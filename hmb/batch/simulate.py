"""Representative synthetic workload for measuring scheduling policies (engineering evidence only).

Runs the REAL router, scheduler, `call_model`, cache, ledger and job manifests over synthetic final-run cells,
with fake synchronous and batch backends whose latencies are time-scaled (`scale` real seconds per
simulated second). Each synthetic cell has the dependency chain of a real cell:

    contestant call -> compile (serialized, C24) -> Q3 (2 judges, when ambiguous) -> Q1 (2 judges per chunk,
    third judge on disagreement) -> done

with the widened deferral collection of the production judge stages (independent slots of one stage are
collected before the stage waits). Counts and latencies come from `WorkloadShape` (defaults: the measured
2026-09-26 live flow, reports/prompt_repair_20260926.md §8). No model is called and nothing is paid.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from hmb.schemas import ModelRequest, ModelSpec
from hmb.vocab import Role
from hmb.store import Layout

from .jobs import BatchStore
from .scheduler import HybridRouter, HybridScheduler, Policy, summarize


@dataclass
class WorkloadShape:
    items: int = 40
    contestants: tuple = ("sim.contestant-a", "sim.contestant-b", "sim.contestant-c", "sim.contestant-d",
                          "sim.contestant-e", "sim.contestant-f")
    batch_contestants: tuple = ("sim.contestant-a", "sim.contestant-b", "sim.contestant-c", "sim.contestant-d")
    judges: tuple = ("sim.judge-1", "sim.judge-2", "sim.judge-3")
    tracks: tuple = ("A", "B")
    contestant_s: float = 7.5            # measured 3-12 s per submission
    judge_s: float = 4.3                 # ~79 s serial per image cell over ~18.5 judge calls
    compile_s: float = 5.0               # serialized sandbox compile (C24)
    q1_chunks: float = 1.5               # measured ~1.5 Q1 calls per judge per cell
    q3_rate: float = 0.5                 # cells with an ambiguous alignment (Q3 asked)
    third_rate: float = 0.06             # share of statements escalated to the third judge
    compile_ok: float = 0.8              # share of cells whose contestant output compiles (reaches judging)
    batch_turnaround_s: float = 1800.0   # submit -> Completed, from the live probes
    min_records: int = 100


def _u(*parts) -> float:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF


class FakeSyncProvider:
    def __init__(self, shape: WorkloadShape, scale: float):
        self.shape, self.scale = shape, scale

    def send(self, req: ModelRequest, images) -> dict:
        s = self.shape.contestant_s if req.purpose == "contestant" else self.shape.judge_s
        time.sleep(s * self.scale)
        return {"text": f"ok:{req.cache_scope}", "stop_reason": "end_turn", "input_tokens": 1000, "output_tokens": 100}


class FakeBatchBackend:
    """In-memory CreateModelInvocationJob stand-in: Completed after `turnaround_s` simulated seconds."""

    def __init__(self, shape: WorkloadShape, scale: float, clock=time.monotonic):
        self.shape, self.scale, self.clock = shape, scale, clock
        self.jobs: dict[str, dict] = {}
        self.lock = threading.Lock()

    def submit(self, job_name: str, model_id: str, jsonl: bytes, tags: dict) -> str:
        recs = [json.loads(ln) for ln in jsonl.decode().splitlines() if ln.strip()]
        with self.lock:
            self.jobs[job_name] = {"t": self.clock(), "recs": recs}
        return f"arn:aws:bedrock:us-east-1:000000000000:model-invocation-job/{job_name}"

    def status(self, arn: str) -> dict:
        j = self.jobs[arn.rsplit("/", 1)[-1]]
        done = self.clock() - j["t"] >= self.shape.batch_turnaround_s * self.scale
        return {"status": "Completed" if done else "InProgress", "message": None}

    def outputs(self, job_name: str):
        lines = []
        for r in self.jobs[job_name]["recs"]:
            text = r["modelInput"]["messages"][0]["content"][-1]["text"]
            lines.append(json.dumps({"recordId": r["recordId"], "modelOutput": {
                "output": {"message": {"content": [{"text": f"ok:{text}"}]}}, "stopReason": "end_turn",
                "usage": {"inputTokens": 1000, "outputTokens": 100}}}))
        return lines, {"totalRecordCount": len(lines)}

    def stop(self, arn: str) -> None:
        pass


_COMPILE = threading.Lock()


def _spec(model_id: str, role: str) -> ModelSpec:
    return ModelSpec(model_id=model_id, provider="bedrock", family=model_id.split("-")[-1], role=Role(role),
                     unit_price_id=f"sim/{model_id}", decoding={"temperature": 0.0})


def cell_task(layout: Layout, ledger, shape: WorkloadShape, provider, scale: float, item: int, model: str, track: str):
    from hmb.providers import BatchDeferred, call_model

    def ask(spec: ModelSpec, purpose: str, scope: str):
        req = ModelRequest(spec=spec, text=scope, max_tokens=256, purpose=purpose, phase="SIM", cache_scope=scope)
        return call_model(req, layout, ledger, allow_live=True, provider_override=provider)

    def wave(slots):                         # one stage: independent slots, deferrals collected
        deferred = []
        for spec, purpose, scope in slots:
            try:
                ask(spec, purpose, scope)
            except BatchDeferred as d:
                deferred.append(d)
        if deferred:
            raise BatchDeferred.merge(deferred)

    def run():
        cid = f"{item}|{model}|{track}"
        ask(_spec(model, "contestant"), "contestant", f"c|{cid}")
        if _u("compile", cid) > shape.compile_ok:
            return {"cell": cid, "outcome": "C_ERROR"}
        done_marker = layout.p("sim", "compiled", hashlib.sha256(cid.encode()).hexdigest() + ".txt")
        if not done_marker.exists():                               # compile once per cell (content-addressed)
            with _COMPILE:
                time.sleep(shape.compile_s * scale)
            done_marker.parent.mkdir(parents=True, exist_ok=True)
            done_marker.write_text("ok")
        j = [_spec(x, "judge") for x in shape.judges]
        for q, n in (("Q3", 1 if _u("q3", cid) < shape.q3_rate else 0),
                     ("Q1", max(1, round(shape.q1_chunks + _u("q1", cid) - 0.5)))):
            if not n:
                continue
            wave([(jj, q, f"{q}|{cid}|{k}|{jj.model_id}") for k in range(n) for jj in j[:2]])
            third = [k for k in range(n) if _u("3rd", q, cid, k) < shape.third_rate]
            if third:
                wave([(j[2], q, f"{q}|{cid}|{k}|{j[2].model_id}") for k in third])
        return {"cell": cid, "outcome": "OK"}
    return run


def _rate_card(path: Path, shape: WorkloadShape) -> None:
    import yaml
    entries = []
    for m in (*shape.contestants, *shape.judges):
        entries.append({"unit_price_id": f"sim/{m}", "model_id": m, "usd_per_mtok_input": 1.0, "usd_per_mtok_output": 2.0,
                        "pricing_url": "https://example.invalid/sim", "retrieved_utc": "2026-09-26"})
        if m in shape.batch_contestants:
            entries.append({"unit_price_id": f"sim-batch/{m}", "model_id": m, "usd_per_mtok_input": 0.5,
                            "usd_per_mtok_output": 1.0, "lane": "batch", "pricing_url": "https://example.invalid/sim",
                            "retrieved_utc": "2026-09-26"})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump({"version": "pricing-1", "entries": entries}), encoding="utf-8")


def _matrix(shape: WorkloadShape) -> dict:
    ents = []
    for m in shape.batch_contestants:
        ents.append({"route_id": f"sim-{m}", "roles": ["contestant"], "model_id": m, "provider": "bedrock", "endpoint": None,
                     "provider_path": "converse_job", "region": "us-east-1", "inference_scope": "in_region",
                     "modality": "text", "structured_output_required": False, "sync_structured_output": "not_required",
                     "batch_submission": "SUPPORTED_LIVE_COMPLETED", "batch_structured_output": "DOCUMENTED_UNSUPPORTED",
                     "min_records": shape.min_records, "pricing_state": "sim", "final_route": "batch",
                     "reason": "simulation", "evidence": ["simulation"]})
    return {"version": "batch-capability-1", "entries": ents}


def run_policy(policy: str, shape: WorkloadShape, *, workers: int, scale: float, fill_target: int,
               window_s: float, sync_concurrency: int = 4, root: Path | None = None) -> dict:
    """One measured run. policy: sequential (the current model-run order), sync, fixed or window."""
    from hmb.ledger import Ledger
    from hmb.providers import set_router
    with tempfile.TemporaryDirectory() as td:
        layout = Layout(Path(root or td))
        _rate_card(layout.pricing, shape)
        ledger = Ledger(layout)
        matrix = _matrix(shape)
        pol = Policy("sync" if policy == "sequential" else policy, fill_target=fill_target,
                     window_seconds=window_s * scale, max_records_per_job=10000)
        router = HybridRouter(BatchStore(layout), ledger, pol, matrix=matrix, retry_max=0,
                              default_sync_concurrency=1 if policy == "sequential" else sync_concurrency)
        backend = FakeBatchBackend(shape, scale)
        prov = FakeSyncProvider(shape, scale)
        # provider_override marks responses fake; the router routes by the matrix only for non-fake calls,
        # so the simulation installs the provider as the live adapter instead.
        import os

        import hmb.providers as PV
        orig = PV.adapter_for
        PV.adapter_for = lambda name: prov
        # At millisecond time scales a real fsync per ledger/cache/attempt write (~ms) would be magnified into
        # seconds of simulated time; real calls take seconds, so the harness removes it for the measurement only.
        orig_fsync = os.fsync
        os.fsync = lambda fd: None
        tasks = {}
        order = [(m, i, t) for m in shape.contestants for i in range(shape.items) for t in shape.tracks]
        for m, i, t in order:
            tasks[f"{m}|{t}|{i}"] = cell_task(layout, ledger, shape, None, scale, i, m, t)
        sched = HybridScheduler(router, backend, workers=1 if policy == "sequential" else workers,
                                poll_seconds=60 * scale, tick_seconds=min(0.05, 5 * scale))
        set_router(router)
        try:
            out = sched.run(tasks)
        finally:
            set_router(None)
            PV.adapter_for = orig
            os.fsync = orig_fsync
        s = summarize(router.metrics, t_end=time.monotonic(), scale=1.0 / scale)
        spent = ledger.spent()
        ev = ledger.events()
        lanes = {"batch": sum(1 for e in ev if e["event"] == "settle" and e.get("lane") == "batch"),
                 "on_demand": sum(1 for e in ev if e["event"] == "settle" and e.get("lane") != "batch")}
        busy = s["sync_busy_seconds_by_model"]
        wall = s["wall_seconds"]
        return {"policy": policy, "workers": 1 if policy == "sequential" else workers, **s,
                "utilization_by_model": {m: busy.get(m, 0.0) / wall for m in sorted(busy)} if wall else {},
                "settled_calls_by_lane": lanes, "ledger_usd": spent,
                "batch_fill_rate": (s["mean_job_records"] / fill_target) if s["remote_jobs"] else None,
                "unfinished_tasks": len(out["parked"]), "task_errors": s["task_errors"]}


def measure(shape: WorkloadShape | None = None, *, workers: int = 16, scale: float = 0.001,
            fill_target: int = 500, window_s: float = 900.0) -> dict:
    shape = shape or WorkloadShape()
    runs = [run_policy(p, shape, workers=workers, scale=scale, fill_target=fill_target, window_s=window_s)
            for p in ("sequential", "sync", "fixed", "window")]
    return {"shape": asdict(shape), "scale_real_seconds_per_simulated_second": scale, "fill_target": fill_target,
            "window_seconds": window_s, "runs": runs}
