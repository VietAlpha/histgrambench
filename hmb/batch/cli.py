"""CLI for batch inference: route plan, hybrid final (MS5) run, job status/collect/cancel, probes, cost projection.

    batch-plan    --roster R [--matrix M]                      route of every roster role/model (no call)
    batch-run     --roster R --batch-config C --policy P ...    MS5 cells through the hybrid scheduler
                  [--allow-live] [--dry-run]                    (--dry-run: route of every first-wave call)
    batch-status                                                job manifests and record outcomes
    batch-collect --batch-config C                              poll and ingest every terminal job (idempotent)
    batch-cancel  --batch-config C --job NAME                   stop exactly one named job
    batch-probe   --batch-config C --out DIR --route R --kind K  submit / --check a capability probe
    batch-cost    --out DIR [--date YYYYMMDD]                    granular cost projection (hmb.batch.projection)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from hmb.store import Layout, write_mutable


def _print(obj) -> None:
    sys.stdout.write(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


# --------------------------------------------------------------------------- MS5 tasks
def cp10_tasks(layout: Layout, th, roster: dict, *, tracks: list[str], split: str, allow_live: bool, phase: str,
               access_date: str, provider_terms_version: str, output_release: str, limit: int | None,
               provider_override=None, judge_override=None) -> tuple[dict, list[dict]]:
    """One replayable task per unfinished (contestant, item, track) cell of the image contestants, with the
    same preconditions as `model-run` (checked per contestant before any task exists)."""
    from hmb.ledger import Ledger
    from hmb.pipeline import RunRefused, contestant_cell, eligible_items, require_official_run
    from hmb.roster import accepts_images
    from hmb.schemas import Card
    from hmb.store import read_json
    from hmb.submission.process import model_dir_name
    judges = roster["judge"]
    contestants = [c for c in roster["contestant"] if accepts_images(c)]
    cap = int(th.value("common_output_cap_tokens"))
    if int(th.value("contestant_max_tokens")) > cap:
        raise RunRefused(f"contestant_max_tokens exceeds the common output cap {cap}")
    fake = provider_override is not None or judge_override is not None
    for c in contestants:
        require_official_run(layout, th, c, judges, fake=fake, split=split)
    release_hash = read_json(layout.release_manifest)["release_hash"] if layout.release_manifest.exists() else None
    ledger = Ledger(layout)
    items = eligible_items(layout, split)[: limit or None]
    tasks, resumed = {}, []
    for rec in items:
        iid = rec["id"]
        card = Card.model_validate(read_json(layout.card(iid)))
        crop = layout.crop_canonical(iid)
        ctx_path = layout.context_text(iid)
        ctx = ctx_path.read_text(encoding="utf-8") if ctx_path.exists() else ""
        for c in contestants:
            for t in tracks:
                if layout.submission_record(model_dir_name(c.model_id), t, iid).exists():
                    resumed.append({"item_id": iid, "model_id": c.model_id, "track": t, "resumed": True})
                    continue

                def task(rec=rec, card=card, crop=crop, ctx=ctx, t=t, c=c):
                    return contestant_cell(layout, th, rec=rec, card=card, crop=crop, ctx=ctx, track=t,
                                           contestant=c, judges=judges, ledger=ledger, allow_live=allow_live,
                                           provider_override=provider_override, judge_override=judge_override,
                                           phase=phase, access_date=access_date,
                                           provider_terms_version=provider_terms_version,
                                           output_release=output_release, release_hash=release_hash)
                tasks[f"{c.model_id}|{t}|{iid}"] = task
    return tasks, resumed


def make_router(layout: Layout, cfg, policy_name: str, *, plan_only: bool = False, ledger=None, matrix=None,
                lane_size: int | None = None):
    """`lane_size`: records one contestant lane will receive in this run (items x tracks); the fill target is
    capped at it so a complete lane flushes at once instead of waiting for the window."""
    from hmb.ledger import Ledger
    from .jobs import BatchStore
    from .scheduler import HybridRouter, Policy
    fill = min(cfg.fill_target, lane_size) if lane_size else cfg.fill_target
    pol = Policy(policy_name, fill_target=max(1, fill), window_seconds=cfg.window_seconds,
                 max_records_per_job=cfg.max_records_per_job)
    return HybridRouter(BatchStore(layout), ledger or Ledger(layout), pol, matrix=matrix,
                        sync_concurrency=cfg.sync_concurrency, default_sync_concurrency=cfg.default_sync_concurrency,
                        plan_only=plan_only)


def _cmd_batch_run(a) -> int:
    from hmb.grading.cli import ReportRefused, resolve_thresholds
    from hmb.pipeline import RunRefused
    from hmb.providers import BatchDeferred, set_router
    from hmb.roster import load_roster
    from .jobs import BedrockBatchBackend, load_config
    from .scheduler import HybridScheduler, summarize
    layout = Layout.from_env(a.root)
    try:
        th = resolve_thresholds(a.thresholds, layout)
    except ReportRefused as e:
        print(f"REFUSED: {e}")
        return 2
    cfg = load_config(a.batch_config)
    roster = load_roster(a.roster)
    try:
        tasks, resumed = cp10_tasks(layout, th, roster, tracks=a.track, split=a.split, allow_live=a.allow_live,
                                    phase=a.phase, access_date=a.access_date,
                                    provider_terms_version=a.provider_terms_version,
                                    output_release=a.output_release, limit=a.limit)
    except RunRefused as e:
        print(f"REFUSED: {e}")
        return 3
    from hmb.pipeline import eligible_items
    lane_size = len(eligible_items(layout, a.split)[: a.limit or None]) * len(a.track)
    router = make_router(layout, cfg, a.policy, plan_only=a.dry_run, lane_size=lane_size)
    set_router(router)
    try:
        if a.dry_run:                                  # one pass, no call: the route of every first-wave call
            for f in tasks.values():
                try:
                    f()
                except BatchDeferred:
                    pass
            plan = router.plan
            _print({"policy": a.policy, "cells": len(tasks), "resumed": len(resumed), "first_wave_calls": len(plan),
                    "by_route": {r: sum(p["route"] == r for p in plan) for r in ("batch", "synchronous")},
                    "calls": plan})
            return 0
        if cfg.compile_workers > 1:                    # C24: lifted only after `sandbox-cputime-check` on this host
            import threading

            from hmb.submission import sandbox
            sandbox._COMPILE_LOCK = threading.BoundedSemaphore(cfg.compile_workers)
        from hmb import cpu_pool
        cpu_pool.configure(cfg.cpu_workers)
        sched = HybridScheduler(router, BedrockBatchBackend(cfg), workers=cfg.workers, poll_seconds=cfg.poll_seconds,
                                max_wall_seconds=a.max_wall_seconds)
        out = sched.run(tasks)
    finally:
        set_router(None)
    summary = summarize(router.metrics, t_end=time.monotonic())
    cells = []
    for tid, res in out["results"].items():
        cells.append(res.summary() if hasattr(res, "summary") else {"task": tid, "error": f"{type(res).__name__}: {res}"})
    rep = {"policy": a.policy, "fill_target": router.policy.fill_target, "profile": th.profile.value, "cells_run": len(cells), "resumed": len(resumed),
           "parked": out["parked"], "stopped_early": out["stopped_early"], "efficiency": summary, "cells": cells,
           "note": "ledger figures are estimates from the dated rate card, not billing guarantees"}
    write_mutable(layout.reports / "batch_run.json", (json.dumps(rep, indent=1, default=str) + "\n").encode())
    _print({k: v for k, v in rep.items() if k != "cells"})
    return 4 if out["stopped_early"] else 0


# --------------------------------------------------------------------------- plan / status / collect / cancel
def route_plan(roster: dict, matrix: dict | None = None) -> list[dict]:
    """Route of every roster model for each request kind its role makes (image; native schema where the role
    requires it). Nothing is called."""
    from hmb.schemas import ModelRequest
    from .capability import decide
    rows = []
    png = str(Path(__file__).with_name("capability_matrix.json"))          # any existing path: never read
    for role, specs in roster.items():
        for s in specs:
            schema = {"name": "x", "schema": {"type": "object"}} if s.structured_output else None
            req = ModelRequest(spec=s, text="", image_paths=[] if s.input_mode == "text_only" else [png],
                               max_tokens=1, purpose=role, phase="plan", cache_scope="plan", response_schema=schema)
            d = decide(req, matrix)
            rows.append({"role": role, "model_id": s.model_id, "provider": s.provider, "route": d.route,
                         "reason": d.reason})
    return rows


def _cmd_batch_plan(a) -> int:
    from hmb.roster import load_roster
    from .capability import load_matrix
    m = load_matrix(a.matrix) if a.matrix else load_matrix()
    rows = route_plan(load_roster(a.roster), m)
    _print({"matrix_version": m["version"], "matrix_updated": m.get("updated_utc"), "routes": rows})
    return 0


def _cmd_batch_status(a) -> int:
    from .jobs import BatchStore
    st = BatchStore(Layout.from_env(a.root))
    jobs = []
    for j in st.jobs():
        counts: dict[str, int] = {}
        for r in j["records"]:
            k = (r.get("outcome") or {}).get("status", "pending")
            counts[k] = counts.get(k, 0) + 1
        jobs.append({"job": j["job_name"], "state": j["state"], "aws_status": j.get("aws_status"),
                     "model_id": j["model_id"], "records": len(j["records"]), "outcomes": counts,
                     "submit_error": j.get("submit_error")})
    _print({"jobs": jobs, "queued_requests": len(st.queued_keys())})
    return 0


def _cmd_batch_collect(a) -> int:
    from hmb.ledger import Ledger
    from .jobs import BatchStore, BedrockBatchBackend, TERMINAL, collect_job, load_config, refresh_job, submit_job
    layout = Layout.from_env(a.root)
    st, led = BatchStore(layout), Ledger(layout)
    be = BedrockBatchBackend(load_config(a.batch_config))
    out = []
    for doc in st.jobs():
        if doc["state"] == "prepared":
            doc = submit_job(st, be, led, doc)
        if doc["state"] == "submitted":
            doc = refresh_job(st, be, doc)
            if doc.get("aws_status") in TERMINAL:
                doc = collect_job(st, be, led, doc)
        out.append({"job": doc["job_name"], "state": doc["state"], "aws_status": doc.get("aws_status")})
    _print({"jobs": out})
    return 0


def _cmd_batch_cancel(a) -> int:
    from .jobs import BatchStore, BedrockBatchBackend, cancel_job, load_config
    try:
        cancel_job(BatchStore(Layout.from_env(a.root)), BedrockBatchBackend(load_config(a.batch_config)), a.job)
    except (ValueError, FileNotFoundError) as e:
        print(f"REFUSED: {e}")
        return 2
    print(f"stop requested for {a.job} (only this job); collect it with batch-collect when it is Stopped")
    return 0


def _cmd_batch_probe(a) -> int:
    from .jobs import load_config
    from .probe import check_probe, probe_state, submit_probe
    cfg = load_config(a.batch_config)
    out = Path(a.out)
    if a.check:
        rows = []
        for p in sorted(out.glob("*__*.json")):
            ev = check_probe(cfg, p)
            rows.append({"probe": p.stem, "status": ev.get("status"), "state": probe_state(ev),
                         "result": {k: v for k, v in (ev.get("result") or {}).items() if k != "error_samples"}})
        _print(rows)
        return 0
    ev = submit_probe(cfg, a.route, a.kind, out, n=a.records, max_tokens=a.max_tokens)
    _print({"probe": ev["probe"], "job": ev.get("job_name"), "submit_error": ev.get("submit_error"),
            "state": probe_state(ev)})
    return 0


def _cmd_batch_cost(a) -> int:
    from .projection import write_projection
    paths = write_projection(Path(a.out), date=a.date, pricing=Path(a.pricing) if a.pricing else None)
    _print({"written": [str(p) for p in paths]})
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("batch-plan", help="route (batch or synchronous, with the reason) of every roster role/model "
                                          "from the versioned capability matrix; nothing is called")
    p.add_argument("--roster", required=True)
    p.add_argument("--matrix", default=None, help="capability matrix (default: hmb/batch/capability_matrix.json)")
    p.set_defaults(func=_cmd_batch_plan)
    p = sub.add_parser("batch-run", help="MS5 final run through the hybrid scheduler: every image contestant x item x track as "
                                         "a replayable cell; batch lanes only where the matrix says "
                                         "SUPPORTED_LIVE_COMPLETED; resumable; same preconditions as model-run")
    p.add_argument("--roster", required=True)
    p.add_argument("--batch-config", required=True)
    p.add_argument("--policy", choices=["sync", "fixed", "window"], default="window")
    p.add_argument("--track", action="append", choices=["A", "B"], required=True)
    p.add_argument("--split", default="test", choices=["development", "test"])
    p.add_argument("--thresholds", default=None)
    p.add_argument("--phase", default="FINAL")
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="route of every first-wave call; no call, no submission")
    p.add_argument("--access-date", required=True)
    p.add_argument("--provider-terms-version", required=True)
    p.add_argument("--output-release", default="public", choices=["public", "withheld"])
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--max-wall-seconds", type=float, default=None,
                   help="stop after this long (state stays on disk; rerun to resume)")
    p.set_defaults(func=_cmd_batch_run)
    p = sub.add_parser("batch-status", help="batch job manifests, AWS status and per-record outcomes")
    p.set_defaults(func=_cmd_batch_status)
    p = sub.add_parser("batch-collect", help="submit prepared jobs, poll, and ingest every terminal job once")
    p.add_argument("--batch-config", required=True)
    p.set_defaults(func=_cmd_batch_collect)
    p = sub.add_parser("batch-cancel", help="stop exactly one named batch job")
    p.add_argument("--batch-config", required=True)
    p.add_argument("--job", required=True)
    p.set_defaults(func=_cmd_batch_cancel)
    p = sub.add_parser("batch-probe", help="engineering capability probe: submit one probe job, or --check all "
                                           "probes in --out (paid; never a benchmark call)")
    p.add_argument("--batch-config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--route", default=None)
    p.add_argument("--kind", choices=["plain", "schema"], default="plain")
    p.add_argument("--records", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--check", action="store_true")
    p.set_defaults(func=_cmd_batch_probe)
    p = sub.add_parser("batch-cost", help="granular cost projection: on-demand vs Global CRIS + hybrid batch")
    p.add_argument("--out", required=True, help="directory for the .md and .json projection")
    p.add_argument("--date", default=None)
    p.add_argument("--pricing", default=None, help="price snapshot JSON (default: hmb/batch/price_snapshot.json)")
    p.set_defaults(func=_cmd_batch_cost)
