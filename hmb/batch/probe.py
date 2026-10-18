"""Minimal live batch capability probes (engineering evidence only; never a benchmark call).

Each probe is one Converse-type batch job of `n` (the service minimum, 100) tiny deterministic records
built by the PRODUCTION record builder (`jobs.batch_record` -> `providers.converse_request`) with the
route's real request family, effort encoding and modality (a synthetic 32x32 PNG, never a benchmark
crop). Every record carries a checkable answer, so correlation and parsing are validated per record:

    plain   "What is k plus 1?"                     valid when the reply contains k+1
    schema  "Write two sentences about k." + native JSON Schema {"value": integer}
            enforced when the reply is exactly a JSON object {"value": <int>} although the prompt asked
            for prose; prose replies mean the schema was accepted but not enforced

Evidence (sanitized: no account id, no credentials) goes to `<out>/<probe>.json`. Probe jobs carry the
tag hmb-purpose=engineering-capability-probe; their spend is reported separately from benchmark spend.
"""
from __future__ import annotations

import json
import re
import struct
import zlib
from pathlib import Path

from hmb.schemas import ModelRequest, ModelSpec

from .jobs import BatchConfig, BedrockBatchBackend, TERMINAL, _utc, batch_record, classify_record_error, \
    classify_submit_error

PROBE_SCHEMA = {"name": "hmb_probe_value", "schema": {"type": "object", "properties": {"value": {"type": "integer"}},
                                                      "required": ["value"], "additionalProperties": False}}

# Routes probed, with the role's real request family (Converse on bedrock-runtime) and effort encoding.
ROUTES: dict[str, dict] = {
    "opus-5-5": dict(model_id="us.anthropic.claude-opus-5-5", family="anthropic", role="builder", effort="medium",
                     decoding={"additional_model_request_fields": {"thinking": {"type": "adaptive"},
                                                                   "output_config": {"effort": "medium"}}}),
    "gpt-6-sol": dict(model_id="global.openai.gpt-6-sol", family="openai", role="builder", effort="medium",
                      decoding={"additional_model_request_fields": {"reasoning": {"effort": "medium"}}}),
    "gpt-6-luna": dict(model_id="global.openai.gpt-6-luna", family="openai", role="verifier", effort="high",
                       decoding={"additional_model_request_fields": {"reasoning": {"effort": "high"}}}),
    "qwen3-vl-runtime": dict(model_id="qwen.qwen3-vl-235b-a22b", family="qwen", role="judge", effort=None,
                             decoding={"temperature": 0.0, "top_p": 1.0}),
    "mistral-large-3-runtime": dict(model_id="mistral.mistral-large-3-675b-instruct", family="mistral", role="judge",
                                    effort=None, decoding={"temperature": 0.0, "top_p": 1.0}),
    "llama4-maverick": dict(model_id="us.meta.llama4-maverick-17b-instruct-v1:0", family="meta", role="contestant",
                            effort=None, decoding={"temperature": 0.0, "top_p": 1.0}),
    "kimi-k2.5": dict(model_id="moonshotai.kimi-k2.5", family="moonshot", role="contestant", effort=None,
                      decoding={"temperature": 0.0, "top_p": 1.0}),
    "nemotron-nano-12b": dict(model_id="nvidia.nemotron-nano-12b-v2", family="nvidia", role="contestant",
                              effort=None, decoding={"temperature": 0.0, "top_p": 1.0}),
    "ministral-3-14b": dict(model_id="mistral.ministral-3-14b-instruct", family="mistral", role="contestant",
                            effort=None, decoding={"temperature": 0.0, "top_p": 1.0}),
    # roster amendment 2026-10-02: the contestant's own decoding (temperature only; Claude rejects it with top_p)
    "haiku-4-5-global": dict(model_id="global.anthropic.claude-haiku-4-5-20251001-v1:0", family="anthropic",
                             role="contestant", effort=None, decoding={"temperature": 0.0}),
}


def probe_png(path: Path) -> Path:
    """A deterministic 32x32 greyscale PNG (a diagonal stroke); written once."""
    if not path.exists():
        w = h = 32
        rows = b"".join(b"\x00" + bytes(0 if abs(x - y) <= 1 else 255 for x in range(w)) for y in range(h))

        def chunk(t: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + t + data + struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF)
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)) + \
            chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b"")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)
    return path


def probe_spec(route: str, *, schema: bool) -> ModelSpec:
    r = ROUTES[route]
    return ModelSpec(model_id=r["model_id"], provider="bedrock", family=r["family"], role=r["role"],
                     snapshot=None, unit_price_id=None, decoding=r["decoding"], effort=r["effort"],
                     structured_output="native_json_schema" if schema else None)


def probe_requests(route: str, kind: str, image: Path, n: int, max_tokens: int) -> list[tuple[str, ModelRequest]]:
    from hmb.providers import cache_key
    spec = probe_spec(route, schema=kind == "schema")
    out = []
    for k in range(n):
        text = (f"What is {k} plus 1? Reply with the number only." if kind == "plain"
                else f"Write two full sentences about the number {k}. Do not use JSON or code.")
        req = ModelRequest(spec=spec, text=text, image_paths=[str(image)], max_tokens=max_tokens,
                           purpose="batch_capability_probe", phase="ENG-PROBE", cache_scope=f"batch-probe-1|{kind}|{k}",
                           response_schema=PROBE_SCHEMA if kind == "schema" else None)
        out.append((cache_key(req), req))
    return out


def validate(kind: str, k: int, text: str) -> tuple[bool, str]:
    t = (text or "").strip()
    if kind == "plain":
        return (re.search(rf"(?<!\d){k + 1}(?!\d)", t) is not None), ("answer" if t else "empty")
    try:
        v = json.loads(t)
    except ValueError:
        return False, "prose (schema not enforced)" if t else "empty"
    ok = isinstance(v, dict) and set(v) == {"value"} and isinstance(v["value"], int) and not isinstance(v["value"], bool)
    return ok, "schema-conformant JSON" if ok else "JSON not matching schema"


def job_arn(cfg: BatchConfig, job_id: str) -> str:
    acct = cfg.role_arn.split(":")[4]
    return f"arn:aws:bedrock:{cfg.region}:{acct}:model-invocation-job/{job_id}"


def _sanitize(s: str) -> str:
    return re.sub(r"\d{12}", "<acct>", s or "")


def submit_probe(cfg: BatchConfig, route: str, kind: str, out_dir: Path, *, n: int = 100, max_tokens: int = 2048,
                 backend=None) -> dict:
    """Submit one probe job (idempotent: an existing evidence file with a job is returned unchanged)."""
    ev_path = out_dir / f"{route}__{kind}.json"
    if ev_path.exists():
        ev = json.loads(ev_path.read_text(encoding="utf-8"))
        if ev.get("job_arn") or ev.get("submit_error"):
            return ev
    backend = backend or BedrockBatchBackend(cfg)
    img = probe_png(out_dir / "probe_32x32.png")
    reqs = probe_requests(route, kind, img, n, max_tokens)
    lines = [json.dumps(batch_record(r, k), sort_keys=True) for k, r in reqs]
    job_name = f"hmb-probe-{route.replace('.', '-')}-{kind}-{_utc()[:19].replace(':', '').replace('-', '')}"[:63]
    ev = {"probe": f"{route}__{kind}", "route": route, "kind": kind, "model_id": ROUTES[route]["model_id"],
          "region": cfg.region, "request_family": "Converse model invocation job (modelInvocationType=Converse)",
          "records": n, "max_tokens": max_tokens, "effort": ROUTES[route]["effort"],
          "request_shape": {k: v for k, v in json.loads(lines[0])["modelInput"].items() if k != "messages"},
          "modality": "image+text", "job_name": job_name, "submitted_utc": _utc(), "job_arn": None,
          "keys": [k for k, _ in reqs]}
    try:
        ev["job_arn"] = _sanitize(backend.submit(job_name, ROUTES[route]["model_id"], ("\n".join(lines) + "\n").encode(),
                                                 {"hmb-purpose": "engineering-capability-probe"}))
        ev["job_arn_raw_suffix"] = ev["job_arn"].rsplit("/", 1)[-1]
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        ev["submit_error"] = {"code": code, "message": _sanitize(str(e))[:800],
                              "state": classify_submit_error(code, str(e))}
    out_dir.mkdir(parents=True, exist_ok=True)
    ev_path.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev


def check_probe(cfg: BatchConfig, ev_path: Path, *, backend=None) -> dict:
    """Refresh a probe's status; when terminal, retrieve, parse (production parser) and validate every record."""
    from hmb.providers import converse_result
    ev = json.loads(ev_path.read_text(encoding="utf-8"))
    if ev.get("submit_error") or ev.get("result"):
        return ev
    backend = backend or BedrockBatchBackend(cfg)
    st = backend.status(job_arn(cfg, ev["job_arn_raw_suffix"]))   # evidence files never store the account id
    ev["status"] = st["status"]
    ev["status_message"] = _sanitize(st.get("message") or "")
    ev["checked_utc"] = _utc()
    if st["status"] in TERMINAL:
        ev["end_time"] = st.get("end_time")
        lines, manifest = backend.outputs(ev["job_name"])
        idx = {k: i for i, k in enumerate(ev["keys"])}
        spec = probe_spec(ev["route"], schema=ev["kind"] == "schema")
        stats = {"lines": len(lines), "valid": 0, "invalid": 0, "errors": 0, "unknown_ids": 0, "duplicates": 0,
                 "stop_reasons": {}, "input_tokens": 0, "output_tokens": 0, "error_samples": [], "invalid_samples": [],
                 "detail_counts": {}}
        seen = set()
        for ln in lines:
            row = json.loads(ln)
            rid = row.get("recordId")
            if rid not in idx:
                stats["unknown_ids"] += 1
                continue
            if rid in seen:
                stats["duplicates"] += 1
                continue
            seen.add(rid)
            if row.get("error") or row.get("modelOutput") is None:
                stats["errors"] += 1
                d, _ = classify_record_error(row.get("error"))
                if len(stats["error_samples"]) < 3:
                    stats["error_samples"].append(_sanitize(d))
                continue
            req = ModelRequest(spec=spec, text="", max_tokens=1, purpose="p", phase="p", cache_scope="p")
            out = converse_result(row["modelOutput"], req)
            stats["input_tokens"] += int(out["input_tokens"] or 0)
            stats["output_tokens"] += int(out["output_tokens"] or 0)
            stats["stop_reasons"][str(out["stop_reason"])] = stats["stop_reasons"].get(str(out["stop_reason"]), 0) + 1
            ok, detail = validate(ev["kind"], idx[rid], out["text"])
            stats["detail_counts"][detail] = stats["detail_counts"].get(detail, 0) + 1
            stats["valid" if ok else "invalid"] += 1
            if not ok and len(stats["invalid_samples"]) < 3:
                stats["invalid_samples"].append((out["text"] or "")[:200])
        stats["missing"] = len(ev["keys"]) - len(seen)
        ev["manifest_out"] = manifest
        ev["result"] = stats
    ev_path.write_text(json.dumps(ev, indent=1), encoding="utf-8")
    return ev


def probe_state(ev: dict) -> str:
    """Capability state implied by one probe's evidence (plain -> batch_submission, schema -> batch+SO).
    Record-level service errors do not change the state (the capability exists); they are reported as
    `record_error_rate` and weighed in the production route decision."""
    if ev.get("submit_error"):
        return ev["submit_error"]["state"]
    if "result" not in ev:
        return "SUPPORTED_SUBMISSION_ONLY"
    r = ev["result"]
    n = len(ev["keys"])
    prose = r["detail_counts"].get("prose (schema not enforced)", 0)
    if ev["kind"] == "schema" and prose >= n // 2:
        return "UNSUPPORTED_LIVE_CONFIRMED"               # accepted, but the schema was not enforced
    if r["errors"] and any(re.search(r"(not support|unsupported|outputconfig|output_config|response_format)", s, re.I)
                           for s in r["error_samples"]):
        return "UNSUPPORTED_LIVE_CONFIRMED"
    if ev.get("status") in ("Completed", "PartiallyCompleted") and r["valid"] > 0 and r["invalid"] == 0             and r["unknown_ids"] == 0 and r["duplicates"] == 0 and r["valid"] + r["errors"] + r["missing"] == n:
        return "SUPPORTED_LIVE_COMPLETED"
    return "INCONCLUSIVE_FORMAT_OR_MINIMUM"


def record_error_rate(ev: dict) -> float | None:
    r = ev.get("result")
    return None if not r else (r["errors"] + r["missing"]) / len(ev["keys"])
