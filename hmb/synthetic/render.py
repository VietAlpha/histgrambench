"""Render generated fixture candidates (track C) through the PRODUCTION harness.

`render_candidate` calls `hmb.submission.process_submission(..., track="C")`: extraction (§9.2),
wrapper normalization with the §9.4 byte-identity invariant, sandboxed LuaLaTeX, canonicalization.
When the harness module or the pinned sandbox image is unavailable the result is BLOCKED with the
reason; there is never a host TeX fallback.

"""
from __future__ import annotations

from hmb.schemas import ModelSpec
from hmb.store import Layout

CONTROL_MODEL = ModelSpec(model_id="trackc-oracle-generator", provider="fake", family="trackc_generator",
                          role="contestant")


def render_candidate(raw: str, stop_reason: str | None, *, candidate_id: str, layout: Layout, th,
                     dpi_delta: int = 0, access_date: str = "1970-01-01") -> dict:
    if dpi_delta:
        return {"status": "BLOCKED", "reason": "harness renders at the pinned DPI only (dpi transform)"}
    try:
        from hmb.submission import process_submission
        from hmb.submission.sandbox import SandboxUnavailable
    except ModuleNotFoundError as e:
        return {"status": "BLOCKED", "reason": f"hmb.submission unavailable: {e}"}
    meta = {"dataset_version": "trackc", "model_version_string": CONTROL_MODEL.model_id,
            "access_date": access_date, "provider_terms_version": "n/a (generated control)",
            "output_release": "public", "image_encoding": {"source": "generated"}}
    try:
        rec = process_submission(raw, stop_reason, item_id=candidate_id, model=CONTROL_MODEL, track="C",
                                 layout=layout, th=th, meta=meta)
    except SandboxUnavailable as e:
        return {"status": "BLOCKED", "reason": f"sandbox unavailable: {e}"}
    return {"status": rec.outcome_code, "record": rec.model_dump(mode="json")}
