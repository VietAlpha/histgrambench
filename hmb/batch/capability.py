"""Versioned batch capability matrix and fail-closed route selection (2026-09-26).

`capability_matrix.json` (next to this module) holds one entry per exact roster route: model or
inference-profile id, transport, request family, Region, inference scope, input modality, and the
separately established states of synchronous structured output, batch submission and batch + structured
output, each with its evidence. Only the states below are allowed.

A call is routed to batch only when ALL hold (anything else routes synchronously, with the reason):
  * the route has an entry and its `final_route` is "batch";
  * `batch_submission` is SUPPORTED_LIVE_COMPLETED (a completed, retrieved, parsed live probe);
  * the request's modality was probed (`image` routes cover image requests; `text` routes text only);
  * a request with a native JSON Schema also needs `batch_structured_output` SUPPORTED_LIVE_COMPLETED.
The loader refuses an entry whose `final_route` is "batch" without those states (no silent upgrade).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

MATRIX_PATH = Path(__file__).with_name("capability_matrix.json")
MATRIX_VERSION = "batch-capability-1"

STATES = (
    "SUPPORTED_LIVE_COMPLETED",       # job completed, output retrieved, parsed and validated
    "SUPPORTED_SUBMISSION_ONLY",      # accepted, no valid completed output yet (never routes production)
    "UNSUPPORTED_LIVE_CONFIRMED",     # capability-specific rejection by the service
    "DOCUMENTED_SUPPORTED_UNTESTED",
    "DOCUMENTED_UNSUPPORTED",
    "INCONCLUSIVE_ACCESS_IAM_QUOTA_REGION",
    "INCONCLUSIVE_FORMAT_OR_MINIMUM",
    "BLOCKED_BY_REQUIRED_STRUCTURED_OUTPUT",
    "NOT_PROBED",                     # absent from the documentation and never probed (never routes production)
)
SYNC_SO_STATES = ("documented", "live_verified", "unsupported", "unknown", "not_required")
PROVIDER_PATHS = ("converse_job", "invoke_job", "openai_batch", "none")
MODALITIES = ("image", "text")
ENTRY_KEYS = {"route_id", "roles", "model_id", "provider", "endpoint", "provider_path", "region", "inference_scope",
              "modality", "structured_output_required", "sync_structured_output", "batch_submission",
              "batch_structured_output", "min_records", "pricing_state", "final_route", "reason", "evidence"}


class CapabilityError(ValueError):
    pass


@dataclass(frozen=True)
class RouteDecision:
    route: str                 # "batch" | "synchronous"
    reason: str
    route_id: str | None
    min_records: int | None = None


def _check(e: dict) -> None:
    missing, extra = ENTRY_KEYS - set(e), set(e) - ENTRY_KEYS
    if missing or extra:
        raise CapabilityError(f"{e.get('route_id')}: missing {sorted(missing)} / unknown {sorted(extra)} keys")
    for k in ("batch_submission", "batch_structured_output"):
        if e[k] not in STATES:
            raise CapabilityError(f"{e['route_id']}: {k}={e[k]!r} is not a capability state")
    if e["sync_structured_output"] not in SYNC_SO_STATES:
        raise CapabilityError(f"{e['route_id']}: sync_structured_output={e['sync_structured_output']!r}")
    if e["provider_path"] not in PROVIDER_PATHS or e["modality"] not in MODALITIES:
        raise CapabilityError(f"{e['route_id']}: provider_path/modality not recognised")
    if e["final_route"] not in ("batch", "synchronous"):
        raise CapabilityError(f"{e['route_id']}: final_route must be batch or synchronous")
    if e["final_route"] == "batch":
        if e["batch_submission"] != "SUPPORTED_LIVE_COMPLETED" or e["provider_path"] != "converse_job":
            raise CapabilityError(f"{e['route_id']}: final_route batch needs a completed live Converse-job probe")
        if e["structured_output_required"] and e["batch_structured_output"] != "SUPPORTED_LIVE_COMPLETED":
            raise CapabilityError(f"{e['route_id']}: the role needs native structured output, unproven in batch")
        if not isinstance(e["min_records"], int) or e["min_records"] < 1:
            raise CapabilityError(f"{e['route_id']}: a batch route needs the service's min_records")
        if not e["evidence"]:
            raise CapabilityError(f"{e['route_id']}: a batch route needs evidence")


def load_matrix(path: str | Path | None = None) -> dict:
    d = json.loads(Path(path or MATRIX_PATH).read_text(encoding="utf-8"))
    if d.get("version") != MATRIX_VERSION:
        raise CapabilityError(f"capability matrix version {d.get('version')!r}, expected {MATRIX_VERSION}")
    seen = set()
    for e in d["entries"]:
        _check(e)
        k = (e["model_id"], e["provider"], e["endpoint"])
        if k in seen:
            raise CapabilityError(f"duplicate route {k}")
        seen.add(k)
    return d


@lru_cache(maxsize=4)
def _default() -> dict:
    return load_matrix()


def entry_for(spec, matrix: dict | None = None) -> dict | None:
    m = matrix or _default()
    for e in m["entries"]:
        if e["model_id"] == spec.model_id and e["provider"] == spec.provider and \
                (e["endpoint"] or None) == (spec.endpoint if spec.provider == "bedrock_mantle" else None):
            return e
    return None


def decide(req, matrix: dict | None = None) -> RouteDecision:
    """Batch or synchronous for one ModelRequest, fail-closed."""
    from hmb.providers import native_schema
    e = entry_for(req.spec, matrix)
    if e is None:
        return RouteDecision("synchronous", f"{req.spec.model_id} ({req.spec.provider}): no capability entry (unknown)",
                             None)
    rid = e["route_id"]
    if e["final_route"] != "batch":
        return RouteDecision("synchronous", f"{rid}: {e['reason']}", rid)
    if req.image_paths and e["modality"] != "image":
        return RouteDecision("synchronous", f"{rid}: image input was not probed in batch", rid)
    if native_schema(req) is not None and e["batch_structured_output"] != "SUPPORTED_LIVE_COMPLETED":
        return RouteDecision("synchronous", f"{rid}: BLOCKED_BY_REQUIRED_STRUCTURED_OUTPUT "
                                            f"(batch + schema is {e['batch_structured_output']})", rid)
    return RouteDecision("batch", f"{rid}: batch_submission SUPPORTED_LIVE_COMPLETED", rid, int(e["min_records"]))
