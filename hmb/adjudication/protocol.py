"""§10.6 judge configuration and §10.9 family independence and drift.

Pure decision functions unless stated. (The §10.6 budget governor was removed with the judging budget
B, D2 2026-09-23; the unwired cost formula and extractor audit were removed 2026-09-25.)
"""
from __future__ import annotations

from pathlib import Path

from typing import Any, Iterable, Mapping, Sequence

from hmb.hashing import H
from hmb.schemas import ModelSpec

from .judges import VERDICT_SCHEMA_VERSION, prompt_hashes, tiebreak_prompt_hash


class JudgeConfigError(ValueError):
    """The judge roster cannot satisfy the protocol for this cell (fewer than two families)."""


# ------------------------------------------------------------------ §15.1 judge_config_hash
def judge_config(judges: Sequence[ModelSpec], extractor_version: str, th) -> dict:
    from hmb.providers import effective_inference
    from .serialize import serializer_identity
    return {
        "prompts": prompt_hashes(),
        "statement_serializer": serializer_identity(),
        "verdict_schema_version": VERDICT_SCHEMA_VERSION,
        "judges": [{"model_id": j.model_id, "snapshot": j.snapshot, "family": j.family,
                    "provider": j.provider} for j in judges],
        "extractor_version": extractor_version,
        "protocol": {"decision_rule": "majority of cast votes, >= 2 concurring, third judge only if undecided",
                     "chunk_size": th.value("judge_chunk_size"),
                     "failed_slot": "one judge of another family, no identical-prompt retry",
                     # 2026-09-26: a verdict file written under the older rule (absence of a model-class entity
                     # exempted its relations) must not be reused as-is
                     "exempt_rule": "anchor B unbound, or an argument entity whose geo entity fact is ABSENT",
                     "label_bind_frac": th.value("label_bind_frac"),
                     "eps_on_frac": th.value("eps_on_frac"), "tangency_rel": th.value("tangency_rel"),
                     "parallel_deg": th.value("parallel_deg"), "perpendicular_deg": th.value("perpendicular_deg"),
                     "circle_fit_resid_rel": th.value("circle_fit_resid_rel"),
                     "subgraph_max_unmatched": th.value("subgraph_max_unmatched"),
                     "max_tokens": th.value("judge_max_tokens")},
        "decoding": [{"model_id": j.model_id, **j.decoding, "effort": j.effort,
                      "request": effective_inference(j, int(th.value("judge_max_tokens")))} for j in judges],
    }


# Forced decisions (owner instruction 2026-09-27). Part of the verdict identity, not of the call-cache scope.
FORCED_DECISION_RULE = ("Q3 entities without a usable binding (no majority, object claimed by another entity, or already bound) route their facts to Q1; a Q1 item still tied after every eligible family takes the answer of the earliest panel judge that answered it")


def judge_config_hash(judges: Sequence[ModelSpec], extractor_version: str, th) -> str:
    """§15.1 identity of a verdict: the call-cache scope, the forced-choice Q3 tie-break prompt and the
    forced-decision rule."""
    return H(judge_cache_scope(judges, extractor_version, th), tiebreak_prompt_hash(), FORCED_DECISION_RULE)


def judge_cache_scope(judges: Sequence[ModelSpec], extractor_version: str, th) -> str:
    """Cache scope of the judge calls: H(Q1/Q3 prompts ‖ verdict schema version ‖ judge model ids with snapshots
    ‖ extractor version ‖ protocol constants ‖ decoding); the judge_config_hash before the 2026-09-27 tie-break."""
    c = judge_config(judges, extractor_version, th)
    return H(c["prompts"], c["verdict_schema_version"], c["judges"], c["extractor_version"],
             c["protocol"], c["decoding"], c["statement_serializer"])


# ------------------------------------------------------------------ §10.9 families
def select_judges(judges: Sequence[ModelSpec], *, builder_families: Iterable[str],
                  contestant_family: str | None, configuration: str = "primary") -> list[ModelSpec]:
    """One judge per family in frozen preference order. Primary configuration: no family that built
    this item's card (per-item rotation) and never the contestant's family. The declared secondary
    configuration deliberately keeps builder families (confirmation effect is measured)."""
    if configuration not in ("primary", "secondary"):
        raise ValueError(configuration)
    banned = set()
    if configuration == "primary":
        banned |= set(builder_families)
    if contestant_family is not None:
        banned.add(contestant_family)
    out, seen = [], set()
    for j in judges:
        if j.family in banned or j.family in seen:
            continue
        seen.add(j.family)
        out.append(j)
    return out


# ------------------------------------------------------------------ §10.9 drift probe
def evaluate_drift(reference: Mapping[str, bool], at_open: Mapping[str, bool | None],
                   at_close: Mapping[str, bool | None], th) -> dict:
    """Agreement with reference verdicts at window open and close must be >= κ_min, and
    open-versus-close agreement >= α_min; otherwise the scoring window is invalid. A missing
    verdict counts as disagreement."""
    ids = sorted(reference)
    if not ids:
        return {"status": "no_triples", "valid": False}
    n = len(ids)
    a_open = sum(at_open.get(i) == reference[i] for i in ids) / n
    a_close = sum(at_close.get(i) == reference[i] for i in ids) / n
    ovc = sum(at_open.get(i) is not None and at_open.get(i) == at_close.get(i) for i in ids) / n
    kmin, amin = float(th.value("kappa_min")), float(th.value("alpha_min"))
    return {"status": "ok", "n": n, "agreement_open": a_open, "agreement_close": a_close,
            "open_vs_close": ovc, "kappa_min": kmin, "alpha_min": amin,
            "valid": bool(a_open >= kmin and a_close >= kmin and ovc >= amin)}


def own_q1_votes(layout, triples: Sequence[Mapping[str, Any]], judge_id: str) -> dict[str, tuple[str, bool]]:
    """{triple_id: (cache key of the judge's original Q1 call, its original answer)} for every frozen triple the
    judge voted on at development time (read from the recorded verdict files; owner decision 2026-09-30)."""
    from hmb.schemas import VerdictFile
    from hmb.store import read_json
    out: dict[str, tuple[str, bool]] = {}
    for t in triples:
        model, track, item_id, fact_id = t["triple_id"].rsplit("|", 3)
        vf = VerdictFile.model_validate(read_json(layout.verdicts(model, track, item_id)))
        fv = next(f for f in vf.facts if f.fact_id == fact_id)
        for v in fv.votes:
            if v.judge_id == judge_id and v.question == "Q1" and v.positive is not None:
                out[t["triple_id"]] = (v.raw_call_sha256, bool(v.positive))
    return out


def recorded_image(layout, recorded: str, sha256: str | None) -> Path:
    """A recorded request's image path on this host. Recorded paths are absolute on the host that made the call
    (a Windows path on the development host, 2026-10-01 EC2 move), so a missing path is resolved to the longest
    suffix of it that exists under this root, and accepted only when its bytes hash to the recorded sha256."""
    from hmb.hashing import sha256_file
    from pathlib import PurePosixPath, PureWindowsPath
    if Path(recorded).exists():
        return Path(recorded)
    parts = (PureWindowsPath(recorded) if "\\" in recorded else PurePosixPath(recorded)).parts
    for i in range(1, len(parts)):
        p = layout.root.joinpath(*parts[i:])
        if p.is_file() and (sha256 is None or sha256_file(p) == sha256):
            return p
    raise FileNotFoundError(f"recorded image {recorded!r} not found under {layout.root}")


def run_drift_probe(triples: Sequence[Mapping[str, Any]], judge: ModelSpec, *, layout, th, ledger,
                    allow_live: bool, provider_override=None, phase: str, window_tag: str) -> dict[str, bool | None]:
    """§10.9 drift probe by replay (owner decision 2026-09-30): for every frozen triple this judge voted on at
    development time, re-send that judge's original Q1 call (the recorded prompt text, chunk and ENTITIES block,
    and render) and read its answer for the triple's statement. The reference is the judge's own original
    answer (`own_q1_votes`), so the probe measures whether the judge changed, not its agreement with the panel.
    `window_tag` ("open|<window_id>" / "close|<window_id>", see `cli.drift_probe`) enters the cache scope, so the two
    ends of a window, and different windows, are independent calls.
    Returns {triple_id: asserted or None} for the triples the judge voted on."""
    import re

    from hmb.store import read_json
    from hmb.vocab import JudgeQuestion
    from .judges import JudgeCtx, ask_once
    ctx = JudgeCtx(layout, th, ledger, allow_live, provider_override, phase,
                   config_hash=f"drift|{window_tag}")
    votes = own_q1_votes(layout, triples, judge.model_id)
    by_call: dict[str, list[Mapping[str, Any]]] = {}
    for t in triples:
        if t["triple_id"] in votes:
            by_call.setdefault(votes[t["triple_id"]][0], []).append(t)
    out: dict[str, bool | None] = {}
    for key, ts in sorted(by_call.items()):
        req = read_json(layout.model_call_cache / key[:2] / f"{key}.json")["request"]
        text = req["text"]
        body = text.rsplit("STATEMENTS", 1)[-1]
        sid_of = {stmt: sid for sid, stmt in re.findall(r"^(s\d+): (.*)$", body, re.M)}
        shas = req.get("image_sha256") or [None] * len(req["image_paths"])
        images = [recorded_image(layout, p, s) for p, s in zip(req["image_paths"], shas)]
        a = ask_once(judge, JudgeQuestion.Q1, text, images, list(sid_of.values()), ctx)
        for t in ts:
            sid = sid_of.get(t["fact_text"])
            out[t["triple_id"]] = a.answers.get(sid) if a.status == "ok" and sid else None
    return out


