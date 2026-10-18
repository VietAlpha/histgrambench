"""§11.2–11.3 per-cell scoring: `score_cell(card, sub, outcome, verdicts, th) -> ScoreRecord`.

Deterministic, CPU only (simplification 2026-09-25; the one-page statement is component 07 §11.3):
  scorable facts   = the item's eligible strict facts (classes E, R, L) minus EXEMPT
  S                = g · #PRESENT / #scorable,   g = 1[outcome == OK]
INDETERMINATE counts as not recovered; ABSENT and CONTRADICTED score the same; EXEMPT leaves both
sides. Cards carry only the strict classes (asc-2.0.0: layout, the coarse class Y, is not a card class and
the material-extra count X was removed with the extras channel, 2026-09-26). There are no class weights, caps, optimistic bound or NOT_SCORED status: every eligible cell is scored, and a
judge failure only leaves the undecided facts INDETERMINATE (flagged in `evaluator_failure`).
Non-OK outcomes score 0 with the card's own counts (no verdicts exist). A verdict file can only remove
a fact from a denominator by a deterministic EXEMPT (source 'deterministic'); judge output never can.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from hmb.hashing import H, sha256_bytes
from hmb.schemas import Card, ScoreRecord, SubmissionRecord, VerdictFile
from hmb.ontology import entity_args
from hmb.vocab import CLASS_LETTER, Outcome, Profile, Verdict

CLASSES = ("E", "R", "L")
HEADLINE_CLASSES = CLASSES                   # every card class is strict and scored
AGGREGATION_RULES = ("S_i = g * PRESENT / scorable over strict classes E, R, L",
                     "EXEMPT removed from numerator and denominator",
                     "INDETERMINATE in the denominator (counts as not recovered)",
                     "every eligible cell is scored for every model (common denominator)",
                     "H = mean over split groups of the mean item score; IM t-interval over group means")


class GradingError(ValueError):
    pass


def _src_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    return {p.name: sha256_bytes(p.read_bytes()) for p in sorted(here.glob("*.py"))}


def metric_constants(th) -> dict[str, Any]:
    return {"headline_classes": list(HEADLINE_CLASSES), "m_c": {k: th.value(f"m_{k}") for k in CLASSES},
            "q5_tau_E": th.value("q5_tau_E")}


def metric_hash(th) -> str:
    """metric_hash = H(metric definitions ‖ caps ‖ aggregation rules) (§15.1); the definitions are
    the grading source files themselves."""
    return H(_src_hashes(), metric_constants(th), list(AGGREGATION_RULES))


def _letter(fact) -> str:
    return CLASS_LETTER[fact.fact_class] if not isinstance(fact.fact_class, str) else \
        {"entity": "E", "relation": "R", "label": "L", "layout": "Y"}[fact.fact_class]


def score_cell(card: Card, sub: SubmissionRecord | None, outcome: Outcome, verdicts: VerdictFile | None,
               th, *, model_id: str | None = None, track: str | None = None) -> ScoreRecord:
    outcome = Outcome(outcome)
    facts = [f for f in card.facts if f.eligible]
    if not facts:
        raise GradingError(f"{card.item_id}: no eligible facts; the item is not in I (§11.12)")
    model_id = model_id or (sub.model_id if sub else verdicts.model_id if verdicts else None)
    track = track or (str(sub.track) if sub else str(verdicts.track) if verdicts else None)
    if model_id is None or track is None:
        raise GradingError("model_id and track are required when there is no submission or verdict file")
    if sub is not None and sub.outcome_code != outcome.value:
        raise GradingError("outcome disagrees with the submission record")
    g = 1 if outcome == Outcome.OK else 0
    vmap: dict[str, str] = {}
    label_info: dict[str, dict] = {}
    failure = None
    if g:
        if verdicts is None:
            raise GradingError("an OK cell needs its verdict file")
        if (verdicts.item_id, verdicts.model_id, str(verdicts.track)) != (card.item_id, model_id, track):
            raise GradingError("verdict file is for another cell")
        for fv in verdicts.facts:
            v = str(fv.verdict)
            if v == Verdict.EXEMPT.value and fv.source != "deterministic":
                raise GradingError(f"{fv.fact_id}: EXEMPT from a judge is not allowed (§10.4)")
            vmap[fv.fact_id] = v
            if fv.binding and "label_match" in fv.binding:
                label_info[fv.fact_id] = fv.binding["label_match"]
        _check_exempt_is_deterministic(facts, verdicts)
        failure = verdicts.evaluator_failure
    per: dict[str, dict[str, Any]] = {}
    for c in CLASSES:
        fc = [f for f in facts if _letter(f) == c]
        cnt = {k.value: 0 for k in Verdict}
        errs = {"attachment": 0, "transcription": 0, "missing": 0}
        for f in fc:
            v = vmap.get(f.fact_id, Verdict.INDETERMINATE.value) if g else Verdict.ABSENT.value
            cnt[v] += 1
            li = label_info.get(f.fact_id)
            if g and li and li.get("error") in errs:
                errs[li["error"]] += 1
        ns = len(fc) - cnt[Verdict.EXEMPT.value]
        r = cnt[Verdict.PRESENT.value]
        per[c] = {"n": len(fc), "n_star": ns, "n_present": r, "n_absent": cnt[Verdict.ABSENT.value],
                  "n_contradicted": cnt[Verdict.CONTRADICTED.value],
                  "n_indeterminate": cnt[Verdict.INDETERMINATE.value], "n_exempt": cnt[Verdict.EXEMPT.value],
                  "R": (g * r / ns) if ns else None, "reportable": ns >= int(th.value(f"m_{c}"))}
        if c == "L":
            per[c]["label_errors"] = errs
    den = sum(per[c]["n_star"] for c in HEADLINE_CLASSES)
    if den <= 0:
        raise GradingError(f"{card.item_id}: no scorable strict facts after EXEMPT removal")
    S = g * sum(per[c]["n_present"] for c in HEADLINE_CLASSES) / den
    profile = Profile.production.value if th.is_production else Profile.fixture.value
    if th.is_production and card.profile != Profile.production.value:
        raise GradingError("production scoring refuses a fixture card")
    if th.is_production and verdicts is not None and verdicts.profile != Profile.production.value:
        raise GradingError("production scoring refuses fixture verdicts")
    return ScoreRecord(item_id=card.item_id, model_id=model_id, track=track, outcome_code=outcome,
                       evaluator_failure=failure, S=S, per_class=per, metric_hash=metric_hash(th),
                       profile=profile)


def _check_exempt_is_deterministic(facts, verdicts: VerdictFile) -> None:
    """Nothing judge-derived may reduce n* (red-team #1/#15). EXEMPT is legal only as
    (a) an anchor-B fact (non-unique deterministic binding), or (b) a fact with an argument entity
    whose own eligible entity fact is ABSENT from the extractor (deterministic alignment absence)
    and that entity never went to Q3. Entity facts are never EXEMPT."""
    vf = {fv.fact_id: fv for fv in verdicts.facts}
    ent_fact = {f.args[0]: f.fact_id for f in facts if _letter(f) == "E"}
    q3 = set(verdicts.alignment_ambiguous)
    for f in facts:
        fv = vf.get(f.fact_id)
        if fv is None or str(fv.verdict) != Verdict.EXEMPT.value:
            continue
        if fv.source != "deterministic":
            raise GradingError(f"{f.fact_id}: EXEMPT from a judge is not allowed (§10.4)")
        if _letter(f) == "E":
            raise GradingError(f"{f.fact_id}: an entity fact is never EXEMPT")
        if f.anchor == "B":
            continue
        ok = False
        for x in entity_args(f.predicate, f.args):
            ef = vf.get(ent_fact.get(x, ""))
            if x not in q3 and ef is not None and str(ef.verdict) == Verdict.ABSENT.value \
                    and ef.source in ("extractor", "deterministic"):
                ok = True
                break
        if not ok:
            raise GradingError(f"{f.fact_id}: EXEMPT without a deterministically absent argument entity")


def write_score(layout, rec: ScoreRecord) -> tuple[Path, str]:
    """Write the score file immutably; returns (path, sha256) for SubmissionRecord.scores(_sha256)."""
    from hmb.store import write_json_immutable
    p = layout.scores(rec.model_id, str(rec.track), rec.item_id)
    return p, write_json_immutable(p, rec.model_dump(mode="json"))
