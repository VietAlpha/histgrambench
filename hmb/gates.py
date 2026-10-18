"""§17.5 gates G1, G2, G6-G9, the §17.6 scale-or-stop decision table, the milestone exits MS1-MS5 with
the book-readiness condition (§17.2, checkpoint simplification 2026-09-26), and the two-act freeze
invariants (§17.3).

Evidence rules (docs/CONTRACTS.md): missing input -> NOT_RUN (or BLOCKED when a prerequisite such as
an unfrozen level is missing), never PASS; fixture-profile evidence -> NOT_RUN; the decision table
returns UNDECIDABLE whenever a blocking gate is not decided. Evidence is read from `score-report`
(the G1 N, G2, G6, G7 and G8 inputs), never assembled by hand (simplification 2026-09-25).

The evaluator-validation gates G3-G5 (V1-V11 on Track C controls) were deleted with CP3 and CP5
(checkpoint simplification 2026-09-26): `geo` verdicts come from deterministic code covered by offline
software tests, and model-judge reliability is reported from the final run itself (three-family
majority, leave-one-judge-family-out G6, INDETERMINATE and evaluator-failure rates).

G9 is informational since the owner's 2026-09-23 decision (reports/decision_audit_20260923.md D2:
spending tracked, not capped; no X_V1 ceiling). It reports the projected V1 cost and never passes or
fails. The 17.6 rule "SCALE when at least 3 of the 4 modulating gates pass" is applied literally with
a G9 that cannot pass, so SCALE needs G2, G6 and G8 all to pass.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .config import REGISTRY, CalibrationMissing, load_thresholds, registry_hash
from .store import FreezeViolation, Layout, iter_jsonl, read_json
from .vocab import Decision, GateResult

PASS, FAIL, NOT_RUN, BLOCKED, INFORMATIONAL = (GateResult.PASS.value, GateResult.FAIL.value,
                                               GateResult.NOT_RUN.value, GateResult.BLOCKED.value,
                                               GateResult.INFORMATIONAL.value)
BLOCKING = ("G1", "G7")
MODULATING = ("G2", "G6", "G8")          # G9 is informational (module docstring)
SCALE_MIN_MODULATING_PASS = 3            # 17.6 "3 of 4", with G9 never passing


def gate(gid: str, result: str, *, metric: Any = None, threshold: Any = None, evidence: dict | None = None,
         reason: str | None = None, profile: str | None = None) -> dict:
    if result in (PASS, FAIL) and profile == "fixture":
        evidence = dict(evidence or {}, engineering_result=result)
        result, reason = NOT_RUN, (reason + "; " if reason else "") + "fixture-profile evidence"
    return {"gate_id": gid, "result": result, "blocking": gid in BLOCKING, "metric": metric,
            "threshold": threshold, "evidence": evidence or {}, "reason": reason, "profile": profile}


def _th(th, name):
    try:
        return th.value(name), None
    except (CalibrationMissing, KeyError) as e:
        return None, str(e)


# ------------------------------------------------------------------------ G1
def g1(layout: Layout, N: int | None, *, profile: str = "production") -> dict:
    """human_touch_rate = distinct items with any scope == "item" row in the human audit log / N."""
    if not layout.human_audit_log.exists():
        return gate("G1", NOT_RUN, reason="no human audit log (pipeline not audited)", profile=profile)
    if not N:
        return gate("G1", NOT_RUN, reason="N (eligible items) unknown", profile=profile)
    items = {r["item_id"] for r in iter_jsonl(layout.human_audit_log) if r.get("scope") == "item"}
    rate = len(items) / N
    return gate("G1", PASS if rate == 0.0 else FAIL, metric=rate, threshold="exactly 0.000",
                evidence={"items_touched": sorted(items), "N": N}, profile=profile)


# ------------------------------------------------------------------------ G2
def g2(ev: dict | None, th) -> dict:
    """ev: {N, N_star?, F_bar, group_sizes: {group: n_items}, profile}. N* read from thresholds."""
    if not ev or ev.get("N") is None or ev.get("F_bar") is None or not ev.get("group_sizes"):
        return gate("G2", NOT_RUN, reason="yield inputs missing (§8.7 output)")
    levels = {k: _th(th, k) for k in ("N_star", "F_bar_min", "G_min", "min_items_per_group")}
    missing = {k: err or "no value" for k, (v, err) in levels.items() if v is None}
    if missing:                                         # red-team #4: never a TypeError
        return gate("G2", BLOCKED, reason=f"levels not set: {missing}")
    N_star, F_min, G_min, per = (levels[k][0] for k in ("N_star", "F_bar_min", "G_min", "min_items_per_group"))
    big = sum(1 for n in ev["group_sizes"].values() if n >= per)
    ok = ev["N"] >= N_star and ev["F_bar"] >= F_min and big >= G_min
    return gate("G2", PASS if ok else FAIL, metric={"N": ev["N"], "F_bar": ev["F_bar"], "groups_ge_min": big},
                threshold={"N_star": N_star, "F_bar_min": F_min, "G_min": G_min, "min_items_per_group": per},
                profile=ev.get("profile"))


# ------------------------------------------------------------------------ G6
def g6(ev: dict | None, th) -> dict:
    """ev: {M, n_judge_families, min_tau_b, profile}. Ranking stability: the minimum over judge
    families of Kendall τ_b between the full headline ordering and the leave-one-judge-family-out
    ordering (score-report `lojfo`), with M >= 4 models and >= 3 judge families (§10.9)."""
    if not ev:
        return gate("G6", NOT_RUN, reason="no model-run ranking evidence")
    if (ev.get("M") or 0) < 4:
        return gate("G6", NOT_RUN, reason="G6 requires M >= 4 models", metric={"M": ev.get("M")})
    if (ev.get("n_judge_families") or 0) < 3:
        return gate("G6", NOT_RUN, reason="G6 requires >= 3 judge families (leave-one-judge-family-out)",
                    metric={"n_judge_families": ev.get("n_judge_families")})
    tmin, err = _th(th, "tau_b_min")
    if tmin is None:
        return gate("G6", BLOCKED, reason=err)
    if ev.get("min_tau_b") is None:
        return gate("G6", NOT_RUN, reason="ranking statistics missing")
    return gate("G6", PASS if ev["min_tau_b"] >= tmin else FAIL, metric={"min_tau_b": ev["min_tau_b"]},
                threshold={"tau_b_min": tmin}, profile=ev.get("profile"))


# ------------------------------------------------------------------------ G7
def g7(ev: dict | None, th) -> dict:
    """Separation (owner decision D21, 2026-09-26): PASS when at least one eligible model pair of the Track A
    all-pairs comparison is separated (BH rejects at bh_q over the IM p-values) with |d̄| >= δ* and
    δ* >= 1.1 × that pair's MDD floor. ev = score-report `gate_evidence.G7`: {G, q, family_size, pairs:
    [{pair, status, diff, bh_adjusted, separated, mdd_floor, ...}], ceiling_flag, profile}. Eligible = the
    pair's IM test is computable on the common Track A item set. `design_ok` (for the §17.6 STOP row) holds
    when δ* >= 1.1 × MDD floor for every eligible pair."""
    if not ev or not ev.get("pairs"):
        return gate("G7", NOT_RUN, reason="all-pairs comparison not computed"
                    + (f" ({ev.get('reason') or ev.get('status')})" if ev else ""))
    ds, err = _th(th, "delta_star")
    G_min, err2 = _th(th, "G_min")
    if ds is None or G_min is None:
        return gate("G7", BLOCKED, reason=err or err2)
    if (ev.get("G") or 0) < G_min:
        return gate("G7", NOT_RUN, reason=f"G = {ev.get('G')} < G_min = {G_min}: no inferential claim (§14.2)",
                    profile=ev.get("profile"))
    eligible = [p for p in ev["pairs"] if p.get("status") == "ok" and p.get("diff") is not None]
    if not eligible:
        return gate("G7", NOT_RUN, reason="no eligible pair (no computable IM test)", profile=ev.get("profile"))
    rows = []
    for p in eligible:
        floor = p.get("mdd_floor")
        design = floor is not None and ds >= 1.1 * floor
        rows.append({**p, "design_ok": design,
                     "qualifies": bool(p.get("separated")) and abs(p["diff"]) >= ds and design})
    qualifying = [r for r in rows if r["qualifies"]]
    design_ok = all(r["design_ok"] for r in rows)
    return gate("G7", PASS if qualifying else FAIL,
                metric={"qualifying_pairs": [{k: r[k] for k in ("pair", "diff", "bh_adjusted", "mdd_floor")}
                                             for r in qualifying],
                        "n_eligible_pairs": len(rows), "n_separated": sum(bool(r.get("separated")) for r in rows)},
                threshold={"delta_star": ds, "bh_q": ev.get("q"), "delta_star_ge_1.1_mdd_floor": "per pair"},
                evidence={"ceiling_flag": ev.get("ceiling_flag"), "design_ok": design_ok, "G": ev.get("G"),
                          "family_size": ev.get("family_size"), "pairs": rows},
                profile=ev.get("profile"))


# ------------------------------------------------------------------------ G8
def g8(ev: dict | None, th) -> dict:
    """ev: {ci: [lo, hi] of the pooled context effect ΔH (B − A), profile}: effect (excludes 0) or
    informative null (within ±δ*)."""
    if not ev or ev.get("ci") is None:
        return gate("G8", NOT_RUN, reason="context effect not computed")
    ds, err = _th(th, "delta_star")
    if ds is None:
        return gate("G8", BLOCKED, reason=err)
    lo, hi = ev["ci"]
    effect = lo > 0 or hi < 0
    null = -ds <= lo and hi <= ds
    return gate("G8", PASS if (effect or null) else FAIL, metric={"ci": ev["ci"], "effect": effect,
                                                                  "informative_null": null},
                threshold={"delta_star": ds}, profile=ev.get("profile"))


# ------------------------------------------------------------------------ G9
def g9(ev: dict | None) -> dict:
    """ev: {projection: hmb.ledger.project_cost output at 2,500 items / 60 books / M models, profile}.
    Informational: the projected V1 cost is reported with no ceiling (D2, 2026-09-23)."""
    if not ev or not ev.get("projection") or ev["projection"].get("total") is None:
        missing = (ev or {}).get("projection", {}).get("missing") if ev else None
        return gate("G9", NOT_RUN, reason=f"projection incomplete (missing {missing})")
    return gate("G9", INFORMATIONAL, metric=ev["projection"]["total"], threshold="none (spend tracked, not capped)",
                evidence={"disclaimer": ev["projection"].get("disclaimer")}, profile=ev.get("profile"))


# ------------------------------------------------------------------------ §17.6 decision table
REMEDIATION = {
    "G2": "more books at constant items per book",
    "G6": "freeze one judge configuration and publish single-judge with the instability reported",
    "G8": "Track B becomes a V1 analysis, not a V1 track requirement",
}


def decide(gates: dict[str, dict]) -> dict:
    """§17.6. `gates` maps G1..G9 to gate results."""
    r = {g: (gates.get(g) or {}).get("result", NOT_RUN) for g in BLOCKING + MODULATING}
    ev7 = (gates.get("G7") or {})
    if r["G1"] == FAIL:
        return {"decision": Decision.STOP_IMMEDIATELY.value, "reason": "G1 failed: zero-annotation premise violated",
                "gates": r}
    undecided = [g for g in BLOCKING if r[g] not in (PASS, FAIL)]
    if r["G1"] != PASS:
        return {"decision": Decision.UNDECIDABLE.value, "reason": "G1 not decided", "undecided": undecided,
                "gates": r}
    if undecided:
        return {"decision": Decision.UNDECIDABLE.value, "reason": f"blocking gates not decided: {undecided}",
                "undecided": undecided, "gates": r}
    if r["G7"] == FAIL:
        evd = ev7.get("evidence", {})
        if evd.get("ceiling_flag") is True:
            return {"decision": Decision.SCALE_REDUCED_WIDENED.value,
                    "reason": "G7 failed with the ceiling-effect flag: widen eligibility", "gates": r}
        if evd.get("ceiling_flag") is False and evd.get("design_ok") is True:
            return {"decision": Decision.STOP.value,
                    "reason": "models do not separate at a difference the design could detect", "gates": r}
        return {"decision": Decision.UNDECIDABLE.value,
                "reason": "G7 failed but the ceiling flag or δ* >= 1.1·MDD_floor for every eligible pair is not "
                          "established", "gates": r}
    passes = [g for g in MODULATING if r[g] == PASS]
    undec_mod = [g for g in MODULATING if r[g] not in (PASS, FAIL)]
    if len(passes) >= SCALE_MIN_MODULATING_PASS:
        return {"decision": Decision.SCALE.value, "reason": f"all blocking pass; modulating pass {passes}",
                "gates": r}
    if len(passes) + len(undec_mod) >= SCALE_MIN_MODULATING_PASS:
        return {"decision": Decision.UNDECIDABLE.value,
                "reason": f"modulating gates not decided {undec_mod} could change SCALE vs SCALE-REDUCED",
                "gates": r}
    failing = [g for g in MODULATING if r[g] == FAIL]
    return {"decision": Decision.SCALE_REDUCED.value, "reason": f"modulating pass {passes}",
            "remediation": {g: REMEDIATION[g] for g in failing}, "gates": r}


def evaluate_all(evidence: dict, layout: Layout, th) -> dict:
    gates = {
        "G1": g1(layout, evidence.get("N"), profile=evidence.get("profile", "production")),
        "G2": g2(evidence.get("G2"), th),
        "G6": g6(evidence.get("G6"), th), "G7": g7(evidence.get("G7"), th),
        "G8": g8(evidence.get("G8"), th), "G9": g9(evidence.get("G9")),
    }
    return {"gates": gates, "decision": decide(gates)}


def collect_evidence(scores: dict | None, projection: dict | None = None) -> dict:
    """The evidence bundle from stored reports: `score-report`'s `gate_evidence` (N, G2, G6, G7, G8) and
    an optional ledger projection for the informational G9."""
    ev = dict((scores or {}).get("gate_evidence") or {})
    if projection is not None:
        ev["G9"] = {"projection": projection, "profile": ev.get("profile")}
    return ev


# ------------------------------------------------------------------------ §17.2 milestones
# Five milestones, each answering one decision (checkpoint simplification 2026-09-26). CP3 (controls) and
# CP5 (evaluator validation) are deleted; CP6, CP7 and CP4 are one development milestone ending in the
# level freeze; CP8-CP9 are one test-build milestone; CP10-CP11 are the final run and its report.
# Book census and rights clearance (formerly CP3b) is a parallel readiness track with one condition,
# `books_ready`, checked before MS4.
MILESTONES = [
    {"ms": "MS1", "name": "Form freeze", "was": ["CP1"],
     "decision": "the protocol forms are fixed before any empirical data that evaluates the benchmark",
     "exit": "owner approval of the current form_freeze_sha256; registry hash recorded", "stop": None,
     "check": "ms1_exit"},
    {"ms": "MS2", "name": "Pilot", "was": ["CP2"],
     "decision": "the real pipeline runs end to end on development crops, and its decoding caps and costs are "
                 "known, before the development set is paid for",
     "exit": "zero crashes; every record validates; c_judge, c_card, c_gen metered; body invariant asserted; "
             "determinism re-run green; font coverage green",
     "stop": "a form change -> new MS1 approval, rerun the pilot; two failed attempts -> re-scope the harness",
     "check": "ms2_exit"},
    {"ms": "MS3", "name": "Development set, sizing and level freeze", "was": ["CP6", "CP7", "CP4"],
     "decision": "the levels, N*, G* and delta* that govern test construction and scoring",
     "exit": "r_hat >= 0.25 and F_bar >= 6; delta* >= 1.1 x MDD_floor; every parameter frozen with value, "
             "artifact id, hash, date",
     "stop": "r_hat < 0.25 or F_bar < 6 -> revise the builder (new MS1 approval); delta* < 1.1 x MDD_floor at "
             "achievable G -> raise delta* or owner decision",
     "check": "ms3_exit"},
    {"ms": "MS4", "name": "Test set and release", "was": ["CP8", "CP9"],
     "decision": "the frozen test set is large enough to run the models",
     "exit": "books_ready; N_cand >= N*/r_hat x 1.25; G2 (yield); release manifest written",
     "stop": "shortfall -> more pages or reserve books; never a gate or level change", "check": "ms4_exit"},
    {"ms": "MS5", "name": "Final run and report", "was": ["CP10", "CP11"],
     "decision": "scale, reduce or stop (§17.6), from one scoring window over the frozen release",
     "exit": "G6, G7, G8 computable; §17.6 decision not UNDECIDABLE", "stop": None, "check": "ms5_exit"},
]
MS3_MIN_R_HAT, MS3_MIN_F_BAR = 0.25, 6       # §17.2 development-set stop rule (plan constants)


def _ms(ms: str, result: str, **kw) -> dict:
    return {"ms": ms, "result": result, **kw}


def ms1_exit(layout: Layout) -> dict:
    gate_params = ("F_bar_min", "G_min", "min_items_per_group", "N_star", "tau_b_min", "delta_star")
    names = {s.name for s in REGISTRY}
    missing = [p for p in gate_params if p not in names]
    if missing:
        return _ms("MS1", FAIL, reason=f"gate parameters not registered: {missing}")
    if not layout.thresholds.exists():
        return _ms("MS1", NOT_RUN, reason="thresholds.json (form-freeze record) absent; owner hash-and-commit pending")
    d = read_json(layout.thresholds)
    ok = d.get("registry_hash") == registry_hash()
    if not ok:
        return _ms("MS1", FAIL, registry_hash=registry_hash(),
                   reason="thresholds.json registry_hash differs from the current forms")
    from .manifests import cp1_approval_problems
    pending = cp1_approval_problems(layout, d.get("frozen_at"))
    return _ms("MS1", NOT_RUN if pending else PASS, registry_hash=registry_hash(),
               reason="; ".join(pending) if pending else None)


def ms2_exit(pilot: dict | None) -> dict:
    need = ("crashes", "invalid_records", "c_judge", "c_card", "c_gen", "body_invariant_ok", "determinism_ok",
            "font_coverage_ok")
    if not pilot or any(pilot.get(k) is None for k in need):
        return _ms("MS2", NOT_RUN, reason="pilot summary missing (paid calls not authorized)")
    ok = pilot["crashes"] == 0 and pilot["invalid_records"] == 0 and pilot["body_invariant_ok"] \
        and pilot["determinism_ok"] and pilot["font_coverage_ok"]
    return _ms("MS2", PASS if ok else FAIL, evidence={k: pilot[k] for k in need})


def ms3_exit(layout: Layout, r_hat: float | None, F_bar: float | None, mdd_floor: float | None) -> dict:
    """Development set (r_hat, F_bar stop rule), sizing design check (delta* >= 1.1 x MDD_floor) and the level
    freeze (every parameter frozen with provenance, read from <root>/thresholds.json)."""
    if r_hat is None or F_bar is None:
        return _ms("MS3", NOT_RUN, reason="development-set cards not measured")
    if r_hat < MS3_MIN_R_HAT or F_bar < MS3_MIN_F_BAR:
        return _ms("MS3", FAIL, r_hat=r_hat, F_bar=F_bar,
                   stop="the builder, not the corpus, is the problem; revise before test construction")
    if not layout.thresholds.exists():
        return _ms("MS3", NOT_RUN, reason="no thresholds.json")
    th = load_thresholds(layout.thresholds)
    if th.profile.value != "production":
        return _ms("MS3", NOT_RUN, reason="thresholds are not the production profile")
    missing = th.missing_for_production()
    if missing:
        return _ms("MS3", NOT_RUN, reason="level freeze incomplete", missing=missing)
    if mdd_floor is None:
        return _ms("MS3", NOT_RUN, reason="variance probe (MDD_floor) not computed")
    ds, _ = _th(th, "delta_star")
    ok = ds >= 1.1 * mdd_floor
    return _ms("MS3", PASS if ok else FAIL, r_hat=r_hat, F_bar=F_bar, delta_star=ds, mdd_floor=mdd_floor,
               frozen_at=th.frozen_at)


def books_ready(th, n_selectable_test_groups: int | None) -> dict:
    """Readiness track (formerly CP3b), checked before MS4: the R3-approved test books form at least G*
    split groups after grouping (development books excluded)."""
    if n_selectable_test_groups is None:
        return _ms("books_ready", NOT_RUN, reason="census not run (owner book approvals)")
    G_star, err = _th(th, "G_star")
    if G_star is None:
        return _ms("books_ready", BLOCKED, reason=err or "G* not frozen (MS3)")
    return _ms("books_ready", PASS if n_selectable_test_groups >= G_star else FAIL,
               n_selectable_test_groups=n_selectable_test_groups, G_star=G_star)


def ms4_exit(th, N_cand: int | None, r_hat: float | None, g2_result: dict | None) -> dict:
    N_star, _ = _th(th, "N_star")
    f, _ = _th(th, "pool_safety_factor")
    if None in (N_star, f, N_cand, r_hat) or not r_hat:
        return _ms("MS4", NOT_RUN, reason="N*, r_hat or N_cand unknown")
    need = N_star / r_hat * f
    if N_cand < need:
        return _ms("MS4", FAIL, N_cand=N_cand, required=need)
    g2r = (g2_result or {}).get("result", NOT_RUN)
    return _ms("MS4", g2r if g2r in (PASS, FAIL) else NOT_RUN, N_cand=N_cand, required=need, gate=g2_result)


def ms5_exit(gates: dict[str, dict], decision: dict) -> dict:
    computable = {g: gates.get(g, {}).get("result") in (PASS, FAIL) for g in ("G6", "G7", "G8")}
    d = decision.get("decision")
    done = all(computable.values()) and d not in (None, Decision.UNDECIDABLE.value)
    return _ms("MS5", PASS if done else NOT_RUN, computable=computable, decision=d)


# ------------------------------------------------------------------------ §17.3 two-act freeze
def freeze_invariants(layout: Layout) -> dict:
    """Form freeze (registry hash recorded), level freeze (provenance complete, timestamped), and the
    ordering invariants. Level changes themselves are refused by `hmb.config.level_freeze`
    (FreezeViolation); this function only checks and reports."""
    out = {"form_freeze": NOT_RUN, "level_freeze": NOT_RUN, "violations": []}
    if layout.thresholds.exists():
        d = read_json(layout.thresholds)
        out["form_freeze"] = PASS if d.get("registry_hash") == registry_hash() else FAIL
        try:
            th = load_thresholds(layout.thresholds)
            if th.frozen_at and not th.missing_for_production():
                out["level_freeze"] = PASS
                out["frozen_at"] = th.frozen_at
        except Exception as e:                      # malformed provenance
            out["level_freeze"] = FAIL
            out["violations"].append(f"thresholds.json: {e}")
    marker = layout.first_test_artifact_marker
    if marker.exists():
        first = read_json(marker)["utc"]
        if out["level_freeze"] != PASS:
            out["violations"].append("test-book artifacts exist without a complete level freeze")
        elif out.get("frozen_at") and out["frozen_at"] > first:
            out["violations"].append(f"level freeze {out['frozen_at']} after first test artifact {first}")
    return out


__all__ = ["g1", "g2", "g6", "g7", "g8", "g9", "decide", "evaluate_all", "collect_evidence", "MILESTONES",
           "ms1_exit", "ms2_exit", "ms3_exit", "books_ready", "ms4_exit", "ms5_exit", "freeze_invariants",
           "FreezeViolation"]


def register_cli(sub) -> None:
    from .config import fixture_thresholds

    def _eval(a) -> int:
        lay = Layout.from_env(a.root)
        path = a.thresholds or lay.thresholds
        if a.thresholds != "fixture" and not Path(path).exists():
            print(f"REFUSED: {path} does not exist; pass --thresholds <path> (or 'fixture' for engineering runs)")
            return 2
        th = fixture_thresholds() if a.thresholds == "fixture" else load_thresholds(path)
        ev = collect_evidence(read_json(a.scores) if a.scores else None)
        res = evaluate_all(ev, lay, th)
        res["freeze"] = freeze_invariants(lay)
        sys.stdout.write(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
        return 0
    p = sub.add_parser("gates-evaluate", help="G1, G2, G6-G9 and the §17.6 decision from the score report")
    p.add_argument("--scores", default=None, help="score-report output (reports/scores_test_<release>.json)")
    p.add_argument("--thresholds", default=None, help="thresholds.json (default <root>/thresholds.json); "
                                                      "'fixture' for engineering runs")
    p.set_defaults(func=_eval)
