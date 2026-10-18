"""Regression tests for the ACCEPTED findings of the B4 Gemini red-team, round 1 (those whose mechanism
survived the 2026-09-25 simplification; the interval-overlap, composite and decomposition regressions
went with those mechanisms)."""
from __future__ import annotations

import pytest
from PIL import Image

from hmb import stats
from hmb.adjudication import adjudicate
from hmb.adjudication.judges import BItem, JudgeCtx, Panel, run_binary
from hmb.grading import GradingError, model_track_table, score_cell
from hmb.schemas import CgxFact, CgxPrimitive, FactVerdict, VerdictFile
from hmb.vocab import JudgeQuestion, Outcome

from w4_fixtures import (card, default_truth, fact, judges, thresholds, triangle_card, triangle_cgx,
                         write_submission_stub)


def _q3_scenario():
    c = triangle_card([fact("e9", "point", ["p_E"])])
    c = c.model_copy(update={"entities": c.entities + [c.entities[0].model_copy(update={"entity_id": "p_E",
                                                                                       "label": None})]})
    prims = [CgxPrimitive(prim_id="P7", type="point", geom={"x": 150.0, "y": 20.0}),
             CgxPrimitive(prim_id="P8", type="point", geom={"x": 120.0, "y": 120.0})]
    facts = [CgxFact(cfact_id="c70", predicate="point", args=["P7"]),
             CgxFact(cfact_id="c71", predicate="point", args=["P8"])]
    return c, triangle_cgx(prims, facts)


def _run(lay, images, c, cg, th=None):
    write_submission_stub(lay)
    return adjudicate(c, cg, images[0], images[1], judges=judges(), layout=lay, th=th or thresholds(),
                      ledger=None, allow_live=False)


# ------------------------------------------------------------------ #1 / #15
def test_q3_absence_never_exempts_and_grading_accepts(fake, lay, images):
    # Q3: D -> none (judge-derived absence), E -> o1. D's own facts become judge-sourced ABSENT, never EXEMPT.
    fake.policy = lambda j, q, t: ("none" if "“D”" in t else "o1") if q == "Q3" else default_truth(t)
    c, cg = _q3_scenario()
    vf = _run(lay, images, c, cg)
    fv = {f.fact_id: f for f in vf.facts}
    assert (fv["e4"].verdict, fv["e4"].source) == ("ABSENT", "judges")
    assert (fv["l4"].verdict, fv["l4"].source) == ("ABSENT", "judges")       # was EXEMPT before the fix
    # r7 endpoint_of(p_D, s_4): s_4 is DETERMINISTICALLY absent (no free segment) -> EXEMPT stays legal
    assert (fv["r7"].verdict, fv["r7"].source) == ("EXEMPT", "deterministic")
    r = score_cell(c, None, Outcome.OK, vf, thresholds(), model_id="contestant_x", track="A")
    assert r.per_class["L"]["n_star"] == 4 and r.per_class["L"]["n_exempt"] == 0


def test_q3_undecided_is_decided_by_q1(fake, lay, images):
    def pol(j, q, t):
        if q != "Q3":
            return default_truth(t)
        return "none" if "“D”" in t else {"J1": "o1", "J2": "o2", "J3": "none"}[j]
    fake.policy = pol
    c, cg = _q3_scenario()
    fv = {f.fact_id: f for f in _run(lay, images, c, cg).facts}
    assert (fv["e9"].verdict, fv["e9"].source) == ("PRESENT", "judges")   # no Q3 majority -> Q1 (2026-09-27)


def test_grading_guard_rejects_judge_derived_exempt():
    c = card("hmb_00000000000000aa", [("p0", "point"), ("p1", "point"), ("c0", "circle")],
             [fact("e0", "point", ["p0"]), fact("e1", "point", ["p1"]), fact("r0", "on", ["p1", "c0"])])

    def vf(e1, e1_src, ambiguous):
        return VerdictFile(item_id=c.item_id, model_id="m", track="A", judge_config_hash="h",
                           facts=[FactVerdict(fact_id="e0", verdict="PRESENT", source="extractor"),
                                  FactVerdict(fact_id="e1", verdict=e1, source=e1_src),
                                  FactVerdict(fact_id="r0", verdict="EXEMPT", source="deterministic")],
                           alignment_ambiguous=ambiguous, profile="fixture")
    th = thresholds()
    score_cell(c, None, Outcome.OK, vf("ABSENT", "extractor", []), th, model_id="m", track="A")   # legal
    for bad in (vf("ABSENT", "judges", []), vf("ABSENT", "extractor", ["p1"]), vf("PRESENT", "extractor", [])):
        with pytest.raises(GradingError):
            score_cell(c, None, Outcome.OK, bad, th, model_id="m", track="A")


# ------------------------------------------------------------------ #2
def test_q3_substitution_does_not_consume_q1_third_family(fake, lay, images):
    fake.malformed["J1"] = 1                  # J1 first slot (Q3) fails -> J3 substitutes on Q3 only
    fake.policy = lambda j, q, t: ("none" if "“D”" in t else "o1") if q == "Q3" else \
        ((j != "J2") if "equal in length" in t else default_truth(t))
    c, cg = _q3_scenario()
    vf = _run(lay, images, c, cg)
    assert fake.count("Q3", "J3") == 1
    r8 = next(f for f in vf.facts if f.fact_id == "r8")
    assert [v.judge_family for v in r8.votes] == ["fam_j1", "fam_j2", "fam_j3"]   # J3 still escalates Q1
    assert fake.count("Q1", "J4") == 0


# ------------------------------------------------------------------ #8
def _ctx(lay, th):
    return JudgeCtx(lay, th, None, False, None, "FINAL", "cfg")


def test_third_judge_chosen_per_fact(fake, lay, images):
    th = thresholds()
    items = [BItem(key=f"k{n:02d}", text=f"fact number {n} holds.") for n in range(30)]
    fails = {"n": 0}

    def j1_fails_chunk2(req):
        if req.spec.model_id == "J1" and "fact number 26 " in req.text and fails["n"] < 1:
            fails["n"] += 1
            return True
        return False
    fake.malformed_if = j1_fails_chunk2
    fake.policy = lambda j, q, t: j != "J2"                  # every fact is undecided after two votes
    res = run_binary(JudgeQuestion.Q1, items, images=[images[0]], ctx=_ctx(lay, th), panel=Panel(judges()),
                     seed="s")
    fam = {k: [v.judge_family for v in res.votes[k]] for k in ("k00", "k26")}
    assert fam["k00"] == ["fam_j1", "fam_j2", "fam_j3"]      # chunk 1 voters J1, J2 -> third J3
    assert fam["k26"] == ["fam_j3", "fam_j2", "fam_j1"]      # chunk 2 voters J3, J2 -> third J1 (per fact)
    assert fake.count(judge="J4") == 0


# ------------------------------------------------------------------ #3, #4
def test_bh_uses_full_family_with_uncomputable_pairs():
    pairs = {("a", "b"): {"p": 0.04, "diff": 0.3}, ("a", "c"): {"p": 0.07, "diff": 0.3},
             ("b", "c"): {"status": "zero_variance", "p": None, "diff": 0.0}}
    r = stats.partial_order(["a", "b", "c"], pairs)
    # m = 3: .04 > .1·1/3 and .07 > .1·2/3 -> nothing rejected (with m = 2 both would have been)
    assert r["family_size"] == 3 and r["separated"] == []
    assert r["pairs"]["b|c"]["p_in_family"] == 1.0 and r["pairs"]["b|c"]["status"] == "zero_variance"
    assert stats.partial_order(["a", "b", "c"], {k: v for k, v in pairs.items() if k != ("b", "c")})["family_size"] == 3


def test_separation_is_directional():
    r = stats.partial_order(["a", "b"], {("a", "b"): {"p": 1e-6, "diff": -0.2}})
    assert r["separated"] == [("b", "a")] and r["rank_bands"] == {"a": [2, 2], "b": [1, 1]}


# ------------------------------------------------------------------ #5
def test_valid_at_1_over_all_eligible_items(fake):
    from test_grading import build_cells
    OK, CE = Outcome.OK, Outcome.C_ERROR
    cells = build_cells({
        "m1": [(OK, {}), (OK, {"e0": "INDETERMINATE", "r0": "INDETERMINATE"}), (CE, {}), (OK, {})],
        "m2": [(OK, {}), (OK, {}), (OK, {}), (CE, {})]})
    groups = {i: f"g{k}" for k, i in enumerate(sorted(cells["m1"]))}
    t = model_track_table(cells, groups, thresholds())
    m1 = t["models"]["m1"]
    assert t["N"] == 4                                                 # nothing dropped list-wise
    assert m1["validity"]["Valid@1"] == pytest.approx(3 / 4)            # #{OK}/N over eligible items
    assert m1["validity"]["derived_rates"]["n"] == 4 and m1["validity"]["outcome_distribution"]["n"] == 4
    assert m1["headline"]["estimate"] == pytest.approx((1 + 0.6 + 0 + 1) / 4)   # one item per group


# ------------------------------------------------------------------ #7, #10, #13
def test_sizing_rule_of_thumb_only_at_g12():
    assert stats.sizing_decision(0.04, 0.12, 10)["decision"] == "use_simulation_N_star"
    assert stats.sizing_decision(0.04, 0.12, 12)["decision"] == "pilot_250_sufficient"


def test_registered_judge_max_tokens(fake, lay, images):
    _run(lay, images, triangle_card(), triangle_cgx(), th=thresholds(judge_max_tokens=777))
    assert {r.max_tokens for r in fake.calls} == {777}
