"""§11 grading and §14 aggregation tests with hand-derived expectations.

The eight worked cases of component 07 §11.3/§14.1 are tested by name (`test_case_*`); every number in
them is reproduced in the plan text.
"""
from __future__ import annotations

import math
import random

import pytest
from scipy import stats as sps

from hmb.grading import (GradingError, ceiling_flag, complete_items, context_effect, derived_rates, headline,
                         label_errors, metric_hash, model_track_table,
                         outcome_distribution, paired_comparison, pairwise_partial_order, per_class_recall, q5,
                         score_cell, separation_evidence, slice_publishable, valid_rate)
from hmb.schemas import FactVerdict, VerdictFile
from hmb.vocab import Outcome

from w4_fixtures import card, fact, thresholds

ITEM = "hmb_00000000000000aa"
OK, CE = Outcome.OK, Outcome.C_ERROR


def mini_card(n_e=3, n_r=2, n_l=0, item=ITEM):
    """n_e point facts (E), n_r on(p, c0) facts (R), n_l labels (L)."""
    ents = [(f"p{k}", "point") for k in range(max(n_e, 2))] + [("c0", "circle")]
    fs = [fact(f"e{k}", "point", [f"p{k}"]) for k in range(n_e)]
    fs += [fact(f"r{k}", "on", [f"p{k % max(n_e, 2)}", "c0"]) for k in range(n_r)]
    fs += [fact(f"l{k}", "label", [chr(65 + k), f"p{k % max(n_e, 2)}"]) for k in range(n_l)]
    return card(item, ents, fs)


def vfile(c, verdicts: dict, model="m1", track="A", label_info=None, failure=None):
    fv = []
    for f in c.facts:
        if not f.eligible:
            continue
        v = verdicts.get(f.fact_id, "PRESENT")
        src = "deterministic" if v == "EXEMPT" else ("judges" if f.decidability == "model" else "extractor")
        b = {"label_match": label_info[f.fact_id]} if label_info and f.fact_id in label_info else None
        fv.append(FactVerdict(fact_id=f.fact_id, verdict=v, source=src, binding=b))
    return VerdictFile(item_id=c.item_id, model_id=model, track=track, judge_config_hash="h",
                       evaluator_failure=failure, facts=fv, profile="fixture")


def sc(c, outcome, vf, th=None, model="m1", track="A"):
    return score_cell(c, None, outcome, vf, th or thresholds(), model_id=model, track=track)


# ------------------------------------------------------------------ the eight worked cases (§11.3, §14.1)
def test_case_1_all_facts_present():
    assert sc(mini_card(), OK, vfile(mini_card(), {})).S == 1.0                      # 5/5


def test_case_2_one_absent_fact():
    assert sc(mini_card(), OK, vfile(mini_card(), {"r0": "ABSENT"})).S == pytest.approx(4 / 5)


def test_case_3_one_contradicted_fact_scores_like_absent_and_is_counted():
    r = sc(mini_card(), OK, vfile(mini_card(), {"r0": "CONTRADICTED"}))
    assert r.S == pytest.approx(4 / 5) and r.per_class["R"]["n_contradicted"] == 1


def test_case_4_one_indeterminate_fact_counts_as_not_recovered():
    r = sc(mini_card(), OK, vfile(mini_card(), {"e2": "INDETERMINATE"}))
    assert r.S == pytest.approx(4 / 5) and r.per_class["E"]["n_indeterminate"] == 1


@pytest.mark.parametrize("code", [o for o in Outcome if o != Outcome.OK])
def test_case_5_compile_or_extraction_failure_scores_zero(code):
    r = sc(mini_card(), code, None)
    assert (r.S, r.evaluator_failure) == (0.0, None) and r.outcome_code == code.value
    # the non-valid cell keeps the card's own counts (common denominator)
    assert r.per_class["E"]["n_star"] == 3 and r.per_class["R"]["n_star"] == 2


def test_case_6_item_with_no_eligible_facts_is_not_scored_at_all():
    c = card(ITEM, [("p0", "point")], [fact("e0", "point", ["p0"], eligible=False)])
    with pytest.raises(GradingError, match="not in I"):
        sc(c, CE, None)


def _cells(model, S_by_item, track="A"):
    """Cells with the requested scores on the 5-fact mini card (S in multiples of 0.2)."""
    out = {}
    for item, S in S_by_item.items():
        c = mini_card(item=item)
        miss = round((1 - S) * 5)
        out[item] = sc(c, OK, vfile(c, {f"e{k}": "ABSENT" for k in range(miss)} if miss <= 3 else
                                    {**{f"e{k}": "ABSENT" for k in range(3)}, **{f"r{k}": "ABSENT" for k in range(miss - 3)}},
                                    model=model, track=track), model=model, track=track)
        assert out[item].S == pytest.approx(S)
    return out


GROUPS = {"i1": "g1", "i2": "g1", "i3": "g1", "i4": "g2", "i5": "g3", "i6": "g3"}


def test_case_7_unequal_group_sizes():
    # g1: 1.0, 0.6, 0.8 (mean 0.8); g2: 0.4; g3: 0.6, 0.6 (mean 0.6)
    # H = (0.8 + 0.4 + 0.6)/3 = 0.6, although the plain item mean is 4.0/6 = 0.667;
    # IM: sd of group means = 0.2, se = 0.2/√3, t_{0.975, 2} = 4.3027 -> CI = 0.6 ± 0.4968
    m = _cells("m1", {"i1": 1.0, "i2": 0.6, "i3": 0.8, "i4": 0.4, "i5": 0.6, "i6": 0.6})
    h = headline(m, sorted(m), GROUPS, thresholds())
    half = sps.t.ppf(0.975, 2) * 0.2 / math.sqrt(3)
    assert h["estimate"] == pytest.approx(0.6) and h["G"] == 3 and h["n_per_group"] == {"g1": 3, "g2": 1, "g3": 2}
    assert h["ci"] == pytest.approx([0.6 - half, 0.6 + half]) and half == pytest.approx(0.4968, abs=1e-4)
    assert h["inferential_claim_allowed"] is False                  # G = 3 < G_min = 6


def test_case_8_model_comparison_with_clustered_observations():
    # b scores 0.2 lower than a on every g1 item, equal on g2 and g3: d_g = (0.2, 0, 0)
    a = _cells("a", {"i1": 1.0, "i2": 0.6, "i3": 0.8, "i4": 0.4, "i5": 0.6, "i6": 0.6})
    b = _cells("b", {"i1": 0.8, "i2": 0.4, "i3": 0.6, "i4": 0.4, "i5": 0.6, "i6": 0.6})
    its, th = sorted(a), thresholds()
    pc = paired_comparison(a, b, its, GROUPS, th)
    # the estimate equals the difference of the headlines (same items, same groups)
    assert pc["estimate"] == pytest.approx(headline(a, its, GROUPS, th)["estimate"] - headline(b, its, GROUPS, th)["estimate"])
    assert pc["estimate"] == pytest.approx(0.2 / 3)
    sd = float(sps.tstd([0.2, 0.0, 0.0]))                            # 0.11547
    half = sps.t.ppf(0.975, 2) * sd / math.sqrt(3)
    assert pc["ci"] == pytest.approx([0.2 / 3 - half, 0.2 / 3 + half]) and pc["G"] == 3
    # the naive item-level t (6 independent items) would be far narrower: the group is the unit
    naive = sps.t.ppf(0.975, 5) * sps.tstd([0.2, 0.2, 0.2, 0, 0, 0]) / math.sqrt(6)
    assert naive < half


# ------------------------------------------------------------------ the rest of the score definition
def test_score_classes_are_the_card_classes():
    # asc-2.0.0: a card carries E, R and L only (layout is extractor-internal), and every class is scored
    from hmb.grading import CLASSES, HEADLINE_CLASSES
    assert CLASSES == HEADLINE_CLASSES == ("E", "R", "L")
    r = sc(mini_card(n_l=1), OK, vfile(mini_card(n_l=1), {}))
    assert set(r.per_class) == {"E", "R", "L"} and not hasattr(r, "X")


def test_labels_are_scored_with_entities_and_relations():
    c = mini_card(n_e=2, n_r=0, n_l=2)
    li = {"l0": {"strict": False, "error": "transcription", "extracted": ["Α"]},
          "l1": {"strict": False, "error": "attachment"}}
    r = sc(c, OK, vfile(c, {"l0": "CONTRADICTED", "l1": "ABSENT"}, label_info=li))
    assert r.S == pytest.approx(2 / 4)                               # 2 points present, 2 labels missed
    assert label_errors({r.item_id: r}, [r.item_id]) == {"attachment": 1, "transcription": 1, "missing": 0}


def test_exempt_removed_from_numerator_and_denominator():
    # p1 deterministically absent (e1 ABSENT) so on(p1, c0) = r1 is EXEMPT; r0 ABSENT
    # -> scorable = 4 (e0 P, e1 A, e2 P, r0 A), present = 2 -> S = 0.5 (not 2/5)
    c = mini_card()
    r = sc(c, OK, vfile(c, {"e1": "ABSENT", "r1": "EXEMPT", "r0": "ABSENT"}))
    assert r.S == pytest.approx(0.5) and r.per_class["R"]["n_star"] == 1 and r.per_class["R"]["n_exempt"] == 1


def test_judge_exempt_rejected():
    c = mini_card()
    vf = vfile(c, {})
    vf.facts[0] = FactVerdict(fact_id="e0", verdict="EXEMPT", source="judges")
    with pytest.raises(GradingError, match="EXEMPT"):
        sc(c, OK, vf)


def test_evaluator_failure_is_recorded_and_the_cell_is_scored():
    c = mini_card()
    r = sc(c, OK, vfile(c, {"r0": "INDETERMINATE", "r1": "INDETERMINATE"}, failure="judge_failure"))
    assert r.S == pytest.approx(3 / 5) and r.evaluator_failure == "judge_failure"


def test_denominators_do_not_depend_on_judge_verdicts():
    """§11.0 invariant: n_i^c and n_i^c* are functions of the card and the outcome code only."""
    c = mini_card(n_e=3, n_r=2, n_l=1)
    ids = [f.fact_id for f in c.facts if f.eligible]
    ref = sc(c, CE, None)
    den = lambda r: {k: (v["n"], v["n_star"]) for k, v in r.per_class.items()}  # noqa: E731
    rng = random.Random(7)
    vals = ["PRESENT", "ABSENT", "CONTRADICTED", "INDETERMINATE"]
    for _ in range(200):
        assert den(sc(c, OK, vfile(c, {i: rng.choice(vals) for i in ids}))) == den(ref)


def test_metric_hash_is_stable_and_readable_in_production():
    from hmb.config import production_template
    assert metric_hash(thresholds()) == metric_hash(thresholds())
    assert metric_hash(thresholds()) != metric_hash(thresholds(q5_tau_E=0.7))
    assert metric_hash(production_template())             # no calibrated input: no level freeze needed


# ------------------------------------------------------------------ diagnostics and aggregation
def build_cells(model_scores: dict[str, list], track="A"):
    """model -> list of (outcome, verdict dict) per item; items share the same 5-fact card."""
    out = {}
    for m, rows in model_scores.items():
        cells = {}
        for k, (o, v) in enumerate(rows):
            c = mini_card(item=f"hmb_{k:016d}")
            cells[c.item_id] = sc(c, o, vfile(c, v, model=m, track=track) if o == OK else None,
                                  model=m, track=track)
        out[m] = cells
    return out


def test_common_denominator_nothing_dropped():
    cells = build_cells({"m1": [(OK, {}), (OK, {"e0": "INDETERMINATE", "r0": "INDETERMINATE"}), (CE, {})],
                         "m2": [(OK, {"r0": "ABSENT"}), (OK, {}), (OK, {})]})
    its = complete_items(cells)
    assert len(its) == 3                                           # every item counts for every model
    assert [cells["m1"][i].S for i in its] == pytest.approx([1.0, 0.6, 0.0])
    assert valid_rate(cells["m1"], its) == pytest.approx(2 / 3)
    del cells["m2"][its[0]]
    with pytest.raises(ValueError, match="no cell"):
        complete_items(cells)


def test_per_class_shares():
    cells = build_cells({"m": [(OK, {"r0": "CONTRADICTED", "e0": "INDETERMINATE"}), (OK, {})]})["m"]
    pc = per_class_recall(cells, list(cells))
    assert pc["R"]["contradicted_share"] == pytest.approx(1 / 4) and pc["E"]["indeterminate_share"] == pytest.approx(1 / 6)
    assert pc["R"]["R"] == pytest.approx(0.75) and set(pc) == {"E", "R", "L"}


def test_outcome_distribution():
    c = mini_card()
    r1 = sc(c, OK, vfile(c, {}))
    r2 = sc(mini_card(item="hmb_00000000000000bb"), CE, None)
    cells = {r1.item_id: r1, r2.item_id: r2}
    od = outcome_distribution(cells.values())
    assert sum(od["fractions"].values()) == pytest.approx(1.0) and od["counts"]["C_ERROR"] == 1
    dr = derived_rates(cells.values())
    assert dr["Valid@1"] == 0.5 and dr["Compile@1"] == 0.5 and dr["Timeout@1"] == 0.0


def test_track_c_never_pooled():
    c = mini_card()
    r = sc(c, OK, vfile(c, {}, track="C"), track="C")
    with pytest.raises(ValueError, match="never pooled"):
        outcome_distribution([r])


def test_q5_by_hand():
    # item 0: R^E = 1, one CONTRADICTED relation -> counts; item 1: R^E = 2/3 < 0.8 -> excluded;
    # item 2: R^E = 1, no contradiction -> Q5 = 1/2
    cells = build_cells({"m": [(OK, {"r0": "CONTRADICTED"}), (OK, {"e0": "ABSENT", "r0": "CONTRADICTED"}), (OK, {})]})["m"]
    r = q5(cells, list(cells), thresholds())
    assert r["n"] == 2 and r["Q5"] == pytest.approx(0.5)


def test_context_effect_by_hand():
    # per item: A = (0.6, 0.6, 0, 0), B = (0.8, 0, 1.0, 0); groups g1 = {0, 1}, g2 = {2, 3}
    def cellset(track, rows):
        out = {}
        for k, (o, v) in enumerate(rows):
            c = mini_card(item=f"hmb_{k:016d}")
            out[c.item_id] = sc(c, o, vfile(c, v, track=track) if o == OK else None, track=track)
        return out
    A = {"m": cellset("A", [(OK, {"e0": "ABSENT", "r0": "ABSENT"}), (OK, {"e0": "ABSENT", "e1": "ABSENT"}),
                            (CE, {}), (CE, {})])}
    B = {"m": cellset("B", [(OK, {"e0": "ABSENT"}), (CE, {}), (OK, {}), (CE, {})])}
    its = sorted(A["m"])
    groups = {its[0]: "g1", its[1]: "g1", its[2]: "g2", its[3]: "g2"}
    r = context_effect(A, B, its, groups, thresholds())
    # d = (0.2, −0.6, 1.0, 0): group means (−0.2, 0.5) -> ΔH = 0.15; ΔValid@1 = 2/4 − 2/4 = 0
    assert r["per_model"]["m"]["delta_H"] == pytest.approx(0.15) and r["per_model"]["m"]["delta_valid"] == 0.0
    assert r["pooled"]["estimate"] == pytest.approx(0.15) and r["pooled"]["G"] == 2


def test_separation_evidence_reads_the_all_pairs_order_with_mdd_floors():
    rng = random.Random(3)
    a = build_cells({"a": [(OK, {"r0": rng.choice(["PRESENT", "ABSENT"])}) for _ in range(24)]})["a"]
    b = build_cells({"b": [(OK, {"r0": "ABSENT", "e0": rng.choice(["PRESENT", "ABSENT"])}) for _ in range(24)]})["b"]
    c = build_cells({"c": [(OK, {}) for _ in range(24)]})["c"]
    cells, its = {"a": a, "b": b, "c": c}, sorted(a)
    groups = {i: f"g{k % 8}" for k, i in enumerate(its)}
    order = pairwise_partial_order(cells, its, groups, thresholds())
    ev = separation_evidence(cells, order, its, groups, thresholds())
    assert ev["G"] == 8 and ev["family_size"] == 3 and ev["q"] == 0.10
    assert [p["pair"] for p in ev["pairs"]] == [["a", "b"], ["a", "c"], ["b", "c"]]
    for p in ev["pairs"]:                                     # the same BH result and difference, no second test
        r = order["pairs"]["|".join(p["pair"])]
        assert (p["diff"], p["separated"], p["bh_adjusted"]) == (r["diff"], r["separated"], r["bh_adjusted"])
        assert p["mdd_floor"] is not None and p["mdd_floor"] >= 0
    ab = paired_comparison(a, b, its, groups, thresholds())
    assert ev["pairs"][0]["diff"] == pytest.approx(ab["estimate"]) and ab["estimate"] > 0
    assert separation_evidence({"a": a}, None, its, groups, thresholds())["status"] == "fewer_than_two_models"


def test_ceiling_flag():
    top = build_cells({"t": [(OK, {})] * 12})["t"]                    # every item 1.0, SD 0 within groups
    groups = {i: f"g{k % 4}" for k, i in enumerate(sorted(top))}
    assert ceiling_flag({"t": top}, sorted(top), groups, thresholds())["flag"] is True
    mid = build_cells({"m": [(OK, {"r0": "ABSENT", "r1": "ABSENT"} if k % 2 else {}) for k in range(12)]})["m"]  # H = 0.8
    assert ceiling_flag({"m": mid}, sorted(mid), groups, thresholds())["flag"] is False


def test_slice_suppression():
    th = thresholds()
    assert slice_publishable(30, 3, 150, th)["publish_scores"]
    assert slice_publishable(29, 3, 150, th)["mode"] == "counts_only"
    assert not slice_publishable(30, 2, 150, th)["publish_scores"]


def test_model_track_table_smoke():
    rng = random.Random(1)
    rows = lambda p: [(OK if rng.random() < p else CE, {"r0": rng.choice(["PRESENT", "ABSENT"])}) for _ in range(12)]  # noqa: E731
    cells = build_cells({"m1": rows(0.9), "m2": rows(0.6)})
    groups = {i: f"g{k % 6}" for k, i in enumerate(sorted(cells["m1"]))}
    t = model_track_table(cells, groups, thresholds())
    m = t["models"]["m1"]
    assert t["N"] == 12 and m["headline"]["G"] == 6 and m["headline"]["method"] == "ibragimov_mueller_group_t"
    assert set(m) == {"headline", "validity", "per_class"}   # one headline, two diagnostic families
    assert t["partial_order"]["status"] == "ok" and t["partial_order"]["family_size"] == 1


def test_score_files_of_a_colon_model_id_are_found(tmp_path):
    # locked contestant ids contain ':' (us.meta.llama4-maverick-17b-instruct-v1:0); write_score stores them under
    # safe_model_dir, and load_cells must read that same directory (live flow 2026-09-26: score-report refused)
    from hmb.grading.cell import write_score
    from hmb.grading.cli import load_cells
    from hmb.store import Layout
    lay = Layout(tmp_path / "r")
    mid = "us.vendor.model-v1:0"
    c = mini_card()
    rec = sc(c, OK, vfile(c, {}, model=mid), model=mid)
    write_score(lay, rec)
    assert list(load_cells(lay, mid, "A", thresholds())) == [ITEM]
