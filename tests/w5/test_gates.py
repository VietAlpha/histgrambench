"""§17.5 gates, §17.6 decision table (every row), NOT_RUN propagation, milestone exits, freeze invariants."""
import pytest

from hmb import gates as Gt
from hmb.config import SPECS, fixture_thresholds, level_freeze, production_template
from hmb.schemas import HumanAction
from hmb.store import Layout, append_jsonl
from hmb.vocab import Decision

SHA = "a" * 64


def row(scope, **kw):
    return HumanAction(utc="2026-09-22T00:00:00Z", actor="owner", scope=scope, action="x",
                       description="d", **kw).model_dump(mode="json")


def test_g1_audit_log(tmp_path):
    lay = Layout(tmp_path)
    assert Gt.g1(lay, 100)["result"] == "NOT_RUN"                     # no audit log at all
    append_jsonl(lay.human_audit_log, row("book", book_id="bk_000000000001"))
    append_jsonl(lay.human_audit_log, row("phase", phase="MS3"))
    assert Gt.g1(lay, None)["result"] == "NOT_RUN"
    ok = Gt.g1(lay, 100)
    assert ok["result"] == "PASS" and ok["metric"] == 0.0            # permitted human work is not counted
    append_jsonl(lay.human_audit_log, row("item", item_id="hmb_0000000000000001"))
    bad = Gt.g1(lay, 100)
    assert bad["result"] == "FAIL" and bad["metric"] == pytest.approx(0.01)
    assert Gt.g1(lay, 100, profile="fixture")["result"] == "NOT_RUN"


def G(result, **ev):
    return {"result": result, "evidence": ev}


def all_gates(**over):
    g = {k: G("PASS") for k in Gt.BLOCKING + Gt.MODULATING}
    g.update(over)
    return g


def test_decision_table_rows():
    assert Gt.decide(all_gates(G1=G("FAIL")))["decision"] == Decision.STOP_IMMEDIATELY.value
    assert Gt.decide(all_gates())["decision"] == Decision.SCALE.value        # G2, G6, G8 pass (G9 never passes)
    assert Gt.decide(all_gates(G9=G("INFORMATIONAL")))["decision"] == Decision.SCALE.value  # G9 does not vote
    r = Gt.decide(all_gates(G8=G("FAIL")))                                    # at most 2 of 4 pass
    assert r["decision"] == Decision.SCALE_REDUCED.value and set(r["remediation"]) == {"G8"}
    assert Gt.BLOCKING == ("G1", "G7")                                        # G3-G5 deleted with CP3/CP5
    assert not hasattr(Decision, "STOP_PUBLISH_METHODS")
    assert Gt.decide(all_gates(G7=G("FAIL", ceiling_flag=True, design_ok=True)))["decision"] == \
        Decision.SCALE_REDUCED_WIDENED.value
    assert Gt.decide(all_gates(G7=G("FAIL", ceiling_flag=False, design_ok=True)))["decision"] == Decision.STOP.value
    assert Gt.decide(all_gates(G7=G("FAIL", ceiling_flag=False, design_ok=False)))["decision"] == \
        Decision.UNDECIDABLE.value


def test_not_run_propagates_to_undecidable():
    for g in Gt.BLOCKING:
        for st in ("NOT_RUN", "BLOCKED"):
            assert Gt.decide(all_gates(**{g: G(st)}))["decision"] == Decision.UNDECIDABLE.value
    # modulating: 2 pass, 1 not run -> could be SCALE or SCALE-REDUCED -> undecidable
    r = Gt.decide(all_gates(G8=G("NOT_RUN")))
    assert r["decision"] == Decision.UNDECIDABLE.value
    r = Gt.decide(all_gates(G6=G("NOT_RUN"), G8=G("FAIL")))                  # at most 1 more can pass
    assert r["decision"] == Decision.SCALE_REDUCED.value
    lay_ev = Gt.evaluate_all({}, Layout(__import__("pathlib").Path(".nonexistent_w5")), fixture_thresholds())
    assert lay_ev["decision"]["decision"] == Decision.UNDECIDABLE.value
    assert all(v["result"] in ("NOT_RUN", "BLOCKED") for v in lay_ev["gates"].values())


def test_g2_to_g9_functions(setp):
    th = fixture_thresholds()
    g2ev = {"N": 450, "F_bar": 9, "group_sizes": {f"g{i}": 40 for i in range(7)}, "profile": "production"}
    assert Gt.g2(g2ev, th)["result"] == "BLOCKED"                      # N* not calibrated
    setp(th, "N_star", 400)
    assert Gt.g2(g2ev, th)["result"] == "PASS"
    assert Gt.g2(dict(g2ev, group_sizes={f"g{i}": 40 for i in range(5)}), th)["result"] == "FAIL"   # < G_min
    assert Gt.g2(dict(g2ev, profile="fixture"), th)["result"] == "NOT_RUN"
    assert Gt.g6({"M": 3}, th)["result"] == "NOT_RUN"                  # requires M >= 4
    # G6: min over judge families of the leave-one-judge-family-out Kendall τ_b (score-report `lojfo`)
    g6ev = {"M": 5, "n_judge_families": 3, "min_tau_b": 0.85}
    assert Gt.g6(g6ev, th)["result"] == "PASS"
    assert Gt.g6(dict(g6ev, min_tau_b=0.6), th)["result"] == "FAIL"
    assert Gt.g6(dict(g6ev, n_judge_families=2), th)["result"] == "NOT_RUN"   # LOJFO needs >= 3 families
    assert Gt.g8({"ci": [0.01, 0.2]}, th)["result"] == "PASS"         # effect
    assert Gt.g8({"ci": [-0.03, 0.04]}, th)["result"] == "PASS"       # informative null within ±0.05
    assert Gt.g8({"ci": [-0.03, 0.09]}, th)["result"] == "FAIL"       # straddles +δ*
    g9 = Gt.g9({"projection": {"total": 900.0, "disclaimer": "x"}})   # informational: no ceiling (D2)
    assert g9["result"] == "INFORMATIONAL" and g9["metric"] == 900.0 and not g9["blocking"]
    assert Gt.g9({"projection": {"total": None, "missing": ["c_judge"]}})["result"] == "NOT_RUN"


def test_milestone_exits(tmp_path, setp):
    lay = Layout(tmp_path)
    th = fixture_thresholds()
    assert Gt.ms1_exit(lay)["result"] == "NOT_RUN"
    assert Gt.ms2_exit(None)["result"] == "NOT_RUN"
    assert Gt.ms3_exit(lay, None, None, None)["result"] == "NOT_RUN"
    assert Gt.ms3_exit(lay, 0.2, 9, 0.04)["result"] == "FAIL"             # r_hat < 0.25: revise the builder
    assert Gt.ms3_exit(lay, 0.3, 6, 0.04)["result"] == "NOT_RUN"          # stop rule met, no level freeze yet
    assert Gt.books_ready(th, None)["result"] == "NOT_RUN"
    assert Gt.books_ready(th, 12)["result"] == "BLOCKED"                  # G* not frozen (fixture: None)
    setp(th, "G_star", 10)
    assert Gt.books_ready(th, 12)["result"] == "PASS" and Gt.books_ready(th, 9)["result"] == "FAIL"
    assert Gt.ms4_exit(th, 1000, 0.5, None)["result"] == "NOT_RUN"        # N* not committed
    setp(th, "N_star", 250)
    assert Gt.ms4_exit(th, 500, 0.5, None)["result"] == "FAIL"            # 500 < 250 / 0.5 x 1.25
    assert Gt.ms4_exit(th, 700, 0.5, {"result": "PASS"})["result"] == "PASS"
    assert Gt.ms4_exit(th, 700, 0.5, None)["result"] == "NOT_RUN"         # yield (G2) not computed
    assert Gt.ms5_exit({}, {"decision": "UNDECIDABLE"})["result"] == "NOT_RUN"
    done = {g: {"result": "PASS"} for g in ("G6", "G7", "G8")}
    assert Gt.ms5_exit(done, {"decision": "SCALE"})["result"] == "PASS"
    assert [m["ms"] for m in Gt.MILESTONES] == ["MS1", "MS2", "MS3", "MS4", "MS5"]
    was = [c for m in Gt.MILESTONES for c in m["was"]]
    assert "CP3" not in was and "CP5" not in was and "CP3b" not in was    # deleted / readiness track
    for gone in ("cp3_exit", "cp5_exit", "g3", "g4", "g5", "attempts_from_log"):
        assert not hasattr(Gt, gone)


def test_freeze_invariants(tmp_path):
    lay = Layout(tmp_path)
    assert Gt.freeze_invariants(lay)["level_freeze"] == "NOT_RUN"
    lay.mark_first_test_artifact("crop", "2026-09-22T00:00:00+00:00")
    v = Gt.freeze_invariants(lay)
    assert "test-book artifacts exist without a complete level freeze" in v["violations"]
    lay2 = Layout(tmp_path / "b")
    prov = {n: {"value": (s.plan_value if s.plan_value is not None else 1.0), "calibration_artifact_id": "cal",
                "sha256": SHA, "date": "2026-09-22", "rule": "test"} for n, s in SPECS.items()
            if s.kind != "form_constant"}
    level_freeze(production_template(), prov, lay2, now="2026-09-21T00:00:00+00:00")
    lay2.mark_first_test_artifact("crop", "2026-09-22T00:00:00+00:00")
    ok = Gt.freeze_invariants(lay2)
    assert ok["form_freeze"] == "PASS" and ok["level_freeze"] == "PASS" and ok["violations"] == []
    ms1 = Gt.ms1_exit(lay2)                                         # no owner approval yet: not done
    assert ms1["result"] == "NOT_RUN" and "no MS1 owner approval" in ms1["reason"]
    from hmb import manifests as M
    M.write_form_freeze(lay2, "2026-09-20T00:00:00+00:00")
    M.record_form_freeze_approval(lay2, "owner:test", "2026-09-20T00:00:00+00:00")
    assert Gt.ms1_exit(lay2)["result"] == "PASS"
    assert Gt.ms3_exit(lay2, 0.5, 9, 0.04)["result"] == "PASS"             # levels frozen, delta* 1.0 >= 1.1 x 0.04


def pair(a, b, diff, *, separated=True, mdd_floor: float | None = 0.04, status="ok"):
    return {"pair": [a, b], "status": status, "diff": diff, "bh_adjusted": 0.01 if separated else 0.5,
            "separated": separated, "mdd_floor": mdd_floor}


def g7ev(*pairs, G=8):
    return {"G": G, "q": 0.10, "family_size": len(pairs), "pairs": list(pairs), "ceiling_flag": False}


def test_g7_generic_separation():
    """D21: G7 passes when at least one eligible pair of the all-pairs comparison is BH-separated with
    |d̄| >= δ* (fixture δ* = 0.05) and δ* >= 1.1 × its MDD floor; model names play no part."""
    th = fixture_thresholds()
    weak = pair("model_x", "model_y", 0.01, separated=False)
    ok = Gt.g7(g7ev(weak, pair("model_x", "model_z", 0.08)), th)
    assert ok["result"] == "PASS" and ok["evidence"]["design_ok"] is True
    assert ok["metric"]["qualifying_pairs"] == [{"pair": ["model_x", "model_z"], "diff": 0.08, "bh_adjusted": 0.01,
                                                 "mdd_floor": 0.04}]
    assert ok["metric"]["n_eligible_pairs"] == 2 and ok["metric"]["n_separated"] == 1
    # reversed orientation (b, a) with the negated difference gives the same result
    assert Gt.g7(g7ev(weak, pair("model_z", "model_x", -0.08)), th)["result"] == "PASS"
    # no pair meets the inferential criterion -> FAIL, and design_ok feeds the §17.6 STOP row
    none = Gt.g7(g7ev(weak, pair("model_x", "model_z", 0.08, separated=False)), th)
    assert none["result"] == "FAIL" and none["evidence"]["design_ok"] is True and none["metric"]["qualifying_pairs"] == []
    assert Gt.decide(all_gates(G7=none))["decision"] == Decision.STOP.value
    # separated but below δ*
    assert Gt.g7(g7ev(pair("model_x", "model_z", 0.03)), th)["result"] == "FAIL"
    # MDD floor: δ* = 0.05 < 1.1 × 0.05, the design cannot detect δ* for this pair
    small = Gt.g7(g7ev(pair("model_x", "model_z", 0.08, mdd_floor=0.05)), th)
    assert small["result"] == "FAIL" and small["evidence"]["design_ok"] is False
    assert Gt.decide(all_gates(G7=small))["decision"] == Decision.UNDECIDABLE.value
    nofloor = Gt.g7(g7ev(pair("model_x", "model_z", 0.08, mdd_floor=None)), th)   # red-team #3: never PASS
    assert nofloor["result"] == "FAIL" and nofloor["evidence"]["design_ok"] is False
    # an ineligible pair (IM test not computable) is ignored, even for design_ok
    bad = pair("model_x", "model_w", 0.5, separated=False, mdd_floor=None, status="zero_variance")
    assert Gt.g7(g7ev(bad, pair("model_x", "model_z", 0.08)), th)["evidence"]["design_ok"] is True
    only_bad = Gt.g7(g7ev(bad), th)
    assert only_bad["result"] == "NOT_RUN" and "no eligible pair" in only_bad["reason"]
    # fewer than G_min groups: no inferential claim; fixture evidence never passes
    assert Gt.g7(g7ev(pair("model_x", "model_z", 0.08), G=5), th)["result"] == "NOT_RUN"
    assert Gt.g7(dict(g7ev(pair("model_x", "model_z", 0.08)), profile="fixture"), th)["result"] == "NOT_RUN"


def test_g7_missing_checks_and_g2_none_levels(setp):
    th = fixture_thresholds()
    assert Gt.g7(None, th)["result"] == "NOT_RUN"
    nc = Gt.g7({"status": "fewer_than_two_models"}, th)
    assert nc["result"] == "NOT_RUN" and "fewer_than_two_models" in nc["reason"]
    setp(th, "delta_star", None)
    assert Gt.g7(g7ev(pair("model_x", "model_z", 0.08)), th)["result"] == "BLOCKED"
    setp(th, "N_star", 400)
    setp(th, "min_items_per_group", None)                            # red-team #4: BLOCKED, no TypeError
    g2 = Gt.g2({"N": 450, "F_bar": 9, "group_sizes": {"g": 40}}, th)
    assert g2["result"] == "BLOCKED" and "min_items_per_group" in g2["reason"]
