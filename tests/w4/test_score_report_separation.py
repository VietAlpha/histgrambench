"""G7 separation evidence from the all-pairs comparison (owner decision D21), MS3 sizing (dev-sizing), the §7.3 context
re-score and the descriptive diagram-type table, on fixture roots (engineering evidence only, never a benchmark
score)."""
from __future__ import annotations

import json

import pytest

from hmb.cli import main
from hmb.grading import score_cell
from hmb.grading.cell import write_score
from hmb.schemas import FactVerdict, JudgeVote, VerdictFile
from hmb.store import Layout, write_json_immutable
from hmb.vocab import DiagramType, Outcome

from w4_fixtures import card, fact, thresholds

M_A, M_B, M_C = "model_a", "model_b", "model_c"
TYPES = [t.value for t in DiagramType]


def build_root(root, models, *, split="test", n=12) -> Layout:
    """n items in n/3 split groups; the second model misses r0 on even items and fails (C_ERROR, S = 0) on
    items 1 and 5; any other model scores like the first; one item carries a context-assisted fact."""
    lay = Layout(root)
    th = thresholds()
    for k in range(n):
        i = f"hmb_{k:016d}"
        write_json_immutable(lay.p("data", split, f"{i}.json"), {
            "split_group": f"g{k // 3}", "extraction": {"context_char_count": 0},
            "automatic_metadata": {"diagram_type": {"label": TYPES[k % 6] if k != 11 else None,
                                                    "status": "predicted" if k != 11 else "invalid_reply"}}})
        fs = [fact("e0", "point", ["p0"]), fact("e1", "point", ["p1"]), fact("e2", "point", ["p2"]),
              fact("r0", "on", ["p0", "c0"]), fact("r1", "on", ["p1", "c0"]), fact("m0", "equal_length", ["p0", "p1"])]
        fs = [f.model_copy(update={"votes": {"b1": "true", "b2": "true", "b3": "true"},
                                   "context_assisted": k == 0 and f.fact_id == "r1"}) for f in fs]
        cd = card(i, [("p0", "point"), ("p1", "point"), ("p2", "point"), ("c0", "circle")], fs)
        write_json_immutable(lay.card(i), cd.model_dump(mode="json", by_alias=True))
        for m_idx, m in enumerate(models):
            for t in ("A", "B"):
                ok = not (m_idx == 1 and t == "A" and k in (1, 5))
                vf = None
                if ok:
                    miss = m_idx == 1 and k % 2 == 0
                    fvs = [FactVerdict(fact_id=f.fact_id, verdict=("ABSENT" if (miss and f.fact_id == "r0") else "PRESENT"),
                                       source="extractor") for f in fs if f.fact_id != "m0"]
                    fvs.append(FactVerdict(fact_id="m0", verdict="PRESENT", source="judges",
                                           votes=[JudgeVote(judge_id=j, judge_family=f"fam_{j.lower()}", question="Q1",
                                                            positive=True) for j in ("J1", "J2")]))
                    vf = VerdictFile(item_id=i, model_id=m, track=t, judge_config_hash="cfg1", facts=fvs,
                                     profile="fixture")
                    write_json_immutable(lay.verdicts(m, t, i), vf.model_dump(mode="json"))
                write_score(lay, score_cell(cd, None, Outcome.OK if ok else Outcome.C_ERROR, vf, th, model_id=m,
                                            track=t))
    return lay


def _th_file(tmp_path, **over):
    p = tmp_path / "th.json"
    p.write_text(json.dumps(thresholds(**over).to_json()), encoding="utf-8")
    return str(p)


def test_score_report_g7_evidence_is_the_all_pairs_comparison(tmp_path, capsys):
    lay = build_root(tmp_path / "r", [M_A, M_B, M_C], n=18)             # 6 split groups = G_min
    th = _th_file(tmp_path)
    assert main(["--root", str(lay.root), "score-report", "--models", M_A, M_B, M_C, "--track", "A",
                 "--track", "B", "--thresholds", th]) == 0
    rep = json.loads((lay.reports / "scores_test_unreleased.json").read_text(encoding="utf-8"))
    assert "primary_contrast" not in rep
    order = rep["tables"]["A"]["partial_order"]
    g7 = rep["gate_evidence"]["G7"]
    assert g7["G"] == 6 and g7["family_size"] == order["family_size"] == 3 and g7["q"] == order["q"]
    by = {tuple(p["pair"]): p for p in g7["pairs"]}
    assert set(by) == {(M_A, M_B), (M_A, M_C), (M_B, M_C)}
    for (a, b), p in by.items():
        assert (p["diff"], p["separated"]) == (order["pairs"][f"{a}|{b}"]["diff"], order["pairs"][f"{a}|{b}"]["separated"])
    assert by[(M_A, M_B)]["diff"] > 0 and by[(M_A, M_B)]["mdd_floor"] is not None
    assert by[(M_A, M_C)]["status"] != "ok"                             # identical scores: no computable test
    assert g7["ceiling_flag"] is not None
    # secondary / descriptive outputs are labelled and never feed G7
    assert "by_diagram_type" in rep["secondary_descriptive"]
    dt = rep["by_diagram_type"]["A"]
    assert dt["role"] == "secondary_descriptive" and set(dt) - {"role"} == set(TYPES) | {"unlabelled_invalid_reply"}
    assert sum(v["n"] for k, v in dt.items() if k != "role") == 18
    cr = rep["context_rescore"]["A"]
    assert cr["n_context_assisted_facts"] == 1 and cr["N"] == 18 and set(cr["H"]) == {M_A, M_B, M_C}
    capsys.readouterr()
    assert main(["--root", str(lay.root), "gates-evaluate", "--scores", str(lay.reports / "scores_test_unreleased.json"),
                 "--thresholds", th]) == 0
    g = json.loads(capsys.readouterr().out)["gates"]["G7"]
    assert g["result"] == "NOT_RUN" and "fixture-profile evidence" in g["reason"]   # fixture never passes
    assert g["evidence"]["engineering_result"] == "PASS"
    q = {tuple(x["pair"]): x["diff"] for x in g["metric"]["qualifying_pairs"]}
    assert set(q) == {(M_A, M_B), (M_B, M_C)} and q[(M_B, M_C)] == pytest.approx(-q[(M_A, M_B)])   # either sign
    assert g["metric"]["n_eligible_pairs"] == 2                         # (model_a, model_c) is ignored


def test_dev_sizing_from_development_scores(tmp_path, capsys):
    lay = build_root(tmp_path / "r", ["m1", "m2"], split="development")
    th = _th_file(tmp_path, delta_star=0.05, sizing_reps=200)
    assert main(["--root", str(lay.root), "dev-sizing", "--models", "m1", "m2", "--G", "6", "12",
                 "--thresholds", th, "--seed", "3"]) == 0
    rep = json.loads((lay.reports / "dev_sizing_unreleased_seed3.json").read_text(encoding="utf-8"))
    assert rep["n_items"] == 12 and rep["n_groups"] == 4 and rep["split"] == "development"
    assert rep["simulation"]["status"] == "ok" and set(rep["simulation"]["N_star"]) == {"6", "12"}
    assert set(rep["decision"]) == {"6", "12"} and rep["sigma_d"] > 0
    assert main(["--root", str(lay.root), "dev-sizing", "--models", "m1", "m2", "--G", "6",
                 "--thresholds", "prefreeze:DEV"]) == 2                   # no production thresholds.json on this root


def test_dev_sizing_before_the_freeze_groups_development_candidates(tmp_path, monkeypatch):
    """C42 fix: in a pre-freeze context the probe items are development candidates without item records;
    their groups come from the split graph over the development books."""
    from hmb.config import fixture_thresholds
    from hmb.grading import cli as GC
    lay = build_root(tmp_path / "r", ["m1", "m2"], split="development")
    th = fixture_thresholds()
    th.params["delta_star"] = dict(th.params["delta_star"], value=0.05)
    th.params["sizing_reps"] = dict(th.params["sizing_reps"], value=200)
    th.context = "DEV"                                            # the branch keys on is_prefreeze only
    cells, _, _ = GC.load_track_cells(lay, ["m1", "m2"], "A", th)
    items = sorted(cells["m1"])
    monkeypatch.setattr(GC, "item_meta", lambda *a: (_ for _ in ()).throw(AssertionError("no item records read")))
    monkeypatch.setattr(GC, "development_groups", lambda layout, th, its: {i: f"sg{k % 4}" for k, i in enumerate(its)})
    rep = GC.dev_sizing(lay, th, ["m1", "m2"], [6, 12], 3)
    assert rep["n_items"] == len(items) and rep["n_groups"] == 4 and rep["simulation"]["status"] == "ok"
