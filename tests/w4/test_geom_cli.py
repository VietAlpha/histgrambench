"""Shared-geometry regressions, red-team round-2 items, and the Scaffold 4 CLI commands."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
from PIL import Image

from hmb.adjudication import CgxIndex, adjudicate, align, judge_config_hash
from hmb.adjudication.core import AdjudicationError, deterministic_verdicts
from hmb.cli import main
from hmb.grading import score_cell
from hmb.grading.cell import write_score
from hmb.schemas import CgxFact, CgxPrimitive, CgxRecord, FactVerdict, JudgeVote, VerdictFile
from hmb.store import Layout, write_json_immutable
from hmb.vocab import Outcome

from w4_fixtures import card, fact, judges, thresholds, triangle_card, triangle_cgx, write_submission_stub

ITEM = "hmb_0000000000000001"


def _cgx(prims, labels, facts=(), model="m", track="A"):
    return CgxRecord(extractor_version="x", item_id=ITEM, model_id=model, track=track, pdf_sha256="a" * 64,
                     diagonal_D=math.hypot(160, 120), primitives=prims, glyphs=[], labels=labels,
                     facts=list(facts), path_op_count=1)


# ------------------------------------------------------------------ geometry (lead item 1)
def test_label_binding_to_non_point_entities_real_format():
    """C28: no junction points; labels bind to circles / segments. Fails on the old x1/cx parser
    (it returned no distance for non-points, so both circles stayed ALIGNMENT_AMBIGUOUS)."""
    c = card(ITEM, [("c_k", "circle", "k"), ("c_m", "circle", "m"), ("s_a", "segment", "a")],
             [fact("e1", "circle", ["c_k"]), fact("e2", "circle", ["c_m"]), fact("e3", "segment", ["s_a"]),
              fact("l1", "label", ["k", "c_k"]), fact("l2", "label", ["m", "c_m"]), fact("l3", "label", ["a", "s_a"])])
    prims = [CgxPrimitive(prim_id="K1", type="circle", geom={"c": [0.0, 0.0], "r": 20.0}),
             CgxPrimitive(prim_id="K2", type="circle", geom={"c": [100.0, 0.0], "r": 20.0}),
             CgxPrimitive(prim_id="S1", type="segment", geom={"p": [0.0, 80.0], "q": [100.0, 80.0]})]
    labels = [{"text": "k", "bbox": [123.0, -2.0, 127.0, 2.0], "kind": "name", "bound_prim": None},
              {"text": "m", "bbox": [23.0, -2.0, 27.0, 2.0], "kind": "name", "bound_prim": None},
              {"text": "a", "bbox": [48.0, 84.0, 52.0, 88.0], "kind": "name", "bound_prim": None}]
    al = align(c, CgxIndex(_cgx(prims, labels)), thresholds())
    assert al.binding == {"c_k": "K2", "c_m": "K1", "s_a": "S1"}
    assert set(al.method.values()) == {"label"} and not al.ambiguous


def test_label_verdict_uses_shared_binding_for_unbound_runs():
    c = card(ITEM, [("c_k", "circle", "k")], [fact("e1", "circle", ["c_k"]), fact("l1", "label", ["k", "c_k"])])
    prims = [CgxPrimitive(prim_id="K1", type="circle", geom={"c": [0.0, 0.0], "r": 20.0})]
    labels = [{"text": "k", "bbox": [18.0, -2.0, 22.0, 2.0], "kind": "name", "bound_prim": None}]
    idx = CgxIndex(_cgx(prims, labels))
    out, _, _ = deterministic_verdicts(c, align(c, idx, thresholds()), idx, thresholds())
    assert out["l1"].verdict == "PRESENT"


def test_unlabelled_axes_aligned_by_orientation_parameter():
    """Two unlabelled axes with no relations: the axis(e, o) parameter decides the match (§10.3 type
    includes the parameter). Identical unlabelled tick marks stay ambiguous (automorphic -> Q3)."""
    c = card(ITEM, [("ax_x", "axis"), ("ax_y", "axis"), ("t1", "length_tick"), ("t2", "length_tick")],
             [fact("e1", "axis", ["ax_x", "horizontal"]), fact("e2", "axis", ["ax_y", "vertical"]),
              fact("e3", "length_tick", ["t1"]), fact("e4", "length_tick", ["t2"])])
    prims = [CgxPrimitive(prim_id="A1", type="axis", geom={"p": [0.0, 50.0], "q": [100.0, 50.0]}),
             CgxPrimitive(prim_id="A2", type="axis", geom={"p": [50.0, 0.0], "q": [50.0, 100.0]}),
             CgxPrimitive(prim_id="T1", type="length_tick", geom={"p": [20.0, 48.0], "q": [20.0, 52.0]}),
             CgxPrimitive(prim_id="T2", type="length_tick", geom={"p": [80.0, 48.0], "q": [80.0, 52.0]})]
    facts = [CgxFact(cfact_id="f1", predicate="axis", args=["A1", "horizontal"]),
             CgxFact(cfact_id="f2", predicate="axis", args=["A2", "vertical"])]
    al = align(c, CgxIndex(_cgx(prims, [], facts)), thresholds())
    assert al.binding == {"ax_x": "A1", "ax_y": "A2"}
    assert al.ambiguous == {"t1", "t2"}


# ------------------------------------------------------------------ red-team round 2
def test_contestant_family_keyword_and_stage_record(fake, lay, images):
    c, th = triangle_card(), thresholds()
    cg = triangle_cgx().model_copy(update={"model_id": "org/model x"})
    with pytest.raises(AdjudicationError, match="contestant_family"):
        adjudicate(c, cg, images[0], images[1], judges=judges(), layout=lay, th=th, ledger=None, allow_live=False)
    vf = adjudicate(c, cg, images[0], images[1], judges=judges(), layout=lay, th=th, ledger=None,
                    allow_live=False, contestant_family="fam_c")
    assert vf.evaluator_failure is None
    # stage-record fallback lives under model_dir_name(model_id)
    write_submission_stub(lay, model="org_model_x")
    from hmb.adjudication.core import _contestant_family
    assert _contestant_family(lay, cg) == "fam_c"


def test_judge_config_hash_covers_tolerances():
    base = judge_config_hash(judges(), "x", thresholds())
    for k, v in (("eps_on_frac", 0.02), ("tangency_rel", 0.03), ("parallel_deg", 2.0),
                 ("perpendicular_deg", 2.0), ("label_bind_frac", 0.07), ("circle_fit_resid_rel", 0.02)):
        assert judge_config_hash(judges(), "x", thresholds(**{k: v})) != base, k


# ------------------------------------------------------------------ Scaffold 4 CLI
def _build_root(root) -> Layout:
    lay = Layout(root)
    th = thresholds()
    items = [f"hmb_{k:016d}" for k in range(8)]
    for k, i in enumerate(items):
        write_json_immutable(lay.p("data", "test", f"{i}.json"),
                             {"split_group": f"g{k // 2}", "extraction": {"context_char_count": 50 * (k % 2)}})
        fs = [fact("e0", "point", ["p0"]), fact("e1", "point", ["p1"]), fact("e2", "point", ["p2"]),
              fact("r0", "on", ["p0", "c0"]), fact("r1", "on", ["p1", "c0"]), fact("m0", "equal_length", ["p0", "p1"])]
        fs = [f.model_copy(update={"votes": ({"b1": "true", "b2": "true"} if f.fact_id == "r1" else
                                             {"b1": "true", "b2": "true", "b3": "true"})}) for f in fs]
        cd = card(i, [("p0", "point"), ("p1", "point"), ("p2", "point"), ("c0", "circle")], fs)
        write_json_immutable(lay.card(i), cd.model_dump(mode="json", by_alias=True))
        for m in ("m1", "m2"):
            for t in ("A", "B"):
                ok = not (m == "m2" and t == "A" and k in (1, 5))
                outcome = Outcome.OK if ok else Outcome.C_ERROR
                vf = None
                if ok:
                    miss = m == "m2" and k % 2 == 0
                    fvs = [FactVerdict(fact_id=f.fact_id, verdict=("ABSENT" if (miss and f.fact_id == "r0") else "PRESENT"),
                                       source="extractor") for f in fs if f.fact_id != "m0"]
                    fvs.append(FactVerdict(fact_id="m0", verdict="PRESENT", source="judges",
                                           votes=[JudgeVote(judge_id=j, judge_family=f"fam_{j.lower()}", question="Q1",
                                                            positive=True) for j in ("J1", "J2")]))
                    vf = VerdictFile(item_id=i, model_id=m, track=t, judge_config_hash="cfg1",
                                     facts=fvs, profile="fixture")
                    write_json_immutable(lay.verdicts(m, t, i), vf.model_dump(mode="json"))
                write_score(lay, score_cell(cd, None, outcome, vf, th, model_id=m, track=t))
    return lay


def test_score_report_command(tmp_path):
    lay = _build_root(tmp_path / "r")
    assert main(["--root", str(lay.root), "score-report", "--models", "m1", "m2", "--track", "A", "--track", "B", "--thresholds", "fixture"]) == 0
    txt = (lay.reports / "scores_test_unreleased.json").read_text(encoding="utf-8")
    rep = json.loads(txt)
    assert rep["profile"] == "fixture" and rep["claim_status"].startswith("engineering evidence")
    assert rep["judge_config_hashes"] == ["cfg1"] and rep["metric_hash"] and rep["thresholds_hash"]
    assert rep["split"] == "test"
    tA = rep["tables"]["A"]
    assert tA["N"] == 8 and tA["models"]["m2"]["validity"]["Valid@1"] == pytest.approx(6 / 8)
    assert tA["models"]["m1"]["headline"]["G"] == 4 and tA["partial_order"]["family_size"] == 1
    assert set(rep["context_effect"]["per_model"]) == {"m1", "m2"}
    ev = rep["gate_evidence"]
    assert ev["N"] == 8 and ev["G2"]["group_sizes"] == {"g0": 2, "g1": 2, "g2": 2, "g3": 2}
    assert ev["G2"]["F_bar"] == pytest.approx(6.0)
    assert ev["G6"]["M"] == 2 and ev["G6"]["n_judge_families"] == 2
    assert ev["G7"]["G"] == 4 and ev["G7"]["family_size"] == 1          # the all-pairs comparison (D21)
    assert ev["G8"]["ci"] is not None
    # LOJFO: without fam_j2, m0 keeps J1's positive vote -> still PRESENT -> ordering unchanged
    assert rep["lojfo"]["judge_families"] == ["fam_j1", "fam_j2"] and rep["lojfo"]["min_tau_b"] == 1.0
    assert "PASS" not in txt


def test_gates_evaluate_reads_the_score_report(tmp_path, capsys):
    """No hand-built evidence: score-report -> gates-evaluate --scores gives G6/G7/G8 computed values."""
    lay = _build_root(tmp_path / "r")
    thp = tmp_path / "th.json"
    thp.write_text(json.dumps(thresholds().to_json()), encoding="utf-8")
    assert main(["--root", str(lay.root), "score-report", "--models", "m1", "m2", "--track", "A", "--track", "B",
                 "--thresholds", str(thp)]) == 0
    capsys.readouterr()
    rep = json.loads((lay.reports / "scores_test_unreleased.json").read_text(encoding="utf-8"))
    g7 = rep["gate_evidence"]["G7"]
    assert g7["pairs"][0]["pair"] == ["m1", "m2"] and g7["pairs"][0]["diff"] > 0
    assert g7["pairs"][0]["mdd_floor"] is not None
    assert g7["ceiling_flag"] is not None
    assert main(["--root", str(lay.root), "gates-evaluate", "--scores", str(lay.reports / "scores_test_unreleased.json"),
                 "--thresholds", str(thp)]) == 0
    out = json.loads(capsys.readouterr().out)
    g = out["gates"]
    assert g["G7"]["result"] == "NOT_RUN" and "G_min" in g["G7"]["reason"]     # 4 groups < G_min = 6
    assert g["G6"]["result"] == "NOT_RUN" and "M >= 4" in g["G6"]["reason"]
    assert g["G2"]["result"] in ("NOT_RUN", "BLOCKED")               # N_star has no fixture value
    # fixture evidence never passes a gate
    assert all(r["result"] not in ("PASS", "FAIL") for r in g.values())


def test_score_report_refuses_track_c(tmp_path):
    lay = _build_root(tmp_path / "r")
    rec = json.loads(lay.scores("m1", "A", "hmb_0000000000000000").read_text(encoding="utf-8"))
    write_json_immutable(lay.p("scores", "m1", "A", "hmb_00000000000000cc.json"), {**rec, "track": "C",
                                                                                 "item_id": "hmb_00000000000000cc"})
    assert main(["--root", str(lay.root), "score-report", "--models", "m1", "m2", "--track", "A",
                 "--thresholds", "fixture"]) == 2
    with pytest.raises(SystemExit):
        main(["--root", str(lay.root), "score-report", "--models", "m1", "--track", "C"])


def _probe_triples(lay: Layout, n: int, judge_ids=("J1", "J2", "J3")) -> list[dict]:
    """Drift triples over a development record (verdict votes + original Q1 calls; replay probe 2026-09-30)."""
    from w4_fixtures import write_drift_dev
    c = triangle_card()
    return write_drift_dev(lay, c, lay.root / "renders" / "t.png", list(judge_ids), [f.fact_id for f in c.facts[:n]])


def test_drift_probe_command(tmp_path, fake):
    lay = Layout(tmp_path / "r")
    (lay.root / "renders").mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(lay.root / "renders" / "t.png")
    trip = _probe_triples(lay, 3)
    tp = lay.root / "triples.json"
    tp.write_text(json.dumps(trip), encoding="utf-8")
    roster = tmp_path / "roster.yaml"
    roster.write_text("version: roster-1\njudge:\n" + "".join(
        f"  - {{model_id: J{i}, provider: fake, family: fam_j{i}}}\n" for i in (1, 2, 3)), encoding="utf-8")
    base = ["--root", str(lay.root), "drift-probe", "--roster", str(roster), "--triples", str(tp),
            "--thresholds", "fixture"]
    assert main(base + ["--window", "open"]) == 0
    assert main(base + ["--window", "close"]) == 0
    res = json.loads(lay.p("manifests", "drift_probe", "result.json").read_text(encoding="utf-8"))
    assert res["profile"] == "fixture" and res["pass"] is None and res["engineering_valid"] is True
    assert res["open_agreement"] == 1.0 and res["close_agreement"] == 1.0 and res["kappa_min"] == 0.8
    assert set(res["per_judge"]) == {"J1", "J2", "J3"}


def test_single_candidate_binds_unique_type():
    """One right_angle_marked on the card, one compatible candidate, no incidence facts: bound
    ("unique_type"), even when other unmatched entities send the remainder to Hungarian."""
    c = card(ITEM, [("m_T", "right_angle_marked"), ("p_U", "point"), ("p_V", "point")],
             [fact("e1", "right_angle_marked", ["m_T"]), fact("e2", "point", ["p_U"]), fact("e3", "point", ["p_V"])])
    prims = [CgxPrimitive(prim_id="ra1", type="right_angle_marked", geom={"vertex": [0.0, 0.0], "size": 4.0}),
             CgxPrimitive(prim_id="P1", type="point", geom={"x": 10.0, "y": 10.0}),
             CgxPrimitive(prim_id="P2", type="point", geom={"x": 50.0, "y": 10.0})]
    for th in (thresholds(), thresholds(subgraph_max_unmatched=0)):
        al = align(c, CgxIndex(_cgx(prims, [])), th)
        assert al.binding["m_T"] == "ra1" and al.method["m_T"] == "unique_type"
        assert al.ambiguous == {"p_U", "p_V"}                     # two indistinguishable options -> Q3


def test_t3_probe_cells_excluded_and_reported_separately(tmp_path):
    lay = _build_root(tmp_path / "r")
    base = "hmb_0000000000000000"
    t3 = base + "__t3"
    th = thresholds()
    from hmb.schemas import ScoreRecord
    for m in ("m1", "m2"):
        rec = ScoreRecord.model_validate(json.loads(lay.scores(m, "B", base).read_text(encoding="utf-8")))
        write_json_immutable(lay.scores(m, "B", t3), {**rec.model_dump(mode="json"), "item_id": t3, "S": 0.0,
                                                       "outcome_code": "C_ERROR"})
        write_json_immutable(lay.submission_record(m, "B", t3), {"prompt_id": "reconstruction-T3-v1"})
    assert main(["--root", str(lay.root), "score-report", "--models", "m1", "m2", "--track", "A", "--track", "B", "--thresholds", "fixture"]) == 0
    rep = json.loads((lay.reports / "scores_test_unreleased.json").read_text(encoding="utf-8"))
    tB = rep["tables"]["B"]
    assert tB["N"] == 8 and tB["models"]["m1"]["validity"]["Valid@1"] == 1.0      # probe cell not in A/B
    probe = rep["t3_probe_s13_6"]["B"]["m1"]
    assert probe["n"] == 1 and probe["mode"] == "counts_only"                  # no rights manifest here
    assert rep["t3_classification_basis"]["B"]["prompt_id"] == 2


# ------------------------------------------------------------------ B6 integration red-team
def test_cli_thresholds_default_and_banner(tmp_path, capsys):
    lay = _build_root(tmp_path / "r")
    args = ["--root", str(lay.root), "score-report", "--models", "m1", "m2", "--track", "A"]
    assert main(args) == 2 and "thresholds.json does not exist" in capsys.readouterr().out
    assert main(args + ["--thresholds", "fixture"]) == 0
    assert "FIXTURE PROFILE" in capsys.readouterr().out
    (lay.reports / "scores_test_unreleased.json").unlink()
    lay.thresholds.write_text(json.dumps(thresholds().to_json()), encoding="utf-8")   # <root>/thresholds.json
    assert main(args) == 0
    assert "FIXTURE PROFILE" not in capsys.readouterr().out


def test_cli_refuses_profile_mismatch(tmp_path, capsys):
    from hmb.config import production_template
    lay = _build_root(tmp_path / "r")                       # fixture-profile score/verdict/card records
    lay.thresholds.write_text(json.dumps(production_template().to_json()), encoding="utf-8")
    assert main(["--root", str(lay.root), "score-report", "--models", "m1", "m2", "--track", "A"]) == 2
    out = capsys.readouterr().out
    assert "profile mismatch" in out and "'production'" in out and "'fixture'" in out


def test_verdict_file_card_uid(fake, lay, images):
    from hmb.adjudication.core import AdjudicationError
    from hmb.vocab import Profile
    c, cg = triangle_card(), triangle_cgx()
    kw = dict(judges=judges(), layout=lay, ledger=None, allow_live=False, contestant_family="fam_c")
    vf = adjudicate(c, cg, images[0], images[1], th=thresholds(), **kw)
    assert vf.card_uid == c.card_uid
    other = c.model_copy(update={"card_uid": "card_" + "1" * 16})
    with pytest.raises(AdjudicationError, match="judged card"):
        adjudicate(other, cg, images[0], images[1], th=thresholds(), **kw)
    # a stored file without card_uid: reused under fixture with a warning, refused in production
    p = lay.verdicts(cg.model_id, "A", cg.item_id)
    legacy = {**json.loads(p.read_text(encoding="utf-8")), "card_uid": None}
    lay2 = Layout(lay.root.parent / "r2")
    write_json_immutable(lay2.verdicts(cg.model_id, "A", cg.item_id), legacy)
    with pytest.warns(UserWarning, match="no card_uid"):
        adjudicate(c, cg, images[0], images[1], th=thresholds(), **{**kw, "layout": lay2})
    th_p = thresholds()
    th_p.profile = Profile.production
    for k in th_p.params:
        th_p.params[k]["status"] = "frozen"
    lay3 = Layout(lay.root.parent / "r3")
    write_json_immutable(lay3.verdicts(cg.model_id, "A", cg.item_id),
                         {**legacy, "judge_config_hash": judge_config_hash(judges(), cg.extractor_version, th_p),
                          "profile": "production"})
    with pytest.raises(AdjudicationError, match="no card_uid"):
        adjudicate(c.model_copy(update={"profile": "production"}), cg, images[0], images[1], th=th_p,
                   **{**kw, "layout": lay3})


def test_drift_probe_default_thresholds_and_profile_mismatch(tmp_path, fake, capsys):
    lay = Layout(tmp_path / "r")
    (lay.root / "renders").mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(lay.root / "renders" / "t.png")
    tp = lay.root / "triples.json"
    tp.write_text(json.dumps(_probe_triples(lay, 1, ("J1",))), encoding="utf-8")
    roster = tmp_path / "roster.yaml"
    roster.write_text("version: roster-1\njudge:\n  - {model_id: J1, provider: fake, family: fam_j1}\n", encoding="utf-8")
    base = ["--root", str(lay.root), "drift-probe", "--roster", str(roster), "--triples", str(tp)]
    assert main(base + ["--window", "open"]) == 2                  # no <root>/thresholds.json, no silent fixture
    assert main(base + ["--window", "open", "--thresholds", "fixture"]) == 0
    open_rec = json.loads(lay.p("manifests", "drift_probe", "open.json").read_text(encoding="utf-8"))
    lay.p("manifests", "drift_probe", "open.json").write_text(
        json.dumps({**open_rec, "profile": "production"}), encoding="utf-8")
    capsys.readouterr()
    assert main(base + ["--window", "close", "--thresholds", "fixture"]) == 2
    assert "profile mismatch" in capsys.readouterr().out
