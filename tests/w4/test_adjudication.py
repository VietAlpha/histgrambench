"""§10 adjudication tests. Expected values are derived by hand in the comments."""
from __future__ import annotations

import pytest

from hmb.adjudication import (Alignment, CgxIndex, adjudicate, decide, deterministic_verdicts, evaluate_drift,
                              judge_config_hash, recompute_without_family, select_judges)
from hmb.adjudication.judges import JudgeCtx, ask_once, parse_response
from hmb.config import production_template
from hmb.schemas import CgxFact, CgxPrimitive, JudgeVote, VerdictFile
from hmb.store import read_json
from hmb.vocab import JudgeQuestion

from w4_fixtures import (card, default_truth, fact, judges, thresholds, triangle_card, triangle_cgx,
                         write_drift_dev, write_submission_stub)


# ------------------------------------------------------------------ the majority rule (§10.5)
def test_majority_rule_by_hand():
    # two concurring answers decide; a disagreement or a missing answer is a tie (-> third judge)
    assert decide([True, True]) == "positive" and decide([False, False]) == "negative"
    assert decide([True, False]) == "tie" and decide([True, None]) == "tie" and decide([]) == "tie"
    # after the third vote: two out of three
    assert decide([True, False, True]) == "positive" and decide([False, True, False]) == "negative"
    assert decide([True, None, False]) == "tie" and decide([True, None, None]) == "tie"
    # leave-one-judge-family-out re-decides the remaining votes with min_concurring = 1
    assert decide([True], min_concurring=1) == "positive" and decide([True, False], min_concurring=1) == "tie"


# ------------------------------------------------------------------ parsing (§10.8)
@pytest.mark.parametrize("text,status,answers", [
    ("not json", "malformed", {}),
    ("[1, 2]", "malformed", {}),
    ('{"answers": [{"id": "s01", "answer": "yes"}]}', "malformed", {}),
    ('{"answers": [{"id": "s09", "answer": true}]}', "malformed", {}),
    ('{"answers": [{"id": "s01", "answer": true}, {"id": "s01", "answer": false}]}', "malformed", {}),
    ('{"answers": []}', "ok", {}),                      # every answer missing -> the third judge
    ('{"refusal": true}', "refusal", {}),
    ('```json\n{"answers": [{"id": "s01", "answer": true}]}\n```', "ok", {"s01": True}),
    # answer-level salvage (2026-09-26): a bad entry costs only its own statement
    ('{"answers": [{"id": "s01", "answer": true}, {"id": "s09", "answer": false}]}', "ok", {"s01": True}),
    ('{"answers": [{"id": "s01", "answer": "yes"}, {"id": "s02", "answer": false}]}', "ok", {"s02": False}),
    ('{"answers": [{"id": "s01", "answer": true}, {"id": "s02", "ans', "ok", {"s01": True}),     # cut off
])
def test_parse_response_answer_level(text, status, answers):
    p = parse_response(text, JudgeQuestion.Q1, ["s01", "s02"])
    assert p.status == status and p.answers == answers


def test_production_refuses_fake_response(fake, lay, images):
    th = production_template()
    th.params["judge_max_tokens"] = {"value": 2048, "status": "frozen", "provenance": None}
    ctx = JudgeCtx(lay, th, None, False, None, "FINAL", "cfg")
    with pytest.raises(RuntimeError, match="fake"):
        ask_once(judges()[0], JudgeQuestion.Q1, "s01: x", [images[0]], ["s01"], ctx)


# ------------------------------------------------------------------ the full triangle cell
def run_triangle(lay, images, **kw):
    write_submission_stub(lay)
    th = kw.pop("th", None) or thresholds()
    return adjudicate(kw.pop("card", None) or triangle_card(), kw.pop("cgx", None) or triangle_cgx(),
                      images[0], images[1], judges=judges(), layout=lay, th=th, ledger=None,
                      allow_live=False, **kw)


def test_triangle_cell_end_to_end(fake, lay, images):
    vf = run_triangle(lay, images)
    v = {f.fact_id: f.verdict for f in vf.facts}
    assert vf.evaluator_failure is None
    for k in ("e1", "e2", "e3", "e5", "e6", "e7", "r1", "r2", "r3", "r4", "r5", "r6", "l1", "l2", "l3"):
        assert v[k] == "PRESENT", k
    assert v["e4"] == "ABSENT" and v["e8"] == "ABSENT"            # D and AD are not drawn
    assert v["r7"] == "EXEMPT" and v["l4"] == "EXEMPT"            # argument entity absent & counted
    assert "z1" not in v                                          # ineligible facts are never judged
    # r8 (model): the first two judges (panel order J1, J2) both say yes -> PRESENT, no third call
    r8 = next(f for f in vf.facts if f.fact_id == "r8")
    assert r8.source == "judges" and len(r8.votes) == 2
    assert {x.judge_family for x in r8.votes} == {"fam_j1", "fam_j2"} and fake.count("Q1", "J3") == 0
    # the extra circle K1 costs no call: the extras channel (ladder, Q2) was removed 2026-09-26
    assert {r.purpose for r in fake.calls} == {"Q1"} and len(fake.calls) == 2
    # Q1 inputs: the render only (never the crop); no evidence string is requested or stored
    q1 = [r for r in fake.calls if r.purpose == "Q1"]
    assert q1 and all(r.image_paths == [str(images[0])] for r in q1)
    assert all("crop" not in r.image_paths[0] for r in q1) and all("evidence" not in r.text for r in q1)
    assert r8.evidence is None
    # immutable verdict file with judge_config_hash
    stored = VerdictFile.model_validate(read_json(lay.verdicts("contestant_x", "A", "hmb_0000000000000001")))
    assert stored.judge_config_hash == vf.judge_config_hash == judge_config_hash(judges(), "cgx-test-1", thresholds())


def test_no_rejudging_after_freeze(fake, lay, images):
    run_triangle(lay, images)
    n = len(fake.calls)
    run_triangle(lay, images)
    assert len(fake.calls) == n


def test_geo_only_cell_makes_no_judge_call(fake, lay, images):
    c = triangle_card()
    c = c.model_copy(update={"facts": [f for f in c.facts if f.fact_id != "r8"]})
    cg = triangle_cgx()
    cg = cg.model_copy(update={"facts": [f for f in cg.facts if f.cfact_id != "c07"],
                               "primitives": [p for p in cg.primitives if p.prim_id != "K1"]})
    vf = run_triangle(lay, images, card=c, cgx=cg)
    assert vf.evaluator_failure is None and fake.calls == []


def test_malformed_reply_is_not_retried_with_the_identical_prompt(fake, lay, images):
    fake.malformed["J1"] = 1                       # one malformed reply -> the other family, never a retry
    vf = run_triangle(lay, images)
    r8 = next(f for f in vf.facts if f.fact_id == "r8")
    assert {x.judge_id for x in r8.votes} == {"J2", "J3"}
    assert fake.count("Q1", "J1") == 1 and vf.evaluator_failure is None


def test_malformed_falls_back_to_other_family(fake, lay, images):
    fake.malformed["J1"] = 99                      # J1 always malformed -> the other family (J3)
    vf = run_triangle(lay, images)
    r8 = next(f for f in vf.facts if f.fact_id == "r8")
    assert {x.judge_id for x in r8.votes} == {"J2", "J3"}
    assert fake.count("Q1", "J1") == 1 and vf.evaluator_failure is None


def test_both_families_fail_flags_the_cell_and_keeps_extractor_verdicts(fake, lay, images):
    # §10.8: J1 and its substitute J3 both malformed -> the Q1 slot fails. The cell is still scored and
    # flagged; the judged fact is forced to the only answer (J2, 2026-09-27); every extractor verdict stands.
    fake.malformed["J1"] = 99
    fake.malformed["J3"] = 99
    fake.malformed["J4"] = 99      # repair 2026-09-26: otherwise J2 + J4 legitimately decide at escalation
    vf = run_triangle(lay, images)
    v = {f.fact_id: f.verdict for f in vf.facts}
    assert vf.evaluator_failure == "judge_failure"
    assert v["r8"] == "PRESENT" and v["r1"] == "PRESENT" and v["e4"] == "ABSENT" and len(v) == 20
    assert next(f for f in vf.facts if f.fact_id == "r8").evidence == "forced: earliest panel judge's answer"


def test_refusal_recorded_separately(fake, lay, images):
    fake.refuse |= {"J1", "J3"}
    vf = run_triangle(lay, images)
    assert vf.evaluator_failure == "judge_refusal"


def test_missing_answer_goes_to_the_third_judge(fake, lay, images):
    fake.policy = lambda j, q, t: None if ("equal in length" in t and j == "J2") else default_truth(t)
    vf = run_triangle(lay, images)
    r8 = next(f for f in vf.facts if f.fact_id == "r8")
    # (+, missing) is not two concurring answers -> J3 (+) -> two out of three -> PRESENT
    assert r8.verdict == "PRESENT" and [x.positive for x in r8.votes] == [True, None, True]
    assert fake.count("Q1", "J3") == 1


def test_disagreement_goes_to_third_judge_only(fake, lay, images):
    fake.policy = lambda j, q, t: (j != "J2") if "equal in length" in t else default_truth(t)
    vf = run_triangle(lay, images)
    r8 = next(f for f in vf.facts if f.fact_id == "r8")
    # (+, −) -> J3 votes (+) -> 2 of 3 -> PRESENT
    assert len(r8.votes) == 3 and r8.verdict == "PRESENT"
    assert fake.count("Q1", "J4") == 0            # never a fourth vote
    # J3 saw only the undecided fact
    j3 = [r for r in fake.calls if r.spec.model_id == "J3" and r.purpose == "Q1"]
    assert len(j3) == 1 and "equal in length" in j3[0].text
    body = j3[0].text.split("STATEMENTS:")[1]
    assert len([ln for ln in body.splitlines() if ln[:1] == "s" and ln[3:4] == ":"]) == 1


def test_undecided_facts_are_forced_and_the_cell_is_scored(fake, lay, images):
    # J2 and J3 never answer Q1: only J1 votes -> no two concurring answers -> forced to J1 (2026-09-27)
    extra = [fact(f"m{k:02d}", "equal_angle", [f"s_{1 + k % 3}", f"s_{1 + (k + 1) % 3}"]) for k in range(10)]
    fake.policy = lambda j, q, t: None if j in ("J2", "J3") and q == "Q1" else True
    vf = run_triangle(lay, images, card=triangle_card(extra))
    judged = [f for f in vf.facts if f.source == "judges"]
    assert len(judged) == 11 and all(f.verdict == "PRESENT" and f.evidence == "forced: earliest panel judge's answer" for f in judged)
    assert vf.evaluator_failure is None                     # answers were missing, no call failed


def test_chunking_25(fake, lay, images):
    extra = [fact(f"m{k:02d}", "equal_angle", [f"s_{1 + k % 3}", f"s_{1 + (k + 1) % 3}"]) for k in range(29)]
    run_triangle(lay, images, card=triangle_card(extra))
    q1 = [r for r in fake.calls if r.purpose == "Q1" and r.spec.model_id == "J1"]
    n_stmts = [len([ln for ln in r.text.split("STATEMENTS:")[1].splitlines() if ln[:1] == "s" and ln[3:4] == ":"])
               for r in q1]
    # 30 model facts (r8 + 29) in fact_id order -> chunks of 25 and 5 (no planted controls)
    assert n_stmts == [25, 5]


def test_contestant_and_builder_families_excluded():
    from hmb.schemas import ModelSpec
    js = [ModelSpec(model_id="B", provider="fake", family="fam_b1", role="judge"),
          ModelSpec(model_id="C", provider="fake", family="fam_c", role="judge")] + judges()
    sel = select_judges(js, builder_families={"fam_b1"}, contestant_family="fam_c")
    assert [j.model_id for j in sel] == ["J1", "J2", "J3", "J4"]
    sec = select_judges(js, builder_families={"fam_b1"}, contestant_family="fam_c", configuration="secondary")
    assert [j.model_id for j in sec][:1] == ["B"]


# ------------------------------------------------------------------ deterministic pieces
def _index(extra_facts=(), extra_prims=()):
    return CgxIndex(triangle_cgx(extra_prims=extra_prims, extra_facts=extra_facts))


def test_contradicted_from_exclusion_table():
    idx = _index(extra_facts=[CgxFact(cfact_id="c50", predicate="on", args=["P1", "K1"])])
    assert idx.e_incompatible("inside", ["P1", "K1"])            # E1: on xor inside
    assert idx.e_incompatible("left_of", ["P2", "P1"])           # E4: render has left_of(P1, P2)
    assert not idx.e_incompatible("left_of", ["P1", "P3"])


def test_geo_contradicted_verdict_and_anchor_b_exempt():
    # CONTRADICTED is kept (Q5): a card relation the render fails with an E-incompatible fact
    fs = [fact("e1", "point", ["p_A"]), fact("e2", "point", ["p_B"]), fact("e3", "point", ["p_C"]),
          fact("e9", "point", ["p_X"]), fact("e5", "segment", ["s_1"]), fact("e6", "segment", ["s_2"]),
          fact("r10", "parallel", ["s_1", "s_2"]), fact("r9", "concyclic", [["p_A", "p_B", "p_C", "p_X"]])]
    c = card("hmb_0000000000000001", [("p_A", "point", "A"), ("p_B", "point", "B"), ("p_C", "point", "C"),
                                      ("p_X", "point"), ("s_1", "segment"), ("s_2", "segment")], fs)
    al = Alignment(binding={"p_A": "P1", "p_B": "P2", "p_C": "P3", "s_1": "S1", "s_2": "S2"}, ambiguous={"p_X"})
    idx = _index(extra_facts=[CgxFact(cfact_id="c51", predicate="intersect_at", args=["S1", "S2", "P2"])])
    out, to_judge, _ = deterministic_verdicts(c, al, idx, thresholds())
    assert out["r10"].verdict == "CONTRADICTED"                  # E2: the render has S1, S2 crossing
    assert out["r9"].verdict == "EXEMPT"                         # anchor B, p_X not uniquely bound
    assert out["e9"].verdict == "INDETERMINATE" and not to_judge


# ------------------------------------------------------------------ re-derivations from stored votes
def test_leave_one_judge_family_out(fake, lay, images):
    vf = run_triangle(lay, images)
    out = recompute_without_family(vf, triangle_card(), "fam_j2")
    # r8 keeps only J1's positive vote -> majority of the remaining vote -> PRESENT; geo facts unchanged
    assert out["r8"] == "PRESENT" and out["r1"] == "PRESENT"
    # a 2-1 split loses its majority when a majority family is left out
    r8 = next(f for f in vf.facts if f.fact_id == "r8")
    split = r8.model_copy(update={"votes": [JudgeVote(judge_id=j, judge_family=f"fam_{j.lower()}", question=JudgeQuestion.Q1,
                                                      positive=a) for j, a in (("J1", True), ("J2", False),
                                                                               ("J3", True))]})
    vf2 = vf.model_copy(update={"facts": [split if f.fact_id == "r8" else f for f in vf.facts]})
    assert recompute_without_family(vf2, triangle_card(), "fam_j1")["r8"] == "INDETERMINATE"
    assert recompute_without_family(vf2, triangle_card(), "fam_j2")["r8"] == "PRESENT"


# ------------------------------------------------------------------ protocol helpers
def test_drift_probe_window_rule():
    th = thresholds()                                      # fixture κ_min = α_min = 0.8
    ref = {f"t{i}": (i % 2 == 0) for i in range(10)}
    good = dict(ref)
    bad = {k: (not v if k in ("t0", "t1", "t2") else v) for k, v in ref.items()}     # 0.7 agreement
    assert evaluate_drift(ref, good, good, th)["valid"]
    r = evaluate_drift(ref, good, bad, th)
    assert r["agreement_close"] == pytest.approx(0.7) and not r["valid"]


def test_judge_config_hash_sensitivity():
    th = thresholds()
    h = judge_config_hash(judges(), "x1", th)
    assert h == judge_config_hash(judges(), "x1", th)
    assert h != judge_config_hash(judges(), "x2", th)
    assert h != judge_config_hash(judges()[::-1], "x1", th)


# ------------------------------------------------------------------ Q3 (ALIGNMENT_AMBIGUOUS)
def _q3_scenario():
    c = triangle_card([fact("e9", "point", ["p_E"])])
    c = c.model_copy(update={"entities": c.entities + [c.entities[0].model_copy(update={"entity_id": "p_E",
                                                                                       "label": None})]})
    prims = [CgxPrimitive(prim_id="P7", type="point", geom={"x": 150.0, "y": 20.0}),
             CgxPrimitive(prim_id="P8", type="point", geom={"x": 120.0, "y": 120.0})]
    facts = [CgxFact(cfact_id="c70", predicate="point", args=["P7"]),
             CgxFact(cfact_id="c71", predicate="point", args=["P8"])]
    return c, triangle_cgx(prims, facts)


def test_q3_resolves_ambiguous_alignment(fake, lay, images):
    # p_D (labelled D, not drawn) and p_E (unlabelled) have no structure: two embeddings onto the free
    # points P7, P8 -> both ALIGNMENT_AMBIGUOUS -> Q3 (o1 = P7, o2 = P8). Judges: D -> none, E -> o1.
    fake.policy = lambda j, q, t: ("none" if "“D”" in t else "o1") if q == "Q3" else default_truth(t)
    c, cg = _q3_scenario()
    vf = run_triangle(lay, images, card=c, cgx=cg)
    v = {f.fact_id: f.verdict for f in vf.facts}
    assert vf.alignment_ambiguous == ["p_D", "p_E"]
    assert v["e4"] == "ABSENT" and v["e9"] == "PRESENT" and v["r7"] == "EXEMPT"
    q3 = [r for r in fake.calls if r.purpose == "Q3"]
    assert [r.spec.model_id for r in q3] == ["J1", "J2"] and all(r.image_paths == [str(images[0])] for r in q3)


def test_q3_disagreement_uses_third_family_majority(fake, lay, images):
    def pol(j, q, t):
        if q != "Q3":
            return default_truth(t)
        if "“D”" in t:
            return "none"
        return {"J1": "o1", "J2": "o2", "J3": "o1"}[j]
    fake.policy = pol
    c, cg = _q3_scenario()
    vf = run_triangle(lay, images, card=c, cgx=cg)
    al = read_json(lay.p("adjudication", "alignment", "contestant_x", "A", "hmb_0000000000000001.json"))
    assert al["binding"]["p_E"] == "P7" and al["method"]["p_E"] == "q3"
    assert fake.count("Q3", "J3") == 1 and fake.count("Q3", "J4") == 0
    assert {f.fact_id: f.verdict for f in vf.facts}["e9"] == "PRESENT"


def test_hungarian_fallback_on_structural_similarity():
    # No labels, so nothing is bound before stage 3. Signatures: p_X {on(·, round)} = P1's; k = K1's;
    # s has no relations -> similarity 0 -> ALIGNMENT_AMBIGUOUS under Hungarian.
    from hmb.adjudication import align
    from hmb.schemas import CgxRecord
    c = card("hmb_0000000000000001", [("p_X", "point"), ("k", "circle"), ("s", "segment")],
             [fact("e1", "point", ["p_X"]), fact("e2", "circle", ["k"]), fact("e3", "segment", ["s"]),
              fact("r1", "on", ["p_X", "k"])])
    cg = CgxRecord(extractor_version="x", item_id="hmb_0000000000000001", model_id="m", track="A",
                   pdf_sha256="a" * 64, diagonal_D=100.0, path_op_count=3, glyphs=[], labels=[],
                   primitives=[CgxPrimitive(prim_id="P1", type="point", geom={"x": 10.0, "y": 0.0}),
                               CgxPrimitive(prim_id="P2", type="point", geom={"x": 50.0, "y": 40.0}),
                               CgxPrimitive(prim_id="K1", type="circle", geom={"c": [0.0, 0.0], "r": 10.0}),
                               CgxPrimitive(prim_id="S1", type="segment",
                                            geom={"p": [30.0, 30.0], "q": [60.0, 30.0]})],
                   facts=[CgxFact(cfact_id="f1", predicate="on", args=["P1", "K1"])])
    al = align(c, CgxIndex(cg), thresholds(subgraph_max_unmatched=0))      # force the Hungarian branch
    # s: the only segment and the only segment candidate -> unique type (not ambiguous)
    assert al.binding == {"p_X": "P1", "k": "K1", "s": "S1"}
    assert al.method == {"p_X": "hungarian", "k": "unique_type", "s": "unique_type"}
    al2 = align(c, CgxIndex(cg), thresholds())                             # exact subgraph branch
    assert al2.binding == al.binding and al2.method["p_X"] == "subgraph"


def test_drift_probe_runner(fake, lay, images):
    """Replay probe (owner decision 2026-09-30): the judge's original development Q1 call is re-sent verbatim
    (chunk, ENTITIES block, render) and its answer is compared with the judge's own recorded answer."""
    from hmb.adjudication import run_drift_probe
    from hmb.adjudication.protocol import own_q1_votes
    c = triangle_card([fact("x1", "circle", ["k_1"])])
    c = c.model_copy(update={"entities": [*c.entities, *card("i", [("k_1", "circle")], []).entities]})
    triples = write_drift_dev(lay, c, images[0], ["J1"], ["r1", "x1"])     # J1 originally said True to both
    th = thresholds()
    got = run_drift_probe(triples, judges()[0], layout=lay, th=th, ledger=None, allow_live=False,
                          phase="FINAL", window_tag="open")
    r1, x1 = (t["triple_id"] for t in triples)
    assert got == {r1: True, x1: False}                       # default_truth: 'circle' statements false
    ref = {t: v for t, (_, v) in own_q1_votes(lay, triples, "J1").items()}
    assert ref == {r1: True, x1: True}
    assert evaluate_drift(ref, got, got, th)["agreement_open"] == 0.5
    assert len(fake.calls) == 1 and fake.calls[0].text.endswith(triples[1]["fact_text"] + chr(10))   # one replayed call
    assert run_drift_probe(triples, judges()[1], layout=lay, th=th, ledger=None, allow_live=False,
                           phase="FINAL", window_tag="open") == {}                      # J2 never voted


def test_drift_probe_window_id_keeps_windows_apart_in_a_shared_cache(fake, lay, images, tmp_path):
    """2026-10-01: a new window on a root whose cache holds an earlier window's probe calls (a copied cache) must
    call the judges again, never replay the earlier window; close reuses its own window's id."""
    import json
    import shutil
    from hmb.adjudication.cli import drift_probe
    c = triangle_card()
    triples = write_drift_dev(lay, c, images[0], ["J1"], ["r1"])
    tp = tmp_path / "triples.json"
    tp.write_text(json.dumps(triples), encoding="utf-8")
    th, js = thresholds(), judges()[:1]
    first = drift_probe(lay, th, window="open", judges=js, triples_path=tp, allow_live=True)
    assert len(fake.calls) == 1 and first["window_id"]
    shutil.move(lay.p("manifests", "drift_probe", "open.json"), tmp_path / "old_open.json")   # fresh window
    second = drift_probe(lay, th, window="open", judges=js, triples_path=tp, allow_live=True)
    assert len(fake.calls) == 2 and second["window_id"] != first["window_id"]
    drift_probe(lay, th, window="close", judges=js, triples_path=tp, allow_live=True)
    closed = json.loads(lay.p("manifests", "drift_probe", "close.json").read_text("utf-8"))
    assert closed["window_id"] == second["window_id"] and len(fake.calls) == 3


def test_q3_failure_sends_the_ambiguous_entities_to_q1(fake, lay, images):
    # Q3 fails (J1 and J3 malformed on Q3 only): p_D and p_E get no binding -> their facts are decided
    # by Q1 (2026-09-27), every other fact is still decided and the cell is flagged, not dropped
    fake.malformed_if = lambda req: req.purpose == "Q3" and req.spec.model_id in ("J1", "J3")
    c, cg = _q3_scenario()
    vf = run_triangle(lay, images, card=c, cgx=cg)
    v = {f.fact_id: f.verdict for f in vf.facts}
    assert vf.evaluator_failure == "judge_failure"
    assert v["e4"] == "PRESENT" and v["e9"] == "PRESENT" and v["r1"] == "PRESENT"
    assert {f.fact_id: f.source for f in vf.facts}["e9"] == "judges"


# ------------------------------------------------------------------ EXEMPT on a model-class argument (2026-09-26)
def _curve_card():
    """The triangle plus a parabola cv_1 the render does not draw: curve(...) is model-decided (Q1), on(p_A, cv_1)
    is a geo relation on it; r10 parallel(s_1, s_2) is E-incompatible with the render's crossing (CONTRADICTED)."""
    base = triangle_card()
    ents = [(e.entity_id, e.type, e.label) for e in base.entities] + [("cv_1", "curve")]
    extra = [fact("e9", "curve", ["cv_1", "parabola"]), fact("r9", "on", ["p_A", "cv_1"]),
             fact("r10", "parallel", ["s_1", "s_2"])]
    return card(base.item_id, ents, base.facts + extra)


@pytest.mark.parametrize("curve_answer", [True, False, "EXEMPT"])
def test_absent_model_class_entity_never_exempts_and_the_cell_scores(fake, lay, images, curve_answer):
    """A point/line/curve cell: alignment finds no primitive for the curve, whose existence fact goes to the
    judges. The relation on it used to be EXEMPT from that absence, and grading refused the whole cell
    ("EXEMPT without a deterministically absent argument entity"). Now EXEMPT depends only on deterministic
    evidence, so n* is the same whatever the judges answer, and a judge's out-of-vocabulary answer ("EXEMPT")
    is a missing answer for that statement only."""
    from hmb.adjudication import recompute_without_family
    from hmb.grading.cell import score_cell
    from hmb.grading.tables import q5
    from hmb.vocab import Outcome
    fake.policy = lambda j, q, t: curve_answer if "curve" in t else default_truth(t)
    cg = triangle_cgx(extra_facts=[CgxFact(cfact_id="c51", predicate="intersect_at", args=["S1", "S2", "P2"])])
    c = _curve_card()
    vf = run_triangle(lay, images, card=c, cgx=cg)
    v = {f.fact_id: f for f in vf.facts}
    assert set(v) == {f.fact_id for f in c.facts if f.eligible}             # every eligible fact has a verdict
    assert v["r9"].verdict != "EXEMPT" and v["r9"].source == "extractor"   # stays in the denominator
    assert v["r7"].verdict == "EXEMPT" and v["r7"].source == "deterministic"   # geo absence still exempts
    assert v["r10"].verdict == "CONTRADICTED"
    assert v["r8"].verdict == "PRESENT" and len(v["r8"].votes) == 2        # unaffected judged answer kept
    e9 = v["e9"]
    if curve_answer == "EXEMPT":                                          # skipped entry -> third judge -> tie
        assert e9.verdict == "INDETERMINATE" and all(x.positive is None for x in e9.votes)
        # J3's one-statement reply has no valid entry: malformed, recorded as an evaluator failure (§10.8)
        assert fake.count("Q1", "J3") >= 1 and vf.evaluator_failure == "judge_failure"
    else:
        assert e9.verdict == ("PRESENT" if curve_answer else "ABSENT")
        assert {x.judge_family for x in e9.votes} == {"fam_j1", "fam_j2"} and vf.evaluator_failure is None
    rec = score_cell(c, None, Outcome.OK, vf, thresholds(), model_id="contestant_x", track="A")
    n_star = {k: rec.per_class[k]["n_star"] for k in ("E", "R", "L")}
    assert n_star == {"E": 9, "R": 9, "L": 3}                              # identical for every judge answer
    assert rec.per_class["R"]["n_exempt"] == 1 and rec.per_class["R"]["n_contradicted"] == 1
    assert q5({c.item_id: rec}, [c.item_id], thresholds())["status"] in ("ok", "insufficient_items")
    lojfo = recompute_without_family(vf, c, "fam_j1")                      # LOJFO inputs intact
    assert lojfo["r7"] == "EXEMPT" and lojfo["r10"] == "CONTRADICTED" and set(lojfo) == set(v)


def test_c43_over_cap_cgx_routes_every_fact_to_q1():
    """C43: a CGX above cgx_max_primitives carries no evidence; no fact is decided by the extractor (so none is
    ABSENT or EXEMPT without a judge) and every eligible fact goes to Q1."""
    from hmb.adjudication import align
    from hmb.adjudication.alignment import card_scope
    from hmb.schemas import CgxRecord
    c = triangle_card()
    cg = CgxRecord(**{**triangle_cgx().model_dump(), "primitives": [], "facts": [], "labels": [], "glyphs": [],
                      "primitives_over_cap": 1185})
    idx = CgxIndex(cg)
    al = align(c, idx, thresholds())
    out, to_judge, _ = deterministic_verdicts(c, al, idx, thresholds())
    _, facts = card_scope(c)
    assert not al.absent and not al.binding and al.not_aligned == set(card_scope(c)[0])
    assert out == {} and {f.fact_id for f in to_judge} == {f.fact_id for f in facts}


def test_q3_tiebreak_forced_choice_by_the_failed_family(fake, lay, images):
    """2026-09-27 tie-break: J2 fails on the full Q3 prompt, J3 substitutes and disagrees with J1; the failed
    family J2 then breaks the tie with the forced-choice prompt listing only the answers already given."""
    fake.malformed_if = lambda req: req.spec.model_id == "J2" and req.purpose == "Q3" and "Options:" not in req.text

    def pol(j, q, t):
        if q != "Q3":
            return default_truth(t)
        if "“D”" in t:
            return "none"
        return {"J1": "o1", "J3": "o2", "J2": "o1"}[j]
    fake.policy = pol
    c, cg = _q3_scenario()
    vf = run_triangle(lay, images, card=c, cgx=cg)
    tb = [r for r in fake.calls if r.purpose == "Q3" and "Options:" in r.text]
    assert [r.spec.model_id for r in tb] == ["J2"]
    assert "Options: o1, o2" in tb[0].text and "do not refuse" in tb[0].text
    al = read_json(lay.p("adjudication", "alignment", "contestant_x", "A", "hmb_0000000000000001.json"))
    assert al["binding"]["p_E"] == "P7" and {f.fact_id: f.verdict for f in vf.facts}["e9"] == "PRESENT"


def test_recorded_image_resolves_a_foreign_host_path_by_suffix_and_hash(tmp_path):
    from hmb.adjudication.protocol import recorded_image
    from hmb.hashing import sha256_file
    from hmb.store import Layout
    lay = Layout(tmp_path / "hmb_data")
    p = lay.root / "compiled" / "m_0" / "A" / "hmb_x" / "render.png"
    p.parent.mkdir(parents=True)
    p.write_bytes(b"png-bytes")
    win = r"C:\Users\someone\proj\hmb_data\compiled\m_0\A\hmb_x\render.png"
    assert recorded_image(lay, win, sha256_file(p)) == p              # Windows path recorded on another host
    assert recorded_image(lay, "/old/root/hmb_data/compiled/m_0/A/hmb_x/render.png", None) == p
    assert recorded_image(lay, str(p), "ignored-when-the-path-exists") == p
    with pytest.raises(FileNotFoundError):
        recorded_image(lay, win, "0" * 64)                              # same suffix, different bytes


def test_cpu_pool_gives_the_same_verdicts_as_inline(fake, lay, images, tmp_path):
    from hmb import cpu_pool
    from hmb.store import Layout
    inline = run_triangle(lay, images).model_dump(mode="json")
    lay2 = Layout(tmp_path / "pooled")
    lay2.root.mkdir()
    cpu_pool.configure(2)
    try:
        assert cpu_pool.active()
        pooled = run_triangle(lay2, images).model_dump(mode="json")
    finally:
        cpu_pool._POOL.shutdown()
        cpu_pool._POOL = None
    assert pooled == inline
    align_dir = ("adjudication", "alignment", "contestant_x", "A", "hmb_0000000000000001.json")
    assert read_json(lay2.p(*align_dir)) == read_json(lay.p(*align_dir))
