"""Extractor round trip (formerly V1x; an offline engineering check): generated TikZ -> CGX -> Γ through the
GEOMETRIC correspondence. The unit tests build a CGX record from the fixture's own geometry in PDF convention (points, y down, an arbitrary
page offset) with the shared evaluator at §10.2 tolerances: a perfect extractor, renamed primitive
ids, so precision and recall must be exactly 1."""
import pytest

from hmb.cards.closure import minimal_generating_set
from hmb.synthetic.admission import admission_reasons
from hmb.synthetic.evaluator import diagonal, oracle_facts, spec_geom
from hmb.synthetic.families import generate
from hmb.synthetic.generate import oracle_graph
from hmb.synthetic.geo_predicates import evaluate_geo, tolerances_10_2
from hmb.hashing import sha256_json
from hmb.schemas import CgxRecord
from hmb.synthetic.roundtrip import PT_PER_CM, roundtrip_cell

OFFSET = (13.0, 700.0)        # arbitrary page placement (PDF points)


def admitted(family, th):
    for seed in range(30):
        s = generate(family, seed)
        if not admission_reasons(s, th):
            return s
    raise AssertionError(family)


def to_pdf(v):
    """cm, y up -> PDF points, y down: x' = s x + ox, y' = oy - s y."""
    if isinstance(v, dict):
        out = {}
        for k, x in v.items():
            if k in ("x", "y") and isinstance(x, (int, float)):
                out[k] = PT_PER_CM * x + OFFSET[0] if k == "x" else OFFSET[1] - PT_PER_CM * x
            elif k in ("r", "rx", "ry", "size"):
                out[k] = PT_PER_CM * x
            else:
                out[k] = to_pdf(x)
        return out
    if isinstance(v, list) and len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
        return [PT_PER_CM * v[0] + OFFSET[0], OFFSET[1] - PT_PER_CM * v[1]]
    if isinstance(v, list):
        return [to_pdf(x) for x in v]
    return v


def perfect_cgx(spec, th, drop_pred=None):
    D = diagonal(spec)
    g = spec_geom(spec, tolerances_10_2(th, D))
    ren = {eid: f"q{i}" for i, eid in enumerate(sorted(g["entities"]))}
    ents = {ren[k]: {"type": v["type"], "label": None, "g": to_pdf(v["g"])} for k, v in g["entities"].items()}
    for v in ents.values():                                   # y down flips the ellipse rotation sense
        if "rot_deg" in v["g"]:
            v["g"]["rot_deg"] = -v["g"]["rot_deg"]
    texts = [{"text": t["text"], "x": PT_PER_CM * t["x"] + OFFSET[0], "y": OFFSET[1] - PT_PER_CM * t["y"]}
             for t in g["texts"]]
    geom = {"y_up": False, "entities": ents, "texts": texts}
    Dpt = PT_PER_CM * D
    raw = evaluate_geo(geom, tolerances_10_2(th, Dpt))
    types = {k: v["type"] for k, v in ents.items()}
    mgs = minimal_generating_set([(p, a) for p, a, _ in raw if p != drop_pred], types)
    bound = {a[1]: a[0] for p, a, _ in raw if p == "label"}
    facts = [{"cfact_id": "cf_" + sha256_json([p, [list(x) if isinstance(x, tuple) else x for x in a]])[:12],
              "predicate": p, "args": [list(x) if isinstance(x, tuple) else x for x in a], "margin": None}
             for p, a in mgs]
    labels = [{"text": t["text"], "bbox": [t["x"] - 3, t["y"] - 3, t["x"] + 3, t["y"] + 3],
               "kind": "name", "bound_prim": None} for t in texts]
    return CgxRecord.model_validate({
        "extractor_version": "perfect-test", "item_id": spec.spec_id, "model_id": "m", "track": "C",
        "pdf_sha256": "0" * 64, "diagonal_D": Dpt,
        "primitives": [{"prim_id": k, "type": v["type"], "geom": v["g"], "label": bound.get(k)}
                       for k, v in ents.items()],
        "glyphs": [], "labels": labels, "facts": facts, "path_op_count": 0})


@pytest.mark.parametrize("family", ["circles_tangents_secants", "coordinate_geometry", "angle_trig"])
def test_perfect_extractor_gives_precision_and_recall_one(family, th):
    s = admitted(family, th)
    og = oracle_graph(s, oracle_facts(s, th))
    c = roundtrip_cell(og, perfect_cgx(s, th), th)
    assert c["mapping"]["method"].startswith("label_points")
    assert c["mapping"]["unassigned_prims"] == [] and c["n_cgx_unmapped"] == 0
    assert c["precision"] == 1.0 and c["recall"] == 1.0
    assert "alignment_ambiguity_rate" in c and "alignment_precision" in c     # secondary, reported apart


def test_missing_predicate_lowers_recall_per_predicate(th):
    s = admitted("circles_tangents_secants", th)
    og = oracle_graph(s, oracle_facts(s, th))
    c = roundtrip_cell(og, perfect_cgx(s, th, drop_pred="tangent_to"), th)
    pp = c["per_predicate"]["tangent_to"]
    assert pp["n_gamma"] >= 1 and pp["recovered"] == 0 and c["recall"] < 1.0 and c["precision"] == 1.0


@pytest.mark.docker
def test_real_sandbox_pdf_to_roundtrip_cell(tmp_path, th):
    from hmb.submission.sandbox import sandbox_available
    if not sandbox_available():
        pytest.skip("pinned sandbox image unavailable")
    from hmb.store import Layout
    from hmb.synthetic.roundtrip import run_roundtrip
    cells = run_roundtrip(Layout(tmp_path), th, families=["circles_tangents_secants"], seeds=1)
    c = cells[0]
    assert c["status"] == "OK" and c["mapping"]["method"].startswith("label_points")
    # measured 2026-09-22 on this family: the real extractor recovers Γ exactly
    assert c["precision"] == 1.0 and c["recall"] == 1.0
