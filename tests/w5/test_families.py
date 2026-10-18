"""Generator families, oracle graphs, TikZ emission and response envelopes (§12.2)."""
import pytest

from hmb.synthetic.admission import admission_reasons
from hmb.synthetic.evaluator import fact_dicts, oracle_facts
from hmb.synthetic.families import FAMILIES, generate
from hmb.synthetic.generate import oracle_graph
from hmb.synthetic.spec import Spec
from hmb.synthetic.tikz import ENVELOPE_KINDS, EXPECTED_PATH, emit_tikz, picture_sha256, wrap


def admitted_seeds(family, th, want=3, scan=40):
    out = []
    for seed in range(scan):
        s = generate(family, seed)
        if not admission_reasons(s, th):
            out.append(s)
            if len(out) == want:
                break
    return out


@pytest.mark.parametrize("family", FAMILIES)
def test_every_family_admits_specs_with_oracle_graphs(family, th):
    specs = admitted_seeds(family, th)
    assert len(specs) == 3, f"{family}: fewer than 3 admitted specs in 40 seeds"
    for s in specs:
        of = oracle_facts(s, th)
        assert len(of.mgs) >= 8
        og = oracle_graph(s, of)
        assert set(og) == {"spec_id", "family", "seed", "partition", "canvas", "entities", "facts", "tikz"}
        # round trip: the stored spec reproduces its own graph
        s2 = Spec.from_json(s.to_json())
        assert fact_dicts(oracle_facts(s2, th).mgs) == og["facts"]
        # every labelled entity's oracle label is the text bound to it
        for e in og["entities"]:
            if not s.has(e["entity_id"]):               # C30 derived stroke-cycle polygons and edges
                assert (e["type"] == "segment" and e["entity_id"].startswith("seg_")) or                     (e["type"] == "polygon" and e["entity_id"].startswith("poly_"))
                continue
            src = s.ent(e["entity_id"])
            if src.label:
                assert e["label"] == src.label
        assert og["tikz"].startswith("\\begin{tikzpicture}")


def test_generation_is_deterministic():
    for f in FAMILIES:
        assert generate(f, 7).to_json() == generate(f, 7).to_json()
        assert generate(f, 7).to_json() != generate(f, 8).to_json()


def test_family_specific_model_facts(th):
    kinds = {}
    for f in FAMILIES:
        for s in admitted_seeds(f, th, want=3):
            for p, a in oracle_facts(s, th).mgs:
                kinds.setdefault(f, set()).add(p)
    assert "tangent_to" in kinds["circles_tangents_secants"]
    assert "solid" in kinds["simple_solids"] and "hidden_edge" in kinds["simple_solids"]
    assert "curve" in kinds["function_plots"] and "intersect_at" in kinds["function_plots"]
    assert "axis" in kinds["coordinate_geometry"] and "measure_label" in kinds["coordinate_geometry"]
    assert kinds["similarity_congruence"] & {"equal_angle", "equal_length"}


@pytest.mark.parametrize("kind", ("none",) + ENVELOPE_KINDS)
def test_envelopes_extract_byte_identical_span(kind, th):
    from hmb.submission import extract_candidate, normalize
    s = admitted_seeds("triangles_polygons", th, want=1)[0]
    tikz = emit_tikz(s)
    raw, stop = wrap(tikz, kind)
    ex = extract_candidate(raw, stop)
    assert ex.ok, ex.outcome
    assert ex.path.value == EXPECTED_PATH[kind]
    nz = normalize(ex, th)
    assert nz.outcome is None
    assert nz.body_sha256 == picture_sha256(tikz)          # §9.4 invariant on the control body


def test_admission_rejects_coincidence_and_canvas(th):
    s = admitted_seeds("triangles_polygons", th, want=1)[0]
    bad = s.copy()
    bad.ent("p_B").geom.update(bad.ent("p_A").geom)            # B coincides with A
    assert any(r.startswith("coincident") for r in admission_reasons(bad, th))
    out = s.copy()
    out.ent("p_A").geom["x"] = -1.0
    assert any(r.startswith("outside_canvas") for r in admission_reasons(out, th))
    lab = s.copy()
    a, b = lab.ent("p_A"), lab.ent("p_B")
    b.label_pos = list(a.label_pos)                           # label B drawn on top of label A
    rs = admission_reasons(lab, th)
    assert any(r.startswith("label_overlap") for r in rs)
