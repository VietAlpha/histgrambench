"""C28 (docs/CONTRACTS.md): Γ point entities need identity (name label or drawn dot); straight
strokes are typed by the shared `classify_straight` on the drawn stroke. Expectations by hand."""
from hmb.synthetic.build import B
from hmb.synthetic.evaluator import oracle_facts
from hmb.synthetic.families import generate
from hmb.synthetic.generate import oracle_graph
from hmb.synthetic.geo_predicates import classify_straight
from hmb.ontology import entity_args, fact_key


def K(pred, *args):
    return fact_key(pred, list(args))


def test_classify_straight_by_hand():
    pts = [[0, 0], [4, 0]]
    assert classify_straight([0, 0], [4, 0], pts, 1e-3)[0] == "segment"          # both ends
    t, a, b = classify_straight([4, 0], [7, 0], pts, 1e-3)
    assert t == "ray" and a == [4.0, 0.0] and b == [7.0, 0.0]                      # origin = pointed end
    assert classify_straight([-1, 0], [5, 0], pts, 1e-3)[0] == "line"              # neither end, extends
    assert classify_straight([0, 3], [4, 3], pts, 1e-3)[0] == "segment"            # zero incident points
    # tipped strokes: long horizontal -> axis; short oblique -> arrow; short horizontal + axis label -> axis
    from hmb.config import fixture_thresholds
    th = fixture_thresholds()                         # axis_min_length_frac = 0.5 (registered FC)
    kw = dict(tipped=True, D=12, parallel_deg=1.5, th=th)
    assert classify_straight([0, 0], [9, 0], [], 1e-3, **kw)[0] == "axis"          # 9 >= 0.5 x 12
    assert classify_straight([0, 0], [5, 0], [], 1e-3, **kw)[0] == "arrow"         # 5 < 6, no label
    assert classify_straight([0, 0], [1, 1], [], 1e-3, **kw)[0] == "arrow"         # oblique
    assert classify_straight([0, 0], [2, 0], [], 1e-3, axis_label=True, **kw)[0] == "axis"
    assert classify_straight([0, 0], [5, 0], [], 1e-3, tipped=True, D=12, parallel_deg=1.5,
                             axis_min_frac=0.4)[0] == "axis"


def mid_point(dot, label):
    b = B("c28", 0)
    b.point("A", (2, 5))
    b.point("B", (8, 5))
    b.add("p_M", "point", {"x": 5.0, "y": 5.0}, "M" if label else None, dot=dot)
    b.seg("p_A", "p_B")
    return b.spec()


def test_identity_less_point_is_internal_geometry(th):
    of = oracle_facts(mid_point(dot=False, label=False), th)
    assert not [f for f in of.facts if "p_M" in entity_args(f[0], list(f[1]))]
    assert "p_M" not in of.types
    dotted = oracle_facts(mid_point(dot=True, label=False), th)
    assert K("between", "p_A", "p_M", "p_B") in dotted.facts and K("point", "p_M") in dotted.facts
    labelled = oracle_facts(mid_point(dot=False, label=True), th)
    assert K("label", "M", "p_M") in labelled.facts and K("on", "p_M", "s_AB") in labelled.facts


def test_stroke_types_from_the_drawing(th):
    b = B("c28", 1)
    b.point("A", (2, 2))
    b.point("P", (3, 6))
    b.point("Q", (7, 6))
    b.add("s_ray", "segment", {"a": "p_A", "b": [8.0, 2.0]})                      # drawn from A to a bare end
    b.add("s_line", "segment", {"a": [1.5, 6.0], "b": [8.5, 6.0]})               # through P, Q, both ends bare
    b.add("s_free", "line", {"a": [2.0, 8.5], "b": [8.0, 8.5], "ext": [0, 1]})    # no incident point
    of = oracle_facts(b.spec(), th)
    assert of.types["s_ray"] == "ray" and K("ray", "s_ray") in of.facts
    assert of.types["s_line"] == "line" and K("line", "s_line") in of.facts
    assert of.types["s_free"] == "segment"
    assert K("on", "p_P", "s_line") in of.facts


def test_family_types_follow_the_rule(th):
    s = generate("coordinate_geometry", 0)
    of = oracle_facts(s, th)
    og = oracle_graph(s, of)
    types = {e["entity_id"]: e["type"] for e in og["entities"]}
    assert types["l_1"] == "line" and types["x_axis"] == "axis" and types["y_axis"] == "axis"
    assert K("line", "l_1") in of.facts


def test_tipped_stroke_requires_the_registered_fraction():
    import pytest
    with pytest.raises(ValueError):
        classify_straight([0, 0], [9, 0], [], 1e-3, tipped=True, D=12, parallel_deg=1.5)


def test_c30_polygon_edges_are_segment_entities(th):
    # triangle ABC drawn as a polygon; AB drawn again as a stroke; BC covered by sub-segments B-M, M-C
    b = B("c30", 0)
    b.point("A", (5, 8))
    b.point("B", (2, 2))
    b.point("C", (8, 2))
    b.point("M", (5, 2))
    b.add("g_1", "polygon", {"v": ["p_A", "p_B", "p_C"]})
    b.seg("p_A", "p_B")
    b.seg("p_B", "p_M")
    b.seg("p_M", "p_C")
    of = oracle_facts(b.spec(), th)
    # One polygon (the cycle A-B-M-C collapses to ABC: M is a straight vertex). AB is the drawn s_AB;
    # BC and CA are synthesized with CONTENT-derived ids from their sorted endpoint ids.
    assert [k for k, t in of.types.items() if t == "polygon"] == ["g_1"]
    assert of.types["s_AB"] == "segment" and "seg_p_A_p_B" not in of.types
    assert of.types["seg_p_B_p_C"] == "segment" and of.types["seg_p_A_p_C"] == "segment"
    assert {K("edge_of", "s_AB", "g_1"), K("edge_of", "seg_p_B_p_C", "g_1"),
            K("edge_of", "seg_p_A_p_C", "g_1")} <= of.facts
    assert of.types["s_BM"] == "segment" and K("between", "p_B", "p_M", "p_C") in of.facts


def test_label_binds_to_point_entity_first(th):
    # the label "M" sits nearer the stroke AB than the dot M, but a point entity within reach wins
    from hmb.synthetic.geo_predicates import bind_labels
    g = {"y_up": True, "texts": [{"text": "M", "x": 5.0, "y": 5.25}],
         "entities": {"p_M": {"type": "point", "g": {"x": 5.0, "y": 5.45}},
                      "s_AB": {"type": "segment", "g": {"p": [2, 5], "q": [8, 5]}}}}
    t = {"eps_on": 1e-3, "tangency_rel": 1e-6, "parallel_deg": 1e-4, "perpendicular_deg": 1e-4, "label_bind": 0.5}
    assert bind_labels(g, t)[0][1] == "p_M"


def test_c30_cevian_triangle_gives_three_polygons_and_marker_first_binding(th):
    # triangle ABC (separate strokes) with M on BC and the cevian AM: cycles ABC, ABM, AMC (hand)
    b = B("c30", 1)
    for n, xy in (("A", (5, 8)), ("B", (2, 2)), ("C", (8, 2)), ("M", (5, 2))):
        b.point(n, xy)
    for p, q in (("p_A", "p_B"), ("p_B", "p_M"), ("p_M", "p_C"), ("p_C", "p_A"), ("p_A", "p_M")):
        b.seg(p, q)
    of = oracle_facts(b.spec(), th)
    polys = sorted(k for k, t in of.types.items() if t == "polygon")
    assert len(polys) == 3
    sizes = sorted(a[1] for p, a in of.facts if p == "polygon")
    assert sizes == [3, 3, 3]
    from hmb.synthetic.geo_predicates import bind_labels
    # a label nearer an undrawn (promoted) point than a drawn dot still binds to the drawn dot
    g = {"y_up": True, "texts": [{"text": "Q", "x": 5.0, "y": 5.0}],
         "entities": {"p_dot": {"type": "point", "g": {"x": 5.0, "y": 5.4, "marker": True}},
                      "p_prom": {"type": "point", "g": {"x": 5.0, "y": 5.1, "marker": False}}}}
    t = {"eps_on": 1e-3, "tangency_rel": 1e-6, "parallel_deg": 1e-4, "perpendicular_deg": 1e-4, "label_bind": 0.5}
    assert bind_labels(g, t)[0][1] == "p_dot"


def test_missing_marker_means_not_drawn():
    from hmb.synthetic.geo_predicates import bind_labels
    # B3 red-team round 2 (a): a point without "marker" is NOT a drawn dot, so a marked point wins
    g = {"y_up": True, "texts": [{"text": "Q", "x": 5.0, "y": 5.0}],
         "entities": {"p_dot": {"type": "point", "g": {"x": 5.0, "y": 5.4, "marker": True}},
                      "p_bare": {"type": "point", "g": {"x": 5.0, "y": 5.1}}}}
    t = {"eps_on": 1e-3, "tangency_rel": 1e-6, "parallel_deg": 1e-4, "perpendicular_deg": 1e-4, "label_bind": 0.5}
    assert bind_labels(g, t)[0][1] == "p_dot"


def test_c34_same_polygons_for_closed_path_or_strokes_and_canonical_order(th):
    """C34: membership does not depend on drawing style; a declared polygon keeps its id with the
    CANONICAL vertex order (start at the smallest corner, positive area)."""
    def fig(declared: bool):
        b = B("c34", 0)
        for n, xy in (("A", (5, 8)), ("B", (2, 2)), ("C", (8, 2))):
            b.point(n, xy)
        if declared:
            b.add("g_1", "polygon", {"v": ["p_A", "p_B", "p_C"]})       # one closed path
        else:
            for p, q in (("p_A", "p_B"), ("p_B", "p_C"), ("p_C", "p_A")):   # three separate strokes
                b.seg(p, q)
        return oracle_facts(b.spec(), th)
    closed, strokes = fig(True), fig(False)
    pc = [e for e in closed.entities if e["type"] == "polygon"]
    ps = [e for e in strokes.entities if e["type"] == "polygon"]
    assert len(pc) == len(ps) == 1
    assert pc[0]["entity_id"] == "g_1" and ps[0]["entity_id"] == "poly_p_A_p_B_p_C"
    # canonical order: B(2,2) first, counter-clockwise B -> C -> A (not the declared A, B, C)
    assert pc[0]["geom"]["v"] == ps[0]["geom"]["v"] == [[2.0, 2.0], [8.0, 2.0], [5.0, 8.0]]


def test_solid_faces_come_from_gamma_polygons(th):
    """Red-team round 2 #3: face_of ranges over the Γ (stroke-cycle) polygons, so a cube drawn only
    with edge segments has its six faces in Γ; a cycle of solid vertices that is not a face is no
    cross-section."""
    for seed in range(20):
        s = generate("simple_solids", seed)
        if s.meta.get("solid") == "cube":
            break
    of = oracle_facts(s, th)
    faces = [a for p, a in of.facts if p == "face_of"]
    assert len(faces) == 6 and all(a[1] == "q_1" for a in faces)
    xs = [a for p, a in of.facts if p == "cross_section_of"]
    assert all(a[0] == "g_x" for a in xs)                          # only the declared cutting triangle
