"""Predicate evaluator on hand-computed geometry (expectations derived by hand, not by the code)."""
import math

import pytest

from hmb.synthetic.geo_predicates import evaluate_geo, holds_geo
from hmb.ontology import fact_key


def P(x, y):
    return {"type": "point", "g": {"x": x, "y": y}}


def SEG(p, q):
    return {"type": "segment", "g": {"p": list(p), "q": list(q)}}


def tol(D, eps_rel=1e-6):
    return {"eps_on": eps_rel * D, "tangency_rel": 1e-6, "parallel_deg": 1e-4, "perpendicular_deg": 1e-4,
            "label_bind": 0.06 * D}


def facts(geom, t):
    return {(p, a) for p, a, _ in evaluate_geo(geom, t)}


def K(pred, *args):
    return fact_key(pred, list(args))


@pytest.fixture
def tri345():
    # Right triangle A(0,0) B(4,0) C(0,3); incircle r = (3+4-5)/2 = 1 centred I(1,1);
    # touch points D(1,0) on AB, E(0,1) on AC, F = I + (3,4)/5 = (1.6, 1.8) on BC (3x + 4y = 12).
    ents = {"p_A": P(0, 0), "p_B": P(4, 0), "p_C": P(0, 3), "p_I": P(1, 1),
            "p_D": P(1, 0), "p_E": P(0, 1), "p_F": P(1.6, 1.8),
            "s_AB": SEG((0, 0), (4, 0)), "s_BC": SEG((4, 0), (0, 3)), "s_CA": SEG((0, 3), (0, 0)),
            "c_1": {"type": "circle", "g": {"c": [1, 1], "r": 1.0}},
            "g_1": {"type": "polygon", "g": {"vertices": [[0, 0], [4, 0], [0, 3]]}}}
    texts = [{"text": "A", "x": -0.2, "y": -0.2}, {"text": "B", "x": 4.2, "y": -0.2},
             {"text": "C", "x": -0.2, "y": 3.2}]
    return {"y_up": True, "entities": ents, "texts": texts}


def test_345_incircle_tangency_and_incidence(tri345):
    D = math.hypot(4.4, 3.4)
    F = facts(tri345, tol(D))
    for f in [K("tangent_to", "s_AB", "c_1", "p_D"), K("tangent_to", "s_CA", "c_1", "p_E"),
              K("tangent_to", "s_BC", "c_1", "p_F"), K("center_of", "p_I", "c_1"),
              K("perpendicular", "s_AB", "s_CA"), K("between", "p_A", "p_D", "p_B"),
              K("between", "p_A", "p_E", "p_C"), K("between", "p_B", "p_F", "p_C"),
              K("on", "p_F", "c_1"), K("on", "p_F", "s_BC"), K("endpoint_of", "p_A", "s_AB"),
              K("vertex_of", "p_B", "g_1"), K("edge_of", "s_BC", "g_1"), K("contains", "g_1", "c_1"),
              K("orientation", "s_AB", "horizontal"), K("orientation", "s_CA", "vertical"),
              K("orientation", "s_BC", "oblique"), K("polygon", "g_1", 3)]:
        assert f in F, f
    # hand reasoning: BC is not perpendicular to either leg; touch points are tangencies, not crossings
    assert K("perpendicular", "s_AB", "s_BC") not in F
    assert not [f for f in F if f[0] == "intersect_at"]
    assert K("parallel", "s_AB", "s_CA") not in F


def test_345_labels_and_layout(tri345):
    D = math.hypot(4.4, 3.4)
    F = facts(tri345, tol(D))
    assert {K("label", "A", "p_A"), K("label", "B", "p_B"), K("label", "C", "p_C")} <= F
    # A(0,0) left of B(4,0); C(0,3) left of B; A and C share x = 0 (tie: no left_of either way)
    assert K("left_of", "p_A", "p_B") in F and K("left_of", "p_C", "p_B") in F
    assert K("left_of", "p_A", "p_C") not in F and K("left_of", "p_C", "p_A") not in F
    assert K("above", "p_C", "p_A") in F and K("above", "p_C", "p_B") in F
    assert K("outside", "p_A", "c_1") in F                 # |A - I| = sqrt(2) > 1
    assert K("inside", "p_A", "g_1") not in F              # a vertex lies on the boundary


def test_y_down_frame_gives_the_same_facts(tri345):
    D = math.hypot(4.4, 3.4)
    flipped = {"y_up": False, "entities": {}, "texts": [dict(t, y=-t["y"]) for t in tri345["texts"]]}
    for k, e in tri345["entities"].items():
        g = dict(e["g"])
        if "y" in g:
            g["y"] = -g["y"]
        for key in ("p", "q", "c"):
            if key in g:
                g[key] = [g[key][0], -g[key][1]]
        if "vertices" in g:
            g["vertices"] = [[x, -y] for x, y in g["vertices"]]
        flipped["entities"][k] = {"type": e["type"], "g": g}
    assert facts(flipped, tol(D)) == facts(tri345, tol(D))


def test_parallel_transversal_crossings():
    ents = {"s_1": SEG((0, 0), (4, 0)), "s_2": SEG((0, 2), (4, 2)), "s_3": SEG((5, 0), (7, 0)),
            "s_t": SEG((1, -1), (1, 3)), "p_P": P(1, 0), "p_Q": P(1, 2)}
    g = {"y_up": True, "entities": ents, "texts": []}
    F = facts(g, tol(9.0))
    assert K("parallel", "s_1", "s_2") in F and K("parallel", "s_2", "s_3") in F
    assert K("parallel", "s_1", "s_3") not in F           # collinear overlapping lines are not parallel
    assert K("intersect_at", "s_1", "s_t", "p_P") in F and K("intersect_at", "s_2", "s_t", "p_Q") in F
    assert K("perpendicular", "s_1", "s_t") in F and K("perpendicular", "s_2", "s_t") in F


def test_tangent_line_at_known_point_and_secant():
    r3 = math.sqrt(3.0)
    ents = {"c_1": {"type": "circle", "g": {"c": [0, 0], "r": 2.0}}, "p_O": P(0, 0), "p_T": P(2, 0),
            "l_t": {"type": "line", "g": {"p": [2, -3], "q": [2, 3]}},
            "l_s": {"type": "line", "g": {"p": [1, -3], "q": [1, 3]}}, "p_U": P(1, r3), "p_V": P(1, -r3)}
    F = facts({"y_up": True, "entities": ents, "texts": []}, tol(8.0))
    assert K("tangent_to", "l_t", "c_1", "p_T") in F
    assert K("tangent_to", "l_s", "c_1", "p_U") not in F and not [f for f in F if f[0] == "tangent_to"
                                                                   and f[1][0] == "l_s"]
    assert K("intersect_at", "c_1", "l_s", "p_U") in F and K("intersect_at", "c_1", "l_s", "p_V") in F
    assert K("parallel", "l_t", "l_s") in F


def test_circle_circle_tangency_and_arc():
    s2 = math.sqrt(2.0)
    ents = {"c_1": {"type": "circle", "g": {"c": [0, 0], "r": 1.0}},
            "c_2": {"type": "circle", "g": {"c": [3, 0], "r": 2.0}}, "p_K": P(1, 0),
            "a_1": {"type": "arc", "g": {"c": [0, 0], "r": 1.0, "p": [1, 0], "q": [0, 1]}},
            "p_M": P(s2 / 2, s2 / 2), "p_N": P(0, 1)}
    F = facts({"y_up": True, "entities": ents, "texts": []}, tol(7.0))
    assert K("tangent_to", "c_1", "c_2", "p_K") in F      # |c1c2| = 3 = r1 + r2, touch at (1, 0)
    assert K("arc_of", "a_1", "c_1") in F
    assert K("endpoint_of", "p_K", "a_1") in F and K("endpoint_of", "p_N", "a_1") in F
    assert K("on", "p_M", "a_1") in F and K("endpoint_of", "p_M", "a_1") not in F


def test_holds_geo_collinear_subset_and_negative():
    ents = {f"p_{i}": P(i, 2 * i) for i in range(4)}
    g = {"y_up": True, "entities": ents, "texts": []}
    t = tol(8.0)
    assert holds_geo(g, "collinear", [["p_0", "p_1", "p_3"]], t)[0]
    assert holds_geo(g, "collinear", [["p_0", "p_1", "p_2", "p_3"]], t)[0]
    ok, m = holds_geo(g, "between", ["p_0", "p_1", "p_2"], t)
    assert ok and m is not None and m >= 3
    assert not holds_geo(g, "between", ["p_0", "p_3", "p_1"], t)[0]
    with pytest.raises(ValueError):
        holds_geo(g, "equal_length", ["s", "t"], t)          # model-class predicate


def test_near_tangent_secant_is_not_tangent_chord_check():
    # circle r = 2 at the origin; tangency_rel 0.015 accepts |d - r| <= 0.03.
    t = {"eps_on": 0.05, "tangency_rel": 0.015, "parallel_deg": 1.5, "perpendicular_deg": 1.5, "label_bind": 0.5}
    def g(y):
        return {"y_up": True, "texts": [], "entities": {
            "c_1": {"type": "circle", "g": {"c": [0, 0], "r": 2.0}},
            "l_1": {"type": "line", "g": {"p": [-3, y], "q": [3, y]}}}}
    # d = 1.975: |d - r| = 0.025 passes the radius test, but the chord half-length
    # sqrt(4 - 1.975^2) = 0.315 > ε_on: two distinct intersections -> a secant, not a tangent
    assert not [f for f in facts(g(1.975), t) if f[0] == "tangent_to"]
    # d = 1.9999: half chord sqrt(4 - 3.9996) = 0.02 <= ε_on -> tangent
    assert K("tangent_to", "l_1", "c_1", "unknown") in facts(g(1.9999), t)


def test_near_tangent_circles_chord_check():
    t = {"eps_on": 0.05, "tangency_rel": 0.015, "parallel_deg": 1.5, "perpendicular_deg": 1.5, "label_bind": 0.5}
    def g(d):
        return {"y_up": True, "texts": [], "entities": {
            "c_1": {"type": "circle", "g": {"c": [0, 0], "r": 2.0}},
            "c_2": {"type": "circle", "g": {"c": [d, 0], "r": 2.0}}}}
    # d = 3.97: |d - 4| = 0.03 <= 0.015 * 2, but the common chord half-length
    # sqrt(4 - 1.985^2) = 0.244 > ε_on -> overlapping circles, not tangent
    assert not [f for f in facts(g(3.97), t) if f[0] == "tangent_to"]
    assert K("tangent_to", "c_1", "c_2", "unknown") in facts(g(3.9999), t)


def test_bind_labels_one_entry_per_instance_in_order():
    from hmb.synthetic.geo_predicates import bind_labels
    g = {"y_up": True, "entities": {"p_A": P(0, 0), "p_B": P(4, 0), "s_1": SEG((0, 0), (4, 0))},
         "texts": [{"text": "X", "x": 0.2, "y": 0.2}, {"text": "X", "x": 3.8, "y": 0.2},
                   {"text": "far", "x": 2.0, "y": 3.0}, {"text": "m", "x": 2.0, "y": 0.2}]}
    t = {"eps_on": 1e-6, "tangency_rel": 1e-6, "parallel_deg": 1e-4, "perpendicular_deg": 1e-4, "label_bind": 0.4}
    r = bind_labels(g, t)
    assert [x[0] for x in r] == ["X", "X", "far", "m"]
    assert r[0][1] == "p_A" and r[1][1] == "p_B"          # duplicate texts bind separately
    assert r[2] == ("far", None, None)                    # unbound instance kept, not dropped
    assert r[3][1] == "s_1" and r[3][2] == pytest.approx(0.2)   # no point within 0.4: nearest non-point


def test_holds_geo_order_along_subsequence_either_direction():
    ents = {"s_1": SEG((0, 0), (8, 0)), "p_A": P(1, 0), "p_B": P(3, 0), "p_C": P(5, 0), "p_D": P(7, 0)}
    texts = [{"text": t, "x": x, "y": 0.3} for t, x in (("A", 1), ("B", 3), ("C", 5), ("D", 7))]
    g = {"y_up": True, "entities": ents, "texts": texts}
    t = tol(8.5) | {"label_bind": 0.4}
    # the evaluator emits the maximal order A, B, C, D along s_1
    assert K("order_along", "s_1", ["p_A", "p_B", "p_C", "p_D"]) in facts(g, t)
    assert holds_geo(g, "order_along", ["s_1", ["p_A", "p_C", "p_D"]], t)[0]      # 3 of 4, same direction
    assert holds_geo(g, "order_along", ["s_1", ["p_D", "p_B", "p_A"]], t)[0]      # reversed direction
    assert not holds_geo(g, "order_along", ["s_1", ["p_B", "p_A", "p_C"]], t)[0]  # not an order


def test_holds_geo_tangent_unknown_matches_any_contact_point():
    ents = {"c_1": {"type": "circle", "g": {"c": [0, 0], "r": 2.0}}, "p_T": P(2, 0),
            "l_t": {"type": "line", "g": {"p": [2, -3], "q": [2, 3]}}}
    g = {"y_up": True, "entities": ents, "texts": []}
    t = tol(8.0)
    assert holds_geo(g, "tangent_to", ["l_t", "c_1", "p_T"], t)[0]
    assert holds_geo(g, "tangent_to", ["l_t", "c_1", "unknown"], t)[0]
    # with no point entity at the contact (C28: internal geometry) the evaluator emits `unknown`
    del ents["p_T"]
    assert K("tangent_to", "l_t", "c_1", "unknown") in facts(g, t)
