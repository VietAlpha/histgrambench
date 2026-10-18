"""C30/C34 polygon rule (CONTRACTS "Polygons"): the shared pure `hmb.cgx.polygons_from_strokes`.
Expected polygon sets are derived by hand from the stroke lists."""
import itertools

import numpy as np

from hmb.cgx import polygons_from_strokes as P

EPS, STRAIGHT = 0.1, 1.5
A, B, C, M = (0.0, 10.0), (-5.0, 0.0), (5.0, 0.0), (0.0, 0.0)


def vsets(polys):
    return sorted(sorted((round(float(x), 3), round(float(y), 3)) for x, y in vs) for vs in polys)


def test_triangle_with_median():
    # BC drawn whole AND as B-M + M-C; median A-M: ABC, ABM, AMC (AM is not a chord of ABC: M is not a corner)
    strokes = [(A, B), (B, C), (C, A), (B, M), (M, C), (A, M)]
    got = vsets(P(strokes, EPS, STRAIGHT, points=[A, B, C, M]))
    assert got == vsets([[A, B, C], [A, B, M], [A, M, C]])


def test_square_with_both_diagonals_documented_case():
    """Documented C34 case: the diagonals are chords of the square, so ONLY the 4 corner-sharing
    triangles are polygons, whatever the drawing style (closed_paths never changes membership)."""
    Q = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
    sides = [(Q[i], Q[(i + 1) % 4]) for i in range(4)]
    diag = [(Q[0], Q[2]), (Q[1], Q[3])]
    tri = [[Q[0], Q[1], Q[2]], [Q[1], Q[2], Q[3]], [Q[2], Q[3], Q[0]], [Q[3], Q[0], Q[1]]]
    assert vsets(P(sides + diag, EPS, STRAIGHT)) == vsets(tri)
    assert vsets(P(sides + diag, EPS, STRAIGHT, points=Q, closed_paths=[Q])) == vsets(tri)
    assert vsets(P(sides, EPS, STRAIGHT)) == vsets([Q])                  # no chords: the square


def test_cube_projection_is_bounded():
    V = {k: (4.0 * k[0] + 1.6 * k[2], 4.0 * k[1] + 1.2 * k[2]) for k in itertools.product((0, 1), repeat=3)}
    E = [(V[a], V[b]) for a, b in itertools.combinations(sorted(V), 2) if sum(i != j for i, j in zip(a, b)) == 1]
    polys = P(E, EPS, STRAIGHT, points=list(V.values()))
    sizes = sorted(len(v) for v in polys)
    assert sizes.count(4) == 6                     # the 6 faces
    assert len(polys) <= 10 and all(s in (4, 6) for s in sizes)   # plus <= 4 induced hexagons (was 15)


def test_membership_independent_of_labelling_and_style():
    """C34: no point-entity corner requirement and no closed-path special case."""
    strokes = [(A, B), (B, C), (C, A)]
    base = vsets(P(strokes, EPS, STRAIGHT))
    assert base == vsets([[A, B, C]])
    assert vsets(P(strokes, EPS, STRAIGHT, points=[])) == base
    assert vsets(P(strokes, EPS, STRAIGHT, points=[A, B, C], closed_paths=[[A, B, C]])) == base


def test_collinear_corners_merge_and_max_corners():
    # a triangle whose side BC is split at M (straight corner) is a triangle, not a quadrilateral
    got = P([(A, B), (B, M), (M, C), (C, A)], EPS, STRAIGHT, points=[A, B, C, M])
    assert vsets(got) == vsets([[A, B, C]])
    ang = [2 * np.pi * k / 9 for k in range(9)]
    nonagon = [(10 * np.cos(t), 10 * np.sin(t)) for t in ang]
    sides = [(nonagon[i], nonagon[(i + 1) % 9]) for i in range(9)]
    assert P(sides, EPS, STRAIGHT) == []                                    # 9 corners > 8
    assert P(sides, EPS, STRAIGHT, points=nonagon, closed_paths=[nonagon]) == []   # in every style


def test_rejects_reversal_and_self_intersection_and_is_deterministic():
    bow = [(0.0, 0.0), (4.0, 4.0), (4.0, 0.0), (0.0, 4.0)]
    strokes = [(bow[i], bow[(i + 1) % 4]) for i in range(4)]
    assert P(strokes, EPS, STRAIGHT, points=bow) == []                      # a bow-tie self-intersects
    s = [(A, B), (B, C), (C, A), (B, M), (M, C), (A, M)]
    r1 = P(s, EPS, STRAIGHT, points=[A, B, C, M])
    r2 = P(list(reversed(s)), EPS, STRAIGHT, points=[M, C, B, A])
    assert vsets(r1) == vsets(r2) and [np.round(v, 6).tolist() for v in r1] == [np.round(v, 6).tolist() for v in r2]
