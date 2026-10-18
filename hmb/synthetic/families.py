"""The eight known-geometry generator families (test fixtures, aligned to the §2.6 taxonomy).

Each generator is a pure function of (family, seed): the construction is solved numerically to
coordinates (feet of perpendiculars, tangent points, circle-line intersections, similar-triangle
points, curve roots), then labels are placed deterministically. No human writes a per-item fact;
Γ(s) is computed from the solved coordinates by `evaluator.oracle_facts`.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np

from .build import B, dir_deg, foot, rng_for
from .spec import Spec, unit

FAMILIES = ("triangles_polygons", "circles_tangents_secants", "similarity_congruence",
            "coordinate_geometry", "conic_sections", "function_plots", "simple_solids",
            "angle_trig")
GENERATOR_VERSION = "tcgen-1"


def _convex_polygon(rng, n: int, c, r_lo: float, r_hi: float) -> list[np.ndarray]:
    while True:
        base = np.sort(rng.uniform(0, 2 * math.pi, n))
        gaps = np.diff(np.concatenate([base, [base[0] + 2 * math.pi]]))
        if gaps.min() > 0.6 * (2 * math.pi / n) * 0.6:
            break
    return [np.array(c) + rng.uniform(r_lo, r_hi) * np.array([math.cos(a), math.sin(a)]) for a in base]


def _angle_marker(b: B, eid: str, v: str, p1, p2, r=0.45, n=1, eq=False) -> None:
    V = b.xy(v)
    a0, a1 = dir_deg(np.array(p1) - V), dir_deg(np.array(p2) - V)
    if (a1 - a0) % 360 > 180:
        a0, a1 = a1, a0
    span = (a1 - a0) % 360
    b.add(eid, "angle_marker", {"v": v, "a0": a0, "a1": a0 + span, "r": r, "n": n, "eq": eq})


def _right_marker(b: B, eid: str, v: str, toward1, toward2, s=0.25) -> None:
    V = b.xy(v)
    b.add(eid, "right_angle_marked", {"v": v, "d1": unit(np.array(toward1) - V).tolist(),
                                      "d2": unit(np.array(toward2) - V).tolist(), "s": s})


def _ticks(b: B, eid: str, seg: str, p: str, q: str, n: int) -> None:
    P, Q = b.xy(p), b.xy(q)
    b.add(eid, "length_tick", {"c": ((P + Q) / 2).tolist(), "dir": unit(Q - P).tolist(), "n": n, "seg": seg})


# ------------------------------------------------------------------------ families
def triangles_polygons(seed: int) -> Spec:
    rng = rng_for("triangles_polygons", seed)
    b = B("triangles_polygons", seed)
    n = int(rng.choice([3, 3, 4, 5]))
    names = "ABCDE"[:n]
    V = _convex_polygon(rng, n, (5.0, 5.0), 2.4, 3.6)
    ids = [b.point(names[i], V[i]) for i in range(n)]
    b.add("g_1", "polygon", {"v": ids})
    for i in range(n):
        b.seg(ids[i], ids[(i + 1) % n])
    variant = int(rng.integers(0, 2))
    if n == 3:
        tH = float((V[0] - V[1]) @ (V[2] - V[1]) / ((V[2] - V[1]) @ (V[2] - V[1])))
        if not 0.2 < tH < 0.8:
            variant = 1                   # foot too close to B or C: use the median construction
    if n == 3 and variant == 0:          # altitude from A with right-angle mark
        A, Bp, Cp = V
        H = foot(A, Bp, Cp)
        b.point("H", H)
        b.seg("p_A", "p_H")
        _right_marker(b, "m_H", "p_H", Cp, A)
        b.meta["construction"] = "altitude"
    elif n == 3:                          # median to BC with equal-length ticks on BM, MC
        b.ents = [e for e in b.ents if e.id != "s_BC"]
        M = (V[1] + V[2]) / 2
        b.point("M", M)
        b.seg("p_B", "p_M")
        b.seg("p_M", "p_C")
        b.seg("p_A", "p_M")
        _ticks(b, "t_1", "s_BM", "p_B", "p_M", 1)
        _ticks(b, "t_2", "s_MC", "p_M", "p_C", 1)
        b.meta["construction"] = "median"
    else:                                 # diagonals AC and BD crossing at O
        A, Bp, Cp, D = V[0], V[1], V[2], V[3]
        d1, d2 = Cp - A, D - Bp
        M = np.array([d1, -d2]).T
        t = np.linalg.solve(M, Bp - A)
        O = A + t[0] * d1
        b.point("O", O)
        b.seg("p_A", "p_C")
        b.seg("p_B", "p_D")
        b.meta["construction"] = "diagonals"
    return b.spec()


def circles_tangents_secants(seed: int) -> Spec:
    rng = rng_for("circles_tangents_secants", seed)
    b = B("circles_tangents_secants", seed)
    r = rng.uniform(1.4, 2.0)
    O = np.array([rng.uniform(3.6, 4.4), rng.uniform(4.2, 5.8)])
    ang = rng.uniform(-25, 25)
    dist = rng.uniform(1.9, 2.4) * r
    P = O + dist * np.array([math.cos(math.radians(ang)), math.sin(math.radians(ang))])
    b.point("O", O)
    b.add("c_1", "circle", {"c": "p_O", "r": float(r)})
    b.point("P", P)
    alpha = math.acos(r / dist)
    base = math.atan2(P[1] - O[1], P[0] - O[0])
    T1 = O + r * np.array([math.cos(base + alpha), math.sin(base + alpha)])
    T2 = O + r * np.array([math.cos(base - alpha), math.sin(base - alpha)])
    b.point("T", T1)
    b.seg("p_P", "p_T")
    b.seg("p_O", "p_T")
    _right_marker(b, "m_T", "p_T", O, P)
    if rng.random() < 0.5:
        b.point("S", T2)
        b.seg("p_P", "p_S")
    # secant through P at an offset angle inside the tangent cone
    phi = rng.uniform(0.25, 0.7) * math.asin(r / dist) * (1 if rng.random() < 0.5 else -1)
    u = -np.array([math.cos(base + phi), math.sin(base + phi)])
    w = P - O
    bq, cq = 2 * (u @ w), w @ w - r * r
    disc = bq * bq - 4 * cq
    s1, s2 = (-bq - math.sqrt(disc)) / 2, (-bq + math.sqrt(disc)) / 2
    Aq, Bq = P + s1 * u, P + s2 * u
    b.point("A", Aq)
    b.point("B", Bq)
    b.seg("p_P", "p_B")
    if rng.random() < 0.4:                          # a smaller circle nested inside (containment)
        off = rng.uniform(0.1, 0.3) * r * np.array([math.cos(rng.uniform(0, 6.28)), math.sin(rng.uniform(0, 6.28))])
        b.add("c_2", "circle", {"c": (O + off).tolist(), "r": float(r * rng.uniform(0.3, 0.45))})
    return b.spec()


def similarity_congruence(seed: int) -> Spec:
    rng = rng_for("similarity_congruence", seed)
    b = B("similarity_congruence", seed)
    if rng.random() < 0.6:               # DE parallel to BC in triangle ABC (similar triangles)
        A, Bp, Cp = _convex_polygon(rng, 3, (5.0, 5.0), 2.8, 3.6)
        t = rng.uniform(0.4, 0.65)
        D, E = A + t * (Bp - A), A + t * (Cp - A)
        for nme, xy in zip("ABCDE", (A, Bp, Cp, D, E)):
            b.point(nme, xy)
        b.seg("p_A", "p_D")
        b.seg("p_D", "p_B")
        b.seg("p_A", "p_E")
        b.seg("p_E", "p_C")
        b.seg("p_B", "p_C")
        b.seg("p_D", "p_E")
        _angle_marker(b, "m_1", "p_D", A, E, r=0.35, n=1, eq=True)
        _angle_marker(b, "m_2", "p_B", A, Cp, r=0.35, n=1, eq=True)
        _angle_marker(b, "m_3", "p_E", A, D, r=0.35, n=2, eq=True)
        _angle_marker(b, "m_4", "p_C", A, Bp, r=0.35, n=2, eq=True)
        b.meta["construction"] = "parallel_cut"
    else:                                # two congruent triangles, sides tick-marked
        tri = _convex_polygon(rng, 3, (0.0, 0.0), 1.4, 1.9)
        th = rng.uniform(-0.6, 0.6)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        left = [np.array([2.7, 5.0]) + p for p in tri]
        right = [np.array([7.3, 5.0]) + R @ p for p in tri]
        for nme, xy in zip("ABCDEF", left + right):
            b.point(nme, xy)
        for (p, q), n in ((("p_A", "p_B"), 1), (("p_B", "p_C"), 2), (("p_C", "p_A"), 3),
                          (("p_D", "p_E"), 1), (("p_E", "p_F"), 2), (("p_F", "p_D"), 3)):
            sid = b.seg(p, q)
            _ticks(b, f"t_{sid[2:]}", sid, p, q, n)
        b.meta["construction"] = "congruent_pair"
    return b.spec()


def _axes(b: B, x0, x1, y0, y1, ox, oy) -> None:
    b.add("x_axis", "axis", {"a": [x0, oy], "b": [x1, oy]})
    b.add("y_axis", "axis", {"a": [ox, y0], "b": [ox, y1]})


def coordinate_geometry(seed: int) -> Spec:
    rng = rng_for("coordinate_geometry", seed)
    b = B("coordinate_geometry", seed)
    u = 0.8                                           # 1 unit = 0.8 cm
    ox, oy = 3.2, 3.2
    _axes(b, 0.8, 9.4, 0.8, 9.4, ox, oy)
    b.point("O", (ox, oy))

    def G(x, y):
        return np.array([ox + u * x, oy + u * y])

    while True:
        p = rng.integers(1, 7, 2)
        q = np.array([rng.integers(-2, 7), rng.integers(1, 7)])
        if not np.array_equal(p, q) and p[1] != q[1] and p[0] != q[0]:
            break
    P, Q = G(*p), G(*q)
    b.point("P", P)
    b.point("Q", Q)
    b.add("l_1", "line", {"a": "p_P", "b": "p_Q", "ext": [-0.35, 1.35]})
    b.measure_label(f"({p[0]}, {p[1]})", "coordinate", P + np.array([0.0, -0.42]), "p_P")
    # x-intercept of PQ if it lies inside the drawn axis
    t = -P[1] + oy
    d = Q - P
    if abs(d[1]) > 1e-9:
        s = (oy - P[1]) / d[1]
        R = P + s * d
        if 1.2 < R[0] < 9.0 and -0.35 < s < 1.35 and abs(R[0] - ox) > 0.5:
            b.point("R", R)
    del t
    return b.spec()


def conic_sections(seed: int) -> Spec:
    rng = rng_for("conic_sections", seed)
    b = B("conic_sections", seed)
    ox, oy = 5.0, 5.0
    _axes(b, 0.8, 9.2, 0.8, 9.2, ox, oy)
    b.point("O", (ox, oy))
    kind = int(rng.integers(0, 3))
    if kind == 0:                                     # ellipse with foci
        a, c = rng.uniform(2.6, 3.4), None
        bb = rng.uniform(1.4, 0.8 * a)
        c = math.sqrt(a * a - bb * bb)
        b.add("e_1", "ellipse", {"c": "p_O", "rx": float(a), "ry": float(bb), "rot": 0.0})
        b.point("F", (ox + c, oy))
        b.point("G", (ox - c, oy))
        b.meta["conic"] = "ellipse"
    elif kind == 1:                                   # parabola y = x^2/(4f) with focus and directrix
        f = rng.uniform(0.6, 1.0)
        xs = np.linspace(-3.4, 3.4, 121)
        pts = [[ox + x, oy - 2.6 + x * x / (4 * f)] for x in xs]
        pts = [p for p in pts if p[1] <= 9.0]
        b.add("k_1", "curve", {"pts": pts, "f": "parabola", "params": {"kind": "parabola"}})
        b.point("F", (ox, oy - 2.6 + f))
        b.add("l_d", "line", {"a": [ox - 1.0, oy - 2.6 - f], "b": [ox + 1.0, oy - 2.6 - f], "ext": [-1.8, 2.8]})
        b.meta["conic"] = "parabola"
    else:                                             # hyperbola x^2/a^2 - y^2/b^2 = 1 with asymptotes
        a, bb = rng.uniform(1.0, 1.6), rng.uniform(0.8, 1.6)
        ts = np.linspace(-1.5, 1.5, 81)
        dirs = [[a, bb], [a, -bb]]
        params = {"kind": "hyperbola", "c": [ox, oy], "asymptote_dirs": [unit(np.array(d)).tolist() for d in dirs]}
        for sgn, eid in ((1, "k_1"), (-1, "k_2")):
            pts = [[ox + sgn * a * math.cosh(t), oy + bb * math.sinh(t)] for t in ts]
            pts = [p for p in pts if 0.9 < p[0] < 9.1 and 0.9 < p[1] < 9.1]
            b.add(eid, "curve", {"pts": pts, "f": "hyperbola", "params": dict(params)})
        k = 3.6 / math.hypot(a, bb)
        b.add("l_1", "line", {"a": [ox - a * k, oy - bb * k], "b": [ox + a * k, oy + bb * k], "ext": [0, 1]},
              dashed=True)
        b.add("l_2", "line", {"a": [ox - a * k, oy + bb * k], "b": [ox + a * k, oy - bb * k], "ext": [0, 1]},
              dashed=True)
        b.point("A", (ox + a, oy))
        b.meta["conic"] = "hyperbola"
    return b.spec()


def function_plots(seed: int) -> Spec:
    rng = rng_for("function_plots", seed)
    b = B("function_plots", seed)
    ox, oy, u = 3.0, 4.0, 0.9
    _axes(b, 0.6, 9.4, 0.6, 9.4, ox, oy)
    b.point("O", (ox, oy))
    kind = ["parabola", "periodic", "monotone_other"][int(rng.integers(0, 3))]
    if kind == "parabola":
        r1, r2 = sorted(rng.choice([-2.0, -1.0, 1.5, 2.5, 3.5, 4.5], 2, replace=False))
        k = rng.uniform(0.35, 0.6)
        f = (lambda x: k * (x - r1) * (x - r2))
        roots = [r1, r2]
    elif kind == "periodic":
        A = rng.uniform(1.2, 2.2)
        f = (lambda x: A * math.sin(x))
        roots = [math.pi, 2 * math.pi]
    else:
        c = rng.uniform(0.8, 1.6)
        f = (lambda x: math.exp(0.45 * x) - c)
        roots = [math.log(c) / 0.45]
    x_lo, x_hi = -2.6, 6.8
    special = {r for r in roots if x_lo < r < x_hi} | {0.0}      # exact samples at roots and x = 0
    xs = sorted(set(np.linspace(x_lo, x_hi, 97).tolist()) | special)
    pts = [[ox + u * x, oy + u * f(x)] for x in xs]
    pts = [p for p in pts if 0.7 < p[1] < 9.3]
    b.add("k_1", "curve", {"pts": pts, "f": kind, "params": {"kind": kind}})
    names = iter("PQR")
    for r in roots:
        X = ox + u * r
        if 1.0 < X < 9.0 and abs(X - ox) > 0.5 and pts[0][0] < X < pts[-1][0]:
            b.point(next(names), (X, oy))
    return b.spec()


def simple_solids(seed: int) -> Spec:
    rng = rng_for("simple_solids", seed)
    b = B("simple_solids", seed)
    kind = ["cube", "prism", "pyramid"][int(rng.integers(0, 3))]
    s = rng.uniform(2.8, 3.6)
    depth, ob = rng.uniform(0.4, 0.55), math.radians(rng.uniform(25, 40))
    dx, dy = s * depth * math.cos(ob), s * depth * math.sin(ob)
    x0, y0 = 5.0 - (s + dx) / 2, 5.0 - (s + dy) / 2
    if kind in ("cube", "prism"):
        h = s if kind == "cube" else s * rng.uniform(0.55, 0.8)
        front = [(x0, y0), (x0 + s, y0), (x0 + s, y0 + h), (x0, y0 + h)]
        back = [(x + dx, y + dy) for x, y in front]
        names = "ABCDEFGH"
        ids = [b.point(names[i], p) for i, p in enumerate(front + back)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        hidden = {(4, 5), (7, 4), (0, 4)}          # edges at the hidden back-bottom-left vertex E
        faces = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [1, 2, 6, 5], [0, 3, 7, 4]]
        for i, j in edges:
            b.seg(ids[i], ids[j], dashed=(i, j) in hidden)
        b.add("q_1", "solid", {"k": kind, "vertices": ids, "edges": [list(e) for e in edges], "faces": faces})
        if rng.random() < 0.5:                     # cross-section through three edges at C
            t = rng.uniform(0.35, 0.6)
            C, Bv, D, G = (np.array(front[2]), np.array(front[1]), np.array(front[3]), np.array(back[2]))
            X, Y, Z = C + t * (Bv - C), C + t * (D - C), C + t * (G - C)
            px = [b.point(n_, p) for n_, p in zip("XYZ", (X, Y, Z))]
            b.add("g_x", "polygon", {"v": px})
        b.meta["solid"] = kind
    else:
        base = [(x0, y0), (x0 + s, y0), (x0 + s + dx, y0 + dy), (x0 + dx, y0 + dy)]
        apex = (x0 + (s + dx) / 2, y0 + dy / 2 + s * rng.uniform(1.0, 1.3))
        names = "ABCDS"
        ids = [b.point(names[i], p) for i, p in enumerate(base + [apex])]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 4), (2, 4), (3, 4)]
        hidden = {(2, 3), (3, 0), (3, 4)}
        faces = [[0, 1, 2, 3], [0, 1, 4], [1, 2, 4], [2, 3, 4], [0, 3, 4]]
        for i, j in edges:
            b.seg(ids[i], ids[j], dashed=(i, j) in hidden)
        b.add("q_1", "solid", {"k": "pyramid", "vertices": ids, "edges": [list(e) for e in edges], "faces": faces})
        b.meta["solid"] = "pyramid"
    return b.spec()


def angle_trig(seed: int) -> Spec:
    rng = rng_for("angle_trig", seed)
    b = B("angle_trig", seed)
    if rng.random() < 0.6:                          # unit-circle triangle with angle and right angle
        ox, oy, R = 4.8, 4.8, rng.uniform(2.6, 3.4)
        th = rng.uniform(25, 65)
        _axes(b, 1.0, 9.4, 1.0, 9.4, ox, oy)
        b.point("O", (ox, oy))
        P = np.array([ox + R * math.cos(math.radians(th)), oy + R * math.sin(math.radians(th))])
        Q = np.array([P[0], oy])
        b.point("P", P)
        b.point("Q", Q)
        b.add("a_1", "arc", {"c": "p_O", "r": float(R), "t0": 0.0, "t1": 90.0})
        b.seg("p_O", "p_P")
        b.seg("p_P", "p_Q")
        _right_marker(b, "m_Q", "p_Q", (ox, oy), P)
        _angle_marker(b, "m_O", "p_O", Q, P, r=0.7)
        mid = math.radians(th / 2)
        b.measure_label(f"{int(round(th))}°", "angle", (ox + 1.0 * math.cos(mid), oy + 1.0 * math.sin(mid)), "m_O")
        b.meta["construction"] = "unit_circle"
    else:                                           # two crossing lines, vertical angles marked equal
        O = np.array([5.0, 5.0])
        a1, a2 = rng.uniform(10, 40), rng.uniform(100, 150)
        L = 3.2
        d1 = np.array([math.cos(math.radians(a1)), math.sin(math.radians(a1))])
        d2 = np.array([math.cos(math.radians(a2)), math.sin(math.radians(a2))])
        b.point("O", O)
        for nme, p in zip("ABCD", (O + L * d1, O - L * d1, O + L * d2, O - L * d2)):
            b.point(nme, p)
        b.seg("p_A", "p_B")
        b.seg("p_C", "p_D")
        _angle_marker(b, "m_1", "p_O", O + d1, O + d2, r=0.55, n=1, eq=True)
        _angle_marker(b, "m_2", "p_O", O - d1, O - d2, r=0.55, n=1, eq=True)
        b.meta["construction"] = "vertical_angles"
    return b.spec()


GENERATORS: dict[str, Callable[[int], Spec]] = {
    "triangles_polygons": triangles_polygons,
    "circles_tangents_secants": circles_tangents_secants,
    "similarity_congruence": similarity_congruence,
    "coordinate_geometry": coordinate_geometry,
    "conic_sections": conic_sections,
    "function_plots": function_plots,
    "simple_solids": simple_solids,
    "angle_trig": angle_trig,
}
assert tuple(GENERATORS) == FAMILIES


def generate(family: str, seed: int) -> Spec:
    if family not in GENERATORS:
        raise KeyError(f"unknown family {family!r}")
    s = GENERATORS[family](seed)
    s.meta["generator_version"] = GENERATOR_VERSION
    return s
