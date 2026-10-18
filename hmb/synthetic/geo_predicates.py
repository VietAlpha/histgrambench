"""Shared geometric predicate evaluator (ONE pure function for Γ and the CGX extractor).

Contract: docs/CONTRACTS.md "Geometric predicate semantics". Owner: Opus 5; the format was agreed
with Opus 3 (CGX) on 2026-09-22. The fixture oracle evaluator calls it with *exact* tolerances;
the CGX extractor calls it with the §10.2 tolerances. Callers canonicalize to the MGS with
`hmb.cards.closure`.

Input ``geom``::

    {"y_up": bool,                       # True: TikZ/oracle frame; False: PDF page frame (y down)
     "entities": {eid: {"type": T, "label": str|None, "g": {...}}},   # "label" is informational
     "texts": [{"text": str, "x": float, "y": float}]}               # label(t,e) comes only from these

Coordinates are floats in any single unit. Per-type ``g``:
    point                          {"x","y"}
    segment / length_tick / arrow  {"p":[x,y],"q":[x,y]}          (arrow: p tail, q head)
    ray                            {"p": origin, "q": a point it passes through}
    line                           {"p","q"}                       (any two distinct points)
    axis                           {"p","q","o"?}                  (o computed when absent)
    circle                         {"c":[x,y],"r"}
    arc                            {"c","r","p","q"}               CCW from p to q in the y-up frame
    ellipse                        {"c","rx","ry","rot_deg"}
    polygon                        {"vertices":[[x,y],...]}         boundary order
    angle_marker                   {"vertex":[x,y],"r","p","q"}
    right_angle_marked             {"vertex":[x,y],"size"}
    curve / curve_generic          {"points":[[x,y],...]}           open polyline; identity opaque
    region                         {"points":[[x,y],...]}           closed boundary; identity opaque
    solid                          {"points"?, "vertices"?, "edges"?:[[i,j]], "faces"?:[[i,j,k..]]}

``tol = {"eps_on": abs length, "tangency_rel", "parallel_deg", "perpendicular_deg", "label_bind": abs length}``

Output of `evaluate_geo`: ``list[(pred, args, margin)]`` with canonical args (`ontology.canonical_args`)
of every positively asserted **geo** fact (decidability geo, §7.1), before closure/MGS.
Margins in ε units: tolerance facts ``tol / max(dev, 1e-9·tol)`` capped at 1000; separation facts
``separation / eps_on``; existence facts ``None``.

Semantics (form choices, CONTRACTS.md):
  * intersect_at(x,y,p): p within eps_on of both, in the relative interior of both (not within
    eps_on of an end or polygon corner) and the local directions differ by more than parallel_deg
    (a transversal crossing; tangency and overlap are excluded).
  * tangent_to(x,y,p): x linear, y curved (circle/arc/ellipse/curve); curved-curved by id order;
    p = the point within eps_on of the touch point, else "unknown".
  * parallel/perpendicular over linear entities {segment, line, ray, axis}; collinear pairs are
    not parallel. collinear: maximal sets >= 3 points; concyclic: maximal sets >= 4 points.
  * between(p,q,r): q within eps_on of segment pr and farther than eps_on from both ends.
  * contains(x,y): closed-in-closed over {circle, ellipse, polygon, region}.
  * label binding (§10.3): nearest point within label_bind, else nearest non-point entity within it.
  * layout predicates only between labelled points (points some text binds to), ε_on tie margin;
    order_along needs >= 3 labelled points on a non-closed 1-D entity, seq[0] < seq[-1];
    orientation on segment/line/ray: horizontal/vertical within parallel_deg, else oblique.
"""
from __future__ import annotations

import itertools
import math
from typing import Any, Iterable

import numpy as np

from hmb.ontology import ONE_D_TYPES, PREDICATES, canonical_args

from .spec import Geo, ellipse_pts, line_dist, polyline_dist, seg_dist, signed_line, unit

GEO_FORMAT_VERSION = "geo-eval-1"
MARGIN_CAP = 1000.0

LINEAR = ("segment", "line", "ray", "axis")
CLOSED = ("circle", "ellipse", "polygon", "region")
OPEN_1D = ("segment", "line", "ray", "axis", "arc", "curve")
ORIENTED = ("segment", "line", "ray")
GEO_EXISTENCE = ("point", "segment", "line", "ray", "circle", "arc", "ellipse", "polygon", "axis",
                 "arrow", "angle_marker", "right_angle_marked", "length_tick")
GEO_PREDICATES = frozenset(p for p, d in PREDICATES.items() if d.decidability.value == "geo")


def _tol_margin(tol: float, dev: float) -> float:
    return float(min(MARGIN_CAP, tol / max(dev, 1e-9 * tol)))


def _ang(a: Iterable[float]) -> float:
    return math.degrees(math.atan2(a[1], a[0]))


# ------------------------------------------------------------------------ format conversion
def to_internal(geom: dict) -> tuple[list[dict], list[dict]]:
    """Shared format -> internal resolved entity dicts (y up) used by `spec.Geo`."""
    sy = 1.0 if geom.get("y_up", True) else -1.0

    def P(v):
        return [float(v[0]), sy * float(v[1])]

    ents: list[dict] = []
    for eid, e in geom["entities"].items():
        t, g = e["type"], e.get("g", {})
        ig: dict[str, Any]
        if t == "point":
            ig = {"x": float(g["x"]), "y": sy * float(g["y"])}
            if "marker" in g:
                ig["marker"] = bool(g["marker"])
        elif t in ("segment", "arrow", "axis", "ray", "line"):
            ig = {"a": P(g["p"]), "b": P(g["q"])}
            if t == "line":
                ig["ext"] = [0.0, 1.0]
            if t == "ray":
                ig["t1"] = 1.0
            if t == "axis" and g.get("o"):
                ig["o"] = g["o"]
        elif t == "length_tick":
            p, q = np.array(P(g["p"])), np.array(P(g["q"]))
            d = q - p
            ig = {"c": ((p + q) / 2).tolist(), "dir": [d[1], -d[0]], "n": int(g.get("n", 1)),
                  "h": float(np.hypot(*d)) / 2}
        elif t == "circle":
            ig = {"c": P(g["c"]), "r": float(g["r"])}
        elif t == "arc":
            c = np.array(P(g["c"]))
            t0 = _ang(np.array(P(g["p"])) - c)
            t1 = _ang(np.array(P(g["q"])) - c)
            span = (t1 - t0) % 360.0 or 360.0
            ig = {"c": c.tolist(), "r": float(g["r"]), "t0": t0, "t1": t0 + span}
        elif t == "ellipse":
            ig = {"c": P(g["c"]), "rx": float(g["rx"]), "ry": float(g["ry"]),
                  "rot": sy * float(g.get("rot_deg", 0.0))}
        elif t == "polygon":
            ig = {"v": [P(v) for v in g["vertices"]]}
        elif t == "angle_marker":
            v = np.array(P(g["vertex"]))
            a0 = _ang(np.array(P(g["p"])) - v)
            a1 = _ang(np.array(P(g["q"])) - v)
            span = (a1 - a0) % 360.0 or 360.0
            ig = {"v": v.tolist(), "r": float(g["r"]), "a0": a0, "a1": a0 + span, "n": int(g.get("n", 1))}
        elif t == "right_angle_marked":
            ig = {"v": P(g["vertex"]), "d1": [1.0, 0.0], "d2": [0.0, sy], "s": float(g["size"])}
        elif t in ("curve", "curve_generic"):
            t = "curve"
            ig = {"pts": [P(v) for v in g["points"]]}
        elif t == "region":
            ig = {"pts": [P(v) for v in g["points"]]}
        elif t == "solid":
            ig = {"vertices": [P(v) for v in g.get("vertices", [])], "edges": g.get("edges", []),
                  "faces": g.get("faces", []), "pts": [P(v) for v in g.get("points", [[0, 0]])]}
        else:
            raise ValueError(f"unknown entity type {t!r} ({eid})")
        ents.append({"entity_id": eid, "type": t, "geom": ig, "style": {}})
    texts = [{"text": x["text"], "x": float(x["x"]), "y": sy * float(x["y"])}
             for x in geom.get("texts", [])]
    return ents, texts


def from_internal(ents: list[dict], texts: list[dict]) -> dict:
    """Internal resolved entity dicts (y up) -> shared format (used for Γ)."""
    out: dict[str, dict] = {}
    for e in ents:
        t, g = e["type"], e["geom"]
        if t == "point":
            sg = {"x": g["x"], "y": g["y"]}
            if "marker" in g:
                sg["marker"] = bool(g["marker"])
        elif t in ("segment", "arrow", "axis", "ray", "line"):
            sg = {"p": list(g["a"]), "q": list(g["b"])}
        elif t == "length_tick":
            c, d = np.array(g["c"]), unit(np.array(g["dir"], float))
            nrm = np.array([-d[1], d[0]]) * g.get("h", 0.12)
            sg = {"p": (c - nrm).tolist(), "q": (c + nrm).tolist()}
            if int(g.get("n", 1)) > 1:
                sg["n"] = int(g["n"])
        elif t == "circle":
            sg = {"c": list(g["c"]), "r": g["r"]}
        elif t == "arc":
            c = np.array(g["c"])
            ends = [c + g["r"] * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
                    for a in (g["t0"], g["t1"])]
            sg = {"c": list(g["c"]), "r": g["r"], "p": ends[0].tolist(), "q": ends[1].tolist()}
        elif t == "ellipse":
            sg = {"c": list(g["c"]), "rx": g["rx"], "ry": g["ry"], "rot_deg": g["rot"]}
        elif t == "polygon":
            sg = {"vertices": [list(v) for v in g["v"]]}
        elif t == "angle_marker":
            v = np.array(g["v"])
            sg = {"vertex": v.tolist(), "r": g["r"],
                  "p": (v + g["r"] * np.array([math.cos(math.radians(g["a0"])),
                                               math.sin(math.radians(g["a0"]))])).tolist(),
                  "q": (v + g["r"] * np.array([math.cos(math.radians(g["a1"])),
                                               math.sin(math.radians(g["a1"]))])).tolist()}
            if int(g.get("n", 1)) > 1:
                sg["n"] = int(g["n"])
        elif t == "right_angle_marked":
            sg = {"vertex": list(g["v"]), "size": g["s"]}
        elif t == "curve":
            sg = {"points": [list(p) for p in g["pts"]]}
        elif t == "region":
            sg = {"points": [list(p) for p in g["pts"]]}
        elif t == "solid":
            sg = {"vertices": [list(v) for v in g["vertices"]], "edges": g["edges"],
                  "faces": g.get("faces", [])}
        else:
            raise ValueError(t)
        out[e["entity_id"]] = {"type": t, "label": None, "g": sg}
    return {"y_up": True, "entities": out,
            "texts": [{"text": x["text"], "x": x["x"], "y": x["y"]} for x in texts]}


# ------------------------------------------------------------------------ helpers
class _Ctx:
    def __init__(self, geom: dict, tol: dict):
        ents, texts = to_internal(geom)
        self.G = Geo(ents, texts)
        self.tol = tol
        self.eps = float(tol["eps_on"])
        self.types = {e["entity_id"]: e["type"] for e in ents}
        self.points = sorted(i for i, t in self.types.items() if t == "point")

    def of(self, *types: str) -> list[str]:
        return sorted(i for i, t in self.types.items() if t in types)

    def local_dir(self, eid: str, p: np.ndarray) -> np.ndarray:
        """Unit tangent direction of entity eid near point p."""
        t = self.types[eid]
        G = self.G
        if t in ("segment", "line", "ray", "axis", "arrow"):
            return G.direction(eid)
        if t in ("circle", "arc"):
            r = p - G.center(eid)
            return unit(np.array([-r[1], r[0]]))
        best, bd = None, math.inf
        for pl in G.polylines(eid):
            for a, b in zip(pl[:-1], pl[1:]):
                d, _ = seg_dist(p, a, b)
                if d < bd and np.any(b != a):
                    bd, best = d, unit(b - a)
        return best if best is not None else np.array([1.0, 0.0])

    def interior(self, eid: str, p: np.ndarray) -> bool:
        return all(float(np.hypot(*(p - q))) > self.eps for q in self.G.endpoints(eid))

    def line_of(self, eid: str) -> tuple[np.ndarray, np.ndarray]:
        a, b = self.G.ab(eid)
        return a, unit(b - a)


def _angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Acute angle between two undirected directions, degrees in [0, 90]."""
    c = abs(float(np.clip(u @ v, -1.0, 1.0)))
    return math.degrees(math.acos(min(1.0, c)))


def _bind_one(C: _Ctx, x: dict) -> tuple[str, float] | None:
    """§10.3 rule for one text, with the C28 binding order (lead decision, cgx-1.3.0): the nearest
    DRAWN point (g["marker"] true) within label_bind, else the nearest other point entity, else the
    nearest non-point entity. A missing "marker" key means False (B3 red-team round 2)."""
    lb = float(C.tol["label_bind"])
    p = np.array([x["x"], x["y"]])

    def nearest(ids):
        best = None
        for eid in ids:
            d = C.G.dist(eid, p)
            if d <= lb and (best is None or d < best[1]):
                best = (eid, d)
        return best
    drawn = [q for q in C.points if C.G.E[q]["geom"].get("marker", False)]
    other = [q for q in C.points if not C.G.E[q]["geom"].get("marker", False)]
    return (nearest(drawn) or nearest(other)
            or nearest([e for e, t in sorted(C.types.items()) if t != "point"]))


def bind_texts(C: _Ctx) -> list[tuple[str, str, float]]:
    """§10.3 label binding: (text, entity, distance) for the texts that bind."""
    out = []
    for x in C.G.texts:
        b = _bind_one(C, x)
        if b is not None:
            out.append((x["text"], b[0], b[1]))
    return out


def bind_labels(geom: dict, tol: dict) -> list[tuple[str, str | None, float | None]]:
    """Public per-instance label binding: one entry per input text, in input order, with
    (text, entity_id | None, distance | None); duplicate texts bind independently."""
    C = _Ctx(geom, tol)
    out: list[tuple[str, str | None, float | None]] = []
    for x in C.G.texts:
        b = _bind_one(C, x)
        out.append((x["text"], b[0], b[1]) if b is not None else (x["text"], None, None))
    return out


# ------------------------------------------------------------------------ C28 stroke classification
AXIS_NAMES = frozenset({"x", "y", "z", "X", "Y", "Z", "t", "r", "θ", "φ"})   # "O" excluded


def classify_straight(p, q, point_xy, eps: float, *, tipped: bool = False, D: float | None = None,
                      parallel_deg: float | None = None, axis_label: bool = False,
                      axis_min_frac: float | None = None, th=None) -> tuple[str, list[float], list[float]]:
    """C28 (docs/CONTRACTS.md): type of a drawn straight stroke p -> q from its incident point entities.

    Untipped: point entities at both ends -> segment; at exactly one end -> ray (a = that end, the
    origin; b = the far drawn end); at neither end with >= 1 incident point entity -> line (a, b = the
    drawn ends); zero incident point entities -> segment. An untipped stroke carrying an axis-name
    label (`axis_label`) is an axis.
    Tipped (arrow tip): axis iff horizontal/vertical within parallel_deg AND (length >=
    axis_min_length_frac·D, registered C28 form constant, or an axis-name label binds to it);
    otherwise arrow (a = tail, b = head). Pass `axis_min_frac` (e.g. tol["axis_min_length_frac"]) or `th`.
    Returns (type, a, b)."""
    P, Q = np.asarray(p, float), np.asarray(q, float)
    pts = [np.asarray(x, float) for x in point_xy]
    if tipped:
        if axis_min_frac is None and th is not None:
            axis_min_frac = float(th.value("axis_min_length_frac"))
        if axis_min_frac is None:
            raise ValueError("classify_straight: a tipped stroke needs axis_min_frac or th "
                             "(registered axis_min_length_frac)")
        u = unit(Q - P)
        ang = abs(math.degrees(math.atan2(u[1], u[0]))) % 180.0
        hv = min(ang, 180.0 - ang, abs(90.0 - ang)) <= (parallel_deg or 0.0)
        long_ = D is not None and float(np.hypot(*(Q - P))) >= axis_min_frac * D
        return ("axis" if hv and (long_ or axis_label) else "arrow"), P.tolist(), Q.tolist()
    if axis_label:
        return "axis", P.tolist(), Q.tolist()
    at_p = any(float(np.hypot(*(P - x))) <= eps for x in pts)
    at_q = any(float(np.hypot(*(Q - x))) <= eps for x in pts)
    incident = [x for x in pts if seg_dist(x, P, Q)[0] <= eps]
    if at_p and at_q:
        return "segment", P.tolist(), Q.tolist()
    if at_p or at_q:
        o, far = (P, Q) if at_p else (Q, P)
        return "ray", o.tolist(), far.tolist()
    if not incident:
        return "segment", P.tolist(), Q.tolist()
    return "line", P.tolist(), Q.tolist()


# ------------------------------------------------------------------------ the evaluator
def evaluate_geo(geom: dict, tol: dict, predicates: Iterable[str] | None = None
                 ) -> list[tuple[str, tuple, float | None]]:
    """Every positively asserted geo fact of `geom` under `tol` (see module docstring)."""
    want = set(predicates) if predicates is not None else set(GEO_PREDICATES)
    C = _Ctx(geom, tol)
    G, eps = C.G, C.eps
    out: dict[tuple, float | None] = {}

    def emit(pred: str, args: list, margin: float | None) -> None:
        if pred not in want:
            return
        key = (pred, canonical_args(pred, args))
        prev = out.get(key, -1.0)
        if key not in out or (margin is not None and (prev is None or margin > prev)):
            out[key] = margin

    # ---- existence
    for eid, t in sorted(C.types.items()):
        if t not in GEO_EXISTENCE:
            continue
        if t == "polygon":
            n = len(G.E[eid]["geom"]["v"])
            if n >= 3:
                emit("polygon", [eid, n], None)
        elif t == "axis":
            emit("axis", [eid, G.E[eid]["geom"].get("o") or _orient(C, eid)[0]], None)
        else:
            emit(t, [eid], None)

    pts = {p: G.point(p) for p in C.points}

    # ---- incidence: on, endpoint_of, vertex_of, center_of
    one_d = [e for e in sorted(C.types) if C.types[e] in ONE_D_TYPES]
    for pid, p in pts.items():
        for x in one_d:
            d = G.dist(x, p)
            if d <= eps:
                emit("on", [pid, x], _tol_margin(eps, d))
        for x in C.of("segment", "ray", "arc"):
            ends = G.endpoints(x)
            for q in ends:
                d = float(np.hypot(*(p - q)))
                if d <= eps:
                    emit("endpoint_of", [pid, x], _tol_margin(eps, d))
        for x in C.of("polygon"):
            for q in G.endpoints(x):
                d = float(np.hypot(*(p - q)))
                if d <= eps:
                    emit("vertex_of", [pid, x], _tol_margin(eps, d))
        for x in C.of("angle_marker"):
            d = float(np.hypot(*(p - G.pt(G.E[x]["geom"]["v"]))))
            if d <= eps:
                emit("vertex_of", [pid, x], _tol_margin(eps, d))
        for x in C.of("solid"):
            for q in G.E[x]["geom"].get("vertices", []):
                d = float(np.hypot(*(p - np.array(q))))
                if d <= eps:
                    emit("vertex_of", [pid, x], _tol_margin(eps, d))
        for x in C.of("circle", "arc", "ellipse"):
            d = float(np.hypot(*(p - G.center(x))))
            if d <= eps:
                emit("center_of", [pid, x], _tol_margin(eps, d))

    # ---- edge_of
    for s in C.of("segment"):
        a, b = G.ab(s)
        for x in C.of("polygon", "solid"):
            if C.types[x] == "polygon":
                v = [np.array(q) for q in G.E[x]["geom"]["v"]]
                pairs = [(v[i], v[(i + 1) % len(v)]) for i in range(len(v))]
            else:
                vs = [np.array(q) for q in G.E[x]["geom"].get("vertices", [])]
                pairs = [(vs[i], vs[j]) for i, j in G.E[x]["geom"].get("edges", [])]
            for u, w in pairs:
                dev = min(max(np.hypot(*(a - u)), np.hypot(*(b - w))),
                          max(np.hypot(*(a - w)), np.hypot(*(b - u))))
                if dev <= eps:
                    emit("edge_of", [s, x], _tol_margin(eps, float(dev)))

    # ---- arc_of
    for a_ in C.of("arc"):
        for c in C.of("circle"):
            dev = max(float(np.hypot(*(G.center(a_) - G.center(c)))),
                      abs(G.E[a_]["geom"]["r"] - G.E[c]["geom"]["r"]))
            if dev <= eps:
                emit("arc_of", [a_, c], _tol_margin(eps, dev))

    # ---- tangency
    tangent_pairs: set[tuple[str, str]] = set()
    for x in C.of(*LINEAR):
        for y in C.of("circle", "arc", "ellipse", "curve"):
            r = _tangent_linear(C, x, y)
            if r is not None:
                touch, margin = r
                tangent_pairs.add((x, y))
                emit("tangent_to", [x, y, _point_at(C, touch)], margin)
    curved = C.of("circle", "arc")
    for x, y in itertools.combinations(curved, 2):
        r = _tangent_circles(C, x, y)
        if r is not None:
            touch, margin = r
            tangent_pairs.add((x, y))
            emit("tangent_to", [x, y, _point_at(C, touch)], margin)

    # ---- intersect_at (transversal crossings at point entities)
    for x, y in itertools.combinations(one_d, 2):
        for pid, p in pts.items():
            dx, dy = G.dist(x, p), G.dist(y, p)
            if dx > eps or dy > eps:
                continue
            if not (C.interior(x, p) and C.interior(y, p)):
                continue
            ang = _angle_between(C.local_dir(x, p), C.local_dir(y, p))
            if ang <= float(tol["parallel_deg"]):
                continue
            sep = min([float(np.hypot(*(p - q))) for q in G.endpoints(x) + G.endpoints(y)]
                      or [MARGIN_CAP * eps])
            margin = min(_tol_margin(eps, dx), _tol_margin(eps, dy), sep / eps if eps > 0 else MARGIN_CAP)
            emit("intersect_at", [x, y, pid], min(margin, MARGIN_CAP))

    # ---- parallel / perpendicular
    for x, y in itertools.combinations(C.of(*LINEAR), 2):
        (ax, ux), (ay, uy) = C.line_of(x), C.line_of(y)
        ang = _angle_between(ux, uy)
        if ang <= float(tol["parallel_deg"]):
            gap = line_dist(ay, ax, ux)
            if gap > eps:
                emit("parallel", [x, y], _tol_margin(float(tol["parallel_deg"]), ang))
        dev = abs(90.0 - ang)
        if dev <= float(tol["perpendicular_deg"]):
            emit("perpendicular", [x, y], _tol_margin(float(tol["perpendicular_deg"]), dev))

    # ---- collinear / concyclic / between
    for S, dev in _maximal_collinear(pts, eps):
        emit("collinear", [list(S)], _tol_margin(eps, dev))
    for S, dev in _maximal_concyclic(pts, eps):
        emit("concyclic", [list(S)], _tol_margin(eps, dev))
    if "between" in want:
        for p_, q_, r_ in itertools.permutations(sorted(pts), 3):
            if p_ > r_:
                continue
            P, Q, R = pts[p_], pts[q_], pts[r_]
            d, t = seg_dist(Q, P, R)
            if d > eps:
                continue
            sep = min(float(np.hypot(*(Q - P))), float(np.hypot(*(Q - R))))
            if sep > eps and 0.0 < t < 1.0:
                emit("between", [p_, q_, r_], min(_tol_margin(eps, d), sep / eps))

    # ---- contains (closed-in-closed)
    for x, y in itertools.permutations(C.of(*CLOSED), 2):
        samples = np.vstack(G.polylines(y))
        s = np.array([G.signed_inside(x, q) for q in samples[:: max(1, len(samples) // 180)]])
        if s.min() >= -eps and s.max() > eps:
            emit("contains", [x, y], _tol_margin(eps, max(0.0, -float(s.min()))))

    # ---- labels
    bound = bind_texts(C)
    labelled: set[str] = set()
    lb = float(tol["label_bind"])
    for text, eid, d in bound:
        if text.strip():
            emit("label", [text, eid], (lb - d) / eps if eps > 0 else MARGIN_CAP)
        if C.types[eid] == "point":
            labelled.add(eid)

    # ---- layout over labelled points
    L = sorted(labelled)
    for a_, b_ in itertools.permutations(L, 2):
        A, B = pts[a_], pts[b_]
        if B[0] - A[0] > eps:
            emit("left_of", [a_, b_], min(MARGIN_CAP, (B[0] - A[0]) / eps))
        if A[1] - B[1] > eps:
            emit("above", [a_, b_], min(MARGIN_CAP, (A[1] - B[1]) / eps))
    for p_ in L:
        for x in C.of(*CLOSED):
            s = G.signed_inside(x, pts[p_])
            if s > eps:
                emit("inside", [p_, x], min(MARGIN_CAP, s / eps))
            elif s < -eps:
                emit("outside", [p_, x], min(MARGIN_CAP, -s / eps))
    for l_ in C.of(*LINEAR):
        a, u = C.line_of(l_)
        for p_, q_ in itertools.combinations(L, 2):
            sp, sq = signed_line(pts[p_], a, u), signed_line(pts[q_], a, u)
            if abs(sp) <= eps or abs(sq) <= eps:
                continue
            m = min(MARGIN_CAP, min(abs(sp), abs(sq)) / eps)
            emit("same_side" if sp * sq > 0 else "opposite_sides", [p_, q_, l_], m)
    for x in C.of(*OPEN_1D):
        on_x = [p_ for p_ in L if G.dist(x, pts[p_]) <= eps]
        if len(on_x) < 3:
            continue
        par = sorted((_param(C, x, pts[p_]), p_) for p_ in on_x)
        gaps = [b[0] - a[0] for a, b in zip(par, par[1:])]
        seq = [p_ for _, p_ in par]
        if seq[0] > seq[-1]:
            seq.reverse()
        sep = min(float(np.hypot(*(pts[s1] - pts[s2]))) for s1, s2 in zip(seq, seq[1:]))
        if min(gaps) > 0 and sep > eps:
            emit("order_along", [x, seq], min(MARGIN_CAP, sep / eps))
    for x in C.of(*ORIENTED):
        o, m = _orient(C, x)
        emit("orientation", [x, o], m)

    return [(k[0], k[1], m) for k, m in sorted(out.items(), key=lambda kv: (kv[0][0], repr(kv[0][1])))]


def _is_subsequence(sub, seq) -> bool:
    it = iter(seq)
    return all(x in it for x in sub)


def holds_geo(geom: dict, pred: str, args: list, tol: dict) -> tuple[bool, float | None]:
    """Single-fact query against the evaluator's output. A fact holds when it is emitted, or when:
      * collinear / concyclic: its set is a subset of an emitted maximal set;
      * order_along(x, S): S is a subsequence of the emitted maximal order along x, in either
        direction (the evaluator canonicalizes the direction so that seq[0] < seq[-1]);
      * tangent_to(x, y, "unknown"): some tangent_to(x, y, p) is emitted for the same x, y."""
    if pred not in GEO_PREDICATES:
        raise ValueError(f"{pred} is not a geo predicate")
    key = canonical_args(pred, args)
    for p, a, m in evaluate_geo(geom, tol, predicates={pred}):
        if a == key:
            return True, m
        if pred in ("collinear", "concyclic") and set(key[0]) <= set(a[0]):
            return True, m
        if pred == "order_along" and a[0] == key[0] and (
                _is_subsequence(key[1], a[1]) or _is_subsequence(key[1], a[1][::-1])):
            return True, m
        if pred == "tangent_to" and key[2] == "unknown" and a[0] == key[0] and a[1] == key[1]:
            return True, m
    return False, None


# ------------------------------------------------------------------------ internals
def _orient(C: _Ctx, x: str) -> tuple[str, float]:
    u = C.G.direction(x)
    a = abs(math.degrees(math.atan2(u[1], u[0]))) % 180.0
    tolp = float(C.tol["parallel_deg"])
    dh = min(a, 180.0 - a)
    dv = abs(90.0 - a)
    if dh <= tolp:
        return "horizontal", _tol_margin(tolp, dh)
    if dv <= tolp:
        return "vertical", _tol_margin(tolp, dv)
    return "oblique", min(MARGIN_CAP, (min(dh, dv) - tolp) / tolp if tolp > 0 else MARGIN_CAP)


def _point_at(C: _Ctx, q: np.ndarray) -> str:
    best, bd = "unknown", math.inf
    for pid in C.points:
        d = float(np.hypot(*(C.G.point(pid) - q)))
        if d <= C.eps and d < bd:
            best, bd = pid, d
    return best


def _on_extent(C: _Ctx, x: str, q: np.ndarray) -> bool:
    return C.G.dist(x, q) <= C.eps


def _half_chord(r: float, d: float) -> float:
    """Half-length of the chord a line at distance d from the centre cuts from a circle of radius r
    (0 when the line misses or touches)."""
    return math.sqrt(max(0.0, r * r - d * d))


def _tangent_linear(C: _Ctx, x: str, y: str) -> tuple[np.ndarray, float] | None:
    G, eps = C.G, C.eps
    a, u = C.line_of(x)
    ty = C.types[y]
    rel = float(C.tol["tangency_rel"])
    if ty in ("circle", "arc"):
        c, r = G.center(y), G.E[y]["geom"]["r"]
        d = line_dist(c, a, u)
        dev = abs(d - r)
        if dev > rel * r:
            return None
        if _half_chord(r, d) > eps:          # §10.2: exactly one intersection within ε_on
            return None
        foot = a + ((c - a) @ u) * u
        touch = c + r * unit(foot - c) if np.hypot(*(foot - c)) > 0 else foot
        if not _on_extent(C, x, touch) or G.dist(y, touch) > max(eps, rel * r):
            return None
        return touch, _tol_margin(rel * r, dev)
    if ty == "ellipse":
        g = G.E[y]["geom"]
        c = G.pt(g["c"])
        rot = math.radians(g["rot"])
        R = np.array([[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]])
        S = np.diag([1 / g["rx"], 1 / g["ry"]])
        a2, b2 = S @ R @ (a - c), S @ R @ (a + u - c)
        u2 = unit(b2 - a2)
        d = line_dist(np.zeros(2), a2, u2)
        dev = abs(d - 1.0)
        if dev > rel:
            return None
        foot2 = a2 + ((-a2) @ u2) * u2
        inv = np.linalg.inv(S @ R)
        h2 = _half_chord(1.0, d)
        if h2 > 0 and float(np.hypot(*(inv @ (2 * h2 * u2)))) / 2 > eps:
            return None                      # the secant chord is longer than ε_on: not tangent
        touch2 = unit(foot2) if np.hypot(*foot2) > 0 else foot2
        touch = inv @ touch2 + c
        if not _on_extent(C, x, touch):
            return None
        return touch, _tol_margin(rel, dev)
    # generic curve: nearest approach within eps and no sign change in the local neighbourhood
    pl = G.polylines(y)[0]
    s = np.array([signed_line(q, a, u) for q in pl])
    i = int(np.argmin(np.abs(s)))
    if abs(s[i]) > eps:
        return None
    lo, hi = max(0, i - 5), min(len(s), i + 6)
    loc = s[lo:hi]
    if (loc.min() < -eps) and (loc.max() > eps):
        return None
    if i in (0, len(s) - 1):
        return None
    touch = pl[i]
    if not _on_extent(C, x, touch):
        return None
    return touch, _tol_margin(eps, abs(float(s[i])))


def _tangent_circles(C: _Ctx, x: str, y: str) -> tuple[np.ndarray, float] | None:
    G = C.G
    c1, r1 = G.center(x), G.E[x]["geom"]["r"]
    c2, r2 = G.center(y), G.E[y]["geom"]["r"]
    d = float(np.hypot(*(c2 - c1)))
    if d == 0:
        return None
    rel = float(C.tol["tangency_rel"]) * min(r1, r2)
    ext, inn = abs(d - (r1 + r2)), abs(d - abs(r1 - r2))
    dev = min(ext, inn)
    if dev > rel:
        return None
    # §10.2 "exactly one intersection within ε_on": the common chord's half-length
    a_ = (d * d + r1 * r1 - r2 * r2) / (2 * d)
    if _half_chord(r1, a_) > C.eps:
        return None
    u = (c2 - c1) / d
    touch = c1 + r1 * u if ext <= inn else (c1 + r1 * u if r1 >= r2 else c1 - r1 * u)
    if G.dist(x, touch) > C.eps or G.dist(y, touch) > C.eps:
        return None
    return touch, _tol_margin(rel, dev)


def _maximal_collinear(pts: dict[str, np.ndarray], eps: float):
    ids = sorted(pts)
    found: dict[frozenset, float] = {}
    for i, j in itertools.combinations(ids, 2):
        a, b = pts[i], pts[j]
        if np.hypot(*(b - a)) <= eps:
            continue
        u = unit(b - a)
        S = frozenset(k for k in ids if line_dist(pts[k], a, u) <= eps)
        if len(S) < 3 or S in found:
            continue
        # verify against the line through the two farthest members
        mem = sorted(S)
        far = max(itertools.combinations(mem, 2), key=lambda pq: np.hypot(*(pts[pq[0]] - pts[pq[1]])))
        a2, u2 = pts[far[0]], unit(pts[far[1]] - pts[far[0]])
        dev = max(line_dist(pts[k], a2, u2) for k in mem)
        if dev <= eps:
            found[S] = dev
    maxi = [(S, d) for S, d in found.items() if not any(S < T for T in found)]
    return [(tuple(sorted(S)), d) for S, d in sorted(maxi, key=lambda sd: sorted(sd[0]))]


def _circumcircle(a, b, c):
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)) / d
    o = np.array([ux, uy])
    return o, float(np.hypot(*(a - o)))


def _maximal_concyclic(pts: dict[str, np.ndarray], eps: float):
    ids = sorted(pts)
    if len(ids) < 4:
        return []
    found: dict[frozenset, float] = {}
    for i, j, k in itertools.combinations(ids, 3):
        cc = _circumcircle(pts[i], pts[j], pts[k])
        if cc is None:
            continue
        o, r = cc
        if r > 1e6 * max(eps, 1e-12):
            continue
        S = frozenset(m for m in ids if abs(float(np.hypot(*(pts[m] - o))) - r) <= eps)
        if len(S) >= 4 and S not in found:
            found[S] = max(abs(float(np.hypot(*(pts[m] - o))) - r) for m in S)
    maxi = [(S, d) for S, d in found.items() if not any(S < T for T in found)]
    return [(tuple(sorted(S)), d) for S, d in sorted(maxi, key=lambda sd: sorted(sd[0]))]


def _param(C: _Ctx, x: str, p: np.ndarray) -> float:
    t = C.types[x]
    G = C.G
    if t in ("segment", "line", "ray", "axis"):
        a, u = C.line_of(x)
        return float((p - a) @ u)
    if t == "arc":
        g = G.E[x]["geom"]
        c = G.center(x)
        return (math.degrees(math.atan2(p[1] - c[1], p[0] - c[0])) - g["t0"]) % 360.0
    pl = G.polylines(x)[0]
    seglen = np.hypot(*np.diff(pl, axis=0).T)
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    best, bd = 0.0, math.inf
    for i, (a, b) in enumerate(zip(pl[:-1], pl[1:])):
        d, t_ = seg_dist(p, a, b)
        if d < bd:
            bd, best = d, cum[i] + t_ * seglen[i]
    return float(best)


def tolerances_10_2(th, D: float) -> dict:
    """§10.2 tolerances (CGX) at render diagonal D."""
    return {"eps_on": th.value("eps_on_frac") * D, "tangency_rel": th.value("tangency_rel"),
            "parallel_deg": th.value("parallel_deg"), "perpendicular_deg": th.value("perpendicular_deg"),
            "label_bind": th.value("label_bind_frac") * D, "axis_min_length_frac": th.value("axis_min_length_frac")}


# Numerical exactness for Γ over solved coordinates (not a threshold: facts that hold by
# construction hold to float precision). Label binding keeps the §10.3 radius.
EXACT_REL = 1e-6
EXACT_DEG = 1e-4


def tolerances_exact(th, D: float) -> dict:
    return {"eps_on": EXACT_REL * D, "tangency_rel": EXACT_REL, "parallel_deg": EXACT_DEG,
            "perpendicular_deg": EXACT_DEG, "label_bind": th.value("label_bind_frac") * D,
            "axis_min_length_frac": th.value("axis_min_length_frac")}


__all__ = ["evaluate_geo", "holds_geo", "bind_labels", "classify_straight", "AXIS_NAMES", "to_internal", "from_internal", "tolerances_10_2",
           "tolerances_exact", "GEO_PREDICATES", "GEO_FORMAT_VERSION"]
_ = ellipse_pts, polyline_dist
