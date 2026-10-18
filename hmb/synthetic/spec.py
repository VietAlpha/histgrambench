"""Known-geometry specification model and geometry (test fixtures; `Geo` is also the production
geometry of CGX and adjudication).

A specification is a set of entities with solved coordinates in TikZ units (cm), plus style.
Geometry values that name a point may be given either as a point entity id (a reference, so
dependent entities move with the point) or as an inline ``[x, y]`` coordinate.

Entity geometry by type (all angles in degrees, counter-clockwise):
  point               {"x", "y"}
  segment/axis/arrow  {"a": ref, "b": ref}
  line                {"a": ref, "b": ref, "ext": [t0, t1]}  semantically infinite; drawn a+t0(b-a)..a+t1(b-a)
  ray                 {"a": ref, "b": ref, "t1": t}           origin a, through b, drawn to a+t(b-a)
  circle              {"c": ref, "r"}
  arc                 {"c": ref, "r", "t0", "t1"}             t0 < t1 < t0 + 360
  ellipse             {"c": ref, "rx", "ry", "rot"}
  polygon             {"v": [ref, ...]}
  region              {"pts": [ref, ...]}                      filled closed boundary
  curve               {"pts": [[x, y], ...], "f": <curve_f>, "params": {...}}
  solid               {"k": <solid_k>, "vertices": [ref...], "edges": [[i, j]...], "faces": [[i, j, k...]...]}
  angle_marker        {"v": ref, "a0", "a1", "r", "n": arcs, "eq": bool}
  right_angle_marked  {"v": ref, "d1": [ux, uy], "d2": [ux, uy], "s": size}
  length_tick         {"c": [x, y], "dir": [ux, uy], "n": ticks, "seg": segment id}

Labels: an entity's ``label`` text is drawn at ``label_pos`` (the node centre). Measure labels
(``measure_label(t, e, kind)``) are free-standing texts with an intended target.
Style carries only allowed-variation quantities (§7.6): line width, colour, font, dot size.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import numpy as np

CM_PER_PT = 2.54 / 72.27
STANDALONE_BORDER_CM = 0.2          # wrapper_v1 `border=2mm` (§9.3)

DEFAULT_STYLE = {"line_width_pt": 0.4, "color": "black", "font": "rm", "dot_pt": 1.2,
                 "label_font_pt": 10.0}
DEFAULT_RENDER = {"dpi_delta": 0, "pad_cm": 0.0, "refactor": False}


@dataclass
class Entity:
    id: str
    type: str
    geom: dict
    label: str | None = None
    label_pos: list[float] | None = None
    style: dict = field(default_factory=dict)      # {"dashed": bool}


@dataclass
class Spec:
    spec_id: str
    family: str
    seed: int
    canvas: dict
    entities: list[Entity]
    measure_labels: list[dict] = field(default_factory=list)   # {text, kind, pos, target}
    style: dict = field(default_factory=lambda: dict(DEFAULT_STYLE))
    render: dict = field(default_factory=lambda: dict(DEFAULT_RENDER))
    meta: dict = field(default_factory=dict)

    # ---------------------------------------------------------------- access
    def ent(self, eid: str) -> Entity:
        for e in self.entities:
            if e.id == eid:
                return e
        raise KeyError(eid)

    def has(self, eid: str) -> bool:
        return any(e.id == eid for e in self.entities)

    def ids(self, *types: str) -> list[str]:
        return [e.id for e in self.entities if not types or e.type in types]

    def entity_types(self) -> dict[str, str]:
        return {e.id: e.type for e in self.entities}

    def copy(self) -> "Spec":
        return copy.deepcopy(self)

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "Spec":
        d = copy.deepcopy(d)
        d["entities"] = [Entity(**e) for e in d["entities"]]
        return cls(**d)

    # ---------------------------------------------------------------- geometry
    def pt(self, ref: Any) -> np.ndarray:
        if isinstance(ref, str):
            e = self.ent(ref)
            if e.type != "point":
                raise ValueError(f"reference {ref!r} is a {e.type}, not a point")
            return np.array([e.geom["x"], e.geom["y"]], float)
        return np.array(ref, float)


# ------------------------------------------------------------------------ primitive geometry
def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 0 else v


def seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Distance from p to segment ab and the clamped projection parameter t in [0, 1]."""
    ab = b - a
    L2 = float(ab @ ab)
    if L2 == 0:
        return float(np.hypot(*(p - a))), 0.0
    t = float(np.clip((p - a) @ ab / L2, 0.0, 1.0))
    return float(np.hypot(*(p - (a + t * ab)))), t


def line_dist(p: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
    d = p - a
    return abs(float(d[0] * u[1] - d[1] * u[0]))


def signed_line(p: np.ndarray, a: np.ndarray, u: np.ndarray) -> float:
    d = p - a
    return float(u[0] * d[1] - u[1] * d[0])


def polyline_dist(p: np.ndarray, pts: np.ndarray) -> float:
    a, b = pts[:-1], pts[1:]
    ab = b - a
    L2 = np.einsum("ij,ij->i", ab, ab)
    L2 = np.where(L2 == 0, 1e-300, L2)
    t = np.clip(np.einsum("ij,ij->i", p - a, ab) / L2, 0, 1)
    q = a + t[:, None] * ab
    return float(np.min(np.hypot(*(p - q).T)))


def point_in_polygon(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if xin > x:
                inside = not inside
    return inside


def ang_in(a: float, t0: float, t1: float) -> bool:
    """Is angle a (deg) within the ccw sweep t0 -> t1?"""
    span = (t1 - t0) % 360.0 or 360.0
    return (a - t0) % 360.0 <= span + 1e-12


def ellipse_pts(c: np.ndarray, rx: float, ry: float, rot: float, n: int = 721,
                t0: float = 0.0, t1: float = 360.0) -> np.ndarray:
    t = np.radians(np.linspace(t0, t1, n))
    ca, sa = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    x, y = rx * np.cos(t), ry * np.sin(t)
    return np.stack([c[0] + ca * x - sa * y, c[1] + sa * x + ca * y], axis=1)


# ------------------------------------------------------------------------ resolution
def resolve_entities(spec: Spec) -> list[dict]:
    """Spec -> internal resolved entity dicts ``{"entity_id","type","geom","style"}``: the per-type
    geometry keys of this module's docstring with every point reference replaced by coordinates."""
    out = []
    for e in spec.entities:
        g = copy.deepcopy(e.geom)
        if e.type == "polygon":
            g["v"] = [spec.pt(r).tolist() for r in g["v"]]
        elif e.type == "region":
            g["pts"] = [spec.pt(r).tolist() for r in g["pts"]]
        elif e.type == "solid":
            g["vertices"] = [spec.pt(r).tolist() for r in g["vertices"]]
        else:
            for k in ("a", "b", "c", "v"):
                if k in g:
                    g[k] = spec.pt(g[k]).tolist()
        out.append({"entity_id": e.id, "type": e.type, "geom": g, "style": dict(e.style)})
    return out


def spec_texts(spec: Spec) -> list[dict]:
    """Texts: entity name labels (kind "name") and measure labels (declared kind)."""
    out = []
    for e in spec.entities:
        if e.label and e.label_pos is not None:
            out.append({"text": e.label, "x": float(e.label_pos[0]), "y": float(e.label_pos[1]),
                        "kind": "name", "intended": e.id})
    for m in spec.measure_labels:
        out.append({"text": m["text"], "x": float(m["pos"][0]), "y": float(m["pos"][1]),
                    "kind": m["kind"], "intended": m.get("target")})
    return out


# ------------------------------------------------------------------------ resolved entity view
class Geo:
    """Geometry over resolved entity dicts (all coordinates inline, y up). Cached sampling."""

    def __init__(self, entities: list[dict], texts: list[dict] | None = None):
        self.ents = entities
        self.E = {e["entity_id"]: e for e in entities}
        self.texts = texts or []
        self._poly: dict[str, list[np.ndarray]] = {}

    @classmethod
    def of_spec(cls, spec: Spec) -> "Geo":
        return cls(resolve_entities(spec), spec_texts(spec))

    @staticmethod
    def pt(ref) -> np.ndarray:
        return np.array(ref, float)

    def type(self, eid: str) -> str:
        return self.E[eid]["type"]

    def ids(self, *types: str) -> list[str]:
        return [e["entity_id"] for e in self.ents if not types or e["type"] in types]

    def point(self, eid: str) -> np.ndarray:
        g = self.E[eid]["geom"]
        return np.array([g["x"], g["y"]], float)

    def ab(self, eid: str) -> tuple[np.ndarray, np.ndarray]:
        g = self.E[eid]["geom"]
        return self.pt(g["a"]), self.pt(g["b"])

    def direction(self, eid: str) -> np.ndarray:
        a, b = self.ab(eid)
        return unit(b - a)

    def drawn_ends(self, eid: str) -> tuple[np.ndarray, np.ndarray]:
        e = self.E[eid]
        a, b = self.ab(eid)
        if e["type"] == "line":
            t0, t1 = e["geom"].get("ext", [0.0, 1.0])
            return a + t0 * (b - a), a + t1 * (b - a)
        if e["type"] == "ray":
            return a, a + e["geom"].get("t1", 1.0) * (b - a)
        return a, b

    def polylines(self, eid: str) -> list[np.ndarray]:
        """Dense sampling of the drawn geometry (distances, bbox, raster)."""
        if eid in self._poly:
            return self._poly[eid]
        e = self.E[eid]
        g = e["geom"]
        t = e["type"]
        if t == "point":
            out = [self.point(eid)[None, :]]
        elif t in ("segment", "axis", "arrow", "line", "ray"):
            a, b = self.drawn_ends(eid)
            out = [np.array([a, b])]
        elif t == "circle":
            out = [ellipse_pts(self.pt(g["c"]), g["r"], g["r"], 0.0)]
        elif t == "arc":
            n = max(8, int(abs(g["t1"] - g["t0"]) * 2) + 1)
            out = [ellipse_pts(self.pt(g["c"]), g["r"], g["r"], 0.0, n, g["t0"], g["t1"])]
        elif t == "ellipse":
            out = [ellipse_pts(self.pt(g["c"]), g["rx"], g["ry"], g["rot"])]
        elif t == "polygon":
            v = np.array([self.pt(r) for r in g["v"]])
            out = [np.vstack([v, v[:1]])]
        elif t == "region":
            v = np.array([self.pt(r) for r in g["pts"]])
            out = [np.vstack([v, v[:1]])]
        elif t == "curve":
            out = [np.array(g["pts"], float)]
        elif t == "solid":
            vs = [self.pt(r) for r in g.get("vertices", [])]
            out = [np.array([vs[i], vs[j]]) for i, j in g.get("edges", [])] or \
                [np.array(g.get("pts", [[0.0, 0.0]]), float)]
        elif t == "angle_marker":
            v = self.pt(g["v"])
            out = []
            for k in range(max(1, int(g.get("n", 1)))):
                r = g["r"] + 0.08 * k
                n = max(8, int(abs(g["a1"] - g["a0"])) + 1)
                out.append(ellipse_pts(v, r, r, 0.0, n, g["a0"], g["a1"]))
        elif t == "right_angle_marked":
            v, d1, d2, s = self.pt(g["v"]), np.array(g["d1"]), np.array(g["d2"]), g["s"]
            out = [np.array([v + s * d1, v + s * d1 + s * d2, v + s * d2])]
        elif t == "length_tick":
            c, d = np.array(g["c"], float), unit(np.array(g["dir"], float))
            nrm = np.array([-d[1], d[0]])
            h = g.get("h", 0.12)
            out = []
            n = int(g.get("n", 1))
            for k in range(n):
                off = (k - (n - 1) / 2) * 0.08
                cc = c + off * d
                out.append(np.array([cc - h * nrm, cc + h * nrm]))
        else:
            raise ValueError(f"unknown entity type {t}")
        self._poly[eid] = out
        return out

    # ---------------------------------------------------------------- distances
    def dist(self, eid: str, p: np.ndarray) -> float:
        """Distance from p to the *semantic* geometry of an entity (lines infinite)."""
        e = self.E[eid]
        g = e["geom"]
        t = e["type"]
        if t == "point":
            return float(np.hypot(*(p - self.point(eid))))
        if t in ("segment", "axis", "arrow"):
            a, b = self.ab(eid)
            return seg_dist(p, a, b)[0]
        if t == "line":
            a, b = self.ab(eid)
            return line_dist(p, a, unit(b - a))
        if t == "ray":
            a, b = self.ab(eid)
            u = unit(b - a)
            if (p - a) @ u < 0:
                return float(np.hypot(*(p - a)))
            return line_dist(p, a, u)
        if t == "circle":
            return abs(float(np.hypot(*(p - self.pt(g["c"])))) - g["r"])
        if t == "arc":
            c = self.pt(g["c"])
            ang = math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
            if ang_in(ang, g["t0"], g["t1"]):
                return abs(float(np.hypot(*(p - c))) - g["r"])
            return min(float(np.hypot(*(p - q))) for q in self.arc_ends(eid))
        return min(polyline_dist(p, pl) for pl in self.polylines(eid))

    def arc_ends(self, eid: str) -> list[np.ndarray]:
        g = self.E[eid]["geom"]
        c = self.pt(g["c"])
        return [c + g["r"] * np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
                for a in (g["t0"], g["t1"])]

    def endpoints(self, eid: str) -> list[np.ndarray]:
        """Points NOT in the relative interior (ends of open entities, polygon corners)."""
        e = self.E[eid]
        t = e["type"]
        if t in ("segment", "axis", "arrow"):
            return list(self.ab(eid))
        if t == "ray":
            return [self.ab(eid)[0]]
        if t == "arc":
            return self.arc_ends(eid)
        if t == "curve":
            p = np.array(e["geom"]["pts"], float)
            return [p[0], p[-1]]
        if t == "polygon":
            return [self.pt(r) for r in e["geom"]["v"]]
        return []

    def center(self, eid: str) -> np.ndarray:
        return self.pt(self.E[eid]["geom"]["c"])

    def signed_inside(self, eid: str, p: np.ndarray) -> float:
        """> 0 inside a closed entity (distance to boundary), < 0 outside."""
        e = self.E[eid]
        if e["type"] == "circle":
            return e["geom"]["r"] - float(np.hypot(*(p - self.center(eid))))
        bd = self.polylines(eid)[0]
        d = polyline_dist(p, bd)
        return d if point_in_polygon(p, bd[:-1]) else -d

    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = [], []
        for e in self.ents:
            for pl in self.polylines(e["entity_id"]):
                xs.extend(pl[:, 0])
                ys.extend(pl[:, 1])
        for t in self.texts:
            xs.append(t["x"])
            ys.append(t["y"])
        if not xs:
            return (0.0, 0.0, 0.0, 0.0)
        return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))

    def diagonal(self) -> float:
        x0, y0, x1, y1 = self.bbox()
        return float(math.hypot(x1 - x0, y1 - y0))


def canonical_px_per_cm(spec: Spec, th=None) -> float:
    """Pixels per cm on the §9.7 canonical canvas: render at 300 DPI (+ standalone 2 mm border),
    s = min(canvas_px / max(W, H), max_upscale)."""
    canvas_px = th.value("canvas_px") if th is not None else 1024
    max_up = th.value("max_upscale") if th is not None else 4.0
    dpi = (th.value("render_dpi_candidate") if th is not None else 300) + spec.render.get("dpi_delta", 0)
    x0, y0, x1, y1 = Geo.of_spec(spec).bbox()
    pad = spec.render.get("pad_cm", 0.0) + STANDALONE_BORDER_CM
    ppc = dpi / 2.54
    W, H = (x1 - x0 + 2 * pad) * ppc, (y1 - y0 + 2 * pad) * ppc
    s = min(canvas_px / max(W, H, 1e-9), max_up)
    return ppc * s


def stroke_px(spec: Spec, th=None) -> float:
    return spec.style.get("line_width_pt", 0.4) * CM_PER_PT * canonical_px_per_cm(spec, th)


# ------------------------------------------------------------------------ transforms
def _is_similarity(A: np.ndarray) -> bool:
    M = A.T @ A
    return abs(M[0, 1]) < 1e-12 and abs(M[0, 0] - M[1, 1]) < 1e-12 * max(1.0, abs(M[0, 0]))


def _ellipse_affine(c, rx, ry, rot, A, t):
    R = np.array([[math.cos(math.radians(rot)), -math.sin(math.radians(rot))],
                  [math.sin(math.radians(rot)), math.cos(math.radians(rot))]])
    M = A @ R @ np.diag([rx, ry])
    U, S, _ = np.linalg.svd(M)
    if np.linalg.det(U) < 0:
        U[:, 1] *= -1
    new_rot = math.degrees(math.atan2(U[1, 0], U[0, 0]))
    return (A @ c + t).tolist(), float(S[0]), float(S[1]), new_rot


def _map_angle(a_deg: float, A: np.ndarray) -> float:
    d = A @ np.array([math.cos(math.radians(a_deg)), math.sin(math.radians(a_deg))])
    return math.degrees(math.atan2(d[1], d[0]))


def apply_affine(spec: Spec, A: Iterable, t: Iterable = (0.0, 0.0)) -> Spec:
    """x -> A x + t over every coordinate. Text is not transformed (TikZ nodes keep their glyphs).

    A non-similarity map turns circles into ellipses and arcs into elliptical-arc curves, which
    is what `xscale`/`yscale` does to a rendered figure.
    """
    A = np.array(A, float)
    t = np.array(t, float)
    sim = _is_similarity(A)
    k = math.sqrt(abs(np.linalg.det(A)))
    flip = np.linalg.det(A) < 0
    s = spec.copy()
    geo_old = Geo.of_spec(spec)

    def mp(ref):
        if isinstance(ref, str):
            return ref
        return (A @ np.array(ref, float) + t).tolist()

    for e in s.entities:
        g = e.geom
        ty = e.type
        if ty == "point":
            x, y = A @ np.array([g["x"], g["y"]]) + t
            g["x"], g["y"] = float(x), float(y)
        elif ty in ("segment", "axis", "arrow", "line", "ray"):
            g["a"], g["b"] = mp(g["a"]), mp(g["b"])
        elif ty in ("polygon",):
            g["v"] = [mp(r) for r in g["v"]]
        elif ty == "region":
            g["pts"] = [mp(r) for r in g["pts"]]
        elif ty == "solid":
            g["vertices"] = [mp(r) for r in g["vertices"]]
        elif ty == "circle":
            if sim:
                g["c"], g["r"] = mp(g["c"]), g["r"] * k
            else:
                c = spec.pt(g["c"])
                cc, rx, ry, rot = _ellipse_affine(c, g["r"], g["r"], 0.0, A, t)
                e.type = "ellipse"
                e.geom = {"c": g["c"] if isinstance(g["c"], str) else cc, "rx": rx, "ry": ry, "rot": rot}
        elif ty == "ellipse":
            c = spec.pt(g["c"])
            cc, rx, ry, rot = _ellipse_affine(c, g["rx"], g["ry"], g["rot"], A, t)
            e.geom = {"c": g["c"] if isinstance(g["c"], str) else cc, "rx": rx, "ry": ry, "rot": rot}
        elif ty == "arc":
            if sim:
                a0, a1 = _map_angle(g["t0"], A), _map_angle(g["t1"], A)
                if flip:
                    a0, a1 = a1, a0
                span = (a1 - a0) % 360.0 or 360.0
                g["c"], g["r"], g["t0"], g["t1"] = mp(g["c"]), g["r"] * k, a0, a0 + span
            else:
                pts = geo_old.polylines(e.id)[0]
                e.type = "curve"
                e.geom = {"pts": ((A @ pts.T).T + t).tolist(), "f": "ellipse_arc", "params": {}}
        elif ty == "curve":
            pts = np.array(g["pts"], float)
            g["pts"] = ((A @ pts.T).T + t).tolist()
            if g.get("params", {}).get("kind") == "hyperbola":
                g["params"] = _hyperbola_affine(g["params"], A, t)
        elif ty == "angle_marker":
            a0, a1 = _map_angle(g["a0"], A), _map_angle(g["a1"], A)
            if flip:
                a0, a1 = a1, a0
            span = (a1 - a0) % 360.0 or 360.0
            g["v"], g["a0"], g["a1"], g["r"] = mp(g["v"]), a0, a0 + span, g["r"] * (k if sim else 1.0)
        elif ty == "right_angle_marked":
            g["v"] = mp(g["v"])
            g["d1"] = unit(A @ np.array(g["d1"])).tolist()
            g["d2"] = unit(A @ np.array(g["d2"])).tolist()
            g["s"] = g["s"] * (k if sim else 1.0)
        elif ty == "length_tick":
            g["c"] = (A @ np.array(g["c"]) + t).tolist()
            g["dir"] = unit(A @ np.array(g["dir"])).tolist()
        if e.label_pos is not None:
            e.label_pos = (A @ np.array(e.label_pos) + t).tolist()
    for m in s.measure_labels:
        m["pos"] = (A @ np.array(m["pos"]) + t).tolist()
    return s


def _hyperbola_affine(params: dict, A: np.ndarray, t: np.ndarray) -> dict:
    """Carry the asymptote directions of a hyperbola through an affine map."""
    p = dict(params)
    p["c"] = (A @ np.array(params["c"], float) + t).tolist()
    p["asymptote_dirs"] = [unit(A @ np.array(d, float)).tolist() for d in params["asymptote_dirs"]]
    return p


def translate(spec: Spec, dx: float, dy: float) -> Spec:
    return apply_affine(spec, np.eye(2), (dx, dy))


def scale_about(spec: Spec, sx: float, sy: float, center: Iterable[float]) -> Spec:
    c = np.array(center, float)
    A = np.diag([sx, sy])
    return apply_affine(spec, A, c - A @ c)



def inline_point(spec: Spec, pid: str) -> None:
    """Replace every reference to point `pid` by its coordinates (used before deleting it)."""
    xy = spec.pt(pid).tolist()

    def sub(r):
        return list(xy) if r == pid else r

    for e in spec.entities:
        g = e.geom
        for k in ("a", "b", "c", "v"):
            if k in g and isinstance(g[k], str) and g[k] == pid:
                g[k] = list(xy)
        for k in ("v", "pts", "vertices"):
            if k in g and isinstance(g[k], list) and g[k] and not isinstance(g[k][0], (int, float)):
                g[k] = [sub(r) for r in g[k]]
