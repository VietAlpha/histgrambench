"""Geometry access for CGX primitives, labels and glyphs — through the SHARED geometry format.

Primitive `geom` dicts are in the shared evaluator format (CONTRACTS.md "Geometric predicate
semantics"; `hmb.synthetic.geo_predicates`): point {x, y}, segment/ray/line/axis/arrow {p, q},
circle {c, r}, arc {c, r, p, q}, ellipse {c, rx, ry, rot_deg}, polygon {vertices}, curve {points}, ...
This module does not parse coordinates itself: `CgxGeometry` rebuilds the evaluator geometry with
`hmb.cgx.cgx_geometry`, resolves it with `geo_predicates.to_internal` into `synthetic.spec.Geo`
(internal frame, y up), and binds label positions with the shared `geo_predicates.bind_labels`
(nearest point within label_bind, else the nearest non-point entity — C28: unlabelled junctions are
not points, so binding to a non-point is the normal case). Label/glyph boxes are in the CGX page
frame; they are converted with the geometry's own `y_up` flag.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np

from hmb.cgx import cgx_geometry
from hmb.synthetic.geo_predicates import bind_labels, to_internal
from hmb.synthetic.spec import Geo

# Card entity type -> CGX primitive types that can realise it (a card line or ray may be drawn and
# extracted as a finite segment).
TYPE_COMPAT: dict[str, frozenset[str]] = {
    "line": frozenset({"line", "segment"}),
    "ray": frozenset({"ray", "segment"}),
    "curve": frozenset({"curve", "curve_generic"}),
}


def compatible(card_type: str, prim_type: str) -> bool:
    return prim_type in TYPE_COMPAT.get(card_type, frozenset({card_type}))


def ontology_type(prim_type: str) -> str:
    return "curve" if prim_type == "curve_generic" else prim_type


def bbox_center(bbox: Iterable[float]) -> tuple[float, float]:
    x0, y0, x1, y1 = list(bbox)
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


class CgxGeometry:
    """Shared-format geometry of one CGX record. Positions returned by `anchor`, `polylines` and
    `extent` are in the internal y-up frame; `dist` and `label_owners` take page-frame points."""

    def __init__(self, cgx):
        self.cgx = cgx
        self.shared = cgx_geometry(cgx)
        self.sy = 1.0 if self.shared.get("y_up", True) else -1.0
        ents, texts = to_internal(self.shared)
        self.G = Geo(ents, texts)
        self.types = {e["entity_id"]: e["type"] for e in ents}

    def internal(self, xy: tuple[float, float]) -> np.ndarray:
        return np.array([float(xy[0]), self.sy * float(xy[1])])

    def dist(self, prim_id: str, xy_page: tuple[float, float]) -> float:
        return float(self.G.dist(prim_id, self.internal(xy_page)))

    def anchor(self, prim_id: str) -> tuple[float, float]:
        t = self.types[prim_id]
        if t == "point":
            p = self.G.point(prim_id)
        elif t in ("circle", "arc", "ellipse"):
            p = self.G.center(prim_id)
        else:
            p = np.vstack(self.G.polylines(prim_id)).mean(axis=0)
        return float(p[0]), float(p[1])

    def polylines(self, prim_id: str) -> list[np.ndarray]:
        return self.G.polylines(prim_id)

    def extent(self) -> tuple[float, float, float, float] | None:
        """Content bounding box (internal frame) of every primitive and text box."""
        pts = [pl for pid in self.types for pl in self.G.polylines(pid)]
        for gl in list(self.cgx.glyphs) + list(self.cgx.labels):
            if gl.get("bbox"):
                x0, y0, x1, y1 = gl["bbox"]
                pts.append(np.array([self.internal((x0, y0)), self.internal((x1, y1))]))
        if not pts:
            return None
        a = np.vstack(pts)
        return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())

    def label_owners(self, centers_page: list[tuple[float, float]], label_bind: float,
                     eps_on: float) -> list[str | None]:
        """Owner of each page-frame text position under the shared §10.3 rule (one entry per input)."""
        if not centers_page or not self.types:
            return [None] * len(centers_page)
        geom = {**self.shared, "texts": [{"text": f"t{i}", "x": float(x), "y": float(y)}
                                         for i, (x, y) in enumerate(centers_page)]}
        return [b[1] for b in bind_labels(geom, {"eps_on": eps_on, "label_bind": label_bind})]

    def px_per_pt(self, canvas_px: int, max_upscale: float, dpi: int) -> float | None:
        """§9.7 scale from PDF points to canonical-render pixels, ESTIMATED from the content extent
        (assumes the standalone page equals the content box; a page border makes the true scale
        smaller). s = min(canvas / max(W, H), max_upscale) with W, H in pixels at `dpi`."""
        ext = self.extent()
        if ext is None:
            return None
        long_px = max(ext[2] - ext[0], ext[3] - ext[1]) * dpi / 72.0
        if long_px <= 0:
            return None
        return min(canvas_px / long_px, max_upscale) * dpi / 72.0


