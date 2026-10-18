"""Construction helpers for the generator families: a spec builder and deterministic label placement."""
from __future__ import annotations

import math

import numpy as np

from .spec import Entity, Geo, Spec, polyline_dist, unit

LABEL_OFFSET_CM = 0.30          # label node centre distance from its point (drawing convention)
LABEL_CHAR_W_CM = 0.22          # label box width per character at 10 pt (drawing convention)
LABEL_H_CM = 0.32


def label_box(text: str, pos) -> tuple[float, float, float, float]:
    w = LABEL_CHAR_W_CM * max(1, len(text)) + 0.06
    return (pos[0] - w / 2, pos[1] - LABEL_H_CM / 2, pos[0] + w / 2, pos[1] + LABEL_H_CM / 2)


def boxes_overlap(a, b) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


class B:
    """Mutable spec under construction."""

    def __init__(self, family: str, seed: int, canvas=(10.0, 10.0)):
        self.family, self.seed = family, seed
        self.canvas = {"w": float(canvas[0]), "h": float(canvas[1])}
        self.ents: list[Entity] = []
        self.measure: list[dict] = []
        self.meta: dict = {}

    def add(self, eid, etype, geom, label=None, **style) -> str:
        self.ents.append(Entity(eid, etype, geom, label, None, dict(style)))
        return eid

    def point(self, name: str, xy, label: bool = True) -> str:
        return self.add(f"p_{name}", "point", {"x": float(xy[0]), "y": float(xy[1])},
                        name if label else None)

    def seg(self, p: str, q: str, eid: str | None = None, **style) -> str:
        return self.add(eid or f"s_{p[2:]}{q[2:]}", "segment", {"a": p, "b": q}, **style)

    def measure_label(self, text: str, kind: str, pos, target: str) -> None:
        self.measure.append({"text": text, "kind": kind, "pos": [float(pos[0]), float(pos[1])],
                             "target": target})

    def xy(self, pid: str) -> np.ndarray:
        for e in self.ents:
            if e.id == pid:
                return np.array([e.geom["x"], e.geom["y"]], float)
        raise KeyError(pid)

    def spec(self) -> Spec:
        s = Spec(spec_id=f"tc-{self.family}-{self.seed}", family=self.family, seed=self.seed,
                 canvas=dict(self.canvas), entities=self.ents, measure_labels=self.measure,
                 meta=dict(self.meta))
        place_labels(s)
        return s


def place_labels(spec: Spec) -> None:
    """Choose each missing label position among 16 directions around its entity, maximizing the
    clearance to all drawn geometry and previously placed labels (deterministic)."""
    G = Geo.of_spec(spec)
    geoms = {e.id: np.vstack(G.polylines(e.id)) for e in spec.entities}
    placed: list[tuple[float, float, float, float]] = []
    for m in spec.measure_labels:
        placed.append(label_box(m["text"], m["pos"]))
    for e in spec.entities:
        if not e.label or e.label_pos is not None:
            continue
        if e.type == "point":
            anchor = np.array([e.geom["x"], e.geom["y"]])
            offsets = [LABEL_OFFSET_CM * np.array([math.cos(a), math.sin(a)])
                       for a in np.radians(np.arange(0, 360, 22.5))]
        else:
            pl = np.vstack(G.polylines(e.id))
            anchor = pl[len(pl) // 2]
            offsets = [0.28 * np.array([math.cos(a), math.sin(a)])
                       for a in np.radians(np.arange(0, 360, 22.5))]
        best, best_score = None, -math.inf
        for off in offsets:
            pos = anchor + off
            box = label_box(e.label, pos)
            if box[0] < 0 or box[1] < 0 or box[2] > spec.canvas["w"] or box[3] > spec.canvas["h"]:
                continue
            corners = [np.array([box[0], box[1]]), np.array([box[2], box[1]]),
                       np.array([box[0], box[3]]), np.array([box[2], box[3]]), pos]
            clear = math.inf
            for oid, g in geoms.items():
                if oid == e.id:
                    continue
                for c in corners:
                    d = polyline_dist(c, g) if len(g) > 1 else float(np.hypot(*(c - g[0])))
                    clear = min(clear, d)
            if any(boxes_overlap(box, b) for b in placed):
                clear -= 10.0
            if clear > best_score:
                best, best_score = pos, clear
        if best is None:
            best = anchor + offsets[0]
        e.label_pos = [float(best[0]), float(best[1])]
        placed.append(label_box(e.label, best))


def rng_for(family: str, seed: int) -> np.random.Generator:
    """Deterministic generator from (family, seed) (independent of Python hash randomization)."""
    from hmb.hashing import sha256_bytes
    h = int(sha256_bytes(f"{family}:{seed}".encode())[:16], 16)
    return np.random.default_rng(h)


def dir_deg(v) -> float:
    return math.degrees(math.atan2(v[1], v[0]))


def foot(p, a, b) -> np.ndarray:
    u = unit(b - a)
    return a + ((p - a) @ u) * u
