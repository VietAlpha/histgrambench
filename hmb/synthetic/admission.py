"""Fixture admission: a known-geometry specification is admitted only if every entity lies inside the canvas, no two
distinct entities coincide within tolerance, and no label overlaps another. Two further checks make
those conditions operational: every label binds (§10.3 rule) to the entity it names, and the exact
and §10.2-tolerance evaluations agree (no accidental near-coincidence in any predicate).
"""
from __future__ import annotations

import itertools

import numpy as np

from .build import boxes_overlap, label_box
from .evaluator import diagonal, tolerance_unstable
from .geo_predicates import _Ctx, bind_texts, tolerances_10_2
from .spec import Geo, Spec, spec_texts
from .geo_predicates import from_internal


def admission_reasons(spec: Spec, th, *, check_binding: bool = True) -> list[str]:
    """Reasons for refusal (empty = admitted). `check_binding=False` for mutated specifications,
    where a label binding elsewhere is the mutation being measured, not a defect."""
    reasons: list[str] = []
    G = Geo.of_spec(spec)
    W, H = spec.canvas["w"], spec.canvas["h"]
    # inside canvas (geometry and label boxes)
    for e in spec.entities:
        for pl in G.polylines(e.id):
            if pl[:, 0].min() < 0 or pl[:, 1].min() < 0 or pl[:, 0].max() > W or pl[:, 1].max() > H:
                reasons.append(f"outside_canvas:{e.id}")
                break
    texts = spec_texts(spec)
    boxes = [(t, label_box(t["text"], (t["x"], t["y"]))) for t in texts]
    for t, bx in boxes:
        if bx[0] < 0 or bx[1] < 0 or bx[2] > W or bx[3] > H:
            reasons.append(f"label_outside_canvas:{t['text']}")
    # coincidence
    D = diagonal(spec)
    eps = th.value("eps_on_frac") * D
    pts = {e.id: np.array([e.geom["x"], e.geom["y"]]) for e in spec.entities if e.type == "point"}
    for a, b in itertools.combinations(sorted(pts), 2):
        if float(np.hypot(*(pts[a] - pts[b]))) <= eps:
            reasons.append(f"coincident:{a},{b}")
    by_type: dict[str, list[str]] = {}
    for e in spec.entities:
        if e.type != "point":
            by_type.setdefault(e.type, []).append(e.id)
    for t, ids in by_type.items():
        for a, b in itertools.combinations(ids, 2):
            pa, pb = np.vstack(G.polylines(a)), np.vstack(G.polylines(b))
            if len(pa) == len(pb) and (np.max(np.hypot(*(pa - pb).T)) <= eps
                                       or np.max(np.hypot(*(pa - pb[::-1]).T)) <= eps):
                reasons.append(f"coincident:{a},{b}")
    # label overlap
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        if boxes_overlap(b1, b2):
            reasons.append(f"label_overlap:{t1['text']},{t2['text']}")
    # label binding to the intended entity under the §10.3 rule
    if check_binding:
        name_texts = [t for t in texts if t["kind"] == "name"]
        from .evaluator import gamma_entities
        t10 = tolerances_10_2(th, D)
        C = _Ctx(from_internal(gamma_entities(spec, t10), name_texts), t10)
        bound = {text: eid for text, eid, _ in bind_texts(C)}
        for t in name_texts:
            if bound.get(t["text"]) != t["intended"]:
                reasons.append(f"label_binding:{t['text']}->{bound.get(t['text'])}")
        if spec.measure_labels:
            from .evaluator import model_facts
            from .geo_predicates import tolerances_exact
            mf = model_facts(spec, tolerances_exact(th, D))
            for m in spec.measure_labels:
                if ("measure_label", [m["text"], m["target"], m["kind"]]) not in mf:
                    reasons.append(f"measure_binding:{m['text']}")
    # every predicate stable between exact and §10.2 tolerances
    unstable = tolerance_unstable(spec, th)
    if unstable:
        reasons.append("tolerance_unstable:" + ";".join(f"{p}{list(a)}" for p, a in unstable[:4]))
    return reasons


def admitted(spec: Spec, th) -> bool:
    return not admission_reasons(spec, th)
