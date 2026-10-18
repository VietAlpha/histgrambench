"""Deterministic predicate evaluator for known-geometry specifications -> oracle graph Γ(s) (test fixtures).

Γ(s) = MGS( geo facts ∪ model-class facts ) under `hmb.cards.closure`, where
  * geo facts come from the shared pure function `geo_predicates.evaluate_geo` (the same code the
    CGX extractor calls), run with exact tolerances over the solved coordinates;
  * model-class facts come from what the specification draws or states, never from a guess:
      region(e), curve(e, f), solid(e, k)       - entity declarations (f, k from the spec)
      equal_length(x, y)                        - only when both segments carry length ticks with
                                                  the same tick count (a tick counts when its centre
                                                  lies on the segment)
      equal_angle(x, y)                         - only equality-marked angle markers (eq=True) with
                                                  the same arc count
      hidden_edge(s)                            - a dashed segment that is an edge of a solid
      face_of(g, s), cross_section_of(g, s)     - polygon matching a declared face of solid s /
                                                  polygon with every vertex on an edge of s, not a face
      asymptote_of(l, c)                        - linear l through the hyperbola centre, parallel to a
                                                  declared asymptote direction
      measure_label(t, e, kind)                 - measure text bound (§10.3 radius) to the nearest
                                                  entity of the kind's domain

C28 (docs/CONTRACTS.md "Geometric predicate semantics"):
  * a point entity exists only when it carries a name label or is drawn as a dot; other points are
    internal geometry (their coordinates shape strokes, but they are never entities or arguments);
  * straight-stroke types (segment / ray / line / axis / arrow) come from `classify_straight` on the
    drawn stroke, the same shared function the CGX extractor uses, not from the generator's intent.
C30: polygons come from the straight-stroke graph with the CGX extractor's own
`polygons_from_strokes`; every polygon edge is a segment entity. Label binding prefers drawn dots,
then other point entities, then non-point entities (`bind_labels`).
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import numpy as np

from hmb.cards.closure import forward_chain, minimal_generating_set
from hmb.hashing import sha256_bytes
from hmb.ontology import canonical_args

from .geo_predicates import (AXIS_NAMES, EXACT_DEG, bind_labels, classify_straight, evaluate_geo,
                             from_internal, tolerances_10_2, tolerances_exact)
from .spec import Geo, Spec, line_dist, resolve_entities, spec_texts, unit

MEASURE_DOMAIN = {"length": ("segment",), "angle": ("angle_marker", "right_angle_marked"),
                  "coordinate": ("point",), "area": ("region", "polygon"), "name": None}


@dataclass
class OracleFacts:
    geo: list[tuple]                 # [(pred, args, margin)]
    model: list[tuple]               # [(pred, args)]
    facts: set                       # all true facts (FactKeys), before closure
    closed: set                      # forward_chain(facts)
    mgs: list                        # Γ(s)
    D: float
    tol: dict = field(default_factory=dict)
    entities: list = field(default_factory=list)     # C28 Γ entities (resolved, derived types)
    types: dict = field(default_factory=dict)


STRAIGHT = ("segment", "line", "ray", "axis", "arrow")


def has_identity(e) -> bool:
    """C28: a point is an entity only when it carries a name label or is drawn as a dot."""
    return bool(e.label) or bool(e.style.get("dot", True))


def gamma_entities(spec: Spec, tol: dict) -> list[dict]:
    """C28/C30 Γ entities: identity-less points dropped; straight strokes typed by `classify_straight`;
    every polygon edge also a segment entity."""
    ident = {e.id for e in spec.entities if e.type != "point" or has_identity(e)}
    ents = [e for e in resolve_entities(spec) if e["entity_id"] in ident]
    for e in ents:                        # C28 binding order: drawn dots carry marker = true
        if e["type"] == "point":
            src = spec.ent(e["entity_id"])
            e["geom"]["marker"] = bool(src.style.get("dot", True))
    texts = [t for t in spec_texts(spec) if t["kind"] == "name"]
    G = Geo(ents, texts)
    pts = [[e["geom"]["x"], e["geom"]["y"]] for e in ents if e["type"] == "point"]
    axis_texts = [t for t in texts if t["text"] in AXIS_NAMES]
    axis_bound = set()
    if axis_texts:
        axis_bound = {eid for _, eid, _ in bind_labels(from_internal(ents, axis_texts), tol) if eid}
    D = G.diagonal()
    out = []
    for e in ents:
        if e["type"] in STRAIGHT:
            a, b = G.drawn_ends(e["entity_id"])
            typ, A, B = classify_straight(a, b, pts, tol["eps_on"], tipped=e["type"] in ("axis", "arrow"),
                                          D=D, parallel_deg=tol["parallel_deg"],
                                          axis_label=e["entity_id"] in axis_bound,
                                          axis_min_frac=tol["axis_min_length_frac"])
            g = {"a": A, "b": B}
            if typ == "line":
                g["ext"] = [0.0, 1.0]
            elif typ == "ray":
                g["t1"] = 1.0
            e = dict(e, type=typ, geom=g)
        out.append(e)
    return c30_polygons(out, tol)


def c30_polygons(ents: list[dict], tol: dict) -> list[dict]:
    """C30/C34: polygons come ONLY from the function the CGX extractor uses
    (`hmb.cgx.polygons_from_strokes`): chordless simple stroke cycles after the collinear merge, <= 8
    corners, independent of drawing style (closed path or separate strokes) and corner labelling.
    A matching declared polygon keeps its id but stores the canonical vertex order.
    Strokes = drawn straight strokes (not axes / arrows) and the edges of drawn polygons, de-duplicated.
    A cycle matching a declared polygon keeps its id; others get the content-derived
    `poly_<sorted corner ids>`. Every polygon edge is a `segment` entity: the drawn stroke with those
    ends (forced to `segment`, not retyped by C28) or a synthesized `seg_<sorted endpoint ids>`.
    Long or coordinate-based names become `poly_h<hash>` / `seg_h<hash>` of the same content."""
    from hmb.cgx import polygons_from_strokes
    eps, straight = tol["eps_on"], tol["parallel_deg"]

    def same(a, b, x, y) -> bool:
        return max(np.hypot(*(a - x)), np.hypot(*(b - y))) <= eps or max(np.hypot(*(a - y)), np.hypot(*(b - x))) <= eps
    strokes: list[tuple[np.ndarray, np.ndarray]] = []

    def add_stroke(a, b):
        if not any(same(a, b, x, y) for x, y in strokes):
            strokes.append((a, b))
    lines = [e for e in ents if e["type"] in ("segment", "line", "ray")]
    G = Geo(ents)
    for e in lines:
        a, b = G.drawn_ends(e["entity_id"])
        add_stroke(np.asarray(a, float), np.asarray(b, float))
    declared = [e for e in ents if e["type"] == "polygon"]
    for e in declared:
        v = [np.asarray(x, float) for x in e["geom"]["v"]]
        for k in range(len(v)):
            add_stroke(v[k], v[(k + 1) % len(v)])
    # C34: ONE rule, independent of drawing style and corner labelling. No closed-path special case
    # is passed; point positions are passed only while the shared function still takes them.
    import inspect
    kw = {}
    if "points" in inspect.signature(polygons_from_strokes).parameters:
        kw["points"] = [[e["geom"]["x"], e["geom"]["y"]] for e in ents if e["type"] == "point"]
    cycles = polygons_from_strokes(strokes, eps, straight, **kw)
    out = [e for e in ents if e["type"] != "polygon"]
    used_decl: set[str] = set()
    pts = [(e["entity_id"], np.array([e["geom"]["x"], e["geom"]["y"]], float)) for e in ents if e["type"] == "point"]

    G0 = Geo(ents)

    def corner_id(v: np.ndarray) -> str:
        """Identity of a corner from ENTITY ids only (red-team round 2 #7), never from coordinates, so
        it is invariant under every preserving transform: the point entity there; otherwise the
        sorted tokens of the drawn entities incident at the corner — `<stroke>.a` / `<stroke>.b` for a
        stroke END there, `<stroke>` for a stroke passing through, `<polygon>#<k>` for vertex k of a
        declared polygon."""
        for pid_, q in pts:
            if np.hypot(*(v - q)) <= eps:
                return pid_
        toks = []
        for e in lines:
            a, b = G0.drawn_ends(e["entity_id"])
            if np.hypot(*(v - np.asarray(a, float))) <= eps:
                toks.append(e["entity_id"] + ".a")
            elif np.hypot(*(v - np.asarray(b, float))) <= eps:
                toks.append(e["entity_id"] + ".b")
            elif G0.dist(e["entity_id"], v) <= eps:
                toks.append(e["entity_id"])
        for e in declared:
            for k, q in enumerate(e["geom"]["v"]):
                if np.hypot(*(v - np.asarray(q, float))) <= eps:
                    toks.append(f"{e['entity_id']}#{k}")
        if not toks:                                   # not reachable: a corner lies on its strokes
            raise ValueError("polygon corner with no incident entity")
        return "c(" + "+".join(sorted(toks)) + ")"

    def stable_id(prefix: str, names: list[str]) -> str:
        """Ids are CONTENT-derived (sorted corner / endpoint ids), so they are stable across mutations
        and preserving transforms; flip_set never reports a renaming as a flip. Long or compound
        names become `<prefix>_h<sha256[:12]>` of the same content."""
        joined = "_".join(sorted(names))
        sid = f"{prefix}_{joined}"
        return sid if len(sid) <= 48 and "c(" not in joined else f"{prefix}_h{sha256_bytes(joined.encode())[:12]}"

    for vs in cycles:
        vs = [np.asarray(x, float) for x in vs]
        pid = None
        for e in declared:
            dv = [np.asarray(x, float) for x in e["geom"]["v"]]
            if len(dv) == len(vs) and all(any(np.hypot(*(x - y)) <= eps for y in dv) for x in vs):
                pid, geom_v = e["entity_id"], [x.tolist() for x in vs]   # declared id, CANONICAL order
                break
        if pid is None:
            pid, geom_v = stable_id("poly", [corner_id(v) for v in vs]), [x.tolist() for x in vs]
        if pid in used_decl:
            continue
        used_decl.add(pid)
        out.append({"entity_id": pid, "type": "polygon", "geom": {"v": geom_v}, "style": {}})
        for k in range(len(vs)):
            a, b = vs[k], vs[(k + 1) % len(vs)]
            hit = None
            for e in out:
                if e["type"] in ("segment", "line", "ray"):
                    x, y = Geo([e]).drawn_ends(e["entity_id"])
                    if same(a, b, np.asarray(x, float), np.asarray(y, float)):
                        hit = e
                        break
            if hit is not None:
                if hit["type"] != "segment":
                    x, y = Geo([hit]).drawn_ends(hit["entity_id"])
                    hit["type"], hit["geom"] = "segment", {"a": list(x), "b": list(y)}
            else:
                out.append({"entity_id": stable_id("seg", [corner_id(a), corner_id(b)]), "type": "segment",
                            "geom": {"a": a.tolist(), "b": b.tolist()}, "style": {}})
    # C34: a declared polygon the rule does not yield is NOT a Γ polygon (one rule on both sides)
    return out


def spec_geom(spec: Spec, tol: dict) -> dict:
    """Spec -> shared geo format over the C28 Γ entities; only NAME labels are texts (measure
    labels are model-class)."""
    texts = [t for t in spec_texts(spec) if t["kind"] == "name"]
    return from_internal(gamma_entities(spec, tol), texts)


def diagonal(spec: Spec) -> float:
    return Geo.of_spec(spec).diagonal()


def model_facts(spec: Spec, tol: dict, types: dict[str, str] | None = None,
                ents: list[dict] | None = None) -> list[tuple]:
    G = Geo.of_spec(spec)
    eps = tol["eps_on"]
    out: list[tuple] = []
    if ents is None:
        ents = gamma_entities(spec, tol)
    types = types if types is not None else {e["entity_id"]: e["type"] for e in ents}
    for e in spec.entities:
        if e.type == "region":
            out.append(("region", [e.id]))
        elif e.type == "curve":
            out.append(("curve", [e.id, e.geom.get("f", "unspecified")]))
        elif e.type == "solid":
            out.append(("solid", [e.id, e.geom["k"]]))
    # equal_length via ticks
    ticked: dict[int, list[str]] = {}
    for e in spec.entities:
        if e.type != "length_tick":
            continue
        seg = e.geom.get("seg")
        if seg in types and types[seg] == "segment" and G.dist(seg, np.array(e.geom["c"])) <= eps:
            ticked.setdefault(int(e.geom.get("n", 1)), []).append(seg)
    for n, segs in ticked.items():
        for x, y in itertools.combinations(sorted(set(segs)), 2):
            out.append(("equal_length", [x, y]))
    # equal_angle via equality-marked angle markers
    marked: dict[int, list[str]] = {}
    for e in spec.entities:
        if e.type == "angle_marker" and e.geom.get("eq"):
            marked.setdefault(int(e.geom.get("n", 1)), []).append(e.id)
    for n, ms in marked.items():
        for x, y in itertools.combinations(sorted(ms), 2):
            out.append(("equal_angle", [x, y]))
    # solids: hidden edges, faces, cross-sections
    for s in spec.entities:
        if s.type != "solid":
            continue
        verts = [spec.pt(r) for r in s.geom["vertices"]]
        edges = [(verts[i], verts[j]) for i, j in s.geom["edges"]]
        faces = [sorted(f) for f in s.geom.get("faces", [])]
        for e in spec.entities:
            if types.get(e.id) == "segment" and e.style.get("dashed"):
                a, b = spec.pt(e.geom["a"]), spec.pt(e.geom["b"])
                if any(min(max(np.hypot(*(a - u)), np.hypot(*(b - w))),
                           max(np.hypot(*(a - w)), np.hypot(*(b - u)))) <= eps for u, w in edges):
                    out.append(("hidden_edge", [e.id]))
        # red-team round 2 #3: faces and cross-sections range over the Γ polygon ENTITIES (C30/C34
        # stroke-cycle polygons), not only declared spec polygons.
        for g in ents:
            if g["type"] != "polygon":
                continue
            pv = [np.asarray(q, float) for q in g["geom"]["v"]]
            idx = []
            for q in pv:
                m = [i for i, v in enumerate(verts) if np.hypot(*(q - v)) <= eps]
                idx.append(m[0] if m else None)
            if None not in idx and sorted(idx) in faces:
                out.append(("face_of", [g["entity_id"], s.id]))
            elif (all(any(_on_seg(q, u, w, eps) for u, w in edges) for q in pv)
                  and any(i is None for i in idx)):
                # a section CUTS the solid: every corner on an edge and at least one corner inside an
                # edge (not a solid vertex); a cycle of solid vertices that is not a face is neither
                out.append(("cross_section_of", [g["entity_id"], s.id]))
    # asymptotes
    for c in spec.entities:
        prm = c.geom.get("params", {}) if c.type == "curve" else {}
        if prm.get("kind") != "hyperbola":
            continue
        cc = np.array(prm["c"], float)
        for l in spec.entities:
            if types.get(l.id) not in ("line", "segment", "ray"):
                continue
            a, b = spec.pt(l.geom["a"]), spec.pt(l.geom["b"])
            u = unit(b - a)
            for d in prm["asymptote_dirs"]:
                ang = math.degrees(math.acos(min(1.0, abs(float(u @ unit(np.array(d, float)))))))
                if ang <= tol["parallel_deg"] and line_dist(cc, a, u) <= eps:
                    out.append(("asymptote_of", [l.id, c.id]))
                    break
    # measure labels
    lb = tol["label_bind"]
    for m in spec.measure_labels:
        p = np.array(m["pos"], float)
        dom = MEASURE_DOMAIN.get(m["kind"])
        best = None
        for e in spec.entities:
            et = types.get(e.id)
            if et is None or (dom is not None and et not in dom):
                continue
            if dom is None and et == "point":
                continue
            d = G.dist(e.id, p)
            if d <= lb and (best is None or d < best[1]):
                best = (e.id, d)
        if best is not None:
            out.append(("measure_label", [m["text"], best[0], m["kind"]]))
    return out


def _on_seg(q, u, w, eps) -> bool:
    ab = w - u
    L2 = float(ab @ ab)
    t = float(np.clip((q - u) @ ab / L2, 0, 1)) if L2 else 0.0
    return float(np.hypot(*(q - (u + t * ab)))) <= eps


def oracle_facts(spec: Spec, th, *, tol: dict | None = None) -> OracleFacts:
    """Γ(s): evaluate every applicable predicate and canonicalize to the MGS."""
    D = diagonal(spec)
    tol = tol or tolerances_exact(th, D)
    ents = gamma_entities(spec, tol)
    types = {e["entity_id"]: e["type"] for e in ents}
    texts = [t for t in spec_texts(spec) if t["kind"] == "name"]
    geo = evaluate_geo(from_internal(ents, texts), tol)
    model = model_facts(spec, tol, types, ents)
    facts = {(p, a) for p, a, _ in geo} | {(p, canonical_args(p, a)) for p, a in model}
    closed = forward_chain(facts, types)
    mgs = minimal_generating_set(facts, types)
    return OracleFacts(geo, model, facts, closed, mgs, D, tol, ents, types)


def tolerance_unstable(spec: Spec, th) -> list:
    """Facts on which exact and §10.2 tolerances disagree (an accidental near-coincidence).

    Admission requires this to be empty so every Γ fact is recoverable by the extractor in
    principle (this is the "no two entities coincide within tolerance" check made complete)."""
    D = diagonal(spec)
    te, t10 = tolerances_exact(th, D), tolerances_10_2(th, D)
    a = {(p, x) for p, x, _ in evaluate_geo(spec_geom(spec, te), te)}
    b = {(p, x) for p, x, _ in evaluate_geo(spec_geom(spec, t10), t10)}
    return sorted(a ^ b, key=repr)


def fact_dicts(mgs) -> list[dict]:
    return [{"predicate": p, "args": _jsonable(a)} for p, a in mgs]


def _jsonable(a):
    return [list(x) if isinstance(x, tuple) else x for x in a]


def flip_set(base: OracleFacts, mutated: OracleFacts) -> tuple[list, list]:
    """Derived diff Φ: base MGS facts not entailed by the mutated closure (removed) and mutated MGS
    facts not entailed by the base closure (added). Uses closure membership so an MGS
    representation change (e.g. a different parallel spanning tree) is never a flip."""
    from hmb.cards.closure import entails
    removed = [f for f in base.mgs if not entails(mutated.closed, f)]
    added = [f for f in mutated.mgs if not entails(base.closed, f)]
    return removed, added


__all__ = ["oracle_facts", "model_facts", "spec_geom", "flip_set", "fact_dicts",
           "tolerance_unstable", "OracleFacts", "EXACT_DEG"]
