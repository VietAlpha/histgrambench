"""§10.3 card-to-candidate alignment (deterministic, before any relation is checked).

1. Label anchoring: each extracted name-label run is bound by the shared §10.3 rule
   (`geo_predicates.bind_labels`: nearest point within `label_bind_frac`·D, else the nearest
   non-point entity); a labelled card entity binds to that owner when the strictly normalized texts
   match and the types are compatible (two entities claiming one primitive leaves both for later).
2. Structural propagation to a fixpoint: an unbound entity whose eligible relations with bound
   entities single out exactly one compatible primitive is bound (a circle with a bound centre, a
   segment with two bound endpoints, ...).
3. Remainder: exact subgraph matching on type and incidence (every eligible relation among the
   bound + unmatched entities must hold on the images) when at most `subgraph_max_unmatched` (15)
   entities remain; entities whose image is the same in every embedding are bound, the others are
   ALIGNMENT_AMBIGUOUS. If there is no embedding, or more than 15 remain, Hungarian assignment on
   structural similarity (Jaccard of typed incidence signatures); ties and zero-similarity picks are
   ALIGNMENT_AMBIGUOUS. Entities with no compatible primitive left are absent.
Ambiguous entities go to Q3. The strict variant (labels must match for a relation to count) is a
post-processing of verdicts, `hmb.adjudication.core.strict_label_variant`, using `Alignment.method`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from hmb.cards.normalize import normalize_label
from hmb.ontology import PREDICATES, entity_args

from .geo import CgxIndex, map_args
from .geom import bbox_center, compatible

SOLUTION_CAP = 64
NODE_CAP = 200_000
MODEL_ONLY_TYPES = frozenset({"region", "solid"})     # never extracted as primitives


@dataclass
class Alignment:
    binding: dict[str, str] = field(default_factory=dict)
    method: dict[str, str] = field(default_factory=dict)
    ambiguous: set[str] = field(default_factory=set)
    absent: set[str] = field(default_factory=set)
    candidates: dict[str, list[str]] = field(default_factory=dict)
    not_aligned: set[str] = field(default_factory=set)       # model-only entity types
    # Judge-derived state (never allowed to reduce a denominator, red-team #1/#15):
    q3_asked: set[str] = field(default_factory=set)          # entities sent to Q3
    q3_absent: set[str] = field(default_factory=set)         # Q3 answered "none"
    det_bound: set[str] | None = None                        # entities bound before Q3 (None = binding keys)

    def deterministically_bound(self) -> set[str]:
        return set(self.binding) if self.det_bound is None else set(self.det_bound)

    @property
    def inverse(self) -> dict[str, str]:
        return {p: e for e, p in self.binding.items()}

    def to_json(self) -> dict:
        return {"binding": dict(sorted(self.binding.items())), "method": dict(sorted(self.method.items())),
                "ambiguous": sorted(self.ambiguous), "absent": sorted(self.absent),
                "q3_asked": sorted(self.q3_asked), "q3_absent": sorted(self.q3_absent),
                "candidates": {k: v for k, v in sorted(self.candidates.items())}}


def _type_group(t: str) -> str:
    if t in ("segment", "line", "ray", "axis", "arrow"):
        return "linear"
    if t in ("circle", "arc", "ellipse"):
        return "round"
    return t


def card_scope(card) -> tuple[dict[str, Any], list[Any]]:
    """Card entities referenced by eligible facts, and the eligible facts."""
    facts = [f for f in card.facts if f.eligible]
    used = {e for f in facts for e in entity_args(f.predicate, f.args)}
    ents = {e.entity_id: e for e in card.entities if e.entity_id in used}
    return ents, facts


def entity_labels(card, facts) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for e in card.entities:
        if e.label:
            out.setdefault(e.entity_id, []).append(e.label)
    for f in facts:
        if f.predicate == "label":
            out.setdefault(f.args[1], [])
            if f.args[0] not in out[f.args[1]]:
                out[f.args[1]].append(f.args[0])
    return out


def _label_objects(cgx) -> list[tuple[str, tuple[float, float]]]:
    """(normalized text, page-frame centre) of every name label run (measure labels never anchor)."""
    out = []
    for o in cgx.labels:
        if o.get("text") and o.get("bbox") and o.get("kind", "name") == "name":
            out.append((normalize_label(o["text"]), bbox_center(o["bbox"])))
    return out


PARAM_PREDICATES = ("axis", "polygon", "curve", "solid")      # entity facts carrying a parameter


def typed_compat(card, facts, index: CgxIndex):
    """§10.3 "type" for compatibility and matching includes the entity's parameter fact (axis
    orientation, polygon n, curve f, solid k). A card parameter and a candidate parameter asserted by
    the CGX record must agree; when either side asserts none, only the base type is compared (no
    geometric tie-break is added)."""
    ctype = {e.entity_id: e.type for e in card.entities}
    cparam = {f.args[0]: f.args[1] for f in facts if f.predicate in PARAM_PREDICATES}
    pparam: dict[str, object] = {}
    for k in index.closed:
        if k[0] in PARAM_PREDICATES:
            pparam.setdefault(k[1][0], k[1][1])

    def ok(eid: str, pid: str) -> bool:
        if not compatible(ctype[eid], index.prims[pid].type):
            return False
        a, b = cparam.get(eid), pparam.get(pid)
        return a is None or b is None or a == b
    return ok


def _structure_constraints(e: str, facts, binding: dict[str, str]):
    for f in facts:
        if PREDICATES[f.predicate].fclass.value != "relation":
            continue
        ea = entity_args(f.predicate, f.args)
        if e in ea and all(o == e or o in binding for o in ea):
            yield f


def align(card, index: CgxIndex, th) -> Alignment:
    cgx = index.cgx
    ents, facts = card_scope(card)
    al = Alignment()
    R = float(th.value("label_bind_frac")) * float(cgx.diagonal_D)
    labels = entity_labels(card, facts)
    tc = typed_compat(card, facts, index)
    lobjs = _label_objects(cgx)
    owners = index.geometry.label_owners([xy for _, xy in lobjs], R,
                                         float(th.value("eps_on_frac")) * float(cgx.diagonal_D))
    prims = list(cgx.primitives)
    for eid, e in ents.items():
        if e.type in MODEL_ONLY_TYPES:
            al.not_aligned.add(eid)
    if cgx.primitives_over_cap is not None:       # C43: the extractor kept no evidence; every fact goes to Q1
        al.not_aligned.update(ents)
    todo = sorted(e for e in ents if e not in al.not_aligned)

    # 1. label anchoring
    proposals: dict[str, str] = {}
    for eid in todo:
        texts = {normalize_label(t) for t in labels.get(eid, [])}
        hits: set[str] = set()
        for (norm, _xy), pid in zip(lobjs, owners):          # shared §10.3 binding rule
            if norm in texts and pid and tc(eid, pid):
                hits.add(pid)
        if len(hits) == 1:
            proposals[eid] = next(iter(hits))
    claimed: dict[str, list[str]] = {}
    for eid, pid in proposals.items():
        claimed.setdefault(pid, []).append(eid)
    for pid, es in claimed.items():
        if len(es) == 1:
            al.binding[es[0]] = pid
            al.method[es[0]] = "label"

    # 2. structural propagation
    changed = True
    while changed:
        changed = False
        used = set(al.binding.values())
        props: dict[str, str] = {}
        for eid in todo:
            if eid in al.binding:
                continue
            cand: set[str] | None = None
            for f in _structure_constraints(eid, facts, al.binding):
                ok = set()
                for p in prims:
                    if p.prim_id in used or not tc(eid, p.prim_id):
                        continue
                    m = map_args(f.predicate, f.args, {**al.binding, eid: p.prim_id})
                    if m is not None and index.holds(f.predicate, m)[0]:
                        ok.add(p.prim_id)
                cand = ok if cand is None else cand & ok
            if cand is not None and len(cand) == 1:
                props[eid] = next(iter(cand))
        rev: dict[str, list[str]] = {}
        for eid, pid in props.items():
            rev.setdefault(pid, []).append(eid)
        for pid, es in rev.items():
            if len(es) == 1:
                al.binding[es[0]] = pid
                al.method[es[0]] = "structure"
                changed = True

    # 3. remainder
    used = set(al.binding.values())
    free = [p for p in prims if p.prim_id not in used]
    U = [e for e in todo if e not in al.binding]
    dom = {e: [p.prim_id for p in free if tc(e, p.prim_id)] for e in U}
    for e in U:
        if not dom[e]:
            al.absent.add(e)
    U = [e for e in U if dom[e]]
    # unique type: an entity with exactly one (parameter-aware) compatible candidate that no other
    # unmatched entity can take — the §10.3 matching is unique; ambiguity needs >= 2 options
    for e in list(U):
        if len(dom[e]) == 1 and sum(dom[e][0] in dom[o] for o in U) == 1:
            al.binding[e] = dom[e][0]
            al.method[e] = "unique_type"
    U = [e for e in U if e not in al.binding]
    if not U:
        return al
    sols, capped = None, False
    if len(U) <= int(th.value("subgraph_max_unmatched")):
        sols, capped = _exact_embeddings(U, dom, facts, al.binding, index)
    if sols and capped:          # too many embeddings to certify any binding as unique
        for e in U:
            al.ambiguous.add(e)
            al.candidates[e] = sorted({s[e] for s in sols})
        return al
    if sols:
        for e in U:
            imgs = sorted({s[e] for s in sols})
            if len(imgs) == 1:
                al.binding[e] = imgs[0]
                al.method[e] = "subgraph"
            else:
                al.ambiguous.add(e)
                al.candidates[e] = imgs
        return al
    _hungarian(U, dom, ents, facts, al, index)
    return al


def _exact_embeddings(U, dom, facts, binding, index) -> tuple[list[dict[str, str]] | None, bool]:
    """Injective embeddings under which every eligible relation among bound ∪ U holds on the
    images. Returns (solutions or None, capped); None when the node budget is exhausted or no
    embedding exists; capped when SOLUTION_CAP embeddings were found (enumeration stopped)."""
    rel = [f for f in facts if PREDICATES[f.predicate].fclass.value == "relation"]
    scope = set(binding) | set(U)
    cons = [f for f in rel if set(entity_args(f.predicate, f.args)) <= scope
            and set(entity_args(f.predicate, f.args)) & set(U)]
    deg = {e: sum(e in entity_args(f.predicate, f.args) for f in cons) for e in U}
    order = sorted(U, key=lambda e: (len(dom[e]), -deg[e], e))
    sols: list[dict[str, str]] = []
    nodes = [0]
    assign: dict[str, str] = {}
    used = set(binding.values())

    def ok_for(e: str) -> bool:
        m_all = {**binding, **assign}
        for f in cons:
            ea = entity_args(f.predicate, f.args)
            if e in ea and all(x in m_all for x in ea):
                m = map_args(f.predicate, f.args, m_all)
                if m is None or not index.holds(f.predicate, m)[0]:
                    return False
        return True

    def rec(i: int) -> bool:
        nodes[0] += 1
        if nodes[0] > NODE_CAP:
            return False
        if i == len(order):
            sols.append(dict(assign))
            return len(sols) < SOLUTION_CAP
        e = order[i]
        for p in dom[e]:
            if p in used:
                continue
            assign[e] = p
            used.add(p)
            if ok_for(e) and not rec(i + 1):
                used.discard(p)
                del assign[e]
                return False
            used.discard(p)
            del assign[e]
        return True

    rec(0)
    if nodes[0] > NODE_CAP:
        return None, False
    return (sols or None), len(sols) >= SOLUTION_CAP


def _signature_card(e, facts, binding, types) -> set:
    sig = set()
    for f in facts:
        if PREDICATES[f.predicate].fclass.value != "relation":
            continue
        ea = entity_args(f.predicate, f.args)
        if e not in ea:
            continue
        others = tuple(sorted(("b", binding[o]) if o in binding else ("t", _type_group(types.get(o, "?")))
                              for o in ea if o != e))
        sig.add((f.predicate, ea.index(e) if PREDICATES[f.predicate].args[0] != "set" else -1, others))
    return sig


def _signature_prim(p, index: CgxIndex, bound_prims: set[str]) -> set:
    sig = set()
    for f in index.closed:
        pred = f[0]
        if pred not in PREDICATES or PREDICATES[pred].fclass.value != "relation":
            continue
        ea = entity_args(pred, [list(a) if isinstance(a, tuple) else a for a in f[1]])
        if p not in ea:
            continue
        others = tuple(sorted(("b", o) if o in bound_prims else ("t", _type_group(index.types.get(o, "?")))
                              for o in ea if o != p))
        sig.add((pred, ea.index(p) if PREDICATES[pred].args[0] != "set" else -1, others))
    return sig


def _hungarian(U, dom, ents, facts, al: Alignment, index: CgxIndex) -> None:
    types = {k: v.type for k, v in ents.items()}
    bound_prims = set(al.binding.values())
    P = sorted({p for e in U for p in dom[e]})
    sim = np.zeros((len(U), len(P)))
    feas = np.zeros_like(sim, dtype=bool)
    psig = {p: _signature_prim(p, index, bound_prims) for p in P}
    for i, e in enumerate(U):
        es = _signature_card(e, facts, al.binding, types)
        for j, p in enumerate(P):
            if p in dom[e]:
                feas[i, j] = True
                un = es | psig[p]
                sim[i, j] = len(es & psig[p]) / len(un) if un else 0.0
    cost = np.where(feas, -sim, 1e6)
    rows, cols = linear_sum_assignment(cost)
    for i, j in zip(rows, cols):
        e = U[i]
        if not feas[i, j] or sim[i, j] <= 0:
            al.ambiguous.add(e)
            al.candidates[e] = list(dom[e])
            continue
        s = sim[i, j]
        tie = any(k != j and feas[i, k] and sim[i, k] >= s - 1e-12 for k in range(len(P))) or \
            any(k != i and feas[k, j] and sim[k, j] >= s - 1e-12 for k in range(len(U)))
        if tie:
            al.ambiguous.add(e)
            al.candidates[e] = [P[k] for k in range(len(P)) if feas[i, k] and sim[i, k] >= s - 1e-12]
        else:
            al.binding[e] = P[j]
            al.method[e] = "hungarian"
    for i, e in enumerate(U):
        if e not in al.binding and e not in al.ambiguous:
            al.ambiguous.add(e)
            al.candidates[e] = list(dom[e])

