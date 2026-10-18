"""§8.2 entailment rule set R, exclusion table E, and the canonical minimal generating set (MGS).

Shared contract (docs/CONTRACTS.md): the card builder, the CGX extractor and the Track C oracle graph
all canonicalize with these functions.

    FactKey = ontology.fact_key(pred, args)       # (pred, canonical args tuple)
    forward_chain(facts, entity_types)  -> set[FactKey]
    contradictions(closed, entity_types) -> list[tuple[FactKey, FactKey]]
    minimal_generating_set(facts, entity_types) -> list[FactKey]

Rule set R (§8.2), range-restricted (every derived argument occurs in a premise), so the fixpoint
terminates:

    R1  endpoint_of(P,X)                         -> on(P,X)
    R2  vertex_of(P,G)                           -> on(P,G)
    R3  edge_of(S,G) & endpoint_of(P,S)          -> vertex_of(P,G)
    R4  center_of(P,C) & circle(C)               -> inside(P,C)
    R5  tangent_to(L,C,P) & P != unknown         -> on(P,L) & on(P,C)
    R6  intersect_at(X,Y,P)                      -> on(P,X) & on(P,Y)
    R7  between(P,Q,R)                           -> collinear({P,Q,R})
    R8  collinear(S) & S' < S & |S'| >= 3        -> collinear(S')   (only the maximal set is kept)
    R9  parallel(X,Y) & parallel(Y,Z)            -> parallel(X,Z)   (the MGS keeps a spanning tree)
    R10 perpendicular(X,Y) & parallel(Y,Z)       -> perpendicular(X,Z)
    R11 center_of(O,C) & endpoint_of(O,S) & endpoint_of(P,S) & on(P,C) -> radius_of(S,C)  (derived only)

Implementation notes (form choices, recorded):
  * `circle(C)` in R4 holds if the fact `circle(C)` is present or `entity_types[C] == "circle"`.
  * R8 is realised as subsumption: after the fixpoint every collinear set that is a strict subset of
    another collinear set in the closure is removed, so the closure carries maximal sets only
    ("keep only the maximal set"). Use `entails()` to test a sub-tuple against a closure.
  * R9/R10 never derive a reflexive fact (X == Z); R7/R11 require distinct entities.
  * Derived-only predicates (`ontology.DERIVED_ONLY`, e.g. `radius_of`) are in the closure but never
    in the MGS, and are not validated by the ontology (they are not builder predicates).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from hmb.ontology import DERIVED_ONLY, PREDICATES, fact_key, mgs_sort_key

from .normalize import normalize_label

FactKey = tuple

RULESET_VERSION = "asc-1.0.0/R-1/E-1"

RULES_TEXT = (
    "R1 endpoint_of(P,X) -> on(P,X)",
    "R2 vertex_of(P,G) -> on(P,G)",
    "R3 edge_of(S,G) & endpoint_of(P,S) -> vertex_of(P,G)",
    "R4 center_of(P,C) & circle(C) -> inside(P,C)",
    "R5 tangent_to(L,C,P) & P != unknown -> on(P,L) & on(P,C)",
    "R6 intersect_at(X,Y,P) -> on(P,X) & on(P,Y)",
    "R7 between(P,Q,R) -> collinear({P,Q,R})",
    "R8 collinear(S) & S' subset S & |S'|>=3 -> collinear(S') [maximal set kept]",
    "R9 parallel(X,Y) & parallel(Y,Z) -> parallel(X,Z) [spanning tree kept]",
    "R10 perpendicular(X,Y) & parallel(Y,Z) -> perpendicular(X,Z)",
    "R11 center_of(O,C) & endpoint_of(O,S) & endpoint_of(P,S) & on(P,C) -> radius_of(S,C) [derived only]",
)
EXCLUSIONS_TEXT = (
    "E1 inside(P,X) xor on(P,X) xor outside(P,X)",
    "E2 parallel(X,Y) xor intersect_at(X,Y,_)",
    "E3 tangent_to(L,C,_) xor two distinct intersect_at(L,C,.)",
    "E4 left_of(A,B) xor left_of(B,A)",
    "E5 above(A,B) xor above(B,A)",
    "E6 same_side(P,Q,L) xor opposite_sides(P,Q,L)",
    "E7 two different normalized labels on one entity",
)


def canon(f) -> FactKey:
    """Canonical FactKey from (pred, args) (list or tuple args accepted)."""
    pred, args = f[0], f[1]
    if pred in DERIVED_ONLY:
        return (pred, tuple(args))
    return fact_key(pred, list(args))    # ontology canonicalizes order_along direction too


def _k(pred: str, *args) -> FactKey:
    return fact_key(pred, list(args))


def _order(f: FactKey) -> tuple:
    return (f[0], repr(f[1]))


def _is_circle(c: str, present: set, types: dict[str, str]) -> bool:
    return types.get(c) == "circle" or ("circle", (c,)) in present


def _derive(S: set, types: dict[str, str]) -> dict[FactKey, tuple[FactKey, ...]]:
    """One naive round: every fact derivable from S by one rule application -> its premises."""
    out: dict[FactKey, tuple[FactKey, ...]] = {}
    by: dict[str, list[FactKey]] = defaultdict(list)
    for f in S:
        by[f[0]].append(f)
    for p in by:
        by[p].sort(key=_order)   # determinism of recorded provenance

    def add(f: FactKey, prem: tuple[FactKey, ...]) -> None:
        if f not in S and f not in out:
            out[f] = prem

    endpoints_of: dict[str, list[FactKey]] = defaultdict(list)   # segment -> endpoint facts
    for f in by["endpoint_of"]:
        p, x = f[1]
        add(_k("on", p, x), (f,))                                   # R1
        endpoints_of[x].append(f)
    for f in by["vertex_of"]:
        p, g = f[1]
        add(_k("on", p, g), (f,))                                   # R2
    for f in by["edge_of"]:
        s, g = f[1]
        for e in endpoints_of.get(s, ()):
            add(_k("vertex_of", e[1][0], g), (f, e))                 # R3
    for f in by["center_of"]:
        p, c = f[1]
        if _is_circle(c, S, types):
            circ = ("circle", (c,))
            add(_k("inside", p, c), (f, circ) if circ in S else (f,))  # R4
    for f in by["tangent_to"]:
        l_, c, p = f[1]
        if p != "unknown":
            add(_k("on", p, l_), (f,))                                # R5
            add(_k("on", p, c), (f,))
    for f in by["intersect_at"]:
        x, y, p = f[1]
        add(_k("on", p, x), (f,))                                     # R6
        add(_k("on", p, y), (f,))
    for f in by["between"]:
        p, q, r = f[1]
        if len({p, q, r}) == 3:
            add(_k("collinear", [p, q, r]), (f,))                     # R7
    par: dict[str, list[tuple[str, FactKey]]] = defaultdict(list)
    for f in by["parallel"]:
        x, y = f[1]
        if x != y:
            par[x].append((y, f))
            par[y].append((x, f))
    for f in by["parallel"]:                                          # R9 (symmetric storage)
        x, y = f[1]
        for a, b in ((x, y), (y, x)):
            for z, g in par.get(b, ()):
                if z != a:
                    add(_k("parallel", a, z), (f, g))
    for f in by["perpendicular"]:                                     # R10 (symmetric storage)
        x, y = f[1]
        for a, b in ((x, y), (y, x)):
            for z, g in par.get(b, ()):
                if z != a:
                    add(_k("perpendicular", a, z), (f, g))
    on_set = set(by["on"])
    for f in by["center_of"]:                                         # R11
        o, c = f[1]
        for e1 in by["endpoint_of"]:
            if e1[1][0] != o:
                continue
            s = e1[1][1]
            for e2 in endpoints_of.get(s, ()):
                p = e2[1][0]
                if p != o and ("on", (p, c)) in on_set:
                    add(("radius_of", (s, c)), (f, e1, e2, ("on", (p, c))))
    return out


def _subsumed_collinear(S: set) -> set:
    cols = [f for f in S if f[0] == "collinear"]
    sets = [(f, frozenset(f[1][0])) for f in cols]
    return {f for f, s in sets if any(s < t for _, t in sets)}


def forward_chain_with_provenance(facts: Iterable, entity_types: dict[str, str]
                                  ) -> dict[FactKey, tuple[FactKey, ...] | None]:
    """Closure under R. Value is None for given facts, else the premises of the first derivation."""
    prov: dict[FactKey, tuple[FactKey, ...] | None] = {canon(f): None for f in facts}
    types = dict(entity_types)
    S = set(prov)
    while True:
        new = _derive(S, types)
        if not new:
            break
        for f in sorted(new, key=_order):
            prov[f] = new[f]
        S |= set(new)
    for f in _subsumed_collinear(S):          # R8: keep only maximal collinear sets
        del prov[f]
    return prov


def forward_chain(facts: Iterable, entity_types: dict[str, str]) -> set[FactKey]:
    """§8.2 C = forward_chain(F, R)."""
    return set(forward_chain_with_provenance(facts, entity_types))


def entails(closed: set[FactKey], f) -> bool:
    """Membership test against a closure, honouring R8 (sub-tuples of a maximal collinear set)."""
    f = canon(f)
    if f in closed:
        return True
    if f[0] == "collinear":
        s = frozenset(f[1][0])
        return any(g[0] == "collinear" and s <= frozenset(g[1][0]) for g in closed)
    return False


def contradictions(closed: set[FactKey], entity_types: dict[str, str]) -> list[tuple[FactKey, FactKey]]:
    """§8.2 exclusion table E over a closed fact set. Each pair (a, b) has a before b (deterministic)."""
    pairs: set[tuple[FactKey, FactKey]] = set()

    def pair(a: FactKey, b: FactKey) -> None:
        if a != b:
            pairs.add((a, b) if _order(a) <= _order(b) else (b, a))

    by: dict[str, list[FactKey]] = defaultdict(list)
    for f in closed:
        by[f[0]].append(f)
    # E1 inside xor on xor outside, per (P, X)
    ioo: dict[tuple, list[FactKey]] = defaultdict(list)
    for p in ("inside", "on", "outside"):
        for f in by[p]:
            ioo[f[1]].append(f)
    for fs in ioo.values():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                pair(fs[i], fs[j])
    # E2 parallel xor intersect_at (intersect_at canonical (x, y) sorted like parallel)
    inter: dict[tuple, list[FactKey]] = defaultdict(list)
    for f in by["intersect_at"]:
        inter[(f[1][0], f[1][1])].append(f)
    for f in by["parallel"]:
        for g in inter.get(f[1], ()):
            pair(f, g)
    # E3 tangent_to(L,C,_) xor two distinct intersect_at(L,C,.) (pair unordered, as in intersect_at)
    for f in by["tangent_to"]:
        l_, c = f[1][0], f[1][1]
        key = (l_, c) if str(l_) <= str(c) else (c, l_)
        pts = {g[1][2] for g in inter.get(key, ())}
        if len(pts) >= 2:
            for g in inter[key]:
                pair(f, g)
    # E4 / E5 antisymmetric layout
    for p in ("left_of", "above"):
        present = set(by[p])
        for f in by[p]:
            a, b = f[1]
            rev = (p, (b, a))
            if rev in present:
                pair(f, rev)
    # E6 same_side xor opposite_sides (same canonical arg order)
    opp = set(by["opposite_sides"])
    for f in by["same_side"]:
        g = ("opposite_sides", f[1])
        if g in opp:
            pair(f, g)
    # E7 two different normalized labels on one entity
    labels: dict[str, list[FactKey]] = defaultdict(list)
    for f in by["label"]:
        labels[f[1][1]].append(f)
    for fs in labels.values():
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                if normalize_label(fs[i][1][0]) != normalize_label(fs[j][1][0]):
                    pair(fs[i], fs[j])
    return sorted(pairs, key=lambda ab: (_order(ab[0]), _order(ab[1])))


def support(f: FactKey, prov: dict[FactKey, tuple[FactKey, ...] | None]) -> frozenset[FactKey]:
    """Leaf premises (given facts) of the recorded derivation of `f`."""
    out: set[FactKey] = set()
    stack, seen = [f], set()
    while stack:
        g = stack.pop()
        if g in seen:
            continue
        seen.add(g)
        pr = prov.get(g)
        if pr is None:
            if g in prov:
                out.add(g)
            continue
        stack.extend(pr)
    return frozenset(out)


def contradiction_participants(facts: Iterable, entity_types: dict[str, str]
                               ) -> tuple[set[FactKey], list[tuple[FactKey, FactKey]]]:
    """Given facts (e.g. an MGS) that participate in an E-violation of their closure.

    A contradicting closure fact that is itself given counts as itself; a derived one is attributed
    to the given facts its recorded derivation rests on. Returns (participants, closure pairs).
    """
    prov = forward_chain_with_provenance(facts, entity_types)
    pairs = contradictions(set(prov), entity_types)
    part: set[FactKey] = set()
    for a, b in pairs:
        for side in (a, b):
            part |= support(side, prov)
    return part, pairs


def _sort_key(f: FactKey) -> tuple:
    return mgs_sort_key(f[0], list(f[1]))


def minimal_generating_set(facts: Iterable, entity_types: dict[str, str]) -> list[FactKey]:
    """§8.2: C = closure; M = C; for f in sort(C, key, descending): drop f if f in closure(M - {f}).

    Derived-only facts are never in M. Returned in ascending key order (deterministic).
    """
    C = forward_chain(facts, entity_types)
    M = {f for f in C if f[0] not in DERIVED_ONLY}
    for f in sorted(M, key=_sort_key, reverse=True):
        rest = M - {f}
        if entails(forward_chain(rest, entity_types), f):   # R8-aware membership (red-team fix #9)
            M = rest
    return sorted(M, key=_sort_key)


def closure_spec() -> dict:
    """Hashable description of R and E (enters asc_version, §8.8)."""
    return {"version": RULESET_VERSION, "rules": list(RULES_TEXT), "exclusions": list(EXCLUSIONS_TEXT),
            "derived_only": sorted(DERIVED_ONLY), "predicates": sorted(PREDICATES)}
