"""Deterministic geo verdicts from the CGX record (§10.2, §10.4) and fact routing (§10.1).

A card fact whose arguments are all bound to candidate primitives is decided on those primitives:
  1. by the shared pure predicate function used by the CGX extractor and the oracle evaluator
     (CONTRACTS.md "Geometric predicate semantics"), when it is registered via
     `hmb.synthetic.geo_predicates.holds_geo` over `hmb.cgx.cgx_geometry(cgx)` with the §10.2
     tolerances, when the index is built with thresholds (production path), else
  2. by entailment against the closure of the CGX facts (`hmb.cards.closure`), honouring maximal
     collinear sets, parallel classes and `tangent_to(·,·,unknown)`.
The basis used is recorded in each verdict's evidence string.
CONTRADICTED = the fact does not hold and adding it (with its consequences under R) to the render's
closure creates an E-violation against a fact the render asserts (exclusion table E, §8.2).
"""
from __future__ import annotations

from typing import Any, Callable

from hmb.cards.closure import canon, contradictions, entails, forward_chain
from hmb.ontology import PREDICATES, OntologyError

from .geom import CgxGeometry, ontology_type

def _shared_for(cgx, th) -> Callable[[str, list], bool | None] | None:
    """The ONE geometric predicate implementation shared with Γ and CGX (CONTRACTS.md, C8)."""
    try:
        from hmb.cgx.extract import cgx_geometry
        from hmb.synthetic.geo_predicates import GEO_PREDICATES, holds_geo, tolerances_10_2
        geom = cgx_geometry(cgx)
        tol = tolerances_10_2(th, float(cgx.diagonal_D))
    except (ImportError, KeyError, TypeError, ValueError):
        return None

    def fn(pred: str, args: list) -> bool | None:
        if pred not in GEO_PREDICATES or pred == "label":
            return None
        try:
            return bool(holds_geo(geom, pred, list(args), tol)[0])
        except (KeyError, TypeError, ValueError, IndexError):
            return None                 # geometry this evaluator cannot read -> closure fallback
    return fn


# ------------------------------------------------------------------ §10.1 routing
def route(fact) -> str:
    """'extractor' for a `geo` fact, 'judges' for a `model` fact (the full extractor is always used;
    its correctness is covered by the offline `cgx-roundtrip` check)."""
    return "judges" if fact.decidability == "model" else "extractor"


# ------------------------------------------------------------------ mapping
def map_args(pred: str, args: list[Any], binding: dict[str, str]) -> list[Any] | None:
    """Substitute bound primitive ids; None if some entity argument is unbound."""
    out: list[Any] = []
    for kind, a in zip(PREDICATES[pred].args, args):
        if kind == "ent" or (kind == "ent?" and a != "unknown"):
            if a not in binding:
                return None
            out.append(binding[a])
        elif kind in ("set", "seq"):
            if any(x not in binding for x in a):
                return None
            out.append([binding[x] for x in a])
        else:
            out.append(a)
    return out


class CgxIndex:
    """Closure of the candidate's asserted facts plus lookups used by alignment and verdicts."""

    def __init__(self, cgx, th=None):
        self.cgx = cgx
        self._shared = _shared_for(cgx, th) if th is not None else None
        self.prims = {p.prim_id: p for p in cgx.primitives}
        self.types = {p.prim_id: ontology_type(p.type) for p in cgx.primitives}
        raw, self.margin, self.unscored = [], {}, []
        for f in cgx.facts:
            try:
                k = canon((f.predicate, list(f.args)))
            except (OntologyError, KeyError, TypeError):
                self.unscored.append(f.cfact_id)
                continue
            raw.append(k)
            if f.margin is not None:
                self.margin[k] = max(self.margin.get(k, float("-inf")), float(f.margin))
        self.raw = set(raw)
        self._geometry: CgxGeometry | None = None
        self.closed = forward_chain(raw, self.types) if raw else set()
        self._par = self._parallel_classes()

    def _parallel_classes(self) -> dict[str, str]:
        parent: dict[str, str] = {}

        def find(x: str) -> str:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for f in self.closed:
            if f[0] == "parallel":
                a, b = find(f[1][0]), find(f[1][1])
                if a != b:
                    parent[max(a, b)] = min(a, b)
        return {x: find(x) for x in list(parent)}

    @property
    def geometry(self) -> CgxGeometry:
        """Shared-format geometry (hmb.cgx.cgx_geometry -> geo_predicates.to_internal)."""
        if self._geometry is None:
            self._geometry = CgxGeometry(self.cgx)
        return self._geometry

    def _cls(self, x: str) -> str:
        return self._par.get(x, x)

    def holds(self, pred: str, args: list[Any]) -> tuple[bool, str]:
        """Truth of a fact over candidate primitives, with its basis. Memoized per index (EC2 MS5 2026-10-01):
        the answer depends only on state fixed at construction, and alignment's structural propagation asks the
        same questions again on every pass."""
        memo = self.__dict__.setdefault("_holds_memo", {})
        k = (pred, repr(args))
        if k not in memo:
            memo[k] = self._holds(pred, args)
        return memo[k]

    def _holds(self, pred: str, args: list[Any]) -> tuple[bool, str]:
        if self._shared is not None:
            r = self._shared(pred, args)
            if r is not None:
                return bool(r), "shared_predicate"
        fb = "cgx_closure"
        key = canon((pred, args))
        if entails(self.closed, key):
            return True, fb
        if pred == "parallel":
            a, b = key[1]
            return (a != b and self._cls(a) == self._cls(b)), fb
        if pred == "perpendicular":
            a, b = key[1]
            ca, cb = self._cls(a), self._cls(b)
            for f in self.closed:
                if f[0] == "perpendicular":
                    x, y = self._cls(f[1][0]), self._cls(f[1][1])
                    if {x, y} == {ca, cb}:
                        return True, fb
            return False, fb
        if pred == "tangent_to" and key[1][2] == "unknown":
            x, y = key[1][0], key[1][1]
            return any(f[0] == "tangent_to" and f[1][0] == x and f[1][1] == y for f in self.closed), fb
        return False, fb

    def e_incompatible(self, pred: str, args: list[Any]) -> bool:
        """True if asserting the fact would violate table E against the render's asserted facts."""
        key = canon((pred, args))
        ext = forward_chain(set(self.closed) | {key}, self.types)
        new = ext - self.closed
        for a, b in contradictions(ext, self.types):
            if (a in new) != (b in new):
                return True
        return False

    def direct_margin(self, pred: str, args: list[Any]) -> float | None:
        return self.margin.get(canon((pred, args)))
