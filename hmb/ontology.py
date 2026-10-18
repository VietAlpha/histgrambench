"""Closed predicate table asc-1.0.0 (plan §7.1) plus canonical-form helpers (§8.2).

This module is the single source of truth for predicate metadata. Card building,
extraction, adjudication, grading and the Track C predicate evaluator all import it.

Argument kinds:
  "ent"   - canonical entity id
  "set"   - sorted maximal set of entity ids (collinear, concyclic)
  "seq"   - ordered list of entity ids (order_along)
  "ent?"  - entity id or the literal "unknown" (tangent_to point)
  "text"  - label string as printed
  "enum:<name>" / "int" - parameter from a closed domain
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .vocab import Anchor, Decidability, FactClass, Strictness

# asc-2.0.0 (2026-09-26 streamlining, reports/answer_key_judge_streamlining_execution_20260926.md): the card
# vocabulary is the strict classes E, R, L only. Layout (Y) predicates stay in this table because the CGX
# extractor and the Track C evaluator compute them (E-exclusions, CONTRADICTED), but no card asserts them.
ASC_MAJOR_MINOR_PATCH = "asc-2.0.0"

ENUMS: dict[str, tuple[str, ...]] = {
    "axis_o": ("horizontal", "vertical", "oblique"),
    "curve_f": ("line", "parabola", "hyperbola", "ellipse_arc", "periodic",
                "monotone_other", "unspecified"),
    "solid_k": ("cube", "prism", "pyramid", "cylinder", "cone", "sphere",
                "polyhedron_other"),
    "measure_kind": ("length", "angle", "area", "coordinate", "name"),
    # ASSUMPTION (recorded in BUILD_STATUS contradictions): §7.1 does not give the
    # domain of orientation(x, o); the axis domain is reused.
    "orientation_o": ("horizontal", "vertical", "oblique"),
}

ENTITY_TYPES = ("point", "segment", "line", "ray", "circle", "arc", "ellipse",
                "polygon", "region", "axis", "arrow", "angle_marker",
                "right_angle_marked", "length_tick", "curve", "solid")

# 1-D entities / curves a point may lie on (§7.1 `on` note).
ONE_D_TYPES = frozenset({"segment", "line", "ray", "circle", "arc", "ellipse",
                         "curve", "axis", "polygon"})


@dataclass(frozen=True)
class Pred:
    name: str
    args: tuple[str, ...]
    fclass: FactClass
    strictness: Strictness
    anchor: Anchor
    decidability: Decidability
    symmetric_pairs: tuple[tuple[int, int], ...] = ()   # positions that commute
    note: str = ""

    @property
    def arity(self) -> int:
        return len(self.args)

    @property
    def category(self) -> str:
        return {"entity": "E", "relation": "R", "label": "L", "layout": "Y"}[self.fclass.value]


_E, _R, _L, _Y = FactClass.entity, FactClass.relation, FactClass.label, FactClass.layout
_S, _C = Strictness.strict, Strictness.coarse
_A, _B = Anchor.A, Anchor.B
_G, _M = Decidability.geo, Decidability.model

_TABLE: list[Pred] = [
    # Entities (E, strict, anchor A); geo except region/curve/solid.
    Pred("point", ("ent",), _E, _S, _A, _G),
    Pred("segment", ("ent",), _E, _S, _A, _G),
    Pred("line", ("ent",), _E, _S, _A, _G),
    Pred("ray", ("ent",), _E, _S, _A, _G),
    Pred("circle", ("ent",), _E, _S, _A, _G),
    Pred("arc", ("ent",), _E, _S, _A, _G),
    Pred("ellipse", ("ent",), _E, _S, _A, _G),
    Pred("polygon", ("ent", "int"), _E, _S, _A, _G),
    Pred("region", ("ent",), _E, _S, _A, _M),
    Pred("axis", ("ent", "enum:axis_o"), _E, _S, _A, _G),
    Pred("arrow", ("ent",), _E, _S, _A, _G),
    Pred("angle_marker", ("ent",), _E, _S, _A, _G),
    Pred("right_angle_marked", ("ent",), _E, _S, _A, _G),
    Pred("length_tick", ("ent",), _E, _S, _A, _G),
    Pred("curve", ("ent", "enum:curve_f"), _E, _S, _A, _M),
    Pred("solid", ("ent", "enum:solid_k"), _E, _S, _A, _M),
    # Relations (R).
    Pred("on", ("ent", "ent"), _R, _S, _A, _G, note="point incident to 1-D entity or curve"),
    Pred("endpoint_of", ("ent", "ent"), _R, _S, _A, _G, note="x in {segment, ray, arc}"),
    Pred("vertex_of", ("ent", "ent"), _R, _S, _A, _G, note="x in {polygon, angle_marker, solid}"),
    Pred("edge_of", ("ent", "ent"), _R, _S, _A, _G, note="x in {polygon, solid}"),
    Pred("face_of", ("ent", "ent"), _R, _S, _B, _M),
    Pred("center_of", ("ent", "ent"), _R, _S, _A, _G, note="x in {circle, arc, ellipse}"),
    Pred("arc_of", ("ent", "ent"), _R, _S, _A, _G),
    Pred("intersect_at", ("ent", "ent", "ent"), _R, _S, _A, _G, ((0, 1),), "symmetric in (x, y)"),
    Pred("tangent_to", ("ent", "ent", "ent?"), _R, _S, _A, _G, note="p may be unknown"),
    Pred("parallel", ("ent", "ent"), _R, _S, _A, _G, ((0, 1),), "symmetric and transitive"),
    Pred("perpendicular", ("ent", "ent"), _R, _S, _A, _G, ((0, 1),), "symmetric"),
    Pred("collinear", ("set",), _R, _S, _A, _G, note="set >= 3; one maximal set per class"),
    Pred("concyclic", ("set",), _R, _S, _B, _G, note="set >= 4"),
    Pred("between", ("ent", "ent", "ent"), _R, _S, _A, _G, ((0, 2),), "canonical id(p) < id(r)"),
    Pred("equal_length", ("ent", "ent"), _R, _S, _A, _M, ((0, 1),), "only when tick-marked or stated"),
    Pred("equal_angle", ("ent", "ent"), _R, _S, _A, _M, ((0, 1),), "only when marked or stated"),
    Pred("contains", ("ent", "ent"), _R, _S, _A, _G, note="closed curve or region containment"),
    Pred("cross_section_of", ("ent", "ent"), _R, _S, _B, _M),
    Pred("hidden_edge", ("ent",), _R, _S, _B, _M, note="dashed-line convention"),
    Pred("asymptote_of", ("ent", "ent"), _R, _S, _B, _M),
    # Labels (L, strict). label: geo, strict normalized string match (no semantic-equivalence question).
    Pred("label", ("text", "ent"), _L, _S, _A, _G),
    Pred("measure_label", ("text", "ent", "enum:measure_kind"), _L, _S, _A, _M),
    # Layout (Y, coarse, anchor A, geo).
    Pred("left_of", ("ent", "ent"), _Y, _C, _A, _G, note="antisymmetric, one canonical direction"),
    Pred("above", ("ent", "ent"), _Y, _C, _A, _G, note="antisymmetric, one canonical direction"),
    Pred("inside", ("ent", "ent"), _Y, _C, _A, _G),
    Pred("outside", ("ent", "ent"), _Y, _C, _A, _G),
    Pred("same_side", ("ent", "ent", "ent"), _Y, _C, _A, _G, ((0, 1),)),
    Pred("opposite_sides", ("ent", "ent", "ent"), _Y, _C, _A, _G, ((0, 1),)),
    Pred("order_along", ("ent", "seq"), _Y, _C, _A, _G),
    Pred("orientation", ("ent", "enum:orientation_o"), _Y, _C, _A, _G),
]

PREDICATES: dict[str, Pred] = {p.name: p for p in _TABLE}

# Classes a card may assert (asc-2.0.0): the scored strict classes. Layout is extractor-internal.
CARD_CLASSES = frozenset({FactClass.entity, FactClass.relation, FactClass.label})
CARD_PREDICATES: dict[str, Pred] = {n: p for n, p in PREDICATES.items() if p.fclass in CARD_CLASSES}

# §7.1 typed argument positions (the plan's signatures endpoint_of(p, x) with x ∈ {segment, ray, arc}, …),
# per position: the allowed entity types, or None = any entity. Only positions the plan fixes are typed;
# layout positions are checked by the builder conventions. Γ uses exactly these types (checked over all
# families, 2026-09-26). A builder fact outside them is abstained at parse time (C41): the plan letters
# alone were not shown to builders, and one writer read endpoint_of(x, p) with the segment first.
_POINT = frozenset({"point"})
ARG_TYPES: dict[str, tuple[frozenset[str] | None, ...]] = {
    "on": (_POINT, ONE_D_TYPES),
    "endpoint_of": (_POINT, frozenset({"segment", "ray", "arc"})),
    "vertex_of": (_POINT, frozenset({"polygon", "angle_marker", "solid"})),
    "edge_of": (frozenset({"segment"}), frozenset({"polygon", "solid"})),
    "center_of": (_POINT, frozenset({"circle", "arc", "ellipse"})),
    "arc_of": (frozenset({"arc"}), None),
    "intersect_at": (None, None, _POINT),
    "tangent_to": (None, None, _POINT),
    "between": (_POINT, _POINT, _POINT),
    "collinear": (_POINT,),
    "concyclic": (_POINT,),
}


def arg_signature(pred: str) -> str:
    """Argument column of the builder predicate table: one entry per position, in order; a typed entity
    position lists its allowed types (`point`, `segment|ray|arc`), an untyped one stays `ent`."""
    p = PREDICATES[pred]
    dom = ARG_TYPES.get(pred, ())
    out = []
    for i, kind in enumerate(p.args):
        d = dom[i] if i < len(dom) else None
        if d is None:
            out.append(kind)
        else:
            t = "|".join(sorted(d))
            out.append(f"set of {t}" if kind == "set" else (f"{t} or \"unknown\"" if kind == "ent?" else t))
    return ", ".join(out)


def arg_type_violation(pred: str, args: list[Any], types: dict[str, str]) -> str | None:
    """None when every typed argument position holds an entity of an allowed type (§7.1)."""
    for i, d in enumerate(ARG_TYPES.get(pred, ())):
        if d is None or i >= len(args):
            continue
        ids = args[i] if isinstance(args[i], (list, tuple)) else [args[i]]
        for a in ids:
            if a == "unknown" and PREDICATES[pred].args[i] == "ent?":
                continue
            if types.get(a) not in d:
                return (f"{pred} argument {i + 1} must be {'|'.join(sorted(d))}, "
                        f"got {types.get(a)} (signature {pred}({arg_signature(pred)}))")
    return None

# §8.2 key(f) uses predicate_rank[f.pred]; the plan does not define the ranking.
# FORM CHOICE (recorded): rank = position in the §7.1 table. It only decides which
# member of a mutually-entailing set (e.g. a parallel triangle) is dropped from the MGS.
PREDICATE_RANK: dict[str, int] = {p.name: i for i, p in enumerate(_TABLE)}

# Derived, never primitive, never separately scored (§7.1). Builders may not emit them.
DERIVED_ONLY = frozenset({"radius_of", "diameter_of", "chord_of", "secant_to"})

# Documented discrepancy: §7.1 says "Total: 47 predicates" but enumerates 46.
PLAN_STATED_TOTAL = 47
assert len(PREDICATES) == 46, len(PREDICATES)

# §7.6 allowed_variation protocol constant. Reflection is not in it; since asc-2.0.0 it is also not scored.
ALLOWED_VARIATION = ("translate", "uniform_scale", "line_width", "color", "font_family",
                     "label_offset", "canvas_pad", "dpi", "source_refactor")


class OntologyError(ValueError):
    pass


def validate_fact(pred: str, args: list[Any]) -> None:
    """Raise OntologyError unless (pred, args) is well-formed under asc-1.0.0."""
    p = PREDICATES.get(pred)
    if p is None:
        raise OntologyError(f"out-of-vocabulary predicate {pred!r}")
    if len(args) != p.arity:
        raise OntologyError(f"{pred}: arity {p.arity}, got {len(args)}")
    for kind, a in zip(p.args, args):
        if kind in ("ent",):
            if not isinstance(a, str) or not a or a == "unknown":
                raise OntologyError(f"{pred}: entity id expected, got {a!r}")
        elif kind == "ent?":
            if not isinstance(a, str) or not a:
                raise OntologyError(f"{pred}: entity id or 'unknown' expected, got {a!r}")
        elif kind in ("set", "seq"):
            minimum = {"collinear": 3, "concyclic": 4}.get(pred, 2)
            if (not isinstance(a, (list, tuple)) or len(a) < minimum
                    or not all(isinstance(x, str) and x for x in a)
                    or len(set(a)) != len(a)):
                raise OntologyError(f"{pred}: {kind} of >= {minimum} distinct ids expected, got {a!r}")
        elif kind == "text":
            if not isinstance(a, str) or not a.strip():
                raise OntologyError(f"{pred}: non-empty label text expected")
        elif kind == "int":
            if not isinstance(a, int) or isinstance(a, bool) or a < 3:
                raise OntologyError(f"{pred}: integer >= 3 expected, got {a!r}")
        elif kind.startswith("enum:"):
            dom = ENUMS[kind[5:]]
            if a not in dom:
                raise OntologyError(f"{pred}: {a!r} not in {dom}")


def canonical_args(pred: str, args: list[Any]) -> tuple:
    """§8.2 canonical form: symmetric pairs in id order, sets sorted, between id(p)<id(r)."""
    validate_fact(pred, args)
    p = PREDICATES[pred]
    out: list[Any] = [tuple(sorted(a)) if k == "set" else (tuple(a) if k == "seq" else a)
                      for k, a in zip(p.args, args)]
    for i, j in p.symmetric_pairs:
        if str(out[i]) > str(out[j]):
            out[i], out[j] = out[j], out[i]
    if pred == "order_along" and out[1][0] > out[1][-1]:
        # CONTRACTS "Geometric predicate semantics": direction canonicalized so seq[0] < seq[-1].
        out[1] = tuple(reversed(out[1]))
    return tuple(out)


def fact_key(pred: str, args: list[Any]) -> tuple:
    """Hashable canonical identity of a fact."""
    return (pred, canonical_args(pred, args))


def mgs_sort_key(pred: str, args: list[Any]) -> tuple:
    """§8.2 key(f) = (predicate_rank, -arity, sorted(args)). Sorted descending by caller.

    The §8.2 key ties for distinct facts with the same argument multiset (endpoint_of(B, s) vs
    endpoint_of(s, B)); a tie then fell back to set iteration order, which varies per process (string
    hash seed), so the MGS, fact ids and verification order changed between runs (C40, 2026-09-26). The
    ordered canonical arguments are the final tie-break, which makes the key total over distinct facts."""
    ca = canonical_args(pred, args)
    flat: list[str] = []
    for a in ca:
        flat.extend(a if isinstance(a, tuple) else [str(a)])
    ar = sum(len(a) if k in ("set", "seq") else 1 for k, a in zip(PREDICATES[pred].args, ca))
    return (PREDICATE_RANK[pred], -ar, tuple(sorted(flat)), repr(ca))


def entity_args(pred: str, args: list[Any]) -> list[str]:
    """Entity ids referenced by a fact (excludes text/enum/int params and 'unknown')."""
    p = PREDICATES[pred]
    out: list[str] = []
    for k, a in zip(p.args, args):
        if k == "ent":
            out.append(a)
        elif k == "ent?" and a != "unknown":
            out.append(a)
        elif k in ("set", "seq"):
            out.extend(a)
    return out
