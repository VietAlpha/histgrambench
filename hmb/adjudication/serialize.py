"""Judge-readable serialization of card facts and entities (§10.3 inputs; repair 2026-09-26, defect B).

This module is the ONLY place that turns card entities and facts into model-visible judge text:
the Q1 statements, the ENTITIES block that defines every identifier used in them, the Q3 entity
list and questions, and the Q3 candidate-object lines. Its source hash and `SERIALIZER_VERSION` are
part of the CP1 form freeze and of `judge_config_hash`; see docs/CONTRACTS.md, "Judge serialization".

Identity rule (deterministic, per item):
  * An entity whose (type, printed label) is unique in the card is named by its label: point “A”.
  * Every other entity (unlabelled, or sharing its label with another entity of its type) gets an
    opaque identifier: a type prefix plus a number, e.g. P2, S3, L1, C1. Numbers are assigned per type
    by sorting on a canonical signature: the labels of the uniquely-labelled entities it is incident
    to (geo incidence facts only), its own printed label, its rounded card anchor, and finally its
    canonical entity id. Nothing depends on Python object identity, dict iteration, file paths,
    model output order or the contestant's drawing.
  * The same card gives the same names in Q1 and Q3, on every judge and rerun.
The ENTITIES block defines an identifier with permitted, verdict-neutral information only: its type,
its printed label, and the eligible GEO incidence facts (endpoints, vertices, centre, edges, arcs,
points on it) that are NOT among the statements of the same call; failing that, its left-to-right
rank among the same-type opaque entities by card anchor. It never states a statement under test, a
hidden-positive fact, a verdict, a model-decidable fact, or a weight.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from hmb.ontology import PREDICATES, entity_args

SERIALIZER_VERSION = "judge-serializer-3"      # 1 = the pre-repair "an unlabelled <type>" describer; 3 = no Q2, no evidence (2026-09-26)

PREFIX: dict[str, str] = {
    "point": "P", "segment": "S", "line": "L", "ray": "R", "circle": "C", "arc": "AR", "ellipse": "EL",
    "polygon": "PG", "region": "RG", "axis": "AX", "arrow": "AW", "angle_marker": "AM",
    "right_angle_marked": "RA", "length_tick": "LT", "curve": "CV", "solid": "SD",
}
# Geo incidence predicates that may define an identifier: (predicate, index of the defined entity,
# index of the other entity, English phrase from the defined entity's side).
DEFINING: tuple[tuple[str, int, int, str], ...] = (
    ("endpoint_of", 1, 0, "endpoint"), ("vertex_of", 1, 0, "vertex"), ("center_of", 1, 0, "centre"),
    ("edge_of", 1, 0, "edge"), ("arc_of", 1, 0, "arc"), ("on", 1, 0, "point on it"),
    ("endpoint_of", 0, 1, "endpoint of"), ("vertex_of", 0, 1, "vertex of"), ("center_of", 0, 1, "centre of"),
    ("edge_of", 0, 1, "edge of"), ("arc_of", 0, 1, "arc of"), ("on", 0, 1, "lies on"),
)


class StatementCollision(ValueError):
    """Two different facts would be shown to a judge with identical wording."""


# ------------------------------------------------------------------ statement templates
TEMPLATES: dict[str, str] = {
    "on": "{0} lies on {1}.",
    "endpoint_of": "{0} is an endpoint of {1}.",
    "vertex_of": "{0} is a vertex of {1}.",
    "edge_of": "{0} is an edge of {1}.",
    "face_of": "{0} is a face of {1}.",
    "center_of": "{0} is the centre of {1}.",
    "arc_of": "{0} is an arc of {1}.",
    "intersect_at": "{0} and {1} cross at {2}.",
    "tangent_to": "{0} is tangent to {1} at {2}.",
    "parallel": "{0} is parallel to {1}.",
    "perpendicular": "{0} is perpendicular to {1}.",
    "collinear": "{0} lie on one straight line.",
    "concyclic": "{0} lie on one circle.",
    "between": "{1} lies between {0} and {2}.",
    "equal_length": "{0} and {1} are marked or stated as equal in length.",
    "equal_angle": "{0} and {1} are marked or stated as equal angles.",
    "contains": "{0} contains {1}.",
    "cross_section_of": "{0} is a cross-section of {1}.",
    "hidden_edge": "{0} is drawn as a hidden (dashed) edge.",
    "asymptote_of": "{0} is an asymptote of {1}.",
    "label": "{1} is labelled “{0}”.",
    "measure_label": "{1} carries the {2} label “{0}”.",
    "left_of": "{0} is to the left of {1}.",
    "above": "{0} is above {1}.",
    "inside": "{0} lies inside {1}.",
    "outside": "{0} lies outside {1}.",
    "same_side": "{0} and {1} lie on the same side of {2}.",
    "opposite_sides": "{0} and {1} lie on opposite sides of {2}.",
    "order_along": "Along {0}, the points {1} appear in this order.",
    "orientation": "{0} is {1}.",
    "polygon": "The drawing contains {0}, a polygon with {1} vertices.",
    "axis": "The drawing contains {0}, a {1} axis.",
    "curve": "The drawing contains {0}, a curve of kind “{1}”.",
    "solid": "The drawing contains {0}, a solid of kind “{1}”.",
}


def verbalize(pred: str, args: Sequence[Any], describe: Callable[[str], str]) -> str:
    """Self-contained English statement of a fact; entity ids are replaced by ``describe(id)``."""
    p = PREDICATES[pred]
    parts: list[str] = []
    for kind, a in zip(p.args, args):
        if kind == "ent" or (kind == "ent?" and a != "unknown"):
            parts.append(describe(a))
        elif kind == "ent?":
            parts.append("some point")
        elif kind in ("set", "seq"):
            parts.append(", ".join(describe(x) for x in a))
        else:
            parts.append(str(a).replace("_", " ") if kind.startswith("enum") else str(a))
    if pred == "tangent_to" and args[2] == "unknown":
        return f"{parts[0]} is tangent to {parts[1]}."
    if p.arity == 1 and p.fclass.value == "entity":
        return f"The drawing contains {parts[0]}."
    return TEMPLATES[pred].format(*parts)


# ------------------------------------------------------------------ entity identity
def _norm(label: str) -> str:
    from hmb.cards.normalize import normalize_label
    return normalize_label(label)


def _card_labels(entities: Iterable[Any], facts: Iterable[Any]) -> dict[str, str]:
    """First printed label per entity: the entity's own label, else its first eligible label fact in
    fact_id order (the same source as alignment.entity_labels)."""
    out: dict[str, str] = {e.entity_id: e.label for e in entities if e.label}
    for f in sorted(facts, key=lambda f: f.fact_id):
        if f.predicate == "label" and f.args[1] not in out:
            out[f.args[1]] = f.args[0]
    return out


class EntityNames:
    """Deterministic identifiers for every entity of one card (see the module docstring)."""

    def __init__(self, entities: Sequence[Any], facts: Sequence[Any]):
        self.types: dict[str, str] = {e.entity_id: e.type for e in entities}
        self.anchor: dict[str, tuple[float, float] | None] = {
            e.entity_id: ((round(float(e.anchor.x), 3), round(float(e.anchor.y), 3)) if e.anchor else None)
            for e in entities}
        self.facts = sorted((f for f in facts if getattr(f, "eligible", True)), key=lambda f: f.fact_id)
        self.label = _card_labels(entities, self.facts)
        count: dict[tuple[str, str], int] = {}
        for e, lab in self.label.items():
            k = (self.types.get(e, "entity"), _norm(lab))
            count[k] = count.get(k, 0) + 1
        self.named = {e for e, lab in self.label.items() if count[(self.types.get(e, "entity"), _norm(lab))] == 1}
        self.dup_count = {e: count[(self.types.get(e, "entity"), _norm(lab))] for e, lab in self.label.items()
                          if e not in self.named}
        self.handle: dict[str, str] = {e: f"“{self.label[e]}”" for e in self.named}
        by_type: dict[str, list[str]] = {}
        for e in self.types:
            if e not in self.named:
                by_type.setdefault(self.types[e], []).append(e)
        for t, es in by_type.items():
            pre = PREFIX.get(t, "E")
            for n, e in enumerate(sorted(es, key=self._signature), 1):
                self.handle[e] = f"{pre}{n}"
        # judges always read an entity as "<type> <handle>" (`describe`), so identity is (type, handle): a point
        # and a line both printed "V" are distinct (measured on a real crop, 2026-09-26: the type-free check
        # aborted the whole cell)
        self.by_handle = {(self.types.get(e, "entity"), h): e for e, h in self.handle.items()}
        if len(self.by_handle) != len(self.handle):
            raise StatementCollision("two entities received the same identifier")

    def _incidence(self, e: str, exclude: frozenset[str] = frozenset()) -> list[tuple[str, str]]:
        """(phrase, other entity id) for every eligible geo incidence fact defining e, minus `exclude`."""
        out = []
        for f in self.facts:
            if f.fact_id in exclude or PREDICATES[f.predicate].decidability.value != "geo":
                continue
            for pred, me, other, phrase in DEFINING:
                if f.predicate == pred and len(f.args) > max(me, other) and f.args[me] == e:
                    out.append((phrase, f.args[other]))
        return out

    def _signature(self, e: str) -> tuple:
        inc = sorted((ph, self.types.get(o, ""), _norm(self.label[o])) for ph, o in self._incidence(e)
                     if o in self.named)
        a = self.anchor.get(e)
        return (tuple(inc), _norm(self.label.get(e, "")), a is None, a or (0.0, 0.0), e)

    def _colors(self) -> dict[str, str]:
        """Colour refinement (1-WL) over the geo incidence graph, seeded with type, unique label and
        anchor: two entities keep one colour only if no chain of incidences to distinguishable entities
        separates them."""
        if getattr(self, "_wl", None) is None:
            col = {e: repr((self.types[e], self.label[e] if e in self.named else None, self.anchor.get(e)))
                   for e in self.types}
            for _ in range(len(col)):
                new = {e: hashlib.sha256(repr((col[e], sorted((ph, col.get(o, "?")) for ph, o in self._incidence(e))))
                                         .encode()).hexdigest() for e in col}
                if len(set(new.values())) == len(set(col.values())):
                    break
                col = new
            self._wl = col
        return self._wl

    def twins(self, e: str) -> list[str]:
        """Other opaque entities of e's type that the reference cannot tell apart from e (same colour
        after refinement; e.g. the two symmetric asymptotes of a hyperbola through one labelled centre).
        Statements that differ only by swapping twins cannot be adjudicated from the card."""
        if e in self.named or e not in self.types:
            return []
        col = self._colors()
        return sorted((x for x in self.handle if x != e and x not in self.named and col[x] == col[e]),
                      key=lambda x: self.handle[x])

    def describe(self, e: str) -> str:
        t = self.types.get(e, "entity").replace("_", " ")
        return f"{t} {self.handle[e]}" if e in self.handle else f"{t} {e}"

    def definition(self, e: str, exclude: Iterable[str] = ()) -> str:
        """One ENTITIES line. Named entities need no definition beyond their printed label."""
        t = self.types.get(e, "entity").replace("_", " ")
        h = self.handle[e]
        if e in self.named:
            return f"{h}: the {t} printed with the label {h}."
        bits = []
        if e in self.label:
            bits.append(f"printed with the label “{self.label[e]}”, which {self.dup_count[e]} {t}s share")
        inc: dict[str, list[str]] = {}
        for ph, o in self._incidence(e, frozenset(exclude)):
            if o in self.handle:
                inc.setdefault(ph, []).append(self.describe(o))
        for ph in sorted(inc):
            bits.append(f"{ph}: {', '.join(sorted(set(inc[ph])))}")
        tw = self.twins(e)
        if tw:                                   # say so rather than imply a distinction that does not exist
            bits.append("the reference does not distinguish it from " + ", ".join(self.describe(x) for x in tw))
        elif not inc:
            same = sorted((x for x in self.handle if x not in self.named and self.types.get(x) == self.types.get(e)
                           and self.anchor.get(x) is not None), key=lambda x: (self.anchor[x], x))
            if e in same and len(same) > 1:
                bits.append(f"number {same.index(e) + 1} of {len(same)} such {t}s counted from the left "
                            "of the figure")
        tail = "; ".join(bits) if bits else "no further identifying information"
        return f"{h}: a {t} with no unique printed label ({tail})."

    def context(self, entity_ids: Iterable[str], exclude: Iterable[str] = ()) -> str:
        """ENTITIES block for one call: every referenced entity, plus every entity its definition uses."""
        ex = frozenset(exclude)
        todo, seen = sorted(set(entity_ids) & set(self.handle)), set()
        while todo:
            e = todo.pop()
            if e in seen:
                continue
            seen.add(e)
            if e not in self.named:
                todo += [o for _, o in self._incidence(e, ex) if o in self.handle and o not in seen]
        rows = sorted(seen, key=lambda x: (x not in self.named, self.handle[x]))
        return "\n".join(self.definition(e, ex) for e in rows) or "(none)"


def names_for_card(card) -> EntityNames:
    return EntityNames(card.entities, card.facts)


def fact_statement(f, names: EntityNames) -> str:
    return verbalize(f.predicate, list(f.args), names.describe)


def fact_refs(f) -> tuple[str, ...]:
    return tuple(entity_args(f.predicate, list(f.args)))


def fact_identity(f) -> str:
    """Canonical identity of a fact: predicate and arguments (not its fact_id)."""
    import json
    return json.dumps([f.predicate, list(f.args)], sort_keys=True, ensure_ascii=False)


def check_unique(facts: Sequence[Any], names: EntityNames) -> None:
    """Refuse to send two facts with different identities but identical wording to judges."""
    seen: dict[str, tuple[str, str]] = {}
    for f in facts:
        text, ident = fact_statement(f, names), fact_identity(f)
        if text in seen and seen[text][1] != ident:
            raise StatementCollision(f"facts {seen[text][0]} and {f.fact_id} both render as {text!r}")
        seen[text] = (f.fact_id, ident)


# ------------------------------------------------------------------ Q3
def q3_entity_text(names: EntityNames, entity_ids: Iterable[str]) -> str:
    return names.context(entity_ids)


def q3_question(names: EntityNames, e: str, candidates: Sequence[str]) -> str:
    return f"Which candidate object is {names.describe(e)}? Candidates: " + ", ".join(candidates)


def object_lines(rows: Sequence[tuple[str, str, tuple[float, float] | None]]) -> str:
    """Q3 candidate objects: (object id, type, (x, y) fractions of the canvas or None)."""
    out = []
    for oid, t, pos in rows:
        where = f" at x≈{pos[0]:.2f}, y≈{pos[1]:.2f}" if pos is not None else ""
        out.append(f"{oid}: {t.replace('_', ' ')}{where}")
    return "\n".join(out)


# ------------------------------------------------------------------ verdict JSON Schema (native constraint)
def verdict_json_schema(question: str, ids: Sequence[str], schema_version: str) -> dict:
    """JSON Schema of an ANSWERING judge reply (the prompt states the same form). Sent as a native
    decoding constraint only for judges whose roster entry sets structured_output. It has no `refusal`
    field: a required refusal boolean beside the answers changed the prompt's meaning (Qwen3 VL at
    medium effort set it to true while answering, live 2026-09-26). A constrained judge that refuses
    does so through the API's own refusal signal, which `parse_response` records as a refusal.
    Ids are the call's own ids."""
    key = "object" if question == "Q3" else "answer"
    props: dict[str, Any] = {"id": {"type": "string", "enum": list(ids)},
                             key: {"type": "string"} if key == "object" else {"type": "boolean"}}
    item = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return {"name": f"hmb_judge_{question}",
            "schema": {"type": "object",
                       "properties": {"schema": {"type": "string", "enum": [schema_version]},
                                      "answers": {"type": "array", "items": item}},
                       "required": ["schema", "answers"], "additionalProperties": False}}


# ------------------------------------------------------------------ frozen identity
def serializer_identity() -> dict[str, str]:
    src = Path(__file__).read_bytes()
    return {"version": SERIALIZER_VERSION, "source_sha256": hashlib.sha256(src).hexdigest()}


__all__ = ["SERIALIZER_VERSION", "PREFIX", "TEMPLATES", "EntityNames", "StatementCollision", "check_unique",
           "fact_identity", "fact_refs", "fact_statement", "names_for_card", "object_lines", "q3_entity_text", "q3_question",
           "serializer_identity", "verbalize", "verdict_json_schema"]

