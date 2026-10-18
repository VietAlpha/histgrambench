"""Writer generation reply (JSON Lines, asc-2.0.0) -> `BuilderOutput` (§7.4), with OOV dropping (§7.2), the
closure requirement (§7.4) and the hard caps `card_max_entities` / `card_max_facts`.

Every line is parsed on its own, so a reply cut at `max_tokens` keeps every complete line and loses only the
unfinished last one (2026-09-26 streamlining: one truncated reply used to discard a writer's whole card).
Hostile or malformed output never raises:

* an entity line `{"id", "type", "label"?, "label_from_context"?, "param"?, "anchor": [x, y, w, h]}` of an
  unknown type -> `oov_dropped`; a duplicate id, the reserved id `unknown`, or a missing/invalid anchor ->
  dropped with a note (a bare centre [x, y] is a zero-extent anchor); entity lines after the first
  `card_max_entities` -> dropped (`over_cap_entities`);
* a fact line `{"fact", "args", "label_from_context"?}` whose predicate is outside the card fact table
  (layout, derived-only, unknown) -> `oov_dropped`; an entity or `label` predicate on a fact line is
  redundant with the entity lines and is skipped with a note; fact lines after the first `card_max_facts`
  -> dropped (`over_cap_facts`);
* a fact that is not self-contained (wrong arity or argument kind, an undeclared entity id, `unknown`
  outside the tangent point), breaks the §7.1 typed positions (`ontology.ARG_TYPES`) or a binding
  convention, or carries `label_from_context` on a non-label fact -> abstained at build time
  (`abstentions`, scope "fact");
* a line that is not a JSON object (the truncated tail, prose, fences) -> counted in `bad_lines`.

Entity-existence facts (the existence layer, §8.2) and `label` facts are derived from the entity lines:
unary types get `type(e)`, parametric types `type(e, param)` when the param is valid (else an entity
abstention), labelled entities `label(text, e)` (context-assisted when `label_from_context` is true).
The reply is malformed (output None) only when no entity line survives.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from hmb.ontology import (ASC_MAJOR_MINOR_PATCH, CARD_PREDICATES, ENTITY_TYPES, PREDICATES, OntologyError,
                          arg_type_violation, entity_args, validate_fact)
from hmb.schemas import BuilderOutput

from .prompts import entity_param, fact_predicates

UNARY_ENTITY_TYPES = frozenset(t for t in ENTITY_TYPES if PREDICATES[t].arity == 1)
FACT_LINE_PREDICATES = frozenset(fact_predicates())


@dataclass
class ParseResult:
    output: BuilderOutput | None
    malformed: str | None = None                 # reason when output is None
    n_in_vocab: int = 0                          # entity and fact lines in the closed vocabulary
    n_oov: int = 0
    bad_lines: int = 0                           # lines that are not a JSON object (e.g. the truncated tail)
    over_cap_entities: int = 0
    over_cap_facts: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def oov_rate(self) -> float | None:
        n = self.n_in_vocab + self.n_oov
        return None if n == 0 else self.n_oov / n


def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _verbatim(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)[:2000]
    except (TypeError, ValueError):
        return repr(obj)[:2000]


def _records(raw: str) -> tuple[list[dict], int]:
    """JSON objects, one per line (a trailing comma, as inside an array, is tolerated)."""
    out, bad = [], 0
    for line in (raw or "").splitlines():
        s = line.strip().rstrip(",").strip()
        if not s or s.startswith("```") or s in ("[", "]"):
            continue
        try:
            d = json.loads(s)
        except (json.JSONDecodeError, RecursionError):
            bad += 1
            continue
        if isinstance(d, dict):
            out.append(d)
        else:
            bad += 1
    return out, bad


def _anchor(a: Any) -> dict | None:
    """[x, y, w, h] (or the {x, y, w, h} object); a bare centre [x, y] is a zero-extent anchor (a writer that
    omitted the extent, measured live 2026-09-26)."""
    if isinstance(a, dict):
        a = [a.get(k) for k in ("x", "y", "w", "h")] if "w" in a or "h" in a else [a.get("x"), a.get("y")]
    if isinstance(a, list) and len(a) == 2:
        a = [*a, 0.0, 0.0]
    if isinstance(a, list) and len(a) == 4 and all(_finite(v) for v in a) and a[2] >= 0 and a[3] >= 0:
        return dict(zip(("x", "y", "w", "h"), (float(v) for v in a)))
    return None


def _set_args(pred: str, args: Any) -> Any:
    """A set predicate (collinear, concyclic) written with its members as flat args is one set argument
    (measured live 2026-09-26: writers wrote collinear(A, B, C, D) instead of collinear([A, B, C, D]))."""
    if (isinstance(args, list) and PREDICATES[pred].args == ("set",) and len(args) >= 3
            and all(isinstance(a, str) for a in args)):
        return [args]
    return args


LINEAR_TYPES = frozenset({"segment", "line", "ray", "axis"})


def convention_violation(pred: str, args: list, types: dict[str, str]) -> str | None:
    """Binding geometric conventions (lead, 2026-09-22). Returns a reason, or None if the fact conforms."""
    if pred in ("parallel", "perpendicular") and any(types.get(a) not in LINEAR_TYPES for a in args):
        return f"{pred} only over segment/line/ray/axis"
    return None


def stage_line_count(raw: str, stage: str) -> int:
    """Complete JSON lines of a staged reply's own kind (entity lines carry "id", fact lines "fact")."""
    recs, _ = _records(raw)
    return sum(1 for d in recs if ("fact" in d) == (stage == "facts") and ("fact" in d or "id" in d))


def parse_builder_output(raw: str, builder_id: str, builder_family: str, *, max_entities: int,
                         max_facts: int, fact_raw: str | None = None) -> ParseResult:
    """Parse one writer's generation. Identity fields come from the caller, not the model.

    Staged generation (`fact_raw` given, 2026-09-26): entity lines are read from `raw` (the entity call) only and
    fact lines from `fact_raw` (the fact call) only; a line of the other kind in either reply is ignored with a
    note, so neither call can add to the other's cap."""
    recs, bad = _records(raw)
    res = ParseResult(None, bad_lines=bad)
    if fact_raw is not None:
        frecs, fbad = _records(fact_raw)
        res.bad_lines += fbad
        off = sum(1 for d in recs if "fact" in d) + sum(1 for d in frecs if "fact" not in d)
        if off:
            res.notes.append(f"{off} line(s) of the other stage's kind ignored")
        recs = [d for d in recs if "fact" not in d] + [d for d in frecs if "fact" in d]
    if not recs:
        res.malformed = "empty response" if not (raw or "").strip() else "no JSON line"
        return res
    oov: list[str] = []
    abst: list[dict] = []
    entities: list[dict] = []
    ids: dict[str, str] = {}
    params: dict[str, Any] = {}
    label_ctx: set[str] = set()
    fact_lines: list[dict] = []
    for d in recs:
        if "fact" in d:
            fact_lines.append(d)
            continue
        if "id" not in d:
            res.bad_lines += 1
            continue
        lid = d["id"].strip() if isinstance(d["id"], str) else ""
        if not lid or lid == "unknown":
            res.notes.append(f"entity dropped (no usable id): {_verbatim(d)}")
            continue
        if d.get("type") not in ENTITY_TYPES:
            oov.append(_verbatim(d))
            res.n_oov += 1
            continue
        res.n_in_vocab += 1
        if len(entities) >= max_entities:
            res.over_cap_entities += 1
            continue
        if lid in ids:
            res.notes.append(f"duplicate id {lid!r} dropped")
            continue
        a = _anchor(d.get("anchor"))
        if a is None:
            res.notes.append(f"entity {lid!r} dropped: anchor [x, y, w, h] required")
            continue
        label = d.get("label")
        if not isinstance(label, str) or not label.strip():
            label = None
        entities.append({"local_id": lid, "type": d["type"], "label": label, "anchor": a})
        ids[lid] = d["type"]
        params[lid] = d.get("param")
        if label is not None and d.get("label_from_context") is True:
            label_ctx.add(lid)
    if not entities:
        res.malformed = "no entity line"
        return res

    facts: list[dict] = []
    seen: set[str] = set()

    def add(pred: str, args: list, ctx: bool) -> None:
        key = _verbatim([pred, args])
        if key not in seen:
            seen.add(key)
            facts.append({"predicate": pred, "args": args, "context_assisted": ctx})

    for e in entities:                              # the existence and label layer, from the entity lines
        lid, t = e["local_id"], e["type"]
        if t in UNARY_ENTITY_TYPES:
            add(t, [lid], False)
        else:
            try:
                validate_fact(t, [lid, params[lid]])
                add(t, [lid, params[lid]], False)
            except OntologyError as err:            # kept for alignment, never silently missing (fix #14)
                abst.append({"scope": "entity", "entity": lid,
                             "reason": f"{t} needs param {entity_param(t)}; existence fact not derived: {err}"})
        if e["label"] is not None:
            add("label", [e["label"], lid], lid in label_ctx)

    n_fact_lines = 0
    for f in fact_lines:
        pred = f.get("fact")
        if isinstance(pred, str) and pred in CARD_PREDICATES and pred not in FACT_LINE_PREDICATES:
            res.notes.append(f"{pred} on a fact line is stated by the entity lines; skipped")
            continue
        if not isinstance(pred, str) or pred not in FACT_LINE_PREDICATES:
            oov.append(_verbatim(f))
            res.n_oov += 1
            continue
        res.n_in_vocab += 1
        n_fact_lines += 1
        if n_fact_lines > max_facts:
            res.over_cap_facts += 1
            continue
        args = _set_args(pred, f.get("args"))
        reason = None
        if not isinstance(args, list):
            reason = "args not a list"
        else:
            try:
                validate_fact(pred, args)
                missing = [x for x in entity_args(pred, args) if x not in ids]
                reason = (f"not self-contained: undeclared entity {missing}" if missing
                          else arg_type_violation(pred, args, ids) or convention_violation(pred, args, ids))
            except (OntologyError, TypeError) as err:
                reason = f"malformed: {err}"
        ctx = f.get("label_from_context") is True
        if reason is None and ctx and PREDICATES[pred].fclass.value != "label":
            reason = "label_from_context on a non-label fact (§7.3)"
        if reason is not None or not isinstance(args, list):
            abst.append({"scope": "fact", "entity": None, "reason": f"{reason}: {_verbatim(f)}"})
            continue
        add(pred, args, ctx)

    res.output = BuilderOutput.model_validate({
        "asc_version": ASC_MAJOR_MINOR_PATCH, "builder_id": builder_id, "builder_family": builder_family,
        "entities": entities, "facts": facts, "abstentions": abst, "oov_dropped": oov})
    return res
