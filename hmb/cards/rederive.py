"""Offline re-derivations from a frozen card (no model calls).

* Context sensitivity (§7.3): the frozen eligible set minus `context_assisted` facts.
* Dedup similarity (§6.7): Jaccard of eligible fact sets after entity canonicalization (entity ->
  type + case/diacritic-insensitive label; unlabelled entities -> type + typed 1-WL colour).

Removed 2026-09-26 (asc-2.0.0 streamlining): leave-one-builder-family-out (with the locked two-writer roster
it reduced to "one writer's facts alone", which the both-writers rule never admits) and the builder-order
card instability (a recorded number with no consumer).
"""
from __future__ import annotations

from collections import Counter

from hmb.ontology import PREDICATES
from hmb.schemas import Card

from .align import wl_colours
from .normalize import dedup_label_key


def frozen_eligible(card: Card) -> set[str]:
    return {r.fact_id for r in card.facts if r.eligible}


def context_excluded(card: Card) -> set[str]:
    return {r.fact_id for r in card.facts if r.eligible and not r.context_assisted}


# ------------------------------------------------------------------ §6.7 dedup
def canonical_fact_multiset(card: Card, wl_rounds: int) -> Counter:
    elig = [r for r in card.facts if r.eligible]
    types = {e.entity_id: e.type for e in card.entities}
    cols = wl_colours([(r.predicate, r.args) for r in elig], types, wl_rounds)[-1]
    name = {}
    for e in card.entities:
        name[e.entity_id] = (f"{e.type}|{dedup_label_key(e.label)}" if e.label
                             else f"{e.type}|#{cols[e.entity_id]}")
    out: Counter = Counter()
    for r in elig:
        p = PREDICATES[r.predicate]
        args = []
        for kind, a in zip(p.args, r.args):
            if kind == "ent" or (kind == "ent?" and a != "unknown"):
                args.append(name.get(a, a))
            elif kind == "set":
                args.append(tuple(sorted(name.get(x, x) for x in a)))
            elif kind == "seq":
                args.append(tuple(name.get(x, x) for x in a))
            elif kind == "text":
                args.append(dedup_label_key(a))
            else:
                args.append(a)
        for i, j in p.symmetric_pairs:
            if str(args[i]) > str(args[j]):
                args[i], args[j] = args[j], args[i]
        out[(r.predicate, tuple(args))] += 1
    return out


def eligible_jaccard(a: Card, b: Card, wl_rounds: int) -> float | None:
    """§6.7 redundancy similarity; None when both eligible sets are empty."""
    A, B = canonical_fact_multiset(a, wl_rounds), canonical_fact_multiset(b, wl_rounds)
    den = sum((A | B).values())
    if den == 0:
        return None
    return sum((A & B).values()) / den
