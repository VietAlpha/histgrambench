"""§8.4 consensus, eligibility, and the §8.5 frozen fact record (asc-2.0.0).

One fixed rule for the locked two-writer roster (2 writers in 2 families, `roster.builder_rule_problems`;
2026-09-26 streamlining: the generic family-count machinery and its one parameter are gone). A fact of the
union MGS is eligible iff
  (1) both writers support it: it is in the closure of the writer's own generation ("generated", unless the
      writer voted false on it when asked, a self-veto), or the writer voted true when asked;
  (2) no writer voted false;
  (3) it participates in no E-violation;
  (4) every argument is a non-ambiguous canonical entity;
  (5) anchor B -> every argument is uniquely bindable.
Card precision on historical crops is not certified (no control suite since 2026-09-26);
the card gates (§8.7) and the development-set measurements report what is observable.

A missing vote is `abstain` (silence is not a vote, §8.3). "Uniquely bindable" (form choice, consistent with
the §10.3 binding order: labels first, then structure): a non-ambiguous entity is bindable if its
§7.5-normalized label is unique among the card's entities, or it is the only entity of its type; then,
propagated to a fixpoint over the closure of U, an unlabelled segment/line/ray is bindable if it has >= 2
distinct bindable incident points (endpoints included, via R1) that no other entity of its type also has, and
a circle is bindable if it has a bindable centre plus a bindable incident point that no other circle shares.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping

from hmb.hashing import card_uid as _card_uid
from hmb.ontology import PREDICATES, entity_args, mgs_sort_key
from hmb.schemas import Card, FactRecord, SemanticCardSummary
from hmb.vocab import CLASS_LETTER, FactClass, Vote

from .align import Alignment
from .closure import FactKey, contradiction_participants
from .normalize import normalize_label
from .verify import Union

STRUCTURAL_REASONS = frozenset({"e_violation", "ambiguous_argument", "anchor_b_unbindable"})


def category_key(pred: str) -> str:
    p = PREDICATES[pred]
    return f"{p.category}_{p.strictness.value}"


SUPPORT = frozenset({Vote.generated, Vote.true})


def decide(votes: Mapping[str, Vote | str], structural: list[str]) -> list[str]:
    """Ineligibility reasons under the §8.4 rule (empty list = eligible). `votes` has one entry per writer."""
    vs = [Vote(v) for v in votes.values()]
    reasons: list[str] = []
    if len(vs) != 2 or not all(v in SUPPORT for v in vs):
        reasons.append("not_supported_by_both_writers")
    if Vote.false in vs:
        reasons.append("writer_voted_false")
    return reasons + [r for r in structural if r not in reasons]


def _fact_args_json(f: FactKey) -> list:
    return [list(a) if isinstance(a, tuple) else a for a in f[1]]


def fact_label(f: FactKey) -> str:
    return f"{f[0]}({', '.join(str(list(a)) if isinstance(a, tuple) else str(a) for a in f[1])})"


def assign_fact_ids(U: list[FactKey]) -> dict[FactKey, str]:
    """Stable ids: class letter + running index over the union MGS in ascending §8.2 key order."""
    counters: dict[str, int] = defaultdict(int)
    out = {}
    for f in sorted(U, key=lambda k: mgs_sort_key(k[0], list(k[1]))):
        letter = CLASS_LETTER[PREDICATES[f[0]].fclass].lower()
        counters[letter] += 1
        out[f] = f"{letter}_{counters[letter]:03d}"
    return out


LINEAR_BIND = frozenset({"segment", "line", "ray"})


def bindable_entities(al: Alignment, closure: set[FactKey]) -> set[str]:
    """Entities that are uniquely bindable (anchor-B condition (6)); see the module docstring."""
    slots = [s for s in al.slots if not s.ambiguous]
    bound: set[str] = set()
    for s in slots:
        if s.label:
            n = normalize_label(s.label)
            if sum(1 for x in al.slots if x.label and normalize_label(x.label) == n) == 1:
                bound.add(s.cid)
        elif sum(1 for x in al.slots if x.type == s.type) == 1:
            bound.add(s.cid)
    incident: dict[str, set[str]] = defaultdict(set)
    centres: dict[str, set[str]] = defaultdict(set)
    for f in closure:
        if f[0] == "on":
            incident[f[1][1]].add(f[1][0])
        elif f[0] == "center_of":
            centres[f[1][1]].add(f[1][0])
    changed = True
    while changed:
        changed = False
        for s in slots:
            if s.cid in bound:
                continue
            ok = False
            if s.type in LINEAR_BIND:
                pts = sorted(incident[s.cid] & bound)
                for i in range(len(pts)):
                    for j in range(i + 1, len(pts)):
                        pair = {pts[i], pts[j]}
                        if not any(t.cid != s.cid and t.type == s.type and pair <= incident[t.cid]
                                   for t in al.slots):
                            ok = True
            elif s.type == "circle":
                for o in sorted(centres[s.cid] & bound):
                    for p in sorted(incident[s.cid] & bound):
                        if not any(t.cid != s.cid and t.type == "circle" and o in centres[t.cid]
                                   and p in incident[t.cid] for t in al.slots):
                            ok = True
            if ok:
                bound.add(s.cid)
                changed = True
    return bound


def consensus(item_id: str, al: Alignment, U: Union, votes: dict[str, dict[FactKey, Vote]],
              family_map: dict[str, str], *, asc_version: str, profile: str,
              abstentions: list[dict] | None = None) -> Card:
    """Pure: alignment + union + verification votes -> frozen Card (§8.4, §8.5). A writer's vote on a fact
    is `generated` when the fact is in the closure of its own generation and it did not vote false on it, else
    its verification vote."""
    types = U.types
    participants, pairs = contradiction_participants(U.facts, types) if U.facts else (set(), [])
    ambiguous = al.ambiguous
    bindable = bindable_entities(al, U.closure)
    ids = assign_fact_ids(U.facts)
    records: list[FactRecord] = []
    for f in sorted(U.facts, key=lambda k: ids[k]):
        pred, args = f[0], list(f[1])
        p = PREDICATES[pred]
        fv = {}
        for b in al.builder_order:
            v = Vote(votes.get(b, {}).get(f, Vote.abstain))
            own = f in U.per_builder_closure.get(b, set())
            fv[b] = Vote.generated if own and v != Vote.false else v      # own facts: supported unless self-vetoed
        structural = []
        if f in participants:
            structural.append("e_violation")
        ents = entity_args(pred, args)
        if any(e in ambiguous for e in ents):
            structural.append("ambiguous_argument")
        if p.anchor.value == "B" and not all(e in bindable for e in ents):
            structural.append("anchor_b_unbindable")
        reasons = decide(fv, structural)
        records.append(FactRecord.model_validate({
            "fact_id": ids[f], "class": p.fclass.value, "predicate": pred, "args": _fact_args_json(f),
            "strictness": p.strictness.value, "decidability": p.decidability.value,
            "anchor": p.anchor.value, "context_assisted": f in U.context_assisted,
            "supporting_families": sorted({family_map[b] for b, v in fv.items() if v in SUPPORT}),
            "votes": {b: v.value for b, v in fv.items()},
            "eligible": not reasons, "ineligible_reasons": reasons}))
    contr = [[ids.get(a, fact_label(a)), ids.get(b, fact_label(b))] for a, b in pairs]
    eligible_set = [{"predicate": r.predicate, "args": r.args} for r in records if r.eligible]
    return Card(item_id=item_id, asc_version=asc_version,
                card_uid=_card_uid(eligible_set, asc_version),
                entities=al.entities(), facts=records, abstained=list(abstentions or []),
                contradictions=contr,
                builder_order=list(al.builder_order), family_map=dict(family_map), profile=profile)


# ------------------------------------------------------------------ card statistics
def eligible_counts(card: Card) -> dict[str, int]:
    out = {"E": 0, "R": 0, "L": 0}
    for r in card.facts:
        if r.eligible:
            out[CLASS_LETTER[FactClass(r.fact_class)]] += 1
    return out


def contradiction_rate(card: Card) -> float | None:
    """|facts in V| / |M| over the union MGS (§8.2); None for an empty card."""
    if not card.facts:
        return None
    return sum(1 for r in card.facts if "e_violation" in r.ineligible_reasons) / len(card.facts)


def ambiguous_share(card: Card) -> float | None:
    if not card.entities:
        return None
    return sum(1 for e in card.entities if e.ambiguous) / len(card.entities)


def card_counts(card: Card) -> dict:
    """Counts for the §5.2 `semantic_card` block (paths/hashes and SC are added by the caller)."""
    elig = [r for r in card.facts if r.eligible]
    return {
        "asc_version": card.asc_version, "card_uid": card.card_uid,
        "union_mgs_count": len(card.facts), "eligible_mgs_count": eligible_counts(card),
        "eligible_geometric_count": sum(1 for r in elig if r.decidability == "geo"),
        "eligible_model_count": sum(1 for r in elig if r.decidability == "model"),
        "abstained_count": len(card.abstained),
        "ambiguous_entity_count": sum(1 for e in card.entities if e.ambiguous),
        "contradiction_rate": contradiction_rate(card),
        "context_assisted_fact_count": sum(1 for r in card.facts if r.context_assisted),
    }


def semantic_card_summary(card: Card, *, generation_manifest: str, generation_manifest_sha256: str,
                          facts: str, facts_sha256: str) -> SemanticCardSummary:
    c = card_counts(card)
    if c["contradiction_rate"] is None:
        raise ValueError("contradiction_rate undefined for an empty card; the item is not eligible (§11.12)")
    return SemanticCardSummary(**c, generation_manifest=generation_manifest,
                               generation_manifest_sha256=generation_manifest_sha256, facts=facts,
                               facts_sha256=facts_sha256)


# ------------------------------------------------------------------ §15.1 consensus_rule_hash
def consensus_rule_hash() -> str:
    """sha256 of this module's source (the fixed rule has no parameter); feeds the §15.1 card_hash."""
    from pathlib import Path as _P

    from hmb.hashing import sha256_file
    return sha256_file(_P(__file__))


def semantic_card_summary_for(card: Card, layout) -> SemanticCardSummary:
    """§5.2 `semantic_card` block with paths taken from the Layout: generation_manifest is the per-item
    manifest `manifests/cards/<card_uid>/<item_id>.json` (C29) and facts is the frozen card file.
    Paths are relative to the layout root; the sha256s are of the stored bytes."""
    from hmb.hashing import sha256_file
    man = layout.card_manifest(card.card_uid, card.item_id)
    facts = layout.card(card.item_id)
    rel = lambda p: p.relative_to(layout.root).as_posix()   # noqa: E731
    return semantic_card_summary(card, generation_manifest=rel(man), generation_manifest_sha256=sha256_file(man),
                                 facts=rel(facts), facts_sha256=sha256_file(facts))
