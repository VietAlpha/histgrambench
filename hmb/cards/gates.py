"""§8.7 item eligibility gates (card part) and the 25 percent book-share cap.

Rights/provenance (R3) and the duplicate-representative rule (§6.7) are gated by their owners; this
module covers the card-derived gates. The structure-coverage proxy SC is recorded, never a gate
(simplification 2026-09-25: it has no calibration). Missing evidence never passes: an undefined contradiction
rate or ambiguous share excludes the item with the corresponding closed-enum reason. There is no degradation
ceiling s*: it was fitted on synthetic degraded controls, deleted with CP3 (2026-09-26).
"""
from __future__ import annotations

from collections import Counter

from hmb.schemas import Card
from hmb.vocab import Exclusion

from .consensus import ambiguous_share, contradiction_rate, eligible_counts


def item_gates(card: Card, th) -> list[Exclusion]:
    """Closed-enum exclusion reasons (empty list = the card gates pass)."""
    out: list[Exclusion] = []
    n = eligible_counts(card)
    n_strict_rel = sum(1 for r in card.facts
                       if r.eligible and r.fact_class == "relation" and r.strictness == "strict")
    if n["E"] < int(th.value("item_min_entity_facts")) or \
            n_strict_rel < int(th.value("item_min_strict_relation_facts")):
        out.append(Exclusion.card_insufficient_facts)
    cr = contradiction_rate(card)
    if cr is None or cr > float(th.value("item_max_contradiction")):
        out.append(Exclusion.card_contradiction_rate)
    am = ambiguous_share(card)
    if am is None or am > float(th.value("item_max_ambiguous_frac")):
        out.append(Exclusion.card_ambiguous_entities)
    return out


def apply_book_cap(items: list[dict], th) -> dict:
    """No book may contribute more than `book_max_share` of the eligible set (§8.7).

    items: [{item_id, book_id, eligible_mgs_count}] (eligible_mgs_count = total eligible facts).
    Deterministic rule (recorded in the manifest): while some book exceeds the cap, the book with the
    most items (ties: smallest book_id) moves its item with the lowest eligible_mgs_count (ties:
    lexicographically smallest item_id) to the unscored pool. Returns kept/moved ids and the rule.
    With fewer than ceil(1/cap) books the cap is unsatisfiable and every item ends up moved; this is
    reported as `infeasible`.
    """
    cap = float(th.value("book_max_share"))
    def total(it):
        return int(it["eligible_mgs_count"])
    pool = sorted(items, key=lambda it: it["item_id"])
    kept = {it["item_id"]: it for it in pool}
    moved: list[dict] = []
    while kept:
        per = Counter(it["book_id"] for it in kept.values())
        N = len(kept)
        book, cnt = min(per.items(), key=lambda kv: (-kv[1], kv[0]))
        if cnt <= cap * N + 1e-12:
            break
        victim = min((it for it in kept.values() if it["book_id"] == book),
                     key=lambda it: (total(it), it["item_id"]))
        del kept[victim["item_id"]]
        moved.append({"item_id": victim["item_id"], "book_id": book,
                      "eligible_mgs_count": total(victim), "reason": Exclusion.book_share_cap.value})
    return {"kept": sorted(kept), "moved": moved, "cap": cap,
            "infeasible": bool(items) and not kept,
            "rule": "while max book share > cap: from the largest book (ties smallest book_id) move the "
                    "item with lowest eligible_mgs_count (ties smallest item_id) to the unscored pool"}
