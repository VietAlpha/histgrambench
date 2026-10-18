"""§8.7 gates + book cap, oracle card (card classes only, asc-2.0.0), §7.3/§6.7
re-derivations."""
import math

from hmb.cards.align import align
from hmb.cards.consensus import consensus
from hmb.cards.gates import apply_book_cap, item_gates
from hmb.cards.oracle import oracle_card
from hmb.cards.rederive import context_excluded, eligible_jaccard
from hmb.cards.verify import build_union, verification_facts
from hmb.vocab import Exclusion, Vote

from w2_scenario import FAM, ORDER, PARAMS, card_th, parse, triangle_json

LINES = [{"id": "l1", "type": "line", "anchor": [0.5, 0.1, 0.9, 0.0]},
         {"id": "l2", "type": "line", "anchor": [0.5, 0.95, 0.9, 0.0]},
         {"id": "pP", "type": "point", "label": "P", "anchor": [0.95, 0.5, 0.02, 0.02]}]


def card_from(extra=None, labels=("A", "B", "C"), extra_entities=()):
    extra = extra or {}
    outs = {b: parse(triangle_json(b, extra_facts=extra.get(b, ()), labels=labels,
                                   extra_entities=extra_entities), b).output for b in ORDER}
    al = align(outs, ORDER, PARAMS)
    U = build_union(al)
    votes = {b: {f: Vote.true for f in verification_facts(U, b)} for b in ORDER}
    return consensus("hmb_t", al, U, votes, FAM, asc_version="a", profile="fixture")


def test_item_gates_each():
    # SC is recorded, never a gate (simplification 2026-09-25); there is no degradation ceiling s* (2026-09-26)
    card = card_from()
    assert item_gates(card, card_th()) == []
    # the triangle has 6 eligible E and 6 strict R; a floor of 7 relations excludes it
    assert item_gates(card, card_th(item_min_strict_relation_facts=7)) == [Exclusion.card_insufficient_facts]
    card.entities[0].ambiguous = True
    card.entities[1].ambiguous = True          # 2/6 > 0.20
    assert Exclusion.card_ambiguous_entities in item_gates(card, card_th())
    bad = card_from(extra={"b1": [{"fact": "parallel", "args": ["l1", "l2"]}],
                           "b3": [{"fact": "intersect_at", "args": ["l1", "l2", "pP"]}]}, extra_entities=LINES)
    # contradiction rate 2/21 = 0.095 <= 0.10; a 0.05 ceiling excludes it
    assert math.isclose(sum("e_violation" in r.ineligible_reasons for r in bad.facts) / len(bad.facts), 2 / 21)
    assert Exclusion.card_contradiction_rate not in item_gates(bad, card_th())
    assert Exclusion.card_contradiction_rate in item_gates(bad, card_th(item_max_contradiction=0.05))


def test_book_cap_deterministic():
    th = card_th()
    items = ([{"item_id": f"a{i}", "book_id": "bkA", "eligible_mgs_count": i} for i in range(6)] +
             [{"item_id": f"{c}{i}", "book_id": f"bk{c}", "eligible_mgs_count": 10} for c in "BCD" for i in range(2)])
    r = apply_book_cap(items, th)
    # hand: N=12, A has 6. Move a0 (6/11), a1 (5/10), a2 (4/9 ... ) until A <= 0.25 N:
    # A=2 of N=8 -> 0.25 ok. Moved a0,a1,a2,a3 (lowest counts first).
    assert [m["item_id"] for m in r["moved"]] == ["a0", "a1", "a2", "a3"]
    assert len(r["kept"]) == 8 and not r["infeasible"]
    r2 = apply_book_cap([{"item_id": "x", "book_id": "b", "eligible_mgs_count": 3}], th)
    assert r2["infeasible"] and r2["kept"] == []


GRAPH = {"spec_id": "tri_1", "family": "triangles_polygons", "seed": 1, "partition": "CAL",
         "canvas": {"w": 10.0, "h": 10.0},
         "entities": [{"entity_id": "p_A", "type": "point", "label": "A", "geom": {"x": 2.0, "y": 2.0}},
                      {"entity_id": "p_B", "type": "point", "label": "B", "geom": {"x": 8.0, "y": 2.0}},
                      {"entity_id": "l_1", "type": "line", "label": "a", "geom": {}},
                      {"entity_id": "l_2", "type": "line", "label": "b", "geom": {}},
                      {"entity_id": "l_3", "type": "line", "label": "c", "geom": {}}],
         "facts": [{"predicate": "on", "args": ["p_A", "l_1"]}, {"predicate": "parallel", "args": ["l_1", "l_2"]},
                   {"predicate": "parallel", "args": ["l_2", "l_3"]}, {"predicate": "left_of", "args": ["p_A", "p_B"]}]}


def test_oracle_card_all_eligible():
    card = oracle_card(GRAPH, profile="fixture")
    assert all(r.eligible for r in card.facts)
    preds = sorted(r.predicate for r in card.facts)
    assert preds.count("label") == 5 and preds.count("point") == 2 and preds.count("line") == 3
    assert preds.count("parallel") == 2 and "left_of" not in preds          # layout is not a card class
    assert card.entities[0].anchor.x == 0.2 and math.isclose(card.entities[0].anchor.y, 0.8)


def test_context_excluded():
    card = card_from()
    assert context_excluded(card) == {r.fact_id for r in card.facts}


def test_dedup_jaccard():
    a = card_from()
    b = card_from(labels=("a", "b", "c"))           # case-insensitive labels: identical sets
    assert eligible_jaccard(a, b, 2) == 1.0
    c = card_from(labels=("A", "B", "D"))
    j = eligible_jaccard(a, c, 2)
    assert 0 < j < 0.9
