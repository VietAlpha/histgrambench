"""§8.3 verification + §8.4 consensus (asc-2.0.0): the fixed two-writer rule, each eligibility condition,
chunked verification with self-veto and line-level vote parsing."""
import math

from hmb.cards.align import align
from hmb.cards.consensus import card_counts, consensus, contradiction_rate
from hmb.cards.verify import (VerificationItem, build_union, fact_text, parse_votes, split_votes,
                              verification_chunks, verification_facts)
from hmb.vocab import Vote

from w2_scenario import FAM, ORDER, PARAMS, parse, triangle_json

T, F, A = Vote.true, Vote.false, Vote.abstain
LINES = [{"id": "l1", "type": "line", "anchor": [0.5, 0.1, 0.9, 0.0]},
         {"id": "l2", "type": "line", "anchor": [0.5, 0.95, 0.9, 0.0]},
         {"id": "pP", "type": "point", "label": "P", "anchor": [0.95, 0.5, 0.02, 0.02]}]


def scenario(extra=None, extra_entities=()):
    extra = extra or {}
    outs = {b: parse(triangle_json(b, extra_facts=extra.get(b, ()), extra_entities=extra_entities), b).output
            for b in ORDER}
    al = align(outs, ORDER, PARAMS)
    return al, build_union(al)


def rec(card, pred, *args):
    for r in card.facts:
        if r.predicate == pred and (not args or tuple(r.args) == args):
            return r
    raise KeyError(pred)


def test_union_is_mgs_and_generated_facts_are_supported_unless_vetoed():
    al, U = scenario()
    # 3 points + 3 segments + 6 endpoint_of + 3 labels (on/vertex facts are entailed, not in U)
    assert len(U.facts) == 15
    assert all(verification_facts(U, b) == U.facts for b in ORDER)   # every writer checks the whole union
    card = consensus("hmb_x", al, U, {}, FAM, asc_version="asc-2.0.0+t", profile="fixture")
    assert all(r.eligible for r in card.facts)
    assert sorted(r.fact_id for r in card.facts)[:3] == ["e_001", "e_002", "e_003"]
    r = card.facts[0]
    assert r.votes == {b: "generated" for b in ORDER} and r.supporting_families == ["fam_x", "fam_y"]
    c = card_counts(card)
    assert c["eligible_mgs_count"] == {"E": 6, "R": 3 * 2, "L": 3} and c["contradiction_rate"] == 0.0
    assert "union_coverage_ratio" not in c
    assert card.card_uid.startswith("card_")


def test_eligibility_both_writers_support_none_votes_false():
    # §8.4: the fact is b1's only; b3 is asked about it and its vote decides
    al, U = scenario({"b1": [{"fact": "perpendicular", "args": ["s1", "s2"]}]})
    f = next(k for k in U.facts if k[0] == "perpendicular")
    assert f in verification_facts(U, "b3") and f in verification_facts(U, "b1")

    def reasons(v):
        card = consensus("i", al, U, {"b3": {f: v}}, FAM, asc_version="a", profile="fixture")
        r = rec(card, "perpendicular")
        return r.ineligible_reasons, r.votes
    assert reasons(T) == ([], {"b1": "generated", "b3": "true"})
    assert reasons(F)[0] == ["not_supported_by_both_writers", "writer_voted_false"]
    assert reasons(A)[0] == ["not_supported_by_both_writers"]          # silence is not a vote
    card = consensus("i", al, U, {}, FAM, asc_version="a", profile="fixture")   # no vote recorded
    assert rec(card, "perpendicular").ineligible_reasons == ["not_supported_by_both_writers"]
    # self-veto: the generating writer votes false on its own fact
    card = consensus("i", al, U, {"b1": {f: F}, "b3": {f: T}}, FAM, asc_version="a", profile="fixture")
    r = rec(card, "perpendicular")
    assert r.votes == {"b1": "false", "b3": "true"} and "writer_voted_false" in r.ineligible_reasons


def test_eligibility_e_violation():
    al, U = scenario({"b1": [{"fact": "parallel", "args": ["l1", "l2"]}],
                      "b3": [{"fact": "intersect_at", "args": ["l1", "l2", "pP"]}]}, LINES)
    votes = {b: {f: T for f in verification_facts(U, b)} for b in ORDER}
    card = consensus("i", al, U, votes, FAM, asc_version="a", profile="fixture")
    bad = [r for r in card.facts if r.predicate in ("parallel", "intersect_at")]
    assert len(bad) == 2 and all(r.ineligible_reasons == ["e_violation"] for r in bad)
    assert len(card.contradictions) == 1
    assert math.isclose(contradiction_rate(card), 2 / len(card.facts))


def test_eligibility_ambiguous_argument():
    al, U = scenario()
    slot = next(s for s in al.slots if s.label == "A")
    slot.ambiguous = True
    card = consensus("i", al, U, {}, FAM, asc_version="a", profile="fixture")
    hit = [r for r in card.facts if slot.cid in str(r.args)]
    assert hit and all("ambiguous_argument" in r.ineligible_reasons for r in hit)
    assert all(r.eligible for r in card.facts if slot.cid not in str(r.args))


def test_eligibility_anchor_b():
    hidden = [{"fact": "hidden_edge", "args": ["s1"]}]
    # labelled triangle: unlabelled s1 binds structurally through its two labelled endpoints (fix #4)
    al, U = scenario({b: hidden for b in ORDER})
    card = consensus("i", al, U, {}, FAM, asc_version="a", profile="fixture")
    assert rec(card, "hidden_edge").eligible
    # unlabelled triangle: no point is bindable, so neither is s1
    outs = {b: parse(triangle_json(b, extra_facts=hidden, labels=(None, None, None)), b).output for b in ORDER}
    al2 = align(outs, ORDER, PARAMS)
    U2 = build_union(al2)
    card2 = consensus("i", al2, U2, {}, FAM, asc_version="a", profile="fixture")
    assert rec(card2, "hidden_edge").ineligible_reasons == ["anchor_b_unbindable"]


def test_bindable_entities_propagation():
    from hmb.cards.consensus import bindable_entities
    from hmb.cards.closure import forward_chain
    al, U = scenario()
    b = bindable_entities(al, U.closure)
    assert {s.cid for s in al.slots} == b                  # 3 labelled points + 3 segments via endpoints
    # give seg[1] (BC) the extra endpoints A, B: segment AB's only bindable pair {A,B} is now shared
    # -> AB is not uniquely bindable; BC still binds through its unique pair {B,C}
    seg = [s for s in al.slots if s.type == "segment"]
    pA, pB = [s.cid for s in al.slots if s.label in ("A", "B")]
    extra = forward_chain(set(U.closure) | {("endpoint_of", (pA, seg[1].cid)), ("endpoint_of", (pB, seg[1].cid))},
                          U.types)
    b2 = bindable_entities(al, extra)
    s_ab = next(s.cid for s in seg if ("on", (pA, s.cid)) in U.closure and ("on", (pB, s.cid)) in U.closure)
    assert s_ab != seg[1].cid and s_ab not in b2 and seg[1].cid in b2


def test_verification_is_chunked_and_deterministic():
    # b3 declares one extra point; every writer checks the whole union (15 triangle facts + b3's point)
    outs = {"b1": parse(triangle_json("b1"), "b1").output,
            "b3": parse('{"id": "q", "type": "point", "anchor": [0.9, 0.1, 0.02, 0.02]}', "b3").output}
    al = align(outs, ORDER, PARAMS)
    U = build_union(al)
    mine = verification_facts(U, "b3")
    assert len(mine) == 16 and ("point", (al.local_to_canon[("b3", "q")],)) in mine
    chunks = verification_chunks(U, al, "hmb_1", "b3", 4)
    assert [len(c) for c in chunks] == [4, 4, 4, 4]
    assert all([i.sid for i in c] == [f"s{k + 1:02d}" for k in range(len(c))] for c in chunks)   # ids restart
    assert sorted((i.fact for c in chunks for i in c), key=repr) == sorted(mine, key=repr)
    assert all(i.text == fact_text(i.fact, al) for c in chunks for i in c)
    again = verification_chunks(U, al, "hmb_1", "b3", 4)
    assert [[(i.sid, i.fact) for i in c] for c in chunks] == [[(i.sid, i.fact) for i in c] for c in again]
    assert [len(c) for c in verification_chunks(U, al, "hmb_1", "b1", 4)] == [4, 4, 4, 4]


def test_label_statement_does_not_leak_label():
    al, U = scenario()
    f = next(k for k in U.facts if k[0] == "label" and k[1][0] == "A")
    t = fact_text(f, al)
    assert t.count("A") == 1 and "unlabelled" not in t


def test_parse_votes_lines_silence_conflict_truncation():
    items = [VerificationItem(f"s0{i}", "x", ("point", (f"p{i}",))) for i in range(1, 6)]
    raw = "s01 TRUE\ns02: false\ns03 true\ns03 false\ns99 true\nnoise line\ns04 abst"
    vp = parse_votes(raw, items)
    assert vp.votes == {"s01": T, "s02": F, "s03": A, "s04": A, "s05": A}
    assert vp.missing == ["s04", "s05"] and len(vp.invalid) == 2      # unknown s99 + conflicting s03
    bad = parse_votes("I think they are all true", items)
    assert set(bad.votes.values()) == {A} and len(bad.missing) == 5
    union = split_votes(items, vp)
    assert union[("point", ("p1",))] == T and union[("point", ("p2",))] == F and union[("point", ("p5",))] == A
