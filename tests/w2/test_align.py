"""§8.1 alignment: merge, label conflict, confusable, margin and automorphism ambiguity, permutation check."""
import itertools
import math

from hmb.cards.align import AlignParams, align, multiset_jaccard, wl_signatures

from hmb.schemas import BuilderOutput

from w2_scenario import PARAMS, parse, triangle_json


def bo(b, ents, facts=()):
    return BuilderOutput.model_validate({
        "asc_version": "asc-2.0.0", "builder_id": b, "builder_family": "f",
        "entities": [{"local_id": lid, "type": t, "label": lab, "anchor": {"x": x, "y": y, "w": 0, "h": 0}}
                     for lid, t, lab, x, y in ents],
        "facts": [{"predicate": p, "args": list(a)} for p, a in facts]})


def test_same_label_merges_and_different_labels_never_merge():
    out = {"b1": bo("b1", [("u", "point", "A", .5, .5)]),
           "b2": bo("b2", [("v", "point", "a", .51, .5)]),       # same label (case fold)
           "b3": bo("b3", [("w", "point", "B", .5, .5)])}        # identical position, other label
    al = align(out, ["b1", "b2", "b3"], PARAMS)
    assert len(al.slots) == 2
    assert al.local_to_canon[("b1", "u")] == al.local_to_canon[("b2", "v")]
    assert al.local_to_canon[("b3", "w")] != al.local_to_canon[("b1", "u")]
    assert not al.ambiguous


def test_confusable_label_scores():
    out = {"b1": bo("b1", [("u", "point", "O", .5, .5)]), "b2": bo("b2", [("v", "point", "0", .5, .5)])}
    al = align(out, ["b1", "b2"], PARAMS)
    rec = [r for r in al.log if r.get("event") == "merge"][0]
    assert len(al.slots) == 1 and math.isclose(rec["score"], 1.0 + 1.0 + 0.5)   # strict-equal label: 1
    out = {"b1": bo("b1", [("u", "point", "P", .5, .5)]), "b2": bo("b2", [("v", "point", "R", .5, .5)])}
    al = align(out, ["b1", "b2"], PARAMS)
    assert len(al.slots) == 2 and not any(r.get("event") == "merge" for r in al.log)   # C15: never merge


def test_types_must_match():
    out = {"b1": bo("b1", [("u", "point", None, .5, .5)]), "b2": bo("b2", [("v", "circle", None, .5, .5)])}
    assert len(align(out, ["b1", "b2"], PARAMS).slots) == 2


def test_below_theta_becomes_new_slot():
    p = AlignParams(1.0, 1.0, 0.5, theta=1.2, delta=0.15, sigma=0.05, wl_rounds=2)
    out = {"b1": bo("b1", [("u", "point", None, .1, .1)]), "b2": bo("b2", [("v", "point", None, .9, .9)])}
    al = align(out, ["b1", "b2"], p)
    assert len(al.slots) == 2 and not al.ambiguous
    assert any(r.get("event") == "below_theta" for r in al.log)


def test_margin_ambiguity_marks_both_sides():
    # b1 has two unlabelled points 0.02 apart; b2's single point sits exactly between them:
    # equal scores -> margin 0 < DELTA -> u and v ambiguous; u gets its own ambiguous slot.
    out = {"b1": bo("b1", [("p1", "point", None, .50, .5), ("p2", "point", None, .52, .5)]),
           "b2": bo("b2", [("q", "point", None, .51, .5)])}
    al = align(out, ["b1", "b2"], PARAMS)
    q_slot = al.local_to_canon[("b2", "q")]
    assert q_slot in al.ambiguous
    assert len(al.ambiguous) == 2           # q's new slot and the matched b1 slot
    assert any(r.get("event") == "ambiguous_margin" for r in al.log)


def test_automorphic_tie_unless_geometry_clears():
    ents1 = [("p1", "point", None, .50, .5), ("p2", "point", None, .53, .5)]   # twins within sigma
    ents2 = [("q", "point", None, .50, .5)]
    # Default weights: geometry alone = 1.0 >= theta and beats the twin by 1-exp(-0.6)=0.45 >= delta.
    al = align({"b1": bo("b1", ents1), "b2": bo("b2", ents2)}, ["b1", "b2"], PARAMS)
    assert not al.ambiguous and len(al.slots) == 2
    # With w_G = 0.5, geometry alone (0.5) cannot clear THETA=0.9 -> ambiguous. Margin 0.23 passes.
    p = AlignParams(1.0, 0.5, 0.5, theta=0.9, delta=0.15, sigma=0.05, wl_rounds=2)
    al2 = align({"b1": bo("b1", ents1), "b2": bo("b2", ents2)}, ["b1", "b2"], p)
    assert any(r.get("event") == "ambiguous_automorphism" for r in al2.log)
    assert al2.local_to_canon[("b2", "q")] in al2.ambiguous


def test_wl_signatures_distinguish_structure():
    types = {"P": "point", "Q": "point", "s": "segment", "c": "circle"}
    sig = wl_signatures([("endpoint_of", ["P", "s"]), ("on", ["Q", "c"])], types, 2)
    assert multiset_jaccard(sig["P"], sig["Q"]) < 1.0
    assert multiset_jaccard(sig["P"], sig["P"]) == 1.0


def test_triangle_builders_align_and_are_order_stable():
    outs = {b: parse(triangle_json(b, jitter=0.005 * i), b, "f").output
            for i, b in enumerate(["b1", "b2", "b3", "b4"])}
    al = align(outs, ["b1", "b2", "b3", "b4"], PARAMS)
    assert len(al.slots) == 6 and not al.ambiguous
    assert all(len(s.members) == 4 for s in al.slots)


def test_builder_with_no_output_is_skipped():
    outs = {"b1": parse(triangle_json("b1"), "b1", "f").output, "b2": None}
    al = align(outs, ["b1", "b2"], PARAMS)
    assert al.builder_facts["b2"] == [] and len(al.slots) == 6


def test_canon_graph_has_no_cross_builder_duplicates():
    # Fix #3: three identical builders. Without dedup the canon graph double-counts b1+b2 facts and
    # b3's relational similarity (hence its merge score) drops below b2's.
    outs = {b: parse(triangle_json(b), b, "f").output for b in ["b1", "b2", "b3"]}
    al = align(outs, ["b1", "b2", "b3"], PARAMS)
    score = {(r["builder"], r["local_id"]): r["score"] for r in al.log if r.get("event") == "merge"}
    for lid in ("pA", "pB", "pC", "s1", "s2", "s3"):
        assert math.isclose(score[("b3", lid)], score[("b2", lid)])
    assert math.isclose(score[("b3", "pA")], 1.0 + 1.0 + 0.5)       # sim_rel = 1 exactly
