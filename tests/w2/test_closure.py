"""§8.2 closure / contradictions / MGS on hand-worked examples (expectations derived by hand)."""
import pytest

from hmb.cards.closure import (canon, contradiction_participants, contradictions, entails,
                               forward_chain, minimal_generating_set)
from hmb.ontology import OntologyError, fact_key


def K(pred, *args):
    return fact_key(pred, list(args))


def test_triangle_with_endpoints_and_edges():
    # Triangle ABC: polygon t with edges s1=AB, s2=BC, s3=CA; builder also said vertex_of and on.
    types = {"A": "point", "B": "point", "C": "point", "s1": "segment", "s2": "segment",
             "s3": "segment", "t": "polygon"}
    F = [K("point", "A"), K("point", "B"), K("point", "C"), K("segment", "s1"), K("segment", "s2"),
         K("segment", "s3"), K("polygon", "t", 3),
         K("endpoint_of", "A", "s1"), K("endpoint_of", "B", "s1"), K("endpoint_of", "B", "s2"),
         K("endpoint_of", "C", "s2"), K("endpoint_of", "C", "s3"), K("endpoint_of", "A", "s3"),
         K("edge_of", "s1", "t"), K("edge_of", "s2", "t"), K("edge_of", "s3", "t"),
         K("vertex_of", "A", "t"), K("on", "A", "s1"), K("on", "B", "t")]
    C = forward_chain(F, types)
    # R1: on(P,s) for each endpoint; R3: vertex_of for every vertex; R2: on(P,t).
    for p, s in [("A", "s1"), ("B", "s1"), ("B", "s2"), ("C", "s2"), ("C", "s3"), ("A", "s3")]:
        assert K("on", p, s) in C
    for p in "ABC":
        assert K("vertex_of", p, "t") in C and K("on", p, "t") in C
    M = set(minimal_generating_set(F, types))
    # Hand result: entity facts, 6 endpoint_of, 3 edge_of. vertex_of/on are all entailed.
    expected = {K("point", "A"), K("point", "B"), K("point", "C"), K("segment", "s1"),
                K("segment", "s2"), K("segment", "s3"), K("polygon", "t", 3),
                K("endpoint_of", "A", "s1"), K("endpoint_of", "B", "s1"), K("endpoint_of", "B", "s2"),
                K("endpoint_of", "C", "s2"), K("endpoint_of", "C", "s3"), K("endpoint_of", "A", "s3"),
                K("edge_of", "s1", "t"), K("edge_of", "s2", "t"), K("edge_of", "s3", "t")}
    assert M == expected
    assert forward_chain(M, types) == C       # the MGS regenerates the closure
    assert contradictions(C, types) == []


def test_parallel_triple_keeps_spanning_tree():
    types = {"a": "line", "b": "line", "c": "line"}
    F = [K("parallel", "a", "b"), K("parallel", "b", "c")]
    C = forward_chain(F, types)
    assert K("parallel", "a", "c") in C and K("parallel", "c", "a") in C  # symmetric canonical
    assert len([f for f in C if f[0] == "parallel"]) == 3
    # All three pairs given: descending key visits (b,c) first -> entailed -> dropped; then (a,c)
    # is not entailed by {(a,b)} alone -> kept; (a,b) kept. Hand result: {(a,b),(a,c)}.
    M = minimal_generating_set(C, types)
    assert set(M) == {K("parallel", "a", "b"), K("parallel", "a", "c")}
    # Four mutually parallel lines -> a spanning tree of 3 edges out of 6.
    types4 = {x: "line" for x in "abcd"}
    F4 = [K("parallel", "a", "b"), K("parallel", "b", "c"), K("parallel", "c", "d")]
    C4 = forward_chain(F4, types4)
    assert len([f for f in C4 if f[0] == "parallel"]) == 6
    M4 = minimal_generating_set(F4, types4)
    assert len(M4) == 3 and forward_chain(M4, types4) == C4
    assert set(M4) == {K("parallel", "a", "b"), K("parallel", "a", "c"), K("parallel", "a", "d")}


def test_no_reflexive_parallel_or_perpendicular():
    types = {"a": "line", "b": "line", "c": "line"}
    C = forward_chain([K("parallel", "a", "b"), K("perpendicular", "c", "a")], types)
    assert all(f[1][0] != f[1][1] for f in C if f[0] in ("parallel", "perpendicular"))
    assert K("perpendicular", "c", "b") in C          # R10


def test_perpendicular_through_parallel_mgs():
    types = {"a": "line", "b": "line", "c": "line"}
    F = [K("parallel", "a", "b"), K("perpendicular", "a", "c"), K("perpendicular", "b", "c")]
    # perpendicular ranks above parallel -> visited first; (b,c) > (a,c) descending -> (b,c) is
    # entailed by perp(a,c) & par(a,b) -> dropped; perp(a,c) no longer entailed -> kept.
    assert set(minimal_generating_set(F, types)) == {K("parallel", "a", "b"), K("perpendicular", "a", "c")}


def test_collinear_subsets_and_between():
    types = {x: "point" for x in "ABCDE"}
    F = [K("collinear", ["A", "B", "C", "D"]), K("collinear", ["A", "B", "C"]),
         K("between", "A", "B", "C")]
    C = forward_chain(F, types)
    cols = [f for f in C if f[0] == "collinear"]
    assert cols == [K("collinear", ["A", "B", "C", "D"])]         # only the maximal set
    assert entails(C, K("collinear", ["D", "B", "A"]))            # sub-tuple entailed
    assert not entails(C, K("collinear", ["A", "B", "E"]))
    M = set(minimal_generating_set(F, types))
    assert M == {K("collinear", ["A", "B", "C", "D"]), K("between", "A", "B", "C")}
    # between alone: collinear({P,Q,R}) is derived and dropped from the MGS
    assert minimal_generating_set([K("between", "C", "B", "A"), K("collinear", ["A", "B", "C"])],
                                  types) == [K("between", "A", "B", "C")]


def test_tangent_known_and_unknown_point():
    types = {"l": "line", "c": "circle", "T": "point"}
    C = forward_chain([K("tangent_to", "l", "c", "T")], types)
    assert K("on", "T", "l") in C and K("on", "T", "c") in C
    C2 = forward_chain([K("tangent_to", "l", "c", "unknown")], types)
    assert not any(f[0] == "on" for f in C2)
    M = minimal_generating_set([K("tangent_to", "l", "c", "T"), K("on", "T", "l"), K("on", "T", "c")], types)
    assert M == [K("tangent_to", "l", "c", "T")]


def test_intersect_at_entails_incidence():
    types = {"x": "line", "y": "line", "P": "point"}
    F = [K("intersect_at", "y", "x", "P"), K("on", "P", "x")]
    assert minimal_generating_set(F, types) == [K("intersect_at", "x", "y", "P")]


def test_center_inside_dropped_and_radius_derived_only():
    # §11.13(a): inside(O,c) is dropped when center_of(O,c) is kept.
    types = {"O": "point", "P": "point", "c": "circle", "s": "segment"}
    F = [K("circle", "c"), K("point", "O"), K("point", "P"), K("segment", "s"),
         K("center_of", "O", "c"), K("inside", "O", "c"), K("endpoint_of", "O", "s"),
         K("endpoint_of", "P", "s"), K("on", "P", "c")]
    C = forward_chain(F, types)
    assert ("radius_of", ("s", "c")) in C
    M = minimal_generating_set(F, types)
    assert K("inside", "O", "c") not in M and K("center_of", "O", "c") in M
    assert not any(f[0] == "radius_of" for f in M)
    assert K("on", "P", "c") in M              # premise of radius, not entailed
    # center of an arc does not entail inside (R4 is circle-only)
    C_arc = forward_chain([K("center_of", "O", "a")], {"O": "point", "a": "arc"})
    assert K("inside", "O", "a") not in C_arc
    # circle-ness from the fact alone (no entity_types)
    assert K("inside", "O", "c") in forward_chain([K("circle", "c"), K("center_of", "O", "c")], {})


def test_exclusion_table_each_pattern():
    t = {"P": "point", "Q": "point", "x": "segment", "l": "line", "m": "line", "c": "circle",
         "A": "point", "B": "point", "T": "point", "U": "point"}
    cases = [
        ([K("inside", "P", "x"), K("on", "P", "x")], 1),
        ([K("outside", "P", "x"), K("endpoint_of", "P", "x")], 1),     # derived on(P,x)
        ([K("parallel", "l", "m"), K("intersect_at", "m", "l", "P")], 1),
        ([K("tangent_to", "l", "c", "T"), K("intersect_at", "l", "c", "T"),
          K("intersect_at", "c", "l", "U")], 2),
        ([K("tangent_to", "l", "c", "T"), K("intersect_at", "l", "c", "T")], 0),   # one point is fine
        ([K("left_of", "A", "B"), K("left_of", "B", "A")], 1),
        ([K("above", "A", "B"), K("above", "B", "A")], 1),
        ([K("same_side", "P", "Q", "l"), K("opposite_sides", "Q", "P", "l")], 1),
        ([K("label", "A", "P"), K("label", "B", "P")], 1),
        ([K("label", "A", "P"), K("label", " a", "P")], 0),     # same normalized label
        ([K("label", "O", "P"), K("label", "0", "P")], 0),      # confusable map
        ([K("label", "A", "P"), K("label", "A", "Q")], 0),      # same text on two entities: two facts
        ([K("left_of", "A", "B"), K("above", "B", "A")], 0),
    ]
    for facts, n in cases:
        C = forward_chain(facts, t)
        assert len(contradictions(C, t)) == n, facts
    # center_of & circle -> inside, which contradicts an asserted on(O,c)
    C = forward_chain([K("circle", "c"), K("center_of", "P", "c"), K("on", "P", "c")], t)
    assert contradictions(C, t) == [(K("inside", "P", "c"), K("on", "P", "c"))]


def test_contradiction_participants_attribute_derived_sides():
    t = {"P": "point", "c": "circle", "l": "line", "m": "line", "n": "line"}
    F = [K("circle", "c"), K("center_of", "P", "c"), K("on", "P", "c")]
    part, pairs = contradiction_participants(F, t)
    # derived inside(P,c) rests on center_of(P,c) and circle(c)
    assert part == {K("center_of", "P", "c"), K("circle", "c"), K("on", "P", "c")}
    F2 = [K("parallel", "l", "m"), K("parallel", "m", "n"), K("intersect_at", "l", "n", "P")]
    part2, _ = contradiction_participants(F2, t)
    assert part2 == set(F2)


def test_polygon_parameters_and_hostile_inputs():
    assert canon(("polygon", ("t", 4))) == ("polygon", ("t", 4))
    with pytest.raises(OntologyError):
        forward_chain([("not_a_predicate", ("a",))], {})
    with pytest.raises(OntologyError):
        forward_chain([("on", ("a",))], {})
    # degenerate between is ignored by R7, not a crash
    assert not any(f[0] == "collinear" for f in forward_chain([K("between", "A", "A", "B")], {}))


def test_mgs_deterministic_under_input_permutation():
    types = {x: "line" for x in "abcde"}
    F = [K("parallel", a, b) for a, b in ["ab", "bc", "cd", "de", "ae", "bd"]] + [K("perpendicular", "a", "e")]
    m1 = minimal_generating_set(F, types)
    m2 = minimal_generating_set(list(reversed(F)), types)
    assert m1 == m2
    # a,e both perpendicular and parallel is not an E-violation in asc-1.0.0 (not in table E)
    assert forward_chain(m1, types) == forward_chain(F, types)


def test_mgs_property_random_graphs():
    import random
    rng = random.Random(7)
    pts, segs, lines, circs = ["P1", "P2", "P3", "P4", "P5"], ["s1", "s2", "s3"], ["l1", "l2", "l3", "l4"], ["c1"]
    types = {**{p: "point" for p in pts}, **{s: "segment" for s in segs},
             **{x: "line" for x in lines}, **{c: "circle" for c in circs}}
    gens = [
        lambda: K("endpoint_of", rng.choice(pts), rng.choice(segs)),
        lambda: K("on", rng.choice(pts), rng.choice(segs + lines + circs)),
        lambda: K("parallel", *rng.sample(lines, 2)),
        lambda: K("perpendicular", *rng.sample(lines, 2)),
        lambda: K("intersect_at", *rng.sample(lines + circs, 2), rng.choice(pts)),
        lambda: K("tangent_to", rng.choice(lines), "c1", rng.choice(pts + ["unknown"])),
        lambda: K("between", *rng.sample(pts, 3)),
        lambda: K("collinear", rng.sample(pts, rng.choice([3, 4]))),
        lambda: K("center_of", rng.choice(pts), "c1"),
        lambda: K("inside", rng.choice(pts), "c1"),
    ]
    for _ in range(150):
        F = [rng.choice(gens)() for _ in range(rng.randint(1, 12))]
        C = forward_chain(F, types)
        M = minimal_generating_set(F, types)
        assert forward_chain(M, types) == C
        assert set(M) <= C
        for f in M:     # irredundant: no MGS fact is entailed by the others
            assert f not in forward_chain(set(M) - {f}, types)
        assert not any(f[0] == "radius_of" for f in M)


def test_mgs_redundancy_is_r8_aware():
    # fix #9: sub-tuples are dropped against the maximal set, whatever the input order
    types = {x: "point" for x in "ABCDE"}
    F = [K("collinear", ["A", "B", "C"]), K("collinear", ["A", "B", "C", "D"]),
         K("collinear", ["A", "B", "C", "D", "E"]), K("collinear", ["B", "D", "E"])]
    assert minimal_generating_set(F, types) == [K("collinear", ["A", "B", "C", "D", "E"])]
    assert minimal_generating_set(list(reversed(F)), types) == [K("collinear", ["A", "B", "C", "D", "E"])]


_TIE_SCRIPT = """
from hmb.cards.closure import minimal_generating_set
from hmb.cards.consensus import assign_fact_ids
from hmb.ontology import fact_key
types = {"B": "point", "s": "segment", "t": "segment", "C": "point"}
F = [fact_key("endpoint_of", ["B", "s"]), fact_key("endpoint_of", ["s", "B"]),
     fact_key("endpoint_of", ["C", "t"]), fact_key("endpoint_of", ["t", "C"])]
M = minimal_generating_set(F, types)
print(repr(M), repr(sorted(assign_fact_ids(M).items())))
"""


def test_mgs_order_independent_of_hash_seed():
    # C40 (2026-09-26, live writer check): a writer's reversed endpoint_of(s, B) ties endpoint_of(B, s) on the §8.2
    # key; the tie fell back to set order, which depends on PYTHONHASHSEED, so a rerun in a new process re-asked the
    # verification calls and conflicted with the frozen card. Every process must give the same order and ids.
    import os
    import subprocess
    import sys
    outs = set()
    for seed in ("0", "1", "2", "3", "4", "5", "6", "7"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run([sys.executable, "-c", _TIE_SCRIPT], env=env, capture_output=True, text=True, check=True)
        outs.add(r.stdout)
    assert len(outs) == 1, outs
