"""Judge serialization contract (repair 2026-09-26, defect B; docs/CONTRACTS.md "Judge serialization").

Every entity a judge statement names has a unique, deterministic identifier; labels stay primary
where unique; the ENTITIES block defines identifiers without leaking statements under test, model
facts, ineligible facts or verdicts; distinct facts never share wording."""
from __future__ import annotations

import random
import re

import pytest

from hmb.adjudication.serialize import (EntityNames, StatementCollision, check_unique, fact_statement,
                                        names_for_card, serializer_identity)
from hmb.schemas import BuilderAnchor, CanonEntity

from w4_fixtures import card, fact, triangle_card


def _solids_card():
    """Two unlabelled segments on labelled points, two anchored free points, two points printed "A"."""
    ents = [("p_A", "point", "A"), ("p_B", "point", "B"), ("p_C", "point", "C"),
            ("s_x", "segment"), ("s_y", "segment"), ("q_1", "point"), ("q_2", "point"),
            ("d_1", "point", "M"), ("d_2", "point", "M"), ("k", "curve")]
    fs = [fact("r1", "endpoint_of", ["p_A", "s_x"]), fact("r2", "endpoint_of", ["p_B", "s_x"]),
          fact("r3", "endpoint_of", ["p_B", "s_y"]), fact("r4", "endpoint_of", ["p_C", "s_y"]),
          fact("m1", "equal_length", ["s_x", "s_y"]), fact("m2", "curve", ["k", "line"]),
          fact("m3", "measure_label", ["5", "s_x", "length"]), fact("m4", "measure_label", ["5", "s_y", "length"]),
          fact("h1", "on", ["q_1", "s_y"], eligible=False)]
    c = card("hmb_0000000000000002", ents, fs)
    anchors = {"q_1": (0.8, 0.2), "q_2": (0.1, 0.5), "d_1": (0.6, 0.6), "d_2": (0.3, 0.9)}
    return c.model_copy(update={"entities": [e.model_copy(update={"anchor": BuilderAnchor(x=anchors[e.entity_id][0],
                                                                                         y=anchors[e.entity_id][1],
                                                                                         w=0.0, h=0.0)})
                                             if e.entity_id in anchors else e for e in c.entities]})


def _names(c):
    n = names_for_card(c)
    return {e.entity_id: n.describe(e.entity_id) for e in c.entities}


def test_unique_labels_stay_primary_and_others_get_typed_opaque_ids():
    got = _names(_solids_card())
    assert got["p_A"] == "point “A”" and got["p_B"] == "point “B”"
    assert got["s_x"] == "segment S1" and got["s_y"] == "segment S2"      # sorted by incidence (A < B)
    assert got["k"] == "curve CV1"
    assert {got["q_1"], got["q_2"], got["d_1"], got["d_2"]} == {"point P1", "point P2", "point P3", "point P4"}
    assert len(set(got.values())) == len(got)                                # unique within the item


def test_duplicate_labels_get_opaque_ids_and_a_defining_line():
    c = _solids_card()
    n = names_for_card(c)
    assert n.describe("d_1") != n.describe("d_2") and "“M”" not in n.describe("d_1")
    line = n.definition("d_1")
    assert "printed with the label “M”, which 2 points share" in line


def test_round_trip_handle_to_entity():
    c = _solids_card()
    n = names_for_card(c)
    for e in c.entities:
        assert n.by_handle[(e.type, n.handle[e.entity_id])] == e.entity_id      # identity = (type, handle)


def test_determinism_and_invariance_to_order_and_entity_ids():
    c = _solids_card()
    base = _names(c)
    rnd = random.Random(7)
    for _ in range(5):
        ents, fs = list(c.entities), list(c.facts)
        rnd.shuffle(ents), rnd.shuffle(fs)
        assert _names(c.model_copy(update={"entities": ents, "facts": fs})) == base
    # renaming canonical ids (the builder/model output order that produced them) keeps every name
    ren = {e.entity_id: f"z{k}" for k, e in enumerate(reversed(c.entities))}
    ents = [e.model_copy(update={"entity_id": ren[e.entity_id]}) for e in c.entities]
    fs = [f.model_copy(update={"args": [ren.get(a, a) if isinstance(a, str) else a for a in f.args]}) for f in c.facts]
    renamed = _names(c.model_copy(update={"entities": ents, "facts": fs}))
    assert {ren[k]: v for k, v in base.items()} == renamed


def test_context_defines_every_identifier_without_leaking():
    c = _solids_card()
    n = names_for_card(c)
    under_test = [f for f in c.facts if f.fact_id in ("m1", "m3")]
    ctx = n.context({"s_x", "s_y"}, exclude={"m1", "m3"})
    for handle in re.findall(r"\b(?:S|P|CV)\d+\b", " ".join(fact_statement(f, n) for f in under_test)):
        assert re.search(rf"^{handle}:", ctx, re.M), handle
    assert "S1: a segment with no unique printed label (endpoint: point “A”, point “B”)" in ctx
    assert "equal" not in ctx and "“5”" not in ctx and "line" not in ctx      # no statement or model fact leaks
    assert "P" not in re.sub(r"point|printed", "", ctx)                        # ineligible on(q_1, s_y) not used


def test_context_drops_a_geo_fact_that_is_itself_under_test():
    c = _solids_card()
    n = names_for_card(c)
    ctx = n.context({"s_x"}, exclude={"r1"})                  # endpoint_of(A, s_x) is being judged
    assert "“A”" not in ctx.split("S1:")[1].splitlines()[0]
    assert n.describe("s_x") == "segment S1"                  # the identifier itself never changes


def test_position_rank_when_nothing_else_distinguishes():
    n = names_for_card(_solids_card())
    lines = [n.definition(e) for e in ("q_1", "q_2")]
    assert any("counted from the left" in ln for ln in lines)


def test_distinct_facts_with_identical_wording_are_refused():
    c = _solids_card()
    n = names_for_card(c)
    check_unique([f for f in c.facts if f.fact_id in ("m3", "m4")], n)        # S1 vs S2: distinct wording

    class Collapse(EntityNames):
        def describe(self, e):
            return "a segment"
    bad = Collapse(c.entities, c.facts)
    with pytest.raises(StatementCollision):
        check_unique([f for f in c.facts if f.fact_id in ("m3", "m4")], bad)


def test_previous_ambiguity_is_gone_on_the_triangle():
    # before the repair both of these rendered as "an unlabelled segment ..." (defect 4 of 2026-09-26)
    c = triangle_card([fact("m1", "equal_angle", ["s_1", "s_2"]), fact("m2", "equal_angle", ["s_2", "s_3"])])
    n = names_for_card(c)
    t = [fact_statement(f, n) for f in c.facts if f.fact_id in ("m1", "m2")]
    assert t[0] != t[1] and "unlabelled" not in " ".join(t)


def test_same_label_on_entities_of_different_types_is_not_a_collision():
    # measured on a real crop (2026-09-26): a point and a line both printed "V" aborted the whole cell
    c = card("hmb_0000000000000003", [("p_V", "point", "V"), ("l_V", "line", "V")],
             [fact("r1", "on", ["p_V", "l_V"])])
    n = names_for_card(c)
    assert {n.describe("p_V"), n.describe("l_V")} == {"point “V”", "line “V”"}
    assert fact_statement(c.facts[0], n).count("“V”") == 2


def test_serializer_identity_is_versioned_and_hashed():
    ident = serializer_identity()
    assert ident["version"] == "judge-serializer-3" and len(ident["source_sha256"]) == 64


def test_card_without_anchors_or_labels_still_unique():
    c = card("hmb_0000000000000003", [("a", "segment"), ("b", "segment"), ("c", "segment")], [])
    got = _names(c)
    assert sorted(got.values()) == ["segment S1", "segment S2", "segment S3"]
    assert CanonEntity  # noqa: B018 (schema import kept for the anchor helper above)


def test_native_verdict_schema_only_when_configured(fake, lay, images):
    from hmb.adjudication.judges import JudgeCtx, ask_once
    from hmb.vocab import JudgeQuestion
    from w4_fixtures import judges, thresholds
    ctx = JudgeCtx(lay, thresholds(), None, False, None, "FINAL", "cfg")
    j = judges()[0]
    ask_once(j, JudgeQuestion.Q1, "STATEMENTS:\ns01: x", [images[0]], ["s01"], ctx)
    assert fake.calls[-1].response_schema is None                     # prompt-constrained judge
    ask_once(j.model_copy(update={"structured_output": "native_json_schema"}), JudgeQuestion.Q1,
             "STATEMENTS:\ns01: x", [images[0]], ["s01"], ctx)
    sch = fake.calls[-1].response_schema
    assert sch is not None
    item = sch["schema"]["properties"]["answers"]["items"]
    assert item["properties"]["id"]["enum"] == ["s01"] and item["properties"]["answer"] == {"type": "boolean"}
    assert sch["schema"]["required"] == ["schema", "answers"] and "refusal" not in sch["schema"]["properties"]


def test_indistinguishable_twins_are_declared_not_ranked():
    # two unlabelled lines through the same labelled point with the same anchor (symmetric asymptotes)
    c = card("hmb_0000000000000004", [("p_O", "point", "O"), ("l_a", "line"), ("l_b", "line"), ("l_c", "line")],
             [fact("r1", "on", ["p_O", "l_a"]), fact("r2", "on", ["p_O", "l_b"])])
    same = BuilderAnchor(x=0.5, y=0.5, w=0.8, h=0.8)
    c = c.model_copy(update={"entities": [e.model_copy(update={"anchor": same}) if e.entity_id in ("l_a", "l_b") else e
                                          for e in c.entities]})
    n = names_for_card(c)
    assert n.twins("l_a") == ["l_b"] and n.twins("l_c") == [] and n.twins("p_O") == []
    line = n.definition("l_a")
    assert "does not distinguish it from line " + n.handle["l_b"] in line and "counted from the left" not in line
