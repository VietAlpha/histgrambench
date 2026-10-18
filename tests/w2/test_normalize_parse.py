"""\u00a77.5 normalization, \u00a77.4 parsing (JSON Lines writer replies, OOV, caps, truncation, closure
requirement), prompts."""
import json

from hmb.cards.normalize import dedup_label_key, normalize_label, sim_label
from hmb.cards.parse import parse_builder_output
from hmb.cards.prompts import prompt_hashes, system_prompt
from hmb.ontology import PREDICATES

from w2_scenario import triangle_json

CYR_O, GREEK_A, FULL_A = chr(0x41E), chr(0x391), chr(0xFF21)


def test_strict_normalization():
    assert normalize_label("  A ") == normalize_label("a")
    assert normalize_label("O") == normalize_label("0") == normalize_label(CYR_O)
    assert normalize_label("l") == normalize_label("1") == normalize_label("I")
    assert normalize_label(GREEK_A) == normalize_label("A") == normalize_label(FULL_A)
    assert normalize_label("i") != normalize_label("l")            # lowercase i is not confusable
    assert normalize_label("A\u0301") == normalize_label("\u00c1")  # NFC
    assert normalize_label("\u00c1") != normalize_label("A")        # strict: diacritics kept
    assert normalize_label(chr(0x3B1)) != normalize_label("a")     # Greek alpha is not case-folded to a
    assert dedup_label_key("\u00c1") == dedup_label_key("a")        # section 6.7: case/diacritic-insensitive


def test_sim_label_values():
    assert sim_label("A", " a") == 1.0
    assert sim_label("O", "0") == 1.0                   # strict map folds confusables (fix #2)
    assert sim_label("AO", "A0") == 1.0
    assert sim_label("P", "R") == float("-inf")        # C15: distinct labels never related
    assert sim_label("AP", "AR") == float("-inf")
    assert sim_label("A", None) == 0.5 and sim_label(None, None) == 0.5
    assert sim_label("A", "B") == float("-inf")        # both labelled and unrelated
    assert sim_label("A", "A'") == float("-inf")       # different labels never related


def P(raw, b="b1", fam="f", max_entities=40, max_facts=60):
    return parse_builder_output(raw, b, fam, max_entities=max_entities, max_facts=max_facts)


def jl(*rows) -> str:
    return "\n".join(r if isinstance(r, str) else json.dumps(r) for r in rows)


def test_parse_valid_triangle_derives_existence_and_label_facts():
    r = P(triangle_json("b1"), fam="fam_x")
    assert r.malformed is None and r.output is not None
    preds = [f.predicate for f in r.output.facts]
    assert preds.count("point") == 3 and preds.count("segment") == 3 and preds.count("label") == 3
    assert preds.count("endpoint_of") == 6
    assert r.oov_rate == 0.0 and r.bad_lines == 0
    assert r.output.builder_id == "b1" and r.output.asc_version == "asc-2.0.0"


def test_parse_hostile_inputs():
    for raw in ["", "not json", "[1,2]", '{"fact": "on", "args": []}', "{" * 5000, None]:
        r = P(raw)
        assert r.output is None and r.malformed
    fenced = "Here you go:\n```json\n" + triangle_json("b1") + "\n```\nThanks"
    assert P(fenced).output is not None
    array = "[\n" + ",\n".join(triangle_json("b1").splitlines()) + "\n]"     # an enclosing array is tolerated
    assert len(P(array).output.facts) == len(P(triangle_json("b1")).output.facts)


def test_truncated_reply_keeps_every_complete_line():
    full = triangle_json("b1")
    cut = full[:full.rindex('{"fact"') + 20]              # the last fact line is cut mid-object
    r = P(cut)
    assert r.output is not None and r.bad_lines == 1
    assert [f.predicate for f in r.output.facts].count("endpoint_of") == 5


def test_hard_caps_discard_lines_beyond_the_limits():
    r = P(triangle_json("b1"), max_entities=4, max_facts=2)
    assert len(r.output.entities) == 4 and r.over_cap_entities == 2
    assert [f.predicate for f in r.output.facts].count("endpoint_of") <= 2 and r.over_cap_facts == 4


def test_bare_centre_anchor_and_flat_set_arguments_are_accepted():
    # measured live 2026-09-26 (Kimi K2.5): anchors written as [x, y] and collinear members as flat args
    r = P(jl({"id": "a", "type": "point", "label": "A", "anchor": [0.1, 0.5]},
             {"id": "b", "type": "point", "label": "B", "anchor": [0.5, 0.5]},
             {"id": "c", "type": "point", "label": "C", "anchor": [0.9, 0.5]},
             {"id": "s", "type": "segment"},                                    # no anchor: dropped, noted
             {"fact": "collinear", "args": ["a", "b", "c"]}))
    out = r.output
    assert [e.local_id for e in out.entities] == ["a", "b", "c"] and out.entities[0].anchor.w == 0.0
    assert ("collinear", [["a", "b", "c"]]) in [(f.predicate, f.args) for f in out.facts]
    assert any("anchor" in n for n in r.notes)


def test_oov_and_abstention_at_build_time():
    raw = jl({"id": "p", "type": "point", "label": "P", "label_from_context": True, "anchor": [.1, .1, 0, 0]},
             {"id": "q", "type": "point", "anchor": [.2, .1, 0, 0]},
             {"id": "l", "type": "line", "anchor": [.5, .5, 1, 0]},
             {"id": "c", "type": "circle", "anchor": [.5, .5, .4, .4]},
             {"id": "z", "type": "hexagram", "anchor": [.5, .5, .4, .4]},           # OOV entity type
             {"id": "p", "type": "point", "anchor": [.9, .9, 0, 0]},                # duplicate id
             {"id": "noanchor", "type": "point"},                                   # no anchor
             {"fact": "on", "args": ["p", "l"]},                         # kept
             {"fact": "radius_of", "args": ["l", "c"]},                  # derived-only -> OOV
             {"fact": "concave_up", "args": ["c"]},                      # OOV
             {"fact": "left_of", "args": ["p", "q"]},                    # layout is not a card class -> OOV
             {"fact": "on", "args": ["p", "ghost"]},                     # undeclared entity -> abstained
             {"fact": "on", "args": ["p"]},                              # arity -> abstained
             {"fact": "parallel", "args": ["l", "c"]},                   # convention: circle not linear
             {"fact": "on", "args": ["q", "c"], "label_from_context": True},   # ctx on non-label -> abstained
             {"fact": "label", "args": ["P", "p"]},                      # stated by the entity line: skipped
             "garbage")                                                  # not a JSON object
    r = P(raw, b="b9")
    out = r.output
    assert {e.local_id for e in out.entities} == {"p", "q", "l", "c"}
    kept = {(f.predicate, tuple(f.args), f.context_assisted) for f in out.facts}
    assert ("on", ("p", "l"), False) in kept and ("label", ("P", "p"), True) in kept
    assert not any(f.predicate in ("parallel", "left_of", "radius_of", "concave_up") for f in out.facts)
    assert len([a for a in out.abstentions if a.scope == "fact"]) == 4
    assert r.n_oov == 4 and r.bad_lines == 1           # hexagram, radius_of, concave_up, left_of; garbage
    assert r.n_in_vocab == 11                            # 6 entity lines of known type + 5 in-table fact lines
    assert abs(r.oov_rate - 4 / 15) < 1e-12


def test_prompts_render_card_tables_caps_and_conventions():
    from hmb.cards.prompts import fact_predicates
    from hmb.ontology import CARD_PREDICATES, ENTITY_TYPES
    s = system_prompt("builder_generation_entities") + system_prompt("builder_generation_facts")
    for name in fact_predicates():
        assert f"\n{name} |" in s
    for t in ENTITY_TYPES:
        assert f"\n{t} |" in s
    for name in PREDICATES:
        if name not in CARD_PREDICATES:
            assert f"\n{name} |" not in s, name         # layout predicates never reach a writer
    assert "\nlabel |" not in s                          # labels come from the entity lines
    assert "at most 40" in s and "at most 60" in s and "transversal crossing" in s
    assert "{" + "PREDICATE_TABLE}" not in s and "{CONVENTIONS}" not in s and "{MAX_" not in s
    v = system_prompt("builder_verification")
    assert "abstain" in v and "transversal crossing" in v and "s01 true" in v
    h = prompt_hashes()
    assert set(h) == {"builder_generation_entities", "builder_generation_facts", "builder_verification"} \
        and all(len(x) == 64 for x in h.values())


def test_parametric_entity_without_parameter_is_abstained():
    # fix #14
    raw = jl({"id": "g1", "type": "polygon", "anchor": [.5, .5, .4, .4]},
             {"id": "g2", "type": "polygon", "param": 4, "anchor": [.2, .2, .1, .1]},
             {"id": "k", "type": "curve", "param": "zigzag", "anchor": [.5, .5, .9, .9]})
    out = P(raw).output
    ab = {a.entity: a.reason for a in out.abstentions if a.scope == "entity"}
    assert set(ab) == {"g1", "k"} and "polygon" in ab["g1"] and "curve" in ab["k"]
    assert [(f.predicate, f.args) for f in out.facts] == [("polygon", ["g2", 4])]


def test_prime_family_normalizes_alike():
    # C25: CGX yields U+2032 (math prime) and U+2019 (text prime); all prime spellings must match
    from hmb.cards.normalize import CONFUSABLE_VERSION
    assert CONFUSABLE_VERSION == "confusables-2"
    one = {normalize_label("A" + chr(c)) for c in (0x27, 0x2019, 0x2032, 0x2B9, 0xFF07)}
    assert one == {"a'"}
    assert normalize_label("A" + chr(0x2033)) == normalize_label("A''") == normalize_label("A" + chr(0x2032) * 2)
    assert normalize_label("A" + chr(0x2034)) == normalize_label("A'''") == normalize_label("A" + chr(0x2019) * 3)
    assert normalize_label("A" + chr(0x2033)) != normalize_label("A'")
    assert sim_label("A" + chr(0x2032), "A" + chr(0x2019)) == 1.0


def test_typed_argument_positions_abstain_reversed_facts():
    # C41 (2026-09-26 live writer check): Claude Sonnet 5 wrote endpoint_of(segment, point) and vertex_of(polygon, point)
    # on every fact; the §7.1 typed positions (point first) are shown in the prompt and enforced by the parser.
    A = [.5, .5, .1, .1]
    ents = [{"id": i, "type": t, **({"label": lab} if lab else {}), **({"param": 3} if t == "polygon" else {}),
             "anchor": A}
            for i, t, lab in (("P", "point", "P"), ("Q", "point", "Q"), ("R", "point", "R"), ("s", "segment", None),
                              ("g", "polygon", None), ("c", "circle", None))]
    facts = [{"fact": "endpoint_of", "args": ["P", "s"]}, {"fact": "endpoint_of", "args": ["s", "Q"]},
             {"fact": "vertex_of", "args": ["P", "g"]}, {"fact": "vertex_of", "args": ["g", "Q"]},
             {"fact": "center_of", "args": ["c", "P"]}, {"fact": "on", "args": ["R", "c"]},
             {"fact": "intersect_at", "args": ["s", "c", "R"]}, {"fact": "intersect_at", "args": ["s", "R", "c"]},
             {"fact": "collinear", "args": [["P", "Q", "R"]]}, {"fact": "collinear", "args": [["P", "Q", "s"]]},
             {"fact": "tangent_to", "args": ["s", "c", "unknown"]}]
    res = P(jl(*ents, *facts))
    out = res.output
    assert out is not None
    kept = {(f.predicate, json.dumps(f.args)) for f in out.facts
            if PREDICATES[f.predicate].fclass.value not in ("entity", "label")}
    assert kept == {("endpoint_of", '["P", "s"]'), ("vertex_of", '["P", "g"]'), ("on", '["R", "c"]'),
                    ("intersect_at", '["s", "c", "R"]'), ("collinear", '[["P", "Q", "R"]]'),
                    ("tangent_to", '["s", "c", "unknown"]')}
    reasons = [a.reason for a in out.abstentions if a.scope == "fact"]
    assert len(reasons) == 5 and any("endpoint_of argument 1 must be point, got segment" in r for r in reasons)
    assert res.n_oov == 0                            # in-vocabulary: abstained, not OOV
    s = system_prompt("builder_generation_entities") + system_prompt("builder_generation_facts")
    assert "\nendpoint_of | point, arc|ray|segment |" in s and "never the segment first" in s


def test_oracle_facts_respect_typed_argument_positions():
    # Γ must never state a fact a builder is forbidden to write (ARG_TYPES is checked against the oracle).
    from hmb.config import fixture_thresholds
    from hmb.synthetic.evaluator import oracle_facts
    from hmb.synthetic.families import FAMILIES, generate
    from hmb.synthetic.generate import oracle_graph
    from hmb.ontology import arg_type_violation
    th, bad, n = fixture_thresholds(), [], 0
    for fam in FAMILIES:
        for seed in range(9000000, 9000010):
            spec = generate(fam, seed)
            og = oracle_graph(spec, oracle_facts(spec, th))
            types = {e["entity_id"]: e["type"] for e in og["entities"]}
            for f in og["facts"]:
                n += 1
                v = arg_type_violation(f["predicate"], f["args"], types)
                if v:
                    bad.append((fam, seed, v))
    assert n > 1000 and bad == [], bad[:5]
