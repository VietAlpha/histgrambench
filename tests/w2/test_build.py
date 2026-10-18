"""build_card orchestration with an injected FAKE provider (engineering evidence only, fixture profile)."""
import json
import re

import numpy as np
import pytest

from hmb.cards import build_card, check_builder_roster, check_primary_disjoint
from hmb.cards.build import CARD_CALL_SCOPE, FakeResponseRefused, RosterError
from hmb.config import CalibrationMissing, production_template
from hmb.roster import LOCKED_CONTESTANTS
from hmb.schemas import ModelResponse, ModelSpec
from hmb.store import ImmutableConflict, Layout, read_json

from w2_scenario import FAM, ORDER, as_production, card_th, triangle_json

PROVIDER = {"b1": "bedrock", "b2": "bedrock", "b3": "bedrock_mantle", "b4": "bedrock"}
ORDER3 = ["b1", "b3"]          # §7.3 as amended 2026-09-26: 2 builders in 2 families (x, y); name kept


def specs(fams=FAM, role="builder"):
    return [ModelSpec(model_id=b, provider=PROVIDER[b], family=fams[b], role=role) for b in ORDER3]


EXTRA = {"b1": [{"fact": "perpendicular", "args": ["s1", "s2"]}]}     # b1 states one fact b3 does not


class FakeCall:
    """Stand-in for hmb.providers.call_model. Every response is marked fake=True."""

    def __init__(self, vote="true", broken=(), stop_reason=None):
        self.reqs = []
        self.vote = vote
        self.broken = set(broken)
        self.stop_reason = stop_reason

    def __call__(self, req, layout, ledger, *, allow_live, provider_override=None):
        assert allow_live is False
        self.reqs.append(req)
        b = req.spec.model_id
        if req.purpose == "card_generation":
            text = "garbage{" if b in self.broken else triangle_json(b, extra_facts=EXTRA.get(b, ()))
        else:
            sids = re.findall(r"^(s\d+):", req.text, re.M)
            text = "\n".join(f"{s} {self.vote}" for s in sids)
        return ModelResponse(text=text, model_version_string=f"{b}-fake", cache_key=f"k{len(self.reqs)}",
                             fake=True, stop_reason=self.stop_reason)


@pytest.fixture
def crop(tmp_path):
    p = tmp_path / "crop.png"
    import cv2
    cv2.imencode(".png", np.full((32, 32), 255, np.uint8))[1].tofile(str(p))
    return p


def test_build_card_end_to_end_fixture(tmp_path, crop):
    lay = Layout(tmp_path / "data")
    call = FakeCall()
    th = card_th()
    card = build_card("hmb_0000000000000001", crop, [{"text": "A", "box": [0, 0, .1, .1]}], "Let ABC be...",
                      builders=specs(), layout=lay, th=th, ledger=None, allow_live=False, call=call)
    # 2 staged generation calls per writer (entities, then facts), then each writer checks the whole union
    # (16 statements, one chunk of <= 20)
    assert [(r.purpose, r.spec.model_id) for r in call.reqs] == [
        ("card_generation", "b1"), ("card_generation", "b1"), ("card_generation", "b3"), ("card_generation", "b3"),
        ("card_verification", "b1"), ("card_verification", "b3")]
    assert all(r.cache_scope == CARD_CALL_SCOPE and r.max_tokens == 4096 for r in call.reqs)
    assert "perpendicular" in call.reqs[5].text and call.reqs[5].text.count("\ns16:") == 1
    assert "Let ABC be" in call.reqs[0].text and "JSON Lines" in call.reqs[0].system
    ent, fac = call.reqs[0], call.reqs[1]
    assert "no fact lines" in ent.system and "FACT TABLE" not in ent.system and "ENTITIES:" not in ent.text
    assert "FACT TABLE" in fac.system and "ENTITIES:\n" in fac.text and '"id": "pA"' in fac.text
    assert card.profile == "fixture" and card.asc_version.startswith("asc-2.0.0+")
    assert len(card.facts) == 16 and all(r.eligible for r in card.facts)
    perp = next(r for r in card.facts if r.predicate == "perpendicular")
    assert perp.votes == {"b1": "generated", "b3": "true"}
    assert card.builder_order == ORDER3 and card.family_map == {b: FAM[b] for b in ORDER3}
    votes = read_json(lay.card_votes("hmb_0000000000000001"))
    assert "planted" not in votes                         # no planted controls (simplification 2026-09-25)
    assert [len(c["statements"]) for c in votes["verification"]["b1"]] == [16]
    man = read_json(lay.card_manifest(card.card_uid, card.item_id))
    assert man["asc_version"] == card.asc_version and man["malformed_builders"] == []
    assert man["oov_rate"] == {b: 0.0 for b in ORDER3} and man["truncated_calls"] == []
    assert man["calls"] == {"b1": {"card_generation": 2, "card_verification": 1},
                            "b3": {"card_generation": 2, "card_verification": 1}}
    assert man["asc_hash_inputs"]["generation_protocol"] == "staged-entities-facts-1"
    assert man["empty_truncated_generation_calls"] == []
    assert "item_gates" not in man["asc_hash_inputs"] and man["asc_hash_inputs"]["caps"]["card_verify_chunk_size"] == 20
    assert read_json(lay.card("hmb_0000000000000001"))["facts"][0]["class"] == "entity"
    # identical rebuild is a no-op; a different outcome for the same item is refused (frozen cards)
    build_card("hmb_0000000000000001", crop, [{"text": "A", "box": [0, 0, .1, .1]}], "Let ABC be...",
               builders=specs(), layout=lay, th=th, ledger=None, allow_live=False, call=FakeCall())
    with pytest.raises(ImmutableConflict):
        build_card("hmb_0000000000000001", crop, [{"text": "A", "box": [0, 0, .1, .1]}], "Let ABC be...",
                   builders=specs(), layout=lay, th=th, ledger=None, allow_live=False, call=FakeCall("abstain"))


def test_malformed_builder_is_recorded_not_fatal(tmp_path, crop):
    lay = Layout(tmp_path / "d")
    card = build_card("hmb_0000000000000002", crop, [], None, builders=specs(), layout=lay, th=card_th(),
                      ledger=None, allow_live=False, call=FakeCall(broken={"b3"}))
    man = read_json(lay.card_manifest(card.card_uid, card.item_id))
    assert man["malformed_builders"] == ["b3"] and man["oov_rate"]["b3"] is None
    assert len(card.facts) == 16
    # b3's generation contributed nothing, so it verifies every one of b1's facts (the card survives)
    assert all(len(s.members) == 1 for s in card.entities)
    assert man["verification_statements"] == {"b1": 16, "b3": 16} and all(r.eligible for r in card.facts)


def test_a_writer_can_veto_its_own_fact(tmp_path, crop):
    """Self-veto (the preregistered fallback): a writer voting false on a fact it generated makes it ineligible."""
    card = build_card("hmb_0000000000000006", crop, [], None, builders=specs(), layout=Layout(tmp_path / "v"),
                      th=card_th(), ledger=None, allow_live=False, call=FakeCall(vote="false"))
    assert not any(r.eligible for r in card.facts)
    assert all(r.votes == {"b1": "false", "b3": "false"} for r in card.facts)


def test_truncated_calls_are_recorded_and_salvaged(tmp_path, crop):
    """A reply cut at max_tokens keeps its complete lines; the stop reason is recorded per call."""
    lay = Layout(tmp_path / "t")
    card = build_card("hmb_0000000000000004", crop, [], None, builders=specs(), layout=lay, th=card_th(),
                      ledger=None, allow_live=False, call=FakeCall(stop_reason="max_tokens"))
    man = read_json(lay.card_manifest(card.card_uid, card.item_id))
    gen = [{"builder_id": b, "purpose": "card_generation", "stage": st, "attempt": 0, "usable_lines": n}
           for b in ("b1", "b3") for st, n in (("entities", 6), ("facts", 6 + (b == "b1")))]
    assert man["truncated_calls"] == gen + [
                                      {"builder_id": "b1", "purpose": "card_verification", "chunk": 0},
                                      {"builder_id": "b3", "purpose": "card_verification", "chunk": 0}]
    assert len(card.facts) == 16 and all(r.eligible for r in card.facts)
    assert man["empty_truncated_generation_calls"] == []      # every cut reply still had complete lines


class StagedCall(FakeCall):
    """Scripted staged generation: `gen[(builder, stage, attempt)] = (text, stop_reason)`; verification votes true."""

    def __init__(self, gen):
        super().__init__()
        self.gen = gen
        self.seen: dict = {}

    def __call__(self, req, layout, ledger, *, allow_live, provider_override=None):
        if req.purpose != "card_generation":
            return super().__call__(req, layout, ledger, allow_live=allow_live)
        self.reqs.append(req)
        b = req.spec.model_id
        st = "facts" if "FACT TABLE" in (req.system or "") else "entities"
        a = self.seen.get((b, st), 0)
        self.seen[(b, st)] = a + 1
        text, stop = self.gen.get((b, st, a), (triangle_json(b), "end_turn"))
        return ModelResponse(text=text, model_version_string=f"{b}-fake", cache_key=f"k{len(self.reqs)}",
                             fake=True, stop_reason=stop, output_tokens=4096 if stop == "max_tokens" else 100)


def test_empty_truncated_generation_is_retried_once_then_degrades(tmp_path, crop):
    """The 2026-09-26 failure: a dense-crop generation turn spent the whole allowance and returned no line. The entity
    stage is asked once more with a different, narrower request; when that is empty too the writer has no entity line,
    gets no fact call, still verifies, and the other writer's card survives."""
    call = StagedCall({("b3", "entities", 0): ("", "max_tokens"), ("b3", "entities", 1): ("", "max_tokens")})
    lay = Layout(tmp_path / "e")
    card = build_card("hmb_0000000000000007", crop, [], None, builders=specs(), layout=lay, th=card_th(),
                      ledger=None, allow_live=False, call=call)
    b3 = [r for r in call.reqs if r.spec.model_id == "b3" and r.purpose == "card_generation"]
    assert len(b3) == 2 and b3[0].text != b3[1].text and "RETRY" in b3[1].text and "at most 20 lines" in b3[1].text
    assert all("FACT TABLE" not in r.system for r in b3)                    # no fact call without entities
    man = read_json(lay.card_manifest(card.card_uid, card.item_id))
    assert man["calls"]["b3"] == {"card_generation": 2, "card_verification": 1}
    assert man["malformed_builders"] == ["b3"] and man["verification_statements"]["b3"] == 15
    assert [(t["stage"], t["attempt"]) for t in man["empty_truncated_generation_calls"]] == [("entities", 0),
                                                                                           ("entities", 1)]
    votes = read_json(lay.card_votes(card.item_id))
    assert votes["generation"]["b3"]["fact_stage"] == "skipped: no entity line"
    assert [c["stop_reason"] for c in votes["generation"]["b3"]["calls"]] == ["max_tokens", "max_tokens"]
    assert len(card.facts) == 15 and all(r.eligible for r in card.facts)   # b1's card, confirmed by b3's votes


def test_retry_recovers_and_partial_fact_reply_is_salvaged_without_dangling_ids(tmp_path, crop):
    """Entity stage: empty cut reply, then the retry answers. Fact stage: cut mid-line; complete lines are kept,
    an entity line in the fact reply is ignored, and a fact on an undeclared id is abstained, never dangling."""
    ents = triangle_json("b1").split("\n")[:6]
    facts = ['{"fact": "endpoint_of", "args": ["pA", "s1"]}', '{"id": "pX", "type": "point", "anchor": [0.1, 0.1, 0, 0]}',
             '{"fact": "endpoint_of", "args": ["pX", "s1"]}', '{"fact": "endpoint_of", "args": ["pB", "s1"]}',
             '{"fact": "endpoint_of", "ar']
    call = StagedCall({("b1", "entities", 0): ("", "max_tokens"),
                       ("b1", "entities", 1): ("\n".join(ents), "end_turn"),
                       ("b1", "facts", 0): ("\n".join(facts), "max_tokens")})
    lay = Layout(tmp_path / "s")
    card = build_card("hmb_0000000000000008", crop, [], None, builders=specs(), layout=lay, th=card_th(),
                      ledger=None, allow_live=False, call=call)
    votes = read_json(lay.card_votes(card.item_id))
    g = votes["generation"]["b1"]
    assert [(c["stage"], c["attempt"], c["usable_lines"]) for c in g["calls"]] == [
        ("entities", 0, 0), ("entities", 1, 6), ("facts", 0, 3)]
    assert g["bad_lines"] == 1 and any("other stage" in n for n in g["notes"])
    kept = [(f["predicate"], f["args"]) for f in g["parsed"]["facts"] if f["predicate"] == "endpoint_of"]
    assert kept == [("endpoint_of", ["pA", "s1"]), ("endpoint_of", ["pB", "s1"])]
    assert "pX" not in {e["local_id"] for e in g["parsed"]["entities"]}
    assert any("undeclared entity ['pX']" in a["reason"] for a in g["parsed"]["abstentions"])
    man = read_json(lay.card_manifest(card.card_uid, card.item_id))
    assert [(t["stage"], t["usable_lines"]) for t in man["truncated_calls"] if t["builder_id"] == "b1"] == [
        ("entities", 0), ("facts", 3)]
    assert [t["attempt"] for t in man["empty_truncated_generation_calls"]] == [0]


def test_generation_cache_identity_is_per_stage_and_per_writer(tmp_path, crop):
    """A stage's request depends only on what can change its answer: the entity request on the crop, OCR and
    context; the fact request additionally on that writer's own entity lines, never on the other writer."""
    args = ("hmb_0000000000000009", crop, [], None)
    a, b = FakeCall(), StagedCall({("b3", "entities", 0): (triangle_json("b3", labels=("P", "Q", "R")), "end_turn")})
    build_card(*args, builders=specs(), layout=Layout(tmp_path / "a"), th=card_th(), ledger=None, allow_live=False,
               call=a)
    build_card(*args, builders=specs(), layout=Layout(tmp_path / "b"), th=card_th(), ledger=None, allow_live=False,
               call=b)
    ga = [r for r in a.reqs if r.purpose == "card_generation"]
    gb = [r for r in b.reqs if r.purpose == "card_generation"]
    same = [x.model_dump(mode="json") == y.model_dump(mode="json") for x, y in zip(ga, gb)]
    assert same == [True, True, True, False]              # only b3's fact request (its own entities) differs


def test_verification_is_chunked_at_the_registered_size(tmp_path, crop):
    lay = Layout(tmp_path / "k")
    call = FakeCall(broken={"b3"})
    card = build_card("hmb_0000000000000005", crop, [], None, builders=specs(), layout=lay,
                      th=card_th(card_verify_chunk_size=5), ledger=None, allow_live=False, call=call)
    sizes = [r.text.count("\ns") for r in call.reqs if r.purpose == "card_verification"]
    assert sizes == [5, 5, 5, 1] * 2 and all(r.eligible for r in card.facts)


def test_roster_rules():
    two = {"b1": "fam_x", "b3": "fam_x"}       # 2 builders but only 1 family
    with pytest.raises(RosterError, match="2 builders in 2 distinct families"):
        check_builder_roster(specs(two))
    with pytest.raises(RosterError):
        check_builder_roster(specs(role="judge"))
    check_builder_roster(specs())
    # D17 as amended 2026-09-26: cards-build applies the roster-check rule, exactly 2 builders (1 or 3 refused)
    with pytest.raises(RosterError, match="2 builders"):
        check_builder_roster(specs()[:1])
    third = ModelSpec(model_id="b5", provider="bedrock", family="fam_w", role="builder")
    with pytest.raises(RosterError, match="2 builders"):
        check_builder_roster(specs() + [third])
    judges =[ModelSpec(model_id="j1", provider="bedrock", family="fam_y", role="judge")]
    with pytest.raises(RosterError):
        check_primary_disjoint(specs(), judges)
    check_primary_disjoint(specs(), [ModelSpec(model_id="j2", provider="bedrock", family="fam_q", role="judge")])


MANTLE = "https://bedrock-mantle.us-east-1.api.aws"


def _approved_shape_roster() -> dict:
    """The shape of the 2026-09-23 roster decision (ids are placeholders, not the owner's file): one
    strict gatekeeper, 3 writers in 3 families at medium, 3 graders at high (one on Mantle), and the
    contestants locked on 2026-09-25 (three vision models and the text-only DeepSeek V3.1 on Mantle)."""
    def m(mid, prov, fam, **kw):
        return {"model_id": mid, "provider": prov, "family": fam, "snapshot": mid, "unit_price_id": "p-" + mid, **kw}
    conv = lambda e: {"additional_model_request_fields": {"effort": e}}  # noqa: E731 (owner pins the real form)
    return {"version": "roster-1", "inclusion_rule_date": "2026-09-23",
            "verifier": [m("gk", "bedrock", "openai", effort="high", decoding=conv("high"))],
            "builder": [m("w1", "bedrock", "anthropic", effort="medium", decoding=conv("medium")),
                        m("w2", "bedrock", "openai", effort="medium", decoding=conv("medium")),
],
            "judge": [m("g1", "bedrock", "qwen", effort="high", decoding=conv("high")),
                      m("google.gemma-4-31b", "bedrock_mantle", "google", effort="high",
                        endpoint=MANTLE + "/openai/v1"),
                      m("g3", "bedrock", "mistral", effort="high", decoding=conv("high"))],
            "contestant": [m(r.model_id, r.provider, r.family, input_mode="vision" if r.image_input else "text_only",
                             **({"endpoint": r.endpoint} if r.endpoint else {})) for r in LOCKED_CONTESTANTS]}


def test_roster_check_gatekeeper_only_three_writers_effort(tmp_path):
    import copy

    import yaml

    from hmb.roster import RosterError, check_roster, load_roster

    def check(d):
        f = tmp_path / "roster.yaml"
        f.write_text(yaml.safe_dump(d), encoding="utf-8")
        return check_roster(load_roster(f))
    good = _approved_shape_roster()
    assert check(good) == []
    r = load_roster(tmp_path / "roster.yaml")
    assert set(r) == {"builder", "judge", "contestant", "verifier"}
    assert r["contestant"][-1].input_mode == "text_only" and r["judge"][1].endpoint.endswith("/openai/v1")
    for role in ("classifier", "metadata", "embedder"):            # cut by the 2026-09-23 role audit
        with pytest.raises(RosterError, match="removed"):
            check(dict(copy.deepcopy(good), **{role: [{"model_id": "x", "provider": "bedrock", "family": "f"}]}))

    def probs_after(mutate):
        d = copy.deepcopy(good)
        mutate(d)
        return " | ".join(check(d))
    assert "no effort pinned" in probs_after(lambda d: d["judge"][0].pop("effort"))
    assert "no effort pinned" in probs_after(lambda d: d["verifier"][0].pop("effort"))
    assert "not pinned per family" in probs_after(lambda d: d["judge"].append(
        dict(d["judge"][0], model_id="g1b", effort="medium")))
    assert "Converse encoding" in probs_after(lambda d: d["builder"][0].pop("decoding"))
    assert "endpoint URL pinned" in probs_after(lambda d: d["judge"][1].pop("endpoint"))
    assert "text_only input is allowed for contestants only" in probs_after(
        lambda d: d["judge"][0].update(input_mode="text_only"))
    assert "exactly one gatekeeper" in probs_after(lambda d: d["verifier"].append(dict(d["verifier"][0], model_id="gk2")))
    assert "2 builders in 2 distinct families" in probs_after(lambda d: d["builder"].append(
        dict(d["builder"][0], model_id="w4", family="xai")))
    assert "2 builders in 2 distinct families" in probs_after(lambda d: d["builder"][1].update(family="anthropic"))
    assert "role 'verifier' is empty" in probs_after(lambda d: d.update(verifier=[]))


def test_production_refuses_uncalibrated_and_fake(tmp_path, crop):
    call = FakeCall()
    with pytest.raises(CalibrationMissing):
        build_card("hmb_0000000000000003", crop, [], None, builders=specs(), layout=Layout(tmp_path / "p"),
                   th=production_template(), ledger=None, allow_live=False, call=call)
    assert call.reqs == []          # refused before any model call
    with pytest.raises(FakeResponseRefused):
        build_card("hmb_0000000000000003", crop, [], None, builders=specs(), layout=Layout(tmp_path / "p"),
                   th=as_production(card_th()), ledger=None, allow_live=False, call=call)


@pytest.mark.live
def test_live_build_card_placeholder():
    pytest.skip("live builder calls require an authorized budget; never run in this build")


def test_identical_cards_for_two_items_do_not_conflict(tmp_path, crop):
    # C29 regression: identical eligible fact sets -> same card_uid, per-item manifests under it
    from hmb.cards.consensus import semantic_card_summary_for
    lay = Layout(tmp_path / "c29")
    cards = [build_card(iid, crop, [], None, builders=specs(), layout=lay, th=card_th(), ledger=None,
                        allow_live=False, call=FakeCall()) for iid in ("hmb_00000000000000c1", "hmb_00000000000000c2")]
    assert cards[0].card_uid == cards[1].card_uid
    for c in cards:
        man = read_json(lay.card_manifest(c.card_uid, c.item_id))
        assert man["item_id"] == c.item_id
        summ = semantic_card_summary_for(c, lay)
        assert summ.generation_manifest == f"manifests/cards/{c.card_uid}/{c.item_id}.json"
        assert summ.facts == f"semantic_cards/{c.item_id}.json"


def test_deferred_replay_builds_the_same_card(tmp_path, crop):
    """Batch transport (2026-09-26): both writers' generation calls are collected in one pass, then the
    verification calls; the replayed card is byte-identical to the synchronous one."""
    from hmb.hashing import sha256_json
    from hmb.providers import BatchDeferred
    args = ("hmb_0000000000000001", crop, [{"text": "A", "box": [0, 0, .1, .1]}], "Let ABC be...")
    direct = build_card(*args, builders=specs(), layout=Layout(tmp_path / "d"), th=card_th(), ledger=None,
                        allow_live=False, call=FakeCall())
    fake, answered, waves = FakeCall(), set(), []

    def deferring(req, layout, ledger, **kw):
        k = sha256_json(req.model_dump(mode="json"))
        if k in answered:
            return fake(req, layout, ledger, **kw)
        raise BatchDeferred({k})
    lay = Layout(tmp_path / "r")
    while True:
        try:
            card = build_card(*args, builders=specs(), layout=lay, th=card_th(), ledger=None, allow_live=False,
                              call=deferring)
            break
        except BatchDeferred as e:
            waves.append(len(e.keys))
            answered |= e.keys
    assert waves == [2, 2, 2]              # entity wave, fact wave, then the two verification calls
    assert json.dumps(card.model_dump(mode="json"), sort_keys=True) == \
        json.dumps(direct.model_dump(mode="json"), sort_keys=True)
