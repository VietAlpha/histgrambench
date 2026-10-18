"""cards-build batch (Scaffold 4) on a tiny fixture root, through the REAL hmb.providers.call_model cache
with an injected FAKE provider (tests only; every response carries fake=True)."""
import argparse
import json
import re

import cv2
import numpy as np
import pytest

from hmb.cards import consensus_rule_hash
from hmb.cards.batch import card_gates_path, cards_build, ocr_words_for
from hmb.cards.build import RosterError
from hmb.providers import LiveCallsDisabled
from hmb.schemas import ModelSpec
from hmb.store import Layout, read_json

from w2_scenario import FAM, ORDER, card_th, triangle_json

PROVIDER = {"b1": "bedrock", "b2": "bedrock", "b3": "bedrock_mantle", "b4": "bedrock"}
BUILDERS = [ModelSpec(model_id=b, provider=PROVIDER[b], family=FAM[b], role="builder")
            for b in ("b1", "b3")]                        # 2 builders in 2 families (§7.3, 2026-09-26)
JUDGES = [ModelSpec(model_id="j1", provider="bedrock", family="fam_q", role="judge")]
GATE_KEYS = {"item_id", "card_uid", "exclusion_reasons", "eligible_counts", "contradiction_rate", "profile"}


class FakeProvider:
    """Injected via provider_override: answers generation with a triangle, verification all-true."""

    def __init__(self):
        self.sent = 0

    def send(self, req, images):
        self.sent += 1
        if req.purpose == "card_generation":
            ocr = json.loads(req.text.split("OCR_WORDS: ", 1)[1].splitlines()[0])
            text = triangle_json(req.spec.model_id, labels=tuple(w["text"] for w in ocr))   # labels as printed
        else:
            text = "\n".join(f"{s} true" for s in re.findall(r"^(s\d+):", req.text, re.M))
        return {"text": text, "input_tokens": 10, "output_tokens": 10, "model_version_string": "fake"}


class StubLedger:
    def estimate_usd(self, req):
        return 0.0

    def reserve(self, *a):
        return "r"

    def settle(self, *a, **k):
        return 0.0

    def release(self, rid):
        pass


def _root(tmp_path):
    lay = Layout(tmp_path / "root")
    for i, iid in enumerate(["hmb_00000000000000a1", "hmb_00000000000000a2"]):
        img = np.full((300, 300), 255, np.uint8)
        for p, q in [((40, 260), (260, 260)), ((260, 260), (150, 40)), ((150, 40), (40, 260))]:
            cv2.line(img, p, q, 0, 2 + i)
        crop = lay.crop_canonical(iid)
        crop.parent.mkdir(parents=True, exist_ok=True)
        cv2.imencode(".png", img)[1].tofile(str(crop))
        ctx = lay.context_text(iid)
        ctx.parent.mkdir(parents=True, exist_ok=True)
        ctx.write_text("Let ABC be a triangle.", encoding="utf-8")
        rec = {"id": iid, "source": {"book_id": "bk_x", "page_index": i},
               "extraction": {"crop_bbox": [0, 0, 300, 300]},
               "assets": {"crop_canonical_image": f"crops_canonical/{iid}.png", "context_text": f"context/{iid}.txt"},
               "ocr_words": [{"text": t, "box": [0, 0, .05, .05]} for t in ("ABC", "DEF")[i]],   # distinct cards
               "automatic_metadata": {"image_stats": {"stats_version": "imgstat-1", "blur_norm": 0.3 + 0.5 * i}}}
        cp = lay.p("data", "candidates", f"{iid}.json")
        cp.parent.mkdir(parents=True, exist_ok=True)
        cp.write_text(json.dumps(rec), encoding="utf-8")
    return lay


def test_cards_build_resumable_and_cache_only(tmp_path):
    lay = _root(tmp_path)
    th = card_th()
    fake = FakeProvider()
    kw = dict(builders=BUILDERS, judges=JUDGES, phase="DEV", ledger=StubLedger(), provider_override=fake)
    # 1. empty cache, live disabled: stops, names every missing generation entry, writes nothing
    with pytest.raises(LiveCallsDisabled) as ei:
        cards_build(lay, th, allow_live=False, **kw)
    msg = str(ei.value)
    assert msg.count("card_generation") == 4 and "hmb_00000000000000a1" in msg and "cache_key=" in msg
    assert fake.sent == 0 and not lay.p("semantic_cards").exists()
    # 2. live (fake provider): 2 items x (2 writers x 2 staged generation calls + 2 verification calls of <= 20)
    s = cards_build(lay, th, allow_live=True, **kw)
    assert sorted(s["built"]) == ["hmb_00000000000000a1", "hmb_00000000000000a2"] and fake.sent == 12
    g1 = read_json(card_gates_path(lay, "hmb_00000000000000a1"))
    g2 = read_json(card_gates_path(lay, "hmb_00000000000000a2"))
    assert set(g1) == GATE_KEYS and g1["profile"] == "fixture"
    assert g1["exclusion_reasons"] == [] and g1["eligible_counts"] == {"E": 6, "R": 6, "L": 3}
    assert g2["exclusion_reasons"] == []                                     # no degradation ceiling (2026-09-26)
    # 3. rerun: everything exists -> skipped, no calls, no error even with live disabled
    s = cards_build(lay, th, allow_live=False, **kw)
    assert sorted(s["skipped"]) == ["hmb_00000000000000a1", "hmb_00000000000000a2"] and fake.sent == 12
    # 4. interrupted after the card: gates recomputed from the stored card
    card_gates_path(lay, "hmb_00000000000000a1").unlink()
    s = cards_build(lay, th, allow_live=False, **kw)
    assert s["gates_only"] == ["hmb_00000000000000a1"] and read_json(card_gates_path(lay, "hmb_00000000000000a1")) == g1
    # 5. card lost but calls cached: rebuilt from cache with live disabled, zero new provider calls
    lay.card("hmb_00000000000000a2").unlink()
    card_gates_path(lay, "hmb_00000000000000a2").unlink()
    s = cards_build(lay, th, allow_live=False, **kw)
    assert s["built"] == ["hmb_00000000000000a2"] and fake.sent == 12
    # 6. a gate level changes nothing that is billed, and one writer's settings never re-bill the other
    lay.card("hmb_00000000000000a2").unlink()
    card_gates_path(lay, "hmb_00000000000000a2").unlink()
    s = cards_build(lay, card_th(item_min_entity_facts=2), builders=BUILDERS,
                    judges=JUDGES, phase="DEV", ledger=StubLedger(), provider_override=fake, allow_live=False)
    assert s["built"] == ["hmb_00000000000000a2"] and fake.sent == 12
    for f in (lay.card("hmb_00000000000000a2"), card_gates_path(lay, "hmb_00000000000000a2"),
              lay.card_votes("hmb_00000000000000a2")):
        f.unlink()                                  # a new writer setting is a new card version
    b3_low = [BUILDERS[0], BUILDERS[1].model_copy(update={"decoding": {"temperature": 0.5}})]
    s = cards_build(lay, th, builders=b3_low, judges=JUDGES, phase="DEV", ledger=StubLedger(),
                    provider_override=fake, allow_live=True)
    assert s["built"] == ["hmb_00000000000000a2"] and fake.sent == 15       # only b3's three calls are new


def test_cards_build_checks_family_disjointness(tmp_path):
    lay = _root(tmp_path)
    bad = [ModelSpec(model_id="j1", provider="bedrock", family="fam_y", role="judge")]
    with pytest.raises(RosterError):
        cards_build(lay, card_th(), builders=BUILDERS, judges=bad, allow_live=True, phase="DEV",
                    ledger=StubLedger(), provider_override=FakeProvider())
    s = cards_build(lay, card_th(), builders=BUILDERS, judges=[], allow_live=True, phase="TEST",
                    ledger=StubLedger(), provider_override=FakeProvider())
    assert s["disjointness_checked"] is False


def test_ocr_words_from_page_rows():
    rec = {"source": {"book_id": "b", "page_index": 3}, "extraction": {"crop_bbox": [100, 100, 300, 200]}}
    rows = [{"book_id": "b", "page_index": 3, "ocr_words": [{"text": "A", "bbox": [150, 120, 160, 130]},
                                                           {"text": "far", "bbox": [0, 0, 10, 10]}]}]
    w = ocr_words_for(rec, rows)
    assert [x["text"] for x in w] == ["A"] and w[0]["box"] == [0.25, 0.2, 0.05, 0.1]
    assert ocr_words_for(rec, []) == []


def test_consensus_rule_hash():
    h = consensus_rule_hash()                       # the fixed two-writer rule has no parameter
    assert len(h) == 64 and h == consensus_rule_hash()


def test_cli_registers():
    from hmb.cards.cli import register_cli
    ap = argparse.ArgumentParser()
    register_cli(ap.add_subparsers(dest="cmd"))
    a = ap.parse_args(["cards-build", "--thresholds", "t.json", "--roster", "r.yaml", "--phase", "DEV"])
    assert a.allow_live is False and a.phase == "DEV" and callable(a.func)


def test_cards_build_parallel_equals_sequential_and_isolates_errors(tmp_path):
    """workers > 1 (owner 2026-09-30) writes the same cards and gates as a sequential run; a call that fails
    is recorded as that item's error and the other items are still built."""
    from hmb.providers import TransportError
    th = card_th()
    seq, par = _root(tmp_path / "seq"), _root(tmp_path / "par")
    kw = dict(builders=BUILDERS, judges=JUDGES, phase="DEV", ledger=StubLedger())
    cards_build(seq, th, allow_live=True, provider_override=FakeProvider(), **kw)
    s = cards_build(par, th, allow_live=True, provider_override=FakeProvider(), workers=4, **kw)
    assert sorted(s["built"]) == ["hmb_00000000000000a1", "hmb_00000000000000a2"]
    for iid in s["built"]:
        assert read_json(par.card(iid)) == read_json(seq.card(iid))
        assert read_json(card_gates_path(par, iid)) == read_json(card_gates_path(seq, iid))

    class Flaky(FakeProvider):
        def send(self, req, images):
            if any("a2" in p for p in req.image_paths):
                raise TransportError("bedrock: InternalServerException HTTP 500")
            return super().send(req, images)
    import hmb.providers as P
    bad = _root(tmp_path / "bad")
    orig = P._sleep
    P._sleep = lambda s: None
    try:
        s = cards_build(bad, th, allow_live=True, provider_override=Flaky(), workers=2, **kw)
    finally:
        P._sleep = orig
    assert s["built"] == ["hmb_00000000000000a1"]
    assert [e["item_id"] for e in s["errors"]] == ["hmb_00000000000000a2"] and "TransportError" in s["errors"][0]["error"]
