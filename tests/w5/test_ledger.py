"""Ledger (§16.4-16.5): rate card, reservations under concurrency (tracked, never capped), settle, projection.
All prices below are TEST FIXTURES with a placeholder URL; no real provider price is used."""
import threading
from pathlib import Path

import pytest
import yaml

from hmb.ledger import (Ledger, PriceMissing, PricingError, c_item,
                        load_rate_card, marginal, project_cost, usd_per_scored_item_per_model)
from hmb.schemas import ModelRequest, ModelSpec
from hmb.store import Layout, iter_jsonl

FIXTURE_CARD = {"version": "pricing-1", "entries": [
    {"unit_price_id": "fake/judge@2026-09-22", "model_id": "fake-judge", "currency": "USD",
     "usd_per_mtok_input": 1.0, "usd_per_mtok_output": 2.0, "usd_per_image": 0.001,
     "pricing_url": "https://example.invalid/fixture-pricing", "retrieved_utc": "2026-09-22"},
    {"unit_price_id": "compute/cpu@2026-09-22", "model_id": "compute/cpu", "usd_per_cpu_second": 0.0001,
     "usd_per_mtok_input": 0.0, "usd_per_mtok_output": 0.0,
     "pricing_url": "https://example.invalid/fixture-cpu", "retrieved_utc": "2026-09-22"}]}


def make(tmp_path, card=FIXTURE_CARD):
    lay = Layout(tmp_path)
    if card is not None:
        lay.pricing.parent.mkdir(parents=True, exist_ok=True)
        lay.pricing.write_text(yaml.safe_dump(card), encoding="utf-8")
    return lay, Ledger(lay)


def req(model="fake-judge", upid="fake/judge@2026-09-22", images=1, max_tokens=1000, text="x" * 500):
    spec = ModelSpec(model_id=model, provider="fake", family="f", role="judge", unit_price_id=upid)
    return ModelRequest(spec=spec, text=text, image_paths=["a.png"] * images, image_sha256=["0" * 64] * images,
                        max_tokens=max_tokens, purpose="Q1", phase="CP5", cache_scope="v")


def test_template_has_no_prices_and_unpriced_model_raises(tmp_path):
    tmpl = Path(__file__).resolve().parents[2] / "configs" / "pricing.template.yaml"
    assert load_rate_card(tmpl) == {}
    lay, led = make(tmp_path, card=None)
    with pytest.raises(PriceMissing):
        led.estimate_usd(req())
    lay2, led2 = make(tmp_path / "b")
    with pytest.raises(PriceMissing):
        led2.estimate_usd(req(model="other", upid=None))


def test_rate_card_requires_url_and_date(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text(yaml.safe_dump({"entries": [{"unit_price_id": "x", "usd_per_mtok_input": 1}]}), encoding="utf-8")
    with pytest.raises(PricingError):
        load_rate_card(p)


def test_estimate_is_worst_case_by_hand(tmp_path):
    lay, led = make(tmp_path)
    # 500 bytes input <= 500 tokens at $1/Mtok + 1000 output tokens at $2/Mtok + 1 image at $0.001
    assert led.estimate_usd(req()) == pytest.approx(500e-6 + 2000e-6 + 0.001)


def test_reserve_settle_release_and_persistence(tmp_path):
    lay, led = make(tmp_path)
    r1 = led.reserve("CP5", "Q1", "fake-judge", 1.0)
    assert led.in_flight("CP5") == 1.0 and led.spent("CP5") == 0.0
    usd = led.settle(r1, input_tokens=1000, output_tokens=500, image_count=1, wall_ms=10,
                     unit_price_id="fake/judge@2026-09-22")
    assert usd == pytest.approx(1000e-6 + 1000e-6 + 0.001)
    r2 = led.reserve("CP5", "Q1", "fake-judge", 2.0)
    led2 = Ledger(lay)                           # a restart counts the open reservation (conservative)
    assert led2.in_flight("CP5") == 2.0 and led2.spent("CP5") == pytest.approx(usd)
    led2.release(r2)
    assert led2.in_flight() == 0.0
    ev = [r["event"] for r in iter_jsonl(lay.ledger)]
    assert ev == ["reserve", "settle", "reserve", "release"]


def test_cache_tokens_are_settled_at_cache_rates(tmp_path):
    """Converse inputTokens excludes cache reads/writes, which AWS bills apart (Sep 2026 invoice gap)."""
    card = {"entries": [dict(FIXTURE_CARD["entries"][0], usd_per_mtok_cache_read=0.1, usd_per_mtok_cache_write=1.25),
                        dict(FIXTURE_CARD["entries"][0], unit_price_id="fake/nocache", model_id="fake-nocache")]}
    lay, led = make(tmp_path, card=card)
    assert led.estimate_usd(req()) == pytest.approx(500 * 1.25e-6 + 2000e-6 + 0.001)   # prompt may be a cache write
    r = led.reserve("CP5", "Q1", "fake-judge", 1.0)
    usd = led.settle(r, input_tokens=1000, output_tokens=500, image_count=0, wall_ms=1,
                     unit_price_id="fake/judge@2026-09-22", cache_read_tokens=2000, cache_write_tokens=4000)
    assert usd == pytest.approx((1000 * 1.0 + 500 * 2.0 + 2000 * 0.1 + 4000 * 1.25) / 1e6)
    r2 = led.reserve("CP5", "Q1", "fake-nocache", 1.0)
    usd2 = led.settle(r2, input_tokens=0, output_tokens=0, image_count=0, wall_ms=1, unit_price_id="fake/nocache",
                      cache_write_tokens=3000)
    assert usd2 == pytest.approx(3000e-6)                       # no cache rate: input rate, with a note
    rows = [x for x in iter_jsonl(lay.ledger) if x["event"] == "settle"]
    assert rows[0]["cache_read_input_tokens"] == 2000 and rows[0]["cache_write_input_tokens"] == 4000
    assert rows[0]["note"] is None and "priced at the input rate" in rows[1]["note"]
    assert Ledger(lay).spent("CP5") == pytest.approx(usd + usd2)


def test_spending_is_tracked_never_capped(tmp_path):
    """D2 (2026-09-23): no phase cap, aggregate ceiling, halt or authorization; every reservation is logged."""
    lay, led = make(tmp_path)
    ok = []
    barrier = threading.Barrier(40)

    def worker():
        barrier.wait()
        ok.append(led.reserve("CP5", "Q1", "fake-judge", 0.3))
    ts = [threading.Thread(target=worker) for _ in range(40)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert len(set(ok)) == 40 and led.in_flight("CP5") == pytest.approx(12.0)
    assert Ledger(lay).in_flight("CP5") == pytest.approx(12.0)     # every row reached the log
    rep = led.report()
    assert rep["phases"]["CP5"] == {"spent": 0.0, "in_flight": pytest.approx(12.0)}
    assert not {"X_total_usd", "cap", "authorized"} & (set(rep) | set(rep["phases"]["CP5"]))


def test_marginal_costs_and_per_item_metric(tmp_path):
    lay, led = make(tmp_path)
    r = led.reserve("CP5", "Q1", "fake-judge", 1.0)
    led.settle(r, input_tokens=1_000_000, output_tokens=0, image_count=0, wall_ms=1,
               unit_price_id="fake/judge@2026-09-22")
    ev = led.events()
    assert c_item(ev, t0="2000", t1="9999", delta_eligible_items=4) == pytest.approx(0.25)
    assert marginal(1.0, 0) is None
    cpu = led.price_for("compute/cpu@2026-09-22")
    m = usd_per_scored_item_per_model(ev, 2, cpu_seconds=100.0, cpu_price=cpu)
    assert m["value"] == pytest.approx((1.0 + 0.01) / 2) and "not billing" in m["disclaimer"]
    assert usd_per_scored_item_per_model(ev, 2, cpu_seconds=100.0, cpu_price=None)["value"] is None


def test_projection_by_hand():
    sym = {"Cost_onetime_engineering": 0, "G": 10, "c_book": 1, "P": 100, "c_render": 0, "c_ocr": 0.01,
           "c_det": 0, "K_f": 50, "c_cls": 0.01, "c_ver": 0.01, "c_meta": 0, "N_cand": 100, "B": 4,
           "c_card": 0.05, "M": 5, "T": 2, "N": 400, "c_gen": 0.02, "Valid1": 0.5, "c_cgx": 0, "q": 0.5,
           "EJ": 2, "c_judge": 0.01, "V": 1, "D": 400, "mu": 10, "c_probe": 2, "c_textprobe": 1}
    r = project_cost(sym)
    # 10 + 1 + 1 + 40 + 5*2*400*(0.02 + 0.5*(0 + 0.5*2*0.01)) + (400*10*2*0.01 + 400*4*4*0.05 + 2) + 1
    expected = 10 + 1 + 1 + 40 + 4000 * (0.02 + 0.005) + (80 + 320 + 2) + 1
    assert r["total"] == pytest.approx(expected)
    r2 = project_cost({k: v for k, v in sym.items() if k != "c_judge"})
    assert r2["total"] is None and r2["missing"] == ["c_judge"]
