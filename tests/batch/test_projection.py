"""Cost projection (2026-09-26): reproducible, scenario-ordered, never invents a price."""
from __future__ import annotations

import json

import pytest

from hmb.batch.projection import Prices, _load, HERE, project, write_projection

_REPORTS = HERE.parents[1] / "reports" / "prompt_repair_20260926_manifest.json"
needs_reports = pytest.mark.skipif(not _REPORTS.exists(), reason="cost-projection inputs (engineering reports/) are not part of the release")


@needs_reports
def test_projection_invariants():
    p = project()
    base = {k: v["base"]["total_usd"] for k, v in p["scenarios"].items()}
    assert base["S2"] > base["S1"] > base["S3"] > base["S4"]          # roster/route change, batch saving, sensitivity
    for sc in p["scenarios"].values():
        assert sc["low"]["total_usd"] < sc["base"]["total_usd"] < sc["high"]["total_usd"]
    s3 = p["scenarios"]["S3"]["base"]["rows"]
    batched = {r["model"] for r in s3 if r["lane"] == "batch"}
    assert batched == {"us.meta.llama4-maverick-17b-instruct-v1:0", "moonshotai.kimi-k2.5",
                       "mistral.ministral-3-14b-instruct"}
    assert all(r["lane"] == "on_demand" for r in s3 if r["phase"] == "PILOT" and r["role"] == "contestant")  # 60 < 100
    # staged writers (2026-09-26 repair): real-crop verification volume puts the pilot's Kimi verification over the minimum
    cp2k = [r for r in s3 if r["phase"] == "PILOT" and r["model"] == "moonshotai.kimi-k2.5"]
    assert {r["stage"]: r["lane"] for r in cp2k} == {"card_generation_entities": "on_demand",
                                                    "card_generation_facts": "on_demand", "card_verification": "batch"}
    assert set(p["unknown_batch_rates"]) == {"global.openai.gpt-6-luna", "global.openai.gpt-6-sol"}
    assert {r["model"] for r in s3 if r["role"] == "answer-key builder"} == {"global.openai.gpt-6-sol",
                                                                              "moonshotai.kimi-k2.5"}
    assert "moonshotai.kimi-k2.5" not in {r["model"] for r in s3 if r["role"] == "contestant"}
    s2 = p["scenarios"]["S2"]["base"]["rows"]                         # the previous configuration
    assert {r["model"] for r in s2 if r["slot"] in ("gatekeeper", "writer_a", "writer_b")} == {
        "openai.gpt-6-luna", "us.anthropic.claude-opus-5-5", "openai.gpt-6-sol"}
    assert "moonshotai.kimi-k2.5" in {r["model"] for r in s2 if r["role"] == "contestant"}


def test_missing_rates_are_never_invented():
    prices = Prices(_load(HERE / "price_snapshot.json"))
    for m in ("global.openai.gpt-6-luna", "global.openai.gpt-6-sol", "us.anthropic.claude-opus-5-5",
              "us.anthropic.claude-sonnet-5"):
        assert not prices.has_batch(m)
        with pytest.raises(KeyError, match="never invented"):
            prices.rate(m, "batch")
    snap = _load(HERE / "price_snapshot.json")
    assert all(e["source"] and e["retrieved_utc"] for e in snap["models"])


@needs_reports
def test_regeneration_is_deterministic(tmp_path):
    a = write_projection(tmp_path / "a", date="20260926")
    b = write_projection(tmp_path / "b", date="20260926")
    assert a[0].read_text() == b[0].read_text() and json.loads(a[1].read_text()) == json.loads(b[1].read_text())
