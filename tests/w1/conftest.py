"""Shared helpers for B1 tests: thresholds, a local ledger stub, synthetic drawings."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from hmb.config import fixture_thresholds


@pytest.fixture
def th():
    return fixture_thresholds()


class StubLedger:
    """Local stand-in for hmb.ledger.Ledger (Opus 5): records the call order, never bills."""

    def __init__(self):
        self.events = []
        self.cache = []                          # (cache_read_tokens, cache_write_tokens) per settle
        self.n = 0

    def estimate_usd(self, req):
        self.events.append(("estimate", req.purpose))
        return 0.01

    def reserve(self, phase, stage, model_id, est_usd):
        self.n += 1
        rid = f"r{self.n}"
        self.events.append(("reserve", rid, phase, stage, model_id, est_usd))
        return rid

    def settle(self, rid, *, input_tokens, output_tokens, image_count, wall_ms, unit_price_id,
               cache_read_tokens=0, cache_write_tokens=0):
        self.events.append(("settle", rid, input_tokens, output_tokens, image_count, unit_price_id))
        self.cache.append((cache_read_tokens, cache_write_tokens))
        return 0.0

    def release(self, rid):
        self.events.append(("release", rid))


@pytest.fixture
def ledger():
    return StubLedger()
