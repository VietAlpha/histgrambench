"""Batch transport (2026-09-26): adjudication replayed through deferrals gives the SAME verdicts as a
synchronous run, and each judge stage collects all its independent slots in one pass (so a batch round
carries every primary Q1/Q2/Q3 call of the stage, not one call at a time)."""
from __future__ import annotations

import json

from hmb.hashing import sha256_json
from hmb.providers import BatchDeferred

from test_adjudication import run_triangle
from test_adjudication_escalation_failures import EXTRA, _contested
from w4_fixtures import triangle_card


class Deferring:
    """First sight of a request defers it (as a batch lane would); after `flush()` it is answered by the fake."""

    def __init__(self, fake):
        self.fake, self.answered, self.pending, self.per_pass = fake, set(), set(), []

    def __call__(self, req, layout, ledger, **kw):
        k = sha256_json(req.model_dump(mode="json"))
        if k in self.answered:
            return self.fake(req, layout, ledger, **kw)
        self.pending.add(k)
        raise BatchDeferred({k})

    def flush(self):
        self.per_pass.append(len(self.pending))
        self.answered |= self.pending
        self.pending = set()


def _vf_json(vf):
    return json.dumps(vf.model_dump(mode="json", by_alias=True), sort_keys=True)


def test_deferred_replay_equals_synchronous_run(fake, lay, images, monkeypatch, tmp_path):
    from hmb.store import Layout
    fake.policy = _contested                                   # forces a third-judge escalation
    card = triangle_card(EXTRA)
    direct = run_triangle(Layout(tmp_path / "direct"), images, card=card)
    d = Deferring(fake)
    monkeypatch.setattr("hmb.adjudication.judges._call_model", d)
    passes = 0
    while True:
        try:
            replayed = run_triangle(lay, images, card=card)
            break
        except BatchDeferred as e:
            assert e.keys == d.pending                          # every pending key is reported to the scheduler
            d.flush()
            passes += 1
            assert passes < 10
    assert _vf_json(replayed) == _vf_json(direct)
    # stage 1 is one wave: both primary judges on every chunk were collected together
    assert d.per_pass[0] >= 2 and passes <= 6


def test_stage_one_collects_both_primary_judges(fake, lay, images, monkeypatch):
    seen = []

    def defer_all(req, layout, ledger, **kw):
        seen.append((req.spec.model_id, req.purpose))
        raise BatchDeferred({sha256_json(req.model_dump(mode="json"))})
    monkeypatch.setattr("hmb.adjudication.judges._call_model", defer_all)
    try:
        run_triangle(lay, images)
    except BatchDeferred as e:
        assert len(e.keys) == len(seen) >= 2
    judges_asked = {m for m, _ in seen}
    assert {"J1", "J2"} <= judges_asked and "J3" not in judges_asked   # no escalation before answers exist
