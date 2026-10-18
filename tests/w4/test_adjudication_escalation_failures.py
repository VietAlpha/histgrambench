"""Repair 2026-09-26 (A): a failed escalation judge never erases statements the first two judges
already decided. Judge 3 (and its substitute) are asked only about the undecided statements; a
missing, failed, truncated, invalid or transport-failed response makes only THOSE statements
INDETERMINATE, flags the cell, and keeps the failed calls' provenance on the affected facts."""
from __future__ import annotations

import json

from hmb.adjudication.judges import BItem, JudgeCtx, Panel, run_binary, run_q3
from hmb.providers import TransportError
from hmb.schemas import ModelResponse
from hmb.vocab import JudgeQuestion

from w4_fixtures import default_truth, fact, judges, thresholds, triangle_card
from test_adjudication import _q3_scenario, run_triangle

# m1 (equal_angle) is the contested statement; r8 (equal_length) is agreed by J1 and J2.
EXTRA = [fact("m1", "equal_angle", ["s_1", "s_2"])]


def _contested(j, q, t):
    if q == "Q1" and "equal angles" in t:
        return j != "J2"                         # J1 +, J2 − -> tie -> escalation
    return default_truth(t)


class Failing:
    """Wraps the fake provider: `mode[model_id]` in {"transport", "truncated", "invalid"}."""

    def __init__(self, fake, mode: dict[str, str]):
        self.fake, self.mode = fake, mode

    def __call__(self, req, layout, ledger, **kw):
        m = self.mode.get(req.spec.model_id)
        if m is None or req.purpose not in ("Q1", "Q3"):
            return self.fake(req, layout, ledger, **kw)
        self.fake.calls.append(req)
        if m == "transport":
            raise TransportError("bedrock_mantle: APIConnectionError: Connection error.")
        text = {"truncated": '{"schema": "hmb-judge-verdict-1", "answers": [{"id": "s01", "ans',
                "invalid": '{"schema": "hmb-judge-verdict-1", "answers": [{"id": "s01", "answer": "yes"}]}'}[m]
        return ModelResponse(text=text, stop_reason="length" if m == "truncated" else "stop",
                             model_version_string=f"{req.spec.model_id}-fake", cache_key=f"k-{m}-{len(self.fake.calls)}",
                             fake=True)


def _verdicts(vf):
    return {f.fact_id: f for f in vf.facts}


def _run(fake, lay, images, monkeypatch, mode):
    fake.policy = _contested
    if mode:
        monkeypatch.setattr("hmb.adjudication.judges._call_model", Failing(fake, mode))
    return run_triangle(lay, images, card=triangle_card(EXTRA))


def test_valid_third_judge_decides_the_contested_statement(fake, lay, images, monkeypatch):
    vf = _run(fake, lay, images, monkeypatch, {})
    v = _verdicts(vf)
    assert v["r8"].verdict == "PRESENT" and v["m1"].verdict == "PRESENT"          # J3 + -> 2 of 3
    assert [x.judge_id for x in v["m1"].votes] == ["J1", "J2", "J3"] and vf.evaluator_failure is None
    j3 = [r for r in fake.calls if r.spec.model_id == "J3" and r.purpose == "Q1"]
    assert len(j3) == 1 and "equal in length" not in j3[0].text                  # only the undecided one


def test_failed_third_judge_keeps_agreed_statements(fake, lay, images, monkeypatch):
    fake.malformed["J3"] = 99
    fake.malformed["J4"] = 99                     # the substitute family fails too
    vf = _run(fake, lay, images, monkeypatch, {})
    v = _verdicts(vf)
    assert vf.evaluator_failure == "judge_failure"
    assert v["r8"].verdict == "PRESENT" and [x.judge_id for x in v["r8"].votes] == ["J1", "J2"]
    assert v["m1"].verdict == "PRESENT" and v["m1"].evidence == "forced: earliest panel judge's answer"   # J1 (+) forced
    assert [x.positive for x in v["m1"].votes] == [True, False]                     # the two real votes stay
    assert v["m1"].binding is not None
    failed = v["m1"].binding["failed_judge_calls"]
    assert [(c["judge_id"], c["status"]) for c in failed] == [("J3", "judge_failure"), ("J4", "judge_failure")]
    assert all(len(c["raw_call_sha256"]) == 1 for c in failed)                      # one attempt per slot
    assert "failed_judge_calls" not in (v["r8"].binding or {})
    assert v["r1"].verdict == "PRESENT"                                             # extractor verdicts stand


def test_transport_failure_of_the_third_judge(fake, lay, images, monkeypatch):
    vf = _run(fake, lay, images, monkeypatch, {"J3": "transport", "J4": "transport"})
    v = _verdicts(vf)
    assert vf.evaluator_failure == "judge_failure"
    assert v["r8"].verdict == "PRESENT" and v["m1"].verdict == "PRESENT" and v["m1"].evidence == "forced: earliest panel judge's answer"
    assert v["m1"].binding is not None
    assert v["m1"].binding["failed_judge_calls"][0]["detail"].startswith("TransportError")


def test_truncated_third_judge(fake, lay, images, monkeypatch):
    vf = _run(fake, lay, images, monkeypatch, {"J3": "truncated", "J4": "truncated"})
    v = _verdicts(vf)
    assert v["r8"].verdict == "PRESENT" and v["m1"].verdict == "PRESENT" and v["m1"].evidence == "forced: earliest panel judge's answer"
    assert v["m1"].binding is not None
    assert v["m1"].binding["failed_judge_calls"][0]["detail"] == "truncated"


def test_invalid_third_judge(fake, lay, images, monkeypatch):
    vf = _run(fake, lay, images, monkeypatch, {"J3": "invalid", "J4": "invalid"})
    v = _verdicts(vf)
    assert v["r8"].verdict == "PRESENT" and v["m1"].verdict == "PRESENT" and v["m1"].evidence == "forced: earliest panel judge's answer"
    assert vf.evaluator_failure == "judge_failure"


def test_partial_third_answer_forces_only_the_unanswered_statement(fake, lay, images):
    # two contested statements; J3 answers one of them only (valid JSON, one id missing)
    extra = EXTRA + [fact("m2", "equal_angle", ["s_2", "s_3"])]

    def pol(j, q, t):
        if q == "Q1" and "equal angles" in t:
            if j == "J3":
                return True if "segment S1" in t or "“AB”" in t or t.startswith("S1") else None
            return j != "J2"
        return default_truth(t)
    fake.policy = pol
    vf = run_triangle(lay, images, card=triangle_card(extra))
    v = _verdicts(vf)
    forced = sorted(k for k in ("m1", "m2") if v[k].evidence == "forced: earliest panel judge's answer")
    assert v["m1"].verdict == v["m2"].verdict == "PRESENT" and len(forced) == 1   # J1 (+) forced on one
    assert v["r8"].verdict == "PRESENT" and vf.evaluator_failure is None          # a missing answer is not a failure


def test_stage_one_failure_keeps_the_other_judges_votes_and_escalates(fake, lay, images):
    # J1 and its substitute J3 fail in stage 1: J2's votes are kept (previously the whole call was lost).
    # Escalation re-asks the first family that did not vote (J1, red-team #8); it fails again and its
    # substitute is J4 (J3 already failed on this chunk); J2 and J4 concur and decide.
    fake.malformed["J1"] = 99
    fake.malformed["J3"] = 99
    th = thresholds()
    items = [BItem("f1", "segment S1 is parallel to segment S2."), BItem("f2", "point “A” lies on segment S1.")]
    res = run_binary(JudgeQuestion.Q1, items, images=[images[0]], ctx=JudgeCtx(lay, th, None, False, None, "FINAL", "c"),
                     panel=Panel(judges()), seed="s")
    assert res.decision == {"f1": "positive", "f2": "positive"}
    assert [v.judge_id for v in res.votes["f1"]] == ["J2", "J4"]
    assert res.failures and res.failures[0]["judge_id"] == "J1"
    assert fake.count("Q1", "J1") == 2 and fake.count("Q1", "J3") == 1            # J3 never re-tried in the chunk


def test_q3_failed_third_judge_keeps_agreed_answers(fake, lay, images):
    # Q3: J1 and J2 agree that D is absent ("none") but disagree on E; J3 and J4 fail -> D stays decided
    def pol(j, q, t):
        if q != "Q3":
            return default_truth(t)
        if "“D”" in t:
            return "none"
        return {"J1": "o1", "J2": "o2"}.get(j, "o1")
    fake.policy = pol
    fake.malformed_if = lambda req: req.purpose == "Q3" and req.spec.model_id in ("J3", "J4")
    c, cg = _q3_scenario()
    vf = run_triangle(lay, images, card=c, cgx=cg)
    v = {f.fact_id: f.verdict for f in vf.facts}
    assert vf.evaluator_failure == "judge_failure"
    assert v["e4"] == "ABSENT"                    # D: agreed "none" by J1 and J2, kept
    assert v["e9"] == "PRESENT"                   # E: contested, escalation failed -> decided by Q1


def test_run_q3_direct_partial(fake, lay, images):
    fake.policy = lambda j, q, t: "o1"
    fake.malformed_if = lambda req: req.spec.model_id == "J1"
    th = thresholds()
    out, got = run_q3([("e", "q")], entities_text="E", objects_text="o1: point", render=images[0],
                      ctx=JudgeCtx(lay, th, None, False, None, "FINAL", "c"), panel=Panel(judges()))
    assert out == {"e": "o1"}                     # J1 failed -> substitute J3; J2 and J3 agree
    assert json.dumps(got)                        # provenance is serializable


def test_model_id_with_colon_is_adjudicated(fake, lay, images):
    # locked contestant ids such as us.anthropic.claude-sonnet-4-5-20250929-v1:0 contain ':' (illegal in Windows
    # paths); every adjudication artifact path goes through safe_model_dir (live flow 2026-09-26)
    from hmb.store import safe_model_dir
    from w4_fixtures import triangle_cgx, write_submission_stub
    from test_adjudication import run_triangle as _rt  # noqa: F401
    mid = "us.vendor.model-v1:0"
    write_submission_stub(lay, model=mid)
    cg = triangle_cgx().model_copy(update={"model_id": mid})
    from hmb.adjudication import adjudicate
    vf = adjudicate(triangle_card(), cg, images[0], images[1], judges=judges(), layout=lay, th=thresholds(),
                    ledger=None, allow_live=False)
    assert vf.model_id == mid
    assert lay.p("adjudication", "alignment", safe_model_dir(mid), "A", "hmb_0000000000000001.json").exists()


def test_auth_failure_stops_adjudication_instead_of_failing_the_judge(fake, lay, images, monkeypatch):
    # an expired key must not be frozen into verdicts as judge_failure (live flow 2026-09-26): it propagates
    import pytest
    from hmb.providers import AuthError

    def expired(req, layout, ledger, **kw):
        raise AuthError("bedrock_mantle: HTTP 401: invalid_api_key")
    monkeypatch.setattr("hmb.adjudication.judges._call_model", expired)
    with pytest.raises(AuthError):
        run_triangle(lay, images, card=triangle_card(EXTRA))
    assert not lay.verdicts("contestant_x", "A", "hmb_0000000000000001").exists()     # nothing frozen
