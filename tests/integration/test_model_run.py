"""CP10 driver: run_model over fixture items (Tracks A and B), resume, and production refusal.

The item is a generated control rendered through the harness and registered as a fixture item with
its oracle card (fixture profile only). The contestant is an EXPLICITLY FAKE provider returning the
clean TikZ; judges are oracle-derived fakes. Expectations come from the generator.
"""
import json
import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

pytestmark = pytest.mark.docker


class FakeContestant:
    def __init__(self, text):
        self.text, self.sends = text, 0

    def send(self, req, images):
        self.sends += 1
        assert len(images) == 1                      # the crop, for both tracks (§2.5)
        return {"text": self.text, "stop_reason": "end_turn", "input_tokens": 50, "output_tokens": 80,
                "model_version_string": "fake-contestant-FAKE"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from hmb.submission.sandbox import sandbox_available
    if not sandbox_available():
        pytest.skip("sandbox image unavailable (Docker)")
    import vertical
    from hmb.cards import oracle_card
    from hmb.synthetic.evaluator import oracle_facts
    from hmb.synthetic.families import generate
    from hmb.synthetic.generate import oracle_graph
    from hmb.synthetic.tikz import emit_tikz
    from hmb.hashing import sha256_file
    from hmb.ledger import Ledger
    from hmb.pipeline import compile_paths, run_cell
    from hmb.store import Layout, write_json_immutable, write_immutable

    root = tmp_path_factory.mktemp("modelrun")
    lay = Layout(root)
    th = vertical.thresholds()
    vertical.write_fixture_pricing(lay)
    spec = generate("circles_tangents_secants", vertical.SEED_BASE + 3)
    og = oracle_graph(spec, oracle_facts(spec, th))
    tikz = emit_tikz(spec)
    # "crop": the clean control rendered through the production harness (engineering item only)
    eng = vertical.CONTESTANT.model_copy(update={"model_id": "eng-crop-maker"})
    card0 = oracle_card(og, profile="fixture")
    iid = "hmb_" + "c" * 16
    res = run_cell(card0.model_copy(update={"item_id": "crop-" + iid}), vertical.fenced(tikz), "end_turn",
                   item_id="crop-" + iid, contestant=eng, track="C", layout=lay, th=th, judges=vertical.JUDGES,
                   ledger=Ledger(lay), allow_live=True, crop_png=None, meta=vertical.META,
                   provider_override=vertical.OracleJudge(lambda m, q, s: None if q == "Q3" else True),
                   phase="ENG")
    render = compile_paths(lay, eng.model_id, "C", "crop-" + iid)[1]
    write_immutable(lay.crop_canonical(iid), render.read_bytes())
    write_immutable(lay.context_text(iid), b"Fig. 3. A circle with a tangent.")
    write_json_immutable(lay.card(iid), card0.model_copy(update={"item_id": iid}).model_dump(mode="json", by_alias=True))
    write_json_immutable(lay.item_record("test", iid), {
        "id": iid, "dataset_version": "fixture", "quality": {"eligible": True},
        "assets": {"crop_canonical_sha256": sha256_file(lay.crop_canonical(iid))}})
    truth = vertical.truth_texts(og, oracle_facts(spec, th).closed)
    judge = vertical.OracleJudge(lambda m, q, s: None if q == "Q3" else (s in truth))
    return dict(lay=lay, th=th, iid=iid, tikz=tikz, judge=judge, res0=res)


def _run(env, contestant_provider, th=None):
    import vertical
    from hmb.pipeline import run_model
    c = vertical.CONTESTANT.model_copy(update={"model_id": "fake-contestant", "unit_price_id": vertical.PRICE_ID})
    return run_model(env["lay"], th or env["th"], contestant=c, judges=vertical.JUDGES, tracks=["A", "B"],
                     split="test", allow_live=True, provider_override=contestant_provider,
                     judge_override=env["judge"], access_date="2026-09-22",
                     provider_terms_version="fixture", output_release="public")


def test_model_run_scores_both_tracks_and_resumes(env):
    import vertical
    fc = FakeContestant(vertical.fenced(env["tikz"]))
    out = _run(env, fc)
    assert out["cells_run"] == 2 and out["stopped"] is None and fc.sends == 2
    for c in out["cells"]:
        assert c["outcome"] == "OK" and c["S"] == pytest.approx(1.0), c
    # the Track B request carries the context text; Track A does not (§2.5, §9.1)
    reqs = sorted(env["lay"].model_call_cache.rglob("*.json"))
    texts = [json.loads(p.read_text(encoding="utf-8"))["request"]["text"] for p in reqs]
    assert any("Fig. 3. A circle with a tangent." in t for t in texts)
    # resume: finalized cells are skipped and no new contestant call is made
    fc2 = FakeContestant("should never be called")
    again = _run(env, fc2)
    assert again["cells_run"] == 0 and fc2.sends == 0


class FakeTextOnlyContestant:
    """A text-only floor (§2.7(c)/(d), e.g. DeepSeek V3.1): must never receive the image."""

    def __init__(self, text):
        self.text, self.sends, self.reqs = text, 0, []

    def send(self, req, images):
        self.sends += 1
        self.reqs.append(req)
        assert images == [] and req.image_paths == []
        return {"text": self.text, "stop_reason": "end_turn", "input_tokens": 40, "output_tokens": 80,
                "model_version_string": "fake-text-only-FAKE"}


def test_text_only_contestant_gets_no_image_and_is_recorded(env):
    """Text-only mode: never the direct-figure track (A) (owner decision 2026-09-25); on Track B no image,
    the frozen no-image prompt plus the context text, records naming the right model, prompt and effort."""
    import vertical
    from hmb.pipeline import RunRefused, run_model
    from hmb.store import read_json
    from hmb.submission import model_dir_name
    from hmb.submission.prompts import load_prompt, load_t3_prompt
    c = vertical.CONTESTANT.model_copy(update={"model_id": "fake-text-only-floor", "family": "deepseek",
                                               "unit_price_id": vertical.PRICE_ID, "input_mode": "text_only"})
    fc = FakeTextOnlyContestant(vertical.fenced(env["tikz"]))
    kw: dict[str, Any] = dict(contestant=c, judges=vertical.JUDGES, split="test", allow_live=True, provider_override=fc,
              judge_override=env["judge"], access_date="2026-09-22", provider_terms_version="fixture",
              output_release="public")
    with pytest.raises(RunRefused, match="direct-figure track"):
        run_model(env["lay"], env["th"], tracks=["A", "B"], **kw)
    assert fc.sends == 0                        # refused before any call
    out = run_model(env["lay"], env["th"], tracks=["B"], **kw)
    assert out["cells_run"] == 1 and out["model_id"] == "fake-text-only-floor"
    assert fc.sends == 1
    t3_pid, t3_text, t3_sha = load_t3_prompt()
    req = fc.reqs[0]
    assert req.text.startswith(t3_text) and req.text.endswith("Fig. 3. A circle with a tangent.")
    _pid, a_text, _ = load_prompt("A")
    for element in a_text.split("\n\n")[1:]:  # every other contestant prompt element intact
        assert element.strip() in req.text
    assert not env["lay"].submission_record(model_dir_name(c.model_id), "A", env["iid"]).exists()
    for t in ("B",):
        rec = read_json(env["lay"].submission_record(model_dir_name(c.model_id), t, env["iid"]))
        assert rec["model_id"] == "fake-text-only-floor" and rec["model_family"] == "deepseek"
        assert rec["prompt_id"] == t3_pid and rec["prompt_sha256"] == t3_sha
        assert rec["image_encoding"] == {"format": "none", "crop_sha256": None}
        assert rec["decoding_params"]["effort"] is None and rec["outcome_code"] == "OK"
    for cell in out["cells"]:
        assert cell["outcome"] == "OK" and cell["S"] is not None


def test_t3_probe_runs_through_the_harness_and_is_scored(env):
    """§13.6 T3 probe (`t3-probe`): run_t3_probe (context text, no image) then run_cell against the item's
    card; stored as <item>__t3 on track B with the T3 prompt; resumable."""
    import vertical
    from hmb.pipeline import run_t3_probe_model
    from hmb.store import read_json
    from hmb.submission import model_dir_name
    from hmb.submission.prompts import load_t3_prompt, t3_item_id
    c = vertical.CONTESTANT.model_copy(update={"model_id": "fake-t3-probe", "unit_price_id": vertical.PRICE_ID})
    fc = FakeTextOnlyContestant(vertical.fenced(env["tikz"]))
    kw: dict[str, Any] = dict(contestant=c, judges=vertical.JUDGES, split="test", allow_live=True, access_date="2026-09-22",
              provider_terms_version="fixture", provider_override=fc, judge_override=env["judge"])
    out = run_t3_probe_model(env["lay"], env["th"], **kw)
    tid = t3_item_id(env["iid"])
    assert fc.sends == 1 and out["stopped"] is None and out["cells"][0]["item_id"] == tid
    assert out["cells"][0]["outcome"] == "OK" and out["cells"][0]["S"] is not None
    rec = read_json(env["lay"].submission_record(model_dir_name(c.model_id), "B", tid))
    assert rec["prompt_id"] == load_t3_prompt()[0] and rec["image_encoding"]["format"] == "none"
    assert env["lay"].scores(c.model_id, "B", tid).exists()
    again = run_t3_probe_model(env["lay"], env["th"], **dict(kw, provider_override=FakeTextOnlyContestant("x")))
    assert again["cells"] == [{"item_id": tid, "resumed": True}]


def test_production_profile_refuses_without_freeze(env):
    from hmb.config import production_template
    from hmb.pipeline import RunRefused
    with pytest.raises(RunRefused):
        _run(env, None, th=production_template())


def test_window_open_check_requires_probe_agreement(tmp_path):
    """§10.9: no open record -> closed; agreement below kappa_min -> closed; at/above -> open."""
    import vertical
    from hmb.hashing import sha256_file
    from hmb.pipeline import window_open_check
    from hmb.store import Layout, write_json_immutable
    lay, th = Layout(tmp_path), vertical.thresholds()          # fixture kappa_min = 0.8
    assert window_open_check(lay, th, vertical.JUDGES)["open"] is False
    tp = lay.p("manifests", "drift_probe", "triples.json")
    triples = [{"triple_id": f"t{i}", "reference": i % 2 == 0, "render_png": "x.png", "fact_text": "f"}
               for i in range(10)]
    write_json_immutable(tp, triples)
    ok = {t["triple_id"]: t["reference"] for t in triples}
    bad = dict(ok, t0=not ok["t0"], t1=not ok["t1"], t2=not ok["t2"])      # 7/10 < 0.8
    # replay probe (owner 2026-09-30): each judge is compared with its own recorded development answers
    refs = {j: ok for j in ("fake-judge-1", "fake-judge-2", "fake-judge-3")}
    write_json_immutable(lay.p("manifests", "drift_probe", "open.json"),
                         {"triples_sha256": sha256_file(tp), "profile": "fixture", "references": refs,
                          "answers": {"fake-judge-1": ok, "fake-judge-2": ok, "fake-judge-3": bad}})
    w = window_open_check(lay, th, vertical.JUDGES)
    assert w["open"] is False and w["agreement_open"]["fake-judge-3"] == 0.7
    assert window_open_check(lay, th, vertical.JUDGES[:2])["open"] is True
    lay2 = Layout(tmp_path / "noref")                       # an open record without references never opens
    tp2 = lay2.p("manifests", "drift_probe", "triples.json")
    write_json_immutable(tp2, triples)
    write_json_immutable(lay2.p("manifests", "drift_probe", "open.json"),
                         {"triples_sha256": sha256_file(tp2), "profile": "fixture", "answers": {"fake-judge-1": ok}})
    assert "no recorded development Q1 vote" in window_open_check(lay2, th, vertical.JUDGES[:1])["reason"]
    # a window opened under another profile is not open for this one (B6 r2 #2)
    from hmb.config import production_template
    prod = production_template()
    prod.params["kappa_min"] = {"value": 0.8, "status": "frozen", "provenance": {}}
    assert window_open_check(lay, prod, vertical.JUDGES[:2])["open"] is False
    # a closed window is never open (B6 r1 #3)
    write_json_immutable(lay.p("manifests", "drift_probe", "close.json"), {"window": "close"})
    assert window_open_check(lay, th, vertical.JUDGES[:2])["open"] is False


def test_close_model_run_writes_run_manifest(env):
    """§15.3 run.json after the window closes; refuses when the window is not closed."""
    import vertical
    from hmb.pipeline import RunRefused, close_model_run
    from hmb.store import read_json, write_json_immutable
    c = vertical.CONTESTANT.model_copy(update={"model_id": "fake-contestant"})
    lay, th = env["lay"], env["th"]
    with pytest.raises(RunRefused):
        close_model_run(lay, th, contestant=c, tracks=["A", "B"])
    for name in ("open.json", "close.json"):
        write_json_immutable(lay.p("manifests", "drift_probe", name), {"window": name})
    write_json_immutable(lay.p("manifests", "drift_probe", "result.json"),
                         {"open_agreement": 1.0, "close_agreement": 1.0, "kappa_min": 0.8, "pass": None,
                          "engineering_valid": True})
    p = close_model_run(lay, th, contestant=c, tracks=["A", "B"])
    run = read_json(p)
    assert run["model"]["attempts"]["cells"] == 2 and run["profile"] == "fixture"
    assert run["scoring_window"]["open"] and run["drift_probe"]["kappa_min"] == 0.8
