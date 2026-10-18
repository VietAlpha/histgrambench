"""Locked contestant roster and routes, D14 connector removal (owner decisions 2026-09-25).

No network: every provider below is an explicit fake or a client that is constructed but never called.
"""
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pydantic
import pytest
import yaml

from hmb import providers as PV
from hmb import roster as R
from hmb.schemas import ModelRequest, ModelSpec
from hmb.store import Layout
from hmb.vocab import Role

MANTLE = "https://bedrock-mantle.us-east-1.api.aws/v1"
MANTLE_OAI = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"
EXPECTED = [   # owner decisions 2026-09-26: one image model per family + a small anchor + the text-only floor;
    # Kimi K2.5 left the contestants when it became an answer-key writer
    ("Llama 4 Maverick 17B-128E Instruct", "us.meta.llama4-maverick-17b-instruct-v1:0", "bedrock", None, True),
    ("Nemotron Nano 12B v2 VL", "nvidia.nemotron-nano-12b-v2", "bedrock", None, True),
    ("Gemma 4 26B A4B", "google.gemma-4-26b-a4b", "bedrock_mantle", MANTLE_OAI, True),
    ("Ministral 3 14B", "mistral.ministral-3-14b-instruct", "bedrock", None, True),
    ("Gemma 4 E2B", "google.gemma-4-e2b", "bedrock_mantle", MANTLE_OAI, True),
    ("DeepSeek V3.1", "deepseek.v3.1", "bedrock_mantle", MANTLE, False),
    # roster amendment (owner, 2026-10-02)
    ("Claude Haiku 4.5", "global.anthropic.claude-haiku-4-5-20251001-v1:0", "bedrock", None, True),
]


def spec_of(route: R.Route, **kw: Any) -> ModelSpec:
    base: dict[str, Any] = dict(model_id=route.model_id, provider=route.provider,
                                family=route.family, role="contestant",
                                snapshot=route.model_id, unit_price_id="p-" + route.model_id, endpoint=route.endpoint,
                                input_mode="vision" if route.image_input else "text_only")
    base.update(kw)
    return ModelSpec.model_validate(base)


def locked_specs(**overrides) -> list[ModelSpec]:
    return [spec_of(r, **overrides.get(r.model_id, {})) for r in R.LOCKED_CONTESTANTS]


def test_exact_locked_roster_and_routes():
    got = [(r.logical_name, r.model_id, r.provider, r.endpoint, r.image_input) for r in R.LOCKED_CONTESTANTS]
    assert got == EXPECTED
    assert R.MANTLE_BASE_URL == MANTLE and R.BEDROCK_REGION == "us-east-1"
    assert set(R.TRANSPORTS) == {"bedrock", "bedrock_mantle"}
    assert R.locked_roster_problems(locked_specs()) == []


def test_image_acceptance_and_direct_figure_eligibility():
    specs = {s.model_id: s for s in locked_specs()}
    for name, mid, _p, _e, image in EXPECTED:
        assert R.accepts_images(specs[mid]) is image, name
        assert R.direct_figure_eligible(specs[mid]) is image, name
    # a text-only locked route stays ineligible even if a roster file wrongly claims vision
    ds = specs["deepseek.v3.1"].model_copy(update={"input_mode": "vision"})
    assert not R.accepts_images(ds) and not R.direct_figure_eligible(ds)
    rec = R.route_record(specs["google.gemma-4-26b-a4b"])
    assert rec["endpoint"] == MANTLE_OAI and rec["region"] == "us-east-1" and rec["service_tier"] == "default"
    assert rec["transport"] == "bedrock-mantle Chat Completions" and rec["direct_figure_eligible"] is True
    rec = R.route_record(specs["us.meta.llama4-maverick-17b-instruct-v1:0"])
    assert rec["transport"] == "bedrock-runtime Converse" and rec["endpoint"] == "bedrock-runtime.us-east-1"


@pytest.mark.parametrize("change,needle", [
    (lambda s: [x for x in s if x.model_id != "deepseek.v3.1"], "is missing"),
    (lambda s: s + [s[0].model_copy(update={"model_id": "us.anthropic.claude-sonnet-4-5-20250929-v1:0"})],
     "not in the locked roster"),
    (lambda s: [x.model_copy(update={"provider": "bedrock"}) if x.model_id == "deepseek.v3.1" else x for x in s],
     "locked route is 'bedrock_mantle'"),
    (lambda s: [x.model_copy(update={"endpoint": MANTLE}) if x.model_id.startswith("google.gemma-4") else x
                for x in s], "endpoint"),
    (lambda s: [x.model_copy(update={"family": "qwen"}) if x.model_id == "nvidia.nemotron-nano-12b-v2" else x for x in s],
     "locked family"),
    (lambda s: [x.model_copy(update={"input_mode": "vision"}) if x.model_id == "deepseek.v3.1" else x for x in s],
     "text_only"),
])
def test_substitution_alias_or_route_change_is_refused(change, needle):
    probs = R.locked_roster_problems(change(locked_specs()))
    assert probs and any(needle in p for p in probs), probs


def test_removed_providers_fail_closed(tmp_path):
    for name in ("anthropic", "openai", "gemini"):
        with pytest.raises(PV.ProviderError, match="removed"):
            PV.adapter_for(name)
        with pytest.raises(pydantic.ValidationError):
            ModelSpec(model_id="x", provider=name, family="f", role="contestant")
        roster = tmp_path / f"{name}.yaml"
        roster.write_text(yaml.safe_dump({"version": "roster-1", "contestant": [
            {"model_id": "x", "provider": name, "family": "f"}]}), encoding="utf-8")
        with pytest.raises(R.RosterError, match="removed"):
            R.load_roster(roster)
    assert not hasattr(PV, "AnthropicAdapter") and not hasattr(PV, "OpenAIAdapter") and not hasattr(PV, "GeminiAdapter")
    assert isinstance(PV.adapter_for("bedrock"), PV.BedrockAdapter)
    assert isinstance(PV.adapter_for("bedrock_mantle"), PV.MantleAdapter)


def test_converse_client_is_pinned_to_us_east_1():
    a = PV.BedrockAdapter()                    # constructing the boto3 client sends nothing
    assert a.client.meta.region_name == "us-east-1"


class NeverSend:
    calls = 0

    def send(self, req, images):              # pragma: no cover - must not be reached
        NeverSend.calls += 1
        raise AssertionError("network send reached")


@pytest.mark.parametrize("allow_live", [True, False])
def test_deepseek_image_payload_refused_before_any_request(tmp_path, ledger, allow_live):
    img = tmp_path / "c.png"
    img.write_bytes(cv2.imencode(".png", np.full((8, 8), 100, np.uint8))[1].tobytes())
    ds = spec_of(R.LOCKED_BY_ID["deepseek.v3.1"])
    req = ModelRequest(spec=ds, text="draw", image_paths=[str(img)], max_tokens=64, purpose="contestant",
                       phase="FINAL", cache_scope="c")
    with pytest.raises(PV.ProviderError, match="text_only"):
        PV.call_model(req, Layout(tmp_path / "r"), ledger, allow_live=allow_live, provider_override=NeverSend())
    assert NeverSend.calls == 0 and ledger.events == []
    for r in R.LOCKED_CONTESTANTS[:3]:            # the three multimodal models accept the same image
        ok = req.model_copy(update={"spec": spec_of(r)})
        out = PV.call_model(ok, Layout(tmp_path / "r"), ledger, allow_live=True, provider_override=_Echo())
        assert out.image_count == 1


class _Echo:
    def send(self, req, images):
        return {"text": "ok", "stop_reason": "end_turn", "model_version_string": req.spec.model_id,
                "input_tokens": 1, "output_tokens": 1}


def test_effective_inference_records_sent_and_unsupported_controls():
    q = spec_of(R.LOCKED_BY_ID["deepseek.v3.1"], decoding={"temperature": 0.0, "top_p": 1.0,
                                                                              "top_k": 5})
    e = PV.effective_inference(q, 8192)
    assert e["sent"] == {"max_tokens": 8192, "temperature": 0.0, "top_p": 1.0} and e["unsupported_not_sent"] == ["top_k"]
    c = spec_of(R.LOCKED_BY_ID["us.meta.llama4-maverick-17b-instruct-v1:0"], decoding={"temperature": 0.0,
                                                                                          "seed": 1})
    e = PV.effective_inference(c, 8192)
    assert e["transport"] == "bedrock-runtime Converse" and e["unsupported_not_sent"] == ["seed"]
    assert e["service_tier"] == "default"


def test_roster_template_lists_only_bedrock_transports():
    text = (Path(__file__).resolve().parents[2] / "configs" / "roster.template.yaml").read_text(encoding="utf-8")
    assert "provider: bedrock | bedrock_mantle\n" in text
    for gone in ("anthropic |", "| openai", "| gemini"):
        assert gone not in text


def test_no_grader_model_and_no_answer_key_company_competes():
    """Owner decision 2026-09-26: graders never compete; answer-key companies are excluded (2026-09-23)."""
    judge = ModelSpec(model_id="google.gemma-4-31b", provider="bedrock_mantle", family="google", role=Role.judge,
                      endpoint=MANTLE_OAI)
    builder = ModelSpec(model_id="w1", provider="bedrock", family="moonshot", role=Role.builder)
    as_contestant = judge.model_copy(update={"role": Role.contestant})
    sonnet = ModelSpec(model_id="global.openai.gpt-6-luna", provider="bedrock", family="openai",
                       role=Role.contestant)   # OpenAI stays an answer-key company (Anthropic removed 2026-10-02)
    r = {k: [] for k in R.ROLES}
    r.update(judge=[judge], builder=[builder], contestant=[*locked_specs(), as_contestant, sonnet])
    probs = " | ".join(R.check_roster(r))
    assert "is also a grader" in probs and "writes the answer key" in probs
    ok = dict(r, contestant=locked_specs())
    assert not any("grader" in p or "answer key" in p for p in R.check_roster(ok))
    assert not {x.family for x in R.LOCKED_CONTESTANTS} & set(R.ANSWER_KEY_COMPANIES)
    assert "moonshot" in R.ANSWER_KEY_COMPANIES and "moonshotai.kimi-k2.5" not in R.LOCKED_BY_ID
