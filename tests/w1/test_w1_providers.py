"""hmb.providers: cache, live-call refusal, ledger ordering, bounded retries, adapters (with FAKE clients).

Every client object in this file is an explicitly named fake; no network is touched.
"""
import hashlib
import json
from types import SimpleNamespace as NS
from typing import Any

import cv2
import numpy as np
import pytest

from hmb import providers as PV
from hmb.schemas import ModelRequest, ModelSpec
from hmb.store import Layout
from hmb.vocab import Role


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    monkeypatch.setattr(PV, "_sleep", lambda s: None)


def _img(tmp_path, name="c.png", v=100):
    p = tmp_path / name
    p.write_bytes(cv2.imencode(".png", np.full((8, 8), v, np.uint8))[1].tobytes())
    return p


def _req(tmp_path, provider="fake", **kw):
    p = _img(tmp_path)
    spec = ModelSpec(model_id="m-1", provider=provider, family="famA", role="judge", snapshot="m-1-2026-09",
                     unit_price_id="price-1")
    base: dict[str, Any] = dict(spec=spec, text="Q?", image_paths=[str(p)], max_tokens=64, purpose="Q1", phase="PILOT",
                                cache_scope="judge-1")
    base.update(kw)
    return ModelRequest(**base)


class FakeProvider:
    """Explicit fake provider: scripted outcomes, records whether the ledger reserved first."""

    def __init__(self, script, ledger=None):
        self.script = list(script)
        self.calls = 0
        self.ledger = ledger
        self.reserved_before = []

    def send(self, req, images):
        self.calls += 1
        if self.ledger is not None:
            self.reserved_before.append(any(e[0] == "reserve" for e in self.ledger.events))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


OK = {"text": "not json at all", "stop_reason": "end_turn", "model_version_string": "m-1-2026-09",
      "input_tokens": 11, "output_tokens": 7}


def test_cache_key_replaces_paths_with_image_hashes(tmp_path):
    r1 = _req(tmp_path)
    moved = tmp_path / "sub"
    moved.mkdir()
    q = moved / "other_name.png"
    q.write_bytes((tmp_path / "c.png").read_bytes())
    r2 = r1.model_copy(update={"image_paths": [str(q)]})
    assert PV.cache_key(r1) == PV.cache_key(r2)
    assert PV.cache_key(r1) != PV.cache_key(r1.model_copy(update={"max_tokens": 65}))
    assert PV.cache_key(r1) != PV.cache_key(r1.model_copy(update={"cache_scope": "judge-2"}))
    assert PV.cache_key(r1) != PV.cache_key(r1, fake=True)
    body = r1.model_dump(mode="json")
    body.pop("image_paths")
    body.pop("phase")
    body["spec"].pop("unit_price_id")
    body["image_sha256"] = [hashlib.sha256((tmp_path / "c.png").read_bytes()).hexdigest()]
    # fields added 2026-09-26 are keyed only when set, so every pre-existing cache entry keeps its key
    assert body["spec"].pop("structured_output") is None and body.pop("response_schema") is None
    from hmb.hashing import sha256_json
    assert PV.cache_key(r1) == sha256_json(body)


def test_phase_and_price_id_do_not_split_the_cache(tmp_path, ledger):
    lay = Layout(tmp_path / "root")
    r1 = _req(tmp_path)
    r2 = r1.model_copy(update={"phase": "DEV", "spec": r1.spec.model_copy(update={"unit_price_id": "price-2"})})
    assert PV.cache_key(r1) == PV.cache_key(r2)
    assert PV.cache_key(r1) != PV.cache_key(r1.model_copy(update={"spec": r1.spec.model_copy(update={"snapshot": "s2"})}))
    fp = FakeProvider([OK])
    PV.call_model(r1, lay, ledger, allow_live=True, provider_override=fp)
    again = PV.call_model(r2, lay, ledger, allow_live=True, provider_override=fp)
    assert fp.calls == 1 and again.cache_hit
    assert [e[0] for e in ledger.events].count("reserve") == 1


@pytest.mark.parametrize("bad", [{"text": "x", "input_tokens": "lots", "output_tokens": 3}, "not a dict"])
def test_billed_response_is_settled_before_parsing(tmp_path, ledger, bad):
    fp = FakeProvider([bad], ledger)
    with pytest.raises(PV.ProviderError):
        PV.call_model(_req(tmp_path), Layout(tmp_path / "root"), ledger, allow_live=True, provider_override=fp)
    kinds = [e[0] for e in ledger.events]
    assert kinds == ["estimate", "reserve", "settle"] and "release" not in kinds
    assert ledger.events[-1][3] == 64                 # unreadable usage settles at worst case (max_tokens)


def test_stated_image_hash_must_match(tmp_path):
    r = _req(tmp_path, image_sha256=["0" * 64])
    with pytest.raises(ValueError):
        PV.cache_key(r)


def test_miss_without_live_raises_before_ledger(tmp_path, ledger):
    with pytest.raises(PV.LiveCallsDisabled):
        PV.call_model(_req(tmp_path), Layout(tmp_path / "root"), ledger, allow_live=False,
                      provider_override=FakeProvider([OK]))
    assert ledger.events == []


def test_live_call_reserves_first_settles_and_caches(tmp_path, ledger):
    lay = Layout(tmp_path / "root")
    fp = FakeProvider([OK], ledger)
    r = PV.call_model(_req(tmp_path), lay, ledger, allow_live=True, provider_override=fp)
    assert fp.reserved_before == [True]
    assert r.fake is True and r.cache_hit is False and r.text == "not json at all"
    kinds = [e[0] for e in ledger.events]
    assert kinds == ["estimate", "reserve", "settle"]
    assert ledger.events[2][2:4] == (11, 7)
    # second call: cache hit, nothing charged, provider not called
    r2 = PV.call_model(_req(tmp_path), lay, ledger, allow_live=False, provider_override=FakeProvider([]))
    assert r2.cache_hit and r2.text == r.text and len(ledger.events) == 3
    log = [json.loads(line) for line in lay.p("manifests", "model_call_attempts.jsonl").read_text().splitlines()]
    assert [x["outcome"] for x in log] == ["ok"]


def test_fake_cache_never_serves_a_real_call(tmp_path, ledger):
    lay = Layout(tmp_path / "root")
    PV.call_model(_req(tmp_path, provider="bedrock"), lay, ledger, allow_live=True,
                  provider_override=FakeProvider([OK]))
    with pytest.raises(PV.LiveCallsDisabled):
        PV.call_model(_req(tmp_path, provider="bedrock"), lay, ledger, allow_live=False)


def test_transport_retries_bounded_then_release(tmp_path, ledger):
    lay = Layout(tmp_path / "root")
    fp = FakeProvider([PV.TransportError("503")] * 10, ledger)
    with pytest.raises(PV.TransportError):
        PV.call_model(_req(tmp_path), lay, ledger, allow_live=True, provider_override=fp)
    assert fp.calls == 4                      # 1 attempt + transport_retry_max (3) retries
    assert [e[0] for e in ledger.events] == ["estimate", "reserve", "release"]
    assert len(lay.p("manifests", "model_call_attempts.jsonl").read_text().splitlines()) == 4
    assert not any(lay.model_call_cache.rglob("*.json")) if lay.model_call_cache.exists() else True


def test_transport_then_success(tmp_path, ledger):
    fp = FakeProvider([PV.TransportError("429"), PV.TransportError("timeout"), OK], ledger)
    r = PV.call_model(_req(tmp_path), Layout(tmp_path / "root"), ledger, allow_live=True, provider_override=fp)
    assert fp.calls == 3 and len(r.attempts) == 3
    assert r.attempts[0].transport_error and r.attempts[2].transport_error is None


def test_non_transport_error_is_not_retried(tmp_path, ledger):
    fp = FakeProvider([PV.ProviderError("400 bad request"), OK], ledger)
    with pytest.raises(PV.ProviderError):
        PV.call_model(_req(tmp_path), Layout(tmp_path / "root"), ledger, allow_live=True, provider_override=fp)
    assert fp.calls == 1 and ledger.events[-1][0] == "release"


def test_live_needs_ledger_and_price(tmp_path, ledger):
    with pytest.raises(PV.ProviderError):
        PV.call_model(_req(tmp_path), Layout(tmp_path / "r"), None, allow_live=True, provider_override=FakeProvider([OK]))
    with pytest.raises(PV.ProviderError):
        PV.call_model(_req(tmp_path, provider="fake"), Layout(tmp_path / "r"), ledger, allow_live=True)


# --------------------------------------------------------------------------- adapters with FAKE clients
class FakeBedrockClient:
    def __init__(self, result=None, error=None):
        self.kw, self.result, self.error = None, result, error

    def converse(self, **kw):
        self.kw = kw
        if self.error:
            raise self.error
        return self.result


def test_bedrock_adapter(tmp_path):
    from botocore.exceptions import ClientError
    res = {"output": {"message": {"content": [{"reasoningContent": {"reasoningText": {"text": "hmm"}}},
                                              {"text": "answer"}]}},
           "stopReason": "end_turn", "usage": {"inputTokens": 50, "outputTokens": 9}}
    fc = FakeBedrockClient(result=res)
    out = PV.BedrockAdapter(fc).send(_req(tmp_path, provider="bedrock"), [(b"JPEGDATA", "image/jpeg")])
    c = fc.kw["messages"][0]["content"]
    assert c[0] == {"image": {"format": "jpeg", "source": {"bytes": b"JPEGDATA"}}} and c[1] == {"text": "Q?"}
    assert fc.kw["inferenceConfig"]["maxTokens"] == 64
    assert (out["text"], out["thinking"], out["stop_reason"], out["input_tokens"], out["output_tokens"]) == \
        ("answer", "hmm", "end_turn", 50, 9)
    thr = ClientError({"Error": {"Code": "ThrottlingException"}, "ResponseMetadata": {"HTTPStatusCode": 429}}, "Converse")
    with pytest.raises(PV.TransportError):
        PV.BedrockAdapter(FakeBedrockClient(error=thr)).send(_req(tmp_path, provider="bedrock"), [])
    val = ClientError({"Error": {"Code": "ValidationException"}, "ResponseMetadata": {"HTTPStatusCode": 400}}, "Converse")
    with pytest.raises(PV.ProviderError):
        PV.BedrockAdapter(FakeBedrockClient(error=val)).send(_req(tmp_path, provider="bedrock"), [])


# --------------------------------------------------------------------------- Bedrock Mantle, effort, text-only
GEMMA_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"      # Gemma 4 model card (2026-09-23)


def _mantle_req(tmp_path, effort="high", images=True, **kw):
    spec = ModelSpec(model_id="google.gemma-4-31b", provider="bedrock_mantle", family="google", role="judge",
                     snapshot="google.gemma-4-31b", unit_price_id="price-g", effort=effort, endpoint=GEMMA_URL,
                     decoding={"temperature": 0.0, "top_p": 1.0})
    return _req(tmp_path, spec=spec, image_paths=[str(_img(tmp_path))] if images else [], **kw)


class FakeChatClient:
    """Stands in for openai.OpenAI(...) at the transport boundary: records chat.completions.create kwargs."""

    def __init__(self, result=None, error=None):
        self.kw, self.result, self.error = None, result, error
        self.chat = NS(completions=self)

    def create(self, **kw):
        self.kw = kw
        if self.error:
            raise self.error
        return self.result


def _chat(content="\\begin{tikzpicture}", finish="stop", refusal=None, usage=(120, 30)):
    return NS(model="google.gemma-4-31b", choices=[NS(message=NS(content=content, refusal=refusal),
                                                         finish_reason=finish)],
              usage=None if usage is None else NS(prompt_tokens=usage[0], completion_tokens=usage[1]))


def test_mantle_adapter_request_and_parse(tmp_path):
    import base64
    fc = FakeChatClient(result=_chat())
    out = PV.MantleAdapter(fc).send(_mantle_req(tmp_path, system="sys"), [(b"PNGDATA", "image/png")])
    msgs = fc.kw["messages"]
    assert msgs[0] == {"role": "system", "content": "sys"} and msgs[1]["role"] == "user"
    assert msgs[1]["content"][0] == {"type": "image_url", "image_url": {
        "url": "data:image/png;base64," + base64.b64encode(b"PNGDATA").decode()}}
    assert msgs[1]["content"][1] == {"type": "text", "text": "Q?"}
    assert fc.kw["model"] == "google.gemma-4-31b" and fc.kw["max_tokens"] == 64
    assert fc.kw["reasoning_effort"] == "high" and fc.kw["temperature"] == 0.0 and fc.kw["top_p"] == 1.0
    assert (out["text"], out["stop_reason"], out["input_tokens"], out["output_tokens"], out["model_version_string"]) \
        == ("\\begin{tikzpicture}", "stop", 120, 30, "google.gemma-4-31b")
    # text-only request: the user turn carries only text; no effort -> no reasoning_effort field
    fc = FakeChatClient(result=_chat(finish="length"))
    out = PV.MantleAdapter(fc).send(_mantle_req(tmp_path, effort=None, images=False), [])
    assert fc.kw["messages"] == [{"role": "user", "content": [{"type": "text", "text": "Q?"}]}]
    assert "reasoning_effort" not in fc.kw and out["stop_reason"] == "length"
    # refusal with empty content is reported as a refusal stop reason
    out = PV.MantleAdapter(FakeChatClient(result=_chat(content=None, refusal="no"))).send(_mantle_req(tmp_path), [])
    assert out["text"] == "" and out["stop_reason"] == "refusal"


def test_mantle_adapter_error_classes(tmp_path):
    import httpx
    import openai

    def resp(code):
        return httpx.Response(code, request=httpx.Request("POST", GEMMA_URL + "/chat/completions"))
    for err, cls in ((openai.InternalServerError("ise", response=resp(500), body=None), PV.TransportError),
                     (openai.RateLimitError("slow", response=resp(429), body=None), PV.TransportError),
                     (openai.APIConnectionError(request=httpx.Request("POST", GEMMA_URL)), PV.TransportError),
                     (openai.BadRequestError("bad", response=resp(400), body=None), PV.ProviderError)):
        with pytest.raises(cls):
            PV.MantleAdapter(FakeChatClient(error=err)).send(_mantle_req(tmp_path), [])


def test_mantle_client_is_pinned_to_the_model_endpoint(tmp_path, monkeypatch):
    for k in PV.MANTLE_KEY_ENV:
        monkeypatch.delenv(k, raising=False)
    assert isinstance(PV.adapter_for("bedrock_mantle"), PV.MantleAdapter)
    req = _mantle_req(tmp_path)
    with pytest.raises(PV.ProviderError, match="API key"):
        PV.MantleAdapter()._client_for(req)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key-not-real")
    client = PV.MantleAdapter()._client_for(req)          # constructing the client makes no request
    assert str(client.base_url).rstrip("/") == GEMMA_URL and client.max_retries == 0
    no_url = req.model_copy(update={"spec": req.spec.model_copy(update={"endpoint": None})})
    with pytest.raises(PV.ProviderError, match="endpoint"):
        PV.MantleAdapter()._client_for(no_url)
    direct = req.model_copy(update={"spec": req.spec.model_copy(update={"endpoint": "https://api.openai.com/v1"})})
    with pytest.raises(PV.ProviderError, match="bedrock-mantle"):
        PV.MantleAdapter()._client_for(direct)              # never a direct OpenAI endpoint
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK")
    monkeypatch.setenv("OPENAI_API_KEY", "direct-openai-key-not-real")
    with pytest.raises(PV.ProviderError, match="API key"):
        PV.MantleAdapter()._client_for(req)                 # a direct OpenAI key is never used


def test_mantle_key_file_is_reread_on_every_call(tmp_path, monkeypatch):
    for k in PV.MANTLE_KEY_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "env-key-not-real")
    key_file = tmp_path / "mantle.key"
    key_file.write_text("file-key-1-not-real\n", encoding="utf-8")
    monkeypatch.setenv(PV.MANTLE_KEY_FILE_ENV, str(key_file))
    req = _mantle_req(tmp_path)
    assert PV.MantleAdapter()._client_for(req).api_key == "file-key-1-not-real"     # the file wins over the env
    key_file.write_text("file-key-2-not-real", encoding="utf-8")                   # a refresher replaced it
    assert PV.MantleAdapter()._client_for(req).api_key == "file-key-2-not-real"
    key_file.write_text("", encoding="utf-8")
    with pytest.raises(PV.ProviderError, match="API key"):
        PV.MantleAdapter()._client_for(req)


def test_mantle_call_is_metered_and_cached_through_call_model(tmp_path, ledger):
    lay = Layout(tmp_path / "root")
    fc = FakeChatClient(result=_chat(usage=(200, 40)))
    r = PV.call_model(_mantle_req(tmp_path), lay, ledger, allow_live=True, provider_override=PV.MantleAdapter(fc))
    assert r.fake and r.input_tokens == 200 and r.output_tokens == 40 and r.image_count == 1
    kinds = [e[0] for e in ledger.events]
    assert kinds.index("reserve") < kinds.index("settle")
    assert ("settle", "r1", 200, 40, 1, "price-g") in ledger.events
    # effort is part of the spec, so a different effort is a different cache entry
    other = _mantle_req(tmp_path, effort="medium")
    assert PV.cache_key(other, fake=True) != PV.cache_key(_mantle_req(tmp_path), fake=True)


def test_cache_tokens_reach_the_ledger(tmp_path, ledger):
    """Cache tokens are billed apart from input tokens (Sep 2026 invoice): both transports report them."""
    fc = FakeBedrockClient(result={"output": {"message": {"content": [{"text": "a"}]}}, "stopReason": "end_turn",
                                   "usage": {"inputTokens": 100, "outputTokens": 5, "cacheReadInputTokens": 30,
                                             "cacheWriteInputTokens": 900}})
    lay = Layout(tmp_path / "bedrock")
    r = PV.call_model(_req(tmp_path, provider="bedrock"), lay, ledger, allow_live=True,
                      provider_override=PV.BedrockAdapter(fc))
    assert r.input_tokens == 100 and ledger.cache[-1] == (30, 900)
    # Mantle: prompt_tokens includes the cached prefix, which is billed as cache reads
    chat = _chat(usage=(200, 40))
    chat.usage.prompt_tokens_details = NS(cached_tokens=150)
    out = PV.MantleAdapter(FakeChatClient(result=chat)).send(_mantle_req(tmp_path), [])
    assert (out["input_tokens"], out["cache_read_input_tokens"]) == (50, 150)
    assert PV.cache_usage(PV.MantleAdapter(FakeChatClient(result=_chat())).send(_mantle_req(tmp_path), [])) == (0, 0)


def test_effort_encoding_per_transport_fails_closed(tmp_path):
    def spec(provider, **kw):
        return ModelSpec(model_id="m-1", provider=provider, family="famA", role="judge", unit_price_id="p",
                         effort="medium", **kw)
    fc = FakeBedrockClient(result={"output": {"message": {"content": [{"text": "a"}]}}, "stopReason": "end_turn",
                                   "usage": {"inputTokens": 1, "outputTokens": 1}})
    with pytest.raises(PV.ProviderError, match="additional_model_request_fields"):
        PV.BedrockAdapter(fc).send(_req(tmp_path, spec=spec("bedrock")), [])
    assert fc.kw is None                                   # refused before any request
    enc = {"output_config": {"effort": "medium"}}
    PV.BedrockAdapter(fc).send(_req(tmp_path, spec=spec("bedrock", decoding={"additional_model_request_fields": enc})),
                               [])
    assert fc.kw["additionalModelRequestFields"] == enc


def test_text_only_model_is_never_sent_an_image(tmp_path, ledger):
    spec = ModelSpec(model_id="deepseek.v3.1", provider="bedrock_mantle", family="deepseek", role="contestant",
                     unit_price_id="p", input_mode="text_only", endpoint="https://bedrock-mantle.us-east-1.api.aws/v1")
    fp = FakeProvider([OK])
    with pytest.raises(PV.ProviderError, match="text_only"):
        PV.call_model(_req(tmp_path, spec=spec), Layout(tmp_path / "r"), ledger, allow_live=True, provider_override=fp)
    assert fp.calls == 0 and ledger.events == []
    r = PV.call_model(_req(tmp_path, spec=spec, image_paths=[]), Layout(tmp_path / "r"), ledger, allow_live=True,
                      provider_override=fp)
    assert fp.calls == 1 and r.image_count == 0


# --------------------------------------------------------------------------- repair 2026-09-26 (D, F)
LUNA_URL = "https://bedrock-mantle.us-east-1.api.aws/openai/v1"


def _kw(fc) -> dict:
    assert fc.kw is not None
    return fc.kw

SCHEMA = {"name": "g8b", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"], "additionalProperties": False}}


def _luna(tmp_path, native=True, decoding=None, **kw):
    """GPT-6 Luna on its Global CRIS Converse route (migration 2026-09-26)."""
    spec = ModelSpec(model_id="global.openai.gpt-6-luna", provider="bedrock", family="openai", role=Role.verifier,
                     unit_price_id="p", effort="high",
                     decoding=decoding if decoding is not None else
                     {"additional_model_request_fields": {"reasoning": {"effort": "high"}}},
                     structured_output="native_json_schema" if native else None)
    return _req(tmp_path, spec=spec, **kw)


def _converse_ok(text="{}"):
    return FakeBedrockClient(result={"output": {"message": {"content": [{"text": text}]}}, "stopReason": "end_turn",
                                     "usage": {"inputTokens": 1, "outputTokens": 1}})


def test_gpt6_global_converse_request_shape(tmp_path):
    fc = _converse_ok()
    PV.BedrockAdapter(fc).send(_luna(tmp_path, response_schema=SCHEMA), [])
    kw = _kw(fc)
    assert kw["modelId"] == "global.openai.gpt-6-luna" and kw["inferenceConfig"] == {"maxTokens": 64}
    assert kw["additionalModelRequestFields"] == {"reasoning": {"effort": "high"}}
    assert kw["outputConfig"]["textFormat"]["structure"]["jsonSchema"]["name"] == "g8b"
    fc2 = _converse_ok()
    with pytest.raises(PV.ProviderError, match="no sampling control"):
        PV.BedrockAdapter(fc2).send(_luna(tmp_path, decoding={"temperature": 0.0, "additional_model_request_fields": {
            "reasoning": {"effort": "high"}}}), [])
    assert fc2.kw is None                                  # refused before any request
    sent = PV.effective_inference(_luna(tmp_path).spec, 64)["sent"]
    assert sent["maxTokens"] == 64 and "temperature" not in sent


def test_mantle_token_field_and_no_sampling_shape(tmp_path, monkeypatch):
    """The per-model Mantle request shape (token field, no sampling) still works for any listed model."""
    from hmb import roster
    monkeypatch.setitem(roster.MODEL_REQUEST_SHAPE, "vendor.mantle-reasoner",
                        {"mantle_token_field": "max_completion_tokens", "sampling_controls": False})
    spec = ModelSpec(model_id="vendor.mantle-reasoner", provider="bedrock_mantle", family="v", role=Role.verifier,
                     unit_price_id="p", effort="high", endpoint=LUNA_URL, decoding={},
                     structured_output="native_json_schema")
    fc = FakeChatClient(result=_chat())
    PV.MantleAdapter(fc).send(_req(tmp_path, spec=spec, response_schema=SCHEMA), [])
    assert _kw(fc)["max_completion_tokens"] == 64 and "max_tokens" not in _kw(fc)
    assert _kw(fc)["reasoning_effort"] == "high" and "temperature" not in _kw(fc) and "top_p" not in _kw(fc)
    assert _kw(fc)["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "g8b", "schema": SCHEMA["schema"], "strict": True}}
    fc2 = FakeChatClient(result=_chat())
    with pytest.raises(PV.ProviderError, match="temperature"):
        PV.MantleAdapter(fc2).send(_req(tmp_path, spec=spec.model_copy(update={"decoding": {"temperature": 0.0}})), [])
    assert fc2.kw is None
    # other models are unchanged: max_tokens, sampling sent, no response_format without native mode
    fc3 = FakeChatClient(result=_chat())
    PV.MantleAdapter(fc3).send(_mantle_req(tmp_path, response_schema=dict(SCHEMA)), [])
    assert _kw(fc3)["max_tokens"] == 64 and "max_completion_tokens" not in _kw(fc3) and "response_format" not in _kw(fc3)
    assert PV.effective_inference(spec, 64)["sent"]["max_completion_tokens"] == 64


def test_converse_structured_output_field(tmp_path):
    fc = FakeBedrockClient(result={"output": {"message": {"content": [{"text": "{}"}]}}, "stopReason": "end_turn",
                                   "usage": {"inputTokens": 1, "outputTokens": 1}})
    spec = ModelSpec(model_id="mistral.mistral-large-3-675b-instruct", provider="bedrock", family="mistral",
                     role=Role.judge, unit_price_id="p", structured_output="native_json_schema")
    PV.BedrockAdapter(fc).send(_req(tmp_path, spec=spec, response_schema=SCHEMA), [])
    tf = _kw(fc)["outputConfig"]["textFormat"]
    assert tf["type"] == "json_schema" and tf["structure"]["jsonSchema"]["name"] == "g8b"
    assert json.loads(tf["structure"]["jsonSchema"]["schema"]) == SCHEMA["schema"]


def test_mantle_reasoning_field_is_kept_as_thinking(tmp_path):
    r = NS(model="google.gemma-4-31b", choices=[NS(message=NS(content="{}", refusal=None, reasoning="because"),
                                                    finish_reason="stop")],
           usage=NS(prompt_tokens=1, completion_tokens=2))
    out = PV.MantleAdapter(FakeChatClient(result=r)).send(_mantle_req(tmp_path), [])
    assert out["text"] == "{}" and out["thinking"] == "because"


def test_structured_output_and_request_shape_are_keyed_and_recorded(tmp_path, ledger):
    base = _luna(tmp_path, native=False)
    native = _luna(tmp_path, response_schema=SCHEMA)
    other = _luna(tmp_path, response_schema={**SCHEMA, "name": "other"})
    keys = {PV.cache_key(r, fake=True) for r in (base, native, other)}
    assert len(keys) == 3
    r = PV.call_model(native, Layout(tmp_path / "r"), ledger, allow_live=True,
                      provider_override=PV.BedrockAdapter(_converse_ok('{"ok": true}')))
    assert r.output_constraint == "native_json_schema"
    r2 = PV.call_model(base.model_copy(update={"text": "x"}), Layout(tmp_path / "r"), ledger, allow_live=True,
                       provider_override=PV.BedrockAdapter(_converse_ok()))
    assert r2.output_constraint == "prompt"


def test_roster_request_shape_and_parity_problems():
    from hmb.roster import request_shape_problems
    luna = ModelSpec(model_id="global.openai.gpt-6-luna", provider="bedrock", family="openai", role=Role.verifier,
                     decoding={"temperature": 0.0, "top_p": 1.0})
    j1 = ModelSpec(model_id="a", provider="bedrock_mantle", family="x", role=Role.judge,
                   structured_output="native_json_schema")
    j2 = ModelSpec(model_id="b", provider="bedrock_mantle", family="y", role=Role.judge)
    c = ModelSpec(model_id="c", provider="bedrock_mantle", family="z", role=Role.contestant,
                  structured_output="native_json_schema")
    probs = " | ".join(request_shape_problems({"verifier": [luna], "judge": [j1, j2], "contestant": [c]}))
    assert "rejected by this model" in probs and "one constraint mode per role" in probs and "raw TikZ" in probs
    ok = request_shape_problems({"verifier": [luna.model_copy(update={"decoding": {}})],
                                 "judge": [j1, j2.model_copy(update={"structured_output": "native_json_schema"})]})
    assert ok == []


def test_sonnet_45_sampling_is_exclusive(tmp_path):
    from hmb.roster import request_shape_problems
    spec = ModelSpec(model_id="us.anthropic.claude-sonnet-4-5-20250929-v1:0", provider="bedrock", family="anthropic",
                     role=Role.contestant, unit_price_id="p")               # default decoding: temperature AND top_p
    fc = FakeBedrockClient(result={"output": {"message": {"content": [{"text": "a"}]}}, "stopReason": "end_turn",
                                   "usage": {"inputTokens": 1, "outputTokens": 1}})
    with pytest.raises(PV.ProviderError, match="together"):
        PV.BedrockAdapter(fc).send(_req(tmp_path, spec=spec), [])
    assert fc.kw is None and "only one of" in request_shape_problems({"contestant": [spec]})[0]
    one = spec.model_copy(update={"decoding": {"temperature": 0.0}})
    PV.BedrockAdapter(fc).send(_req(tmp_path, spec=one), [])
    assert _kw(fc)["inferenceConfig"] == {"maxTokens": 64, "temperature": 0.0}


@pytest.mark.live
def test_live_bedrock_roundtrip(tmp_path):  # pragma: no cover - never run in this build
    raise AssertionError("live test must not run without authorization")


def test_transport_read_timeout_is_long_enough_for_reasoning_builders(monkeypatch):
    # botocore's 60 s default cut off a live Grok 4.6 card generation (2026-09-26)
    PV.BedrockAdapter()                                        # constructing the client makes no request
    assert PV.BedrockAdapter().client.meta.config.read_timeout == PV.READ_TIMEOUT_S >= 600
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "test-key-not-real")
    spec = ModelSpec(model_id="m", provider="bedrock_mantle", family="f", role=Role.judge,
                     endpoint="https://bedrock-mantle.us-east-1.api.aws/v1")
    c = PV.MantleAdapter()._client_for(ModelRequest(spec=spec, text="x", max_tokens=1, purpose="p", phase="x",
                                                    cache_scope="s"))
    assert c.timeout == PV.READ_TIMEOUT_S


def test_auth_failures_are_a_distinct_error(tmp_path):
    # an expired Bedrock key (live 2026-09-26: Mantle 401 invalid_api_key; Converse AccessDenied) is an operator
    # problem, raised as AuthError so callers stop instead of recording a model failure
    import httpx
    import openai
    from botocore.exceptions import ClientError
    resp = httpx.Response(401, request=httpx.Request("POST", GEMMA_URL + "/chat/completions"))
    with pytest.raises(PV.AuthError):
        PV.MantleAdapter(FakeChatClient(error=openai.AuthenticationError(  # pyright: ignore[reportArgumentType]
            "expired", response=resp, body=None))).send(
            _mantle_req(tmp_path), [])
    den = ClientError({"Error": {"Code": "AccessDeniedException"}, "ResponseMetadata": {"HTTPStatusCode": 403}}, "Converse")
    with pytest.raises(PV.AuthError):
        PV.BedrockAdapter(FakeBedrockClient(error=den)).send(_req(tmp_path, provider="bedrock"), [])
    assert issubclass(PV.AuthError, PV.ProviderError)
