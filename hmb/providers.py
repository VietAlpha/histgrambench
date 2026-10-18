"""Model calls with a content-addressed immutable cache, ledger accounting and bounded retries.

Contract (docs/CONTRACTS.md):
  * cache key = sha256 of the canonical request with `image_paths` replaced by `image_sha256`
    (spec, max_tokens and cache_scope included). Responses from an injected provider
    (`provider_override`, test fakes) are keyed separately and carry `fake=True`, so a fake answer
    can never be served to a real call.
  * a miss with `allow_live=False` raises `LiveCallsDisabled` before anything else happens.
  * a live miss reserves on the ledger BEFORE dispatch, retries at most `transport_retry_max`
    times on transport errors / HTTP 5xx / rate limits only, logs every attempt, then settles with
    provider-reported usage (or releases the reservation on final failure). Parse/format failures
    are never retried here.

Transports (D14 closed 2026-09-25): exactly two, both Amazon Bedrock in us-east-1 on the standard
(default) service tier: `bedrock` (bedrock-runtime Converse) and `bedrock_mantle` (Chat Completions on
Bedrock's OpenAI-compatible bedrock-mantle endpoint, through the pinned `openai` SDK as a client
library only). The direct Anthropic, OpenAI and Gemini connectors are removed; a removed provider name
fails closed. They send the images and text and parse text, thinking, stop reason, model version and
usage. SDK-level retries are disabled so this module's retry count is the only one. There is no
fallback to another model or transport. They are exercised offline only through injected fake client
objects.

Per-model settings on the ModelSpec (roster decision 2026-09-23): `effort` (medium|high) is sent in
the form each transport documents (Mantle `reasoning_effort`); a Converse model's effort must be
encoded explicitly in `decoding.additional_model_request_fields`. A text-only model (spec `text_only`,
or a locked route without image input) is refused an image before anything else happens.
"""
from __future__ import annotations

import base64
import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .hashing import sha256_file, sha256_json
from .schemas import Attempt, ModelRequest, ModelResponse
from .store import Layout, append_jsonl, read_json, write_json_immutable

_sleep = time.sleep                 # tests replace this
BACKOFF_BASE_S = 2.0
# Transport read timeout (engineering constant, not a scientific parameter). botocore's 60 s default
# cut off a medium-effort reasoning builder (Grok 4.6 card generation, live 2026-09-26); 600 s matches
# the openai SDK default the Mantle transport already uses.
READ_TIMEOUT_S = 600


class LiveCallsDisabled(RuntimeError):
    """Cache miss while live calls are disabled."""


class BatchDeferred(LiveCallsDisabled):
    """Cache miss whose answer will come from a batch job (or from another in-flight call of the same key).
    Raised only while a batch router is installed (`hmb.batch`); the caller stops before writing anything
    and is replayed from the cache once `keys` are settled. A subclass of LiveCallsDisabled, so every
    existing stop-on-miss path treats it the same way (2026-09-26)."""

    def __init__(self, keys, detail: str = ""):
        self.keys = frozenset(keys)
        super().__init__(detail or f"deferred until {len(self.keys)} model call(s) settle")

    @classmethod
    def merge(cls, errs: list["BatchDeferred"]) -> "BatchDeferred":
        return cls(frozenset().union(*(e.keys for e in errs)), f"deferred on {len(errs)} independent slot(s)")


# Installed by `hmb.batch.scheduler` for a hybrid run; None otherwise (every path behaves as before).
_ROUTER = None
_CLIENT_LOCK = threading.Lock()


def set_router(router) -> None:
    global _ROUTER
    _ROUTER = router


def get_router():
    return _ROUTER


class TransportError(RuntimeError):
    """Retryable: connection/timeout failure, HTTP 5xx, or rate limit."""


class ProviderError(RuntimeError):
    """Non-retryable provider failure (4xx other than rate limit, malformed response...)."""


class AuthError(ProviderError):
    """Authentication/authorization refused (HTTP 401/403, AccessDenied, expired key). An operator problem:
    callers stop and resume after it is fixed; it is never recorded as a model's failure (2026-09-26)."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _retry_max() -> int:
    from .config import production_template
    return int(production_template().value("transport_retry_max"))   # form constant (§9.1)


# --------------------------------------------------------------------------- cache key
def resolve_images(req: ModelRequest) -> list[str]:
    """sha256 of every image file; if the request states hashes they must match the bytes."""
    got = [sha256_file(p) for p in req.image_paths]
    if req.image_sha256 and list(req.image_sha256) != got:
        raise ValueError("image_sha256 does not match the image bytes on disk")
    return got




def cache_key(req: ModelRequest, *, fake: bool = False) -> str:
    """Everything that can change the output (model id, snapshot, decoding, system, text, image
    sha256s, max_tokens, cache_scope, purpose, role/family) is keyed; the ledger phase and the
    price id are not, so the same request in two phases is one paid call."""
    body = req.model_dump(mode="json")
    body.pop("image_paths")
    body.pop("phase")
    body["spec"].pop("unit_price_id")
    body["image_sha256"] = resolve_images(req)
    # Fields added 2026-09-26 enter the key only when set, so every earlier cache entry keeps its key.
    if body["spec"].get("structured_output") is None:
        body["spec"].pop("structured_output", None)
    if body.get("response_schema") is None:
        body.pop("response_schema", None)
    from .roster import request_shape
    shape = request_shape(req.spec.model_id)
    if shape:
        body["request_shape"] = shape          # the effective token field / sampling policy of this model
    if fake:
        body = {"fake_provider": True, "request": body}
    return sha256_json(body)


def _cache_path(layout: Layout, key: str) -> Path:
    return layout.model_call_cache / key[:2] / f"{key}.json"


def _attempt_log(layout: Layout) -> Path:
    return layout.p("manifests", "model_call_attempts.jsonl")


# --------------------------------------------------------------------------- main entry
def call_model(req: ModelRequest, layout: Layout, ledger, *, allow_live: bool,
               provider_override=None) -> ModelResponse:
    fake = provider_override is not None
    from .roster import accepts_images
    if req.image_paths and not accepts_images(req.spec):
        raise ProviderError(f"{req.spec.model_id} is text_only (no image input); refusing to send it an image")
    if req.spec.provider == "fake" and not fake:
        raise ProviderError("provider 'fake' requires an explicit provider_override (tests only)")
    key = cache_key(req, fake=fake)
    path = _cache_path(layout, key)
    if path.exists():
        return _cached(path, key, fake)
    router = _ROUTER
    if router is not None:
        # Batch routing (hmb.batch): defers the call to a batch job, waits on an in-flight identical call,
        # raises the recorded final failure of a batch record, or returns a context that holds this
        # model's synchronous concurrency slot and the key's single-flight claim.
        with router.live_slot(req, key, layout=layout, allow_live=allow_live, fake=fake):
            if path.exists():                    # filled while waiting for the slot
                return _cached(path, key, fake)
            return _call_live(req, layout, ledger, key, path, fake=fake, allow_live=allow_live,
                              provider_override=provider_override)
    return _call_live(req, layout, ledger, key, path, fake=fake, allow_live=allow_live,
                      provider_override=provider_override)


def _cached(path: Path, key: str, fake: bool) -> ModelResponse:
    stored = read_json(path)
    resp = ModelResponse.model_validate(stored["response"])
    if resp.fake != fake:
        raise ProviderError(f"cache entry {key} fake={resp.fake} does not match this call")
    return resp.model_copy(update={"cache_hit": True, "cache_key": key})


def _call_live(req: ModelRequest, layout: Layout, ledger, key: str, path: Path, *, fake: bool, allow_live: bool,
               provider_override) -> ModelResponse:
    if not allow_live:
        raise LiveCallsDisabled(f"cache miss for {req.spec.model_id} ({req.purpose}) with live calls disabled")
    if ledger is None:
        raise ProviderError("a live call requires a ledger (reserve before dispatch)")
    if not fake and not req.spec.unit_price_id:
        raise ProviderError(f"{req.spec.model_id}: no unit_price_id; a live call needs a dated price entry")
    provider = provider_override or adapter_for(req.spec.provider)
    images = [(Path(p).read_bytes(), _mime(p)) for p in req.image_paths]
    est = float(ledger.estimate_usd(req))
    rid = ledger.reserve(req.phase, req.purpose, req.spec.model_id, est)
    attempts: list[Attempt] = []
    max_attempts = 1 + _retry_max()
    try:
        for i in range(1, max_attempts + 1):
            t0 = time.monotonic()
            utc = _utc()
            try:
                out = provider.send(req, images)
            except TransportError as e:
                attempts.append(Attempt(index=i, utc=utc, transport_error=str(e)[:500]))
                append_jsonl(_attempt_log(layout), {"cache_key": key, "model_id": req.spec.model_id,
                                                    "purpose": req.purpose, "attempt": i, "utc": utc,
                                                    "outcome": "transport_error", "error": str(e)[:500]})
                if i == max_attempts:
                    raise
                _sleep(BACKOFF_BASE_S * 2 ** (i - 1))
                continue
            except Exception as e:
                attempts.append(Attempt(index=i, utc=utc, transport_error=None))
                append_jsonl(_attempt_log(layout), {"cache_key": key, "model_id": req.spec.model_id,
                                                    "purpose": req.purpose, "attempt": i, "utc": utc,
                                                    "outcome": "error", "error": f"{type(e).__name__}: {e}"[:500]})
                raise
            wall_ms = int((time.monotonic() - t0) * 1000)
            attempts.append(Attempt(index=i, utc=utc, transport_error=None))
            append_jsonl(_attempt_log(layout), {"cache_key": key, "model_id": req.spec.model_id,
                                                "purpose": req.purpose, "attempt": i, "utc": utc,
                                                "outcome": "ok", "wall_ms": wall_ms})
            break
    except BaseException:
        ledger.release(rid)                      # no billable response was received
        raise
    # The provider answered, so the call is billed: settle BEFORE any parsing or validation.
    in_tok, out_tok, usage_ok = _usage(out, req)
    c_read, c_write = cache_usage(out)
    ledger.settle(rid, input_tokens=in_tok, output_tokens=out_tok, image_count=len(images), wall_ms=wall_ms,
                  unit_price_id=req.spec.unit_price_id or "fake", cache_read_tokens=c_read, cache_write_tokens=c_write)
    if not isinstance(out, dict) or not usage_ok:
        raise ProviderError(f"{req.spec.model_id}: provider returned an unparseable result "
                            "(settled at worst-case output tokens)")
    resp = ModelResponse(text=out.get("text") or "", thinking=out.get("thinking"),
                         stop_reason=out.get("stop_reason"),
                         model_version_string=out.get("model_version_string") or req.spec.model_id,
                         input_tokens=in_tok, output_tokens=out_tok, image_count=len(images), wall_ms=wall_ms,
                         attempts=attempts, cache_key=key, cache_hit=False, fake=fake,
                         output_constraint=output_constraint(req))
    req_body = req.model_dump(mode="json")
    req_body["image_sha256"] = resolve_images(req)
    write_json_immutable(path, {"cache_key": key, "request": req_body, "response": resp.model_dump(mode="json")})
    return resp


def store_batch_result(req: ModelRequest, out: dict, layout: Layout, *, key: str, utc: str,
                       transport: dict) -> ModelResponse:
    """Write one batch record's answer as the cache entry of `req` (same key, same ModelResponse fields as a
    synchronous call; `transport` records the batch lane and job). The caller settles the ledger first,
    exactly as `call_model` does. Idempotent: an identical entry is a no-op (write_json_immutable)."""
    if cache_key(req) != key:
        raise ProviderError(f"batch record key {key} does not match its request")
    in_tok, out_tok, _ok = _usage(out, req)
    resp = ModelResponse(text=out.get("text") or "", thinking=out.get("thinking"), stop_reason=out.get("stop_reason"),
                         model_version_string=out.get("model_version_string") or req.spec.model_id,
                         input_tokens=in_tok, output_tokens=out_tok, image_count=len(req.image_paths), wall_ms=0,
                         attempts=[Attempt(index=1, utc=utc, transport_error=None)], cache_key=key, cache_hit=False,
                         fake=False, output_constraint=output_constraint(req))
    req_body = req.model_dump(mode="json")
    req_body["image_sha256"] = resolve_images(req)
    write_json_immutable(_cache_path(layout, key), {"cache_key": key, "request": req_body,
                                                    "response": resp.model_dump(mode="json"), "transport": transport})
    return resp


def native_schema(req: ModelRequest) -> dict | None:
    """The schema sent as a native decoding constraint, or None (prompt-stated only)."""
    return req.response_schema if (req.response_schema and req.spec.structured_output == "native_json_schema") else None


def output_constraint(req: ModelRequest) -> str:
    return "native_json_schema" if native_schema(req) else "prompt"


def _usage(out, req: ModelRequest) -> tuple[int, int, bool]:
    """(input_tokens, output_tokens, ok). Unreadable usage settles at worst case (max_tokens out)."""
    try:
        i, o = int(out.get("input_tokens") or 0), int(out.get("output_tokens") or 0)
        if i < 0 or o < 0:
            raise ValueError
        return i, o, True
    except (AttributeError, TypeError, ValueError):
        return 0, int(req.max_tokens), False


def cache_usage(out) -> tuple[int, int]:
    """(cache_read_tokens, cache_write_tokens) reported apart from input_tokens; (0, 0) when absent."""
    try:
        r, w = int(out.get("cache_read_input_tokens") or 0), int(out.get("cache_write_input_tokens") or 0)
        return (r, w) if r >= 0 and w >= 0 else (0, 0)
    except (AttributeError, TypeError, ValueError):
        return 0, 0


def _mime(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp",
            ".gif": "image/gif"}.get(ext, "application/octet-stream")


def adapter_for(provider: str):
    from .roster import REMOVED_PROVIDERS
    if provider in REMOVED_PROVIDERS:
        raise ProviderError(f"provider {provider!r} was removed (D14, 2026-09-25); only bedrock and bedrock_mantle exist")
    cls = {"bedrock": BedrockAdapter, "bedrock_mantle": MantleAdapter}.get(provider)
    if cls is None:
        raise ProviderError(f"no adapter for provider {provider!r}")
    return cls()


# Decoding keys each transport sends (everything else in `decoding` is recorded as unsupported).
_SENT_DECODING = {"bedrock": ("temperature", "top_p"), "bedrock_mantle": ("temperature", "top_p", "seed")}


def effective_inference(spec, max_tokens: int) -> dict:
    """The generation controls actually sent for this spec (run provenance), and the configured ones
    this transport does not support, instead of pretending the two APIs are identical."""
    from .roster import SERVICE_TIER, TRANSPORTS
    from .roster import request_shape
    dec = dict(spec.decoding or {})
    keys = _SENT_DECODING.get(spec.provider, ())
    tok = request_shape(spec.model_id).get("mantle_token_field", "max_tokens") if spec.provider == "bedrock_mantle"         else "maxTokens"
    sent: dict[str, Any] = {tok: int(max_tokens), **{k: dec[k] for k in keys if dec.get(k) is not None}}
    if spec.structured_output:
        sent["structured_output"] = spec.structured_output
    if spec.provider == "bedrock" and dec.get("additional_model_request_fields"):
        sent["additional_model_request_fields"] = dec["additional_model_request_fields"]
    if spec.provider == "bedrock_mantle" and spec.effort is not None:
        sent["reasoning_effort"] = spec.effort
    unsupported = sorted(k for k in dec if k not in keys and k != "additional_model_request_fields")
    return {"transport": TRANSPORTS.get(spec.provider, spec.provider), "service_tier": SERVICE_TIER,
            "sent": sent, "unsupported_not_sent": unsupported}


def _decoding(req: ModelRequest) -> dict:
    return dict(req.spec.decoding or {})


def _model_name(req: ModelRequest) -> str:
    return req.spec.snapshot or req.spec.model_id


# --------------------------------------------------------------------------- Bedrock (Converse)
_BEDROCK_TRANSPORT_CODES = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException",
                            "ModelNotReadyException", "ServiceQuotaExceededException", "TooManyRequestsException"}


class BedrockAdapter:
    def __init__(self, client=None):
        if client is None:
            import boto3
            from botocore.config import Config

            from .roster import BEDROCK_REGION
            with _CLIENT_LOCK:                   # boto3's default session is not thread-safe (parallel cards-build)
                client = boto3.client("bedrock-runtime", region_name=BEDROCK_REGION,
                                      config=Config(retries={"total_max_attempts": 1, "mode": "standard"},
                                                    read_timeout=READ_TIMEOUT_S))
        self.client = client

    def send(self, req: ModelRequest, images: list[tuple[bytes, str]]) -> dict:
        from botocore.exceptions import (ClientError, ConnectionClosedError, ConnectTimeoutError,
                                         EndpointConnectionError, ReadTimeoutError)
        kw = converse_request(req, images)
        try:
            r = self.client.converse(**kw)
        except (EndpointConnectionError, ConnectTimeoutError, ReadTimeoutError, ConnectionClosedError) as e:
            raise TransportError(f"bedrock: {type(e).__name__}: {e}") from e
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            status = e.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0) or 0
            if code in _BEDROCK_TRANSPORT_CODES or status >= 500 or status == 429:
                raise TransportError(f"bedrock: {code} HTTP {status}") from e
            if status in (401, 403) or code in ("AccessDeniedException", "UnrecognizedClientException",
                                                 "ExpiredTokenException"):
                raise AuthError(f"bedrock: {code} HTTP {status}: {e}") from e
            raise ProviderError(f"bedrock: {code} HTTP {status}: {e}") from e
        return converse_result(r, req)


def converse_request(req: ModelRequest, images: list[tuple[bytes, str]]) -> dict:
    """The exact Converse request of `req` (modelId included). Shared by the synchronous adapter and the
    Converse-type batch record (`hmb.batch`), so both transports send the same body (2026-09-26)."""
    content: list[dict] = []
    for b, mime in images:
        fmt = {"image/png": "png", "image/jpeg": "jpeg", "image/webp": "webp", "image/gif": "gif"}.get(mime)
        if fmt is None:
            raise ProviderError(f"bedrock: unsupported image type {mime}")
        content.append({"image": {"format": fmt, "source": {"bytes": b}}})
    content.append({"text": req.text})
    dec = _decoding(req)
    from .roster import request_shape
    shape = request_shape(req.spec.model_id)
    excl = shape.get("sampling_exclusive") or []
    if excl and sum(dec.get(k) is not None for k in excl) > 1:
        raise ProviderError(f"bedrock: {req.spec.model_id} rejects {excl} together; configure only one")
    if shape.get("sampling_controls") is False and any(dec.get(k) is not None for k in ("temperature", "top_p")):
        raise ProviderError(f"bedrock: {req.spec.model_id} accepts no sampling control; configure `decoding` without "
                            "temperature/top_p")
    inf: dict[str, Any] = {"maxTokens": req.max_tokens}
    if dec.get("temperature") is not None:
        inf["temperature"] = dec["temperature"]
    if dec.get("top_p") is not None:
        inf["topP"] = dec["top_p"]
    kw: dict[str, Any] = {"modelId": _model_name(req), "messages": [{"role": "user", "content": content}],
                          "inferenceConfig": inf}
    if req.system:
        kw["system"] = [{"text": req.system}]
    schema = native_schema(req)
    if schema is not None:                   # Converse structured outputs (structured-output.html)
        kw["outputConfig"] = {"textFormat": {"type": "json_schema", "structure": {"jsonSchema": {
            "schema": json.dumps(schema["schema"], sort_keys=True), "name": schema["name"]}}}}
    if dec.get("additional_model_request_fields"):
        kw["additionalModelRequestFields"] = dec["additional_model_request_fields"]
    elif req.spec.effort is not None:
        raise ProviderError(f"bedrock: effort {req.spec.effort!r} for {req.spec.model_id} needs an explicit "
                            "decoding.additional_model_request_fields encoding (family-specific)")
    return kw


def converse_result(r: dict, req: ModelRequest) -> dict:
    """Text, thinking, stop reason and usage of a Converse response (synchronous or batch `modelOutput`)."""
    blocks = (((r or {}).get("output") or {}).get("message") or {}).get("content") or []
    texts, thinking = [], []
    for bl in blocks:
        if "text" in bl:
            texts.append(bl["text"])
        elif "reasoningContent" in bl:
            thinking.append(((bl["reasoningContent"] or {}).get("reasoningText") or {}).get("text", ""))
    u = r.get("usage") or {}
    return {"text": "".join(texts), "thinking": "\n".join(thinking) or None, "stop_reason": r.get("stopReason"),
            "model_version_string": _model_name(req), "input_tokens": u.get("inputTokens", 0),
            "output_tokens": u.get("outputTokens", 0),
            # billed on their own SKUs and not part of inputTokens (reconciled with the Sep 2026 invoice)
            "cache_read_input_tokens": u.get("cacheReadInputTokens", 0),
            "cache_write_input_tokens": u.get("cacheWriteInputTokens", 0)}


# --------------------------------------------------------------------------- Bedrock Mantle (Chat Completions)
# A Bedrock API key; never a direct OpenAI key. HMB_MANTLE_API_KEY (2026-09-30) keeps the key away from boto3, which
# signs every Bedrock call with AWS_BEARER_TOKEN_BEDROCK when it is set (that identity cannot create batch jobs).
MANTLE_KEY_ENV = ("HMB_MANTLE_API_KEY", "AWS_BEARER_TOKEN_BEDROCK")
# A file holding the key, re-read on every call, so a refresher can replace a short-term key during a long run
# (EC2, 2026-10-01). Set, it takes precedence over MANTLE_KEY_ENV.
MANTLE_KEY_FILE_ENV = "HMB_MANTLE_API_KEY_FILE"


def _field(obj, name: str):
    """Attribute of an SDK object or key of a plain dict (test fakes)."""
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


class MantleAdapter:
    """Amazon Bedrock's OpenAI-compatible `bedrock-mantle` endpoint through the pinned `openai` SDK
    (Chat Completions). The base URL is the model card's In-Region endpoint URL, pinned per model in
    the roster as `endpoint` (it differs by model: Gemma 4 is served under `/openai/v1`, DeepSeek V3.1
    under `/v1`). Authentication is a Bedrock API key sent as the bearer token. Mantle-only models
    (Gemma 4) are never routed through Converse."""

    def __init__(self, client=None):
        self.client = client

    def _client_for(self, req: ModelRequest):
        if self.client is not None:
            return self.client
        import os

        import openai
        from .roster import mantle_endpoint_problem
        base = req.spec.endpoint
        bad = mantle_endpoint_problem(base)
        if bad:
            raise ProviderError(f"bedrock_mantle: {req.spec.model_id}: {bad}")
        key_file = os.environ.get(MANTLE_KEY_FILE_ENV)
        key = (Path(key_file).read_text(encoding="utf-8").strip() if key_file
               else next((os.environ[k] for k in MANTLE_KEY_ENV if os.environ.get(k)), None))
        if not key:
            raise ProviderError("bedrock_mantle: no Bedrock API key in " + " or ".join(MANTLE_KEY_ENV))
        return openai.OpenAI(base_url=base, api_key=key, max_retries=0, timeout=READ_TIMEOUT_S)

    def send(self, req: ModelRequest, images: list[tuple[bytes, str]]) -> dict:
        import openai
        content: list[dict] = [{"type": "image_url", "image_url": {
            "url": f"data:{mime};base64,{base64.b64encode(b).decode('ascii')}"}} for b, mime in images]
        content.append({"type": "text", "text": req.text})
        messages: list[dict] = [{"role": "system", "content": req.system}] if req.system else []
        messages.append({"role": "user", "content": content})
        from .roster import request_shape
        shape = request_shape(req.spec.model_id)
        kw: dict[str, Any] = {"model": _model_name(req), "messages": messages,
                              shape.get("mantle_token_field", "max_tokens"): req.max_tokens}
        dec = _decoding(req)
        for k in ("temperature", "top_p", "seed"):
            if dec.get(k) is not None:
                if shape.get("sampling_controls") is False:
                    raise ProviderError(f"bedrock_mantle: {req.spec.model_id} rejects {k!r}; configure `decoding: {{}}`")
                kw[k] = dec[k]
        if req.spec.effort is not None:
            kw["reasoning_effort"] = req.spec.effort
        schema = native_schema(req)
        if schema is not None:                   # Chat Completions response_format (live-verified 2026-09-26)
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": schema["name"], "schema": schema["schema"], "strict": True}}
        try:
            r = self._client_for(req).chat.completions.create(**kw)
        except (openai.APIConnectionError, openai.RateLimitError, openai.InternalServerError) as e:
            raise TransportError(f"bedrock_mantle: {type(e).__name__}: {e}") from e
        except openai.APIStatusError as e:
            if getattr(e, "status_code", 0) >= 500 or getattr(e, "status_code", 0) == 429:
                raise TransportError(f"bedrock_mantle: HTTP {e.status_code}") from e
            if getattr(e, "status_code", 0) in (401, 403):
                raise AuthError(f"bedrock_mantle: HTTP {e.status_code}: {e}") from e
            raise ProviderError(f"bedrock_mantle: HTTP {getattr(e, 'status_code', '?')}: {e}") from e
        choices = _field(r, "choices") or []
        c0 = choices[0] if choices else None
        msg = _field(c0, "message") if c0 is not None else None
        text = (_field(msg, "content") if msg is not None else None) or ""
        stop = _field(c0, "finish_reason") if c0 is not None else None
        if msg is not None and _field(msg, "refusal") and not text:
            stop = "refusal"
        u = _field(r, "usage")
        in_tok = _field(u, "prompt_tokens") if u is not None else None
        out_tok = _field(u, "completion_tokens") if u is not None else None
        details = _field(u, "prompt_tokens_details") if u is not None else None
        cached = (_field(details, "cached_tokens") if details is not None else None) or 0
        if in_tok and 0 < cached <= in_tok:      # prompt_tokens includes the cached prefix; Bedrock bills it as cache reads
            in_tok -= cached
        else:
            cached = 0
        reasoning = (_field(msg, "reasoning") or _field(msg, "reasoning_content")) if msg is not None else None
        return {"text": text, "thinking": reasoning if isinstance(reasoning, str) and reasoning else None,
                "stop_reason": stop,
                "model_version_string": _field(r, "model") or _model_name(req),
                "input_tokens": in_tok or 0, "output_tokens": out_tok or 0, "cache_read_input_tokens": cached}


# --------------------------------------------------------------------------- CLI
def _cmd_cache_stats(args) -> int:
    lay = Layout.from_env(args.root)
    n = sum(1 for _ in lay.model_call_cache.rglob("*.json")) if lay.model_call_cache.exists() else 0
    print(json.dumps({"cache_dir": str(lay.model_call_cache), "entries": n}))
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("model-cache-stats", help="count cached model calls")
    p.set_defaults(func=_cmd_cache_stats)


__all__ = ["LiveCallsDisabled", "TransportError", "ProviderError", "AuthError", "call_model", "cache_key", "adapter_for",
           "cache_usage",
           "effective_inference", "BedrockAdapter", "MantleAdapter"]
