"""Batch inference transport, capability routing, job lifecycle and hybrid scheduler (2026-09-26).
Offline only: fake Bedrock/S3 clients and fake backends; nothing is called or paid."""
from __future__ import annotations

import base64
import json
import random
import threading
import time
from pathlib import Path

import pytest
import yaml
from PIL import Image

import hmb.providers as PV
from hmb.batch import capability as CAP
from hmb.batch.jobs import (BatchConfigError, BatchStore, BedrockBatchBackend, batch_record, cancel_job,
                            classify_record_error, classify_submit_error, collect_job, lane_key, load_config,
                            prepare_job, refresh_job, submit_job)
from hmb.batch.scheduler import HybridRouter, HybridScheduler, Policy
from hmb.ledger import Ledger, PriceMissing
from hmb.schemas import ModelRequest, ModelSpec
from hmb.store import Layout
from hmb.vocab import Role

SCHEMA = {"name": "g8b", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}},
                                    "required": ["ok"], "additionalProperties": False}}


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def lay(tmp_path) -> Layout:
    lay = Layout(tmp_path / "root")
    entries = []
    for m in ("moonshotai.kimi-k2.5", "global.openai.gpt-6-luna", "qwen.judge"):
        entries.append({"unit_price_id": f"od/{m}", "model_id": m, "usd_per_mtok_input": 0.6, "usd_per_mtok_output": 3.0, "image_input_tokens": 1600,
                        "pricing_url": "https://aws.amazon.com/bedrock/pricing/", "retrieved_utc": "2026-09-26"})
    entries.append({"unit_price_id": "batch/moonshotai.kimi-k2.5", "model_id": "moonshotai.kimi-k2.5", "lane": "batch",
                    "usd_per_mtok_input": 0.3, "usd_per_mtok_output": 1.5, "image_input_tokens": 1600,
                    "pricing_url": "https://aws.amazon.com/bedrock/pricing/", "retrieved_utc": "2026-09-26"})
    lay.pricing.parent.mkdir(parents=True, exist_ok=True)
    lay.pricing.write_text(yaml.safe_dump({"version": "pricing-1", "entries": entries}), encoding="utf-8")
    return lay


@pytest.fixture
def img(tmp_path) -> Path:
    p = tmp_path / "crop.png"
    Image.new("L", (16, 16), 255).save(p)
    return p


def kimi(**kw) -> ModelSpec:
    kw.setdefault("decoding", {"temperature": 0.0, "top_p": 1.0})
    return ModelSpec(model_id="moonshotai.kimi-k2.5", provider="bedrock", family="moonshot", role=Role.contestant,
                     unit_price_id="od/moonshotai.kimi-k2.5", **kw)


def luna() -> ModelSpec:
    return ModelSpec(model_id="global.openai.gpt-6-luna", provider="bedrock", family="openai", role=Role.verifier,
                     unit_price_id="od/global.openai.gpt-6-luna", effort="high",
                     decoding={"additional_model_request_fields": {"reasoning": {"effort": "high"}}},
                     structured_output="native_json_schema")


def req(spec, text="draw it", images=(), **kw) -> ModelRequest:
    return ModelRequest(spec=spec, text=text, image_paths=[str(p) for p in images], max_tokens=64, purpose="contestant",
                        phase="FINAL", cache_scope="t", **kw)


def matrix(**over) -> dict:
    e = {"route_id": "kimi", "roles": ["contestant"], "model_id": "moonshotai.kimi-k2.5", "provider": "bedrock",
         "endpoint": None, "provider_path": "converse_job", "region": "us-east-1", "inference_scope": "in_region",
         "modality": "image", "structured_output_required": False, "sync_structured_output": "not_required",
         "batch_submission": "SUPPORTED_LIVE_COMPLETED", "batch_structured_output": "DOCUMENTED_UNSUPPORTED",
         "min_records": 2, "pricing_state": "explicit", "final_route": "batch", "reason": "probe", "evidence": ["x"]}
    e.update(over)
    g = {**e, "route_id": "luna", "model_id": "global.openai.gpt-6-luna", "roles": ["verifier"],
         "structured_output_required": True, "batch_submission": "UNSUPPORTED_LIVE_CONFIRMED",
         "batch_structured_output": "UNSUPPORTED_LIVE_CONFIRMED", "final_route": "synchronous", "min_records": None,
         "reason": "batch rejected"}
    return {"version": CAP.MATRIX_VERSION, "entries": [e, g]}


def converse_out(text, stop="end_turn", i=10, o=5):
    return {"output": {"message": {"content": [{"text": text}]}}, "stopReason": stop,
            "usage": {"inputTokens": i, "outputTokens": o}}


class FakeBackend:
    """In-memory job backend; `answer(record) -> output line dict` decides every record."""

    def __init__(self, answer=None, fail_submit=None):
        self.jobs: dict[str, dict] = {}
        self.answer = answer or (lambda r: {"recordId": r["recordId"], "modelOutput": converse_out(
            "ok:" + r["modelInput"]["messages"][0]["content"][-1]["text"])})
        self.fail_submit = fail_submit
        self.submits = 0
        self.stopped: list[str] = []

    def submit(self, job_name, model_id, jsonl, tags):
        self.submits += 1
        if self.fail_submit:
            raise self.fail_submit
        recs = [json.loads(x) for x in jsonl.decode().splitlines()]
        self.jobs.setdefault(job_name, {"recs": recs, "status": "InProgress", "model": model_id})
        return f"arn:aws:bedrock:us-east-1:000000000000:model-invocation-job/{job_name}"

    def status(self, arn):
        return {"status": self.jobs[arn.rsplit("/", 1)[-1]]["status"], "message": None}

    def complete(self, status="Completed"):
        for j in self.jobs.values():
            j["status"] = status

    def outputs(self, job_name):
        lines = [json.dumps(self.answer(r)) for r in self.jobs[job_name]["recs"]]
        lines = [x for x in lines if x != "null"]
        random.Random(1).shuffle(lines)                                # out of order
        return lines, {"totalRecordCount": len(lines)}

    def stop(self, arn):
        self.stopped.append(arn.rsplit("/", 1)[-1])


def queue(lay, reqs):
    st = BatchStore(lay)
    out = []
    for r in reqs:
        k = PV.cache_key(r)
        st.save_request(r, k, lane_key(r))
        out.append((k, r))
    return st, out


# --------------------------------------------------------------------------- 1-4 records and envelopes
def test_record_is_the_synchronous_converse_body(img):
    r = req(kimi(), images=[img])
    rec = batch_record(r, "k" * 64)
    sync = PV.converse_request(r, [(img.read_bytes(), "image/png")])
    assert rec["recordId"] == "k" * 64 and "modelId" not in rec["modelInput"]
    b = rec["modelInput"]["messages"][0]["content"][0]["image"]["source"]["bytes"]
    assert base64.b64decode(b) == img.read_bytes()                     # Converse JSON wire format: base64
    sync.pop("modelId")
    sync["messages"][0]["content"][0]["image"]["source"]["bytes"] = b
    assert rec["modelInput"] == sync                                   # same prompt bytes, decoding, effort
    text = batch_record(req(kimi(), "only text"), "t" * 64)["modelInput"]
    assert text["messages"][0]["content"] == [{"text": "only text"}]
    native = batch_record(req(luna(), response_schema=SCHEMA), "n" * 64)["modelInput"]
    assert native["outputConfig"]["textFormat"]["type"] == "json_schema"
    assert native["additionalModelRequestFields"] == {"reasoning": {"effort": "high"}}
    assert "outputConfig" not in batch_record(req(luna().model_copy(update={"structured_output": None}),
                                                  response_schema=SCHEMA), "p" * 64)["modelInput"]


def test_lanes_group_only_compatible_records(img):
    a = lane_key(req(kimi(), images=[img]))
    assert a == lane_key(req(kimi(), "other text", images=[img]))
    assert a != lane_key(req(kimi()))                                    # modality
    assert a != lane_key(req(kimi(decoding={"temperature": 0.5}), images=[img]))   # decoding
    assert lane_key(req(luna(), response_schema=SCHEMA)) != lane_key(req(luna()))  # schema mode


# --------------------------------------------------------------------------- 5 capability, fail-closed
def test_route_selection_is_fail_closed(img):
    m = matrix()
    assert CAP.decide(req(kimi(), images=[img]), m).route == "batch"
    unknown = kimi().model_copy(update={"model_id": "vendor.unknown"})
    d = CAP.decide(req(unknown), m)
    assert d.route == "synchronous" and "no capability entry" in d.reason
    blocked = CAP.decide(req(luna(), response_schema=SCHEMA), m)
    assert blocked.route == "synchronous" and "batch rejected" in blocked.reason
    txt = matrix(modality="text")
    assert CAP.decide(req(kimi(), images=[img]), txt).route == "synchronous"
    so = matrix(batch_structured_output="DOCUMENTED_UNSUPPORTED")
    so_req = req(kimi(structured_output="native_json_schema"), response_schema=SCHEMA)
    assert "BLOCKED_BY_REQUIRED_STRUCTURED_OUTPUT" in CAP.decide(so_req, so).reason


@pytest.mark.parametrize("over,msg", [
    ({"batch_submission": "SUPPORTED_SUBMISSION_ONLY"}, "completed live"),
    ({"batch_submission": "DOCUMENTED_SUPPORTED_UNTESTED"}, "completed live"),
    ({"provider_path": "invoke_job"}, "completed live"),
    ({"structured_output_required": True}, "native structured output"),
    ({"batch_submission": "SUPPORTED"}, "not a capability state"),
    ({"evidence": []}, "evidence"),
])
def test_matrix_loader_refuses_unproven_batch_routes(tmp_path, over, msg):
    p = tmp_path / "m.json"
    p.write_text(json.dumps(matrix(**over)))
    with pytest.raises(CAP.CapabilityError, match=msg):
        CAP.load_matrix(p)


def test_shipped_matrix_loads_and_covers_the_roster_routes():
    m = CAP.load_matrix()
    ids = {e["model_id"] for e in m["entries"]}
    for mid in ("global.openai.gpt-6-luna", "global.openai.gpt-6-sol", "us.anthropic.claude-opus-5-5",
                "us.meta.llama4-maverick-17b-instruct-v1:0", "moonshotai.kimi-k2.5", "deepseek.v3.1"):
        assert mid in ids
    from hmb.roster import RETIRED_ROUTES
    assert not ids & set(RETIRED_ROUTES)                          # retired GPT-6 Mantle / US Geo ids
    by = {e["model_id"]: e for e in m["entries"]}
    assert by["us.openai.gpt-6.1-sol"]["roles"] == ["builder"]      # writer since 2026-09-30 (US CRIS)
    assert by["us.openai.gpt-6.1-sol"]["final_route"] == "synchronous"
    for mid in ("global.openai.gpt-6-luna", "global.openai.gpt-6-sol", "us.anthropic.claude-opus-5-5"):
        assert by[mid]["batch_submission"] == "UNSUPPORTED_LIVE_CONFIRMED" and by[mid]["final_route"] == "synchronous"


# --------------------------------------------------------------------------- 6-10 jobs, ingestion, resume, ledger
def test_submit_collect_out_of_order_duplicates_and_ledger_lane(lay, img):
    reqs = [req(kimi(), f"item {i}", images=[img]) for i in range(5)]
    st, pairs = queue(lay, reqs)
    led = Ledger(lay)
    lane = lane_key(reqs[0])

    def answer(r):
        return {"recordId": r["recordId"], "modelOutput": converse_out("ok")}
    be = FakeBackend(answer)
    doc = submit_job(st, be, led, prepare_job(st, led, pairs, lane=lane))
    assert doc["state"] == "submitted" and led.in_flight() > 0
    assert all(led.is_open(r["reservation_id"]) for r in doc["records"])
    be.complete()
    orig = be.outputs

    def with_duplicate_and_unknown(job_name):
        lines, man = orig(job_name)
        return lines + [lines[0], json.dumps({"recordId": "f" * 64, "modelOutput": {}})], man
    be.outputs = with_duplicate_and_unknown                          # type: ignore[method-assign]
    doc = collect_job(st, be, led, refresh_job(st, be, doc))
    assert doc["state"] == "collected" and all(r["outcome"]["status"] == "ok" for r in doc["records"])
    settles = [e for e in led.events() if e["event"] == "settle"]
    assert len(settles) == 5 and {e["lane"] for e in settles} == {"batch"}
    assert {e["unit_price_id"] for e in settles} == {"batch/moonshotai.kimi-k2.5"}
    assert settles[0]["usd"] == pytest.approx((10 * 0.3 + 5 * 1.5) / 1e6)       # the batch rate, not on-demand
    # collecting again is a no-op: no second settle, no second cache write
    collect_job(st, be, led, st.job(doc["job_name"]))
    assert len([e for e in Ledger(lay).events() if e["event"] == "settle"]) == 5
    # the cache entry is the ordinary entry of the request: a synchronous call now hits it
    for k, r in pairs:
        resp = PV.call_model(r, lay, led, allow_live=False)
        assert resp.cache_hit and resp.cache_key == k and resp.text == "ok"
        stored = json.loads((lay.model_call_cache / k[:2] / f"{k}.json").read_text())
        assert stored["transport"]["lane"] == "batch" and stored["transport"]["job_name"] == doc["job_name"]


def test_record_failures_are_classified_and_isolated(lay, img):
    reqs = [req(kimi(), f"item {i}", images=[img]) for i in range(5)]
    st, pairs = queue(lay, reqs)
    led = Ledger(lay)
    keys = [k for k, _ in pairs]

    def answer(r):
        i = keys.index(r["recordId"])
        if i == 0:
            return {"recordId": r["recordId"], "error": {"errorCode": "ThrottlingException", "errorMessage": "slow"}}
        if i == 1:
            return {"recordId": r["recordId"], "error": {"errorCode": "ValidationException", "errorMessage": "bad"}}
        if i == 2:
            return None                                                  # missing output
        if i == 3:
            return {"recordId": r["recordId"], "modelOutput": converse_out("\\begin{tikz", stop="max_tokens")}
        return {"recordId": r["recordId"], "modelOutput": converse_out("not json at all")}
    be = FakeBackend(answer)
    doc = submit_job(st, be, led, prepare_job(st, led, pairs, lane=lane_key(reqs[0])))
    be.complete("PartiallyCompleted")
    doc = collect_job(st, be, led, refresh_job(st, be, doc))
    o = {keys.index(r["key"]): r["outcome"] for r in doc["records"]}
    assert o[0]["status"] == "service_error" and o[0]["retryable"] is True
    assert o[1]["status"] == "service_error" and o[1]["retryable"] is False
    assert o[2]["status"] == "missing_output"
    assert o[3]["status"] == "ok" and o[3]["stop_reason"] == "max_tokens"   # truncation is an answer: cached
    assert o[4]["status"] == "ok"                                           # invalid output: parser's business
    led2 = Ledger(lay)
    assert led2.in_flight() == 0 and len([e for e in led2.events() if e["event"] == "settle"]) == 2
    # the router: non-retryable -> ProviderError, retryable -> requeued to batch (no synchronous fallback)
    router = HybridRouter(st, led2, Policy("window", fill_target=100), matrix=matrix(), retry_max=1)
    with pytest.raises(PV.ProviderError, match="no synchronous fallback"):
        with router.live_slot(pairs[1][1], keys[1], layout=lay, allow_live=True, fake=False):
            pass
    with pytest.raises(PV.BatchDeferred):
        with router.live_slot(pairs[0][1], keys[0], layout=lay, allow_live=True, fake=False):
            pass
    assert keys[0] in router.queued


def test_job_level_failure_releases_everything(lay, img):
    reqs = [req(kimi(), f"i{i}", images=[img]) for i in range(3)]
    st, pairs = queue(lay, reqs)
    led = Ledger(lay)
    be = FakeBackend(lambda r: None)
    doc = submit_job(st, be, led, prepare_job(st, led, pairs, lane=lane_key(reqs[0])))
    be.complete("Failed")
    doc = collect_job(st, be, led, refresh_job(st, be, doc))
    assert {r["outcome"]["status"] for r in doc["records"]} == {"job_failed"}
    assert Ledger(lay).in_flight() == 0 and Ledger(lay).spent() == 0


def test_prepared_job_resumes_without_a_second_submission(lay, img):
    reqs = [req(kimi(), f"i{i}", images=[img]) for i in range(3)]
    st, pairs = queue(lay, reqs)
    led = Ledger(lay)
    doc = prepare_job(st, led, pairs, lane=lane_key(reqs[0]))          # crash here: nothing sent yet
    assert st.active_keys().keys() == {k for k, _ in pairs}
    be = FakeBackend()
    doc = submit_job(st, be, Ledger(lay), st.job(doc["job_name"]))
    doc = submit_job(st, be, Ledger(lay), doc)                         # idempotent: state already submitted
    assert be.submits == 1 and doc["state"] == "submitted"
    reserves = [e for e in Ledger(lay).events() if e["event"] == "reserve"]
    assert len(reserves) == 3 and {e["lane"] for e in reserves} == {"batch"}


def test_submit_refusal_is_classified_not_mislabelled(lay, img):
    class E(Exception):
        def __init__(self, code, msg):
            super().__init__(msg)
            self.response = {"Error": {"Code": code}}
    reqs = [req(kimi(), "a", images=[img])]
    st, pairs = queue(lay, reqs)
    led = Ledger(lay)
    doc = submit_job(st, FakeBackend(fail_submit=E("ValidationException", "Batch inference is not supported for the "
                                                                          "requested model")), led,
                     prepare_job(st, led, pairs, lane=lane_key(reqs[0])))
    assert doc["state"] == "submit_failed" and doc["submit_error"]["state"] == "UNSUPPORTED_LIVE_CONFIRMED"
    assert Ledger(lay).in_flight() == 0
    assert classify_submit_error("AccessDeniedException", "not authorized") == "INCONCLUSIVE_ACCESS_IAM_QUOTA_REGION"
    assert classify_submit_error("ValidationException", "minimum number of records is 100") == \
        "INCONCLUSIVE_FORMAT_OR_MINIMUM"
    assert classify_submit_error("ServiceQuotaExceededException", "x") == "INCONCLUSIVE_ACCESS_IAM_QUOTA_REGION"
    assert classify_submit_error("ValidationException", "Malformed input request") == "INCONCLUSIVE_FORMAT_OR_MINIMUM"
    assert classify_record_error({"errorCode": "ModelTimeoutException", "errorMessage": "t"})[1] is True


def test_no_batch_rate_means_no_batch_lane(lay, img):
    led = Ledger(lay)
    with pytest.raises(PriceMissing, match="no discount is assumed"):
        led.batch_price("qwen.judge")
    spec = kimi().model_copy(update={"model_id": "qwen.judge", "unit_price_id": "od/qwen.judge"})
    m = matrix(model_id="qwen.judge")
    router = HybridRouter(BatchStore(lay), led, Policy("window"), matrix=m, retry_max=0, plan_only=True)
    with pytest.raises(PV.BatchDeferred):
        with router.live_slot(req(spec, images=[img]), "k" * 64, layout=lay, allow_live=True, fake=False):
            pass
    assert router.plan[0]["route"] == "synchronous" and "no dated batch-lane rate" in router.plan[0]["reason"]


def test_cancel_stops_only_the_named_job(lay, img):
    st, pairs = queue(lay, [req(kimi(), "a", images=[img])])
    be = FakeBackend()
    doc = submit_job(st, be, Ledger(lay), prepare_job(st, Ledger(lay), pairs, lane=lane_key(pairs[0][1])))
    st2, pairs2 = queue(lay, [req(kimi(), "b", images=[img])])
    doc2 = submit_job(st2, be, Ledger(lay), prepare_job(st2, Ledger(lay), pairs2, lane=lane_key(pairs2[0][1])))
    cancel_job(st, be, doc["job_name"])
    assert be.stopped == [doc["job_name"]] and doc2["job_name"] not in be.stopped
    be.complete()
    collect_job(st, be, Ledger(lay), refresh_job(st, be, st.job(doc["job_name"])))
    with pytest.raises(ValueError, match="nothing to cancel"):
        cancel_job(st, be, doc["job_name"])


def test_config_validation(tmp_path):
    good = {"version": "batch-1", "region": "us-east-1", "s3_bucket": "b", "s3_prefix": "hmb/run",
            "role_arn": "arn:aws:iam::123456789012:role/service-role/r"}
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(good))
    assert load_config(p).poll_seconds == 60 and load_config(p).compile_workers == 1
    p.write_text(yaml.safe_dump({**good, "compile_workers": 6}))
    assert load_config(p).compile_workers == 6
    for bad in ({**good, "secret_key": "x"}, {**good, "role_arn": "nope"}, {**good, "s3_prefix": "/x/"},
                {**good, "timeout_hours": 2}, {**good, "workers": 0}, {**good, "compile_workers": 0}):
        p.write_text(yaml.safe_dump(bad))
        with pytest.raises(BatchConfigError):
            load_config(p)


def test_bedrock_backend_request_shape():
    calls = {}

    class B:
        def create_model_invocation_job(self, **kw):
            calls["job"] = kw
            return {"jobArn": "arn:x/job"}

    class S:
        def put_object(self, **kw):
            calls["put"] = kw
    cfg = load_config_from({"version": "batch-1", "region": "us-east-1", "s3_bucket": "bkt", "s3_prefix": "hmb/run",
                            "role_arn": "arn:aws:iam::123456789012:role/service-role/r", "tags": {"Project": "HMB"}})
    BedrockBatchBackend(cfg, bedrock=B(), s3=S()).submit("hmb-x-1", "moonshotai.kimi-k2.5", b"{}\n", {"t": "v"})
    j = calls["job"]
    assert j["modelInvocationType"] == "Converse" and j["clientRequestToken"] == "hmb-x-1"
    assert j["inputDataConfig"]["s3InputDataConfig"]["s3Uri"] == "s3://bkt/hmb/run/jobs/hmb-x-1/input/records.jsonl"
    assert j["outputDataConfig"]["s3OutputDataConfig"]["s3Uri"] == "s3://bkt/hmb/run/jobs/hmb-x-1/output/"
    assert {"key": "Project", "value": "HMB"} in j["tags"] and calls["put"]["Bucket"] == "bkt"


def load_config_from(d):
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.safe_dump(d, f)
    return load_config(f.name)


# --------------------------------------------------------------------------- 11-15 router and scheduler
class SlowProvider:
    def __init__(self, delays: dict[str, float]):
        self.delays, self.lock = delays, threading.Lock()
        self.active: dict[str, int] = {}
        self.peak: dict[str, int] = {}

    def send(self, r, images):
        m = r.spec.model_id
        with self.lock:
            self.active[m] = self.active.get(m, 0) + 1
            self.peak[m] = max(self.peak.get(m, 0), self.active[m])
        time.sleep(self.delays.get(m, 0.01))
        with self.lock:
            self.active[m] -= 1
        return {"text": "ok", "stop_reason": "end_turn", "input_tokens": 3, "output_tokens": 2}


def _sched_env(lay, monkeypatch, provider, policy="sync", fill=100, window=0.3, conc=None, **pol):
    monkeypatch.setattr(PV, "adapter_for", lambda name: provider)
    led = Ledger(lay)
    router = HybridRouter(BatchStore(lay), led, Policy(policy, fill_target=fill, window_seconds=window, **pol),
                          matrix=matrix(), sync_concurrency=conc or {}, default_sync_concurrency=2, retry_max=0)
    return led, router


def _cell(lay, led, model_id, n_judge, log, label):
    judge = kimi().model_copy(update={"model_id": "qwen.judge", "unit_price_id": "od/qwen.judge", "role": "judge"})

    def run():
        c = kimi().model_copy(update={"model_id": model_id, "unit_price_id": f"od/{model_id}"})
        PV.call_model(req(c, f"contestant {label}"), lay, led, allow_live=True)
        log.append(("compiled", label, time.monotonic()))
        deferred = []
        for k in range(n_judge):
            try:
                PV.call_model(req(judge, f"judge {label} {k}"), lay, led, allow_live=True)
            except PV.BatchDeferred as d:
                deferred.append(d)
        if deferred:
            raise PV.BatchDeferred.merge(deferred)
        log.append(("adjudicated", label, time.monotonic()))
        return label
    return run


def test_fast_cells_reach_adjudication_while_a_slow_model_runs(lay, monkeypatch):
    slow_price = {"unit_price_id": "od/slow.model", "model_id": "slow.model", "usd_per_mtok_input": 1,
                  "usd_per_mtok_output": 1, "pricing_url": "https://aws.amazon.com/bedrock/pricing/",
                  "retrieved_utc": "2026-09-26"}
    card = yaml.safe_load(lay.pricing.read_text())
    card["entries"].append(slow_price)
    lay.pricing.write_text(yaml.safe_dump(card))
    prov = SlowProvider({"slow.model": 0.6, "moonshotai.kimi-k2.5": 0.01, "qwen.judge": 0.01})
    led, router = _sched_env(lay, monkeypatch, prov)
    log: list = []
    tasks = {f"slow|{i}": _cell(lay, led, "slow.model", 2, log, f"slow{i}") for i in range(2)}
    tasks.update({f"fast|{i}": _cell(lay, led, "moonshotai.kimi-k2.5", 3, log, f"fast{i}") for i in range(4)})
    PV.set_router(router)
    try:
        out = HybridScheduler(router, FakeBackend(), workers=6, tick_seconds=0.01).run(tasks)
    finally:
        PV.set_router(None)
    assert not out["parked"] and len(out["results"]) == 6
    done = {lbl: t for ev, lbl, t in log if ev == "adjudicated"}
    first_slow_contestant_done = min(t for ev, lbl, t in log if ev == "compiled" and lbl.startswith("slow"))
    assert all(done[f"fast{i}"] < first_slow_contestant_done for i in range(4))   # no global barrier
    assert prov.peak["qwen.judge"] <= 2 and prov.peak["slow.model"] <= 2          # bounded per-model concurrency


def test_backpressure_never_exceeds_workers(lay, monkeypatch):
    prov = SlowProvider({"moonshotai.kimi-k2.5": 0.02, "qwen.judge": 0.02})
    led, router = _sched_env(lay, monkeypatch, prov, conc={"moonshotai.kimi-k2.5": 50, "qwen.judge": 50})
    active, peak, lock = [0], [0], threading.Lock()

    def wrap(f):
        def g():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            try:
                return f()
            finally:
                with lock:
                    active[0] -= 1
        return g
    log: list = []
    tasks = {f"t{i}": wrap(_cell(lay, led, "moonshotai.kimi-k2.5", 1, log, f"c{i}")) for i in range(20)}
    PV.set_router(router)
    try:
        HybridScheduler(router, FakeBackend(), workers=3, tick_seconds=0.005).run(tasks)
    finally:
        PV.set_router(None)
    assert peak[0] <= 3


def _batch_run(lay, monkeypatch, n, policy, fill, window, backend=None):
    prov = SlowProvider({})
    led, router = _sched_env(lay, monkeypatch, prov, policy=policy, fill=fill, window=window)
    log: list = []
    tasks = {f"t{i}": _cell(lay, led, "moonshotai.kimi-k2.5", 1, log, f"c{i}") for i in range(n)}
    be = backend or FakeBackend()
    orig = be.submit

    def submit_and_finish(*a, **k):
        arn = orig(*a, **k)
        be.complete()
        return arn
    be.submit = submit_and_finish
    PV.set_router(router)
    try:
        out = HybridScheduler(router, be, workers=4, poll_seconds=0.0, tick_seconds=0.005).run(tasks)
    finally:
        PV.set_router(None)
    return out, router, be, led


def test_fixed_size_flush_and_spill_below_minimum(lay, monkeypatch):
    out, router, be, led = _batch_run(lay, monkeypatch, 5, "fixed", fill=4, window=100)
    assert not out["parked"] and len(out["results"]) == 5
    sizes = [len(j["recs"]) for j in be.jobs.values()]
    assert sizes == [4]                                  # one full job; the 1 leftover < min_records 2 -> spilled
    assert router.metrics.spills == 1 and router.metrics.sync_calls == 1 + 5     # 1 spilled contestant + 5 judges
    lanes = [e.get("lane") for e in led.events() if e["event"] == "settle"]
    assert lanes.count("batch") == 4


def test_window_flush_holds_only_a_bounded_time(lay, monkeypatch):
    out, router, be, _ = _batch_run(lay, monkeypatch, 3, "window", fill=100, window=0.05)
    assert not out["parked"] and len(be.jobs) == 1 and len(next(iter(be.jobs.values()))["recs"]) == 3


def test_sync_policy_never_batches(lay, monkeypatch):
    out, router, be, _ = _batch_run(lay, monkeypatch, 4, "sync", fill=2, window=0.0)
    assert not be.jobs and router.metrics.sync_calls == 8 and not out["parked"]


def test_interrupted_scheduler_resumes_without_duplicates(lay, monkeypatch):
    prov = SlowProvider({})
    led, router = _sched_env(lay, monkeypatch, prov, policy="fixed", fill=3, window=100)
    log: list = []
    mk = lambda: {f"t{i}": _cell(lay, led, "moonshotai.kimi-k2.5", 1, log, f"c{i}") for i in range(3)}  # noqa: E731
    be = FakeBackend()                                                   # jobs stay InProgress
    PV.set_router(router)
    try:
        out = HybridScheduler(router, be, workers=3, poll_seconds=0.0, tick_seconds=0.005,
                              max_wall_seconds=0.2).run(mk())
    finally:
        PV.set_router(None)
    assert out["stopped_early"] and be.submits == 1 and len(BatchStore(lay).active_keys()) == 3
    be.complete()                                                        # the job finishes while we are down
    led2 = Ledger(lay)
    router2 = HybridRouter(BatchStore(lay), led2, Policy("fixed", fill_target=3, window_seconds=100), matrix=matrix(),
                           default_sync_concurrency=2, retry_max=0)
    PV.set_router(router2)
    try:
        out2 = HybridScheduler(router2, be, workers=3, poll_seconds=0.0, tick_seconds=0.005).run(mk())
    finally:
        PV.set_router(None)
    assert not out2["parked"] and len(out2["results"]) == 3 and be.submits == 1      # nothing resubmitted
    ev = Ledger(lay).events()
    assert len([e for e in ev if e["event"] == "settle" and e.get("lane") == "batch"]) == 3
    assert len([e for e in ev if e["event"] == "reserve" and e.get("lane") == "batch"]) == 3


def test_single_flight_no_duplicate_paid_call(lay, monkeypatch):
    prov = SlowProvider({"qwen.judge": 0.2})
    led, router = _sched_env(lay, monkeypatch, prov)
    judge = kimi().model_copy(update={"model_id": "qwen.judge", "unit_price_id": "od/qwen.judge", "role": "judge"})
    same = req(judge, "identical")
    tasks = {f"t{i}": (lambda: PV.call_model(same, lay, led, allow_live=True).text) for i in range(4)}
    PV.set_router(router)
    try:
        out = HybridScheduler(router, FakeBackend(), workers=4, tick_seconds=0.01).run(tasks)
    finally:
        PV.set_router(None)
    assert all(v == "ok" for v in out["results"].values())
    assert len([e for e in Ledger(lay).events() if e["event"] == "settle"]) == 1


def test_structured_output_blocked_role_stays_synchronous_in_a_hybrid_flow(lay, monkeypatch, img):
    prov = SlowProvider({})
    led, router = _sched_env(lay, monkeypatch, prov, policy="window", fill=2, window=0.05)
    be = FakeBackend()
    orig = be.submit

    def fin(*a, **k):
        arn = orig(*a, **k)
        be.complete()
        return arn
    be.submit = fin

    def gatekeeper():
        return PV.call_model(req(luna(), "g8b", images=[img], response_schema=SCHEMA), lay, led, allow_live=True).text

    def contestant(i):
        return lambda: PV.call_model(req(kimi(), f"c{i}", images=[img]), lay, led, allow_live=True).text
    tasks = {"g8b": gatekeeper, **{f"c{i}": contestant(i) for i in range(2)}}
    PV.set_router(router)
    try:
        out = HybridScheduler(router, be, workers=3, poll_seconds=0.0, tick_seconds=0.005).run(tasks)
    finally:
        PV.set_router(None)
    assert not out["parked"] and len(be.jobs) == 1
    assert [r["modelInput"]["messages"][0]["content"][-1]["text"] for r in next(iter(be.jobs.values()))["recs"]] \
        and all("g8b" not in json.dumps(r) for j in be.jobs.values() for r in j["recs"])
    settles = Ledger(lay).events()
    luna_rows = [e for e in settles if e["event"] == "settle" and e["model_id"] == "global.openai.gpt-6-luna"]
    assert len(luna_rows) == 1 and luna_rows[0].get("lane") is None                 # on-demand


# --------------------------------------------------------------------------- CLI
def test_cli_help_and_route_plan(tmp_path, capsys):
    from hmb.cli import main
    for cmd in ("batch-plan", "batch-run", "batch-status", "batch-collect", "batch-cancel", "batch-probe", "batch-cost"):
        with pytest.raises(SystemExit) as e:
            main([cmd, "--help"])
        assert e.value.code == 0
    roster = {"version": "roster-1", "verifier": [
        {"model_id": "global.openai.gpt-6-luna", "provider": "bedrock", "family": "openai", "effort": "high",
         "decoding": {"additional_model_request_fields": {"reasoning": {"effort": "high"}}},
         "structured_output": "native_json_schema", "unit_price_id": "x"}],
        "contestant": [{"model_id": "moonshotai.kimi-k2.5", "provider": "bedrock", "family": "moonshot",
                        "unit_price_id": "y", "snapshot": "s"}]}
    rp = tmp_path / "r.yaml"
    rp.write_text(yaml.safe_dump(roster))
    capsys.readouterr()
    assert main(["batch-plan", "--roster", str(rp)]) == 0
    out = json.loads(capsys.readouterr().out)
    routes = {r["model_id"]: r for r in out["routes"]}
    assert routes["global.openai.gpt-6-luna"]["route"] == "synchronous"
    assert "UNSUPPORTED_LIVE_CONFIRMED" in json.dumps(CAP.load_matrix()) and routes["moonshotai.kimi-k2.5"]["reason"]


def test_global_cris_route_and_stale_ids_refused():
    from hmb.roster import check_roster, route_record
    stale = [ModelSpec(model_id="openai.gpt-6-sol", provider="bedrock_mantle", family="openai", role=Role.builder,
                       effort="medium", endpoint="https://bedrock-mantle.us-east-1.api.aws/openai/v1", decoding={},
                       unit_price_id="x", snapshot="s"),
             ModelSpec(model_id="us.openai.gpt-6-sol", provider="bedrock", family="openai", role=Role.builder,
                       effort="medium", decoding={"additional_model_request_fields": {"reasoning": {"effort": "medium"}}},
                       unit_price_id="x", snapshot="s")]
    probs = " | ".join(check_roster({"builder": stale, "judge": [], "contestant": [], "verifier": []}))
    assert probs.count("retired route; use global.openai.gpt-6-sol") == 2
    g = ModelSpec(model_id="global.openai.gpt-6-sol", provider="bedrock", family="openai", role=Role.builder,
                  effort="medium", decoding={"additional_model_request_fields": {"reasoning": {"effort": "medium"}}},
                  unit_price_id="x", snapshot="s")
    probs = check_roster({"builder": [g], "judge": [], "contestant": [], "verifier": []})
    assert not any("retired" in p or "Global CRIS" in p for p in probs)
    assert route_record(g)["inference_scope"] == "global_cris" and route_record(g)["endpoint"] == "bedrock-runtime.us-east-1"
    wrong = g.model_copy(update={"provider": "bedrock_mantle", "endpoint": "https://bedrock-mantle.us-east-1.api.aws/openai/v1"})
    assert any("served on bedrock-runtime" in p for p in check_roster({"builder": [wrong], "judge": [], "contestant": [],
                                                                        "verifier": []}))


def test_fill_target_below_the_service_minimum_never_submits_a_short_job(lay, monkeypatch):
    out, router, be, _ = _batch_run(lay, monkeypatch, 3, "window", fill=1, window=100)   # min_records is 2
    assert not out["parked"] and all(len(j["recs"]) >= 2 for j in be.jobs.values())


def test_writer_roster_sonnet5_and_kimi():
    """Owner decision 2026-09-26: Sonnet 5 + Kimi K2.5 write the answer key; Kimi has no effort control and no longer
    competes; its writer calls are batch-routed, Sonnet 5 stays synchronous (not probed, no batch SKU)."""
    from hmb.roster import check_roster, effort_problems
    sonnet = ModelSpec(model_id="us.anthropic.claude-sonnet-5", provider="bedrock", family="anthropic", role=Role.builder,
                       effort="medium", unit_price_id="x", snapshot="s", decoding={"additional_model_request_fields": {
                           "thinking": {"type": "adaptive"}, "output_config": {"effort": "medium"}}})
    kimi_w = ModelSpec(model_id="moonshotai.kimi-k2.5", provider="bedrock", family="moonshot", role=Role.builder,
                       unit_price_id="x", snapshot="s")
    assert effort_problems({"builder": [sonnet, kimi_w]}) == []
    assert "no effort control" in effort_problems({"builder": [kimi_w.model_copy(update={"effort": "medium"})]})[0]
    contestant = kimi_w.model_copy(update={"role": Role.contestant})
    probs = " | ".join(check_roster({"builder": [sonnet, kimi_w], "judge": [], "verifier": [], "contestant": [contestant]}))
    assert "writes the answer key" in probs and "not in the locked roster" in probs
    m = CAP.load_matrix()
    assert CAP.decide(req(kimi_w, images=[Path(CAP.MATRIX_PATH)]), m).route == "batch"
    d = CAP.decide(req(sonnet, images=[Path(CAP.MATRIX_PATH)]), m)
    assert d.route == "synchronous" and "not probed" in d.reason


def test_cards_build_reports_batch_deferrals(monkeypatch, tmp_path):
    """cards_build turns batch deferrals into one BatchDeferred (replayed by the scheduler), never 'missing'."""
    import hmb.cards.batch as CB
    from hmb.config import fixture_thresholds
    lay = Layout(tmp_path / "r")
    (lay.p("data", "candidates")).mkdir(parents=True)
    crop = lay.p("images", "crops_canonical", "i1.png")
    crop.parent.mkdir(parents=True)
    Image.new("L", (8, 8), 255).save(crop)
    (lay.p("data", "candidates", "i1.json")).write_text(json.dumps(
        {"id": "i1", "assets": {"crop_canonical_image": "crops_canonical/i1.png"}}))

    def fake_build(*a, **kw):
        raise PV.BatchDeferred({"k" * 64})
    monkeypatch.setattr(CB, "build_card", fake_build)
    monkeypatch.setattr(CB, "check_builder_roster", lambda b: None)
    with pytest.raises(PV.BatchDeferred) as e:
        CB.cards_build(lay, fixture_thresholds(), builders=[], judges=[], allow_live=True, phase="DEV", ledger=None)
    assert e.value.keys == {"k" * 64} and not lay.card("i1").exists()
