"""Batch config, Converse-type batch records, the S3/Bedrock job backend, job manifests and ingestion.

Job lifecycle (all state under `<root>/manifests/batch/`; every change also appended to `events.jsonl`):
  prepared   manifest written and one ledger reservation per record made (batch-lane estimate), BEFORE
             anything is sent; resubmitting a `prepared` job reuses its clientRequestToken, so an
             interrupted submission never creates a second job
  submitted  AWS accepted the job; the job ARN is added atomically
  <AWS status>  InProgress / Completed / PartiallyCompleted / Failed / Stopped / Expired ... (status/poll)
  collected  output retrieved and every record classified exactly once:
               ok               cache entry written, ledger settled at the batch rate
               service_error    the record's `error` (retryable or not, from its code)
               missing_output   no output line for the record
               job_failed       the job ended without output for the record
             A truncated or invalid model answer is an ANSWER: it is cached like a synchronous one and the
             caller's parser handles it (judge retries, E_TRUNCATED, card malformed), exactly as today.
  submit_failed  AWS refused the job; reservations released; the refusal is classified.

Record ids are the call's cache key, so out-of-order output lines correlate exactly and a line for a key
that is not in the job, or a second line for the same key, is ignored and logged.
"""
from __future__ import annotations

import base64
import json
import re
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hmb.schemas import ModelRequest
from hmb.store import Layout, append_jsonl, read_json, write_json_immutable, write_mutable

CONFIG_VERSION = "batch-1"
TERMINAL = ("Completed", "PartiallyCompleted", "Failed", "Stopped", "Expired")
# Record-level service errors worth resubmitting (the transport-error class of call_model's retry policy).
RETRYABLE_CODES = ("ThrottlingException", "ServiceUnavailableException", "InternalServerException",
                   "ModelTimeoutException", "ModelNotReadyException", "TooManyRequestsException", "429", "500", "503")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class BatchConfig:
    """`--batch-config` YAML (template: configs/batch.template.yaml). No secret is ever stored here."""
    region: str
    s3_bucket: str
    s3_prefix: str
    role_arn: str
    poll_seconds: int = 60
    timeout_hours: int = 24
    max_records_per_job: int = 10000
    fill_target: int = 1000          # records that flush a lane at once
    window_seconds: int = 900        # oldest-record age that flushes a lane (>= min_records) or spills it to sync
    workers: int = 8                 # concurrent cells (bounded; backpressure)
    sync_concurrency: dict = field(default_factory=dict)   # model_id -> concurrent synchronous calls
    default_sync_concurrency: int = 4
    compile_workers: int = 1         # concurrent scoring compiles; >1 only after the C24 check on that host
    cpu_workers: int = 1             # worker processes for CGX/alignment (hmb.cpu_pool); 1 = inline
    kms_key_id: str | None = None
    tags: dict = field(default_factory=dict)

    @property
    def is_real(self) -> bool:
        return bool(self.s3_bucket and self.role_arn)


_CFG_KEYS = {"version", "region", "s3_bucket", "s3_prefix", "role_arn", "poll_seconds", "timeout_hours",
             "max_records_per_job", "fill_target", "window_seconds", "workers", "sync_concurrency",
             "default_sync_concurrency", "compile_workers", "cpu_workers", "kms_key_id", "tags"}


class BatchConfigError(ValueError):
    pass


def load_config(path: str | Path) -> BatchConfig:
    d = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if d.get("version") != CONFIG_VERSION:
        raise BatchConfigError(f"{path}: expected version {CONFIG_VERSION}")
    extra = set(d) - _CFG_KEYS
    if extra:
        raise BatchConfigError(f"{path}: unknown keys {sorted(extra)}")
    for k in ("region", "s3_bucket", "s3_prefix", "role_arn"):
        if not isinstance(d.get(k), str) or not d[k]:
            raise BatchConfigError(f"{path}: {k} is required")
    if not re.fullmatch(r"arn:aws:iam::\d{12}:role/[\w+=,.@/-]+", d["role_arn"]):
        raise BatchConfigError(f"{path}: role_arn is not an IAM role ARN")
    if d["s3_prefix"].startswith("/") or d["s3_prefix"].endswith("/"):
        raise BatchConfigError(f"{path}: s3_prefix has no leading or trailing '/'")
    ints = ("poll_seconds", "timeout_hours", "max_records_per_job", "fill_target", "window_seconds", "workers",
            "default_sync_concurrency", "compile_workers", "cpu_workers")
    for k in ints:
        if k in d and (not isinstance(d[k], int) or d[k] < 1):
            raise BatchConfigError(f"{path}: {k} must be a positive integer")
    if not 24 <= int(d.get("timeout_hours", 24)) <= 168:
        raise BatchConfigError(f"{path}: timeout_hours must be within 24..168 (service range)")
    sc = d.get("sync_concurrency") or {}
    if not isinstance(sc, dict) or any(not isinstance(v, int) or v < 1 for v in sc.values()):
        raise BatchConfigError(f"{path}: sync_concurrency maps model_id -> positive int")
    kw = {k: d[k] for k in _CFG_KEYS - {"version"} if k in d and d[k] is not None}
    kw["sync_concurrency"] = dict(sc)
    kw["tags"] = {str(k): str(v) for k, v in (d.get("tags") or {}).items()}
    return BatchConfig(**kw)


# --------------------------------------------------------------------------- records
def record_id(key: str) -> str:
    return key                                   # the cache key: collision-resistant and self-correlating


def batch_record(req: ModelRequest, key: str) -> dict:
    """One Converse-type JSONL record: the synchronous Converse body (modelId removed, image bytes base64 as
    in the Converse JSON wire format). Built by the same function the synchronous adapter uses."""
    from hmb.providers import _mime, converse_request
    images = [(Path(p).read_bytes(), _mime(p)) for p in req.image_paths]
    body = converse_request(req, images)
    body.pop("modelId")
    for m in body["messages"]:
        for c in m["content"]:
            if "image" in c:
                c["image"]["source"]["bytes"] = base64.b64encode(c["image"]["source"]["bytes"]).decode("ascii")
    return {"recordId": record_id(key), "modelInput": body}


def job_model_id(req: ModelRequest) -> str:
    return req.spec.snapshot or req.spec.model_id    # the id the synchronous Converse call sends


def lane_key(req: ModelRequest) -> str:
    """Records share a job only when model, Region, request family, modality, schema mode, decoding, effort
    and role contract all match."""
    from hmb.hashing import sha256_json
    from hmb.providers import native_schema
    from hmb.roster import BEDROCK_REGION
    sig = {"model": job_model_id(req), "region": BEDROCK_REGION, "family": "converse", "provider": req.spec.provider,
           "image": bool(req.image_paths), "schema": bool(native_schema(req)), "decoding": req.spec.decoding,
           "effort": req.spec.effort, "role": str(req.spec.role), "structured_output": req.spec.structured_output}
    return f"{re.sub(r'[^a-z0-9]+', '-', job_model_id(req).lower()).strip('-')[:40]}-{sha256_json(sig)[:10]}"


def classify_submit_error(code: str, message: str) -> str:
    """Map a CreateModelInvocationJob refusal to a capability state (never 'unsupported' for access/format)."""
    m = (message or "").lower()
    if code in ("AccessDeniedException", "UnrecognizedClientException", "ExpiredTokenException") \
            or "not authorized" in m or "access denied" in m:
        return "INCONCLUSIVE_ACCESS_IAM_QUOTA_REGION"
    if code in ("ServiceQuotaExceededException", "ThrottlingException", "TooManyRequestsException"):
        return "INCONCLUSIVE_ACCESS_IAM_QUOTA_REGION"
    if "minimum" in m and "record" in m:
        return "INCONCLUSIVE_FORMAT_OR_MINIMUM"
    if code == "ValidationException" and re.search(r"(does not support|not supported|unsupported).*(batch|model invocation job)"
                                                  r"|batch.*(not supported|unsupported)", m):
        return "UNSUPPORTED_LIVE_CONFIRMED"
    if code == "ValidationException" and ("region" in m):
        return "INCONCLUSIVE_ACCESS_IAM_QUOTA_REGION"
    return "INCONCLUSIVE_FORMAT_OR_MINIMUM"


def classify_record_error(err: Any) -> tuple[str, bool]:
    """(detail, retryable) of a record-level `error` field."""
    if isinstance(err, dict):
        code = str(err.get("errorCode") or err.get("code") or "")
        msg = str(err.get("errorMessage") or err.get("message") or "")
    else:
        code, msg = "", str(err)
    return f"{code}: {msg}"[:500], any(c and (c == code or c in msg) for c in RETRYABLE_CODES)


# --------------------------------------------------------------------------- backend
class BedrockBatchBackend:
    """CreateModelInvocationJob (modelInvocationType=Converse) with S3 input/output under
    s3://<bucket>/<prefix>/jobs/<job_name>/. Clients are injectable for tests."""

    def __init__(self, cfg: BatchConfig, bedrock=None, s3=None):
        import boto3
        self.cfg = cfg
        self.bedrock = bedrock or boto3.client("bedrock", region_name=cfg.region)
        self.s3 = s3 or boto3.client("s3", region_name=cfg.region)

    def _base(self, job_name: str) -> str:
        return f"{self.cfg.s3_prefix}/jobs/{job_name}"

    def uris(self, job_name: str) -> tuple[str, str]:
        b = self._base(job_name)
        return f"s3://{self.cfg.s3_bucket}/{b}/input/records.jsonl", f"s3://{self.cfg.s3_bucket}/{b}/output/"

    def submit(self, job_name: str, model_id: str, jsonl: bytes, tags: dict) -> str:
        in_uri, out_uri = self.uris(job_name)
        put: dict[str, Any] = {"Bucket": self.cfg.s3_bucket, "Key": f"{self._base(job_name)}/input/records.jsonl",
                               "Body": jsonl}
        if self.cfg.kms_key_id:
            put.update(ServerSideEncryption="aws:kms", SSEKMSKeyId=self.cfg.kms_key_id)
        self.s3.put_object(**put)
        out_cfg: dict[str, Any] = {"s3Uri": out_uri}
        if self.cfg.kms_key_id:
            out_cfg["s3EncryptionKeyId"] = self.cfg.kms_key_id
        r = self.bedrock.create_model_invocation_job(
            jobName=job_name, roleArn=self.cfg.role_arn, clientRequestToken=job_name.replace(".", "-"),
            modelId=model_id, modelInvocationType="Converse",
            inputDataConfig={"s3InputDataConfig": {"s3Uri": in_uri, "s3InputFormat": "JSONL"}},
            outputDataConfig={"s3OutputDataConfig": out_cfg}, timeoutDurationInHours=self.cfg.timeout_hours,
            tags=[{"key": k, "value": v} for k, v in {**self.cfg.tags, **tags}.items()])
        return r["jobArn"]

    def status(self, job_arn: str) -> dict:
        r = self.bedrock.get_model_invocation_job(jobIdentifier=job_arn)
        return {"status": r.get("status"), "message": r.get("message"),
                "submit_time": str(r.get("submitTime") or ""), "end_time": str(r.get("endTime") or "")}

    def outputs(self, job_name: str) -> tuple[list[str], dict | None]:
        """Output JSONL lines (every *.jsonl.out under the job's output prefix) and manifest.json.out."""
        prefix = f"{self._base(job_name)}/output/"
        lines: list[str] = []
        manifest = None
        pag = self.s3.get_paginator("list_objects_v2")
        for page in pag.paginate(Bucket=self.cfg.s3_bucket, Prefix=prefix):
            for o in page.get("Contents", []) or []:
                k = o["Key"]
                if k.endswith(".jsonl.out") or k.endswith("manifest.json.out"):
                    body = self.s3.get_object(Bucket=self.cfg.s3_bucket, Key=k)["Body"].read().decode("utf-8")
                    if k.endswith("manifest.json.out"):
                        manifest = json.loads(body)
                    else:
                        lines += [ln for ln in body.splitlines() if ln.strip()]
        return lines, manifest

    def stop(self, job_arn: str) -> None:
        self.bedrock.stop_model_invocation_job(jobIdentifier=job_arn)


# --------------------------------------------------------------------------- store
class BatchStore:
    """Job manifests and queued requests under <root>/manifests/batch/ (resumable, idempotent)."""

    def __init__(self, layout: Layout):
        self.layout = layout
        self.dir = layout.p("manifests", "batch")
        self._lock = threading.RLock()

    def request_path(self, key: str) -> Path:
        return self.dir / "requests" / key[:2] / f"{key}.json"

    def job_path(self, name: str) -> Path:
        return self.dir / "jobs" / f"{name}.json"

    def log(self, **rec) -> None:
        append_jsonl(self.dir / "events.jsonl", {"utc": _utc(), **rec})

    def save_request(self, req: ModelRequest, key: str, lane: str) -> None:
        write_json_immutable(self.request_path(key), {"cache_key": key, "lane": lane,
                                                      "request": req.model_dump(mode="json")})

    def load_request(self, key: str) -> tuple[ModelRequest, str]:
        d = read_json(self.request_path(key))
        return ModelRequest.model_validate(d["request"]), d["lane"]

    def queued_keys(self) -> list[str]:
        """Saved requests not yet assigned to a job and not yet answered (resume restores the queues)."""
        d = self.dir / "requests"
        assigned = {r["key"] for j in self.jobs() for r in j["records"]}
        out = []
        for p in sorted(d.rglob("*.json")) if d.exists() else []:
            k = p.stem
            if k not in assigned and not (self.layout.model_call_cache / k[:2] / f"{k}.json").exists():
                out.append(k)
        return out

    def put_job(self, doc: dict) -> None:
        with self._lock:
            write_mutable(self.job_path(doc["job_name"]), (json.dumps(doc, indent=1, sort_keys=True) + "\n").encode())

    def job(self, name: str) -> dict:
        return read_json(self.job_path(name))

    def jobs(self) -> list[dict]:
        d = self.dir / "jobs"
        return [read_json(p) for p in sorted(d.glob("*.json"))] if d.exists() else []

    def active_keys(self) -> dict[str, str]:
        """key -> job name for every record of a job that is not yet collected or failed to submit."""
        out = {}
        for j in self.jobs():
            if j["state"] in ("prepared", "submitted"):
                for r in j["records"]:
                    out[r["key"]] = j["job_name"]
        return out

    def record_outcomes(self) -> dict[str, list[dict]]:
        """key -> every collected outcome of that key across jobs (attempt history)."""
        out: dict[str, list[dict]] = {}
        for j in self.jobs():
            if j["state"] == "collected" or j["state"] == "submit_failed":
                for r in j["records"]:
                    if r.get("outcome"):
                        out.setdefault(r["key"], []).append({"job": j["job_name"], **r["outcome"]})
        return out


# --------------------------------------------------------------------------- operations
def prepare_job(store: BatchStore, ledger, reqs: list[tuple[str, ModelRequest]], *, lane: str, phase_tag: str = "") -> dict:
    """Write the manifest and reserve every record at the batch-lane rate before anything is sent.
    Raises PriceMissing when no dated batch rate exists (no discount is ever assumed)."""
    first = reqs[0][1]
    model_id = job_model_id(first)
    price = ledger.batch_price(first.spec.model_id)
    slug, lh = lane.rsplit("-", 1)
    name = f"hmb-{slug[:20].strip('-')}-{lh}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:6]}"
    recs = []
    for key, req in reqs:
        if job_model_id(req) != model_id or lane_key(req) != lane:
            raise ValueError("records of one job must share one lane")
        est = float(ledger.estimate_usd(req, unit_price_id=price.unit_price_id))
        rid = ledger.reserve(req.phase, req.purpose, req.spec.model_id, est, lane="batch")
        recs.append({"key": key, "record_id": record_id(key), "purpose": req.purpose, "phase": req.phase,
                     "reservation_id": rid, "outcome": None})
    doc = {"job_name": name, "lane": lane, "model_id": model_id, "spec_model_id": first.spec.model_id,
           "batch_unit_price_id": price.unit_price_id, "invocation_type": "Converse", "state": "prepared",
           "aws_status": None, "job_arn": None, "created_utc": _utc(), "submitted_utc": None, "terminal_utc": None,
           "collected_utc": None, "records": recs, "tag": phase_tag, "submit_error": None, "manifest_out": None}
    store.put_job(doc)
    store.log(event="prepared", job=name, lane=lane, records=len(recs))
    return doc


def submit_job(store: BatchStore, backend, ledger, doc: dict) -> dict:
    """Upload and create the job (idempotent through the job-name clientRequestToken)."""
    if doc["state"] != "prepared":
        return doc
    lines = []
    for r in doc["records"]:
        req, _ = store.load_request(r["key"])
        lines.append(json.dumps(batch_record(req, r["key"]), ensure_ascii=False, sort_keys=True))
    body = ("\n".join(lines) + "\n").encode("utf-8")
    try:
        arn = backend.submit(doc["job_name"], doc["model_id"], body, {"hmb-lane": doc["lane"][:200]})
    except Exception as e:                                     # classified, reservations released
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        msg = str(e)
        doc.update(state="submit_failed", submit_error={"code": code, "message": msg[:800],
                                                        "state": classify_submit_error(code, msg)})
        for r in doc["records"]:
            ledger.release(r["reservation_id"])
            r["outcome"] = {"status": "submit_failed", "detail": f"{code}: {msg[:300]}", "retryable": False}
        store.put_job(doc)
        store.log(event="submit_failed", job=doc["job_name"], code=code)
        return doc
    doc.update(state="submitted", job_arn=arn, submitted_utc=_utc(), aws_status="Submitted")
    store.put_job(doc)
    store.log(event="submitted", job=doc["job_name"], records=len(doc["records"]))
    return doc


def refresh_job(store: BatchStore, backend, doc: dict) -> dict:
    if doc["state"] != "submitted":
        return doc
    st = backend.status(doc["job_arn"])
    if st["status"] != doc.get("aws_status"):
        doc["aws_status"] = st["status"]
        doc["aws_message"] = st.get("message")
        if st["status"] in TERMINAL:
            doc["terminal_utc"] = _utc()
        store.put_job(doc)
        store.log(event="status", job=doc["job_name"], status=st["status"])
    return doc


def collect_job(store: BatchStore, backend, ledger, doc: dict) -> dict:
    """Ingest a terminal job exactly once per record (re-running is a no-op)."""
    from hmb.providers import cache_usage, converse_result, store_batch_result
    if doc["state"] != "submitted" or doc.get("aws_status") not in TERMINAL:
        return doc
    lines, manifest = backend.outputs(doc["job_name"])
    by_key = {r["key"]: r for r in doc["records"]}
    seen: set[str] = set()
    utc = _utc()
    for ln in lines:
        try:
            row = json.loads(ln)
        except json.JSONDecodeError:
            store.log(event="unparseable_output_line", job=doc["job_name"])
            continue
        key = row.get("recordId")
        r = by_key.get(key)
        if r is None or key in seen:
            store.log(event="ignored_output_line", job=doc["job_name"], reason="unknown or duplicate recordId")
            continue
        seen.add(key)
        if r.get("outcome"):                                  # already ingested by an interrupted collect
            continue
        req, _ = store.load_request(key)
        if "modelOutput" in row and row["modelOutput"] is not None and not row.get("error"):
            out = converse_result(row["modelOutput"], req)
            in_tok, out_tok = int(out.get("input_tokens") or 0), int(out.get("output_tokens") or 0)
            if ledger.is_open(r["reservation_id"]):
                c_read, c_write = cache_usage(out)
                ledger.settle(r["reservation_id"], input_tokens=in_tok, output_tokens=out_tok,
                              image_count=len(req.image_paths), wall_ms=0, unit_price_id=doc["batch_unit_price_id"],
                              lane="batch", cache_read_tokens=c_read, cache_write_tokens=c_write)
            store_batch_result(req, out, store.layout, key=key, utc=utc,
                               transport={"lane": "batch", "job_name": doc["job_name"], "record_id": r["record_id"],
                                          "invocation_type": "Converse", "model_id": doc["model_id"]})
            r["outcome"] = {"status": "ok", "stop_reason": out.get("stop_reason"),
                            "input_tokens": in_tok, "output_tokens": out_tok}
        else:
            detail, retry = classify_record_error(row.get("error"))
            ledger.release(r["reservation_id"])
            r["outcome"] = {"status": "service_error", "detail": detail, "retryable": retry}
    status_missing = "job_failed" if doc["aws_status"] in ("Failed", "Stopped", "Expired") else "missing_output"
    for r in doc["records"]:
        if not r.get("outcome"):
            ledger.release(r["reservation_id"])
            r["outcome"] = {"status": status_missing, "detail": f"no output line (job {doc['aws_status']})",
                            "retryable": doc["aws_status"] != "Stopped"}
    doc.update(state="collected", collected_utc=utc, manifest_out=manifest)
    store.put_job(doc)
    counts: dict[str, int] = {}
    for r in doc["records"]:
        counts[r["outcome"]["status"]] = counts.get(r["outcome"]["status"], 0) + 1
    store.log(event="collected", job=doc["job_name"], outcomes=counts)
    return doc


def cancel_job(store: BatchStore, backend, job_name: str) -> dict:
    """Stop exactly the named job (nothing else). Its records are collected later as job_failed/Stopped."""
    doc = store.job(job_name)
    if doc["state"] != "submitted" or doc.get("aws_status") in TERMINAL:
        raise ValueError(f"{job_name} is {doc['state']}/{doc.get('aws_status')}: nothing to cancel")
    backend.stop(doc["job_arn"])
    store.log(event="stop_requested", job=job_name)
    return doc
