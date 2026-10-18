"""Hybrid batch/on-demand router and the dependency-aware scheduler (2026-09-26).

Router (`HybridRouter`, installed into `hmb.providers.call_model` for the duration of a run). On a
cache miss it:
  1. raises the recorded final failure of that key's batch record as ProviderError (non-retryable error,
     or retryable attempts exhausted: `transport_retry_max` is the same budget as a synchronous call);
     it never falls back to a synchronous call after a submitted batch record failed;
  2. defers (BatchDeferred) when the key is queued or in an active job, or another thread is calling it
     synchronously (single flight: no duplicate paid call);
  3. queues the request on its lane and defers, when the capability matrix routes it to batch and the
     policy is not `sync` and the key was not spilled;
  4. otherwise holds one of the model's synchronous slots (bounded concurrency = backpressure) and lets
     the synchronous call proceed.

Scheduler (`HybridScheduler`). Tasks are replayable callables (a cell, a card item, a gatekeeper crop):
a task runs until it returns or defers; a deferred task is parked on the keys it waits for and re-queued
the moment all of them are settled (answered, failed, or spilled to synchronous). Each task carries its
own dependency chain (contestant call -> extraction -> compile -> Q3 -> Q1 -> score), so a finished
contestant output advances to compilation and adjudication immediately; unrelated tasks never wait for it.
Cross-model reports (score-report, gates) run after the scheduler, over the full cell set.

Lane flush policies (records of one lane share model, Region, request family, modality, schema mode,
decoding, effort and role):
  sync    nothing is batched (immediate dispatch; the synchronous baseline)
  fixed   flush at `fill_target` records (never below the service minimum); when the run drains, flush if >= the
          minimum, else spill. `batch-run` sets fill_target to the run's lane size (items x tracks, capped by the
          config): measured best (reports/batch_inference_compatibility_20260926.md §7), since a lane's contestant
          records all arrive within seconds of the start and any further hold only delays their cells
  window  as fixed, and also flush when the oldest record waited `window_seconds` (>= minimum), or spill it
          to synchronous when the window expires below the minimum (bounded hold; never padded with
          scientifically meaningless calls)
State is resumable: queued requests and job manifests live under <root>/manifests/batch/, finished work
is content-addressed, so a restarted run re-derives every cell from the cache and resumes active jobs.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

from hmb.providers import BatchDeferred, ProviderError

from . import capability
from .jobs import BatchStore, TERMINAL, collect_job, lane_key, prepare_job, refresh_job, submit_job

POLICIES = ("sync", "fixed", "window")


@dataclass
class Policy:
    name: str = "window"
    fill_target: int = 1000
    window_seconds: float = 900.0
    max_records_per_job: int = 10000

    def __post_init__(self):
        if self.name not in POLICIES:
            raise ValueError(f"policy must be one of {POLICIES}")


@dataclass
class Metrics:
    t0: float = 0.0
    task_spans: dict = field(default_factory=dict)      # task id -> [start, end]
    busy: list = field(default_factory=list)            # (model_id, start, end, lane) of every paid call/job
    deferrals: int = 0
    spills: int = 0
    jobs: list = field(default_factory=list)            # (lane, records, submit_t, done_t)
    sync_calls: int = 0
    batch_records: int = 0
    task_errors: dict = field(default_factory=dict)


class HybridRouter:
    def __init__(self, store: BatchStore, ledger, policy: Policy, *, matrix: dict | None = None,
                 sync_concurrency: dict | None = None, default_sync_concurrency: int = 4,
                 retry_max: int | None = None, clock: Callable[[], float] = time.monotonic,
                 plan_only: bool = False):
        self.store, self.ledger, self.policy, self.matrix = store, ledger, policy, matrix
        self.clock = clock
        self.plan_only = plan_only
        self.lock = threading.RLock()
        self.lanes: dict[str, list[tuple[str, float]]] = {}
        self.lane_min: dict[str, int] = {}
        self.queued: set[str] = set()
        self.spilled: set[str] = set()
        self.claims: set[str] = set()
        self.sems: dict[str, threading.BoundedSemaphore] = {}
        self.sync_concurrency = dict(sync_concurrency or {})
        self.default_sync = int(default_sync_concurrency)
        if retry_max is None:
            from hmb.config import production_template
            retry_max = int(production_template().value("transport_retry_max"))
        self.max_attempts = 1 + int(retry_max)
        self.plan: list[dict] = []
        self.metrics = Metrics(t0=clock())
        self._active = store.active_keys()
        self._outcomes = store.record_outcomes()
        for key in store.queued_keys():                   # resume: restore the ready queues
            req, lane = store.load_request(key)
            dec = capability.decide(req, matrix)
            self._enqueue(key, lane, dec.min_records or 1)

    # ------------------------------------------------------------------ state
    def _has_batch_price(self, model_id: str) -> bool:
        from hmb.ledger import PriceMissing
        try:
            self.ledger.batch_price(model_id)
            return True
        except PriceMissing:
            return False

    def refresh_indexes(self) -> None:
        with self.lock:
            self._active = self.store.active_keys()
            self._outcomes = self.store.record_outcomes()

    def _sem(self, model_id: str) -> threading.BoundedSemaphore:
        with self.lock:
            if model_id not in self.sems:
                self.sems[model_id] = threading.BoundedSemaphore(self.sync_concurrency.get(model_id, self.default_sync))
            return self.sems[model_id]

    def _enqueue(self, key: str, lane: str, min_records: int) -> None:
        with self.lock:
            if key in self.queued:
                return
            self.lanes.setdefault(lane, []).append((key, self.clock()))
            self.lane_min[lane] = min_records
            self.queued.add(key)

    def settled(self, key: str) -> bool:
        """True once a parked task waiting on `key` can make progress."""
        with self.lock:
            if key in self.spilled or key in self._outcomes:
                return key not in self.queued and key not in self._active
            return (self.store.layout.model_call_cache / key[:2] / f"{key}.json").exists() or \
                (key not in self.queued and key not in self._active and key not in self.claims)

    # ------------------------------------------------------------------ call_model hook
    @contextmanager
    def live_slot(self, req, key: str, *, layout, allow_live: bool, fake: bool):
        with self.lock:
            hist = [o for o in self._outcomes.get(key, []) if o.get("status") != "ok"]
            active = key in self._active or key in self.queued
            claimed = key in self.claims
        if active or claimed:
            self.metrics.deferrals += 1
            raise BatchDeferred({key}, f"{req.spec.model_id}: waiting for an in-flight identical call")
        if hist:
            last = hist[-1]
            if not last.get("retryable") or len(hist) >= self.max_attempts:
                raise ProviderError(f"{req.spec.model_id}: batch record failed ({last.get('status')}: "
                                    f"{last.get('detail')}); no synchronous fallback after a submitted batch record")
        dec = capability.decide(req, self.matrix) if not fake else capability.RouteDecision(
            "synchronous", "fake provider (tests)", None)
        to_batch = dec.route == "batch" and self.policy.name != "sync" and key not in self.spilled
        if to_batch and not self._has_batch_price(req.spec.model_id):
            to_batch = False
            dec = capability.RouteDecision("synchronous", f"{dec.route_id}: no dated batch-lane rate-card entry "
                                                          "(no discount is assumed)", dec.route_id)
        if self.plan_only:
            self.plan.append({"cache_key": key, "model_id": req.spec.model_id, "purpose": req.purpose,
                              "route": "batch" if to_batch else "synchronous",
                              "reason": dec.reason if dec.route == "synchronous" or to_batch else
                              f"policy {self.policy.name}" if self.policy.name == "sync" else "spilled below minimum"})
            raise BatchDeferred({key}, "plan only: no call is made")
        if to_batch:
            if not allow_live:
                from hmb.providers import LiveCallsDisabled
                raise LiveCallsDisabled(f"{req.spec.model_id}: batch submission needs --allow-live")
            lane = lane_key(req)
            self.store.save_request(req, key, lane)
            self._enqueue(key, lane, dec.min_records or 1)
            self.metrics.deferrals += 1
            raise BatchDeferred({key}, f"{req.spec.model_id}: queued on batch lane {lane}")
        with self.lock:
            if key in self.claims:
                raise BatchDeferred({key}, "single flight")
            self.claims.add(key)
        sem = self._sem(req.spec.model_id)
        sem.acquire()
        t0 = self.clock()
        try:
            yield
        finally:
            sem.release()
            with self.lock:
                self.claims.discard(key)
                self.metrics.busy.append((req.spec.model_id, t0, self.clock(), "sync"))
                self.metrics.sync_calls += 1

    # ------------------------------------------------------------------ lanes
    def due_lanes(self, draining: bool) -> tuple[list[tuple[str, list[str]]], list[str]]:
        """(lanes to flush with their keys, keys to spill to synchronous) under the policy."""
        now = self.clock()
        flush, spill = [], []
        p = self.policy
        with self.lock:
            for lane, items in list(self.lanes.items()):
                if not items:
                    continue
                n, age, mn = len(items), now - items[0][1], self.lane_min.get(lane, 1)
                take = None
                if n >= p.fill_target and n >= mn:          # never a job below the service minimum
                    take = min(n, p.max_records_per_job)
                elif draining or (p.name == "window" and age >= p.window_seconds):
                    if n >= mn:
                        take = min(n, p.max_records_per_job)
                    else:
                        spill += [k for k, _ in items]
                        self.lanes[lane] = []
                        continue
                if take:
                    flush.append((lane, [k for k, _ in items[:take]]))
                    self.lanes[lane] = items[take:]
            for k in spill:
                self.queued.discard(k)
                self.spilled.add(k)
            self.metrics.spills += len(spill)
        return flush, spill

    def mark_submitted(self, keys: list[str], job_name: str) -> None:
        with self.lock:
            for k in keys:
                self.queued.discard(k)
                self._active[k] = job_name

    def has_queued(self) -> bool:
        with self.lock:
            return any(self.lanes.values())


class HybridScheduler:
    """Bounded worker pool over replayable tasks, with batch lanes flushed by policy and jobs polled."""

    def __init__(self, router: HybridRouter, backend, *, workers: int = 8, poll_seconds: float = 60.0,
                 tick_seconds: float = 0.5, max_wall_seconds: float | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.router, self.backend = router, backend
        self.workers, self.poll_seconds, self.tick = workers, poll_seconds, tick_seconds
        self.max_wall = max_wall_seconds
        self.sleep = sleep
        self._last_poll = -1e18

    def _submit_lane(self, lane: str, keys: list[str]) -> None:
        r = self.router
        reqs = [(k, r.store.load_request(k)[0]) for k in keys]
        doc = prepare_job(r.store, r.ledger, reqs, lane=lane)
        doc = submit_job(r.store, self.backend, r.ledger, doc)
        r.mark_submitted(keys, doc["job_name"])
        r.metrics.jobs.append({"job": doc["job_name"], "lane": lane, "records": len(keys),
                               "submit_t": r.clock(), "done_t": None, "state": doc["state"]})
        r.metrics.batch_records += len(keys)
        if doc["state"] == "submit_failed":
            r.refresh_indexes()

    def _poll(self, force: bool = False) -> None:
        r = self.router
        now = r.clock()
        if not force and now - self._last_poll < self.poll_seconds:
            return
        self._last_poll = now
        changed = False
        for doc in r.store.jobs():
            if doc["state"] == "prepared":                    # interrupted before acceptance: resubmit (idempotent)
                submit_job(r.store, self.backend, r.ledger, doc)
                changed = True
            elif doc["state"] == "submitted":
                doc = refresh_job(r.store, self.backend, doc)
                if doc.get("aws_status") in TERMINAL:
                    collect_job(r.store, self.backend, r.ledger, doc)
                    changed = True
                    for j in r.metrics.jobs:
                        if j["job"] == doc["job_name"]:
                            j["done_t"], j["state"] = r.clock(), doc["aws_status"]
                            r.metrics.busy.append((doc["model_id"], j["submit_t"], j["done_t"], "batch"))
        if changed:
            r.refresh_indexes()

    def run(self, tasks: dict[str, Callable[[], Any]]) -> dict:
        r = self.router
        ready: deque[str] = deque(tasks)
        running: dict[Future, str] = {}
        parked: dict[str, frozenset] = {}
        results: dict[str, Any] = {}
        t_start = r.clock()
        m = r.metrics
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            while ready or running or parked:
                while ready and len(running) < self.workers:          # backpressure: never more than `workers`
                    tid = ready.popleft()
                    m.task_spans.setdefault(tid, [r.clock(), None])
                    running[pool.submit(tasks[tid])] = tid
                if running:
                    done, _ = wait(list(running), timeout=self.tick, return_when=FIRST_COMPLETED)
                else:
                    done = set()
                    self.sleep(self.tick)
                for f in done:
                    tid = running.pop(f)
                    try:
                        results[tid] = f.result()
                        m.task_spans[tid][1] = r.clock()
                    except BatchDeferred as d:
                        parked[tid] = d.keys
                    except Exception as e:                            # isolated: other tasks continue
                        results[tid] = e
                        m.task_errors[tid] = f"{type(e).__name__}: {e}"[:300]
                        m.task_spans[tid][1] = r.clock()
                draining = not ready and not running
                flush, _spill = r.due_lanes(draining)
                for lane, keys in flush:
                    self._submit_lane(lane, keys)
                self._poll(force=bool(flush))
                for tid, keys in list(parked.items()):
                    if all(r.settled(k) for k in keys):
                        del parked[tid]
                        ready.append(tid)
                if parked and not ready and not running and not r.has_queued() and not r.store.active_keys():
                    for tid in list(parked):                          # nothing can settle them otherwise: replay
                        del parked[tid]
                        ready.append(tid)
                if self.max_wall is not None and r.clock() - t_start > self.max_wall and not running:
                    break                                             # resumable stop: state is on disk
        return {"results": results, "parked": {t: sorted(k) for t, k in parked.items()},
                "stopped_early": bool(parked or ready)}


def summarize(metrics: Metrics, *, t_end: float, scale: float = 1.0) -> dict:
    """Run-efficiency figures: critical path, stage idle, utilization, batch fill, jobs, calls/min."""
    spans = [s for s in metrics.task_spans.values() if s[1] is not None]
    wall = (t_end - metrics.t0) * scale
    by_model: dict[str, float] = {}
    for mid, a, b, lane in metrics.busy:
        if lane == "sync":
            by_model[mid] = by_model.get(mid, 0.0) + (b - a) * scale
    jobs = metrics.jobs
    fill = [j["records"] for j in jobs]
    calls = metrics.sync_calls + metrics.batch_records
    return {"wall_seconds": wall,
            "critical_path_seconds": max((s[1] - metrics.t0) * scale for s in spans) if spans else 0.0,
            "tasks": len(spans), "sync_calls": metrics.sync_calls, "batch_records": metrics.batch_records,
            "remote_jobs": len(jobs), "mean_job_records": (sum(fill) / len(fill)) if fill else 0.0,
            "spilled_to_sync": metrics.spills, "deferrals": metrics.deferrals,
            "calls_per_minute": (calls / (wall / 60.0)) if wall else 0.0,
            "sync_busy_seconds_by_model": by_model, "task_errors": len(metrics.task_errors)}
