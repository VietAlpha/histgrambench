"""Cost ledger, dated rate card, marginal costs and the §16.2 projection (plan §16).

Contract (docs/CONTRACTS.md): `Ledger(layout)` with estimate_usd / reserve / settle / release / spent.
Spending is tracked, not capped (owner decision 2026-09-23, reports/decision_audit_20260923.md D2):
there is no aggregate ceiling, no phase cap, no 80 percent halt and no phase authorization. Every
call is still metered per phase, stage and model against the dated rate card.

ESTIMATES ARE NOT BILLING GUARANTEES. `estimate_usd` is a conservative worst case (output =
max_tokens; input tokens bounded above by the UTF-8 byte length of the prompt text, since a token
spans at least one byte; images priced per the dated rate card). Settled amounts apply the dated
rate card to provider-reported usage; the provider's invoice is the only billing truth.

Rate card `layout.pricing` (manifests/pricing.yaml; template: configs/pricing.template.yaml):
    entries:
      - unit_price_id: <id>            # referenced by ModelSpec.unit_price_id and by ledger rows
        model_id: <provider model id>  # (compute entries: "compute/<name>")
        currency: USD
        usd_per_mtok_input: <number>
        usd_per_mtok_output: <number>
        usd_per_mtok_cache_read: <number>   # optional; prompt-cache reads (billed apart from input tokens)
        usd_per_mtok_cache_write: <number>  # optional; prompt-cache writes
        usd_per_image: <number>        # or image_input_tokens: <int>; needed for image requests
        usd_per_cpu_second: <number>   # compute entries only
        lane: on_demand | batch        # pricing lane (default on_demand); a batch rate is its own entry
        pricing_url: https://...       # the provider's official pricing page
        retrieved_utc: <ISO date>      # retrieval date of that page
No price is invented anywhere (a batch discount is never derived from an on-demand rate): a model without an entry makes `estimate_usd` raise `PriceMissing`,
which blocks the live call. Cache tokens: Converse `inputTokens` excludes the cache-read and cache-write tokens,
which AWS bills on their own SKUs (GPT-6.1 Sol cache writes at 1.25x input were the September 2026 gap); they are
settled at the entry's cache rates, or at its input rate with a note when the entry has none. A reservation is logged before dispatch and settled or released after it;
reservations found unsettled in the log at start-up (a crashed run) stay counted as in flight until
released.
"""
from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import yaml

from .schemas import LedgerEvent, ModelRequest
from .store import Layout, append_jsonl, iter_jsonl

DISCLAIMER = "Estimates and projections are not billing guarantees; the provider invoice is authoritative."


class PriceMissing(RuntimeError):
    """No dated rate-card entry prices this call (blocks live calls)."""


class PricingError(ValueError):
    """Malformed rate card entry (missing URL or retrieval date, negative price...)."""


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Price:
    unit_price_id: str
    model_id: str
    in_per_mtok: float | None
    out_per_mtok: float | None
    per_image: float | None
    image_tokens: int | None
    per_cpu_second: float | None
    pricing_url: str
    retrieved_utc: str
    lane: str = "on_demand"
    cache_read_per_mtok: float | None = None
    cache_write_per_mtok: float | None = None


LANES = ("on_demand", "batch")


def load_rate_card(path: str | Path) -> dict[str, Price]:
    p = Path(path)
    if not p.exists():
        return {}
    d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, Price] = {}
    for e in d.get("entries") or []:
        uid = e.get("unit_price_id")
        if not uid:
            raise PricingError("rate card entry without unit_price_id")
        url, when = e.get("pricing_url"), e.get("retrieved_utc")
        if not (isinstance(url, str) and url.startswith("http")) or not when:
            raise PricingError(f"{uid}: entries need the provider pricing_url and retrieved_utc")
        if e.get("currency", "USD") != "USD":
            raise PricingError(f"{uid}: only USD entries are supported")
        nums = {k: e.get(k) for k in ("usd_per_mtok_input", "usd_per_mtok_output", "usd_per_image",
                                      "usd_per_cpu_second", "usd_per_mtok_cache_read", "usd_per_mtok_cache_write")}
        for k, v in nums.items():
            if v is not None and (not isinstance(v, (int, float)) or v < 0):
                raise PricingError(f"{uid}: {k} must be a non-negative number")
        if uid in out:
            raise PricingError(f"duplicate unit_price_id {uid}")
        lane = e.get("lane", "on_demand")
        if lane not in LANES:
            raise PricingError(f"{uid}: lane must be one of {LANES}")
        out[uid] = Price(uid, str(e.get("model_id", "")), nums["usd_per_mtok_input"],
                         nums["usd_per_mtok_output"], nums["usd_per_image"], e.get("image_input_tokens"),
                         nums["usd_per_cpu_second"], url, str(when), lane,
                         nums["usd_per_mtok_cache_read"], nums["usd_per_mtok_cache_write"])
    return out


class Ledger:
    def __init__(self, layout: Layout):
        self.layout = layout
        self.path = layout.ledger
        self._lock = threading.Lock()
        self.prices = load_rate_card(layout.pricing)
        self._inflight: dict[str, tuple[str, float]] = {}
        self._meta: dict[str, dict] = {}              # reservation id -> {stage, model_id}
        self._spent: dict[str, float] = {}
        for r in iter_jsonl(self.path):
            self._apply(r)

    # ---------------------------------------------------------------- state
    def _apply(self, r: dict) -> None:
        ev = r["event"]
        if ev == "reserve":
            self._inflight[r["reservation_id"]] = (r["phase"], float(r["usd"]))
            self._meta[r["reservation_id"]] = {"stage": r.get("stage", ""), "model_id": r.get("model_id")}
        elif ev == "settle":
            self._inflight.pop(r["reservation_id"], None)
            self._spent[r["phase"]] = self._spent.get(r["phase"], 0.0) + float(r["usd"])
        elif ev == "release":
            self._inflight.pop(r["reservation_id"], None)

    def _append(self, **kw) -> dict:
        rec = LedgerEvent.model_validate({"utc": _utc(), **kw}).model_dump(mode="json")
        append_jsonl(self.path, rec)
        self._apply(rec)
        return rec

    # ---------------------------------------------------------------- pricing
    def price_for(self, unit_price_id: str | None, model_id: str | None = None) -> Price:
        if unit_price_id and unit_price_id in self.prices:
            return self.prices[unit_price_id]
        if unit_price_id:
            raise PriceMissing(f"unit_price_id {unit_price_id!r} not in {self.layout.pricing}")
        cands = [p for p in self.prices.values() if model_id and p.model_id == model_id and p.lane == "on_demand"]
        if not cands:
            raise PriceMissing(f"no dated rate-card entry for model {model_id!r}")
        return max(cands, key=lambda p: p.retrieved_utc)

    def batch_price(self, model_id: str) -> Price:
        """The newest explicit batch-lane entry for `model_id` (PriceMissing when AWS publishes none)."""
        cands = [p for p in self.prices.values() if p.model_id == model_id and p.lane == "batch"]
        if not cands:
            raise PriceMissing(f"no dated batch-lane rate-card entry for model {model_id!r} (no discount is assumed)")
        return max(cands, key=lambda p: p.retrieved_utc)

    def estimate_usd(self, req: ModelRequest, unit_price_id: str | None = None) -> float:
        """Worst case: max_tokens output; input bounded by prompt bytes; images per the rate card."""
        p = self.price_for(unit_price_id or req.spec.unit_price_id, req.spec.model_id)
        if p.in_per_mtok is None or p.out_per_mtok is None:
            raise PriceMissing(f"{p.unit_price_id}: token prices missing")
        in_tok = len(((req.system or "") + req.text).encode("utf-8"))
        n_img = max(len(req.image_paths), len(req.image_sha256))
        img = 0.0
        if n_img:
            if p.per_image is not None:
                img = n_img * p.per_image
            elif p.image_tokens is not None:
                in_tok += n_img * int(p.image_tokens)
            else:
                raise PriceMissing(f"{p.unit_price_id}: request has images but the entry prices none")
        in_rate = max(p.in_per_mtok, p.cache_write_per_mtok or 0.0)   # the prompt may be billed as a cache write
        return (in_tok * in_rate + req.max_tokens * p.out_per_mtok) / 1e6 + img

    def cost_usd(self, unit_price_id: str, *, input_tokens: int, output_tokens: int,
                 image_count: int, cache_read_tokens: int = 0, cache_write_tokens: int = 0) -> float:
        return self._cost(unit_price_id, input_tokens=input_tokens, output_tokens=output_tokens,
                          image_count=image_count, cache_read_tokens=cache_read_tokens,
                          cache_write_tokens=cache_write_tokens)[0]

    def _cost(self, unit_price_id: str, *, input_tokens: int, output_tokens: int, image_count: int,
              cache_read_tokens: int, cache_write_tokens: int) -> tuple[float, list[str]]:
        """(usd, notes). Cache tokens without a cache rate on the entry are priced at its input rate."""
        p = self.price_for(unit_price_id)
        if p.in_per_mtok is None or p.out_per_mtok is None:
            raise PriceMissing(f"{unit_price_id}: token prices missing")
        img = 0.0
        if image_count and p.per_image is not None:
            img = image_count * p.per_image          # otherwise image tokens are inside input_tokens
        notes: list[str] = []
        cache = 0.0
        for kind, n, rate in (("cache_read", cache_read_tokens, p.cache_read_per_mtok),
                              ("cache_write", cache_write_tokens, p.cache_write_per_mtok)):
            if n and rate is None:
                notes.append(f"{n} {kind} tokens priced at the input rate (no {kind} rate on {unit_price_id})")
                rate = p.in_per_mtok
            cache += n * (rate or 0.0)
        return (input_tokens * p.in_per_mtok + output_tokens * p.out_per_mtok + cache) / 1e6 + img, notes

    # ---------------------------------------------------------------- metering
    def reserve(self, phase: str, stage: str, model_id: str, est_usd: float, *, lane: str | None = None) -> str:
        """Log the worst-case estimate before dispatch (tracked, never capped). `lane` is recorded for the
        batch lane only (None = on-demand, the rows written before 2026-09-26)."""
        if est_usd < 0:
            raise ValueError("negative estimate")
        with self._lock:
            rid = uuid.uuid4().hex
            self._append(event="reserve", reservation_id=rid, phase=phase, stage=stage,
                         model_id=model_id, usd=float(est_usd), **({"lane": lane} if lane else {}))
            return rid

    def is_open(self, reservation_id: str) -> bool:
        return reservation_id in self._inflight

    def settle(self, reservation_id: str, *, input_tokens: int, output_tokens: int, image_count: int,
               wall_ms: int, unit_price_id: str, lane: str | None = None, cache_read_tokens: int = 0,
               cache_write_tokens: int = 0) -> float:
        with self._lock:
            if reservation_id not in self._inflight:
                raise KeyError(f"unknown or closed reservation {reservation_id}")
            phase, est = self._inflight[reservation_id]
            usd, notes = self._cost(unit_price_id, input_tokens=input_tokens, output_tokens=output_tokens,
                                    image_count=image_count, cache_read_tokens=cache_read_tokens,
                                    cache_write_tokens=cache_write_tokens)
            if usd > est:
                notes.append(f"actual {usd:.6f} exceeded reservation {est:.6f}")
            res = self._reserve_row(reservation_id)
            cache = {k: v for k, v in (("cache_read_input_tokens", cache_read_tokens),
                                       ("cache_write_input_tokens", cache_write_tokens)) if v}
            self._append(event="settle", reservation_id=reservation_id, phase=phase,
                         stage=res.get("stage", ""), model_id=res.get("model_id"),
                         input_tokens=input_tokens, output_tokens=output_tokens, image_count=image_count,
                         wall_ms=wall_ms, unit_price_id=unit_price_id, usd=usd,
                         note="; ".join(notes) or None, **cache, **({"lane": lane} if lane else {}))
            return usd

    def _reserve_row(self, rid: str) -> dict:
        return self._meta.get(rid, {})

    def release(self, reservation_id: str) -> None:
        with self._lock:
            if reservation_id not in self._inflight:
                return
            phase, _ = self._inflight[reservation_id]
            res = self._reserve_row(reservation_id)
            self._append(event="release", reservation_id=reservation_id, phase=phase,
                         stage=res.get("stage", ""), model_id=res.get("model_id"), usd=0.0)

    def spent(self, phase: str | None = None) -> float:
        if phase is None:
            return float(sum(self._spent.values()))
        return float(self._spent.get(phase, 0.0))

    def in_flight(self, phase: str | None = None) -> float:
        return float(sum(u for p, u in self._inflight.values() if phase is None or p == phase))

    def events(self) -> list[dict]:
        return list(iter_jsonl(self.path))

    def report(self) -> dict:
        phases = sorted(set(self._spent) | {p for p, _ in self._inflight.values()})
        return {"disclaimer": DISCLAIMER, "spent_total": self.spent(), "in_flight_total": self.in_flight(),
                "phases": {p: {"spent": self.spent(p), "in_flight": self.in_flight(p)} for p in phases},
                "rate_card_entries": sorted(self.prices)}


# ------------------------------------------------------------------------ marginal costs (§16.4)
def window_cost(events: Iterable[dict], *, t0: str | None = None, t1: str | None = None,
                select: Callable[[dict], bool] | None = None) -> float:
    return float(sum(e["usd"] for e in events if e["event"] == "settle"
                     and (t0 is None or e["utc"] >= t0) and (t1 is None or e["utc"] < t1)
                     and (select is None or select(e))))


def marginal(delta_cost: float, delta_units: float) -> float | None:
    """c = Δ ledger cost / Δ units; None (not 0) when no units were added."""
    return None if not delta_units else delta_cost / delta_units


def c_item(events, *, t0: str, t1: str, delta_eligible_items: int) -> float | None:
    """Over the last complete extraction batch [t0, t1)."""
    return marginal(window_cost(events, t0=t0, t1=t1), delta_eligible_items)


def c_book(events, *, t0: str, t1: str, delta_books: int) -> float | None:
    return marginal(window_cost(events, t0=t0, t1=t1), delta_books)


def c_model(events, *, t0: str, t1: str, delta_models: int) -> float | None:
    return marginal(window_cost(events, t0=t0, t1=t1), delta_models)


def usd_per_scored_item_per_model(model_events: Iterable[dict], n_scored_items: int, *,
                                  cpu_seconds: float | None, cpu_price: Price | None) -> dict:
    """§11.10: (generation tokens + judge tokens + compile CPU seconds) at the dated rate card, per
    scored item. `model_events` are the settled rows attributable to this contestant model."""
    tok = window_cost(model_events)
    if cpu_seconds and (cpu_price is None or cpu_price.per_cpu_second is None):
        return {"value": None, "reason": "compile CPU seconds are not priced on the rate card",
                "disclaimer": DISCLAIMER}
    cpu = (cpu_seconds or 0.0) * (cpu_price.per_cpu_second if cpu_price else 0.0)
    if not n_scored_items:
        return {"value": None, "reason": "no scored items", "disclaimer": DISCLAIMER}
    return {"value": (tok + cpu) / n_scored_items, "token_usd": tok, "cpu_usd": cpu,
            "n_scored_items": n_scored_items, "disclaimer": DISCLAIMER}


# ------------------------------------------------------------------------ §16.2 projection

def project_cost(sym: dict) -> dict:
    """Symbolic cost equation §16.2 as a projection. Missing symbols leave their term None and the
    total None (never 0). A projection, not a billing guarantee."""
    g = sym.get

    def term(expr: Callable[[], float], needs: tuple[str, ...]):
        miss = [k for k in needs if g(k) is None]
        return (None, miss) if miss else (float(expr()), [])
    terms = {
        "onetime_engineering": term(lambda: g("Cost_onetime_engineering"), ("Cost_onetime_engineering",)),
        "per_book": term(lambda: g("G") * g("c_book"), ("G", "c_book")),
        "per_page": term(lambda: g("P") * (g("c_render") + g("c_ocr") + 3 * g("c_det")),
                         ("P", "c_render", "c_ocr", "c_det")),
        "per_crop_G8": term(lambda: g("K_f") * (g("c_cls") + g("c_ver") + g("c_meta")),
                            ("K_f", "c_cls", "c_ver", "c_meta")),
        "per_item_cards": term(lambda: g("N_cand") * 2 * g("B") * g("c_card"), ("N_cand", "B", "c_card")),
        "per_model_run": term(lambda: g("M") * g("T") * g("N") * (g("c_gen") + g("Valid1") * (
            g("c_cgx") + g("q") * g("EJ") * g("c_judge"))),
            ("M", "T", "N", "c_gen", "Valid1", "c_cgx", "q", "EJ", "c_judge")),
        "per_evaluator_version": term(lambda: g("V") * (g("D") * g("mu") * 2 * g("c_judge")
                                                        + g("D") * 4 * g("B") * g("c_card") + g("c_probe")),
                                      ("V", "D", "mu", "c_judge", "B", "c_card", "c_probe")),
        "text_probe": term(lambda: g("c_textprobe"), ("c_textprobe",)),
    }
    missing = sorted({m for _, ms in terms.values() for m in ms})
    total = None if missing else sum(v for v, _ in terms.values())
    return {"total": total, "terms": {k: v for k, (v, _) in terms.items()}, "missing": missing,
            "disclaimer": DISCLAIMER}


def register_cli(sub) -> None:
    import json
    import sys

    def _report(a) -> int:
        led = Ledger(Layout.from_env(a.root))
        sys.stdout.write(json.dumps(led.report(), indent=1, sort_keys=True, default=str) + "\n")
        return 0
    p = sub.add_parser("ledger-report", help="spend and in-flight reservations per phase (not billing guarantees)")
    p.set_defaults(func=_report)
