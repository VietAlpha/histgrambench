"""`cards-build`: build cards and card gates for every `data/candidates/<item_id>.json` (Scaffold 4).

Per candidate record (§5.2, written by acquisition G8):
  * skip when both `semantic_cards/<item_id>.json` and `data/card_gates/<item_id>.json` exist;
    when only the card exists (interrupted run), the gates are recomputed from the stored card;
  * otherwise build the card through `hmb.providers.call_model` (content-addressed cache: a rebuilt
    item whose calls are cached spends nothing);
  * the §8.7 item gates;
  * write `data/card_gates/<item_id>.json` immutably.

Without `allow_live`, a cache miss stops that item before any artifact is written; the run continues
over the remaining items (still without spend) and then raises `LiveCallsDisabled` naming every
missing cache entry (item, model, purpose, cache key). Builder/judge family disjointness (§10.9) is
checked against the roster before any call.
"""
from __future__ import annotations

import threading
from pathlib import Path

from hmb.schemas import Card
from hmb.store import Layout, read_json, write_json_immutable

from .build import build_card, check_builder_roster, check_primary_disjoint, generation_requests
from .consensus import contradiction_rate, eligible_counts
from .gates import item_gates


def candidates_dir(layout: Layout) -> Path:
    return layout.p("data", "candidates")


def card_selection_path(layout: Layout) -> Path:
    """Current pre-card selection written by `card-select` (hmb.sampling.select_for_cards); absent = every candidate."""
    return layout.p("manifests", "card_selection.json")


def card_selection_excluded(layout: Layout) -> dict[str, str]:
    """{item_id: exclusion reason} of the current selection ({} when no selection was made)."""
    p = card_selection_path(layout)
    return dict(read_json(p)["excluded"]) if p.exists() else {}


def card_selection_lock_path(layout: Layout) -> Path:
    """Owner lock of the current selection (`card-lock`): the exact test candidates production cards-build cards."""
    return layout.p("manifests", "card_selection_lock.json")


def card_selection_locked(layout: Layout) -> set[str] | None:
    """Locked test candidate ids, or None when no lock exists. Raises ValueError when the current
    card_selection.json is not the selection the lock names (a later `card-select` would change the set)."""
    p = card_selection_lock_path(layout)
    if not p.exists():
        return None
    lock = read_json(p)
    sp = card_selection_path(layout)
    cur = read_json(sp)["sha256"] if sp.exists() else None
    if cur != lock["selection_sha256"]:
        raise ValueError(f"card_selection.json ({(cur or 'absent')[:12]}) is not the locked selection "
                         f"{lock['selection_sha256'][:12]} ({p})")
    return set(lock["test_item_ids"])


def card_gates_path(layout: Layout, item_id: str) -> Path:
    return layout.p("data", "card_gates", f"{item_id}.json")


def _asset(layout: Layout, rec: dict, key: str) -> Path | None:
    rel = (rec.get("assets") or {}).get(key)
    return None if not rel else layout.p("images", *Path(rel).parts)


def ocr_words_for(rec: dict, page_rows: list[dict]) -> list[dict]:
    """OCR words for the crop: from the record (`ocr_words` or `automatic_metadata.ocr_words`), else
    from a page funnel row of the same (book_id, page_index) carrying `ocr_words` with page-pixel
    `bbox` [x0, y0, x1, y1] (kept when the box centre lies in the crop; converted to crop-relative
    x, y, w, h), else []."""
    for src in (rec.get("ocr_words"), (rec.get("automatic_metadata") or {}).get("ocr_words")):
        if isinstance(src, list):
            return src
    src_ = rec.get("source") or {}
    box = (rec.get("extraction") or {}).get("crop_bbox")
    if not box:
        return []
    x0, y0, x1, y1 = box
    W, Hh = max(1, x1 - x0), max(1, y1 - y0)
    for row in page_rows:
        if row.get("book_id") == src_.get("book_id") and row.get("page_index") == src_.get("page_index") \
                and isinstance(row.get("ocr_words"), list):
            out = []
            for w in row["ocr_words"]:
                b = w.get("bbox")
                if not b or len(b) != 4:
                    continue
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    out.append({"text": w.get("text", ""),
                                "box": [(b[0] - x0) / W, (b[1] - y0) / Hh, (b[2] - b[0]) / W, (b[3] - b[1]) / Hh]})
            return out
    return []


def card_gate_record(card: Card, th) -> dict:
    return {"item_id": card.item_id, "card_uid": card.card_uid,
            "exclusion_reasons": [e.value for e in item_gates(card, th)],
            "eligible_counts": eligible_counts(card),
            "contradiction_rate": contradiction_rate(card), "profile": card.profile}


def cards_build(layout: Layout, th, *, builders: list, judges: list, allow_live: bool, phase: str, ledger,
                provider_override=None, max_tokens: int | None = None, only_books: set[str] | None = None,
                skip_items: set[str] | None = None, workers: int = 1) -> dict:
    """`only_books`: restrict to candidates of these books (the pre-freeze development books).
    `skip_items`: candidates excluded by the current `card-select` selection (never carded).
    `workers`: items carded concurrently (owner 2026-09-30); each item is independent and its files are written
    atomically, so the cards equal a sequential run's. A call that fails after its retries is recorded as that
    item's error and the pass continues (rerun to retry it)."""
    from concurrent.futures import ThreadPoolExecutor

    from hmb.providers import BatchDeferred, LiveCallsDisabled, cache_key, call_model

    check_builder_roster(builders)
    disjoint_checked = bool(judges)
    if judges:
        check_primary_disjoint(builders, judges)
    try:
        from hmb.acquisition.funnel import page_rows as _page_rows
        rows = _page_rows(layout)
    except (ImportError, FileNotFoundError, ValueError):
        rows = []
    summary: dict = {"built": [], "gates_only": [], "skipped": [], "missing": [], "errors": [],
                     "disjointness_checked": disjoint_checked}
    missing: list[dict] = []
    deferred: set[str] = set()          # keys answered by batch jobs (hybrid runs, hmb.batch): item replayed later
    lock = threading.Lock()

    def one(path: Path) -> None:
        rec = read_json(path)
        iid = rec.get("id") or path.stem
        if only_books is not None and (rec.get("source") or {}).get("book_id") not in only_books:
            return
        if skip_items and iid in skip_items:
            return
        gpath, cpath = card_gates_path(layout, iid), layout.card(iid)
        if gpath.exists() and cpath.exists():
            with lock:
                summary["skipped"].append(iid)
            return
        crop = _asset(layout, rec, "crop_canonical_image")
        if crop is None or not crop.exists():
            with lock:
                summary["errors"].append({"item_id": iid, "error": "crop_canonical_image missing"})
            return
        words = ocr_words_for(rec, rows)
        if cpath.exists():
            card = Card.model_validate(read_json(cpath))
            with lock:
                summary["gates_only"].append(iid)
        else:
            ctx_p = _asset(layout, rec, "context_text")
            ctx = ctx_p.read_text(encoding="utf-8") if ctx_p and ctx_p.exists() else ""

            def call(req, lay, led, *, allow_live, provider_override=None, _iid=iid):
                try:
                    return call_model(req, lay, led, allow_live=allow_live, provider_override=provider_override)
                except BatchDeferred:
                    raise
                except LiveCallsDisabled:
                    with lock:
                        missing.append({"item_id": _iid, "model_id": req.spec.model_id, "purpose": req.purpose,
                                        "cache_key": cache_key(req, fake=provider_override is not None)})
                    raise
            try:
                card = build_card(iid, crop, words, ctx or None, builders=builders, layout=layout, th=th,
                                  ledger=ledger, allow_live=allow_live, provider_override=provider_override,
                                  max_tokens=max_tokens, phase=phase, call=call)
            except BatchDeferred as d:
                with lock:
                    deferred.update(d.keys)
                return                        # nothing written for this item; replayed once the keys settle
            except LiveCallsDisabled:
                # name every missing generation entry of this item, not only the first one hit
                with lock:
                    seen = {m["cache_key"] for m in missing}
                    for greq in generation_requests(crop, words, ctx or None, builders=builders, th=th,
                                                    max_tokens=max_tokens, phase=phase):
                        key = cache_key(greq, fake=provider_override is not None)
                        if key not in seen and not (layout.model_call_cache / key[:2] / f"{key}.json").exists():
                            missing.append({"item_id": iid, "model_id": greq.spec.model_id,
                                            "purpose": greq.purpose, "cache_key": key})
                return                        # nothing written for this item
            except Exception as e:            # isolated (owner 2026-09-30): one failed call never stops the pass
                with lock:
                    summary["errors"].append({"item_id": iid, "error": f"{type(e).__name__}: {e}"[:500]})
                return
            with lock:
                summary["built"].append(iid)
        gate_rec = card_gate_record(card, th)
        if th.levels_block():                     # built before the level freeze (pre-freeze context)
            gate_rec["levels"] = th.levels_block()
        write_json_immutable(gpath, gate_rec)

    paths = sorted(candidates_dir(layout).glob("*.json"))
    if workers <= 1:
        for path in paths:
            one(path)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for f in [pool.submit(one, p) for p in paths]:
                f.result()
    summary["missing"] = missing
    if deferred and not missing:
        raise BatchDeferred(deferred, f"cards-build: {len(deferred)} call(s) pending in batch jobs")
    if missing:
        lines = "\n".join(f"  {m['item_id']}  {m['model_id']}  {m['purpose']}  cache_key={m['cache_key']}"
                          for m in missing)
        raise LiveCallsDisabled(
            f"cards-build: {len(missing)} model-call cache entr{'y' if len(missing) == 1 else 'ies'} missing and "
            f"live calls are disabled (items stopped before writing anything; rerun with --allow-live under an "
            f"authorized budget; verification entries become enumerable once generation is cached):\n{lines}")
    return summary


def dev_measurements(layout: Layout, th) -> dict:
    """MS3 development-set measurements (component 10, MS3 row) from the stored candidates, cards and
    card gates, no model calls: retention r̂ (eligible / carded), mean eligible facts F̄ over retained
    items, mean contradiction rate and per-fact family agreement (supporting families / builders over the
    union). In a pre-freeze context only the declared development books are read."""
    import numpy as np

    from hmb.acquisition.drivers import development_books
    from hmb.hashing import sha256_file
    from hmb.roster import N_BUILDERS
    dev = development_books(layout) if th.is_prefreeze else None
    cands = {}
    for path in sorted(candidates_dir(layout).glob("*.json")):
        rec = read_json(path)
        if dev is None or (rec.get("source") or {}).get("book_id") in dev:
            cands[rec.get("id") or path.stem] = (path, rec)
    carded = [i for i in cands if layout.card(i).exists() and card_gates_path(layout, i).exists()]
    cards = {i: Card.model_validate(read_json(layout.card(i))) for i in carded}
    gates = {i: read_json(card_gates_path(layout, i)) for i in carded}
    retained = [i for i in carded if not gates[i].get("exclusion_reasons") and any(f.eligible for f in cards[i].facts)]
    union = [f for i in carded for f in cards[i].facts]
    contra = [g["contradiction_rate"] for g in gates.values() if g.get("contradiction_rate") is not None]
    return {
        "n_candidates": len(cands), "n_carded": len(carded), "n_retained": len(retained),
        "retention_r_hat": len(retained) / len(carded) if carded else None,
        "F_bar": (sum(sum(f.eligible for f in cards[i].facts) for i in retained) / len(retained)) if retained else None,
        "contradiction_rate_mean": float(np.mean(contra)) if contra else None,
        "family_agreement_mean": (float(np.mean([len(set(f.supporting_families)) / N_BUILDERS for f in union]))
                                  if union else None),
        "inputs": {"candidates": {i: sha256_file(cands[i][0]) for i in sorted(cands)},
                   "cards": {i: sha256_file(layout.card(i)) for i in sorted(carded)},
                   "card_gates": {i: sha256_file(card_gates_path(layout, i)) for i in sorted(carded)}},
    }
