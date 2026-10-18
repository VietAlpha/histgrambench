"""Resumable drivers past the CPU ladder: paid gates G8 -> candidate records (`acq-g8`), and
item finalization after cards (`items-finalize`).

Both drivers read the append-only funnel (latest row per proposal), never rewrite anything, and
never re-call a cached model (hmb.providers.call_model caches by content). Re-running after an
interruption continues where the previous run stopped: rows that already carry a G8b record are
skipped; immutable records are re-written only with identical bytes.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..hashing import item_id as make_item_id, sha256_bytes, sha256_file
from ..providers import LiveCallsDisabled
from ..rights import load_manifests, r3_gate
from ..schemas import Card, ItemRecord
from ..store import Layout, iter_jsonl, read_json, write_immutable, write_json_immutable
from ..vocab import Exclusion, FunnelOutcome
from . import collection as col
from . import gates as G
from .context import context_window, inside_share
from .funnel import (append_row, assemble_item, canonical_crop, default_automatic_metadata, finalize,
                     latest_rows, page_funnel_path)
from .render import encode_png

G8_PHASE = "TEST"            # ledger phase outside a pre-freeze context; inside one, the context name
OCR_WORD_LEVEL = "block"      # PaddleOCR-VL gives layout blocks, not word boxes


def candidates_dir(layout: Layout) -> Path:
    return layout.p("data", "candidates")


# --------------------------------------------------------------------------- pre-freeze development books
def development_books_path(layout: Layout) -> Path:
    return layout.p("sources", "development_books.jsonl")


def development_books(layout: Layout) -> set[str]:
    """Books used in a pre-freeze PILOT/DEV context: development books for good (append-only log)."""
    return {r["book_id"] for r in iter_jsonl(development_books_path(layout))}


def declare_development_book(layout: Layout, book_id: str, source_name: str, context: str) -> None:
    if book_id not in development_books(layout):
        from datetime import datetime, timezone

        from ..store import append_jsonl
        append_jsonl(development_books_path(layout), {
            "book_id": book_id, "source_name": source_name, "context": context,
            "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "rule": "a book processed before the level freeze is a development book; items-finalize refuses "
                    "to place it in the test split"})


def require_development_context(th, what: str) -> None:
    """In a pre-freeze context only the development stages acquire (never test books)."""
    from ..config import PREFREEZE_CONTEXTS
    if th.is_prefreeze and PREFREEZE_CONTEXTS[th.context]["partition"] != "development":
        raise G.GateBlocked(f"{what} runs only in a development pre-freeze context (PILOT/DEV), not {th.context}")


def _page_rows(layout: Layout) -> dict[tuple[str, int], dict]:
    out = {}
    for r in iter_jsonl(page_funnel_path(layout)):
        if r.get("page_index") is not None:
            out[(r["book_id"], r["page_index"])] = r
    return out


def _cpu_passed(row: dict) -> bool:
    try:
        G.require_cpu_ladder_passed(row)
        return True
    except G.LadderIncomplete:
        return False


def _source_crop_and_text(layout: Layout, row: dict) -> tuple[np.ndarray, list[dict]]:
    """The item's Paddle crop (re-verified against the snapshot the ladder used) and the page's Paddle
    text blocks, both in Paddle's frame. Raises GateBlocked when the snapshot or the file is gone."""
    sc = row["source_crop"]
    sp = layout.p("sources", "collection_snapshots", f"{row['book_id']}_{sc['snapshot_id']}.json")
    if not sp.exists():
        raise G.GateBlocked(f"{row['proposal_id']}: collection snapshot {sc['snapshot_id']} not stored")
    snap = read_json(sp)
    try:
        crop = col.load_crop_gray(snap, sc["file"])
        pages = col.parse_ocr(snap)["pages"]
    except (col.SnapshotChanged, col.OcrMalformed) as e:
        raise G.GateBlocked(f"{row['proposal_id']}: {e}") from e
    return crop, col.frame_text_blocks(pages[row["page_index"]])


def ocr_words_in_crop(text_blocks: list[dict], box) -> list[dict]:
    """[{text, bbox, conf, level}] for OCR units >= 50% inside the crop (block level; conf unknown)."""
    return [{"text": u["text"], "bbox": list(u["bbox"]), "conf": None, "level": OCR_WORD_LEVEL}
            for u in text_blocks if inside_share(u["bbox"], box) >= 0.5]


# --------------------------------------------------------------------------- acq-g8
def run_g8(layout: Layout, th, roster: dict, *, ledger, allow_live: bool, book_id: str | None = None,
           max_tokens: int | None = None, provider_override=None, dataset_version: str = "v1.0") -> dict:
    """G8b, the strict gatekeeper alone (role audit 2026-09-23), over rows whose CPU ladder passed;
    survivors become candidate records."""
    if th.is_production and provider_override is not None:
        raise G.GateBlocked("fake providers are refused under the production profile")
    require_development_context(th, "acq-g8")
    dev_only = development_books(layout) if th.is_prefreeze else None
    mt = int(max_tokens if max_tokens is not None else th.value("g8_max_tokens"))
    verifiers = roster.get("verifier", [])
    if len(verifiers) != 1:
        raise G.GateBlocked(f"roster needs exactly one gatekeeper (verifier) for G8b, got {len(verifiers)}")
    gatekeeper = verifiers[0]
    pages = _page_rows(layout)
    mans = load_manifests(layout.rights_manifest)
    summary = {"processed": 0, "skipped_done": 0, "survivors": 0, "rejected": 0, "pending": 0,
               "stopped": None, "candidates": []}
    todo = []
    for pid, r in sorted(latest_rows(layout).items()):
        if book_id and r["book_id"] != book_id:
            continue
        if dev_only is not None and r["book_id"] not in dev_only:
            continue                              # pre-freeze: development books only
        if any(g["gate_id"] == "G8b" for g in r.get("gates", [])):
            summary["skipped_done"] += 1
        elif _cpu_passed(r):
            todo.append(r)
    for i, r in enumerate(todo):
        try:
            out = _g8_one(layout, th, r, pages, mans, gatekeeper, ledger=ledger,
                          allow_live=allow_live, max_tokens=mt, provider_override=provider_override,
                          dataset_version=dataset_version)
        except LiveCallsDisabled as e:
            summary["stopped"] = f"live calls disabled: {e}"
            summary["pending"] = len(todo) - i
            break
        summary["processed"] += 1
        if out:
            summary["survivors"] += 1
            summary["candidates"].append(out)
        else:
            summary["rejected"] += 1
    return summary


def _g8_one(layout, th, row, pages, mans, gatekeeper, *, ledger, allow_live, max_tokens,
            provider_override, dataset_version) -> str | None:
    book, pi = row["book_id"], row["page_index"]
    prow = pages.get((book, pi))
    if prow is None:
        raise G.GateBlocked(f"{row['proposal_id']}: no page row for {book}:{pi}")
    page_png = layout.page_image(book, pi)
    data = page_png.read_bytes()
    if prow.get("page_image_sha256") and sha256_bytes(data) != prow["page_image_sha256"]:
        raise G.GateBlocked(f"{page_png}: page image differs from its funnel record")
    src, text_blocks = _source_crop_and_text(layout, row)
    box = [int(v) for v in row["box_xyxy_page_px"]]                 # Paddle frame
    ox, oy = row["source_crop"]["block_bbox"][:2]
    iid = make_item_id(book, pi, box)
    from ..manifests import ManifestError, assert_id_usable
    try:
        assert_id_usable(layout, iid)             # §15.4: a tombstoned id is never issued again
    except ManifestError as e:
        raise G.GateBlocked(str(e)) from e
    crop_png = encode_png(np.ascontiguousarray(src[box[1] - oy:box[3] - oy, box[0] - ox:box[2] - ox]))
    write_immutable(layout.crop(iid), crop_png)
    canon_png, _ = canonical_crop(crop_png, th)
    canon_path = layout.crop_canonical(iid)
    write_immutable(canon_path, canon_png)
    new: dict[str, Any] = dict(row, gates=list(row["gates"]), model_calls=list(row.get("model_calls", [])),
                          exclusion_reasons=list(row.get("exclusion_reasons", [])))
    rec_b, rs_b, out_b, calls_b = G.g8b(row, canon_path, gatekeeper, layout=layout, ledger=ledger, phase=th.context or G8_PHASE,
                                        max_tokens=max_tokens, allow_live=allow_live,
                                        provider_override=provider_override)
    new["gates"].append(rec_b)
    new["model_calls"] += calls_b
    new["verifier_pred"], new["item_id"] = out_b["verifier_pred"], iid
    new["diagram_type"] = out_b.get("diagram_type")
    new["exclusion_reasons"] += rs_b
    append_row(layout, finalize(new))
    if rs_b:
        return None
    ctx = context_window(box, pi, {pi: text_blocks}, row.get("caption_text"), None, th)
    am = default_automatic_metadata(row)
    _, psha = G.load_prompt(G.GATEKEEPER_PROMPT)
    am.update(ocr_words=ocr_words_in_crop(text_blocks, box),
              ocr_words_level=OCR_WORD_LEVEL,
              # descriptive only (owner decision 2026-09-25): predicted by the gatekeeper, never checked by a
              # human, never a gate, score input or quota
              diagram_type={"label": out_b.get("diagram_type"), "status": out_b.get("diagram_type_status"),
                            "labeler_model_id": gatekeeper.model_id, "labeler_effort": gatekeeper.effort,
                            "gate_version": G.GATE_VERSION["G8b"], "prompt": G.GATEKEEPER_PROMPT,
                            "prompt_sha256": psha},
              # provenance only: the page had no prose, caption or body text (never a pass, penalty or score term)
              figure_only_page=bool(prow.get("figure_only")))
    if th.levels_block():
        am["levels"] = th.levels_block()          # written before the level freeze (pre-freeze context)
    man = mans.get(book) or {}
    rec = assemble_item(layout, dict(new, box_xyxy_page_px=box), page_png_path=page_png,
                        page_png_sha256=sha256_bytes(data), crop_png=crop_png, context=ctx,
                        page_meta=prow.get("render") or {"render_dpi": 0, "renderer_version": "unknown"},
                        ocr_engine_version=(col.PRECROPPED_ENGINE if row["source_crop"]["frame"] == col.PRECROPPED_FRAME
                                            else col.OCR_ENGINE), ocr_status=prow.get("ocr_status") or "unavailable",
                        work_id=man.get("work_id") or "wk_unknown", th=th, dataset_version=dataset_version,
                        rights_manifest_sha256=man.get("manifest_sha256"), automatic_metadata=am)
    write_json_immutable(candidates_dir(layout) / f"{iid}.json", rec)
    return iid


# --------------------------------------------------------------------------- items-finalize
def _production_frozen(th) -> bool:
    return th.is_production and bool(th.frozen_at) and not th.missing_for_production()


def finalize_items(layout: Layout, th, *, release_version: str, seed: int,
                   development_only: bool = False) -> dict:
    """§6.7 dedup, R3 + duplicate exclusions, §8.7 card gates and book cap, split groups and the
    development/test assignment, final §5.2 records, final funnel outcomes."""
    from ..cards.consensus import semantic_card_summary_for
    from ..cards.gates import apply_book_cap
    from ..cards.rederive import eligible_jaccard
    from ..sampling import apply_collapse_guard, assign_splits, dedup_representatives, split_inputs, write_split_groups
    if th.is_prefreeze:
        return {"refused": "items-finalize assigns the development/test split and writes final records; it never "
                           "runs in a pre-freeze context"}
    cdir = candidates_dir(layout)
    cands = {p.stem: read_json(p) for p in sorted(cdir.glob("*.json"))} if cdir.exists() else {}
    mans = load_manifests(layout.rights_manifest)
    rows = latest_rows(layout)
    report = {"candidates": len(cands), "awaiting_card": [], "outcomes": {}, "records": {}, "refused": None}
    cards, gates = {}, {}
    from ..cards.batch import card_selection_excluded
    not_selected = card_selection_excluded(layout)      # `card-select`: never carded; funnel rows already written
    report["not_selected"] = len(not_selected)
    for iid in cands:
        if iid in not_selected:
            continue
        cp, gp = layout.card(iid), layout.p("data", "card_gates", f"{iid}.json")
        if not cp.exists() or not gp.exists():
            report["awaiting_card"].append(iid)
            continue
        cards[iid] = Card.model_validate(read_json(cp))
        gates[iid] = read_json(gp)
    ready = sorted(cards)
    status: dict[str, dict] = {iid: {"reasons": [], "outcome": None, "exit": None} for iid in ready}
    # R3 (value-checked, current manifests)
    for iid in ready:
        book = cands[iid]["source"]["book_id"]
        man = mans.get(book)
        if man is None or not r3_gate(man, th.profile)["passed"]:
            status[iid].update(reasons=[Exclusion.rights_unresolved.value], outcome="excluded", exit="S4")
    # §6.7 dedup over pHash clusters (cards' eligible-fact Jaccard)
    live = [i for i in ready if status[i]["outcome"] is None]
    clusters: dict[str, list[str]] = {}
    for iid in live:
        cid = cands[iid]["quality"]["phash_cluster_id"]
        clusters.setdefault(cid if cid != "singleton" else f"singleton:{iid}", []).append(iid)
    ecount = {i: sum(1 for f in cards[i].facts if f.eligible) for i in live}
    frag = {i: (cands[i]["automatic_metadata"].get("image_stats") or {}).get("frag_index") for i in live}
    dd = dedup_representatives({c: m for c, m in clusters.items() if len(m) > 1}, {i: [] for i in live}, ecount,
                               frag, th, similarity=lambda a, b: eligible_jaccard(cards[a], cards[b], int(th.value("wl_rounds"))))
    for iid in live:
        d = dd.get(iid)
        status[iid]["representative"] = True if d is None else d["is_cluster_representative"]
        if d and not d["is_cluster_representative"]:
            status[iid].update(reasons=[Exclusion.duplicate_representative_lost.value], outcome="excluded", exit="S5")
    # §8.7 item gates (from cards-build) and the book-share cap
    for iid in [i for i in ready if status[i]["outcome"] is None]:
        reasons = list(gates[iid].get("exclusion_reasons") or [])
        if ecount.get(iid, 0) == 0:
            status[iid].update(reasons=reasons or [Exclusion.card_insufficient_facts.value],
                               outcome="unscored_pool", exit="S7")
        elif reasons:
            status[iid].update(reasons=reasons, outcome="unscored_pool", exit="S8")
    pre = [i for i in ready if status[i]["outcome"] is None]
    cap = apply_book_cap([{"item_id": i, "book_id": cands[i]["source"]["book_id"],
                           "eligible_mgs_count": ecount[i]} for i in pre], th)
    for m in cap["moved"]:
        status[m["item_id"]].update(reasons=[m["reason"]], outcome="unscored_pool", exit="S8")
    for i in pre:
        if status[i]["outcome"] is None:
            status[i]["outcome"] = "eligible"
    report["book_cap"] = {k: cap[k] for k in ("cap", "infeasible", "rule")} | {"moved": [m["item_id"] for m in cap["moved"]]}
    # split groups (collapse guard resolved by the standing rule, sampling.COLLAPSE_RULE)
    eligible = {i for i in ready if status[i]["outcome"] == "eligible"}
    books, items, elig = split_inputs(layout, eligible_item_ids=eligible)
    graph = apply_collapse_guard(books, items, elig, th)
    try:
        split = assign_splits(graph, seed, th, development_books(layout))
    except ValueError as e:                     # too few split groups for 3 dev groups + a test split
        report["refused"] = f"split assignment: {e}"
        return report
    write_split_groups(layout, graph, split)
    split_of = {}
    for iid in ready:
        g = graph["book_group"].get(cands[iid]["source"]["book_id"])
        split_of[iid] = (g, split["assignment"][g]) if g else (None, "reserved")
    dev_in_test = sorted({cands[i]["source"]["book_id"] for i in ready if split_of[i][1] == "test"}
                         & development_books(layout))
    if dev_in_test:                               # partition separation: pre-freeze books never become test books
        report["refused"] = (f"development books used before the level freeze would enter the test split: "
                             f"{dev_in_test}; nothing written (owner: choose the split seed or the book set)")
        return report
    n_test = sum(1 for i in ready if split_of[i][1] == "test")
    if n_test and not _production_frozen(th) and not development_only:
        report["refused"] = (f"{n_test} test-split records need a production level-frozen thresholds.json; "
                             "nothing written (use --development-only for engineering runs)")
        return report
    for iid in ready:
        gid, sp = split_of[iid]
        if sp == "test" and development_only:
            continue
        rec = dict(cands[iid])
        man = mans.get(rec["source"]["book_id"])
        # provenance: the rights record this run's R3 check read (its manifest_sha256), not a schema tag
        rec["source"] = dict(rec["source"], rights_manifest_sha256=(man or {}).get("manifest_sha256"))
        st = status[iid]
        card = cards[iid]
        summary = None
        try:
            summary = semantic_card_summary_for(card, layout).model_dump(mode="json")
        except (ValueError, FileNotFoundError) as e:
            report.setdefault("summary_unavailable", {})[iid] = str(e)
        if summary is None and st["outcome"] == "eligible":       # the card itself is unreadable
            st.update(outcome="unscored_pool", reasons=[Exclusion.card_insufficient_facts.value], exit="S8")
        rec.update(dataset_version=release_version, semantic_card=summary, split_group=gid, split=sp,
                   quality={"eligible": st["outcome"] == "eligible", "exclusion_reasons": st["reasons"],
                            "funnel_exit_stage": None if st["outcome"] == "eligible" else st["exit"],
                            "phash_cluster_id": rec["quality"]["phash_cluster_id"],
                            "is_cluster_representative": bool(st.get("representative", True))})
        rec = ItemRecord.model_validate(rec).model_dump(mode="json")
        if sp == "test" and not layout.test_artifacts_exist():
            from datetime import datetime, timezone
            layout.mark_first_test_artifact(f"first test-split item record {iid}",
                                            datetime.now(timezone.utc).isoformat(timespec="seconds"))
        write_json_immutable(layout.item_record(sp, iid), rec)
        report["records"][iid] = sp
        report["outcomes"][st["outcome"]] = report["outcomes"].get(st["outcome"], 0) + 1
        pid = rec["extraction"]["proposal_id"]
        prev = rows.get(pid)
        if prev and (prev.get("outcome"), prev.get("exclusion_reasons"), prev.get("item_id")) != (
                st["outcome"], st["reasons"], iid):
            append_row(layout, dict(prev, outcome=FunnelOutcome(st["outcome"]).value, item_id=iid,
                                    exclusion_reasons=st["reasons"],
                                    is_cluster_representative=bool(st.get("representative", True))))
    report["split"] = {"groups": len(graph["groups"]), "held_out_claim": split["held_out_claim"],
                       "collapse_guard_action": graph["collapse_guard"]["action"]}
    return report


def load_roster_file(path: str) -> dict:
    from ..roster import load_roster
    return load_roster(path)


__all__ = ["run_g8", "finalize_items", "ocr_words_in_crop", "load_roster_file"]
