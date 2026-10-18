"""CLI for acquisition: list collector sources, snapshot, run the CPU ladder, fixture smoke.

The smoke command runs only under the fixture profile, into a scratch root, with `fixture:`
manifests; its outputs are engineering evidence and never benchmark items.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..config import fixture_thresholds, load_thresholds
from ..rights import (apply_us_rule, dump_yaml, load_manifests, manifest_sha256, probe_master, r3_gate,
                      render_titlepage_png, write_evidence)
from ..store import Layout, write_mutable
from ..vocab import Profile
from . import collection as col
from .ladder import process_source, summarize


def fixture_manifest(layout: Layout, snap: dict) -> dict:
    """A manifest for engineering runs ONLY: every human field says `fixture`, approved_by is
    `fixture:smoke`. The production R3 gate refuses it by construction."""
    master = col.verified_master_path(snap)
    probe = probe_master(master)
    yr = ((snap.get("year_evidence") or {}).get("record") or {})
    book = snap["book_id"]
    cat = write_evidence(layout, book, "catalog.json",
                         (json.dumps({"source": "corpus/metadata/file_years.json", "record": yr,
                                      "file_sha256": (snap.get("year_evidence") or {}).get("file_sha256")},
                                     indent=1, ensure_ascii=False) + "\n").encode())
    tp = write_evidence(layout, book, "titlepage.png", render_titlepage_png(master, 0, 100))
    terms = write_evidence(layout, book, "terms.html", b"fixture: repository terms not recorded\n")
    rec = {
        "book_id": book, "work_id": "wk_fixture_" + book[3:], "display_title": "fixture:" + snap["source_name"],
        "author": "fixture", "publisher": "fixture", "publisher_id": "pub_fixture", "edition_statement": "fixture",
        "is_reprint_of_earlier_work": "unknown", "publication_year": yr.get("year"), "published": "yes",
        "publication_place": "fixture", "first_publication_country": "XX", "language": "und",
        "acquisition": {"source_repository": "fixture", "source_url": "fixture", "download_utc": "fixture",
                        "source_file_sha256": probe["source_file_sha256"],
                        "source_file_bytes": probe["source_file_bytes"], "page_count": probe["page_count"],
                        "source_codec": probe["source_codec"], "source_bitdepth": probe["source_bitdepth"] or 8,
                        "source_colorspace": probe["source_colorspace"] or "gray", "scanning_agent": None,
                        "source_dpi_nominal": None},
        "publication_year_evidence": {"catalog_record": cat["path"], "catalog_record_sha256": cat["sha256"],
                                      "titlepage_asset": tp["path"], "titlepage_asset_sha256": tp["sha256"],
                                      "titlepage_page_index": 0, "agree": True},
        "terms": {"terms_url": "fixture", "terms_snapshot": terms["path"], "terms_snapshot_sha256": terms["sha256"],
                  "terms_snapshot_utc": "fixture", "repository_rights_statement_verbatim": "fixture",
                  "redistribution_permitted": "unknown", "commercial_reuse_permitted": "unknown",
                  "ai_use_clause": "unknown", "required_attribution_string": None},
        "determination": {"us_status": "not_determined", "us_rule_id": "us-pd-pre1931-v1", "us_determined_by": "rule",
                          "source_country": "XX", "source_country_status": "not_determined",
                          "release_country_status": "not_determined", "determined_by_model": False},
        "prominence": {"hosted_by_major_aggregators": 0, "public_fulltext_layer": False},
        "release_mode": "server_only", "approved_by": "fixture:smoke", "approved_utc": "fixture",
        "manifest_version": "rights-1"}
    rec["manifest_sha256"] = manifest_sha256(rec)
    return apply_us_rule(rec)


def _pages_with_images(snap: dict, k: int) -> list[int]:
    try:
        ocr = col.parse_ocr(snap)
    except col.OcrMalformed:
        return list(range(k))
    return [p["page_index"] for p in ocr["pages"]
            if p["page_index"] >= 5 and any(b["label"] in col.ITEM_BLOCK_LABELS for b in p["blocks"])][:k]


def run_smoke(corpus: Path, root: Path, sources: list[str], n_pages: int) -> dict:
    layout = Layout(root.resolve())
    th = fixture_thresholds()
    report = {"profile": "fixture", "note": "engineering smoke; not benchmark items", "sources": []}
    for name in sources:
        snap = col.snapshot_source(corpus, name)
        col.store_snapshot(layout, snap)
        man = fixture_manifest(layout, snap)
        prod_r3 = r3_gate(man, Profile.production)
        pages = _pages_with_images(snap, n_pages)
        summ = process_source(layout, snap, man, th, profile=Profile.fixture, page_indices=pages)
        s = summarize(summ)
        s["production_r3_on_fixture_manifest"] = prod_r3
        s["production_r3_without_manifest"] = process_source(
            Layout((root / "_prod_refusal").resolve()), snap, None, th, profile=Profile.production,
            page_indices=pages)["r3"]
        s["page_indices"] = pages
        report["sources"].append(s)
    out = layout.reports / "acq_smoke_report.json"
    write_mutable(out, (json.dumps(report, indent=1, ensure_ascii=False) + "\n").encode())
    report["report_path"] = str(out)
    return report


def _cmd_sources(args) -> int:
    src = col.list_sources(args.corpus)
    by: dict[str, int] = {}
    for s in src:
        by[str(s["state"])] = by.get(str(s["state"]), 0) + 1
    print(json.dumps({"by_state": by}, indent=1))
    return 0


def _cmd_run(args) -> int:
    from ..config import CalibrationMissing, ThresholdsError, prefreeze_record, resolve_thresholds_arg
    from ..store import FreezeViolation
    from .drivers import declare_development_book, require_development_context
    from .gates import GateBlocked
    lay = Layout.from_env(args.root)
    try:
        th = resolve_thresholds_arg(args.thresholds, lay) if args.thresholds else fixture_thresholds()
        require_development_context(th, "acq-run")
    except (ThresholdsError, FreezeViolation, GateBlocked, FileNotFoundError) as e:
        print(f"REFUSED: {e}")
        return 2
    if args.profile == "production" and not th.is_production:
        print("production runs need a production thresholds file")
        return 2
    snap = col.snapshot_source(args.corpus, args.source)
    col.store_snapshot(lay, snap)
    man = load_manifests(lay.rights_manifest).get(snap["book_id"])
    pages = [int(x) for x in args.pages.split(",")] if args.pages else None
    if th.is_prefreeze:           # a book processed before the level freeze is a development book
        declare_development_book(lay, snap["book_id"], snap["source_name"], th.context or "")
    try:
        out = summarize(process_source(lay, snap, man, th, profile=args.profile, page_indices=pages))
    except CalibrationMissing as e:
        print(f"REFUSED: {e}")
        return 2
    prefreeze_record(th, lay, f"acq-run {args.source}", [lay.funnel, lay.p("manifests", "funnel_pages.jsonl")],
                     inputs={"book_id": snap["book_id"], "snapshot_id": snap["snapshot_id"]})
    print(json.dumps(out, indent=1, ensure_ascii=False))
    return 0


def _cmd_smoke(args) -> int:
    rep = run_smoke(Path(args.corpus), Path(args.root_dir), args.source, args.pages)
    print(json.dumps({"report": rep["report_path"], "sources": rep["sources"]}, indent=1, ensure_ascii=False))
    return 0


def _cmd_fixture_manifest(args) -> int:
    lay = Layout.from_env(args.root)
    snap = col.snapshot_source(args.corpus, args.source)
    print(dump_yaml(fixture_manifest(lay, snap)).decode())
    return 0


def _load_th(lay: Layout, path: str | None):
    if path:
        from ..config import resolve_thresholds_arg
        return resolve_thresholds_arg(path, lay)
    if lay.thresholds.exists():
        return load_thresholds(lay.thresholds)
    return fixture_thresholds()


def _cmd_g8(args) -> int:
    from .drivers import load_roster_file, run_g8
    from ..config import prefreeze_record
    lay = Layout.from_env(args.root)
    try:
        th = _load_th(lay, args.thresholds)
    except Exception as e:            # ThresholdsError, FreezeViolation: a closed or unknown context
        print(f"acq-g8 refused: {type(e).__name__}: {e}")
        return 2
    book = None
    if args.source:
        book = col.snapshot_source(args.corpus, args.source)["book_id"]
    ledger = None
    if args.allow_live:
        from ..ledger import Ledger
        ledger = Ledger(lay)
    try:
        out = run_g8(lay, th, load_roster_file(args.roster), ledger=ledger, allow_live=args.allow_live,
                     book_id=book, max_tokens=args.max_tokens, dataset_version=args.dataset_version)
    except Exception as e:            # GateBlocked, roster/ledger errors: report, never guess
        print(f"acq-g8 refused: {type(e).__name__}: {e}")
        return 2
    from .drivers import candidates_dir
    prefreeze_record(th, lay, "acq-g8", [candidates_dir(lay) / f"{i}.json" for i in out["candidates"]] + [lay.funnel])
    print(json.dumps(out, indent=1))
    return 0 if not out["stopped"] else 3


def _cmd_finalize(args) -> int:
    from .drivers import finalize_items
    lay = Layout.from_env(args.root)
    try:
        th = _load_th(lay, args.thresholds)
    except Exception as e:            # ThresholdsError, FreezeViolation
        print(f"items-finalize refused: {type(e).__name__}: {e}")
        return 2
    rep = finalize_items(lay, th, release_version=args.release_version, seed=args.seed,
                         development_only=args.development_only)
    print(json.dumps(rep, indent=1, default=str))
    return 2 if rep.get("refused") else 0


def register_cli(sub) -> None:
    p = sub.add_parser("acq-g8", help="paid gate G8b (strict gatekeeper) -> candidate records (resumable, cached)")
    p.add_argument("--roster", required=True)
    p.add_argument("--source", default=None, help="restrict to one collector source")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--thresholds", default=None)
    p.add_argument("--allow-live", action="store_true")
    p.add_argument("--max-tokens", type=int, default=None, help="engineering override of g8_max_tokens")
    p.add_argument("--dataset-version", default="v1.0")
    p.set_defaults(func=_cmd_g8)
    p = sub.add_parser("items-finalize", help="dedup, gates, book cap, splits, final item records")
    p.add_argument("--release-version", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--thresholds", default=None)
    p.add_argument("--development-only", action="store_true",
                   help="engineering: write development records only (test records need frozen levels)")
    p.set_defaults(func=_cmd_finalize)
    p = sub.add_parser("acq-sources", help="collector sources by status state (read-only)")
    p.add_argument("--corpus", default="corpus")
    p.set_defaults(func=_cmd_sources)
    p = sub.add_parser("acq-run", help="CPU ladder (R3, G0-G7d) for one completed source")
    p.add_argument("source")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--profile", choices=["fixture", "production"], default="production")
    p.add_argument("--thresholds", default=None, help="a thresholds.json path, or prefreeze:PILOT|DEV (development "
                                                      "books before the level freeze)")
    p.add_argument("--pages", default=None, help="comma-separated 0-based page indices")
    p.set_defaults(func=_cmd_run)
    p = sub.add_parser("acq-smoke", help="fixture-profile smoke on real pages into a scratch root")
    p.add_argument("--corpus", default="corpus")
    p.add_argument("--root-dir", required=True)
    p.add_argument("--source", action="append", required=True)
    p.add_argument("--pages", type=int, default=4)
    p.set_defaults(func=_cmd_smoke)
    p = sub.add_parser("acq-fixture-manifest", help="print a fixture-only manifest (engineering use)")
    p.add_argument("source")
    p.add_argument("--corpus", default="corpus")
    p.set_defaults(func=_cmd_fixture_manifest)
