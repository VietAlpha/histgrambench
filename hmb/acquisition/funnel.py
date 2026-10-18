"""Append-only selection funnel (§6.6) and item record assembly (§5.1-§5.2).

Funnel rows are validated `FunnelRow` records appended with `hmb.store.append_jsonl`; rows are
never rewritten or deleted. A later stage (G8, cards) appends a newer row for the same
`proposal_id`; readers take the last row per proposal (`latest_rows`). Page-level gate records
(G0a-G0d) go to a sibling append-only file `manifests/funnel_pages.jsonl`.

Crop canonicalisation is the shared §9.7 pipeline owned by Opus 3
(`hmb.submission.render.canonicalize_image`). There is deliberately no local copy: until it
exists, `canonical_crop` raises `CanonicalizerUnavailable`.
"""
from __future__ import annotations

from pathlib import Path

from ..hashing import item_id as make_item_id, sha256_json
from ..rights import MANIFEST_VERSION
from ..schemas import FunnelRow, ItemRecord
from ..store import Layout, append_jsonl, iter_jsonl, write_immutable
from ..vocab import FunnelOutcome
from .imgstats import STATS_VERSION

PIPELINE_VERSION = "extract-1.0.0"


class CanonicalizerUnavailable(RuntimeError):
    """The shared §9.7 canonicalization (Opus 3) is not importable yet."""


def page_funnel_path(layout: Layout) -> Path:
    return layout.p("manifests", "funnel_pages.jsonl")


def proposal_id(book: str, page_index: int, box, box_source: str, parent: str | None) -> str:
    return "prop_" + sha256_json([book, page_index, [int(v) for v in box], box_source, parent])[:16]


def config_hash(th, extra: dict | None = None) -> str:
    return sha256_json({"thresholds_hash": th.hash, "pipeline_version": PIPELINE_VERSION, **(extra or {})})


def finalize(row: dict) -> dict:
    """terminal_gate = first failed gate or 'passed'; outcome = excluded on any failure/reason."""
    failed = next((g["gate_id"] for g in row["gates"] if not g["passed"]), None)
    row["terminal_gate"] = failed or "passed"
    if failed or row.get("exclusion_reasons"):
        row["outcome"] = FunnelOutcome.excluded.value
    return row


def append_row(layout: Layout, row: dict) -> dict:
    """Validate against FunnelRow and append; returns the stored dict."""
    rec = FunnelRow.model_validate(row).model_dump(mode="json")
    append_jsonl(layout.funnel, rec)
    return rec


def append_page_row(layout: Layout, row: dict) -> None:
    append_jsonl(page_funnel_path(layout), row)


def latest_rows(layout: Layout) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in iter_jsonl(layout.funnel):
        out[r["proposal_id"]] = r
    return out


def page_rows(layout: Layout) -> list[dict]:
    return list(iter_jsonl(page_funnel_path(layout)))


# --------------------------------------------------------------------------- canonical crop
def _canonicalize_fn():
    try:
        from ..submission.render import canonicalize_image   # type: ignore[import-not-found]
        return canonicalize_image
    except ImportError:
        pass
    try:
        from ..submission import canonicalize_image           # type: ignore[attr-defined]
        return canonicalize_image
    except ImportError as e:
        raise CanonicalizerUnavailable(
            "hmb.submission.render.canonicalize_image (Opus 3, §9.7) is not available; "
            "no local substitute is used") from e


def canonical_crop(crop_png: bytes, th) -> tuple[bytes, dict]:
    return _canonicalize_fn()(crop_png, th)


# --------------------------------------------------------------------------- item record §5.2
def default_automatic_metadata(row: dict) -> dict:
    """§5.2 automatic_metadata from a funnel row (G7a border measures + image statistics). Image
    statistics describe Paddle's processed crop, not the source page (D3, 2026-09-23). The border
    measures are recorded, never an exclusion (D10); growth_steps is always 0."""
    return {"crop_border": {"border_ink": row.get("border_ink"), "touching_cc": row.get("touching_cc"),
                            "growth_steps": row.get("growth_steps")},
            "image_stats": {"stats_version": STATS_VERSION, "measured_on": "paddle_crop",
                            **(row.get("img_stats") or {})}}


def assemble_item(layout: Layout, row: dict, *, page_png_path: Path, page_png_sha256: str, crop_png: bytes,
                  context: dict, page_meta: dict, ocr_engine_version: str, ocr_status: str, work_id: str,
                  th, dataset_version: str, rights_manifest_sha256: str | None,
                  printed_page_label: str | None = None, automatic_metadata: dict | None = None) -> dict:
    """Write crop / canonical crop / context-text assets immutably and return a validated ItemRecord dict.

    Every path field gets a `*_sha256` sibling over the bytes as stored (§5.1). Paths are relative
    to `images/`. `quality` starts from the funnel row; eligibility is decided downstream. The item
    image is a Paddle crop (`row["source_crop"]`); there is no §6.4 context image, because the
    flattened page it would be cut from was never saved (D3, 2026-09-23). `rights_manifest_sha256` is
    the `manifest_sha256` of the book's rights record the caller loaded (None when the book has none);
    `items-finalize` replaces it with the hash of the record R3 approved.
    """
    box = [int(v) for v in row["box_xyxy_page_px"]]
    iid = make_item_id(row["book_id"], row["page_index"], box)
    crop_sha = write_immutable(layout.crop(iid), crop_png)
    canon_png, _canon_meta = canonical_crop(crop_png, th)
    canon_sha = write_immutable(layout.crop_canonical(iid), canon_png)
    ctx_text = context.get("context_text", "").encode("utf-8")
    ctx_sha = write_immutable(layout.context_text(iid), ctx_text)
    rel = lambda p: Path(p).relative_to(layout.p("images")).as_posix()   # noqa: E731
    rec = {
        "id": iid, "dataset_version": dataset_version,
        "source": {"book_id": row["book_id"], "work_id": work_id, "rights_manifest_version": MANIFEST_VERSION,
                   "rights_manifest_sha256": rights_manifest_sha256,
                   "page_index": row["page_index"], "printed_page_label": printed_page_label},
        "extraction": {
            "pipeline_version": PIPELINE_VERSION, "render_dpi": int(page_meta.get("render_dpi") or 0),
            "color_mode": "gray8", "renderer_version": page_meta["renderer_version"],
            "proposal_id": row["proposal_id"], "crop_bbox": box, "context_bbox": context["context_bbox"],
            "source_crop": row["source_crop"], "ocr_engine_version": ocr_engine_version, "ocr_status": ocr_status,
            "caption_stripped": bool(row.get("caption_stripped")), "panel_of": row.get("parent_proposal_id"),
            "context_char_count": context["context_char_count"],
            "context_block_count": context["context_block_count"], "context_window_rule": "ctxwin-1"},
        "assets": {"page_image": rel(page_png_path), "page_image_sha256": page_png_sha256,
                   "crop_image": rel(layout.crop(iid)), "crop_image_sha256": crop_sha,
                   "crop_canonical_image": rel(layout.crop_canonical(iid)), "crop_canonical_sha256": canon_sha,
                   "context_text": rel(layout.context_text(iid)), "context_text_sha256": ctx_sha},
        "automatic_metadata": {"ocr_words": [], **(automatic_metadata or default_automatic_metadata(row))},
        "quality": {"eligible": False, "exclusion_reasons": row.get("exclusion_reasons", []),
                    "funnel_exit_stage": None,
                    "phash_cluster_id": row.get("phash_cluster_id") or "singleton",
                    "is_cluster_representative": bool(row.get("is_cluster_representative"))},
    }
    return ItemRecord.model_validate(rec).model_dump(mode="json")


__all__ = ["PIPELINE_VERSION", "CanonicalizerUnavailable", "proposal_id", "config_hash", "finalize",
           "append_row", "append_page_row", "latest_rows", "page_rows", "canonical_crop", "assemble_item"]
