"""Read-only adapter to the running collector's outputs (corpus/).

Boundary rules:
  * never writes, moves or renames anything under the corpus root;
  * a source is consumed only when `status.json` says `"state": "done"`;
  * `snapshot_source` records size + sha256 of every input file it will consume (master,
    status.json, result.json, raw chunks, image crops, the year-metadata file), and every later
    read goes through `read_verified`, which re-checks size and hash, so a file that was
    partially written or changed after the snapshot raises `SnapshotChanged` instead of being used.

Items are Paddle's saved crops (owner decision D3/D4, 2026-09-23, reports/decision_audit_20260923.md).
The collector ran PaddleOCR-VL with `useDocUnwarping` and `useDocOrientationClassify`, so every
`block_bbox` and every saved crop lives in Paddle's preprocessed frame (rotated upright, dewarped),
not the page frame. The crops are used as they are: Paddle is the only region source, there is no
matching back to the page, no re-OCR and no reconstruction. Each item carries `source_crop_record`:
the collector file and its snapshot sha256, the box labelled as Paddle-frame coordinates, and the
disclosed modifications (rotated where the page was, dewarped, resampled, JPEG re-compressed, bounded by
Paddle's box). A crop cannot be regenerated from the master. Paddle's text blocks share the crop frame.
"""
from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import numpy as np

from ..hashing import book_id as make_book_id, sha256_bytes, sha256_json
from ..store import Layout, write_json_immutable

SNAPSHOT_VERSION = "collect-snap-1"
DONE = "done"
MASTER_DIR = "wikimedia-2026-09-21"
OCR_DIR = "corpus_output"
YEARS_FILE = "metadata/file_years.json"
OCR_ENGINE = "PaddleOCR-VL-1.6"
# Paddle `image` and `chart` crops become items (D13, 2026-09-23); non-target charts are rejected
# downstream by the G8b gatekeeper's target-type test.
ITEM_BLOCK_LABELS = ("image", "chart")
_IMG_RE = re.compile(r"^imgs/p(\d{5})/img_in_(?:image|chart)_box_(\d+)_(\d+)_(\d+)_(\d+)\.jpg$")
REGION_SOURCE = "paddle_layout@1.6"          # the only region source (D4 revised by D3)
FRAME = "paddle_preprocessed"                # coordinate frame of block_bbox and of every saved crop
CROP_MODIFICATIONS = ("dewarped", "resampled", "jpeg_recompressed", "paddle_bounds")


# Pre-cropped sources: a collector directory with figure images only (no status.json, no result.json, no master
# page file), e.g. Project Gutenberg HTML illustrations, figures extracted from a PDF, or Paddle crops whose pages
# were not kept. Each image is one source page and one item box (the whole image). The master is the listing
# of the image files (name, bytes, sha256), so the book_id is fixed by the image contents.
PRECROPPED = "precropped"
PRECROPPED_FRAME = "precropped_image"
PRECROPPED_SOURCE = "precropped@1"
PRECROPPED_ENGINE = "none (pre-cropped source)"
LISTING_VERSION = "precropped-listing-1"
_PRECROP_EXT = (".png", ".jpg", ".jpeg", ".gif")
_PAGE_RE = re.compile(r"(?:^|/)p(\d{3,5})(?:[/_]|$)")
PRECROPPED_MODS = {"paddle_crop_without_page": ["cropped_by_collector", "dewarped", "resampled", "jpeg_recompressed",
                                                "paddle_bounds"],
                   "pdf_embedded_image": ["cropped_by_collector"],
                   "plate_figure_crop": ["cropped_by_collector", "painted_over"],
                   "html_image": ["cropped_by_collector"]}


class SnapshotChanged(RuntimeError):
    """An input file differs from the snapshot (partially written or changed later)."""


class SourceNotReady(RuntimeError):
    """The source is not a completed collector artifact."""


class OcrMalformed(ValueError):
    """The OCR output does not have the expected structure; the source's OCR is unusable."""


# --------------------------------------------------------------------------- listing
def list_sources(corpus_root: str | Path) -> list[dict]:
    """All OCR output directories with their status state (None if status.json is unreadable)."""
    base = Path(corpus_root) / OCR_DIR
    out = []
    for d in sorted(os.listdir(base)):
        p = base / d / "status.json"
        state = None
        try:
            state = json.loads(p.read_text(encoding="utf-8")).get("state")
        except (OSError, ValueError, AttributeError):
            pass
        out.append({"name": d, "state": state})
    return out


def done_sources(corpus_root: str | Path) -> list[str]:
    return [s["name"] for s in list_sources(corpus_root) if s["state"] == DONE]


# --------------------------------------------------------------------------- snapshot
def _file_entry(path: Path) -> dict:
    data = path.read_bytes()
    return {"bytes": len(data), "sha256": sha256_bytes(data)}


def snapshot_source(corpus_root: str | Path, name: str) -> dict:
    """Immutable snapshot boundary for one completed source. Raises SourceNotReady otherwise."""
    root = Path(corpus_root)
    if "/" in name or "\\" in name or name in (".", ".."):
        raise SourceNotReady(f"bad source name {name!r}")
    ocr_dir = root / OCR_DIR / name
    status_path = ocr_dir / "status.json"
    if not status_path.exists() and not (root / MASTER_DIR / name).exists():
        return snapshot_precropped(root, name)
    try:
        status_bytes = status_path.read_bytes()
        status = json.loads(status_bytes.decode("utf-8"))
    except (OSError, ValueError) as e:
        raise SourceNotReady(f"{name}: status.json unreadable ({e})") from e
    if not isinstance(status, dict) or status.get("state") != DONE:
        raise SourceNotReady(f"{name}: state={status.get('state') if isinstance(status, dict) else None!r}")
    master = root / MASTER_DIR / name
    if not master.is_file() and not (ocr_dir / "result.json").exists():
        return snapshot_precropped(root, name)   # collector exported figures only (no master, no OCR JSON)
    if not master.is_file():
        raise SourceNotReady(f"{name}: master file missing")
    return _snapshot_paddle(root, name, ocr_dir, status_path, status_bytes, status, master)


def _snapshot_paddle(root: Path, name: str, ocr_dir: Path, status_path: Path, status_bytes: bytes, status: dict,
                     master: Path) -> dict:
    files: dict[str, dict] = {}
    for rel in ["status.json", "result.json"] + sorted(
            f"raw/{p.name}" for p in (ocr_dir / "raw").glob("*.jsonl")):
        if not (ocr_dir / rel).is_file():
            raise SourceNotReady(f"{name}: {rel} missing")
        files[rel] = _file_entry(ocr_dir / rel)
    imgs = ocr_dir / "imgs"
    if imgs.is_dir():
        for p in sorted(imgs.rglob("*")):
            if p.is_file():
                files[p.relative_to(ocr_dir).as_posix()] = _file_entry(p)
    if files["status.json"]["sha256"] != sha256_bytes(status_bytes):
        raise SnapshotChanged(f"{name}: status.json changed while snapshotting")
    master_entry = _file_entry(master)
    years = root / YEARS_FILE
    year_ev = None
    if years.is_file():
        yb = years.read_bytes()
        rec = json.loads(yb.decode("utf-8-sig")).get(name)
        year_ev = {"file": YEARS_FILE, "file_bytes": len(yb), "file_sha256": sha256_bytes(yb),
                   "record": rec}
    snap = {"snapshot_version": SNAPSHOT_VERSION, "corpus_root": str(root.resolve()),
            "source_name": name, "kind": status.get("kind"), "state": DONE,
            "master": {"path": f"{MASTER_DIR}/{name}", **master_entry},
            "book_id": make_book_id(master_entry["sha256"]),
            "ocr_dir": f"{OCR_DIR}/{name}", "files": files, "year_evidence": year_ev}
    # A status.json rewritten during the pass would already differ; check once more.
    if sha256_bytes(status_path.read_bytes()) != files["status.json"]["sha256"]:
        raise SnapshotChanged(f"{name}: status.json changed while snapshotting")
    snap["snapshot_id"] = "snap_" + sha256_json(snap)[:16]
    return snap


def _precropped_items(ocr_dir: Path) -> tuple[str, list[str]]:
    """(origin, image paths relative to the source directory) of a pre-cropped source."""
    man = ocr_dir / "manifest.json"
    if man.is_file():
        try:
            data = json.loads(man.read_text(encoding="utf-8"))
        except ValueError as e:
            raise SourceNotReady(f"{ocr_dir.name}: manifest.json unreadable ({e})") from e
        figs = data.get("figures") or data.get("images") or []
        rels = [f["file"].replace("\\", "/") for f in figs if isinstance(f, dict) and f.get("file")]
        origin = "pdf_embedded_image" if data.get("source_pdf") else "plate_figure_crop"
    else:
        imgs = ocr_dir / "imgs"
        found = sorted(p.relative_to(ocr_dir).as_posix() for p in imgs.rglob("*")
                       if p.is_file() and p.suffix.lower() in _PRECROP_EXT) if imgs.is_dir() else []
        paddle = [r for r in found if _IMG_RE.match(r)]
        origin, rels = ("paddle_crop_without_page", paddle) if paddle else ("html_image", found)
    missing = [r for r in rels if not (ocr_dir / r).is_file()]
    if missing:
        raise SourceNotReady(f"{ocr_dir.name}: listed images missing: {missing[:3]}")
    return origin, sorted(dict.fromkeys(rels))


def _listing_bytes(name: str, origin: str, items: list[str], files: dict) -> bytes:
    body = {"listing_version": LISTING_VERSION, "source_name": name, "origin": origin,
            "items": [{"file": r, "bytes": files[r]["bytes"], "sha256": files[r]["sha256"]} for r in items]}
    return (json.dumps(body, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def is_precropped(snap: dict) -> bool:
    return snap.get("kind") == PRECROPPED


def precropped_listing(snap: dict) -> bytes:
    """The master of a pre-cropped source: its image listing, rebuilt from the snapshot."""
    pc = snap["precropped"]
    return _listing_bytes(snap["source_name"], pc["origin"], pc["items"], snap["files"])


def snapshot_precropped(root: Path, name: str) -> dict:
    ocr_dir = root / OCR_DIR / name
    if not ocr_dir.is_dir():
        raise SourceNotReady(f"{name}: no source directory")
    origin, items = _precropped_items(ocr_dir)
    if not items:
        raise SourceNotReady(f"{name}: no status.json and no images")
    files = {r: _file_entry(ocr_dir / r) for r in items}
    for idx in ("manifest.json", "source.json", "status.json"):
        if (ocr_dir / idx).is_file():
            files[idx] = _file_entry(ocr_dir / idx)
    listing = _listing_bytes(name, origin, items, files)
    lsha = sha256_bytes(listing)
    snap = {"snapshot_version": SNAPSHOT_VERSION, "corpus_root": str(root.resolve()), "source_name": name,
            "kind": PRECROPPED, "state": PRECROPPED, "precropped": {"origin": origin, "items": items},
            "master": {"path": None, "listing_version": LISTING_VERSION, "bytes": len(listing), "sha256": lsha},
            "book_id": make_book_id(lsha), "ocr_dir": f"{OCR_DIR}/{name}", "files": files, "year_evidence": None}
    snap["snapshot_id"] = "snap_" + sha256_json(snap)[:16]
    return snap


def store_snapshot(layout: Layout, snap: dict) -> Path:
    path = layout.p("sources", "collection_snapshots", f"{snap['book_id']}_{snap['snapshot_id']}.json")
    write_json_immutable(path, snap)
    return path


def read_verified(snap: dict, rel: str, *, master: bool = False) -> bytes:
    """Read one snapshotted input file, re-verifying size and sha256."""
    root = Path(snap["corpus_root"])
    if master and is_precropped(snap):
        data = precropped_listing(snap)
        if sha256_bytes(data) != snap["master"]["sha256"]:
            raise SnapshotChanged(f"{snap['source_name']}: image listing differs from snapshot {snap['snapshot_id']}")
        return data
    if master:
        path, want = root / snap["master"]["path"], snap["master"]
    else:
        want = snap["files"].get(rel)
        if want is None:
            raise SnapshotChanged(f"{rel}: not part of snapshot {snap['snapshot_id']}")
        path = root / snap["ocr_dir"] / rel
    try:
        data = path.read_bytes()
    except OSError as e:
        raise SnapshotChanged(f"{path}: unreadable ({e})") from e
    if len(data) != want["bytes"] or sha256_bytes(data) != want["sha256"]:
        raise SnapshotChanged(f"{path}: differs from snapshot {snap['snapshot_id']}")
    return data


def verified_master_path(snap: dict) -> Path:
    """Re-verify the master and return its path (renderers read by path)."""
    if is_precropped(snap):
        raise SourceNotReady(f"{snap['source_name']}: pre-cropped source has no master page file")
    read_verified(snap, "", master=True)
    return Path(snap["corpus_root"]) / snap["master"]["path"]


# --------------------------------------------------------------------------- OCR parsing
def parse_ocr(snap: dict) -> dict:
    """Parse result.json into per-page records; raises OcrMalformed on structural problems.

    Page index (0-based) = sequential position of `layoutParsingResults` across `pages`, which
    is the collector's chunk order; it is cross-checked against status.json page counts and
    against the `imgs/pNNNNN/` page numbers of every saved crop.
    """
    if is_precropped(snap):
        return _parse_precropped(snap)
    status = json.loads(read_verified(snap, "status.json").decode("utf-8"))
    try:
        res = json.loads(read_verified(snap, "result.json").decode("utf-8"))
    except ValueError as e:
        raise OcrMalformed(f"result.json is not JSON: {e}") from e
    if not isinstance(res, dict) or not isinstance(res.get("pages"), list):
        raise OcrMalformed("result.json lacks a pages list")
    pages: list[dict] = []
    for grp in res["pages"]:
        lprs = grp.get("layoutParsingResults") if isinstance(grp, dict) else None
        if not isinstance(lprs, list):
            raise OcrMalformed("page group lacks layoutParsingResults")
        for lp in lprs:
            pages.append(_parse_page(lp, len(pages), snap))
    expected = status.get("pages_returned")
    if not isinstance(expected, int) or expected != len(pages) or res.get("page_count") != len(pages):
        raise OcrMalformed(f"page count mismatch: parsed {len(pages)}, status {expected}, "
                           f"result {res.get('page_count')}")
    return {"engine": res.get("model") or OCR_ENGINE, "optional_payload": res.get("optional_payload"),
            "pages": pages}


def _parse_precropped(snap: dict) -> dict:
    """One page per image: the frame is the image, the only block is the whole image, no text."""
    import io

    from PIL import Image
    pages = []
    for i, rel in enumerate(snap["precropped"]["items"]):
        try:
            with Image.open(io.BytesIO(read_verified(snap, rel))) as im:
                w, h = im.size
        except SnapshotChanged:
            raise
        except Exception:         # undecodable image: the page has no crop
            w = h = 0
        box = (0, 0, w, h)
        pages.append({"page_index": i, "frame_w": w, "frame_h": h, "angle": None,
                      "blocks": [{"label": "image", "bbox": list(box), "content": ""}] if w and h else [],
                      "image_crops": {box: rel} if w and h else {}})
    return {"engine": PRECROPPED_ENGINE, "optional_payload": None, "pages": pages}


def _parse_page(lp: dict, index: int, snap: dict) -> dict:
    try:
        pr = lp["prunedResult"]
        fw, fh = int(pr["width"]), int(pr["height"])
        blocks_in = pr["parsing_res_list"]
        images = (lp.get("markdown") or {}).get("images") or {}
    except (KeyError, TypeError, ValueError) as e:
        raise OcrMalformed(f"page {index}: missing prunedResult fields ({e})") from e
    if fw <= 0 or fh <= 0 or not isinstance(blocks_in, list):
        raise OcrMalformed(f"page {index}: bad frame {fw}x{fh}")
    blocks = []
    for b in blocks_in:
        try:
            bb = [int(v) for v in b["block_bbox"]]
            label = str(b["block_label"])
        except (KeyError, TypeError, ValueError) as e:
            raise OcrMalformed(f"page {index}: bad block ({e})") from e
        if len(bb) != 4 or bb[2] <= bb[0] or bb[3] <= bb[1]:
            raise OcrMalformed(f"page {index}: degenerate bbox {bb}")
        content = b.get("block_content")
        blocks.append({"label": label, "bbox": bb, "content": content if isinstance(content, str) else ""})
    crops = {}
    for local in images.values():
        m = _IMG_RE.match(local or "")
        if m is None:
            continue          # the collector failed to save this crop; no evidence from it
        if int(m.group(1)) != index + 1:
            raise OcrMalformed(f"page {index}: crop {local} carries page {int(m.group(1))}")
        if local in snap["files"]:
            crops[tuple(int(m.group(i)) for i in range(2, 6))] = local
    angle = ((pr.get("doc_preprocessor_res") or {}).get("angle"))
    return {"page_index": index, "frame_w": fw, "frame_h": fh, "angle": angle,
            "blocks": blocks, "image_crops": crops}


# --------------------------------------------------------------------------- Paddle crops as items
def frame_text_blocks(ocr_page: dict | None) -> list[dict]:
    """Paddle text blocks with content, in Paddle's preprocessed frame (the frame of its image crops).
    Item blocks (`image`, `chart`) are regions, not text."""
    if ocr_page is None:
        return []
    return [{"bbox": list(b["bbox"]), "text": b["content"], "label": b["label"]}
            for b in ocr_page["blocks"] if b["label"] not in ITEM_BLOCK_LABELS and b["content"].strip()]


def load_crop_gray(snap: dict, rel: str) -> np.ndarray:
    """A saved Paddle crop as 8-bit luma (Pillow "L", the render-1 conversion), re-verified against the
    snapshot. Raises SnapshotChanged, or OcrMalformed when the file does not decode."""
    import io

    from PIL import Image
    data = read_verified(snap, rel)
    try:
        with Image.open(io.BytesIO(data)) as im:
            return np.asarray(im.convert("L"), dtype=np.uint8).copy()
    except Exception as e:  # truncated or corrupt JPEG written by the collector
        raise OcrMalformed(f"{rel}: crop does not decode ({type(e).__name__}: {e})") from e


def source_crop_record(snap: dict, ocr_page: dict, bbox, rel: str) -> dict:
    """Per-item provenance and disclosure of a Paddle crop (owner decision D3, 2026-09-23)."""
    if is_precropped(snap):
        origin = snap["precropped"]["origin"]
        m = _PAGE_RE.search(rel)
        return {"source": PRECROPPED_SOURCE, "snapshot_id": snap["snapshot_id"], "file": rel,
                "file_sha256": snap["files"][rel]["sha256"], "block_bbox": [int(v) for v in bbox],
                "frame": PRECROPPED_FRAME, "frame_wh": [ocr_page["frame_w"], ocr_page["frame_h"]],
                "rotation_deg": None, "modifications": list(PRECROPPED_MODS[origin]),
                "regenerable_from_master": True, "precropped_origin": origin,
                "source_page": int(m.group(1)) if m else None}
    angle = ocr_page.get("angle")
    mods = (["rotated"] if angle not in (0, None) else []) + list(CROP_MODIFICATIONS)
    return {"source": REGION_SOURCE, "snapshot_id": snap["snapshot_id"], "file": rel,
            "file_sha256": snap["files"][rel]["sha256"], "block_bbox": [int(v) for v in bbox],
            "frame": FRAME, "frame_wh": [ocr_page["frame_w"], ocr_page["frame_h"]], "rotation_deg": angle,
            "modifications": mods, "regenerable_from_master": False}


# --------------------------------------------------------------------------- script detection
_SCRIPTS = (("Han", ("CJK UNIFIED", "CJK COMPATIBILITY IDEOGRAPH")), ("Kana", ("HIRAGANA", "KATAKANA")),
            ("Hangul", ("HANGUL",)), ("Greek", ("GREEK",)), ("Cyrillic", ("CYRILLIC",)),
            ("Arabic", ("ARABIC",)), ("Hebrew", ("HEBREW",)), ("Latin", ("LATIN",)))


def detect_script(text: str) -> str | None:
    """Dominant Unicode script of OCR text content (frame-independent). None if no letters."""
    counts: dict[str, int] = {}
    for ch in text:
        if not ch.isalpha():
            continue
        name = unicodedata.name(ch, "")
        script = next((s for s, keys in _SCRIPTS if any(k in name for k in keys)), "Other")
        counts[script] = counts.get(script, 0) + 1
    return max(sorted(counts), key=lambda s: counts[s]) if counts else None
