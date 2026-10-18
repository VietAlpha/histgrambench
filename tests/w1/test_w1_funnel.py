"""§6.6 funnel rows and §5.2 item record assembly (content ids, *_sha256 siblings, shared §9.7 canonicalizer)."""
import hashlib
import sys

import cv2
import pytest
from pydantic import ValidationError

from hmb.acquisition import funnel as F
from hmb.hashing import item_id
from hmb.schemas import ItemRecord
from hmb.store import Layout, iter_jsonl
from w1_helpers import blank, draw_diagram, source_crop


def _row(**kw):
    r = {"proposal_id": "prop_1", "book_id": "bk_0123456789ab", "page_index": 3, "box_xyxy_page_px": [10, 20, 110, 140],
         "box_source": "paddle_layout@1.6", "box_frame": "paddle_preprocessed",
         "source_crop": source_crop([10, 20, 110, 140]),
         "gates": [{"gate_id": "G2", "version": "g2-1", "passed": False, "measured_value": None, "threshold": None}],
         "exclusion_reasons": ["box_too_small"], "img_stats": {"ink_frac": 0.1},
         "pipeline_version": F.PIPELINE_VERSION, "config_hash": "c"}
    r.update(kw)
    return r


def test_finalize_and_append_validates(tmp_path):
    lay = Layout(tmp_path)
    r = F.finalize(_row())
    assert r["terminal_gate"] == "G2" and r["outcome"] == "excluded"
    F.append_row(lay, r)
    F.append_row(lay, dict(r, terminal_gate="G2", caption_text="later row"))
    assert F.latest_rows(lay)["prop_1"]["caption_text"] == "later row"
    assert len(list(iter_jsonl(lay.funnel))) == 2                     # append-only: both rows kept
    with pytest.raises(ValidationError):
        F.append_row(lay, dict(r, exclusion_reasons=["free text reason"]))   # closed vocabulary
    with pytest.raises(ValidationError):
        F.append_row(lay, dict(r, unknown_field=1))


def test_proposal_ids_are_content_derived():
    a = F.proposal_id("bk_x", 1, [1, 2, 3, 4], "paddle_layout@1.6", None)
    assert a == F.proposal_id("bk_x", 1, (1, 2, 3, 4), "paddle_layout@1.6", None)
    assert a != F.proposal_id("bk_x", 1, [1, 2, 3, 5], "paddle_layout@1.6", None)
    assert a != F.proposal_id("bk_x", 1, [1, 2, 3, 4], "paddle_layout@1.6", "prop_parent")


def test_assemble_item_with_shared_canonicalizer(tmp_path, th):
    lay = Layout(tmp_path)
    crop = cv2.imencode(".png", draw_diagram(blank(120, 100), 50, 60, 40))[1].tobytes()
    page = lay.page_image("bk_0123456789ab", 3)
    page.parent.mkdir(parents=True)
    page.write_bytes(b"page")
    row = F.finalize(_row(gates=[], exclusion_reasons=[]))
    rec = F.assemble_item(lay, row, page_png_path=page, page_png_sha256=hashlib.sha256(b"page").hexdigest(),
                          crop_png=crop, context={"context_text": "Fig. 3.", "context_bbox": [0, 0, 200, 200],
                                                  "context_char_count": 7, "context_block_count": 0},
                          page_meta={"render_dpi": 400, "renderer_version": "pymupdf x"}, ocr_engine_version="none",
                          ocr_status="unavailable", work_id="wk_1", th=th, dataset_version="v1.0",
                          rights_manifest_sha256="ab" * 32)
    iid = item_id("bk_0123456789ab", 3, [10, 20, 110, 140])
    assert rec["source"]["rights_manifest_sha256"] == "ab" * 32
    assert rec["id"] == iid
    assert rec["assets"]["crop_image"] == f"crops/{iid}.png"
    assert rec["assets"]["crop_image_sha256"] == hashlib.sha256(crop).hexdigest()
    canon = lay.crop_canonical(iid).read_bytes()
    assert rec["assets"]["crop_canonical_sha256"] == hashlib.sha256(canon).hexdigest()
    assert canon[:8] == b"\x89PNG\r\n\x1a\n"
    assert rec["extraction"]["source_crop"]["frame"] == "paddle_preprocessed" and "context_image" not in rec["assets"]
    assert rec["automatic_metadata"]["image_stats"]["measured_on"] == "paddle_crop"
    ItemRecord.model_validate(rec)


def test_canonicalizer_missing_fails_loudly(monkeypatch, th):
    monkeypatch.setitem(sys.modules, "hmb.submission.render", None)
    monkeypatch.setitem(sys.modules, "hmb.submission", None)
    with pytest.raises(F.CanonicalizerUnavailable):
        F.canonical_crop(b"png", th)
