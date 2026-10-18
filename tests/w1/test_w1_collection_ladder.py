"""Collection adapter (snapshot boundary, OCR parsing, Paddle crops as items) and the CPU ladder end
to end, on a synthetic corpus written to tmp_path (never the real corpus)."""
import json

import cv2
import fitz
import numpy as np
import pytest

from hmb import rights
from hmb.acquisition import collection as col
from hmb.acquisition import context as C
from hmb.acquisition.ladder import process_source, summarize
from hmb.schemas import FunnelRow
from hmb.store import Layout, iter_jsonl
from w1_helpers import blank, draw_diagram, draw_text_rows

FW, FH = 600, 800           # OCR frame = 2 x PDF points (as the collector's PaddleOCR output)
DIAG = (300, 450, 150)      # centre x, y, radius in frame pixels
PBOX = [145, 295, 455, 605]  # Paddle image block around the diagram (frame pixels)


def _page_image(figure_only=False):
    img = blank(FH, FW)
    if not figure_only:
        draw_text_rows(img, 40, 40, 6, 44, glyph_h=7, glyph_w=4, gap=2, row_gap=9)
    draw_diagram(img, *DIAG, thickness=2)
    return img


def make_corpus(root, name="Book (IA test).pdf", state="done", angle=0, n_pages_status=None, kind="image",
                figure_only=False):
    """`figure_only`: a plate with the figure alone (no prose, caption or body text on the page)."""
    img = _page_image(figure_only)
    (root / "wikimedia-2026-09-21").mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    pg = doc.new_page(width=FW / 2, height=FH / 2)
    pg.insert_image(pg.rect, stream=cv2.imencode(".png", img)[1].tobytes())
    master = root / "wikimedia-2026-09-21" / name
    doc.save(master)
    out = root / "corpus_output" / name
    (out / "raw").mkdir(parents=True, exist_ok=True)
    # Paddle's saved crop = our own render at frame scale (the fixture frame has no dewarp)
    with fitz.open(master) as d:
        pix = d[0].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csGRAY, alpha=False)
        frame = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.stride)[:, :pix.width]
    x0, y0, x1, y1 = PBOX
    crop = frame[y0:y1, x0:x1]
    local = f"imgs/p00001/img_in_{kind}_box_{x0}_{y0}_{x1}_{y1}.jpg"
    (out / "imgs" / "p00001").mkdir(parents=True, exist_ok=True)
    (out / local).write_bytes(cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    lp = {"prunedResult": {"width": FW, "height": FH, "doc_preprocessor_res": {"angle": angle},
                           "parsing_res_list": ([] if figure_only else [
                               {"block_label": "text", "block_bbox": [40, 40, 300, 110], "block_content": "Some prose"}])
                           + [{"block_label": kind, "block_bbox": PBOX, "block_content": ""}]},
          "markdown": {"text": "", "images": {local.replace("p00001/", ""): local}}}
    n = 1 if n_pages_status is None else n_pages_status
    (out / "result.json").write_text(json.dumps({"model": "PaddleOCR-VL-1.6", "page_count": 1, "pages_returned": 1,
                                                 "pages": [{"layoutParsingResults": [lp]}]}), encoding="utf-8")
    (out / "raw" / "chunk_0000.jsonl").write_text(json.dumps({"result": {"layoutParsingResults": [lp]}}) + "\n")
    (out / "status.json").write_text(json.dumps({"source": name, "kind": "pdf", "state": state, "pages": 1,
                                                 "pages_returned": n}), encoding="utf-8")
    (root / "metadata").mkdir(exist_ok=True)
    (root / "metadata" / "file_years.json").write_text(json.dumps({name: {"year": 1905, "basis": "ia"}}))
    return master


def _manifest_for(snap, approver="fixture:test"):
    from test_w1_rights import valid_record
    rec = valid_record(approver=approver)
    probe = rights.probe_master(col.verified_master_path(snap))
    rec["book_id"] = snap["book_id"]
    rec["acquisition"].update(source_file_sha256=probe["source_file_sha256"],
                              source_file_bytes=probe["source_file_bytes"], page_count=probe["page_count"])
    rec["manifest_sha256"] = rights.manifest_sha256(rec)
    return rights.apply_us_rule(rec)


# --------------------------------------------------------------------------- snapshot boundary
def test_snapshot_refuses_pending_and_detects_changes(tmp_path):
    make_corpus(tmp_path, name="P.pdf", state="pending")
    with pytest.raises(col.SourceNotReady):
        col.snapshot_source(tmp_path, "P.pdf")
    make_corpus(tmp_path, name="D.pdf")
    snap = col.snapshot_source(tmp_path, "D.pdf")
    assert set(snap["files"]) >= {"status.json", "result.json", "raw/chunk_0000.jsonl"}
    assert snap["book_id"].startswith("bk_") and snap["year_evidence"]["record"]["year"] == 1905
    assert col.read_verified(snap, "result.json")
    p = tmp_path / "corpus_output" / "D.pdf" / "result.json"
    p.write_bytes(p.read_bytes() + b" ")                     # the collector rewrote the file
    with pytest.raises(col.SnapshotChanged):
        col.read_verified(snap, "result.json")
    with pytest.raises(col.SnapshotChanged):
        col.read_verified(snap, "not/in/snapshot.json")
    with pytest.raises(col.SourceNotReady):
        col.snapshot_source(tmp_path, "../escape")


def test_parse_ocr_and_malformed_inputs(tmp_path):
    make_corpus(tmp_path, name="D.pdf")
    ocr = col.parse_ocr(col.snapshot_source(tmp_path, "D.pdf"))
    pg = ocr["pages"][0]
    assert (pg["frame_w"], pg["frame_h"], pg["angle"]) == (FW, FH, 0)
    assert tuple(PBOX) in pg["image_crops"]
    make_corpus(tmp_path, name="C.pdf", kind="chart")          # D13: chart crops are items too
    pc = col.parse_ocr(col.snapshot_source(tmp_path, "C.pdf"))["pages"][0]
    assert pc["image_crops"] == {tuple(PBOX): f"imgs/p00001/img_in_chart_box_{'_'.join(map(str, PBOX))}.jpg"}
    assert [b["label"] for b in col.frame_text_blocks(pc)] == ["text"]
    make_corpus(tmp_path, name="M.pdf", n_pages_status=2)      # status claims 2 pages, result has 1
    with pytest.raises(col.OcrMalformed):
        col.parse_ocr(col.snapshot_source(tmp_path, "M.pdf"))
    make_corpus(tmp_path, name="J.pdf")
    (tmp_path / "corpus_output" / "J.pdf" / "result.json").write_text("{not json")
    with pytest.raises(col.OcrMalformed):
        col.parse_ocr(col.snapshot_source(tmp_path, "J.pdf"))
    make_corpus(tmp_path, name="B.pdf")
    rp = tmp_path / "corpus_output" / "B.pdf" / "result.json"
    d = json.loads(rp.read_text())
    d["pages"][0]["layoutParsingResults"][0]["prunedResult"]["parsing_res_list"][0]["block_bbox"] = [5, 5, 1, 1]
    rp.write_text(json.dumps(d))
    with pytest.raises(col.OcrMalformed):
        col.parse_ocr(col.snapshot_source(tmp_path, "B.pdf"))


def test_paddle_crop_provenance_and_text_blocks(tmp_path):
    make_corpus(tmp_path, name="G.pdf")
    snap = col.snapshot_source(tmp_path, "G.pdf")
    page = col.parse_ocr(snap)["pages"][0]
    (bbox, rel), = page["image_crops"].items()
    crop = col.load_crop_gray(snap, rel)
    assert crop.shape == (bbox[3] - bbox[1], bbox[2] - bbox[0]) and crop.dtype == np.uint8
    src = col.source_crop_record(snap, page, bbox, rel)
    assert src["frame"] == "paddle_preprocessed" and src["block_bbox"] == list(PBOX)
    assert src["file_sha256"] == snap["files"][rel]["sha256"] and src["regenerable_from_master"] is False
    assert "rotated" not in src["modifications"] and "dewarped" in src["modifications"]
    assert col.frame_text_blocks(page) == [{"bbox": [40, 40, 300, 110], "text": "Some prose", "label": "text"}]
    make_corpus(tmp_path, name="R.pdf", angle=270)
    snap_r = col.snapshot_source(tmp_path, "R.pdf")
    page_r = col.parse_ocr(snap_r)["pages"][0]
    (bb, rr), = page_r["image_crops"].items()
    assert col.source_crop_record(snap_r, page_r, bb, rr)["modifications"][0] == "rotated"


def test_script_detection():
    assert col.detect_script("The circle ABC") == "Latin"
    assert col.detect_script("圓之切線") == "Han"
    assert col.detect_script("Рис. 5 окружность") == "Cyrillic"
    assert col.detect_script("12 + 3") is None


# --------------------------------------------------------------------------- context rules
def test_figure_numbers_and_labels():
    assert C.figure_number("Fig. 46.") == "46"
    assert C.figure_number("第三圖") == "三"
    assert C.figure_number("Рис. 12") == "12"
    assert C.figure_number("the figure shows") is None
    for tok in ("A", "B'", "P_1", "45°", "α", "AB", "甲"):
        assert C.is_math_label(tok), tok
    for tok in ("circle", "Theorem", "ABCD"):
        assert not C.is_math_label(tok), tok


def test_context_window_rules(th):
    crop = [100, 400, 300, 600]          # height 200 -> window 100..900
    units = {5: [{"bbox": [100, 50, 300, 80], "text": "far above"},
                 {"bbox": [100, 250, 300, 280], "text": "near above"},
                 {"bbox": [100, 650, 300, 680], "text": "near below"},
                 {"bbox": [120, 450, 200, 470], "text": "inside the crop"}],
             6: [{"bbox": [0, 0, 10, 10], "text": "as shown in Fig. 7 the tangent"},
                 {"bbox": [0, 20, 10, 30], "text": "unrelated text"}]}
    ctx = C.context_window(crop, 5, units, "Fig. 7.", "7", th)
    assert ctx["context_text"].split("\n") == ["Fig. 7.", "near above", "near below", "as shown in Fig. 7 the tangent"]
    assert ctx["context_block_count"] == 3 and ctx["context_bbox"][1] == 250
    empty = C.context_window(crop, 5, None, None, None, th)
    assert empty["context_char_count"] == 0 and empty["context_bbox"] == crop


def test_context_cap_truncates_at_block_boundaries(th):
    crop = [0, 1000, 100, 1100]
    units = {0: [{"bbox": [0, 900 + i, 10, 901 + i], "text": "x" * 500} for i in range(3)]}
    ctx = C.context_window(crop, 0, units, None, None, th)
    assert ctx["context_block_count"] == 2 and ctx["context_char_count"] == 1001   # 2 x 500 + newline


def test_panel_split_bisects_per_level(th):
    from hmb.acquisition import gates as G
    page = blank(400, 1300)
    for k in range(3):
        draw_diagram(page, 200 + 450 * k, 200, 120)
    out = G.g7b(page, [40, 60, 1260, 340], None, th, 300, None)
    assert out["action"] == "split" and len(out["panels"]) == 3        # level 1: 1|2, level 2: 2 -> 1|1


def test_multi_panel_unresolved(th):
    from hmb.acquisition import gates as G
    th.params["g7b_max_depth"] = dict(th.params["g7b_max_depth"], value=3)   # deeper recursion allowed
    page = blank(400, 2400)
    for k in range(6):
        draw_diagram(page, 180 + 400 * k, 200, 120)          # gaps of ~155 px >= 0.04 x 2380
    out = G.g7b(page, [10, 60, 2390, 340], None, th, 300, None)
    assert out["action"] == "exclude" and out["reasons"] == ["multi_panel_unresolved"]


# --------------------------------------------------------------------------- ladder end to end
def test_ladder_fixture_run_paddle_crop_to_g7d(tmp_path, th):
    make_corpus(tmp_path / "corpus", name="E.pdf")
    snap = col.snapshot_source(tmp_path / "corpus", "E.pdf")
    lay = Layout(tmp_path / "root")
    summ = process_source(lay, snap, _manifest_for(snap), th, profile="fixture")
    s = summarize(summ)
    assert s["r3_passed"] and s["pages_by_terminal_gate"] == {"passed": 1}
    assert s["ocr_status"] == {"ok": 1} and s["crops_unreadable"] == 0
    assert len(summ["rows"]) == 1
    row = summ["rows"][0]
    assert row["box_source"] == "paddle_layout@1.6" and row["box_frame"] == "paddle_preprocessed"
    assert row["source_crop"]["block_bbox"] == list(PBOX) and row["source_crop"]["snapshot_id"] == snap["snapshot_id"]
    assert [g["gate_id"] for g in row["gates"]] == ["G2", "G3", "G4", "G5", "G6", "G7a", "G7b", "G7c", "G7d"]
    assert PBOX[0] <= row["box_xyxy_page_px"][0] and row["box_xyxy_page_px"][2] <= PBOX[2]   # Paddle frame
    assert row["terminal_gate"] == "passed" and row["exclusion_reasons"] == [] and row["outcome"] is None
    assert row["phash64"] and row["phash_cluster_id"] == "singleton"
    assert set(row["img_stats"]) >= {"ink_frac", "endpoint_density", "eff_res_xheight_px"}
    stored = list(iter_jsonl(lay.funnel))
    assert len(stored) == len(summ["rows"])
    for r in stored:
        FunnelRow.model_validate(r)
    assert lay.page_image(snap["book_id"], 0).exists()


def test_ladder_refuses_to_render_without_rights(tmp_path, th):
    make_corpus(tmp_path / "corpus", name="N.pdf")
    snap = col.snapshot_source(tmp_path / "corpus", "N.pdf")
    lay = Layout(tmp_path / "root")
    summ = process_source(lay, snap, None, th, profile="production")
    assert not summ["r3"]["passed"] and summ["rows"] == [] and summ["pages"] == []
    assert not lay.page_image(snap["book_id"], 0).exists()
    fixture_man = _manifest_for(snap)                                   # fixture approver in production
    summ2 = process_source(lay, snap, fixture_man, th, profile="production")
    assert "fixture_approver_in_production" in summ2["r3"]["failures"]
    pages = list(iter_jsonl(lay.p("manifests", "funnel_pages.jsonl")))
    assert all(p["exclusion_reasons"] == ["rights_unresolved"] for p in pages)


def test_ladder_skips_an_undecodable_crop(tmp_path, th):
    make_corpus(tmp_path / "corpus", name="U.pdf")
    bad = next((tmp_path / "corpus" / "corpus_output" / "U.pdf" / "imgs" / "p00001").glob("*.jpg"))
    bad.write_bytes(b"not a jpeg")                                   # the collector wrote a broken file
    snap = col.snapshot_source(tmp_path / "corpus", "U.pdf")
    summ = process_source(Layout(tmp_path / "root"), snap, _manifest_for(snap), th, profile="fixture")
    assert summ["rows"] == [] and summarize(summ)["crops_unreadable"] == 1
    assert "does not decode" in summ["pages"][0]["crops_unreadable"][0]["error"]


def test_g0a_refuses_manifest_for_a_different_file(tmp_path, th):
    make_corpus(tmp_path / "corpus", name="W.pdf")
    snap = col.snapshot_source(tmp_path / "corpus", "W.pdf")
    man = _manifest_for(snap)
    man["acquisition"]["source_file_sha256"] = man["book_id"][3:] + "0" * 52     # same book_id prefix, other bytes
    man["manifest_sha256"] = rights.manifest_sha256(man)
    summ = process_source(Layout(tmp_path / "root"), snap, man, th, profile="fixture")
    assert summ["rows"] == [] and summ["pages"][0]["exclusion_reasons"] == ["rights_unresolved"]


def test_cluster_ids_stable_and_updates_appended(tmp_path, th):
    from hmb.acquisition import ladder as L
    from hmb.acquisition.funnel import append_row, latest_rows
    from hmb.acquisition.phash import cluster_id
    lay = Layout(tmp_path)
    base = {"book_id": "bk_x", "page_index": 0, "pipeline_version": "p", "config_hash": "c", "gates": []}
    first = [dict(base, proposal_id="prop_b", phash64="0" * 16), dict(base, proposal_id="prop_c", phash64="0" * 16)]
    assert L._cluster(lay, first, th) == []
    for r in first:
        append_row(lay, r)
    cid = first[0]["phash_cluster_id"]
    assert cid == cluster_id(["prop_b", "prop_c"]) == cluster_id(["prop_b"])
    later = [dict(base, proposal_id="prop_d", phash64="0" * 16)]          # sorts after prop_b: id unchanged
    upd = L._cluster(lay, later, th)
    assert later[0]["phash_cluster_id"] == cid
    assert all(u["phash_cluster_id"] == cid and u["cluster_size"] == 3 for u in upd)   # only sizes change
    for r in later + upd:
        append_row(lay, r)
    smaller = [dict(base, proposal_id="prop_a", phash64="0" * 16)]       # new smallest member -> id changes
    updates = L._cluster(lay, smaller, th)
    assert {u["proposal_id"] for u in updates} == {"prop_b", "prop_c", "prop_d"}
    assert all(u["phash_cluster_id"] == cluster_id(["prop_a"]) for u in updates)
    n_before = len(latest_rows(lay))
    for r in updates:
        append_row(lay, r)
    assert len(latest_rows(lay)) == n_before                              # appended, never rewritten


# --------------------------------------------------------------------------- figure-only pages (C21, 2026-09-25)
class _Gatekeeper:
    """Explicit fake G8b gatekeeper with a scripted verdict and diagram type."""

    def __init__(self, accept: bool):
        self.accept, self.calls = accept, 0

    def send(self, req, images):
        self.calls += 1
        v = "true" if self.accept else "false"
        return {"text": f'{{"target_type": {v}, "readable": true, "tikz_representable": true, '
                        '"diagram_type": "geometry"}', "input_tokens": 10, "output_tokens": 5,
                "model_version_string": "fake"}


def _figure_only_root(tmp_path, th, name):
    from test_w1_scaffold4 import write_manifests
    make_corpus(tmp_path / "corpus", name=name, figure_only=True)
    snap = col.snapshot_source(tmp_path / "corpus", name)
    lay = Layout(tmp_path / "root")
    col.store_snapshot(lay, snap)
    man = _manifest_for(snap)
    write_manifests(lay, [man])
    return lay, process_source(lay, snap, man, th, profile="fixture")


def test_figure_only_page_reaches_the_gatekeeper(tmp_path, th, ledger):
    from hmb.acquisition.drivers import run_g8
    from hmb.schemas import ItemRecord, ModelSpec
    lay, summ = _figure_only_root(tmp_path, th, "F.pdf")
    page = summ["pages"][0]
    g0b = next(g for g in page["gates"] if g["gate_id"] == "G0b")
    assert page["figure_only"] is True and g0b["passed"] and g0b["measured_value"]["figure_only"] is True
    assert page["exclusion_reasons"] == [] and len(summ["rows"]) == 1
    assert summ["rows"][0]["terminal_gate"] == "passed"               # same CPU gates as any other crop
    gk = _Gatekeeper(accept=True)
    roster = {"verifier": [ModelSpec(model_id="gk", provider="fake", family="B", role="verifier",
                                     unit_price_id="p", effort="high")]}
    out = run_g8(lay, th, roster, ledger=ledger, allow_live=True, provider_override=gk)
    assert gk.calls == 1 and out["survivors"] == 1
    rec = ItemRecord.model_validate(json.loads(lay.p("data", "candidates", f"{out['candidates'][0]}.json")
                                               .read_text(encoding="utf-8")))
    am = rec.automatic_metadata
    assert am["figure_only_page"] is True
    assert am["diagram_type"]["label"] == "geometry" and am["diagram_type"]["gate_version"] == "g8b-4"
    assert am["diagram_type"]["labeler_model_id"] == "gk" and am["diagram_type"]["status"] == "predicted"


def test_unusable_figure_only_candidate_is_rejected_by_the_gatekeeper(tmp_path, th, ledger):
    from hmb.acquisition.drivers import run_g8
    from hmb.acquisition.funnel import latest_rows
    from hmb.schemas import ModelSpec
    from hmb.sampling import funnel_report
    lay, summ = _figure_only_root(tmp_path, th, "G.pdf")
    assert summ["pages"][0]["figure_only"] is True and summ["rows"][0]["terminal_gate"] == "passed"
    gk = _Gatekeeper(accept=False)
    roster = {"verifier": [ModelSpec(model_id="gk", provider="fake", family="B", role="verifier",
                                     unit_price_id="p", effort="high")]}
    out = run_g8(lay, th, roster, ledger=ledger, allow_live=True, provider_override=gk)
    assert gk.calls == 1 and out["survivors"] == 0 and out["rejected"] == 1
    row = next(iter(latest_rows(lay).values()))
    assert row["exclusion_reasons"] == ["verifier_reject"] and row["diagram_type"] is None
    rep = funnel_report(lay)["figure_only_pages"]
    assert rep["n_pages"] == 1 and rep["n_crops_reaching_g8b"] == 1 and rep["n_crops_passing_g8b"] == 0


def test_acq_run_in_prefreeze_dev_declares_a_development_book(tmp_path):
    """acq-run --thresholds prefreeze:DEV (after MS1, before the level freeze) runs the real ladder on plan starting
    values, declares the book a development book and logs the run's provisional levels; an unknown context
    (the deleted CAL context) is refused, and the context is refused before the MS1 approval."""
    from hmb.cli import main
    from test_w1_scaffold4 import write_manifests
    make_corpus(tmp_path / "corpus", name="D.pdf")
    snap = col.snapshot_source(tmp_path / "corpus", "D.pdf")
    lay = Layout(tmp_path / "root")
    write_manifests(lay, [_manifest_for(snap, approver="owner:alice")])
    run = ["--root", str(lay.root), "acq-run", "D.pdf", "--corpus", str(tmp_path / "corpus"), "--thresholds"]
    assert main(run + ["prefreeze:DEV"]) == 2                              # no production thresholds yet
    assert main(["--root", str(lay.root), "thresholds-init", "--profile", "production"]) == 0
    assert main(run + ["prefreeze:DEV"]) == 2                              # no MS1 approval yet
    assert main(["--root", str(lay.root), "form-freeze"]) == 0
    assert main(["--root", str(lay.root), "form-freeze", "--record-owner-approval", "--actor", "owner:test"]) == 0
    assert main(run + ["prefreeze:CAL"]) == 2                              # the CAL context no longer exists
    assert main(run + ["prefreeze:DEV"]) == 0
    from hmb.acquisition.drivers import development_books
    assert development_books(lay) == {snap["book_id"]}
    rows = list(iter_jsonl(lay.funnel))
    assert rows and rows[0]["terminal_gate"] == "passed"
    rec = list(iter_jsonl(lay.p("manifests", "prefreeze_runs.jsonl")))[-1]
    assert rec["command"] == "acq-run D.pdf" and rec["partition"] == "development" and rec["levels"] == "provisional"
    assert rec["params_read"]["g0b_min_xheight_px"] == 12 and "manifests/funnel_rows.jsonl" in rec["outputs"]


# --------------------------------------------------------------------------- pre-cropped sources (2026-09-29)
def test_precropped_source_snapshot_r3_ladder_and_gatekeeper(tmp_path, th, ledger):
    """A directory of figure images only (no status.json, result.json or master): each image is one page and one
    item; the image listing is the master (book_id), G2 skips the page-area test, and a G8b survivor is a valid item."""
    from hmb.acquisition.drivers import run_g8
    from hmb.schemas import ItemRecord, ModelSpec
    from test_w1_scaffold4 import write_manifests
    name = "Author - Some Geometry, 1700 (Gutenberg 1)"
    out = tmp_path / "corpus" / "corpus_output" / name / "imgs"
    out.mkdir(parents=True)
    img = blank(320, 320)
    draw_diagram(img, 160, 160, 120, thickness=2)
    (out / "fig1.png").write_bytes(cv2.imencode(".png", img)[1].tobytes())
    (out.parent / "source.json").write_text('{"images": 1}', encoding="utf-8")
    snap = col.snapshot_source(tmp_path / "corpus", name)
    assert col.is_precropped(snap) and snap["precropped"] == {"origin": "html_image", "items": ["imgs/fig1.png"]}
    probe = rights.probe_precropped(snap)
    assert probe["page_count"] == 1 and probe["source_codec"] == "png"
    assert snap["book_id"] == rights.make_book_id(probe["source_file_sha256"])
    lay = Layout(tmp_path / "root")
    col.store_snapshot(lay, snap)
    from test_w1_rights import valid_record
    man = valid_record(approver="fixture:test")
    man["book_id"] = snap["book_id"]
    man["acquisition"].update(source_file_sha256=probe["source_file_sha256"], source_file_bytes=probe["source_file_bytes"],
                              page_count=1, source_dpi_nominal=300)
    man["manifest_sha256"] = rights.manifest_sha256(man)
    man = rights.apply_us_rule(man)
    write_manifests(lay, [man])
    summ = process_source(lay, snap, man, th, profile="fixture")
    assert [g["passed"] for g in summ["pages"][0]["gates"]] == [True] * 4 and len(summ["rows"]) == 1
    row = summ["rows"][0]
    g2 = next(g for g in row["gates"] if g["gate_id"] == "G2")
    assert row["terminal_gate"] == "passed" and g2["measured_value"]["area_frac"] is None
    assert row["box_frame"] == "precropped_image" and row["source_crop"]["regenerable_from_master"] is True
    FunnelRow.model_validate({k: v for k, v in row.items() if k in FunnelRow.model_fields})
    gk = _Gatekeeper(accept=True)
    roster = {"verifier": [ModelSpec(model_id="gk", provider="fake", family="B", role="verifier",
                                     unit_price_id="p", effort="high")]}
    res = run_g8(lay, th, roster, ledger=ledger, allow_live=True, provider_override=gk)
    assert gk.calls == 1 and res["survivors"] == 1
    rec = ItemRecord.model_validate(json.loads(lay.p("data", "candidates", f"{res['candidates'][0]}.json")
                                               .read_text(encoding="utf-8")))
    assert rec.extraction.source_crop.frame == "precropped_image" and rec.extraction.crop_bbox == [0, 0, 320, 320]
