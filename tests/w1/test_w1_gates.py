"""§6.1/§6.3/§6.4 gates on synthetic crops with known answers; G8 ordering and parsing with FAKE providers."""
import json
import re

import cv2
import numpy as np
import pytest

from hmb.vocab import DiagramType

from hmb.acquisition import gates as G
from hmb.acquisition.imgstats import analyze, sauvola_ink, sauvola_window
from hmb.schemas import ModelSpec
from hmb.store import Layout
from w1_helpers import blank, draw_diagram, draw_text_rows


def an_of(img, th):
    return analyze(img, sauvola_ink(img, sauvola_window(th, 300), th.value("sauvola_k")))


def test_record_shape(th):
    rec, reasons = G.g3(0.1, th)
    assert set(rec) == {"gate_id", "version", "passed", "measured_value", "threshold"}
    assert rec["passed"] and reasons == []


def test_page_gates(th):
    rec, rs = G.g0a({"passed": False, "failures": ["no_manifest_record"]}, 10, None, "a" * 64, "a" * 64)
    assert not rec["passed"] and rs == ["rights_unresolved"]
    rec, rs = G.g0a({"passed": True, "failures": []}, 10, 11, "a" * 64, "a" * 64)
    assert not rec["passed"] and rs == ["page_render_fail"]
    assert G.g0a({"passed": True, "failures": []}, 10, 10, "a" * 64, "a" * 64)[0]["passed"]
    rec, rs = G.g0a({"passed": True, "failures": []}, 10, 10, "a" * 64, "b" * 64)   # manifest is another file
    assert not rec["passed"] and rs == ["rights_unresolved"]
    assert not G.g0a({"passed": True, "failures": []}, 10, 10, "a" * 64, None)[0]["passed"]
    assert G.g0b(True, 12.0, th)[0]["passed"] and not G.g0b(True, 11.0, th)[0]["passed"]
    fo, rs0 = G.g0b(True, None, th, figure_only=True)  # figure-only page: recorded, not excluded (C21)
    assert fo["passed"] and rs0 == [] and fo["measured_value"]["figure_only"] is True
    assert G.g0b(True, 3.0, th, figure_only=True)[0]["passed"] and not G.g0b(False, None, th, figure_only=True)[0]["passed"]
    rec, rs = G.g0b(True, None, th)                  # not measurable: never a pass
    assert not rec["passed"] and rs == ["page_low_resolution"]
    assert G.g0c(0.0021, th)[0]["passed"] and G.g0c(0.002, th)[1] == ["page_blank"]
    rec, rs = G.g0d("unavailable", None, None, None)
    assert rec["passed"] and rs == []


def test_g2_geometry_and_unknown_dpi(th):
    # 400 dpi: 0.7 cm = 110.2 px
    assert G.g2([0, 0, 120, 200], (2000, 1500), 400.0, th)[0]["passed"]
    assert G.g2([0, 0, 100, 200], (2000, 1500), 400.0, th)[1] == ["box_too_small"]
    assert "box_too_large" in G.g2([0, 0, 1400, 1900], (2000, 1500), 400.0, th)[1]
    assert "aspect_extreme" in G.g2([0, 0, 130, 900], (2000, 1500), 400.0, th)[1]
    rec, rs = G.g2([0, 0, 500, 500], (2000, 1500), None, th)
    assert not rec["passed"] and rs == ["physical_size_unknown"]


def test_g3_bounds(th):
    assert G.g3(0.004, th)[1] == ["ink_too_low"]
    assert G.g3(0.46, th)[1] == ["ink_too_high"]
    assert G.g3(None, th)[1] == ["ink_too_low"]


def test_g4_solid_block(th):
    img = blank(100, 100)
    img[2:98, 2:98] = 0                      # one component covering 92% of the crop
    rec, rs = G.g4(an_of(img, th), th)
    assert rs[0] == "solid_block"
    assert rec["measured_value"]["largest_component_share"] == pytest.approx(96 * 96 / 10000)
    line = draw_diagram(blank(300, 300), 150, 150, 100)
    assert G.g4(an_of(line, th), th)[0]["passed"]


def test_g5_line_art_vs_photograph(th):
    line = draw_diagram(blank(300, 300), 150, 150, 100)
    assert G.g5(line, th)[0]["passed"]
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[0:300, 0:300]
    photo = np.clip(128 + 60 * np.sin(xx / 17.0) * np.cos(yy / 23.0) + rng.normal(0, 12, (300, 300)), 0, 255)
    rec, rs = G.g5(photo.astype(np.uint8), th)
    assert not rec["passed"] and rs == ["photo_grayscale"]
    assert G.bimodality_ratio(blank(10, 10), 0.1) is None


def test_g6_long_strokes_vs_specks(th):
    line = draw_diagram(blank(300, 300), 150, 150, 100)
    assert G.g6(an_of(line, th), th)[0]["passed"]
    specks = blank(300, 300)
    for y in range(20, 280, 25):
        for x in range(20, 280, 25):
            specks[y:y + 4, x:x + 4] = 0
    rec, rs = G.g6(an_of(specks, th), th)
    assert not rec["passed"] and rs == ["no_long_strokes"]


def test_g7a_records_border_measures_and_never_excludes(th):
    # D10: a crop cut through its diagram fires the border test, which is recorded, not an exclusion
    page = blank(800, 800)
    draw_diagram(page, 400, 400, 150)
    rec, rs, border = G.g7a(page, [260, 260, 540, 480], th, 300)    # cuts through the bottom of the circle
    assert rec["passed"] and rs == [] and rec["threshold"]["action"] == "records only"
    assert border["touching_cc"] and border["offending"] and rec["measured_value"]["offending"] == border["offending"]
    rec, rs, border = G.g7a(page, [200, 200, 600, 600], th, 300)    # whole figure with a margin
    assert rec["passed"] and rs == [] and not border["offending"] and not border["touching_cc"]


def test_g7b_panel_split(th):
    page = blank(600, 1000)
    draw_diagram(page, 250, 300, 150)
    draw_diagram(page, 750, 300, 150)
    out = G.g7b(page, [80, 130, 920, 470], None, th, 300, None)
    assert out["action"] == "split" and len(out["panels"]) == 2
    assert out["reasons"] == ["split_into_panels"]
    left, right = sorted(out["panels"])
    assert left[2] < 500 < right[0]


def test_g7b_geometric_caption_strip_and_prose(th):
    page = blank(800, 800)
    draw_diagram(page, 400, 300, 180)
    draw_text_rows(page, 300, 560, 1, 10, glyph_h=14)          # caption-like line below the figure
    out = G.g7b(page, [200, 100, 600, 580], 14.0, th, 300, None)
    assert out["action"] == "pass" and out["caption"]["caption_stripped"] and out["caption"]["mode"] == "geometric"
    assert out["box"][3] < 560
    page2 = blank(800, 800)
    cv2.circle(page2, (400, 300), 180, 0, 3)
    draw_text_rows(page2, 300, 300, 1, 16, glyph_h=14)         # a 16-glyph prose line inside the figure
    out2 = G.g7b(page2, [200, 100, 600, 500], 14.0, th, 300, None)
    assert out2["action"] == "exclude" and out2["reasons"] == ["prose_in_crop"]


def test_g7b_ocr_path_caption_and_label_tokens(th):
    page = blank(800, 800)
    draw_diagram(page, 400, 300, 180)
    draw_text_rows(page, 330, 505, 2, 11, glyph_h=14, row_gap=6)            # a two-line caption, 34 px tall
    units = [{"bbox": [328, 503, 470, 540], "text": "Fig. 46. The tangent drawn from P."},
             {"bbox": [390, 290, 404, 304], "text": "A"}]
    out = G.g7b(page, [200, 100, 600, 545], 14.0, th, 300, units)
    # the raw block height (37 px) is far from x-height 14, but its glyph height matches -> stripped
    assert out["caption"]["mode"] == "ocr" and out["caption"]["caption_stripped"]
    assert out["caption"]["figure_number"] == "46" and out["action"] == "pass"
    assert out["box"][3] <= 503


def test_caption_needs_body_text_glyph_height(th):
    from hmb.acquisition.context import strip_caption
    from hmb.acquisition.imgstats import sauvola_ink, sauvola_window
    page = blank(600, 600)
    draw_diagram(page, 300, 250, 150)
    draw_text_rows(page, 200, 520, 1, 10, glyph_h=30, glyph_w=14)          # display-size lettering
    crop = [100, 80, 500, 560]
    ink = sauvola_ink(page[80:560, 100:500], sauvola_window(th, 300), th.value("sauvola_k"))
    res = strip_caption(crop, ink, [{"bbox": [198, 518, 480, 552], "text": "Figure 12 title"}], 14.0, th)
    assert not res["caption_stripped"]


def test_prose_gate_ignores_paragraph_touching_the_border(th):
    from hmb.acquisition.context import prose_run_chars
    crop = [100, 100, 500, 500]
    para = [{"bbox": [0, 480, 800, 600], "text": "This paragraph continues well below the figure area"}]
    assert prose_run_chars(para, crop, None, 14.0, th)["prose_chars"] == 0      # small share inside -> ignored
    inside = [{"bbox": [150, 300, 450, 320], "text": "which proves the proposition"}]
    assert prose_run_chars(inside, crop, None, 14.0, th)["prose_chars"] >= 12


def test_g7c_text_block(th):
    page = blank(600, 800)
    draw_text_rows(page, 50, 50, 10, 40, glyph_h=14, gap=3)
    rec, rs = G.g7c(page, [40, 40, 560, 360], 14.0, th, 300, None)
    assert rs == ["text_block"]
    fig = draw_diagram(blank(600, 600), 300, 300, 200)
    assert G.g7c(fig, [50, 50, 550, 550], 14.0, th, 300, None)[0]["passed"]


def test_g7c_unmeasurable_passes_without_evidence_claim(th):
    fig = draw_diagram(blank(300, 300), 150, 150, 100)
    rec, rs = G.g7c(fig, [10, 10, 290, 290], None, th, 300, None)      # no OCR, no x-height
    assert rec["passed"] and rec["measured_value"] is None and rs == []


def test_g7d_clusters_only(th):
    rec, rs, ph = G.g7d(draw_diagram(blank(200, 200), 100, 100, 80), th)
    assert rec["passed"] and rs == [] and len(ph) == 16


# --------------------------------------------------------------------------- paid gates (FAKE providers)
class FakeJSONProvider:
    """Explicit fake provider returning scripted texts."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    def send(self, req, images):
        self.calls += 1
        return {"text": self.texts.pop(0), "input_tokens": 5, "output_tokens": 3, "model_version_string": "fake"}


def _passed_row():
    return {"proposal_id": "p1", "exclusion_reasons": [],
            "gates": [{"gate_id": g, "passed": True} for g in G.CPU_PROPOSAL_GATES]}


def _spec(mid, fam, role):
    return ModelSpec(model_id=mid, provider="fake", family=fam, role=role, unit_price_id="p")


@pytest.fixture
def crop(tmp_path):
    p = tmp_path / "crop.png"
    p.write_bytes(cv2.imencode(".png", draw_diagram(blank(100, 100), 50, 50, 30))[1].tobytes())
    return p


def _g8b(row, crop, spec, fp, tmp_path, ledger, allow_live=True):
    return G.g8b(row, crop, spec, layout=Layout(tmp_path), ledger=ledger, phase="PILOT", max_tokens=50,
                 allow_live=allow_live, provider_override=fp)


ACCEPT = '{"target_type": true, "readable": true, "tikz_representable": true}'


def test_g8_refuses_before_cpu_ladder(tmp_path, crop, ledger):
    row = _passed_row()
    row["gates"] = row["gates"][:-1]
    with pytest.raises(G.LadderIncomplete):
        _g8b(row, crop, _spec("v", "B", "verifier"), FakeJSONProvider([ACCEPT]), tmp_path, ledger)
    row = _passed_row()
    row["gates"][3]["passed"] = False
    with pytest.raises(G.LadderIncomplete):
        _g8b(row, crop, _spec("v", "B", "verifier"), FakeJSONProvider([]), tmp_path, ledger)
    assert ledger.events == []


def test_g8b_gatekeeper_alone_accepts_only_three_trues(tmp_path, crop, ledger):
    """Gatekeeper-only G8 (role audit 2026-09-23): one call, no sorter; pass iff all three answers true."""
    fv = FakeJSONProvider([ACCEPT])
    rec, rs, o, calls = _g8b(_passed_row(), crop, _spec("v", "B", "verifier"), fv, tmp_path / "a", ledger)
    assert fv.calls == 1 and rec["passed"] and rs == [] and o["verifier_pred"] == "accept"
    assert rec["version"] == "g8b-4" and rec["measured_value"]["target_type"] is True and calls[0]["fake"]
    rejects = ['{"target_type": false, "readable": true, "tikz_representable": true}',    # target-type test
               '{"target_type": true, "readable": false, "tikz_representable": true}',
               '{"target_type": true, "readable": true, "tikz_representable": false}',
               '{"target_type": true, "readable": true}',                                # missing answer
               '{"target_type": "true", "readable": true, "tikz_representable": true}',  # not a boolean
               '{"target_type": null, "readable": true, "tikz_representable": true}']    # unsure
    for i, text in enumerate(rejects):
        rec, rs, o, _ = _g8b(_passed_row(), crop, _spec("v", "B", "verifier"), FakeJSONProvider([text]),
                             tmp_path / f"r{i}", ledger)
        assert not rec["passed"] and rs == ["verifier_reject"] and o["verifier_pred"] == "reject", text


def test_g8b_diagram_type_is_descriptive_only(tmp_path, crop, ledger):
    """One of six labels is recorded for a survivor; the label never changes the decision, an invalid label
    is recorded as such (no seventh class), and a rejected crop gets no label."""
    for i, t in enumerate(DiagramType):
        text = ACCEPT[:-1] + f', "diagram_type": "{t.value}"}}'
        rec, rs, o, _ = _g8b(_passed_row(), crop, _spec("v", "B", "verifier"), FakeJSONProvider([text]),
                             tmp_path / f"t{i}", ledger)
        assert rec["passed"] and rs == [] and o["diagram_type"] == t.value and o["diagram_type_status"] == "predicted"
    for j, bad in enumerate(('"unknown"', '"plane_geometry"', "7", "null")):
        rec, rs, o, _ = _g8b(_passed_row(), crop, _spec("v", "B", "verifier"),
                             FakeJSONProvider([ACCEPT[:-1] + f', "diagram_type": {bad}}}']), tmp_path / f"b{j}", ledger)
        assert rec["passed"] and o["diagram_type"] is None and o["diagram_type_status"] == "invalid_reply"
    rec, rs, o, _ = _g8b(_passed_row(), crop, _spec("v", "B", "verifier"), FakeJSONProvider(
        ['{"target_type": false, "readable": true, "tikz_representable": true, "diagram_type": "geometry"}']),
        tmp_path / "rej", ledger)
    assert not rec["passed"] and "diagram_type" not in o and rec["measured_value"]["diagram_type"] == "geometry"


def test_g8b_malformed_reply_rejects_without_retry(tmp_path, crop, ledger):
    fv = FakeJSONProvider(["I think it is a diagram."])
    rec, rs, o, _ = _g8b(_passed_row(), crop, _spec("v", "B", "verifier"), fv, tmp_path, ledger)
    assert fv.calls == 1 and not rec["passed"] and rs == ["verifier_reject"] and o["verifier_pred"] is None


def test_g8b_requires_verifier_role(tmp_path, crop, ledger):
    with pytest.raises(G.GateBlocked):
        _g8b(_passed_row(), crop, _spec("v", "A", "judge"), FakeJSONProvider([]), tmp_path, ledger)


def test_gatekeeper_prompt_is_strict_with_target_type_test():
    """The strict prompt carries the reject-when-in-doubt rule, the §2.6 target types and the deferred
    non-target kinds, and asks for exactly the three parsed booleans."""
    text, sha = G.load_prompt(G.GATEKEEPER_PROMPT)
    low = text.lower()
    assert "when in doubt, reject" in low and "if you are unsure" in low
    for t in ("plane geometry", "solid geometry", "functions and curves", "trigonometry and measurement"):
        assert t in low, t
    for t in ("statistical charts", "maps", "decorative", "astronomy charts", "engineering", "isolated formulas"):
        assert t in low, t
    obj = G.parse_json_object(text[text.rindex("{"):].replace("true|false", "true").replace(
        '"geometry"|"graph_plot"|"table_grid"|"number_line_scale"|"relational_schematic"|"mixed_other"', '"geometry"'))
    assert tuple(obj) == G.GATEKEEPER_FIELDS + ("diagram_type",)
    assert obj is not None
    # 2026-09-26: the stated question count matches the fields requested (was "three" for four fields)
    words = {3: "three", 4: "four", 5: "five"}
    assert f"answer {words[len(obj)]} questions" in low and "answer three questions" not in low
    assert len(re.findall(r"^\d\. ", text, re.M)) == len(obj)
    for t in DiagramType:
        assert f'"{t.value}"' in text
    assert not any((G.PROMPT_DIR / n).exists() for n in ("g8a_category.txt", "g8d_metadata.txt", "g8b_verifier.txt"))
    assert not any(hasattr(G, f) for f in ("g8a", "g8c", "g8d", "g8c_merge"))


def test_g8_without_live_or_cache_raises(tmp_path, crop, ledger):
    from hmb.providers import LiveCallsDisabled
    with pytest.raises(LiveCallsDisabled):
        _g8b(_passed_row(), crop, _spec("v", "B", "verifier"), FakeJSONProvider(['{}']), tmp_path, ledger,
             allow_live=False)


def test_gatekeeper_native_schema_matches_the_prompt_fields():
    sch = G.gatekeeper_json_schema()["schema"]
    assert tuple(sch["required"]) == G.GATEKEEPER_FIELDS + ("diagram_type",)
    assert sch["properties"]["diagram_type"]["enum"] == [t.value for t in DiagramType]
    assert sch["additionalProperties"] is False
