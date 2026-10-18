"""§6.5 statistics on hand-built images with independently derived expectations."""
import cv2
import numpy as np
import pytest

from hmb.acquisition import imgstats as S
from w1_helpers import blank, draw_text_rows


def stats(img, th, **kw):
    return S.compute_image_stats(img, th, **kw)


def test_contrast_and_ink_fraction(th):
    img = blank(100, 100)
    img[:, :50] = 0                      # half black: contrast = std of {0,1} halves = 0.5
    img[0, 99] = 128                     # a third grey level so the crop is not bitonal
    s = stats(img, th)
    assert s["contrast_rms"] == pytest.approx(0.5, abs=0.002)
    img2 = blank(200, 200)
    cv2.rectangle(img2, (50, 50), (149, 59), 0, -1)     # 100 x 10 bar = 1000 px
    s2 = stats(img2, th)
    assert s2["ink_frac"] == pytest.approx(1000 / 40000, rel=0.05)


def test_stroke_width_and_endpoints_of_a_straight_bar(th):
    img = blank(100, 600)
    img[48:53, 50:550] = 0               # 5 px thick, 500 px long
    s = stats(img, th)
    # 2 x distance-to-background at the centre row: centre is 3 px from the nearest background row
    assert 5.0 <= s["stroke_w_mean"] <= 6.5
    # one straight skeleton: 2 endpoints over ~500 px of skeleton
    assert s["endpoint_density"] == pytest.approx(2 / 500, rel=0.25)


def test_frag_index_counts_small_components(th):
    img = blank(300, 300)
    img[100:110, 20:280] = 0             # 2600 px bar
    for k in range(5):                   # five 2x2 specks (4 px < 0.005 * total ink)
        img[200 + 10 * k:202 + 10 * k, 50:52] = 0
    total = 2600 + 5 * 4
    s = stats(img, th)
    assert s["frag_index"] == pytest.approx(5 / total * 1000, rel=0.02)


def test_bitonal_undefined_fields(th):
    img = blank(100, 100)
    img[40:60, 10:90] = 0
    s = stats(img, th)
    for k in S.BITONAL_UNDEFINED:
        assert s[k] is None
    assert s["ink_frac"] is not None and s["stats_version"] == "imgstat-2"


def test_midtone_and_speckle_defined_for_grayscale(th):
    rng = np.random.default_rng(1)
    img = np.clip(rng.normal(220, 3, (120, 120)), 0, 255).astype(np.uint8)
    img[50:70, 10:110] = 30
    img[30:35, 10:110] = 120             # midtone band
    s = stats(img, th)
    assert s["midtone_frac"] is not None and s["midtone_frac"] > 0
    assert 0 < s["speckle_mad"] < 0.05
    assert s["blur_norm"] is not None and s["blur_norm"] > 0


def test_midtone_strictly_between_modes(th):
    img = np.full((100, 100), 220, np.uint8)          # paper mode 220
    img[10:40, 10:90] = 30                             # ink mode 30 (2400 px)
    img[60:62, 10:90] = 200                            # 160 px strictly between the modes (near paper)
    img[70:72, 10:90] = 40                             # 160 px strictly between the modes (near ink)
    s = stats(img, th)
    ink_n = int(S.sauvola_ink(img, S.sauvola_window(th, 300), th.value("sauvola_k")).sum())
    assert s["midtone_frac"] == pytest.approx(320 / ink_n)
    assert s["stats_version"] == "imgstat-2"


def test_modal_text_height(th):
    img = blank(600, 800)
    draw_text_rows(img, 20, 20, 6, 40, glyph_h=14)
    draw_text_rows(img, 20, 400, 1, 5, glyph_h=20)
    ink = S.sauvola_ink(img, S.sauvola_window(th, 300), th.value("sauvola_k"))
    assert S.modal_text_height(ink) == 14
    assert S.modal_text_height(np.zeros((50, 50), bool)) is None


def test_skew_estimate_on_rotated_text(th):
    img = blank(800, 800)
    draw_text_rows(img, 100, 100, 12, 45, glyph_h=12, row_gap=30)
    M = cv2.getRotationMatrix2D((400, 400), 2.0, 1.0)
    rot = cv2.warpAffine(img, M, (800, 800), borderValue=255, flags=cv2.INTER_NEAREST)
    ink_rot = rot < 128
    ink0 = img < 128
    assert abs(S.estimate_skew_deg(ink0)) <= 0.1
    assert abs(abs(S.estimate_skew_deg(ink_rot)) - 2.0) <= 0.3


def test_label_contact_counts_glyph_touching_a_long_line(th):
    img = blank(400, 600)
    img[200:203, 20:580] = 0                          # long stroke
    cv2.rectangle(img, (300, 188), (300, 199), 0, -1)  # 12 px spur touching the line (x-height 12)
    s = stats(img, th, eff_res_xheight_px=12.0)
    assert s["label_contact_cc"] == 1.0
    img2 = blank(400, 600)
    img2[200:203, 20:580] = 0
    assert stats(img2, th, eff_res_xheight_px=12.0)["label_contact_cc"] == 0.0
    assert stats(img2, th)["label_contact_cc"] is None     # no x-height -> undefined


def test_fixture_tag_without_thresholds_and_bad_input():
    assert S.compute_image_stats(blank(20, 20))["profile"] == "fixture"
    with pytest.raises(ValueError):
        S.compute_image_stats(np.zeros((5, 5, 3), np.uint8))
    with pytest.raises(ValueError):
        S.compute_image_stats(np.zeros((5, 5), np.float32))
