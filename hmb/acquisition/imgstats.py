"""Deterministic image statistics (§6.5), Sauvola ink mask (§6.3.1), shared crop analysis.

`compute_image_stats(gray_uint8, th=None, ...)` is the public entry point; Opus 5's degradation
calibration imports it. Every field is a float or None (None = undefined for this input, with
the reason implied by the definition, never 0 by default).

Operational choices inside `imgstat-2` (the plan names the statistic, not every detail):
  * Sauvola uses the standard 8-bit dynamic range R = 128;
  * connected components are 8-connected; skeleton length = skeleton pixel count;
  * a skeleton endpoint is a skeleton pixel with <= 1 skeleton neighbour;
  * the two Otsu modes are the histogram argmax on each side of the Otsu threshold;
    `midtone_frac` = pixels strictly between the two modes / ink pixels (imgstat-2; §6.5);
  * `eff_res_xheight_px` is the modal height of ink components with height >= 3 px and
    area >= 4 px (specks excluded), requiring >= 20 such components;
  * `skew_deg` maximises row-projection variance over [-5, 5] deg in 0.1 deg steps;
  * `label_contact_cc` counts long-stroke components (skeleton length >= g6_multi_frac x
    crop diagonal) that carry a spur branch (junction-free skeleton branch ending in an
    endpoint) whose extent is within +/- text_height_tol of the page x-height.
These constants are part of the statistic's definition and change only with `STATS_VERSION`.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage
from skimage.filters import threshold_otsu, threshold_sauvola
from skimage.morphology import skeletonize

STATS_VERSION = "imgstat-2"
SAUVOLA_R = 128.0     # Sauvola dynamic range of the standard deviation for 8-bit images
STAT_FIELDS = ("eff_res_xheight_px", "contrast_rms", "ink_frac", "stroke_w_mean", "stroke_w_cv",
               "frag_index", "endpoint_density", "midtone_frac", "speckle_mad", "blur_norm",
               "skew_deg", "label_contact_cc")
BITONAL_UNDEFINED = ("contrast_rms", "blur_norm", "midtone_frac", "speckle_mad")

XH_MIN_H = 3
XH_MIN_AREA = 4
XH_MIN_COMPONENTS = 20
SKEW_RANGE_DEG = 5.0
SKEW_STEP_DEG = 0.1
SKEW_MAX_POINTS = 200_000
_K8 = np.ones((3, 3), np.uint8)


def sauvola_window(th, dpi: float) -> int:
    """Sauvola window scaled from its 300-dpi-equivalent value; odd, >= 3."""
    w = int(round(th.value("sauvola_window_px") * float(dpi) / 300.0))
    w = max(3, w)
    return w if w % 2 == 1 else w + 1


def sauvola_ink(gray: np.ndarray, window: int, k: float) -> np.ndarray:
    """Boolean ink mask (dark on light) by Sauvola thresholding."""
    g = _check_gray(gray)
    if g.size == 0:
        return np.zeros(g.shape, bool)
    thr = threshold_sauvola(g.astype(np.float64), window_size=window, k=k, r=SAUVOLA_R)
    return g.astype(np.float64) < thr


def _check_gray(gray) -> np.ndarray:
    g = np.asarray(gray)
    if g.ndim != 2 or g.dtype != np.uint8:
        raise ValueError(f"expected a 2-D uint8 grayscale array, got {g.dtype} {g.shape}")
    return g


@dataclass
class CropAnalysis:
    """Everything the §6.3 gates and §6.5 statistics need from one crop, computed once."""
    gray: np.ndarray
    ink: np.ndarray
    labels: np.ndarray                  # 8-connected component labels of ink (0 = background)
    areas: np.ndarray                   # area per label (index 0 unused)
    bboxes: np.ndarray                  # [x, y, w, h] per label (index 0 unused)
    skel: np.ndarray
    skel_nbrs: np.ndarray               # skeleton neighbour count at skeleton pixels, else 0
    skel_len: np.ndarray                # skeleton pixel count per label

    @property
    def n_components(self) -> int:
        return int(len(self.areas) - 1)

    @property
    def total_ink(self) -> int:
        return int(self.ink.sum())

    @property
    def diagonal(self) -> float:
        h, w = self.gray.shape
        return float(np.hypot(h, w))


def analyze(gray: np.ndarray, ink: np.ndarray) -> CropAnalysis:
    g = _check_gray(gray)
    ink = np.asarray(ink, bool)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    areas = stats[:, cv2.CC_STAT_AREA].astype(np.int64)
    bboxes = stats[:, :4].astype(np.int64)
    skel = skeletonize(ink) if ink.any() else np.zeros_like(ink)
    nb = ndimage.convolve(skel.astype(np.uint8), _K8, mode="constant", cval=0).astype(np.int32) - 1
    nb = np.where(skel, nb, 0)
    skel_len = np.bincount(labels[skel], minlength=n).astype(np.int64)
    return CropAnalysis(g, ink, labels, areas, bboxes, skel, nb, skel_len)


# --------------------------------------------------------------------------- page-level measures
def modal_text_height(ink: np.ndarray) -> float | None:
    """Modal ink-component height (px) = eff_res_xheight_px (§6.1). None when not measurable."""
    ink = np.asarray(ink, bool)
    if not ink.any():
        return None
    _, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    h = stats[1:, cv2.CC_STAT_HEIGHT]
    a = stats[1:, cv2.CC_STAT_AREA]
    h = h[(h >= XH_MIN_H) & (a >= XH_MIN_AREA)]
    if len(h) < XH_MIN_COMPONENTS:
        return None
    counts = np.bincount(h)
    return float(int(np.argmax(counts)))     # argmax returns the smallest height on ties


def estimate_skew_deg(ink: np.ndarray) -> float | None:
    """Dominant text orientation by projection-profile variance maximisation (page level)."""
    ys, xs = np.nonzero(np.asarray(ink, bool))
    if len(xs) < 100:
        return None
    if len(xs) > SKEW_MAX_POINTS:
        idx = np.linspace(0, len(xs) - 1, SKEW_MAX_POINTS).astype(np.int64)   # deterministic
        xs, ys = xs[idx], ys[idx]
    xs = xs.astype(np.float64)
    ys = ys.astype(np.float64)
    best, best_a = -1.0, 0.0
    n_steps = int(round(2 * SKEW_RANGE_DEG / SKEW_STEP_DEG))
    for i in range(n_steps + 1):
        a = -SKEW_RANGE_DEG + i * SKEW_STEP_DEG
        t = np.deg2rad(a)
        proj = ys * np.cos(t) - xs * np.sin(t)
        proj -= proj.min()
        hist = np.bincount(proj.astype(np.int64))
        v = float(hist.astype(np.float64).var())
        if v > best + 1e-9:
            best, best_a = v, a
    return round(best_a, 3)


# --------------------------------------------------------------------------- crop statistics
def otsu_modes(gray: np.ndarray) -> tuple[int, int, int] | None:
    """(threshold, dark mode, light mode) or None if the crop has a single grey level."""
    g = _check_gray(gray)
    if g.size == 0 or g.min() == g.max():
        return None
    t = int(threshold_otsu(g))
    hist = np.bincount(g.ravel(), minlength=256)
    lo = int(np.argmax(hist[: t + 1]))
    hi = t + 1 + int(np.argmax(hist[t + 1:]))
    return t, lo, hi


def is_bitonal(gray: np.ndarray) -> bool:
    return len(np.unique(gray)) <= 2


def compute_from_analysis(an: CropAnalysis, th, *, eff_res_xheight_px: float | None = None,
                          skew_deg: float | None = None) -> dict:
    g = an.gray.astype(np.float64) / 255.0
    n_pix = g.size
    ink_n = an.total_ink
    out: dict = {k: None for k in STAT_FIELDS}
    out["eff_res_xheight_px"] = None if eff_res_xheight_px is None else float(eff_res_xheight_px)
    out["skew_deg"] = None if skew_deg is None else float(skew_deg)
    if n_pix == 0:
        return out
    out["contrast_rms"] = float(g.std())
    out["ink_frac"] = ink_n / n_pix
    sk = an.skel
    if sk.any():
        dt = cv2.distanceTransform(an.ink.astype(np.uint8), cv2.DIST_L2, 5)
        w = 2.0 * dt[sk].astype(np.float64)
        out["stroke_w_mean"] = float(w.mean())
        out["stroke_w_cv"] = float(w.std() / w.mean()) if w.mean() > 0 else None
        n_end = int(((an.skel_nbrs <= 1) & sk).sum())
        out["endpoint_density"] = n_end / int(sk.sum())
    if ink_n > 0:
        small = th.value("frag_area_frac") * ink_n
        n_small = int((an.areas[1:] < small).sum())
        out["frag_index"] = n_small / ink_n * 1000.0
    modes = otsu_modes(an.gray)
    if modes and ink_n > 0:
        _, lo, hi = modes
        mid = (an.gray > lo) & (an.gray < hi)
        out["midtone_frac"] = float(mid.sum()) / ink_n
    paper = g[~an.ink]
    if paper.size:
        out["speckle_mad"] = float(np.median(np.abs(paper - np.median(paper))))
    if out["contrast_rms"] and out["contrast_rms"] > 0:
        lap = cv2.Laplacian(g, cv2.CV_64F)
        out["blur_norm"] = float(lap.var() / out["contrast_rms"] ** 2)
    out["label_contact_cc"] = label_contact_cc(an, th, eff_res_xheight_px)
    if is_bitonal(an.gray):
        for k in BITONAL_UNDEFINED:
            out[k] = None
    return out


def label_contact_cc(an: CropAnalysis, th, xh: float | None) -> float | None:
    if xh is None or xh <= 0:
        return None
    long_min = th.value("g6_multi_frac") * an.diagonal
    tol = th.value("text_height_tol")
    lo, hi = (1 - tol) * xh, (1 + tol) * xh
    long_labels = np.nonzero(an.skel_len >= long_min)[0]
    long_labels = long_labels[long_labels > 0]
    if len(long_labels) == 0:
        return 0.0
    junction = (an.skel_nbrs >= 3) & an.skel
    junction = cv2.dilate(junction.astype(np.uint8), _K8) > 0
    branches = an.skel & ~junction
    _, blab, bstats, _ = cv2.connectedComponentsWithStats(branches.astype(np.uint8), connectivity=8)
    extent = np.maximum(bstats[:, cv2.CC_STAT_WIDTH], bstats[:, cv2.CC_STAT_HEIGHT])
    qualifying = (extent >= lo) & (extent <= hi)
    qualifying[0] = False
    endpoints = an.skel & (an.skel_nbrs <= 1) & branches
    # spur branches (those holding an endpoint) of qualifying extent, mapped to their component
    comp_of = an.labels[endpoints]
    br_of = blab[endpoints]
    hit = qualifying[br_of]
    comps_with_spur = set(np.unique(comp_of[hit]).tolist())
    return float(sum(1 for lab in long_labels.tolist() if lab in comps_with_spur))


def compute_image_stats(gray_uint8, th=None, *, dpi: float = 300, eff_res_xheight_px: float | None = None,
                        skew_deg: float | None = None) -> dict:
    """§6.5 statistics vector on an 8-bit grayscale crop and its Sauvola ink mask.

    `th` is a `Thresholds`; production callers must pass it. With `th=None` the Sauvola window
    and k come from `fixture_thresholds()` and the result carries `profile: "fixture"`.
    `eff_res_xheight_px` and `skew_deg` are page-level and passed in by the caller.
    """
    profile = None
    if th is None:
        from ..config import fixture_thresholds
        th, profile = fixture_thresholds(), "fixture"
    g = _check_gray(gray_uint8)
    ink = sauvola_ink(g, sauvola_window(th, dpi), th.value("sauvola_k"))
    out = compute_from_analysis(analyze(g, ink), th, eff_res_xheight_px=eff_res_xheight_px,
                                skew_deg=skew_deg)
    out["stats_version"] = STATS_VERSION
    if profile:
        out["profile"] = profile
    return out


def numeric_stats(stats: dict) -> dict:
    """Only the §6.5 fields (floats or None), in fixed order; for the funnel row."""
    return {k: (None if stats.get(k) is None else float(stats[k])) for k in STAT_FIELDS}
