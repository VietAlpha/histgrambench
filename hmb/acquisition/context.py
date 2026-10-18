"""Panels, captions, residual prose, text blocks, and the context window (§6.4, G7b/G7c).

All boxes are page pixels [x0, y0, x1, y1] (exclusive max) unless a name says `local`.
OCR units are dicts {bbox, text} in page pixels (PaddleOCR-VL gives layout blocks, not words;
each block is treated as one OCR unit). When OCR is unavailable, the caption, prose and
text-block rules fall back to their geometric components (band position, modal text height,
text-height component counts); no proposal is excluded *because* OCR failed.

Geometric text model (part of `ctxwin-1`/`g7-geom-1`): a text-height component has height
within +/- text_height_tol of the page x-height; components form a line when their vertical
centres lie within 0.5 x-height and horizontal gaps are <= 1.5 x-heights; a line breaks into
words at gaps > 0.5 x-height.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

CTX_RULE = "ctxwin-1"
GEOM_TEXT_VERSION = "g7-geom-1"
LINE_CENTER_XH = 0.5
LINE_GAP_XH = 1.5
WORD_GAP_XH = 0.5

_PATTERNS_PATH = Path(__file__).with_name("data") / "figure_number_patterns.json"
MATH_LABEL_RE = re.compile(
    r"^[A-Za-z]'{0,2}$|^[A-Za-z]_?\d{1,2}$|^[0-9]{1,3}(°|′|″)?$|^[Ͱ-Ͽ]$|^[A-Z]{2,3}$"
    r"|^[㐀-䶿一-鿿豈-﫿]$")


def load_figure_patterns() -> dict:
    d = json.loads(_PATTERNS_PATH.read_text(encoding="utf-8"))
    comp = {lang: [re.compile(p, re.IGNORECASE) for p in pats] for lang, pats in d["patterns"].items()}
    return {"version": d["version"], "patterns": comp}


def figure_number(text: str, patterns: dict | None = None) -> str | None:
    """First figure number found by any language's pattern (languages in sorted order)."""
    pats = (patterns or load_figure_patterns())["patterns"]
    for lang in sorted(pats):
        for rx in pats[lang]:
            m = rx.search(text or "")
            if m:
                return m.group("n").lower()
    return None


def is_math_label(token: str) -> bool:
    return bool(MATH_LABEL_RE.match(token))


# --------------------------------------------------------------------------- geometry helpers
def to_local(b, crop):
    return [b[0] - crop[0], b[1] - crop[1], b[2] - crop[0], b[3] - crop[1]]


def inside(b, box) -> bool:
    return b[0] >= box[0] and b[1] >= box[1] and b[2] <= box[2] and b[3] <= box[3]


def overlaps(a, b) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def text_components(ink_local: np.ndarray, xh: float | None, tol: float) -> list[list[int]]:
    """Bboxes (local) of components whose height is within +/- tol of the x-height."""
    if not xh:
        return []
    n, _, st, _ = cv2.connectedComponentsWithStats(ink_local.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        x, y, w, h = (int(v) for v in st[i, :4])
        if (1 - tol) * xh <= h <= (1 + tol) * xh:
            out.append([x, y, x + w, y + h])
    return out


def text_lines(comps: list[list[int]], xh: float) -> list[list[list[int]]]:
    """Group text-height components into lines (sorted left to right)."""
    lines: list[list[list[int]]] = []
    for c in sorted(comps, key=lambda b: ((b[1] + b[3]) / 2, b[0])):
        cy = (c[1] + c[3]) / 2
        for ln in lines:
            last = ln[-1]
            if abs((last[1] + last[3]) / 2 - cy) <= LINE_CENTER_XH * xh and c[0] - max(b[2] for b in ln) <= LINE_GAP_XH * xh:
                ln.append(c)
                break
        else:
            lines.append([c])
    return [sorted(ln) for ln in lines]


def line_words(line: list[list[int]], xh: float) -> list[int]:
    """Component counts per geometric word in a line."""
    words, cur, right = [], 0, None
    for c in line:
        if right is not None and c[0] - right > WORD_GAP_XH * xh:
            words.append(cur)
            cur = 0
        cur += 1
        right = c[2] if right is None else max(right, c[2])
    if cur:
        words.append(cur)
    return words


# --------------------------------------------------------------------------- G7b rule 1: panels
def panel_split(ink_local: np.ndarray, th, depth: int = 0) -> list[list[int]]:
    """Recursive projection-profile panel split (§6.4 rule 1); returns local panel boxes.

    At each level the single widest zero-ink gap band >= g7b_gap_frac x max(w, h) (either axis)
    whose two sides each hold >= g7b_side_ink_min of this region's (the parent's) ink bisects the
    region; each side recurses until g7b_max_depth. One box back means no split. The caller
    applies g7b_max_panels (more panels -> multi_panel_unresolved).
    """
    H, W = ink_local.shape
    total = int(ink_local.sum())
    if total == 0 or depth >= th.value("g7b_max_depth"):
        return [[0, 0, W, H]]
    min_gap = th.value("g7b_gap_frac") * max(W, H)
    side_min = th.value("g7b_side_ink_min")
    best = None
    for axis in (0, 1):                    # 0: vertical gap bands (split columns), 1: horizontal
        prof = ink_local.sum(axis=axis)
        cum = np.cumsum(prof)
        i, n = 0, len(prof)
        while i < n:
            if prof[i] == 0:
                j = i
                while j < n and prof[j] == 0:
                    j += 1
                if i > 0 and j < n and (j - i) >= min_gap:
                    left = int(cum[i - 1])
                    if left >= side_min * total and total - left >= side_min * total:
                        if best is None or (j - i) > best[0]:
                            best = (j - i, axis, i, j)
                i = j
            else:
                i += 1
    if best is None:
        return [[0, 0, W, H]]
    _, axis, a, b = best
    cut = (a + b) // 2
    parts = [(0, 0, cut, H), (cut, 0, W, H)] if axis == 0 else [(0, 0, W, cut), (0, cut, W, H)]
    out = []
    for x0, y0, x1, y1 in parts:
        for p in panel_split(ink_local[y0:y1, x0:x1], th, depth + 1):
            out.append([x0 + p[0], y0 + p[1], x0 + p[2], y0 + p[3]])
    return out


def tighten(ink_local: np.ndarray, box) -> list[int]:
    x0, y0, x1, y1 = box
    ys, xs = np.nonzero(ink_local[y0:y1, x0:x1])
    if len(xs) == 0:
        return list(box)
    return [x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1]


# --------------------------------------------------------------------------- G7b rule 2: captions
CHAR_MIN_H = 3


def block_char_height(ink_local: np.ndarray, b) -> float | None:
    """Character-height estimate of an OCR block: the modal height of the glyph components
    (height >= 3 px) inside the block box, comparable with the page's modal body-text height.
    A multi-line block therefore reads as one line height, not its full box height."""
    H, W = ink_local.shape
    x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(W, b[2]), min(H, b[3])
    if x1 <= x0 or y1 <= y0:
        return None
    _, _, st, _ = cv2.connectedComponentsWithStats(ink_local[y0:y1, x0:x1].astype(np.uint8), connectivity=8)
    h = st[1:, cv2.CC_STAT_HEIGHT]
    h = h[h >= CHAR_MIN_H]
    return float(int(np.argmax(np.bincount(h)))) if len(h) else None


def inside_share(b, crop) -> float:
    """Share of block b's area that lies inside the crop."""
    ix = max(0, min(b[2], crop[2]) - max(b[0], crop[0]))
    iy = max(0, min(b[3], crop[3]) - max(b[1], crop[1]))
    a = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return ix * iy / a if a else 0.0


PROSE_INSIDE_MIN = 0.5   # a block counts for the prose gate only if >= 50% of its area is inside the crop
def strip_caption(crop: list[int], ink_local: np.ndarray, ocr_units: list[dict] | None,
                  xh: float | None, th, patterns: dict | None = None) -> dict:
    """Return {crop (possibly shrunk), caption_stripped, caption_text, figure_number, mode}."""
    H, W = ink_local.shape
    band = th.value("caption_band_frac")
    tol = th.value("text_height_tol")
    top_lim, bot_lim = band * H, (1 - band) * H
    res = {"crop": list(crop), "caption_stripped": False, "caption_text": None,
           "figure_number": None, "mode": "ocr" if ocr_units is not None else "geometric"}
    if ocr_units is not None:
        loc = [(to_local(u["bbox"], crop), u["text"]) for u in ocr_units if inside(u["bbox"], crop)]
        cut_top, cut_bot, texts = 0, H, []
        for b, text in loc:
            in_top, in_bot = b[3] <= top_lim, b[1] >= bot_lim
            if not (in_top or in_bot):
                continue
            ch = block_char_height(ink_local, b)
            h_ok = xh is None or (ch is not None and (1 - tol) * xh <= ch <= (1 + tol) * xh)
            fig = figure_number(text, patterns)
            if h_ok and (len(text.strip()) >= th.value("caption_min_chars") or fig is not None):
                texts.append((b[1], text.strip()))
                res["figure_number"] = res["figure_number"] or fig
                if in_top:
                    cut_top = max(cut_top, b[3])
                else:
                    cut_bot = min(cut_bot, b[1])
        if texts:
            res.update(caption_stripped=True, caption_text=" ".join(t for _, t in sorted(texts)))
            res["crop"] = [crop[0], crop[1] + cut_top, crop[2], crop[1] + cut_bot]
        return res
    # geometric fallback: an ink band wholly in the top/bottom band of text-height glyphs
    if not xh:
        return res
    rows = ink_local.sum(axis=1)
    bands, i = [], 0
    while i < H:
        if rows[i] > 0:
            j = i
            while j < H and rows[j] > 0:
                j += 1
            bands.append((i, j))
            i = j
        else:
            i += 1
    if len(bands) < 2:
        return res
    cut_top, cut_bot = 0, H
    for idx, (a, b) in enumerate(bands):
        is_top, is_bot = b <= top_lim and idx == 0, a >= bot_lim and idx == len(bands) - 1
        if not (is_top or is_bot):
            continue
        comps = text_components(ink_local[a:b], xh, tol)
        n_all = cv2.connectedComponents(ink_local[a:b].astype(np.uint8), connectivity=8)[0] - 1
        if len(comps) >= th.value("caption_min_chars") and len(comps) >= 0.5 * n_all:
            if is_top:
                cut_top = b
            else:
                cut_bot = a
    if cut_top > 0 or cut_bot < H:
        res.update(caption_stripped=True, crop=[crop[0], crop[1] + cut_top, crop[2], crop[1] + cut_bot])
    return res


# --------------------------------------------------------------------------- G7b rule 3: prose
def prose_run_chars(ocr_units: list[dict] | None, crop: list[int], ink_local: np.ndarray,
                    xh: float | None, th) -> dict:
    """Longest non-label run inside the crop: OCR characters, or glyph count of a text line."""
    if ocr_units is not None:
        best = 0
        for u in ocr_units:
            if inside_share(u["bbox"], crop) < PROSE_INSIDE_MIN:   # never a paragraph touching the border
                continue
            run = 0
            for tok in (u["text"] or "").split():
                if is_math_label(tok):
                    best, run = max(best, run), 0
                else:
                    run += len(tok)
            best = max(best, run)
        return {"mode": "ocr", "prose_chars": best}
    if not xh:
        return {"mode": "geometric", "prose_chars": None}
    lines = text_lines(text_components(ink_local, xh, th.value("text_height_tol")), xh)
    return {"mode": "geometric", "prose_chars": max((len(ln) for ln in lines), default=0)}


# --------------------------------------------------------------------------- G7c text block
def text_block_measures(ocr_units: list[dict] | None, crop: list[int], ink_local: np.ndarray,
                        xh: float | None, th) -> dict:
    """{word_cover, median_word_len, mode}: share of crop ink under text, median word length."""
    total = int(ink_local.sum())
    if total == 0:
        return {"mode": "none", "word_cover": 0.0, "median_word_len": None}
    if ocr_units is not None:
        mask = np.zeros_like(ink_local, bool)
        lens = []
        for u in ocr_units:
            if not overlaps(u["bbox"], crop):
                continue
            b = to_local(u["bbox"], crop)
            mask[max(0, b[1]):max(0, b[3]), max(0, b[0]):max(0, b[2])] = True
            lens.extend(len(t) for t in (u["text"] or "").split())
        return {"mode": "ocr", "word_cover": float((mask & ink_local).sum()) / total,
                "median_word_len": float(np.median(lens)) if lens else None}
    if not xh:
        return {"mode": "geometric", "word_cover": None, "median_word_len": None}
    comps = text_components(ink_local, xh, th.value("text_height_tol"))
    mask = np.zeros_like(ink_local, bool)
    for b in comps:
        mask[b[1]:b[3], b[0]:b[2]] = True
    lens = [w for ln in text_lines(comps, xh) for w in line_words(ln, xh)]
    return {"mode": "geometric", "word_cover": float((mask & ink_local).sum()) / total,
            "median_word_len": float(np.median(lens)) if lens else None}


# --------------------------------------------------------------------------- context window
def facing_page(page_index: int) -> int | None:
    """Spread partner assuming index 0 is a recto: (1,2), (3,4), ... ; index 0 has none."""
    if page_index == 0:
        return None
    return page_index + 1 if page_index % 2 == 1 else page_index - 1


def context_window(crop: list[int], page_index: int, units_by_page: dict[int, list[dict]] | None,
                   caption_text: str | None, fig_no: str | None, th, patterns: dict | None = None) -> dict:
    """ctxwin-1: nearby blocks (+/- 1.5 crop heights, same page) plus same/facing-page blocks
    referencing the recovered figure number; reading order; capped at block boundaries;
    stripped caption prepended. OCR unavailable -> caption only (possibly empty)."""
    cap = int(th.value("ctx_cap_chars"))
    k = th.value("ctx_window_heights")
    h = crop[3] - crop[1]
    chosen: list[tuple[int, dict]] = []
    if units_by_page is not None:
        lo, hi = crop[1] - k * h, crop[3] + k * h
        for u in units_by_page.get(page_index, []):
            b = u["bbox"]
            if b[1] >= lo and b[3] <= hi and not inside(b, crop):
                chosen.append((page_index, u))
        if fig_no:
            for p in (page_index, facing_page(page_index)):
                if p is None:
                    continue
                for u in units_by_page.get(p, []):
                    if any(x is u for _, x in chosen) or (p == page_index and inside(u["bbox"], crop)):
                        continue
                    if figure_number(u["text"], patterns) == fig_no:
                        chosen.append((p, u))
    chosen.sort(key=lambda pu: (pu[0], pu[1]["bbox"][1], pu[1]["bbox"][0]))
    parts, n_chars, used = [], 0, []
    for p, u in chosen:
        t = (u["text"] or "").strip()
        if not t:
            continue
        if n_chars + len(t) > cap:
            break
        parts.append(t)
        n_chars += len(t)
        used.append((p, u))
    text = "\n".join(([caption_text.strip()] if caption_text else []) + parts)
    same = [u["bbox"] for p, u in used if p == page_index]
    ctx_box = [min([crop[0]] + [b[0] for b in same]), min([crop[1]] + [b[1] for b in same]),
               max([crop[2]] + [b[2] for b in same]), max([crop[3]] + [b[3] for b in same])]
    return {"context_window_rule": CTX_RULE, "context_text": text, "context_char_count": len(text),
            "context_block_count": len(used), "context_bbox": ctx_box}
