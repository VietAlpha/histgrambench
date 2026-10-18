"""Figure 3. Worked example (Track A, item hmb_284751dbdd7d38c6): (a) the benchmark crop and (b)-(g) the six
models' compiled drawings, in table order. Drawings are the retained compiled PDFs (cand.pdf), rasterised with
PyMuPDF and shown at one common physical scale (the largest that fits a 3 x 2 grid beside the crop), so their size differences are real."""
from __future__ import annotations

import fitz
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from . import data, style

ITEM = "hmb_284751dbdd7d38c6"
TRACK = "A"
RASTER_DPI = 600
CROP_MAX_FRAC = 0.27  # the 161 x 105 px benchmark crop is shown no wider than this share of the figure


def raster(pdf) -> np.ndarray:
    page = fitz.open(pdf)[0]
    pix = page.get_pixmap(dpi=RASTER_DPI, alpha=False)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)


def main() -> list:
    style.apply()
    crop = np.asarray(Image.open(data.DATA / "images" / "crops" / f"{ITEM}.png").convert("RGB"))
    models = data.model_order(TRACK)
    pdfs = {m: raster(data.item_artifacts(m, TRACK, ITEM)["pdf"]) for m in models}
    # Natural sizes in TeX inches. Layout: the crop on the left spanning two rows, the six drawings in a
    # 3 x 2 grid on the right, all drawings at one common scale.
    nat = {m: (a.shape[1] / RASTER_DPI, a.shape[0] / RASTER_DPI) for m, a in pdfs.items()}
    tallest = max(h for _, h in nat.values())
    widest = max(w for w, _ in nat.values())
    book = data.rights()[data.item_record(ITEM)[1]["source"]["book_id"]]
    titles = [f"(a)  Original, {book['publication_year']}"]
    titles += [f"({'bcdefg'[k]})  {style.MODEL_NAMES[m]}" for k, m in enumerate(models)]

    W, gap_x, gap_y, cap, cols = style.FULL_W, 0.16, 0.08, 0.2, 3
    probe = plt.figure()
    r = probe.canvas.get_renderer()  # type: ignore[attr-defined]  (Agg canvas)
    label_w = max(probe.text(0, 0, t, fontsize=style.SMALL).get_window_extent(r).width / probe.dpi
                  for t in titles[1:])
    plt.close(probe)
    aspect = crop.shape[1] / crop.shape[0]

    def geometry(s):
        row_h = tallest * s + cap
        H = 2 * row_h + gap_y
        crop_w = min((H - cap) * aspect, CROP_MAX_FRAC * W)
        cell_w = (W - crop_w - gap_x * cols) / cols
        return row_h, H, crop_w, cell_w

    lo, hi = 0.05, 2.0      # largest common drawing scale whose cells hold every drawing and label
    for _ in range(50):
        s = (lo + hi) / 2
        _, _, _, cell_w = geometry(s)
        lo, hi = (s, hi) if cell_w >= max(widest * s, label_w) else (lo, s)
    s = lo
    row_h, H, crop_w, cell_w = geometry(s)
    fig = plt.figure(figsize=(W, H))

    def place(img, x, y, w, h):
        ax = fig.add_axes((x / W, y / H, w / W, h / H))
        ax.imshow(img)
        ax.set_axis_off()
        ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False, ec=style.FRAME, lw=0.4))

    crop_h = crop_w / aspect
    crop_y = cap + (H - cap - crop_h) / 2
    place(crop, 0, crop_y, crop_w, crop_h)
    fig.text(crop_w / 2 / W, (crop_y - 0.06) / H, titles[0], ha="center", va="top", fontsize=style.SMALL)
    for k, m in enumerate(models):
        row, col = divmod(k, cols)
        x0 = crop_w + gap_x + col * (cell_w + gap_x)
        y0 = H - (row + 1) * row_h - row * gap_y + cap
        w, h = nat[m][0] * s, nat[m][1] * s
        place(pdfs[m], x0 + (cell_w - w) / 2, y0, w, h)   # bottom-aligned on the caption
        fig.text((x0 + cell_w / 2) / W, (y0 - 0.06) / H, titles[k + 1], ha="center", va="top",
                 fontsize=style.SMALL)
    return style.save(fig, "fig3_worked_example", png_dpi=400)


if __name__ == "__main__":
    main()
