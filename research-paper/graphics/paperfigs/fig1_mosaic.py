"""Figure 1. Corpus variety mosaic: the 14 prepared crops in chronological order, all drawn at one common
pixel scale (native aspect ratio and relative size preserved). Year and figure type come from
the rights and item records; panels outside the scored test set are marked with a dagger."""
from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

from . import data, style

# Short titles (editorial abbreviations of the recorded titles; author only where the rights record names one).
SHORT_TITLE = {
    "hmb_75c22eeb0cf610c1": "Proclus, In primum Euclidis",
    "hmb_42c5af403cf570cd": "Ramus, The Way to Geometry",
    "hmb_491810cb0ca5a7da": "Clairaut, Elementos de geometria",
    "hmb_4f0c47442b557c40": "Nattes's Practical Geometry",
    "hmb_26ecf2913d35c65f": "The First Six Books of Euclid",
    "hmb_5b3d06093d1b8902": "Elements of Analytical Geometry",
    "hmb_00f7d4fd9d34e34a": "Nouv. Ann. Math., 1872",
    "hmb_7ff916e03626fc21": "Cremona, Géométrie projective",
    "hmb_c89f03aedcca4f50": "Halsted, Elements of Geometry",
    "hmb_2cc47734c97d4b03": "Hardy, Elements of Analytic Geometry",
    "hmb_26a06dff91245fa9": "Archimedes, Geometrical Solutions",
    "hmb_7aa417baf3985cbc": "Киселёв, Элементарная геометрия",
    "hmb_8f5e92297c2c46ed": "Analytic Geometry, with Calculus",
    "hmb_088039a3639b4026": "Schoenflies & Dehn, Analyt. Geometrie",
}
TYPE_NAMES = {"geometry": "Geometry", "graph_plot": "Graph or plot", "relational_schematic": "Relational schematic",
              "number_line_scale": "Number line or scale", "table_grid": "Table or grid", "mixed_other": "Mixed"}
ROWS = (5, 5, 4)
UNSCORED_MARK = "†"


def _record(item_id: str) -> dict | None:
    try:
        return data.item_record(item_id)[1]
    except FileNotFoundError:
        return None


def panels() -> list[dict]:
    out = []
    with open(data.MEDIA_MANIFEST, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rec = _record(r["item_id"])
            img = Image.open(data.PAPER / "media" / r["file"])
            out.append({
                "id": r["item_id"], "img": np.asarray(img.convert("RGB")), "w": img.width, "h": img.height,
                "year": int(r["year"]),
                "type": TYPE_NAMES[rec["automatic_metadata"]["diagram_type"]["label"] if rec else r["diagram_type"]],
                "title": SHORT_TITLE[r["item_id"]], "scored": rec is not None and data.is_scored(r["item_id"]),
            })
    return sorted(out, key=lambda p: p["year"])


def main() -> list:
    style.apply()
    ps = panels()
    fs = style.SMALL - 0.5
    line_h = 1.25 * fs / 72
    label_h = 2 * line_h + 0.05
    gap_x, gap_y = 0.10, 0.10
    probe = plt.figure()
    r = probe.canvas.get_renderer()  # type: ignore[attr-defined]  (Agg canvas)
    for p in ps:
        p["lines"] = [f"{p['year']}{UNSCORED_MARK if not p['scored'] else ''} · {p['type']}", p["title"]]
        p["label_w"] = max(probe.text(0, 0, s, fontsize=fs, fontweight=style.EMPH if i == 0 else style.BODY)
                           .get_window_extent(r).width / probe.dpi for i, s in enumerate(p["lines"]))
    plt.close(probe)

    rows, k = [], 0
    for n in ROWS:
        rows.append(ps[k:k + n])
        k += n

    def row_width(row, s):
        return sum(max(p["w"] * s, p["label_w"]) for p in row) + gap_x * (len(row) - 1)

    if max(row_width(r, 0) for r in rows) > style.FULL_W:
        raise RuntimeError("labels alone are wider than the page; shorten a title or move a panel")
    lo, hi = 1e-4, 0.05
    for _ in range(60):
        s = (lo + hi) / 2
        lo, hi = (s, hi) if max(row_width(r, s) for r in rows) <= style.FULL_W else (lo, s)
    s = lo

    heights = [max(p["h"] for p in r) * s + label_h for r in rows]
    H = sum(heights) + gap_y * (len(rows) - 1)
    fig = plt.figure(figsize=(style.FULL_W, H))
    y_top = H
    for r, rh in zip(rows, heights):
        x = (style.FULL_W - row_width(r, s)) / 2
        img_h = rh - label_h
        for p in r:
            cw = max(p["w"] * s, p["label_w"])
            w, h = p["w"] * s, p["h"] * s
            x0, y0 = x + (cw - w) / 2, y_top - img_h  # bottom-aligned above the label
            ax = fig.add_axes((x0 / style.FULL_W, y0 / H, w / style.FULL_W, h / H))
            ax.imshow(p["img"], interpolation="antialiased", aspect="auto")
            ax.set_axis_off()
            ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                   edgecolor=style.FRAME, lw=0.4, clip_on=False))
            cx = x + cw / 2
            for i, line in enumerate(p["lines"]):
                fig.text(cx / style.FULL_W, (y_top - img_h - 0.04 - i * line_h) / H, line,
                         ha="center", va="top", fontsize=fs,
                         color=style.INK if i == 0 else style.INK_2,
                         fontweight=style.EMPH if i == 0 else style.BODY)
            x += cw + gap_x
        y_top -= rh + gap_y
    return style.save(fig, "fig1_mosaic", png_dpi=400)


if __name__ == "__main__":
    main()
