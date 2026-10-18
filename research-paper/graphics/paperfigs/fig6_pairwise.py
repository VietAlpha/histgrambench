"""Figure 6. Pairwise separation, Track A and Track B.
Lower triangle: H difference (row minus column). Upper triangle: BH-adjusted p-value, with markers for
pairs separated at the report's FDR level q and for |difference| above the registered margin delta_star."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.ticker import FuncFormatter

from . import data, style

SEP_MARK = "*"
MARGIN_MARK = "†"


def pair(po: dict, a: str, b: str) -> tuple[float, dict]:
    """Difference H(a) - H(b) and the stored pair record, whichever key order the report used."""
    if f"{a}|{b}" in po["pairs"]:
        r = po["pairs"][f"{a}|{b}"]
        return r["diff"], r
    r = po["pairs"][f"{b}|{a}"]
    return -r["diff"], r


def fmt_p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def main() -> list:
    style.apply()
    rep = data.scores()
    order = data.model_order("A")
    names = [style.MODEL_NAMES[m] for m in order]
    delta_star = data.thresholds().value("delta_star")
    n = len(order)

    pos = [abs(r["diff"]) for t in "AB" for r in rep["tables"][t]["partial_order"]["pairs"].values()]
    vmax = float(np.ceil(max(pos) * 10) / 10)
    cmap = LinearSegmentedColormap.from_list("hmb_div", style.DIVERGING)
    norm = Normalize(-vmax, vmax)

    fig, axes = plt.subplots(1, 2, figsize=(style.FULL_W, 3.15), gridspec_kw={"wspace": 0.08})
    qs = set()
    for ax, track in zip(axes, "AB"):
        po = rep["tables"][track]["partial_order"]
        if po["family_size"] != n * (n - 1) // 2 or len(po["pairs"]) != po["family_size"]:
            raise RuntimeError(f"Track {track}: pair family incomplete")
        qs.add(po["q"])
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                x, y = j, i
                if i == j:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color="#ececec", lw=0))
                    continue
                d, r = pair(po, a, b)
                if i > j:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, color=cmap(norm(d)), lw=0))
                    ink = "white" if abs(d) > 0.6 * vmax else style.INK
                    ax.text(x, y, f"{d:+.2f}".replace("-", "−"), ha="center", va="center",
                            fontsize=style.SMALL, color=ink)
                else:
                    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, facecolor="white",
                                               edgecolor="none"))
                    marks = (SEP_MARK if r["separated"] else "") + (MARGIN_MARK if abs(d) > delta_star else "")
                    ax.text(x, y, fmt_p(r["bh_adjusted"]) + (f"\n{marks}" if marks else "\n"),
                            ha="center", va="center", fontsize=style.SMALL - 0.5, color=style.INK,
                            linespacing=1.15)
        for k in range(n + 1):
            ax.plot([-0.5, n - 0.5], [k - 0.5, k - 0.5], color=style.GRID, lw=0.5, zorder=3)
            ax.plot([k - 0.5, k - 0.5], [-0.5, n - 0.5], color=style.GRID, lw=0.5, zorder=3)
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(n), names, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_yticks(range(n), names if track == "A" else [])
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(style.TRACK_NAMES[track], loc="left", pad=4)

    if len(qs) != 1:
        raise RuntimeError(f"tracks use different q: {qs}")
    q = qs.pop()
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cax = fig.add_axes((0.935, 0.30, 0.012, 0.52))
    cb = fig.colorbar(sm, cax=cax)
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(width=0.5, length=2)
    cb.set_label("H difference (row − column)", fontsize=style.SMALL)
    cb.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:+.1f}".replace("-", "−")
                                                      if v else "0"))
    fig.text(0.5, -0.12,
             f"Upper triangle: BH-adjusted p.   "
             f"{SEP_MARK} separated at FDR q = {q:.2f}   {MARGIN_MARK} |difference| > δ* = {delta_star:.2f}",
             ha="center", va="top", fontsize=style.SMALL, color=style.INK_2)
    return style.save(fig, "fig6_pairwise")


if __name__ == "__main__":
    main()
