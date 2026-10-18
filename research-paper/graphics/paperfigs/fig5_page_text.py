"""Figure 5. Effect of page text: per-model H in Track A and Track B, each with its 95% group-t interval.
Per-model difference intervals were never computed, so none are drawn; the pooled difference belongs in
the caption (score report context_effect.pooled)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import data, style

TRACK_COLOR = {"A": style.MUTED, "B": style.ACCENT}
OFFSET = {"A": -0.13, "B": 0.13}


def main() -> list:
    style.apply()
    rep = data.scores()
    order = data.model_order("A")
    H = {t: {m: rep["tables"][t]["models"][m]["headline"] for m in order} for t in "AB"}

    fig, ax = plt.subplots(figsize=(style.COL_W, 2.3))
    for y, m in enumerate(order):
        ax.plot([H["A"][m]["estimate"], H["B"][m]["estimate"]], [y + OFFSET["A"], y + OFFSET["B"]],
                color=style.FRAME, lw=0.6, zorder=1)
        for t in "AB":
            h = H[t][m]
            yy = y + OFFSET[t]
            ax.plot(h["ci"], [yy, yy], color=TRACK_COLOR[t], lw=1.0, zorder=2)
            ax.plot(h["estimate"], yy, "o", ms=3.6, color=TRACK_COLOR[t], mec="white", mew=0.5, zorder=3)

    ax.set_yticks(range(len(order)), [style.MODEL_NAMES[m] for m in order])
    ax.set_ylim(len(order) - 0.5, -0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(0, 0.7)
    ax.set_xlabel("H (95% interval)")
    ax.xaxis.grid(True)
    ax.legend(handles=[Line2D([], [], color=TRACK_COLOR[t], marker="o", ms=3.6, mec="white", mew=0.5, lw=1.0,
                              label=style.TRACK_NAMES[t]) for t in "AB"],
              loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, handlelength=1.6, columnspacing=1.2)
    return style.save(fig, "fig5_page_text")


if __name__ == "__main__":
    main()
