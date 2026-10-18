"""Figure 5 (outline r1). What is recovered, Track A: per-class recall R^c, relation-tier recall, Q5 and the
contradicted share, one row per model. Class recall, Q5 and the contradicted share come from the score report;
tier recall from research-paper/analysis/revision_numbers.json (`relation_tiers`). Marked exact equalities
(35-40 items per model) are too sparse for a rate and stay in the text as counts."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from . import data, style

CLASSES = {"E": ("Entity", style.CAT[0]), "R": ("Relation", style.CAT[1]), "L": ("Label", style.CAT[2])}
TIERS = {"coexact": ("Co-exact", style.CAT[3]), "depicted_exact": ("Depicted exact", style.CAT[5])}
CONTRA = {"R": ("Relation", style.CAT[1]), "L": ("Label", style.CAT[2])}
MS = 3.6


def _dots(ax, order, series, colors):
    """One row per model; each series sits at its own small vertical offset so near-equal values stay visible."""
    step = 0.26 if len(series) == 2 else 0.2
    offs = [step * (k - (len(series) - 1) / 2) for k in range(len(series))]
    for y, m in enumerate(order):
        for k, dy in zip(series, offs):
            ax.plot(series[k][m], y + dy, "o", ms=MS, color=colors[k], mec="white", mew=0.5, zorder=3)


def _legend(ax, entries):
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ms=MS, mec="white", mew=0.5, lw=0, label=lab)
                       for lab, c in entries],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=len(entries), handlelength=0.8,
              handletextpad=0.3, columnspacing=0.9, borderaxespad=0.2)


def main() -> list:
    style.apply()
    rep = data.scores()["tables"]["A"]["models"]
    tiers = data.revision()["relation_tiers"]
    order = data.model_order("A")

    cls = {c: {m: rep[m]["per_class"]["recall"][c]["R"] for m in order} for c in CLASSES}
    tier = {t: {m: tiers[f"{data.SHORT[m]}|A"][t]["R"] for m in order} for t in TIERS}
    q5 = {m: rep[m]["per_class"]["Q5"]["Q5"] for m in order}
    contra = {c: {m: 100 * rep[m]["per_class"]["recall"][c]["contradicted_share"] for m in order} for c in CONTRA}

    fig, axes = plt.subplots(1, 4, figsize=(style.FULL_W, 2.05), sharey=True,
                             gridspec_kw={"wspace": 0.16, "width_ratios": [1.25, 1.25, 0.8, 0.8]})
    a_cls, a_tier, a_q5, a_con = axes

    _dots(a_cls, order, cls, {c: v[1] for c, v in CLASSES.items()})
    _dots(a_tier, order, tier, {t: v[1] for t, v in TIERS.items()})
    for ax in (a_cls, a_tier):
        ax.set_xlim(0, 0.7)
        ax.set_xticks([0, 0.2, 0.4, 0.6])
        ax.set_xlabel("Recall")
    a_cls.set_title("Class recall", loc="left", pad=13)
    _legend(a_cls, list(CLASSES.values()))
    a_tier.set_title("Relation tiers", loc="left", pad=13)
    _legend(a_tier, list(TIERS.values()))

    for y, m in enumerate(order):
        a_q5.barh(y, 100 * q5[m], height=0.55, color=style.NEUTRAL_EDGE, lw=0)
        a_q5.text(100 * q5[m] + 1.2, y, f"{100 * q5[m]:.0f}", va="center", ha="left",
                  fontsize=style.SMALL, color=style.INK)
    a_q5.set_xlim(0, 50)
    a_q5.set_xticks([0, 20, 40])
    a_q5.set_xlabel("Drawings (%)")
    a_q5.set_title("Q5", loc="left", pad=13)

    _dots(a_con, order, contra, {c: v[1] for c, v in CONTRA.items()})
    a_con.set_xlim(0, 5)
    a_con.set_xticks([0, 1, 2, 3, 4, 5])
    a_con.set_xlabel("Checkable facts (%)")
    a_con.set_title("Contradicted", loc="left", pad=13)
    _legend(a_con, list(CONTRA.values()))

    for ax in axes:
        ax.xaxis.grid(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    a_cls.set_yticks(range(len(order)), [style.MODEL_NAMES[m] for m in order])
    a_cls.set_ylim(len(order) - 0.5, -0.5)
    return style.save(fig, "fig5_class_recall")


if __name__ == "__main__":
    main()
