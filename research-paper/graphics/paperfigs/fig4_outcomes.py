"""Figure 4. Outcome decomposition: one stacked bar per model, Track A and Track B facets."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from . import data, style

# Outcome codes of the score report -> article categories. Every code not listed is "Other"
# (missing glyph, no TikZ, timeout, oversize, empty/raster/multipage output, resource, sandbox, multi-candidate).
CATEGORY = {"OK": "Valid drawing", "C_ERROR": "Compile error", "E_TRUNCATED": "Output limit reached",
            "P_POLICY_REJECT": "Policy rejection"}


def counts(track: str) -> dict[str, dict[str, int]]:
    out = {}
    for m, v in data.scores()["tables"][track]["models"].items():
        dist = v["validity"]["outcome_distribution"]
        c = {k: 0 for k in style.OUTCOME_COLORS}
        for code, n in dist["counts"].items():
            c[CATEGORY.get(code, "Other")] += n
        if sum(c.values()) != dist["n"]:
            raise RuntimeError(f"{m} {track}: categories do not sum to n")
        out[m] = c
    return out


def main() -> list:
    style.apply()
    order = data.model_order("A")
    fig, axes = plt.subplots(1, 2, figsize=(style.FULL_W, 1.95), sharey=True,
                             gridspec_kw={"wspace": 0.17})
    for ax, track in zip(axes, "AB"):
        c = counts(track)
        n = data.scores()["tables"][track]["N"]
        for y, m in enumerate(order):
            left = 0
            for cat, col in style.OUTCOME_COLORS.items():
                w = c[m][cat]
                ax.barh(y, w, left=left, height=0.62, color=col, edgecolor="white", linewidth=0.5,
                        label=cat if y == 0 else None)
                left += w
            valid = data.scores()["tables"][track]["models"][m]["validity"]["Valid@1"]
            ax.text(n + n * 0.012, y, f"{100 * valid:.1f}%", va="center", ha="left",
                    fontsize=style.SMALL, color=style.INK)
        ax.set_xlim(0, n)
        ax.set_ylim(len(order) - 0.5, -0.5)
        ax.set_yticks(range(len(order)), [style.MODEL_NAMES[m] for m in order])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
        ax.set_xlabel(f"Items (N = {n:,})")
        ax.set_title(style.TRACK_NAMES[track], loc="left", pad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.22),
               handlelength=1.0, handleheight=0.8, columnspacing=1.4)
    return style.save(fig, "fig4_outcomes")


if __name__ == "__main__":
    main()
