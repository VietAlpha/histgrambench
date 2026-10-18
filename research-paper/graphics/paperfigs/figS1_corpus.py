"""Figure S1. Scored test items by half-century of publication, stacked by source repository,
with the share of items from pages without running text (item flag automatic_metadata.figure_only_page)."""
from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

from . import data, style


def main() -> list:
    style.apply()
    books = data.rights()
    rows = []
    for d in data.scored_items():
        b = books[d["source"]["book_id"]]
        rows.append((b["publication_year"] // 50 * 50, data.repository(b),
                     bool(d["automatic_metadata"]["figure_only_page"])))

    bins = list(range(min(r[0] for r in rows), max(r[0] for r in rows) + 50, 50))
    repos = [r for r, _ in Counter(r[1] for r in rows).most_common()]
    if len(repos) > len(style.CAT):
        raise RuntimeError("more repositories than validated categorical slots")
    cnt = Counter((r[0], r[1]) for r in rows)
    tot = Counter(r[0] for r in rows)
    fig_only = Counter(r[0] for r in rows if r[2])

    fig, (ax, axs) = plt.subplots(2, 1, figsize=(style.COL_W + 0.6, 3.1), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1], "hspace": 0.12})
    x = range(len(bins))
    bottom = [0] * len(bins)
    for repo, col in zip(repos, style.CAT):
        h = [cnt[(b, repo)] for b in bins]
        ax.bar(x, h, bottom=bottom, width=0.7, color=col, edgecolor="white", linewidth=0.5, label=repo)
        bottom = [a + c for a, c in zip(bottom, h)]
    for i, b in enumerate(bins):
        ax.text(i, tot[b] + 15, f"{tot[b]:,}", ha="center", va="bottom", fontsize=style.SMALL)
    ax.set_ylabel("Scored items")
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    ax.set_ylim(0, max(tot.values()) * 1.12)
    ax.yaxis.grid(True)
    ax.legend(loc="upper left", ncol=1, handlelength=1.0, handleheight=0.8)
    ax.set_title(f"N = {len(rows):,} scored items", loc="left", fontsize=style.SMALL, color=style.INK_2, pad=2)

    share = [100 * fig_only[b] / tot[b] if tot[b] else None for b in bins]
    xs = [i for i, s in zip(x, share) if s is not None]
    axs.plot(xs, [s for s in share if s is not None], "o", ms=3.5, color=style.INK, mec="white", mew=0.5)
    axs.set_ylim(-10, 110)
    axs.spines["bottom"].set_position(("data", -10))
    axs.set_yticks([0, 50, 100])
    axs.yaxis.grid(True)
    axs.set_ylabel("Figure-only\npage (%)")
    axs.set_xticks(list(x), [f"{b}–{b + 49}" for b in bins], rotation=30, ha="right", rotation_mode="anchor")
    axs.set_xlabel("Publication period")
    return style.save(fig, "figS1_corpus")


if __name__ == "__main__":
    main()
