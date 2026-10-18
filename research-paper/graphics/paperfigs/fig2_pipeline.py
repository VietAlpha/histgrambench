"""Figure 2. Benchmark pipeline in three lanes (corpus, reference, evaluation) with funnel counts and the
geometric-check / judge-panel fork. Accent boxes are model-generated steps, neutral boxes deterministic code.
Counts: funnel report (S0, S1, S2, S4, S5), test item records, the score report (N, G, answers) and the
eligible facts of the scored items' answer keys (route split by eligible_geometric/model counts)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from . import data, style

W, H = style.FULL_W, 5.0
BOX_H = 0.27


def counts() -> dict:
    f = data.funnel()
    rep = data.scores()
    scored = data.scored_items()
    geo = sum(d["semantic_card"]["eligible_geometric_count"] for d in scored)
    judge = sum(d["semantic_card"]["eligible_model_count"] for d in scored)
    mgs = sum(sum(d["semantic_card"]["eligible_mgs_count"].values()) for d in scored)
    if mgs != geo + judge:
        raise RuntimeError("fact class counts and route counts disagree")
    N, G = rep["tables"]["A"]["N"], rep["gate_evidence"]["G7"]["G"]
    answers = sum(v["validity"]["outcome_distribution"]["n"] for t in rep["tracks"]
                  for v in rep["tables"][t]["models"].values())
    return {
        "books": f["S0_books_cleared"], "pages": f["S1_pages_rendered"],
        "proposals": f["overall"]["S2"]["n"], "screened": f["overall"]["S4"]["n"],
        "representatives": f["overall"]["S5"]["n"], "test": len(data.items("test")),
        "N": N, "G": G, "books_scored": len({d["source"]["book_id"] for d in scored}),
        "facts": geo + judge, "mean": rep["gate_evidence"]["G2"]["F_bar"], "geo": geo, "judge": judge,
        "models": len(rep["models"]), "tracks": len(rep["tracks"]), "answers": answers,
    }


def main() -> list:
    style.apply()
    c = counts()
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_axis_off()

    def box(x, y, text, model=False, w=1.55, bold=False):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - BOX_H / 2), w, BOX_H,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=style.ACCENT_FILL if model else style.NEUTRAL_FILL,
                                    ec=style.ACCENT if model else style.NEUTRAL_EDGE, lw=0.7))
        ax.text(x, y, text, ha="center", va="center", fontsize=style.SMALL, color=style.INK,
                fontweight=style.EMPH if bold else style.BODY, linespacing=1.1)

    def arrow(p, q, conn="arc3"):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=style.INK_2,
                                     shrinkA=0, shrinkB=0, connectionstyle=conn))

    def path_arrow(pts):
        for a, b in zip(pts[:-2], pts[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=style.INK_2, lw=0.7, solid_capstyle="butt")
        arrow(pts[-2], pts[-1])

    renderer = fig.canvas.get_renderer()  # type: ignore[attr-defined]  (Agg canvas)

    def count(x, y, num, unit):
        t = ax.text(x + 0.08, y, f"{num}", ha="left", va="center", fontsize=style.SMALL, fontweight=style.EMPH)
        w = t.get_window_extent(renderer).width / fig.dpi
        ax.text(x + 0.08 + w + 0.05, y, unit, ha="left", va="center", fontsize=style.SMALL,
                color=style.INK_2, linespacing=1.1)

    lanes = [(1.15, "Corpus"), (3.55, "Reference"), (5.85, "Evaluation")]
    top = H - 0.12
    for k, (x, name) in enumerate(lanes):
        ax.text(x, top, f"{k + 1}  {name}", ha="center", va="top", fontsize=style.FONT_SIZE, fontweight=style.EMPH)
    for xd in (2.35, 4.72):
        ax.plot([xd, xd], [0.35, top], color=style.GRID, lw=0.6)

    # Lane 1: corpus funnel.
    x1 = lanes[0][0]
    y = top - 0.42
    ax.text(x1, y, f"{c['books']:,} public-domain books", ha="center", va="center", fontsize=style.SMALL,
            fontweight=style.EMPH)
    steps = [("Page rendering", False, (c["pages"], "pages")),
             ("Figure detection (PaddleOCR-VL)", True, (c["proposals"], "figure proposals")),
             ("Automatic image checks", False, None),
             ("Screening model", True, (c["screened"], "passing screening")),
             ("Near-duplicate reduction", False, (c["representatives"], "representatives")),
             ("Source-group split, test selection", False, (c["test"], "test items"))]
    step = 0.58
    prev = y - 0.08
    for text, model, out in steps:
        y -= step
        arrow((x1, prev), (x1, y + BOX_H / 2))
        box(x1, y, text, model, w=1.9)
        prev = y - BOX_H / 2
        if out and out[1] != "test items":
            count(x1, y - BOX_H / 2 - (step - BOX_H) / 2, f"{out[0]:,}", out[1])
    y_l1_end = y - BOX_H / 2 - (step - BOX_H) - 0.02
    ax.plot([x1, x1], [prev, y_l1_end], color=style.INK_2, lw=0.7)
    lane1_out = (x1, y_l1_end)
    y_lt = prev

    # Lane 2: reference construction.
    x2 = lanes[1][0]
    y = top - 0.62
    bus = y + BOX_H / 2 + 0.16
    for a, b in (((x1, y_l1_end), (2.35, y_l1_end)), ((2.35, y_l1_end), (2.35, bus)),
                 ((2.35, bus), (x2 + 0.5, bus))):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=style.INK_2, lw=0.7)
    box(x2 - 0.5, y, "Writer: GPT-6.1 Sol", True, w=0.95)
    box(x2 + 0.5, y, "Writer: Kimi K2.5", True, w=0.95)
    for dx in (-0.5, 0.5):
        arrow((x2 + dx, bus), (x2 + dx, y + BOX_H / 2))
    y_w = y
    lane2 = [("Entity alignment", False), ("Cross-vote by both writers", True),
             ("Unanimity rule (Eq. 1)", False), ("Figure-level key checks", False)]
    step2 = 0.5
    for i, (text, model) in enumerate(lane2):
        y -= step2
        if i == 0:
            for dx in (-0.5, 0.5):
                ax.plot([x2 + dx, x2 + dx], [y_w - BOX_H / 2, y_w - BOX_H / 2 - 0.08], color=style.INK_2, lw=0.7)
            ax.plot([x2 - 0.5, x2 + 0.5], [y_w - BOX_H / 2 - 0.08] * 2, color=style.INK_2, lw=0.7)
            arrow((x2, y_w - BOX_H / 2 - 0.08), (x2, y + BOX_H / 2))
        else:
            arrow((x2, y + step2 - BOX_H / 2), (x2, y + BOX_H / 2))
        box(x2, y, text, model, w=1.75)
    y -= 0.85
    arrow((x2, y + 0.85 - BOX_H / 2), (x2, y + 0.24))
    count(x2, y + 0.24 + (0.85 - BOX_H / 2 - 0.24) / 2, f"{c['N']:,}", f"scored items\nin {c['G']} groups")
    ref_y = y
    ax.add_patch(FancyBboxPatch((x2 - 0.95, ref_y - 0.24), 1.9, 0.48, boxstyle="round,pad=0,rounding_size=0.04",
                                fc="white", ec=style.INK, lw=0.9))
    ax.text(x2, ref_y + 0.08, "Reference", ha="center", va="center", fontsize=style.SMALL, fontweight=style.EMPH)
    ax.text(x2, ref_y - 0.1, f"{c['facts']:,} facts  (mean {c['mean']:.1f} per item)", ha="center",
            va="center", fontsize=style.SMALL)

    # Lane 3: evaluation.
    x3 = lanes[2][0]
    y = top - 0.42
    box(x3, y, f"Contestant answer (TikZ)", True, w=1.7)
    ax.text(x3, y - BOX_H / 2 - 0.08,
            f"{c['models']} models × {c['tracks']} tracks × {c['N']:,} items = {c['answers']:,}",
            ha="center", va="top", fontsize=style.SMALL - 0.5, color=style.INK_2)
    lane3 = [("Extraction", False), ("Sandbox compilation", False), ("Vector geometry", False),
             ("Entity binding", False)]
    step3 = 0.44
    prev = y - BOX_H / 2
    y -= 0.12
    for text, model in lane3:
        y -= step3
        arrow((x3, prev if text != "Extraction" else prev - 0.2), (x3, y + BOX_H / 2))
        box(x3, y, text, model, w=1.7)
        prev = y - BOX_H / 2
    bind_y = y
    path_arrow([(x2 + 0.95, ref_y), (4.72, ref_y), (4.72, bind_y), (x3 - 0.85, bind_y)])

    fork_y = bind_y - 0.66
    for dx, label, model, n, share in ((-0.55, "Geometric check", False, c["geo"], c["geo"] / c["facts"]),
                                       (0.55, "Judge panel", True, c["judge"], c["judge"] / c["facts"])):
        arrow((x3, prev), (x3 + dx, fork_y + 0.23), conn="arc3")
        ax.add_patch(FancyBboxPatch((x3 + dx - 0.5, fork_y - 0.23), 1.0, 0.46,
                                    boxstyle="round,pad=0,rounding_size=0.04",
                                    fc=style.ACCENT_FILL if model else style.NEUTRAL_FILL,
                                    ec=style.ACCENT if model else style.NEUTRAL_EDGE, lw=0.7))
        ax.text(x3 + dx, fork_y + 0.11, label, ha="center", va="center", fontsize=style.SMALL)
        ax.text(x3 + dx, fork_y - 0.04, f"{100 * share:.2f}%", ha="center", va="center",
                fontsize=style.FONT_SIZE + 1, fontweight=style.EMPH)
        ax.text(x3 + dx, fork_y - 0.17, f"{n:,} facts", ha="center", va="center", fontsize=style.SMALL - 0.5,
                color=style.INK_2)
    y = fork_y - 0.56
    for dx in (-0.55, 0.55):
        arrow((x3 + dx, fork_y - 0.23), (x3, y + BOX_H / 2))
    box(x3, y, "Item score S", False, w=1.3)
    y2 = y - 0.42
    arrow((x3, y - BOX_H / 2), (x3, y2 + BOX_H / 2))
    box(x3, y2, f"{c['G']} group means", False, w=1.3)
    y3 = y2 - 0.42
    arrow((x3, y2 - BOX_H / 2), (x3, y3 + BOX_H / 2))
    box(x3, y3, "Benchmark score H", False, w=1.3, bold=True)

    ax.legend(handles=[Patch(fc=style.ACCENT_FILL, ec=style.ACCENT, lw=0.7, label="Model-generated step"),
                       Patch(fc=style.NEUTRAL_FILL, ec=style.NEUTRAL_EDGE, lw=0.7, label="Deterministic step")],
              loc="lower left", bbox_to_anchor=(0.01, 0.0), ncol=2, handlelength=1.6, handleheight=0.9)
    count(x1, (lane1_out[1] + y_lt) / 2, f"{c['test']:,}", "test items")
    return style.save(fig, "fig2_pipeline")


if __name__ == "__main__":
    main()
