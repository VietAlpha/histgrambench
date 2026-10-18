"""Regenerate the article figures into research-paper/graphics/out/.

Usage (from the repository root):
    .venv/Scripts/python.exe research-paper/graphics/make_figures.py            # all figures
    .venv/Scripts/python.exe research-paper/graphics/make_figures.py fig4 fig6  # a subset
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

FIGURES = {
    "fig1": "paperfigs.fig1_mosaic",
    "fig2": "paperfigs.fig2_pipeline",
    "fig3": "paperfigs.fig3_worked_example",
    "fig4": "paperfigs.fig4_outcomes",
    "fig5": "paperfigs.fig5_page_text",
    "fig5_recall": "paperfigs.fig5_class_recall",
    "fig6": "paperfigs.fig6_pairwise",
    "figS1": "paperfigs.figS1_corpus",
    "tables": "paperfigs.tables",
    "table_png": "paperfigs.table_png",   # needs Docker; run after "tables"
}


def main(argv: list[str]) -> int:
    names = argv or list(FIGURES)
    unknown = [n for n in names if n not in FIGURES]
    if unknown:
        print(f"unknown figure(s): {unknown}; choose from {list(FIGURES)}")
        return 2
    for n in names:
        paths = importlib.import_module(FIGURES[n]).main()
        print(n, *[p.name for p in paths])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
