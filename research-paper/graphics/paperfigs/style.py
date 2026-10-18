"""Shared publication style: fonts, sizes, colours, model display names, export."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

OUT_DIR = Path(__file__).resolve().parents[1] / "out"
# Source Sans 3 (SIL OFL, fonts/LICENSE.md) is vendored so every machine renders the same typeface.
FONT_DIR = Path(__file__).resolve().parents[1] / "fonts"
FONT = "Source Sans 3"

# Page geometry (inches): two-column paper, full width and single column.
FULL_W = 7.0
COL_W = 3.35

# Source Sans has a small x-height, so it is set half a point above the usual 7/6 pt.
FONT_SIZE = 7.5
SMALL = 6.5
BODY = "medium"       # base weight; Regular reads too thin at 6-7 pt
EMPH = "bold"         # the one emphasis weight, a step above the medium body text

# Ink and structure.
INK = "#1a1a1a"
INK_2 = "#4d4d4d"
MUTED = "#8c8c8c"
GRID = "#dedede"
FRAME = "#c4c4c4"
NEUTRAL_FILL = "#f3f3f2"
NEUTRAL_EDGE = "#6b6b6b"

# One accent, used for "model-generated" in Fig. 2.
ACCENT = "#2a78d6"
ACCENT_FILL = "#e3eefb"

# Categorical palette (validated with the dataviz validator, light surface;
# adjacent-pair CVD dE >= 9.1, normal-vision dE >= 19.6).
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

# Fig. 4 outcome categories, in stacking order. Adjacent hues validated
# (worst adjacent CVD dE 24.7); "Other" is a neutral, identified by legend.
OUTCOME_COLORS = {
    "Valid drawing": "#2a78d6",
    "Compile error": "#eb6834",
    "Output limit reached": "#4a3aa7",
    "Policy rejection": "#eda100",
    "Other": "#b9b7b0",
}

# Diverging poles (Fig. 6).
DIVERGING = ["#b8481a", "#f0a27c", "#f4f4f2", "#8fb6e6", "#1f5fae"]

# Official display names as used in the article (Section 5.1).
MODEL_NAMES = {
    "google.gemma-4-26b-a4b": "Gemma 4 26B",
    "us.meta.llama4-maverick-17b-instruct-v1_0": "Llama 4 Maverick",
    "mistral.ministral-3-14b-instruct": "Ministral 3 14B",
    "google.gemma-4-e2b": "Gemma 4 E2B",
    "global.anthropic.claude-haiku-4-5-20251001-v1_0": "Claude Haiku 4.5",
    "nvidia.nemotron-nano-12b-v2": "Nemotron Nano 12B v2",
}

TRACK_NAMES = {"A": "Track A (image only)", "B": "Track B (image + page text)"}


def apply() -> None:
    for f in sorted(FONT_DIR.glob("*.ttf")):
        font_manager.fontManager.addfont(str(f))
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.weight": BODY,
        "font.sans-serif": [FONT, "Arial", "DejaVu Sans"],
        "mathtext.fontset": "custom",
        "mathtext.rm": FONT,
        "mathtext.it": FONT,
        "mathtext.bf": f"{FONT}:semibold",
        "font.size": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "axes.titleweight": "semibold",
        "axes.labelweight": "medium",
        "axes.labelsize": FONT_SIZE,
        "xtick.labelsize": SMALL,
        "ytick.labelsize": SMALL,
        "legend.fontsize": SMALL,
        "text.color": INK,
        "axes.labelcolor": INK,
        "axes.edgecolor": INK_2,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
        "axes.grid": False,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "image.interpolation": "antialiased",
    })


def panel_label(ax, letter: str, x: float = -0.02, y: float = 1.02, **kw) -> None:
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=FONT_SIZE, fontweight=EMPH,
            ha="right", va="bottom", **kw)


def save(fig, name: str, png_dpi: int = 300) -> list[Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext, dpi in (("pdf", 600), ("svg", 600), ("png", png_dpi)):
        p = OUT_DIR / f"{name}.{ext}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight", pad_inches=0.02,
                    metadata={"Date": None} if ext in ("pdf", "svg") else None)
        paths.append(p)
    plt.close(fig)
    return paths
