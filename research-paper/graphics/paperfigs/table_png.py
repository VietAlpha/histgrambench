"""PNG previews of the generated tables in research-paper/graphics/out/latex_tables/, typeset with the article's own
preamble (build_latex.PREAMBLE) so a preview matches the compiled paper. Two images per table: the table alone and
the table with its numbered caption.

pdflatex runs in the `texlive/texlive:latest-medium` Docker image (no TeX on the Windows host); PyMuPDF crops each
page to its ink and rasterises it at 600 dpi. Only the tables in NUMBERS are rendered (outline r1, Part D numbering).
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fitz

from . import data

GRAPHICS = Path(__file__).resolve().parents[1]
OUT = GRAPHICS / "out" / "latex_tables"
IMAGE = "texlive/texlive:latest-medium"
DPI = 600
PAD_PT = 4
NUMBERS = {"table3_composition": "3", "table5_stability": "5", "table6_strata": "6", "table7_page_text": "7",
           "tableS2_exclusions": "S2", "tableS3_outcomes": "S3", "tableS4_families": "S4",
           "tableS5_key_retention": "S5"}


def _preamble() -> str:
    sys.path.insert(0, str(data.PAPER))
    from build_latex import PREAMBLE
    return PREAMBLE.split(r"\title{")[0] + "\\pagestyle{empty}\n\\begin{document}\n"


def _page(stem: str, number: str, caption: bool) -> str:
    prefix, n = (number[0], number[1:]) if number.startswith("S") else ("", number)
    hide = "" if caption else "\\renewcommand{\\caption}[1]{}\n"
    return (f"\\setcounter{{table}}{{{int(n) - 1}}}\\renewcommand{{\\thetable}}{{{prefix}\\arabic{{table}}}}\n"
            f"{{{hide}\\input{{{stem}}}}}\n\\clearpage\n")


def _ink_rect(page: fitz.Page) -> fitz.Rect:
    r = fitz.Rect()
    for b in page.get_text("blocks"):
        r |= fitz.Rect(b[:4])
    for d in page.get_drawings():
        r |= d["rect"]
    return r + (-PAD_PT, -PAD_PT, PAD_PT, PAD_PT)


def main() -> list[Path]:
    OUT.mkdir(parents=True, exist_ok=True)
    stems = list(NUMBERS)
    paths = []
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        for s in stems:
            shutil.copy(GRAPHICS / f"{s}.tex", work / f"{s}.tex")
        body = "".join(_page(s, NUMBERS[s], cap) for s in stems for cap in (False, True))
        (work / "tables.tex").write_text(_preamble() + body + "\\end{document}\n", encoding="utf-8")
        r = subprocess.run(["docker", "run", "--rm", "-v", f"{work}:/work", "-w", "/work", IMAGE,
                            "pdflatex", "-interaction=nonstopmode", "-halt-on-error", "tables.tex"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace",
                           env={**os.environ, "MSYS_NO_PATHCONV": "1"})
        if r.returncode:
            raise RuntimeError("pdflatex failed:\n" + r.stdout[-3000:])
        log = (work / "tables.log").read_text(encoding="utf-8", errors="replace")
        over = sorted({re.findall(r"\(\./(\w+)\.tex", log[:m.start()])[-1] + ": " + m.group(1)
                       for m in re.finditer(r"Overfull \\hbox \(([\d.]+pt) too wide\)", log)})
        if over:
            raise RuntimeError("overfull box in a table: " + "; ".join(over))
        doc = fitz.open(work / "tables.pdf")
        if doc.page_count != 2 * len(stems):
            raise RuntimeError(f"{doc.page_count} pages for {len(stems)} tables; a table spilled over a page")
        for k, page in enumerate(doc):
            s = stems[k // 2]
            p = OUT / (f"{s}_with_caption.png" if k % 2 else f"{s}.png")
            page.get_pixmap(dpi=DPI, clip=_ink_rect(page)).save(p)
            paths.append(p)
        doc.close()
    return paths


if __name__ == "__main__":
    main()
