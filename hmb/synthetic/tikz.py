"""Clean TikZ for a known-geometry specification, the source-refactor variant, and response envelopes.

The emitted picture is a single `tikzpicture` (the §9.1 output contract); the harness wraps it with
wrapper_v1. `source_refactor` gives a byte-different source with the same render (inline coordinates,
`\\path[draw]` for `\\draw`), which is the §7.6 `source_refactor` preserving transform.

Envelopes (fixtures for the §9.2 extraction paths: prose plus fenced block, multiple blocks, truncated
tail, bare body):
  prose_fenced    prose, one ```latex block, closing prose                   -> extraction path fenced
  multi_block     a draft picture block, then the final block (last wins, §9.2) -> fenced, n_blocks 2
  truncated_tail  complete block, prose cut mid-word, stop_reason "length"   -> fenced (span complete)
  bare_body       the picture with no fence                                  -> bare_pic
"""
from __future__ import annotations

import math

import numpy as np

from hmb.hashing import sha256_bytes

from .spec import Spec, unit

ENVELOPE_KINDS = ("prose_fenced", "multi_block", "truncated_tail", "bare_body")
EXPECTED_PATH = {"none": "fenced", "prose_fenced": "fenced", "multi_block": "fenced",
                 "truncated_tail": "fenced", "bare_body": "bare_pic"}


def _f(x: float) -> str:
    s = f"{x:.5f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _xy(p) -> str:
    return f"({_f(float(p[0]))},{_f(float(p[1]))})"


def _tex_text(t: str) -> str:
    rep = {"\\": r"\textbackslash{}", "{": r"\{", "}": r"\}", "$": r"\$", "&": r"\&",
           "#": r"\#", "_": r"\_", "%": r"\%", "^": r"\^{}", "~": r"\~{}"}
    return "".join(rep.get(c, c) for c in t)


FONTS = {"rm": r"\rmfamily", "sf": r"\sffamily", "tt": r"\ttfamily"}


def emit_tikz(spec: Spec, *, refactor: bool | None = None) -> str:
    refactor = spec.render.get("refactor", False) if refactor is None else refactor
    st = spec.style
    draw = r"\path[draw" if refactor else r"\draw"
    fill = r"\path[fill" if refactor else r"\fill"

    def cmd(base: str, opts: list[str]) -> str:
        if refactor:
            return base + ("," + ",".join(opts) if opts else "") + "]"
        return base + ("[" + ",".join(opts) + "]" if opts else "")

    def P(ref) -> str:
        if isinstance(ref, str) and not refactor:
            return f"({ref})"
        return _xy(spec.pt(ref))

    opts = [f"line width={_f(st.get('line_width_pt', 0.4))}pt", f"color={st.get('color', 'black')}",
            f"font={FONTS.get(st.get('font', 'rm'), FONTS['rm'])}"]
    lines = [r"\begin{tikzpicture}[" + ", ".join(opts) + "]"]
    pad = spec.render.get("pad_cm", 0.0)
    if pad > 0:
        from .spec import Geo
        x0, y0, x1, y1 = Geo.of_spec(spec).bbox()
        lines.append(rf"\path {_xy((x0 - pad, y0 - pad))} rectangle {_xy((x1 + pad, y1 + pad))};")
    if not refactor:
        for e in spec.entities:
            if e.type == "point":
                lines.append(rf"\coordinate ({e.id}) at {_xy((e.geom['x'], e.geom['y']))};")
    # regions first (fills under strokes)
    for e in spec.entities:
        if e.type == "region":
            pts = " -- ".join(P(r) for r in e.geom["pts"])
            lines.append(cmd(fill, [f"{st.get('color', 'black')}!18"]) + f" {pts} -- cycle;")
    for e in spec.entities:
        g, t = e.geom, e.type
        dashed = ["dashed"] if e.style.get("dashed") else []
        if t in ("segment",):
            lines.append(cmd(draw, dashed) + f" {P(g['a'])} -- {P(g['b'])};")
        elif t == "line":
            a, b = spec.pt(g["a"]), spec.pt(g["b"])
            t0, t1 = g.get("ext", [0.0, 1.0])
            lines.append(cmd(draw, dashed) + f" {_xy(a + t0 * (b - a))} -- {_xy(a + t1 * (b - a))};")
        elif t == "ray":
            a, b = spec.pt(g["a"]), spec.pt(g["b"])
            lines.append(cmd(draw, dashed) + f" {P(g['a'])} -- {_xy(a + g.get('t1', 1.0) * (b - a))};")
        elif t in ("axis", "arrow"):
            lines.append(cmd(draw, ["->"] + dashed) + f" {P(g['a'])} -- {P(g['b'])};")
        elif t == "circle":
            lines.append(cmd(draw, dashed) + f" {P(g['c'])} circle[radius={_f(g['r'])}];")
        elif t == "arc":
            c = spec.pt(g["c"])
            start = c + g["r"] * np.array([math.cos(math.radians(g["t0"])), math.sin(math.radians(g["t0"]))])
            lines.append(cmd(draw, dashed) + f" {_xy(start)} arc[start angle={_f(g['t0'])}, "
                         f"end angle={_f(g['t1'])}, radius={_f(g['r'])}];")
        elif t == "ellipse":
            c = spec.pt(g["c"])
            lines.append(cmd(draw, dashed + [f"rotate around={{{_f(g['rot'])}:{_xy(c)}}}"])
                         + f" {_xy(c)} ellipse[x radius={_f(g['rx'])}, y radius={_f(g['ry'])}];")
        elif t == "polygon":
            lines.append(cmd(draw, dashed) + " " + " -- ".join(P(r) for r in g["v"]) + " -- cycle;")
        elif t == "curve":
            pts = " ".join(_xy(p) for p in g["pts"])
            lines.append(cmd(draw, dashed) + f" plot coordinates {{{pts}}};")
        elif t == "angle_marker":
            v = spec.pt(g["v"])
            for k in range(max(1, int(g.get("n", 1)))):
                r = g["r"] + 0.08 * k
                start = v + r * np.array([math.cos(math.radians(g["a0"])), math.sin(math.radians(g["a0"]))])
                lines.append(cmd(draw, []) + f" {_xy(start)} arc[start angle={_f(g['a0'])}, "
                             f"end angle={_f(g['a1'])}, radius={_f(r)}];")
        elif t == "right_angle_marked":
            v, d1, d2, s = spec.pt(g["v"]), np.array(g["d1"]), np.array(g["d2"]), g["s"]
            lines.append(cmd(draw, []) + f" {_xy(v + s * d1)} -- {_xy(v + s * d1 + s * d2)} -- {_xy(v + s * d2)};")
        elif t == "length_tick":
            c, d = np.array(g["c"], float), unit(np.array(g["dir"], float))
            nrm = np.array([-d[1], d[0]])
            h = g.get("h", 0.12)
            n = int(g.get("n", 1))
            for k in range(n):
                cc = c + (k - (n - 1) / 2) * 0.08 * d
                lines.append(cmd(draw, []) + f" {_xy(cc - h * nrm)} -- {_xy(cc + h * nrm)};")
    dot = st.get("dot_pt", 1.2)
    for e in spec.entities:
        if e.type == "point" and e.style.get("dot", True):
            lines.append(cmd(fill, []) + f" {P(e.id)} circle[radius={_f(dot)}pt];")
    for e in spec.entities:
        if e.label and e.label_pos is not None:
            lines.append(rf"\node at {_xy(e.label_pos)} {{{_tex_text(e.label)}}};")
    for m in spec.measure_labels:
        lines.append(rf"\node at {_xy(m['pos'])} {{{_tex_text(m['text'])}}};")
    lines.append(r"\end{tikzpicture}")
    return "\n".join(lines) + "\n"


def picture_sha256(tikz: str) -> str:
    """sha256 of the `\\begin{tikzpicture}`..`\\end{tikzpicture}` span (the §9.4 body hash)."""
    i = tikz.index(r"\begin{tikzpicture}")
    j = tikz.rindex(r"\end{tikzpicture}") + len(r"\end{tikzpicture}")
    return sha256_bytes(tikz[i:j].encode("utf-8"))


# ------------------------------------------------------------------------ envelopes
def wrap(tikz: str, kind: str) -> tuple[str, str | None]:
    """(raw response text, stop_reason). The picture span is embedded byte-identically."""
    body = tikz.rstrip("\n")
    if kind == "none":
        return f"```latex\n{body}\n```\n", "end_turn"
    if kind == "prose_fenced":
        return ("Here is a TikZ reconstruction of the figure.\n\n"
                f"```latex\n{body}\n```\n\nThe figure uses only the allowed libraries.\n", "end_turn")
    if kind == "multi_block":
        draft = "\\begin{tikzpicture}\n\\draw (0,0) -- (1,0);\n\\end{tikzpicture}"
        return ("A first sketch:\n\n```latex\n" + draft + "\n```\n\nRefined version:\n\n"
                f"```latex\n{body}\n```\n", "end_turn")
    if kind == "truncated_tail":
        return (f"```latex\n{body}\n```\n\nNotes: the construction places the poi", "length")
    if kind == "bare_body":
        return (f"The reconstruction follows.\n{body}\n", "end_turn")
    raise ValueError(kind)
