r"""§9.3 wrapper normalization (classes D/P/B, preamble allowlist grammar) and the §9.4 invariant.

The static checker is a preamble allowlist grammar only, not a security control (§9.3).

Form choices (recorded):
  * The grammar is applied per *statement* (a control sequence with its balanced arguments), so a
    `\\tikzset{...}` spanning several physical lines is one statement; comments and blank lines are
    allowed and copied verbatim.
  * `\\usepgfplotslibrary` allowed set PGFPLOTSLIB_ALLOW: the plan says "from the allowed set" without
    listing it; the set below is every pgfplots library except `external` and `clickable` (shell /
    JavaScript), matching the §9.3 exclusions.
  * LIB_BASE (preloaded in wrapper_v1) = LIB_ALLOW.
  * Class D: the contestant document is compiled as submitted after the grammar check, with the
    fixed harness font block (fontspec fallback chain + `\\tracinglostchars=3`) inserted right after the
    `\\documentclass` statement (bytes added before `\\begin{document}`, allowed by §9.4), so every
    class uses one font setup (§9.9 rejects per-item font selection).
  * Class P: text after the picture inside the span may be only whitespace or comments; anything
    else is P_POLICY_REJECT naming the token (deleting it would be a hidden repair).
  * Class B: unreachable from §9.2 as written (BUILD_STATUS C6); implemented for direct calls.
    The invariant span for B is the bare body between the wrapper-added picture delimiters.
  * A class-D document without any tikzpicture environment: invariant span = the document body
    `\\begin{document}`..`\\end{document}` if present, else all of S.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from hmb.hashing import sha256_bytes
from hmb.vocab import Outcome, WrapperClass

from .extract import BEGIN, DOCCLASS, Extracted, first_picture_span

WRAPPER_VERSION = "wrapper_v1"

PKG_ALLOW = ("tikz", "pgfplots", "amsmath", "amssymb", "xcolor", "tikz-3dplot")
LIB_ALLOW = ("calc", "intersections", "angles", "quotes", "positioning", "arrows.meta",
             "decorations.markings", "decorations.pathmorphing", "patterns", "through",
             "shapes.geometric", "shapes.misc", "3d", "perspective", "math", "fit", "backgrounds",
             "babel")
PGFPLOTSLIB_ALLOW = ("fillbetween", "polar", "groupplots", "colormaps", "colorbrewer", "statistics",
                     "ternary", "smithchart", "patchplots", "units", "dateplot",
                     "decorations.softclip")
LIB_BASE = LIB_ALLOW

_WRAPPER_FILE = Path(__file__).with_name("wrapper_v1.tex")


def wrapper_text() -> str:
    return _WRAPPER_FILE.read_text(encoding="utf-8")


def wrapper_sha256() -> str:
    return sha256_bytes(_WRAPPER_FILE.read_bytes())


def _wrapper_parts() -> tuple[str, str]:
    """(font block, full header up to the contestant-preamble marker)."""
    t = wrapper_text()
    fb = t[t.index("%<FONTBLOCK>\n") + len("%<FONTBLOCK>\n"):t.index("%</FONTBLOCK>\n")]
    return fb, t


def harness_font_block() -> str:
    return _wrapper_parts()[0]


class HarnessInvariantError(RuntimeError):
    """§9.4 violation: a harness bug that fails the run; never an item outcome."""


@dataclass
class Normalized:
    wrapper_class: WrapperClass | None
    outcome: Outcome | None = None          # P_POLICY_REJECT or None
    reject_token: str | None = None
    tex: str | None = None
    body_sha256: str | None = None

    @property
    def tex_bytes(self) -> bytes | None:
        return None if self.tex is None else self.tex.encode("utf-8")


# ----------------------------------------------------------------------------- scanner
class _Reject(Exception):
    def __init__(self, token: str):
        super().__init__(token)
        self.token = token


def _skip_ws(s: str, i: int) -> int:
    while i < len(s) and s[i] in " \t\r\n":
        i += 1
    return i


def _group(s: str, i: int, open_: str, close: str) -> int:
    """s[i] == open_; return index after the balanced close (brace-aware, escapes skipped)."""
    depth = 0
    j = i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "%":
            nl = s.find("\n", j)
            j = len(s) if nl < 0 else nl + 1
            continue
        if c == open_:
            depth += 1
        elif c == close:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    raise _Reject(s[i:i + 40] + "…(unbalanced)")


def _brace(s: str, i: int) -> tuple[str, int]:
    i = _skip_ws(s, i)
    if i >= len(s) or s[i] != "{":
        raise _Reject(s[i:i + 40] or "<end of preamble>")
    j = _group(s, i, "{", "}")
    return s[i + 1:j - 1], j


def _opt(s: str, i: int) -> int:
    k = _skip_ws(s, i)
    if k < len(s) and s[k] == "[":
        return _group_bracket(s, k)
    return i


def _group_bracket(s: str, i: int) -> int:
    depth_b, j = 0, i
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == "{":
            j = _group(s, j, "{", "}")
            continue
        if c == "[":
            depth_b += 1
        elif c == "]":
            depth_b -= 1
            if depth_b == 0:
                return j + 1
        j += 1
    raise _Reject(s[i:i + 40] + "…(unbalanced)")


def _csname(s: str, i: int) -> tuple[str, int]:
    assert s[i] == "\\"
    j = i + 1
    if j < len(s) and s[j].isalpha():
        while j < len(s) and (s[j].isalpha() or s[j] == "@"):
            j += 1
    else:
        j += 1
    return s[i:j], j


def _list_items(txt: str) -> list[str]:
    return [x.strip() for x in re.sub(r"%[^\n]*", "", txt).split(",") if x.strip()]


def check_preamble(pre: str) -> None:
    """Raise _Reject(token) unless every statement of `pre` matches the §9.3 grammar."""
    i = 0
    while True:
        i = _skip_ws(pre, i)
        if i >= len(pre):
            return
        if pre[i] == "%":
            nl = pre.find("\n", i)
            i = len(pre) if nl < 0 else nl + 1
            continue
        if pre[i] != "\\":
            raise _Reject(pre[i:i + 40].split("\n")[0])
        name, j = _csname(pre, i)
        if name == "\\usepackage":
            j = _opt(pre, j)
            body, j = _brace(pre, j)
            for p in _list_items(body):
                if p not in PKG_ALLOW:
                    raise _Reject(f"\\usepackage{{{p}}}")
            j = _opt(pre, j)
        elif name in ("\\usetikzlibrary", "\\usepgfplotslibrary"):
            allow = LIB_ALLOW if name == "\\usetikzlibrary" else PGFPLOTSLIB_ALLOW
            body, j = _brace(pre, j)
            for lib in _list_items(body):
                if lib not in allow:
                    raise _Reject(f"{name}{{{lib}}}")
        elif name in ("\\pgfplotsset", "\\tikzset"):
            _, j = _brace(pre, j)
        elif name == "\\definecolor":
            j = _opt(pre, j)
            for _ in range(3):
                _, j = _brace(pre, j)
        elif name in ("\\newcommand", "\\renewcommand"):
            k = _skip_ws(pre, j)
            if k < len(pre) and pre[k] == "*":
                j = k + 1
            k = _skip_ws(pre, j)
            if k < len(pre) and pre[k] == "{":
                _, j = _brace(pre, k)
            elif k < len(pre) and pre[k] == "\\":
                _, j = _csname(pre, k)
            else:
                raise _Reject(name)
            j = _opt(pre, j)
            j = _opt(pre, j)
            _, j = _brace(pre, j)
        elif name == "\\def":
            k = _skip_ws(pre, j)
            if k >= len(pre) or pre[k] != "\\":
                raise _Reject(name)
            _, j = _csname(pre, k)
            while j < len(pre) and pre[j] != "{":
                if pre[j] in "\\%}":
                    raise _Reject(name + " parameter text")
                j += 1
            _, j = _brace(pre, j)
        elif name == "\\let":
            k = _skip_ws(pre, j)
            if k >= len(pre) or pre[k] != "\\":
                raise _Reject(name)
            _, j = _csname(pre, k)
            k = _skip_ws(pre, j)
            if k < len(pre) and pre[k] == "=":
                k = _skip_ws(pre, k + 1)
            if k >= len(pre):
                raise _Reject(name)
            if pre[k] == "\\":
                _, j = _csname(pre, k)
            else:
                j = k + 1
        elif name.startswith("\\pgfdeclare"):
            while True:
                k = _skip_ws(pre, j)
                if k < len(pre) and pre[k] == "{":
                    _, j = _brace(pre, k)
                elif k < len(pre) and pre[k] == "[":
                    j = _group_bracket(pre, k)
                else:
                    break
        else:
            raise _Reject(name)
        i = j


def _only_ws_or_comments(s: str) -> str | None:
    """None if s is whitespace/comments only, else the first offending token."""
    for line in s.splitlines():
        t = line.strip()
        if t and not t.startswith("%"):
            return t[:40]
    return None


# ----------------------------------------------------------------------------- invariant
def invariant_span(text: str, wclass: WrapperClass, *, body: str | None = None) -> str:
    sp = first_picture_span(text)
    if sp is not None:
        return text[sp[0]:sp[1]]
    if wclass == WrapperClass.B and body is not None:
        return body
    a = find_begin_document(text)
    ends = list(_END_DOC_RX.finditer(text))
    if a >= 0 and ends and ends[-1].start() > a:
        return text[a:ends[-1].end()]
    return text


def assert_byte_identity(span_raw: str, normalized: str, wclass: WrapperClass) -> str:
    """§9.4; returns body_sha256. Raises HarnessInvariantError on any mismatch."""
    if (wclass == WrapperClass.D and first_picture_span(span_raw) is None
            and not (find_begin_document(span_raw) >= 0 and _END_DOC_RX.search(span_raw))):
        # A class-D document with no picture and no document body (e.g. a lone `\documentclass` line; EC2 MS5
        # 2026-10-02, owner-instructed fix). There is no span to protect, and the whole-text comparison below
        # always failed because of the font block inserted after the class line, which §9.4 allows ("may add
        # bytes before \begin{document}"). The check is instead that every raw byte survives in order around
        # exactly that one insertion. This branch only replaces a path that raised for every input.
        d = find_uncommented(span_raw, DOCCLASS)
        j = _brace(span_raw, _opt(span_raw, d + len(DOCCLASS)))[1] if d >= 0 else -1
        font_block, _ = _wrapper_parts()
        rebuilt = span_raw[:j] + "\n" + font_block + span_raw[j:]
        if j >= 0 and not rebuilt.endswith("\n"):          # normalize() ends the file with a newline
            rebuilt += "\n"
        if j < 0 or normalized != rebuilt:
            raise HarnessInvariantError(f"§9.4 byte-identity violated (class {wclass.value})")
        return sha256_bytes(span_raw.encode("utf-8"))
    if wclass == WrapperClass.B:
        a = normalized.find(BEGIN + "\n")
        z = normalized.rfind("\n\\end{tikzpicture}")
        got = normalized[a + len(BEGIN) + 1:z] if a >= 0 and z > a else None
        want = span_raw
    else:
        want = invariant_span(span_raw, wclass)
        got = invariant_span(normalized, wclass)
    h_want = sha256_bytes(want.encode("utf-8"))
    if got is None or sha256_bytes(got.encode("utf-8")) != h_want:
        raise HarnessInvariantError(f"§9.4 byte-identity violated (class {wclass.value})")
    return h_want


# ----------------------------------------------------------------------------- normalize
_BEGIN_DOC_RX = re.compile(r"\\begin\s*\{document\}")
_END_DOC_RX = re.compile(r"\\end\s*\{document\}")


def find_uncommented(s: str, token: str, start: int = 0) -> int:
    """Offset of the first occurrence of `token` not inside a TeX comment (`%` not preceded by an odd
    number of backslashes, to end of line); -1 if none. Classification only; bytes are untouched."""
    i = s.find(token, start)
    while i >= 0:
        ls = s.rfind("\n", 0, i) + 1
        line = s[ls:i]
        in_comment = False
        k = 0
        while k < len(line):
            if line[k] == "\\":
                k += 2
                continue
            if line[k] == "%":
                in_comment = True
                break
            k += 1
        if not in_comment:
            return i
        i = s.find(token, i + 1)
    return -1


def find_begin_document(s: str, start: int = 0) -> int:
    m = _BEGIN_DOC_RX.search(s, start)
    return m.start() if m else -1


def classify(span: str) -> WrapperClass:
    if find_uncommented(span, DOCCLASS) >= 0:
        return WrapperClass.D
    if BEGIN in span:
        return WrapperClass.P
    return WrapperClass.B


def normalize(ex: Extracted, th=None) -> Normalized:
    """§9.3. `th` is accepted for the contract signature; no registered number is used here."""
    if not ex.ok or ex.span is None:
        raise ValueError("normalize() requires a successful extraction")
    s = ex.span
    wc = classify(s)
    font_block, wrapper = _wrapper_parts()
    header = wrapper[:wrapper.index("%<CONTESTANT_PREAMBLE>\n")]
    try:
        if wc == WrapperClass.D:
            d = find_uncommented(s, DOCCLASS)
            lead = _only_ws_or_comments(s[:d])
            if lead is not None:
                raise _Reject(lead)
            j = d + len(DOCCLASS)
            j = _opt(s, j)
            _, j = _brace(s, j)
            bd = find_begin_document(s, j)
            stop = bd if bd >= 0 else (s.find(BEGIN, j) if BEGIN in s[j:] else len(s))
            check_preamble(s[j:stop])
            tex = s[:j] + "\n" + font_block + s[j:]
            if not tex.endswith("\n"):
                tex += "\n"
        elif wc == WrapperClass.P:
            p = s.find(BEGIN)
            pre = s[:p]
            check_preamble(pre)
            sp = first_picture_span(s)
            assert sp is not None  # balanced by §9.2 step 5
            tail = _only_ws_or_comments(s[sp[1]:])
            if tail is not None:
                raise _Reject(tail)
            pre_lines = pre if pre.endswith("\n") or pre == "" else pre + "\n"
            tex = header + pre_lines + "\\begin{document}\n" + s[p:].rstrip() + "\n\\end{document}\n"
        else:
            tex = (header + "\\begin{document}\n" + BEGIN + "\n" + s + "\n\\end{tikzpicture}\n"
                   + "\\end{document}\n")
    except _Reject as r:
        return Normalized(wc, Outcome.P_POLICY_REJECT, reject_token=r.token)
    body = assert_byte_identity(s, tex, wc)
    return Normalized(wc, None, tex=tex, body_sha256=body)
