"""§9.3 wrapper classes + preamble allowlist grammar; §9.4 byte-identity invariant."""
import pytest

from hmb.hashing import sha256_bytes
from hmb.submission import wrapper as W
from hmb.submission.extract import extract_candidate
from hmb.submission.wrapper import HarnessInvariantError, assert_byte_identity, normalize
from hmb.vocab import Outcome, WrapperClass

PIC = "\\begin{tikzpicture}[scale=2]\n  \\draw (0,0) -- (1,1) node[right] {$A_1$};   % comment\n\\end{tikzpicture}"


def norm(span, th):
    ex = extract_candidate("```latex\n" + span + "\n```", "end_turn")
    assert ex.ok, ex.outcome
    return normalize(ex, th)


@pytest.mark.parametrize("pre", [
    "",
    "\\usepackage{tikz}\n",
    "\\usepackage[dvipsnames]{xcolor}\n\\usepackage{amsmath,amssymb}\n",
    "\\usepackage{pgfplots}\n\\pgfplotsset{compat=1.18}\n\\usepgfplotslibrary{fillbetween, polar}\n",
    "\\usetikzlibrary{calc,\n  intersections, arrows.meta}\n",
    "\\tikzset{\n  every node/.style={font=\\small},\n  dot/.style={circle,fill,inner sep=1pt}\n}\n",
    "\\definecolor{myblue}{RGB}{0,0,200}\n",
    "\\newcommand{\\pt}[1]{\\fill (#1) circle (1pt);}\n\\renewcommand*\\familydefault{\\sfdefault}\n",
    "\\newcommand\\R[2][x]{#1 #2}\n",
    "\\def\\myr{2}\n\\def\\f#1#2{#1+#2}\n\\let\\oldd=\\draw\n\\let\\x\\y\n",
    "\\pgfdeclarelayer{bg}\n\\pgfdeclareverticalshading{s}{1cm}{color(0cm)=(red); color(1cm)=(blue)}\n",
    "% a comment line\n\n\\usepackage{tikz-3dplot}\n",
])
def test_class_p_allowed_preambles(pre, th):
    n = norm(pre + PIC, th)
    assert n.outcome is None and n.wrapper_class == WrapperClass.P
    assert pre in n.tex                             # copied verbatim, in order
    assert n.body_sha256 == sha256_bytes(PIC.encode("utf-8"))


@pytest.mark.parametrize("pre,token", [
    ("\\usepackage{graphicx}\n", "\\usepackage{graphicx}"),
    ("\\usepackage{tikz,fontspec}\n", "\\usepackage{fontspec}"),
    ("\\usetikzlibrary{calc,external}\n", "\\usetikzlibrary{external}"),
    ("\\usetikzlibrary{datavisualization}\n", "\\usetikzlibrary{datavisualization}"),
    ("\\usepgfplotslibrary{external}\n", "\\usepgfplotslibrary{external}"),
    ("\\input{/etc/passwd}\n", "\\input"),
    ("\\directlua{os.execute('x')}\n", "\\directlua"),
    ("\\immediate\\write18{ls}\n", "\\immediate"),
    ("\\setmainfont{Arial}\n", "\\setmainfont"),
    ("\\begin{document}\n", "\\begin"),
    ("\\tdplotsetmaincoords{70}{110}\n", "\\tdplotsetmaincoords"),
    ("stray text\n", "stray text"),
    ("\\tikzset{unbalanced\n", "…(unbalanced)"),
])
def test_class_p_rejections_name_the_token(pre, token, th):
    n = norm(pre + PIC, th)
    assert n.outcome == Outcome.P_POLICY_REJECT
    assert token in n.reject_token


def test_class_p_trailing_code_rejected_trailing_comment_ok(th):
    n = norm(PIC + "\n\\end{document}\n", th)
    assert n.outcome == Outcome.P_POLICY_REJECT and "\\end{document}" in n.reject_token
    n = norm(PIC + "\n% done\n", th)
    assert n.outcome is None


def test_class_p_document_structure(th):
    n = norm("\\usetikzlibrary{calc}\n" + PIC, th)
    t = n.tex
    assert t.startswith("\\documentclass[tikz,border=2mm]{standalone}\n")
    assert t.index("\\tracinglostchars=3") < t.index("\\usetikzlibrary{calc}\n") < t.index("\\begin{document}")
    assert t.rstrip().endswith("\\end{document}")
    assert t.count("\\begin{tikzpicture}") == 1


def test_class_d_compiled_as_submitted_with_font_block(th):
    doc = ("% leading comment\n\\documentclass[tikz,border=1pt]{standalone}\n\\usepackage{amsmath}\n"
           "\\begin{document}\n" + PIC + "\n\\end{document}")
    n = norm(doc, th)
    assert n.wrapper_class == WrapperClass.D and n.outcome is None
    fb = W.harness_font_block()
    assert fb in n.tex and n.tex.index(fb) < n.tex.index("\\usepackage{amsmath}")
    assert n.tex.replace("\n" + fb, "", 1).rstrip("\n") == doc
    assert n.body_sha256 == sha256_bytes(PIC.encode())


def test_class_d_preamble_grammar_and_leading_code(th):
    n = norm("\\documentclass{standalone}\n\\usepackage{graphicx}\n\\begin{document}\n" + PIC + "\n\\end{document}", th)
    assert n.outcome == Outcome.P_POLICY_REJECT and n.reject_token == "\\usepackage{graphicx}"
    n = norm("\\RequirePackage{x}\n\\documentclass{standalone}\n\\begin{document}\n" + PIC + "\n\\end{document}", th)
    assert n.outcome == Outcome.P_POLICY_REJECT and "RequirePackage" in n.reject_token


def test_class_b_wraps_bare_body(th):
    from hmb.submission.extract import Extracted
    body = "\\draw (0,0) -- (1,0);\n\\node at (0,0) {A};"
    n = normalize(Extracted(raw=body, stop_reason=None, span=body), th)
    assert n.wrapper_class == WrapperClass.B and n.outcome is None
    assert "\\begin{tikzpicture}\n" + body + "\n\\end{tikzpicture}" in n.tex
    assert n.body_sha256 == sha256_bytes(body.encode())


def test_invariant_holds_for_unicode_and_crlf(th):
    pic = "\\begin{tikzpicture}\r\n\\node {中國 ệ Ж α};\r\n\\end{tikzpicture}"
    n = norm(pic, th)
    assert n.body_sha256 == sha256_bytes(pic.encode("utf-8"))


def test_invariant_violation_is_a_harness_error(th):
    n = norm(PIC, th)
    tampered = n.tex.replace("(1,1)", "(1,2)")
    with pytest.raises(HarnessInvariantError):
        assert_byte_identity(PIC, tampered, WrapperClass.P)
    with pytest.raises(HarnessInvariantError):
        assert_byte_identity(PIC, n.tex.replace("% comment", "%comment"), WrapperClass.P)


def test_hidden_repair_would_fail_invariant(monkeypatch, th):
    # If normalization ever rewrote a byte inside the span, normalize() must raise, not return:
    # simulate a rewritten picture span in the normalized text only.
    ex = extract_candidate("```latex\n" + PIC + "\n```", "end_turn")
    real = W.invariant_span

    def bad_span(text, wclass, body=None):
        out = real(text, wclass, body=body)
        return out if text == ex.span else out.replace("1,1", "1,9")
    monkeypatch.setattr(W, "invariant_span", bad_span)
    with pytest.raises(HarnessInvariantError):
        normalize(ex, th)


def test_wrapper_file_published_and_hashed():
    t = W.wrapper_text()
    assert "\\tracinglostchars=3" in t and "fontspec" in t
    for fam in ("Noto Serif}", "Noto Serif CJK TC", "Noto Serif CJK JP", "Noto Serif CJK SC"):
        assert fam in t
    assert len(W.wrapper_sha256()) == 64
    assert "\\usetikzlibrary{" + ",".join(W.LIB_BASE) + "}" in t


def test_commented_documentclass_does_not_make_class_d(th):
    """Red-team #7: classification ignores TeX comments; bytes stay identical."""
    comment = "% \\documentclass{article} would be wrong here\n"
    n = norm(comment + PIC, th)
    assert n.wrapper_class == WrapperClass.P and n.outcome is None
    assert comment in n.tex
    assert W.classify("50\\% \\documentclass") == WrapperClass.D      # an escaped percent is not a comment
    doc = ("%\\documentclass{old}\n\\documentclass[tikz]{standalone}\n\\begin{document}\n" + PIC
           + "\n\\end{document}")
    n = norm(doc, th)
    assert n.wrapper_class == WrapperClass.D and n.outcome is None
    assert n.tex.index(W.harness_font_block()) > n.tex.index("{standalone}")   # after the REAL one


def test_whitespace_tolerant_begin_document(th):
    """Red-team #9: `\\begin {document}` still delimits the class-D preamble."""
    doc = ("\\documentclass[tikz]{standalone}\n\\usepackage{tikz}\n\\begin {document}\n" + PIC
           + "\n\\end {document}")
    n = norm(doc, th)
    assert n.outcome is None and n.wrapper_class == WrapperClass.D
    body = "\\begin {document}x\\end {document}"
    assert W.invariant_span(body, WrapperClass.D) == body


@pytest.mark.parametrize("span", ["\\documentclass[tikz,border=10pt]{standalone}",
                                  "\\documentclass{standalone}\n\\usepackage{tikz}\n"])
def test_class_d_without_body_normalizes_and_still_catches_a_rewrite(span, th):
    # EC2 MS5 2026-10-02: a lone class line used to raise HarnessInvariantError (whole-text fallback vs inserted fonts)
    ex = extract_candidate("```latex\n" + span + "\n```", "end_turn")
    nz = normalize(ex, th)
    assert nz.wrapper_class == WrapperClass.D and nz.outcome is None and ex.span is not None
    assert nz.body_sha256 == sha256_bytes(ex.span.encode("utf-8"))
    with pytest.raises(HarnessInvariantError):
        assert_byte_identity(ex.span, nz.tex.replace("standalone", "standalonE"), WrapperClass.D)
