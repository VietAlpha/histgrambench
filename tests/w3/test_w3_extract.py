"""§9.2 extraction: edge cases with hand-written expected outcome codes."""
import pytest

from hmb.submission.extract import (extract_candidate, fenced_blocks, reextract_first_vs_last,
                                    split_inline_thinking)
from hmb.vocab import ExtractionPath, Outcome

PIC = "\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n\\end{tikzpicture}"
PIC2 = "\\begin{tikzpicture}\n\\draw (0,0) circle (1);\n\\end{tikzpicture}"
DOC = "\\documentclass[tikz]{standalone}\n\\begin{document}\n" + PIC + "\n\\end{document}"


def fence(body, info="latex", ch="```"):
    return f"{ch}{info}\n{body}\n{ch}\n"


@pytest.mark.parametrize("raw,stop,code", [
    ("", "end_turn", Outcome.E_NO_OUTPUT),
    ("   \n\t ", "end_turn", Outcome.E_NO_OUTPUT),
    ("I cannot help with that.", "refusal", Outcome.E_NO_OUTPUT),
    (fence(PIC), "refusal", Outcome.E_NO_OUTPUT),                       # refusal wins over content
    ("No code here at all.", "end_turn", Outcome.E_NO_TIKZ),
    ("No code here at all.", "length", Outcome.E_TRUNCATED),
    (fence("\\draw (0,0) -- (1,1);"), "end_turn", Outcome.E_NO_TIKZ),    # bare body never qualifies (C6)
    (fence(PIC, info="python"), "end_turn", None),                      # not in B, but bare_pic finds it
    ("```latex\n\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n", "length", Outcome.E_TRUNCATED),
    ("```latex\n\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n", "end_turn", Outcome.E_NO_TIKZ),
    (fence(PIC + "\n" + PIC2), "end_turn", Outcome.E_MULTI_CANDIDATE),
    (fence("\\begin{tikzpicture}\\end{tikzpicture}\\end{tikzpicture}"), "end_turn", Outcome.E_NO_TIKZ),
    (fence("\\begin{tikzpicture}\\node{\\begin{tikzpicture}\\end{tikzpicture}};\\end{tikzpicture}"),
     "end_turn", None),                                                 # nested, one top-level
    ("prose " + PIC + " more prose " + PIC2, "end_turn", None),         # bare_pic takes the first only
    ("prose " + DOC + " trailing", "end_turn", None),
    ("prose \\documentclass{article} and nothing else", "length", None),  # bare_doc, no picture: span returned
])
def test_outcome_codes(raw, stop, code):
    ex = extract_candidate(raw, stop)
    assert ex.outcome == code


def test_fenced_path_and_last_block():
    raw = "first try\n" + fence(PIC) + "better:\n" + fence(PIC2) + "done"
    ex = extract_candidate(raw, "end_turn")
    assert ex.ok and ex.path == ExtractionPath.fenced and ex.n_blocks == 2
    assert ex.span.strip() == PIC2
    assert raw[ex.start:ex.end] == ex.span          # exact byte slice of R


def test_info_strings_case_and_empty_and_tilde():
    for info in ("latex", "LaTeX", "tex", "tikz", ""):
        ex = extract_candidate(fence(PIC, info=info), "end_turn")
        assert ex.path == ExtractionPath.fenced, info
    ex = extract_candidate(fence(PIC, info="latex", ch="~~~"), "end_turn")
    assert ex.path == ExtractionPath.fenced
    ex = extract_candidate(fence(PIC, info="python"), "end_turn")
    assert ex.path == ExtractionPath.bare_pic and ex.n_blocks == 0


def test_non_qualifying_blocks_do_not_count():
    raw = fence("just text") + fence(PIC) + fence("\\draw (0,0);")
    ex = extract_candidate(raw, "end_turn")
    assert ex.n_blocks == 1 and ex.span.strip() == PIC


def test_documentclass_block_qualifies_and_is_last():
    raw = fence(PIC) + fence(DOC)
    ex = extract_candidate(raw, "end_turn")
    assert ex.span.strip() == DOC and ex.n_blocks == 2


def test_bare_doc_span_first_documentclass_to_last_end_document():
    raw = "x \\documentclass{standalone}\\begin{document}" + PIC + "\\end{document} y \\end{document} z"
    ex = extract_candidate(raw, "end_turn")
    assert ex.path == ExtractionPath.bare_doc
    assert ex.span.startswith("\\documentclass") and ex.span.endswith("\\end{document}")
    assert ex.span.count("\\end{document}") == 2


def test_bare_pic_balanced_end():
    raw = "Answer: " + PIC + " and then " + PIC2
    ex = extract_candidate(raw, "end_turn")
    assert ex.path == ExtractionPath.bare_pic and ex.span == PIC


def test_never_auto_close():
    raw = "```latex\n\\begin{tikzpicture}\n\\draw (0,0) -- (1,1);\n```\n"
    ex = extract_candidate(raw, "length")
    assert ex.outcome == Outcome.E_TRUNCATED and ex.span is not None
    assert "\\end{tikzpicture}" not in ex.span


def test_crlf_and_indented_fence_preserved():
    raw = "text\r\n```latex\r\n" + PIC.replace("\n", "\r\n") + "\r\n```\r\n"
    ex = extract_candidate(raw, "end_turn")
    assert ex.ok and "\r\n" in ex.span
    ex = extract_candidate("  ```latex\n" + PIC + "\n  ```\n", "end_turn")
    assert ex.path == ExtractionPath.fenced


def test_unclosed_fence_runs_to_end():
    blocks = fenced_blocks("```latex\nabc\n")
    assert len(blocks) == 1 and not blocks[0].closed


def test_longer_fence_contains_shorter():
    raw = "````latex\n```\n" + PIC + "\n````\n"
    ex = extract_candidate(raw, "end_turn")
    assert ex.path == ExtractionPath.fenced and PIC in ex.span


def test_first_vs_last_sensitivity():
    raw = fence(PIC) + fence(PIC2)
    r = reextract_first_vs_last(raw, "end_turn")
    assert r["differs"] and r["first"].span.strip() == PIC and r["last"].span.strip() == PIC2
    r = reextract_first_vs_last(fence(PIC), "end_turn")
    assert not r["differs"]


def test_inline_thinking_split():
    ans, th = split_inline_thinking("<think>plan \\begin{tikzpicture}</think>\n" + fence(PIC))
    assert th.startswith("plan") and extract_candidate(ans, "end_turn").ok
    ans, th = split_inline_thinking("<think>never closed")
    assert ans == "" and extract_candidate(ans, "length").outcome == Outcome.E_NO_OUTPUT
    assert split_inline_thinking("no tags")[1] is None
