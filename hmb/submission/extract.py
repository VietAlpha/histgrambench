r"""§9.2 deterministic extraction of the candidate span from a raw contestant response.

Implemented exactly as the plan's algorithm. Form choices (recorded, not in the plan text):
  * "R is empty" includes whitespace-only text.
  * Fenced blocks follow CommonMark fences: an opening line of >= 3 backticks or tildes (<= 3 spaces
    indent), info string = first word, compared case-insensitively against {latex, tex, tikz, ""};
    closed by a line of >= as many of the same character; an unclosed fence runs to the end of R.
    Block content is the exact byte slice of R between the fence lines (no de-indentation), so the
    span in R and the span handed to normalization are the same bytes.
  * Environment counting is textual on the literal strings `\\begin{tikzpicture}` / `\\end{tikzpicture}`
    (a commented-out occurrence counts). "Top-level" = opened at nesting depth 0.
  * bare_doc with no `\\end{document}` after the first `\\documentclass` runs to the end of R (no
    auto-close); bare_pic with no balanced end runs to the end of R and step 5 then applies.
  * Provider thinking is excluded by the caller (ModelResponse.thinking); `split_inline_thinking`
    removes a leading `<think>...</think>` segment for providers that inline it.
  * "Truncated" is the provider's own stop reason for hitting the output cap, in either transport's
    spelling (TRUNCATION_STOP_REASONS: Bedrock Converse `max_tokens`, Bedrock Mantle Chat Completions
    `length`). The raw stop reason is kept unchanged in the submission record.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from hmb.vocab import ExtractionPath, Outcome

EXTRACTION_RULES_VERSION = "extract-9.2-v2"      # v2: Converse `max_tokens` is truncation

BEGIN = "\\begin{tikzpicture}"
END = "\\end{tikzpicture}"
DOCCLASS = "\\documentclass"
END_DOC = "\\end{document}"
INFO_ALLOWED = frozenset({"latex", "tex", "tikz", ""})
# Output-cap stop reasons of the two Bedrock transports (Converse stopReason, Mantle finish_reason).
TRUNCATION_STOP_REASONS = frozenset({"max_tokens", "length"})

_OPEN = re.compile(r"^( {0,3})(`{3,}|~{3,})([^\r\n]*)$")


@dataclass
class Extracted:
    raw: str
    stop_reason: str | None
    outcome: Outcome | None = None        # set on failure; None means a span was extracted
    span: str | None = None
    start: int | None = None              # offsets of the span in raw
    end: int | None = None
    path: ExtractionPath = ExtractionPath.none
    n_blocks: int = 0
    n_fenced_all: int = 0                 # telemetry: all fenced blocks in R
    notes: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.outcome is None


@dataclass(frozen=True)
class Block:
    info: str
    start: int      # content start offset in R
    end: int        # content end offset in R
    closed: bool


def split_inline_thinking(text: str) -> tuple[str, str | None]:
    """Strip a leading `<think>...</think>` (or `<thinking>`) segment; returns (answer, thinking)."""
    m = re.match(r"\s*<(think|thinking)>", text)
    if not m:
        return text, None
    tag = m.group(1)
    close = text.find(f"</{tag}>", m.end())
    if close < 0:
        return "", text[m.end():]          # everything is thinking (truncated inside it)
    return text[close + len(tag) + 3:], text[m.end():close]


def _lines(raw: str):
    pos = 0
    for line in raw.splitlines(keepends=True):
        yield pos, line
        pos += len(line)


def fenced_blocks(raw: str) -> list[Block]:
    out: list[Block] = []
    it = list(_lines(raw))
    i = 0
    while i < len(it):
        pos, line = it[i]
        body = line.rstrip("\r\n")
        m = _OPEN.match(body)
        if not m or (m.group(2)[0] == "`" and "`" in m.group(3)):
            i += 1
            continue
        fch, flen = m.group(2)[0], len(m.group(2))
        words = m.group(3).strip().split()
        info = words[0].lower() if words else ""
        cstart = pos + len(line)
        j = i + 1
        closer = re.compile(r"^ {0,3}" + re.escape(fch) + "{" + str(flen) + r",}[ \t]*$")
        while j < len(it):
            if closer.match(it[j][1].rstrip("\r\n")):
                break
            j += 1
        if j < len(it):
            out.append(Block(info, cstart, it[j][0], True))
            i = j + 1
        else:
            out.append(Block(info, cstart, len(raw), False))
            i = len(it)
    return out


def tikz_events(s: str) -> list[tuple[int, str]]:
    ev = [(m.start(), "b") for m in re.finditer(re.escape(BEGIN), s)]
    ev += [(m.start(), "e") for m in re.finditer(re.escape(END), s)]
    return sorted(ev)


def env_stats(s: str) -> tuple[int, bool]:
    """(# top-level tikzpicture environments, balanced?)"""
    depth, top, balanced = 0, 0, True
    for _, k in tikz_events(s):
        if k == "b":
            if depth == 0:
                top += 1
            depth += 1
        else:
            if depth == 0:
                balanced = False
                continue
            depth -= 1
    return top, balanced and depth == 0


def first_picture_span(s: str) -> tuple[int, int] | None:
    """Offsets [a, b) of the first top-level picture through its matching end; None if unmatched."""
    depth, a = 0, None
    for pos, k in tikz_events(s):
        if k == "b":
            if depth == 0 and a is None:
                a = pos
            depth += 1
        elif depth > 0:
            depth -= 1
            if depth == 0 and a is not None:
                return a, pos + len(END)
    return None


def _qualifies(content: str) -> bool:
    return BEGIN in content or DOCCLASS in content


def extract_candidate(raw: str, stop_reason: str | None, *, choose: str = "last") -> Extracted:
    """§9.2. `choose="first"` exists only for the one-off first-vs-last sensitivity re-extraction."""
    ex = Extracted(raw=raw, stop_reason=stop_reason)
    trunc_or_notikz = Outcome.E_TRUNCATED if stop_reason in TRUNCATION_STOP_REASONS else Outcome.E_NO_TIKZ
    # 1
    if raw is None or raw.strip() == "" or stop_reason == "refusal":
        ex.outcome = Outcome.E_NO_OUTPUT
        return ex
    # 2
    blocks = fenced_blocks(raw)
    ex.n_fenced_all = len(blocks)
    bq = [b for b in blocks if b.info in INFO_ALLOWED and _qualifies(raw[b.start:b.end])]
    # 3
    if bq:
        b = bq[-1] if choose == "last" else bq[0]
        ex.start, ex.end, ex.path, ex.n_blocks = b.start, b.end, ExtractionPath.fenced, len(bq)
    elif DOCCLASS in raw:
        a = raw.find(DOCCLASS)
        z = raw.rfind(END_DOC)
        ex.start, ex.end = a, (z + len(END_DOC) if z >= a else len(raw))
        ex.path = ExtractionPath.bare_doc
        if z < a:
            ex.notes.append("bare_doc_without_end_document")
    elif BEGIN in raw:
        a = raw.find(BEGIN)
        sp = first_picture_span(raw[a:])
        ex.start, ex.end = a, (a + sp[1] if sp else len(raw))
        ex.path = ExtractionPath.bare_pic
    else:
        ex.outcome = trunc_or_notikz
        return ex
    s = raw[ex.start:ex.end]
    ex.span = s
    top, balanced = env_stats(s)
    # 4
    if top > 1:
        ex.outcome = Outcome.E_MULTI_CANDIDATE
        return ex
    # 5
    if not balanced:
        ex.outcome = trunc_or_notikz
        return ex
    return ex


def reextract_first_vs_last(raw: str, stop_reason: str | None) -> dict:
    """§9.2 sensitivity: re-extract with the FIRST qualifying block (no new model calls)."""
    last = extract_candidate(raw, stop_reason, choose="last")
    first = extract_candidate(raw, stop_reason, choose="first")
    return {"last": last, "first": first,
            "differs": (last.outcome, last.span) != (first.outcome, first.span)}
