"""Helpers for W3 tests (imported as w3_helpers; unique name avoids module clashes)."""
from __future__ import annotations

from pathlib import Path

from hmb.config import fixture_thresholds
from hmb.submission import compile_in_sandbox, extract_candidate, normalize


def tikz(body: str, info: str = "latex", pre: str = "") -> str:
    return (f"Here is my reconstruction.\n```{info}\n{pre}\\begin{{tikzpicture}}\n{body}\n"
            f"\\end{{tikzpicture}}\n```\n")


def fast_th(**over):
    """Fixture thresholds with engineering overrides for fast hostile tests (fixture profile only)."""
    th = fixture_thresholds()
    for k, v in over.items():
        th.params[k]["value"] = v
    return th


def compile_response(raw: str, th, workdir: Path, stop_reason: str = "end_turn"):
    """Response -> (extraction, normalization, CompileResult|None) through the production harness."""
    ex = extract_candidate(raw, stop_reason)
    if not ex.ok:
        return ex, None, None
    nz = normalize(ex, th)
    if nz.outcome is not None:
        return ex, nz, None
    return ex, nz, compile_in_sandbox(nz.tex_bytes, th, Path(workdir))
