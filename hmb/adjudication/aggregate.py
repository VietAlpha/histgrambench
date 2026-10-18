"""§10.5 vote aggregation: one majority rule for every judge question (simplification 2026-09-25).

A decision needs at least `min_concurring` (2) judges from distinct families giving the same answer,
and they must outnumber the opposite answer. Two judges are asked first; a third family is asked
only when the first two do not already decide (they disagree, or one answer is missing). With at
most three votes this is "two agree, or two out of three"; anything else is a TIE, which the caller
records as INDETERMINATE. There are no fitted judge reliabilities; judge reliability is reported from the run
itself (INDETERMINATE and evaluator-failure rates, leave-one-judge-family-out G6). Judge self-reported confidence is
never an input.
"""
from __future__ import annotations

from typing import Iterable

from hmb.vocab import Verdict

POSITIVE, NEGATIVE, TIE = "positive", "negative", "tie"


def decide(answers: Iterable[bool | None], min_concurring: int = 2) -> str:
    """POSITIVE / NEGATIVE when that answer has >= `min_concurring` votes and more votes than the
    other; otherwise TIE. A missing answer (None) is not a vote."""
    a = list(answers)
    yes, no = sum(v is True for v in a), sum(v is False for v in a)
    if yes >= min_concurring and yes > no:
        return POSITIVE
    if no >= min_concurring and no > yes:
        return NEGATIVE
    return TIE


def recall_verdict(decision: str, e_incompatible: bool) -> Verdict:
    if decision == POSITIVE:
        return Verdict.PRESENT
    if decision == NEGATIVE:
        return Verdict.CONTRADICTED if e_incompatible else Verdict.ABSENT
    return Verdict.INDETERMINATE
