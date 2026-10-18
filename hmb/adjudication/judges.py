"""Judge questions Q1 and Q3 (§10.3), the failed-reply / refusal protocol (§10.8) and the two-judge rule
with a third judge when the first two do not decide (§10.6, `aggregate.decide`).

Inputs per question are fixed here and nowhere else:
  Q1  image = candidate render only; text = statements + rendering-convention note.
      Never the crop, context, abstention list, TikZ source, or other facts' verdicts.
  Q3  image = candidate render; text = card entity list with types + candidate objects.
Judge self-reported confidence and explanations are never requested or used.

2026-09-26 streamlining (verdict schema hmb-judge-verdict-2): Q2 source-support and the extras channel are
removed; answers carry no `evidence` string; a reply is read answer by answer, so one malformed or cut-off
entry costs only its own statement (a missing answer is not a vote: the third judge is asked); a slot that
yields no usable answer is not retried with the identical prompt but goes once to a judge of another family.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable, Mapping, Sequence

from hmb.hashing import sha256_bytes, sha256_file
from hmb.ontology import PREDICATES
from hmb.providers import BatchDeferred
from hmb.schemas import JudgeVote, ModelRequest, ModelResponse, ModelSpec
from hmb.vocab import EvaluatorFailure, JudgeQuestion

from .aggregate import NEGATIVE, POSITIVE, TIE, decide

_TRUNCATED = ("length", "max_tokens", "model_length")

PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
PROMPT_FILES = ("Q1.txt", "Q3.txt", "convention_note.txt")
# Forced-choice Q3 tie-break for the escalation judge (owner instruction 2026-09-27). Kept out of PROMPT_FILES so
# the call-cache scope of the Q1/Q3 calls is unchanged; it enters judge_config_hash and the form freeze separately.
TIEBREAK_PROMPT = "Q3_tiebreak.txt"
VERDICT_SCHEMA_VERSION = "hmb-judge-verdict-2"


def prompt_text(name: str) -> str:
    return (PROMPT_DIR / name).read_text(encoding="utf-8")


def prompt_hashes() -> dict[str, str]:
    return {n: sha256_bytes((PROMPT_DIR / n).read_bytes()) for n in PROMPT_FILES}


def tiebreak_prompt_hash() -> dict[str, str]:
    return {TIEBREAK_PROMPT: sha256_bytes((PROMPT_DIR / TIEBREAK_PROMPT).read_bytes())}


# ------------------------------------------------------------------ statement serialization
# Statements, identifiers and the ENTITIES block come from `serialize` (repair 2026-09-26, defect B).
from .serialize import EntityNames, StatementCollision, verbalize, verdict_json_schema  # noqa: E402,F401


# ------------------------------------------------------------------ response parsing
_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.S)
_ENTRY = re.compile(r"\{[^{}]*\}")


@dataclass
class Parsed:
    status: str                         # ok | malformed | refusal
    answers: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


def _entries(s: str) -> tuple[list | None, bool]:
    """(answer entries, refusal flag). The whole reply as JSON first; otherwise every flat `{...}` object in
    it (a reply cut at max_tokens keeps its complete entries)."""
    try:
        obj = json.loads(s)
    except (json.JSONDecodeError, RecursionError):
        obj = None
    if isinstance(obj, dict):
        if obj.get("refusal") is True:
            return None, True
        ans = obj.get("answers")
        return (ans if isinstance(ans, list) else None), False
    out = []
    for m in _ENTRY.finditer(s):
        try:
            d = json.loads(m.group(0))
        except (json.JSONDecodeError, RecursionError):
            continue
        if isinstance(d, dict) and "id" in d:
            out.append(d)
    return (out or None), False


def parse_response(text: str, question: JudgeQuestion, expected_ids: Sequence[str],
                   stop_reason: str | None = None) -> Parsed:
    """Answer-level parsing. Accepts a bare JSON object, one fenced JSON block, or (a cut-off reply) its
    complete answer entries. An entry with an unknown id or a wrongly typed answer is skipped; an id
    answered twice with different values is dropped; a skipped or missing id is not a vote. The reply is
    malformed when it has entries and none survives, or none can be read; a well-formed empty answer list
    is a reply with every answer missing (the third judge is asked)."""
    if stop_reason in ("refusal", "content_filter", "safety"):
        return Parsed("refusal", detail=f"stop_reason={stop_reason}")
    s = (text or "").strip()
    m = _FENCE.match(s)
    if m:
        s = m.group(1).strip()
    ans, refused = _entries(s)
    if refused:
        return Parsed("refusal", detail="refusal flag")
    cut = "truncated" if stop_reason in _TRUNCATED else ""
    if ans is None:
        return Parsed("malformed", detail=cut or "no answers")
    exp = set(expected_ids)
    key = "object" if question == JudgeQuestion.Q3 else "answer"
    out: dict[str, Any] = {}
    conflict: set[str] = set()
    for a in ans:
        if not isinstance(a, dict) or a.get("id") not in exp:
            continue
        v = a.get(key)
        if (key == "answer" and not isinstance(v, bool)) or (key == "object" and not isinstance(v, str)):
            continue
        i = a["id"]
        if i in out and out[i] != v:
            conflict.add(i)
        out.setdefault(i, v)
    for i in conflict:
        del out[i]
    if ans and not out:
        return Parsed("malformed", detail=cut or "no valid answer")
    return Parsed("ok", out)


# ------------------------------------------------------------------ calls
def _call_model(req: ModelRequest, layout, ledger, *, allow_live: bool,
                provider_override=None) -> ModelResponse:
    """Indirection to the Opus 1 contract (tests replace this symbol with a marked fake)."""
    from hmb.providers import call_model
    return call_model(req, layout, ledger, allow_live=allow_live, provider_override=provider_override)


@dataclass
class JudgeCtx:
    layout: Any
    th: Any
    ledger: Any
    allow_live: bool
    provider_override: Any
    phase: str
    config_hash: str


@dataclass
class Ask:
    status: str                          # ok | judge_failure | judge_refusal
    judge: ModelSpec | None = None
    answers: dict[str, Any] = field(default_factory=dict)
    raw_sha: str | None = None
    attempts: int = 0
    failed_shas: list[str] = field(default_factory=list)   # cache keys of failed attempts (provenance)
    detail: str = ""                                       # why the slot failed: truncated, not JSON, ...
    prior_failed: list[dict] = field(default_factory=list)  # a failed first judge this substitute replaced


def ask_once(judge: ModelSpec, question: JudgeQuestion, prompt: str, images: Sequence[Path],
             ids: Sequence[str], ctx: JudgeCtx) -> Ask:
    """One slot, one attempt (§10.8, 2026-09-26: no identical-prompt retry; a deterministic judge repeats a
    malformed or cut-off reply). A provider failure (transport retries exhausted, or a non-retryable
    provider error) fails the slot like a reply with no usable answer; it never aborts the cell."""
    from hmb.providers import AuthError, ProviderError, TransportError
    req = ModelRequest(spec=judge, system=None, text=prompt,
                       image_paths=[str(p) for p in images],
                       image_sha256=[sha256_file(p) for p in images],
                       max_tokens=int(ctx.th.value("judge_max_tokens")), purpose=question.value, phase=ctx.phase,
                       cache_scope=ctx.config_hash,
                       response_schema=(verdict_json_schema(question.value, ids, VERDICT_SCHEMA_VERSION)
                                        if judge.structured_output else None))
    try:
        resp = _call_model(req, ctx.layout, ctx.ledger, allow_live=ctx.allow_live,
                           provider_override=ctx.provider_override)
    except AuthError:
        raise                                # credentials, not the judge: stop the run, resume after the fix
    except (TransportError, ProviderError) as e:
        return Ask(EvaluatorFailure.judge_failure.value, judge, attempts=1, detail=f"{type(e).__name__}: {e}"[:300])
    if resp.fake and ctx.th.is_production:
        raise RuntimeError("production adjudication refuses a fake provider response")
    p = parse_response(resp.text, question, ids, resp.stop_reason)
    if p.status == "ok":
        return Ask("ok", judge, p.answers, resp.cache_key, 1)
    reason = EvaluatorFailure.judge_refusal.value if p.status == "refusal" else EvaluatorFailure.judge_failure.value
    return Ask(reason, judge, attempts=1, failed_shas=[resp.cache_key], detail=p.detail)


class Panel:
    """Judges eligible for one cell, one per family, in frozen preference order. Stateless by design:
    substitution and third-family escalation are decided per call from the families that vote on
    the items of THAT call, never from other questions, chunks or facts."""

    def __init__(self, judges: Sequence[ModelSpec]):
        self.judges = list(judges)

    def primary(self) -> list[ModelSpec]:
        return self.judges[:2]

    def first_excluding(self, exclude: set[str]) -> ModelSpec | None:
        for j in self.judges:
            if j.family not in exclude:
                return j
        return None


@dataclass
class CellFailure(Exception):
    """One failed slot (the judge and its substitute). Callers record it and keep every statement
    already decided; only the statements this slot left undecided become INDETERMINATE."""
    reason: str
    detail: str = ""
    failed: list = field(default_factory=list)      # failed_call() records of every judge tried


def failed_call(a: Ask) -> dict:
    """Provenance of one failed judge slot, stored on the facts it left undecided."""
    return {"judge_id": a.judge.model_id if a.judge else None, "judge_family": a.judge.family if a.judge else None,
            "status": a.status, "detail": a.detail, "raw_call_sha256": list(a.failed_shas)}


def ask_slot(judge: ModelSpec, question: JudgeQuestion, prompt: str, images: Sequence[Path],
             ids: Sequence[str], ctx: JudgeCtx, panel: Panel, exclude: set[str]) -> Ask:
    """§10.8: the judge; on failure the first judge of another family outside `exclude` (the families
    voting on, or already failed on, these items); both fail → CellFailure."""
    a = ask_once(judge, question, prompt, images, ids, ctx)
    if a.status == "ok":
        return a
    sub = panel.first_excluding(set(exclude) | {judge.family})
    if sub is None:
        raise CellFailure(a.status, f"{judge.model_id} failed and no other judge family is available",
                          [failed_call(a)])
    b = ask_once(sub, question, prompt, images, ids, ctx)
    if b.status == "ok":
        b.prior_failed = [failed_call(a)]
        return b
    reason = EvaluatorFailure.judge_refusal.value if (a.status == b.status == "judge_refusal") \
        else EvaluatorFailure.judge_failure.value
    raise CellFailure(reason, f"{judge.model_id} and {sub.model_id} both failed", [failed_call(a), failed_call(b)])


# ------------------------------------------------------------------ binary question (Q1)
@dataclass
class BItem:
    key: str                  # fact_id
    text: str
    refs: tuple = ()          # card entity ids the statement names (defined in the ENTITIES block)


@dataclass
class BResult:
    votes: dict[str, list[JudgeVote]]
    decision: dict[str, str]            # positive | negative | tie (aggregate.decide)
    failures: list[dict] = field(default_factory=list)          # one record per failed slot (cell flag)
    failed_calls: dict[str, list[dict]] = field(default_factory=dict)   # item key -> failed slots on it
    forced: set[str] = field(default_factory=set)       # item keys decided by the forced-side rule


def _chunks(items: list[BItem], size: int) -> list[list[BItem]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


def _prompt(question: JudgeQuestion, lines: list[tuple[str, str]], extra: dict[str, str] | None = None) -> str:
    stmts = "\n".join(f"{i}: {t}" for i, t in lines)
    subs = {"statements": stmts, "schema": VERDICT_SCHEMA_VERSION,
            "convention_note": prompt_text("convention_note.txt").strip(), "entities": "(none)"}
    subs.update(extra or {})
    return Template(prompt_text(f"{question.value}.txt")).substitute(subs)


def run_binary(question: JudgeQuestion, items: list[BItem], *, images: Sequence[Path], ctx: JudgeCtx,
               panel: Panel, seed: str, names: EntityNames | None = None) -> BResult:
    """Two judges from distinct families on every item; a third family only on items the first two do
    not decide (disagreement or a missing answer), chosen PER ITEM as the first panel judge whose
    family did not vote on it; never a fourth vote.

    Failure isolation (repair 2026-09-26): a failed slot (§10.8: the judge and its substitute both
    failed, including transport failure, truncation or an invalid reply) is recorded and the run
    continues. Statements two judges already decided are final and never depend on a later call;
    only the statements the failed slot left without two concurring answers stay a tie (the caller's
    INDETERMINATE). Within one chunk, a family that already failed is not used as a substitute."""
    size = int(ctx.th.value("judge_chunk_size"))
    items = sorted(items, key=lambda b: b.key)
    votes: dict[str, list[JudgeVote]] = {b.key: [] for b in items}
    failures: list[dict] = []
    failed_calls: dict[str, list[dict]] = {}
    failed_fams: dict[str, set[str]] = {b.key: set() for b in items}
    deferred: list = []                  # BatchDeferred of independent slots (hybrid batch runs only)

    def run_chunk(judges_: list[ModelSpec], chunk: list[BItem], stage_name: str, ci: int,
                  prior_voters: set[str]) -> None:
        allit = list(chunk)
        rng = random.Random(int(sha256_bytes(f"{seed}|{stage_name}|{ci}".encode())[:16], 16))
        rng.shuffle(allit)
        ids = [f"s{n + 1:02d}" for n in range(len(allit))]
        idmap = dict(zip(ids, allit))
        imgs = list(images)
        ents = names.context({r for it in allit for r in it.refs}, exclude={it.key for it in allit})             if names is not None else "(none)"
        prompt = _prompt(question, [(i, idmap[i].text) for i in ids], {"entities": ents})
        answered: set[str] = set()
        planned = {j.family for j in judges_}
        tried_failed = {f for it in allit for f in failed_fams[it.key]}
        for j in judges_:
            try:
                a = ask_slot(j, question, prompt, imgs, ids, ctx, panel,
                             prior_voters | planned | answered | tried_failed)
            except BatchDeferred as d:       # answer pending in a batch job: keep collecting independent slots
                deferred.append(d)
                continue
            except CellFailure as cf:
                failures.append({"stage": stage_name, "chunk": ci, "reason": cf.reason, "detail": cf.detail,
                                 "judge_id": j.model_id, "items": sorted(it.key for it in allit),
                                 "calls": cf.failed})
                for it in allit:
                    failed_calls.setdefault(it.key, []).extend(cf.failed)
                    failed_fams[it.key] |= {c["judge_family"] for c in cf.failed}
                tried_failed |= {c["judge_family"] for c in cf.failed}
                continue
            for it in allit:
                if a.prior_failed:
                    failed_calls.setdefault(it.key, []).extend(a.prior_failed)
                    failed_fams[it.key] |= {c["judge_family"] for c in a.prior_failed}
            answered.add(a.judge.family)
            for i, it in idmap.items():
                votes[it.key].append(JudgeVote(judge_id=a.judge.model_id, judge_family=a.judge.family,
                                               question=question, positive=a.answers.get(i),
                                               raw_call_sha256=a.raw_sha))

    def groups_of(its: list[BItem]) -> list[list[BItem]]:
        return _chunks(its, size)

    if items:
        prim = panel.primary()
        if len(prim) < 2 or prim[0].family == prim[1].family:
            raise CellFailure(EvaluatorFailure.judge_failure.value, "fewer than two judge families")
        for ci, chunk in enumerate(groups_of(items)):
            run_chunk(prim, chunk, "stage1", ci, set())
        if deferred:                          # every stage-1 slot is independent; stage 2 needs their answers
            raise BatchDeferred.merge(deferred)

    def settle(b: BItem) -> str:
        return decide(v.positive for v in votes[b.key])

    decision = {b.key: settle(b) for b in items}
    third_of: dict[str, list[BItem]] = {}
    for b in items:
        if decision[b.key] != TIE:
            continue
        t = panel.first_excluding({v.judge_family for v in votes[b.key]})
        if t is not None:                       # no family left for THIS item -> stays a tie
            third_of.setdefault(t.model_id, []).append(b)
    order = {j.model_id: n for n, j in enumerate(panel.judges)}
    for tid in sorted(third_of, key=lambda k: order[k]):
        judge = panel.judges[order[tid]]
        for ci, chunk in enumerate(groups_of(third_of[tid])):
            prior = {v.judge_family for b in chunk for v in votes[b.key]}
            run_chunk([judge], chunk, f"stage2|{tid}", ci, prior)
        for b in third_of[tid]:
            decision[b.key] = settle(b)
    if deferred:
        raise BatchDeferred.merge(deferred)
    # Forced side (owner instruction 2026-09-27): an item still tied after every eligible family takes the
    # answer of the earliest panel judge that answered it; it stays a tie only when no judge answered.
    rank = {j.family: n for n, j in enumerate(panel.judges)}
    forced: set[str] = set()
    for b in items:
        if decision[b.key] == TIE:
            cast = sorted((rank.get(v.judge_family, len(rank)), v.positive is True)
                          for v in votes[b.key] if v.positive is not None)
            if cast:
                decision[b.key] = POSITIVE if cast[0][1] else NEGATIVE
                forced.add(b.key)
    return BResult(votes, decision, failures, failed_calls, forced)


# ------------------------------------------------------------------ Q3 (categorical)
def run_q3(questions: list[tuple[str, str]], *, entities_text: str, objects_text: str, render: Path,
           ctx: JudgeCtx, panel: Panel, options: Mapping[str, Sequence[str]] | None = None
           ) -> tuple[dict[str, str | None], dict[str, list[dict]]]:
    """Q3 alignment questions [(entity_id, question text)]. Two judges from distinct families; an
    answer is accepted when both agree; otherwise a third family and a 2-of-3 majority; otherwise
    unresolved (None). ASSUMPTION: Q3 answers are categorical, so the 'same two-judge rule' (§10.6) is
    applied as agreement/majority rather than an LLR.

    Tie-break (owner instruction 2026-09-27): the third judge is the first panel family that did not vote
    on the pending questions, including a family whose reply to the full Q3 prompt failed, and it gets the
    forced-choice TIEBREAK_PROMPT: per question the distinct answers already given (or, with fewer than
    two, the entity's candidate objects from `options`), no other answer and no refusal."""
    if not questions:
        return {}, {}
    ids = [f"q{n + 1:02d}" for n in range(len(questions))]
    idmap = dict(zip(ids, [e for e, _ in questions]))
    prompt = _prompt(JudgeQuestion.Q3, [(i, t) for i, (_, t) in zip(ids, questions)],
                     {"entities": entities_text, "objects": objects_text})
    prim = panel.primary()
    if len(prim) < 2 or prim[0].family == prim[1].family:
        raise CellFailure(EvaluatorFailure.judge_failure.value, "fewer than two judge families")
    got: dict[str, list[dict]] = {e: [] for e in idmap.values()}
    answered: set[str] = set()
    failed: set[str] = set()
    failures: list[CellFailure] = []
    planned = {j.family for j in prim}

    def record(a: Ask, sub_ids: Sequence[str]) -> None:
        assert a.judge is not None
        for i in sub_ids:
            got[idmap[i]].append({"judge_id": a.judge.model_id, "judge_family": a.judge.family,
                                  "object": a.answers.get(i), "raw_call_sha256": a.raw_sha})

    deferred: list = []
    for j in prim:            # a failed slot is recorded; the other judge's answers stand
        try:
            a = ask_slot(j, JudgeQuestion.Q3, prompt, [render], ids, ctx, panel, planned | answered | failed)
        except BatchDeferred as d:            # hybrid batch runs: collect the other primary slot too
            deferred.append(d)
            continue
        except CellFailure as cf:
            failures.append(cf)
            failed |= {c["judge_family"] for c in cf.failed}
            continue
        failed |= {c["judge_family"] for c in a.prior_failed}
        assert a.judge is not None
        answered.add(a.judge.family)
        record(a, ids)
    if deferred:
        raise BatchDeferred.merge(deferred)
    out: dict[str, str | None] = {}

    def agreed(e: str) -> str | None:
        objs = [v["object"] for v in got[e] if v["object"] is not None]
        top = max(set(objs), key=lambda o: (objs.count(o), o)) if objs else None
        return top if top is not None and objs.count(top) >= 2 and objs.count(top) * 2 > len(objs) else None

    pending = [e for e in idmap.values() if agreed(e) is None]
    for e in idmap.values():
        if e not in pending:
            out[e] = agreed(e)                # decided by two judges: final, never re-asked
    if pending:
        voters = {v["judge_family"] for e in pending for v in got[e]}
        third = panel.first_excluding(voters)
        if third is not None:
            sub_ids = [i for i, e in idmap.items() if e in pending]
            text = dict(zip(ids, (t for _, t in questions)))

            def allowed(e: str) -> list[str]:
                given = sorted({v["object"] for v in got[e] if v["object"] is not None})
                return given if len(given) >= 2 else sorted(set(given) | set((options or {}).get(e, ())))
            stmts = "\n".join(f"{i}: {text[i].split('? Candidates:')[0]}? Options: {', '.join(allowed(idmap[i]))}"
                              for i in sub_ids)
            sub_prompt = Template(prompt_text(TIEBREAK_PROMPT)).substitute(
                {"statements": stmts, "schema": VERDICT_SCHEMA_VERSION, "entities": entities_text,
                 "objects": objects_text})
            try:
                record(ask_slot(third, JudgeQuestion.Q3, sub_prompt, [render], sub_ids, ctx, panel, voters | failed),
                       sub_ids)
            except CellFailure as cf:
                failures.append(cf)
        for e in pending:
            out[e] = agreed(e)
    if failures:              # the caller flags the cell; decided answers are returned with the failure
        raise Q3PartialFailure(failures[0].reason, "; ".join(f.detail for f in failures),
                               [c for f in failures for c in f.failed], out, got)
    return out, got


@dataclass
class Q3PartialFailure(CellFailure):
    """A Q3 slot failed. `answers` still carries every entity two judges agreed on."""
    answers: dict = field(default_factory=dict)
    votes: dict = field(default_factory=dict)
