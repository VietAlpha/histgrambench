"""§8.3 verification pass (asc-2.0.0, 2026-09-26 streamlining): union MGS, chunked statements, votes.

* U = MGS(union over writers of MGS(K_w)) on canonical ids (after alignment).
* Every writer is asked about every union fact, in chunks of at most `card_verify_chunk_size` statements, so no
  call carries an unbounded list; statements are shuffled deterministically per (item, writer). A fact in the
  closure of a writer's own generation counts as that writer's support unless it votes false on it (self-veto);
  any other fact needs its true vote.
* Votes are one `<id> <vote>` line per statement and are read line by line: a reply cut at `max_tokens`
  keeps every complete line. Silence is not a vote: an unanswered statement is `abstain` and counted in
  `missing`.
"""
from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass, field

from hmb.ontology import PREDICATES
from hmb.vocab import Vote

from .align import Alignment
from .closure import FactKey, forward_chain, minimal_generating_set



@dataclass
class Union:
    facts: list[FactKey]
    types: dict[str, str]
    context_assisted: set[FactKey]
    per_builder_mgs: dict[str, list[FactKey]]
    closure: set[FactKey] = field(default_factory=set)
    per_builder_closure: dict[str, set[FactKey]] = field(default_factory=dict)   # what each writer supports


def build_union(al: Alignment) -> Union:
    types = al.types
    per_mgs: dict[str, list[FactKey]] = {}
    per_cl: dict[str, set[FactKey]] = {}
    ctx: set[FactKey] = set()
    pool: set[FactKey] = set()
    for b in al.builder_order:
        fs = al.builder_facts.get(b, [])
        keys = [k for k, _ in fs]
        ctx |= {k for k, c in fs if c}
        per_mgs[b] = minimal_generating_set(keys, types) if keys else []
        per_cl[b] = forward_chain(per_mgs[b], types) if per_mgs[b] else set()
        pool |= set(per_mgs[b])
    U = minimal_generating_set(pool, types) if pool else []
    CU = forward_chain(U, types) if U else set()
    return Union(U, types, ctx, per_mgs, CU, per_cl)


# ------------------------------------------------------------------ statements
def describe(cid: str, al: Alignment, hide_label: bool = False) -> str:
    s = {x.cid: x for x in al.slots}.get(cid)
    if s is None:
        return cid
    t = s.type.replace("_", " ")
    if s.label and not hide_label:
        return f"{t} {s.label}"
    if s.anchor:
        return f"the {'' if hide_label else 'unlabelled '}{t} near ({s.anchor['x']:.2f}, {s.anchor['y']:.2f})"
    return f"an unlabelled {t}"


def fact_text(f: FactKey, al: Alignment) -> str:
    """Natural-language rendering of a canonical card fact (classes E, R, L)."""
    pred, a = f
    hide = pred in ("label", "measure_label")

    def D(x):
        return describe(x, al, hide_label=hide)

    def L(xs):
        return ", ".join(D(x) for x in xs)
    T = {
        "polygon": lambda: f"{D(a[0])} is a polygon with {a[1]} sides.",
        "axis": lambda: f"{D(a[0])} is a {a[1]} coordinate axis.",
        "curve": lambda: f"{D(a[0])} is a curve of kind '{a[1]}'.",
        "solid": lambda: f"{D(a[0])} is a solid of kind '{a[1]}'.",
        "on": lambda: f"{D(a[0])} lies on {D(a[1])}.",
        "endpoint_of": lambda: f"{D(a[0])} is an endpoint of {D(a[1])}.",
        "vertex_of": lambda: f"{D(a[0])} is a vertex of {D(a[1])}.",
        "edge_of": lambda: f"{D(a[0])} is an edge of {D(a[1])}.",
        "face_of": lambda: f"{D(a[0])} is a face of {D(a[1])}.",
        "center_of": lambda: f"{D(a[0])} is the centre of {D(a[1])}.",
        "arc_of": lambda: f"{D(a[0])} is an arc of {D(a[1])}.",
        "intersect_at": lambda: f"{D(a[0])} and {D(a[1])} cross at {D(a[2])}.",
        "tangent_to": lambda: (f"{D(a[0])} is tangent to {D(a[1])}." if a[2] == "unknown"
                               else f"{D(a[0])} is tangent to {D(a[1])} at {D(a[2])}."),
        "parallel": lambda: f"{D(a[0])} is parallel to {D(a[1])}.",
        "perpendicular": lambda: f"{D(a[0])} is perpendicular to {D(a[1])}.",
        "collinear": lambda: f"The points {L(a[0])} lie on one straight line.",
        "concyclic": lambda: f"The points {L(a[0])} lie on one circle.",
        "between": lambda: f"{D(a[1])} lies between {D(a[0])} and {D(a[2])} on one straight line.",
        "equal_length": lambda: f"{D(a[0])} and {D(a[1])} are marked or stated as equal in length.",
        "equal_angle": lambda: f"{D(a[0])} and {D(a[1])} are marked or stated as equal angles.",
        "contains": lambda: f"{D(a[0])} contains {D(a[1])}.",
        "cross_section_of": lambda: f"{D(a[0])} is a cross-section of {D(a[1])}.",
        "hidden_edge": lambda: f"{D(a[0])} is drawn as a hidden (dashed) edge.",
        "asymptote_of": lambda: f"{D(a[0])} is an asymptote of {D(a[1])}.",
        "label": lambda: f"The label \"{a[0]}\" is printed for {D(a[1])}.",
        "measure_label": lambda: f"The {a[2]} label \"{a[0]}\" is printed for {D(a[1])}.",
    }
    if pred in T:
        return T[pred]()
    if PREDICATES[pred].fclass.value == "entity":
        return f"There is a {pred.replace('_', ' ')}: {D(a[0])}."
    raise KeyError(pred)


# ------------------------------------------------------------------ statements
def _seed(*parts: str) -> int:
    return int(hashlib.sha256("".join(parts).encode()).hexdigest()[:16], 16)


@dataclass
class VerificationItem:
    sid: str
    text: str
    fact: FactKey


def verification_facts(U: Union, builder_id: str) -> list[FactKey]:
    """The union facts one writer is asked about: all of them. Its own generated facts are asked too, as a
    self-veto (the preregistered fallback of the 2026-09-26 streamlining: asking only about the other writer's
    facts measured pooled P_card 0.566 against the 0.568 floor; the self-veto removes a writer's own false facts)."""
    return list(U.facts)


def verification_chunks(U: Union, al: Alignment, item_id: str, builder_id: str,
                        chunk_size: int) -> list[list[VerificationItem]]:
    """One writer's statements in a seeded order, cut into calls of at most `chunk_size`; ids restart at
    s01 in every chunk."""
    rows = verification_facts(U, builder_id)
    random.Random(_seed(item_id, builder_id, "order")).shuffle(rows)
    out = []
    for c in range(0, len(rows), chunk_size):
        part = rows[c:c + chunk_size]
        width = max(2, len(str(len(part))))
        out.append([VerificationItem(f"s{i + 1:0{width}d}", fact_text(f, al), f) for i, f in enumerate(part)])
    return out


# ------------------------------------------------------------------ votes
@dataclass
class VoteParse:
    votes: dict[str, Vote]                 # sid -> vote (abstain for missing/invalid)
    missing: list[str]
    invalid: list[str]


_VOTE_LINE = re.compile(r"^\W*(s\d+)\W+(true|false|abstain)\W*$", re.I)


def parse_votes(raw: str, items: list[VerificationItem]) -> VoteParse:
    """One `<id> <vote>` per line; every complete line counts (a truncated tail loses only its own
    statements). A line naming an unknown id, or two lines with different votes for one id, is invalid."""
    sids = [it.sid for it in items]
    known = set(sids)
    got: dict[str, Vote] = {}
    invalid: list[str] = []
    conflict: set[str] = set()
    for line in (raw or "").splitlines():
        m = _VOTE_LINE.match(line.strip())
        if not m:
            continue
        sid, v = m.group(1), Vote(m.group(2).lower())
        if sid not in known:
            invalid.append(line.strip()[:200])
            continue
        if sid in got and got[sid] != v:
            conflict.add(sid)
        got.setdefault(sid, v)
    votes: dict[str, Vote] = {}
    missing: list[str] = []
    for sid in sids:
        if sid in conflict:
            votes[sid] = Vote.abstain
            invalid.append(f"{sid}: conflicting duplicate votes")
        elif sid in got:
            votes[sid] = got[sid]
        else:
            votes[sid] = Vote.abstain
            missing.append(sid)
    return VoteParse(votes, missing, invalid)


def split_votes(items: list[VerificationItem], vp: VoteParse) -> dict[FactKey, Vote]:
    """Votes keyed by fact."""
    return {it.fact: vp.votes[it.sid] for it in items}
