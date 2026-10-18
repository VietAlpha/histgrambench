"""§8.1 deterministic entity alignment with abstention.

Builder cards K_1..K_n are aligned, in the frozen builder order, to one accumulating canonical
entity set CANON:

    for b in order:
        cost[u][v] = -score(u, v)  for u in K_b.entities, v in CANON, same type, finite score
        match = hungarian(cost)                                      # scipy linear_sum_assignment
        for (u, v) in match:
            score(u, v) < THETA_MATCH           -> u becomes a new slot
            margin(u, v) < DELTA_MARGIN         -> u and v ambiguous (either side's margin)
            automorphic_tie(u, v)               -> u and v ambiguous (unless geometry clears both)
            else                                -> merge u into v
        every u not merged -> new slot (ambiguous if marked)

score = w_L*sim_label + w_G*sim_geom + w_R*sim_rel
  sim_label  normalize.sim_label (1 / 0.5 one confusable substitution / 0.5 neutral / -inf unrelated)
  sim_geom   exp(-d/sigma) on anchor centres for point-like entities; IoU of anchor boxes for
             extended entities (falls back to exp(-d/sigma) when both boxes are degenerate)
  sim_rel    multiset Jaccard of typed 1-WL signatures after `wl_rounds` rounds over the fact graph

Form choices (recorded):
  * "compatible types" = identical entity type.
  * margin(u, v) = min(score(u,v) - best alternative v' for u, score(u,v) - best alternative u' for v);
    +inf when no alternative exists.
  * automorphic tie: an unlabelled same-type twin of u in K_b (or of v in CANON) within distance
    sigma with an identical 1-WL colour after `wl_rounds` rounds (1-WL equivalence is the tractable
    proxy for a swapping automorphism). "Geometry alone clears both thresholds" means
    w_G*sim_geom(u,v) >= THETA_MATCH and w_G*(sim_geom(u,v) - sim_geom(twin side)) >= DELTA_MARGIN
    for every twin.
  * an ambiguous u still gets its own (ambiguous) slot so its facts are carried and are ineligible.
  * slot anchor = running mean of member anchors; slot label = first non-empty member label.
"""
from __future__ import annotations

import hashlib
import itertools
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment

from hmb.ontology import PREDICATES
from hmb.schemas import BuilderAnchor, BuilderOutput, CanonEntity

from .closure import FactKey, canon
from .normalize import sim_label

POINT_LIKE = frozenset({"point", "angle_marker", "right_angle_marked", "length_tick"})
PREFIX = {"point": "p", "segment": "s", "line": "l", "ray": "r", "circle": "c", "arc": "a",
          "ellipse": "el", "polygon": "g", "region": "rg", "axis": "ax", "arrow": "ar",
          "angle_marker": "am", "right_angle_marked": "ra", "length_tick": "lt", "curve": "cv",
          "solid": "so"}
_BIG = 1e9


@dataclass(frozen=True)
class AlignParams:
    w_L: float
    w_G: float
    w_R: float
    theta: float
    delta: float
    sigma: float
    wl_rounds: int

    @classmethod
    def from_thresholds(cls, th) -> "AlignParams":
        return cls(float(th.value("align_w_L")), float(th.value("align_w_G")), float(th.value("align_w_R")),
                   float(th.value("align_theta_match")), float(th.value("align_delta_margin")),
                   float(th.value("align_sigma")), int(th.value("wl_rounds")))

    def as_dict(self) -> dict:
        return {"w_L": self.w_L, "w_G": self.w_G, "w_R": self.w_R, "theta_match": self.theta,
                "delta_margin": self.delta, "sigma": self.sigma, "wl_rounds": self.wl_rounds}


# ------------------------------------------------------------------ typed 1-WL
def _h(obj) -> str:
    return hashlib.sha256(repr(obj).encode("utf-8")).hexdigest()[:16]


def _incidences(facts: Iterable, nodes: set[str]) -> dict[str, list[tuple]]:
    """node -> [(pred, my_role, other_role, other_node|None)] over entity-argument positions."""
    inc: dict[str, list[tuple]] = {n: [] for n in nodes}
    for pred, args in facts:
        p = PREDICATES.get(pred)
        if p is None:
            continue
        sym = {i for pair in p.symmetric_pairs for i in pair}
        roles: list[tuple[str, str]] = []
        for i, (kind, a) in enumerate(zip(p.args, args)):
            if kind == "ent" or (kind == "ent?" and a != "unknown"):
                roles.append(("s" if i in sym else str(i), a))
            elif kind == "set":
                roles.extend(("S", x) for x in a)
            elif kind == "seq":
                roles.extend((f"Q{j}", x) for j, x in enumerate(a))
        roles = [(r, a) for r, a in roles if a in inc]
        if p.fclass.value == "entity":
            continue                      # the type is the initial colour
        if len(roles) == 1:
            inc[roles[0][1]].append((pred, roles[0][0], None, None))
        for (ri, a), (rj, b) in itertools.permutations(roles, 2):
            inc[a].append((pred, ri, rj, b))
    return inc


def wl_colours(facts: Iterable, types: dict[str, str], rounds: int) -> list[dict[str, str]]:
    """Colours per round (round 0 = entity type). Content-hashed, so comparable across graphs."""
    facts = list(facts)
    nodes = set(types)
    inc = _incidences(facts, nodes)
    cols = [{n: _h(("t", types[n])) for n in nodes}]
    for _ in range(rounds):
        prev = cols[-1]
        cols.append({n: _h((prev[n], sorted((p, r1, r2, prev[o] if o else "")
                                            for p, r1, r2, o in inc[n]))) for n in nodes})
    return cols


def wl_signatures(facts: Iterable, types: dict[str, str], rounds: int) -> dict[str, Counter]:
    """Typed 1-WL signature multiset per node: (0, type) plus, for each round r >= 1, the incidence
    features (r, pred, my_role, other_role, colour_{r-1}(other))."""
    facts = list(facts)
    inc = _incidences(facts, set(types))
    cols = wl_colours(facts, types, rounds)
    sig: dict[str, Counter] = {}
    for n in types:
        c: Counter = Counter()
        c[(0, types[n])] = 1
        for r in range(1, rounds + 1):
            for p, r1, r2, o in inc[n]:
                c[(r, p, r1, r2, cols[r - 1][o] if o else "")] += 1
        sig[n] = c
    return sig


def multiset_jaccard(a: Counter, b: Counter) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 1.0
    num = sum(min(a[k], b[k]) for k in keys)
    den = sum(max(a[k], b[k]) for k in keys)
    return num / den if den else 1.0


# ------------------------------------------------------------------ geometry
def _dist(a: dict, b: dict) -> float:
    return math.hypot(a["x"] - b["x"], a["y"] - b["y"])


def _iou(a: dict, b: dict) -> float | None:
    if a["w"] * a["h"] <= 0 and b["w"] * b["h"] <= 0:
        return None
    ax0, ax1, ay0, ay1 = a["x"] - a["w"] / 2, a["x"] + a["w"] / 2, a["y"] - a["h"] / 2, a["y"] + a["h"] / 2
    bx0, bx1, by0, by1 = b["x"] - b["w"] / 2, b["x"] + b["w"] / 2, b["y"] - b["h"] / 2, b["y"] + b["h"] / 2
    iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def sim_geom(a: dict | None, b: dict | None, etype: str, sigma: float) -> float:
    if a is None or b is None:
        return 0.0
    if etype not in POINT_LIKE:
        iou = _iou(a, b)
        if iou is not None:
            return iou
    return math.exp(-_dist(a, b) / sigma)


# ------------------------------------------------------------------ alignment
@dataclass
class Slot:
    cid: str
    type: str
    label: str | None
    anchor: dict | None
    members: dict[str, str] = field(default_factory=dict)
    ambiguous: bool = False
    n: int = 0

    def add_anchor(self, a: dict) -> None:
        if self.anchor is None:
            self.anchor, self.n = dict(a), 1
            return
        self.n += 1
        self.anchor = {k: self.anchor[k] + (a[k] - self.anchor[k]) / self.n for k in ("x", "y", "w", "h")}


@dataclass
class Alignment:
    builder_order: list[str]
    slots: list[Slot]
    local_to_canon: dict[tuple[str, str], str]
    log: list[dict]
    builder_facts: dict[str, list[tuple[FactKey, bool]]]    # builder -> canon facts (+context flag)

    @property
    def types(self) -> dict[str, str]:
        return {s.cid: s.type for s in self.slots}

    @property
    def ambiguous(self) -> set[str]:
        return {s.cid for s in self.slots if s.ambiguous}

    def entities(self) -> list[CanonEntity]:
        return [CanonEntity(entity_id=s.cid, type=s.type, label=s.label,
                            anchor=BuilderAnchor(**s.anchor) if s.anchor else None,
                            ambiguous=s.ambiguous, members=dict(s.members)) for s in self.slots]


def _map_fact(pred: str, args: list, mp: dict[str, str]) -> FactKey:
    p = PREDICATES[pred]
    out = []
    for kind, a in zip(p.args, args):
        if kind == "ent" or (kind == "ent?" and a != "unknown"):
            out.append(mp[a])
        elif kind in ("set", "seq"):
            out.append([mp[x] for x in a])
        else:
            out.append(a)
    return canon((pred, out))


def align(outputs: dict[str, BuilderOutput | None], builder_order: list[str], params: AlignParams
          ) -> Alignment:
    """Align builder outputs (keyed by builder_id) in `builder_order`. Pure and deterministic."""
    slots: list[Slot] = []
    counters: Counter = Counter()
    l2c: dict[tuple[str, str], str] = {}
    log: list[dict] = []
    canon_facts: dict[FactKey, None] = {}          # ordered set: no cross-builder duplicates (fix #3)
    builder_facts: dict[str, list[tuple[FactKey, bool]]] = {}

    def new_slot(etype: str, label, anchor, b: str, lid: str, ambiguous: bool) -> Slot:
        counters[etype] += 1
        s = Slot(f"{PREFIX[etype]}_{counters[etype]}", etype, label, None, {b: lid}, ambiguous)
        s.add_anchor(anchor)
        slots.append(s)
        return s

    for b in builder_order:
        bo = outputs.get(b)
        if bo is None:
            builder_facts[b] = []
            log.append({"builder": b, "event": "no_output"})
            continue
        ents = sorted(bo.entities, key=lambda e: e.local_id)
        btypes = {e.local_id: e.type for e in ents}
        bfacts = [(f.predicate, f.args) for f in bo.facts]
        ctypes = {s.cid: s.type for s in slots}
        R = params.wl_rounds
        sig_b = wl_signatures(bfacts, btypes, R)
        sig_c = wl_signatures(list(canon_facts), ctypes, R)
        col_b = wl_colours(bfacts, btypes, R)[-1]
        col_c = wl_colours(list(canon_facts), ctypes, R)[-1]
        anc = {e.local_id: e.anchor.model_dump() for e in ents}

        S = np.full((len(ents), len(slots)), -np.inf)
        for i, u in enumerate(ents):
            for j, v in enumerate(slots):
                if u.type != v.type:
                    continue
                sl = sim_label(u.label, v.label)
                if sl == -math.inf:
                    continue
                S[i, j] = (params.w_L * sl + params.w_G * sim_geom(anc[u.local_id], v.anchor, u.type, params.sigma)
                           + params.w_R * multiset_jaccard(sig_b[u.local_id], sig_c[v.cid]))
        pairs: list[tuple[int, int]] = []
        if len(ents) and len(slots):
            cost = np.where(np.isfinite(S), -S, _BIG)
            rows, cols = linear_sum_assignment(cost)
            pairs = [(int(i), int(j)) for i, j in zip(rows, cols) if np.isfinite(S[i, j])]

        merged: dict[str, Slot] = {}
        marked: set[str] = set()
        for i, j in pairs:
            u, v = ents[i], slots[j]
            sc = float(S[i, j])
            rec = {"builder": b, "local_id": u.local_id, "slot": v.cid, "score": sc}
            if sc < params.theta:
                log.append({**rec, "event": "below_theta"})
                continue
            alt_u = [S[i, jj] for jj in range(len(slots)) if jj != j and np.isfinite(S[i, jj])]
            alt_v = [S[ii, j] for ii in range(len(ents)) if ii != i and np.isfinite(S[ii, j])]
            margin = min(sc - max(alt_u) if alt_u else math.inf, sc - max(alt_v) if alt_v else math.inf)
            rec["margin"] = margin
            if margin < params.delta:
                v.ambiguous = True
                marked.add(u.local_id)
                log.append({**rec, "event": "ambiguous_margin"})
                continue
            if _automorphic_tie(u, v, ents, slots, anc, col_b, col_c, params):
                v.ambiguous = True
                marked.add(u.local_id)
                log.append({**rec, "event": "ambiguous_automorphism"})
                continue
            merged[u.local_id] = v
            log.append({**rec, "event": "merge"})

        for u in ents:
            if u.local_id in merged:
                v = merged[u.local_id]
                v.members[b] = u.local_id
                v.add_anchor(anc[u.local_id])
                if v.label is None and u.label is not None:
                    v.label = u.label
                l2c[(b, u.local_id)] = v.cid
            else:
                s = new_slot(u.type, u.label, anc[u.local_id], b, u.local_id, u.local_id in marked)
                l2c[(b, u.local_id)] = s.cid
                log.append({"builder": b, "local_id": u.local_id, "slot": s.cid, "event": "new_slot",
                            "ambiguous": s.ambiguous})
        mp = {lid: l2c[(b, lid)] for lid in btypes}
        mapped = []
        for f in bo.facts:
            mapped.append((_map_fact(f.predicate, f.args, mp), bool(f.context_assisted)))
        builder_facts[b] = mapped
        canon_facts.update(dict.fromkeys(k for k, _ in mapped))
    return Alignment(list(builder_order), slots, l2c, log, builder_facts)


def _automorphic_tie(u, v, ents, slots, anc, col_b, col_c, p: AlignParams) -> bool:
    twins_u = [e for e in ents if e.local_id != u.local_id and e.type == u.type and u.label is None
               and e.label is None and _dist(anc[e.local_id], anc[u.local_id]) <= p.sigma
               and col_b[e.local_id] == col_b[u.local_id]]
    twins_v = [s for s in slots if s.cid != v.cid and s.type == v.type and v.label is None
               and s.label is None and s.anchor is not None and v.anchor is not None
               and _dist(s.anchor, v.anchor) <= p.sigma and col_c.get(s.cid) == col_c.get(v.cid)]
    if not twins_u and not twins_v:
        return False
    g = p.w_G * sim_geom(anc[u.local_id], v.anchor, u.type, p.sigma)
    if g < p.theta:
        return True
    for t in twins_u:
        if g - p.w_G * sim_geom(anc[t.local_id], v.anchor, u.type, p.sigma) < p.delta:
            return True
    for t in twins_v:
        if g - p.w_G * sim_geom(anc[u.local_id], t.anchor, u.type, p.sigma) < p.delta:
            return True
    return False
