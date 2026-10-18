"""Split groups, collapse guard, two-way split (§13.1-13.2); dedup representatives (§6.7);
funnel reporting and selection-bias companions (§13.3); family overlap (§13.5); prominence (§13.6).

Pure functions over plain dicts; file writes only in `write_split_groups`. Sizing (§13.7) belongs
to hmb.stats.
"""
from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from itertools import combinations
from typing import Any, Iterable

import numpy as np
import yaml

from .hashing import sha256_bytes
from .store import Layout, read_json, write_json_immutable, write_mutable
from .vocab import Exclusion

SPLIT_METHOD = "python-random-shuffle-v2"   # v2 (2026-09-26): groups holding pre-freeze development books are development
SPLIT_VERSION = "split-1"
STAGES = ("S0", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8")
PROPOSAL_STAGES = ("S2", "S3", "S4", "S5", "S6", "S7", "S8")


class PipelineBug(RuntimeError):
    pass


# --------------------------------------------------------------------------- normalization
_GREEK = dict(zip("αβγδεζηθικλμνξοπρσςτυφχψω",
                  ["a", "b", "g", "d", "e", "z", "e", "th", "i", "k", "l", "m", "n", "x", "o", "p", "r", "s", "s",
                   "t", "y", "ph", "ch", "ps", "o"]))
_CYR = dict(zip("абвгдеёжзийклмнопрстуфхцчшщъыьэюяіѣѳѵ",
                ["a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "i", "k", "l", "m", "n", "o", "p", "r", "s", "t",
                 "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "iu", "ia", "i", "e", "f", "i"]))


def fold(s: str | None) -> str:
    """Script folding for work identity: NFKC, casefold, Greek/Cyrillic -> Latin, strip diacritics,
    drop punctuation, collapse whitespace. CJK characters are kept as-is (no romanization table)."""
    if not s:
        return ""
    s = unicodedata.normalize("NFKC", s).casefold()
    s = "".join(_GREEK.get(ch, _CYR.get(ch, ch)) for ch in unicodedata.normalize("NFD", s)
                if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def work_key(title: str | None, author: str | None) -> str | None:
    t, a = fold(title), fold(author)
    return f"{t}|{a}" if t and a else None


# --------------------------------------------------------------------------- §13.1 graph
def split_graph(books: list[dict], items: list[dict], th) -> dict:
    """Hard and soft edges over books; split groups = connected components over hard edges (§13.1).

    books: rights-manifest records (`hmb.schemas.RightsManifest` fields: book_id, work_id, display_title,
    author, publisher_id, publication_year). items: {item_id, book_id, phash_cluster_id} with the
    STORED cluster id (G7d pHash clustering); clusters are never recomputed here.

    Hard edges: equal work_id (§4.5: shared across editions, translations and reprints, so it is also
    the edition/translation edge); equal folded (display_title, author); a shared non-singleton
    phash_cluster_id. Soft edges: same folded author without a title match; same publisher_id with
    publication years within soft_edge_series_years. The manifest has no series field, so the
    "same publisher and series within ten years" edge uses the publisher_id + year proxy
    (`same_publisher_within_years`).
    """
    from .acquisition.phash import SINGLETON
    ids = sorted(b["book_id"] for b in books)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate book_id")
    by = {b["book_id"]: b for b in books}
    hard: list[dict] = []
    soft: list[dict] = []
    members: dict[str, set] = defaultdict(set)
    for it in items:
        cid = it.get("phash_cluster_id")
        if cid and cid != SINGLETON and it.get("book_id") in by:
            members[cid].add(it["book_id"])
    for cid, bs in sorted(members.items()):
        for a, b in combinations(sorted(bs), 2):
            hard.append({"a": a, "b": b, "type": "duplicate", "evidence": cid})
    years = th.value("soft_edge_series_years")
    for a, b in combinations(ids, 2):
        A, B = by[a], by[b]
        ka, kb = work_key(A.get("display_title"), A.get("author")), work_key(B.get("display_title"), B.get("author"))
        if A.get("work_id") and A.get("work_id") == B.get("work_id"):
            hard.append({"a": a, "b": b, "type": "work_identity", "evidence": "work_id"})
        elif ka and ka == kb:
            hard.append({"a": a, "b": b, "type": "work_identity", "evidence": "title+author"})
        if fold(A.get("author")) and fold(A.get("author")) == fold(B.get("author")) and not (ka and ka == kb):
            soft.append({"a": a, "b": b, "type": "same_author"})
        ya, yb = A.get("publication_year"), B.get("publication_year")
        if (A.get("publisher_id") and A.get("publisher_id") == B.get("publisher_id")
                and isinstance(ya, int) and isinstance(yb, int) and abs(ya - yb) <= years):
            soft.append({"a": a, "b": b, "type": "same_publisher_within_years"})
    groups = components(ids, [(e["a"], e["b"]) for e in hard])
    return {"groups": groups, "book_group": {b: g for g, bs in groups.items() for b in bs},
            "hard_edges": hard, "soft_edges": soft, "duplicate_edges_from": "stored phash_cluster_id (G7d)"}


def components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, list[str]]:
    parent = {n: n for n in nodes}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    out: dict[str, list[str]] = defaultdict(list)
    for n in parent:
        out[find(n)].append(n)
    return {"sg_" + sha256_bytes("\n".join(sorted(v)).encode())[:10]: sorted(v) for v in out.values()}


# --------------------------------------------------------------------------- collapse guard
def collapse_check(groups: dict[str, list[str]], eligible_by_book: dict[str, int], th) -> dict:
    total = sum(eligible_by_book.get(b, 0) for bs in groups.values() for b in bs)
    shares = {g: (sum(eligible_by_book.get(b, 0) for b in bs) / total if total else 0.0) for g, bs in groups.items()}
    largest = max(shares, key=lambda g: (shares[g], g)) if shares else None
    frac = th.value("collapse_frac")
    return {"total_eligible": total, "largest_group": largest, "largest_share": shares.get(largest, 0.0),
            "collapse_frac": frac, "collapsed": bool(largest) and shares[largest] > frac}


COLLAPSE_RULE = ("collapse-rule-2: drop the book with the smallest eligible count (ties: book_id) whose removal "
                 "splits the collapsed component and leaves no group above collapse_frac, never the component's "
                 "largest-eligible-count book; if no such drop works, single_group (owner standing rule D7, "
                 "floor D12, 2026-09-23)")


def apply_collapse_guard(books: list[dict], items: list[dict], eligible_by_book: dict[str, int], th) -> dict:
    """Graph after the §13.1 collapse guard, resolved by the owner's standing rule (COLLAPSE_RULE).

    No manual decision is read: when the largest group holds more than collapse_frac of eligible items,
    books of that component are tried in (eligible count, book_id) order and the first whose removal
    splits the component into >= 2 groups with none above collapse_frac is dropped. The last book in
    that order, the component's largest-eligible-count book, is never a candidate (D12). If none works
    the pilot becomes single-group and the held-out claim is abandoned."""
    g = split_graph(books, items, th)
    chk = collapse_check(g["groups"], eligible_by_book, th)
    g["collapse_guard"] = {**chk, "rule": COLLAPSE_RULE, "action": None, "held_out_claim": True}
    if not chk["collapsed"]:
        return g
    component = g["groups"][chk["largest_group"]]
    for drop in sorted(component, key=lambda b: (eligible_by_book.get(b, 0), b))[:-1]:
        g2 = split_graph([b for b in books if b["book_id"] != drop], [i for i in items if i["book_id"] != drop], th)
        if len({g2["book_group"][b] for b in component if b != drop}) < 2:
            continue
        chk2 = collapse_check(g2["groups"], {b: n for b, n in eligible_by_book.items() if b != drop}, th)
        if chk2["collapsed"]:
            continue
        g2["collapse_guard"] = {**chk, "rule": COLLAPSE_RULE, "action": "drop_book", "dropped_books": [drop],
                                "after_drop": chk2, "held_out_claim": True}
        return g2
    ids = sorted(b["book_id"] for b in books)
    g["groups"] = {"sg_single": ids}
    g["book_group"] = {b: "sg_single" for b in ids}
    g["collapse_guard"].update(action="single_group", held_out_claim=False)
    return g


# --------------------------------------------------------------------------- §13.2 two-way split
def assign_splits(graph: dict, seed: int, th, development_books: frozenset[str] | set[str] = frozenset()) -> dict:
    """Deterministic seeded assignment. Every group holding a pre-freeze development book (MS2/MS3) is
    development, because those books set levels and sizing and may never be test books; the remaining
    group ids, sorted and shuffled with random.Random(seed), fill the development split up to `dev_groups`,
    and the rest are test. single_group -> all test, no held-out claim."""
    gids = sorted(graph["groups"])
    if not graph.get("collapse_guard", {}).get("held_out_claim", True):
        return {"method": SPLIT_METHOD, "seed": seed, "assignment": {g: "test" for g in gids},
                "held_out_claim": False}
    forced = sorted(g for g in gids if set(graph["groups"][g]) & set(development_books))
    order = [g for g in gids if g not in forced]
    random.Random(seed).shuffle(order)
    k = int(th.value("dev_groups"))
    fill = max(0, k - len(forced))
    if len(order) <= fill:
        raise ValueError(f"{len(gids)} groups cannot give {max(k, len(forced))} development groups and a test split")
    dev = set(forced) | set(order[:fill])
    assign = {g: ("development" if g in dev else "test") for g in gids}
    n_test = len(gids) - len(dev)
    return {"method": SPLIT_METHOD, "seed": seed, "assignment": assign, "held_out_claim": True,
            "forced_development_groups": forced, "n_test_groups": n_test, "below_G_min": n_test < th.value("G_min")}


def soft_edge_crossings(graph: dict, split: dict) -> list[dict]:
    out = []
    for e in graph["soft_edges"]:
        sa = split["assignment"].get(graph["book_group"].get(e["a"]))
        sb = split["assignment"].get(graph["book_group"].get(e["b"]))
        if {sa, sb} == {"development", "test"}:
            out.append({**e, "splits": {e["a"]: sa, e["b"]: sb}})
    return out


def residual_cross_split_duplicates(items: list[dict], graph: dict, split: dict, max_hamming: int) -> list[str]:
    """Development-side item ids to drop: members of a graded-similarity pair that crosses splits."""
    from .acquisition.phash import hamming
    sp = {i["item_id"]: split["assignment"][graph["book_group"][i["book_id"]]] for i in items
          if i["book_id"] in graph["book_group"]}
    drop = set()
    ph = [i for i in items if i.get("phash64") and i["item_id"] in sp]
    for a, b in combinations(ph, 2):
        if sp[a["item_id"]] != sp[b["item_id"]] and hamming(a["phash64"], b["phash64"]) <= max_hamming:
            drop.add(a["item_id"] if sp[a["item_id"]] == "development" else b["item_id"])
    return sorted(drop)


def write_split_groups(layout: Layout, graph: dict, split: dict) -> str:
    doc = {"version": SPLIT_VERSION, "duplicate_edges_from": graph.get("duplicate_edges_from"), "method": split["method"], "seed": split["seed"],
           "held_out_claim": split["held_out_claim"],
           "groups": {g: {"books": bs, "split": split["assignment"][g]} for g, bs in sorted(graph["groups"].items())},
           "hard_edges": graph["hard_edges"], "soft_edges": graph["soft_edges"],
           "soft_edge_crossings": soft_edge_crossings(graph, split),
           "collapse_guard": graph.get("collapse_guard")}
    return write_mutable(layout.split_groups, yaml.safe_dump(doc, sort_keys=False, allow_unicode=True).encode())


# --------------------------------------------------------------------------- §6.7 dedup
def fold_label(s: str) -> str:
    s = unicodedata.normalize("NFD", s.casefold())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").strip()


def canonical_fact_set(facts: Iterable, entity_names: dict[str, str] | None = None) -> set:
    """Facts as (predicate, args) with entity ids replaced by canonical names and label text folded.

    Default entity canonicalization: an entity is named by the folded text of its `label` fact when
    it has one, otherwise it keeps its id. `entity_names` overrides."""
    fl = [(f["predicate"], f["args"]) if isinstance(f, dict) else (f[0], f[1]) for f in facts]
    names = dict(entity_names or {})
    if entity_names is None:
        for p, a in fl:
            if p == "label" and isinstance(a[0], str):
                names.setdefault(a[1], "L:" + fold_label(a[0]))

    def canon(x):
        if isinstance(x, (list, tuple)):
            return tuple(sorted(canon(y) for y in x))
        return names.get(x, x) if isinstance(x, str) else x
    out = set()
    for p, a in fl:
        args = tuple(("T:" + fold_label(x)) if (p in ("label", "measure_label") and i == 0) else canon(x)
                     for i, x in enumerate(a))
        out.add((p, args))
    return out


def jaccard(a: set, b: set) -> float:
    u = a | b
    return len(a & b) / len(u) if u else 1.0


def dedup_representatives(clusters: dict[str, list[str]], fact_sets: dict[str, Iterable], eligible_count: dict[str, int],
                          frag_index: dict[str, float | None], th,
                          entity_names: dict[str, dict[str, str]] | None = None,
                          similarity=None) -> dict[str, dict]:
    """§6.7: within each pHash cluster, items with Jaccard >= dedup_jaccard are redundant; per
    redundancy component the representative has the larger eligible count, then lower frag_index,
    then the smallest item_id. Others lose with `duplicate_representative_lost`.

    `similarity(a, b) -> float | None` overrides the fact-list Jaccard (items-finalize passes the
    cards' eligible-fact Jaccard, hmb.cards.rederive.eligible_jaccard); None counts as not redundant."""
    jmin = th.value("dedup_jaccard")
    out: dict[str, dict] = {}
    for cid, mem in sorted(clusters.items()):
        miss = [m for m in mem if m not in fact_sets or m not in eligible_count]
        if miss:
            raise PipelineBug(f"unresolved duplicate in {cid}: no frozen card for {miss}")
        if similarity is None:
            canon = {m: canonical_fact_set(fact_sets[m], (entity_names or {}).get(m)) for m in mem}
            sim = lambda a, b: jaccard(canon[a], canon[b])            # noqa: E731
        else:
            sim = similarity
        edges = [(a, b) for a, b in combinations(sorted(mem), 2) if (sim(a, b) or 0.0) >= jmin]
        for comp in components(mem, edges).values():
            rep = min(comp, key=lambda m: (-eligible_count[m],
                                           math.inf if frag_index.get(m) is None else frag_index[m], m))
            for m in comp:
                out[m] = {"cluster_id": cid, "is_cluster_representative": m == rep,
                          "representative": rep, "exclusion": None if m == rep else "duplicate_representative_lost"}
    return out


# --------------------------------------------------------------------------- pre-card selection (owner 2026-09-28)
SELECTION_RULE = ("card-select-1: per test split group (1) keep one scan: the book with the most candidates "
                  "(ties: smallest book_id); other books -> precard_duplicate; (2) per non-singleton pHash cluster "
                  "keep the member with the lowest seeded rank sha256('<seed>:<item_id>'); others -> "
                  "precard_duplicate; (3) quota Q = the largest integer with sum_g min(n_g, Q) <= budget; keep the Q "
                  "lowest-ranked; others -> selection_budget. Development groups are never touched. The rule never "
                  "reads a model output beyond the G8b pass that made the candidate")


def select_for_cards(cands: list[dict], dev_groups: set[str], budget: int, seed: int) -> dict:
    """Pre-card selection (owner instruction 2026-09-28: deduplicate the large groups aggressively and keep at
    most `budget` test candidates). cands: [{item_id, book_id, group, phash_cluster_id}].
    Returns {selected: [...], excluded: {item_id: reason}, quota, per_group, rule}."""
    from .acquisition.phash import SINGLETON
    rank = {c["item_id"]: sha256_bytes(f"{seed}:{c['item_id']}".encode()) for c in cands}
    excluded: dict[str, str] = {}
    kept: dict[str, list[str]] = defaultdict(list)
    by_group: dict[str, list[dict]] = defaultdict(list)
    for c in cands:
        by_group[c["group"]].append(c)
    for g, cs in sorted(by_group.items()):
        if g in dev_groups:
            continue
        per_book: dict[str, int] = defaultdict(int)
        for c in cs:
            per_book[c["book_id"]] += 1
        primary = min(per_book, key=lambda b: (-per_book[b], b))
        seen: set[str] = set()
        for c in sorted(cs, key=lambda c: rank[c["item_id"]]):
            cid = c.get("phash_cluster_id") or SINGLETON
            if c["book_id"] != primary or (cid != SINGLETON and cid in seen):
                excluded[c["item_id"]] = Exclusion.precard_duplicate.value
                continue
            seen.add(cid)
            kept[g].append(c["item_id"])
    sizes = {g: len(v) for g, v in kept.items()}
    quota = max(sizes.values(), default=0)
    while quota > 0 and sum(min(n, quota) for n in sizes.values()) > budget:
        quota -= 1
    for g, ids in kept.items():
        for iid in ids[quota:]:                     # ids are already in seeded-rank order
            excluded[iid] = Exclusion.selection_budget.value
    selected = sorted(c["item_id"] for c in cands if c["item_id"] not in excluded)
    per_group = {g: {"candidates": len(by_group[g]), "after_dedup": sizes.get(g, len(by_group[g])),
                     "selected": sum(1 for c in by_group[g] if c["item_id"] not in excluded),
                     "development": g in dev_groups} for g in sorted(by_group)}
    return {"selected": selected, "excluded": dict(sorted(excluded.items())), "quota": quota,
            "budget": budget, "seed": seed, "per_group": per_group, "rule": SELECTION_RULE}


# --------------------------------------------------------------------------- §13.3 funnel reporting
def stages_of_row(row: dict, card: dict | None = None) -> list[str]:
    """Proposal stages reached (nested). card: {built: bool, n_eligible_facts: int} or None."""
    g = {x["gate_id"]: x["passed"] for x in row.get("gates", [])}
    reached = ["S2"]
    # S3 "proposals matched" is vacuous with Paddle as the only region source (D3/D4), so S3 == S2
    conds = [True, g.get("G8b") is True, row.get("is_cluster_representative") is True,
             bool(card and card.get("built")), bool(card and (card.get("n_eligible_facts") or 0) >= 1),
             row.get("outcome") == "eligible"]
    for st, ok in zip(PROPOSAL_STAGES[1:], conds):
        if not ok:
            break
        reached.append(st)
    return reached


def funnel_table(units: list[dict], stratum: str | None = None, stages: tuple = PROPOSAL_STAGES) -> dict:
    """units: {stages: [...], <stratum>: value}. Returns {stratum_value: {stage: {n, retention, survival}}}."""
    groups: dict[Any, list[dict]] = defaultdict(list)
    for u in units:
        groups["all" if stratum is None else u.get(stratum)].append(u)
    out = {}
    for gv, us in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = {s: sum(1 for u in us if s in u["stages"]) for s in stages}
        tab, prev = {}, None
        first = n[stages[0]]
        for s in stages:
            tab[s] = {"n": n[s], "retention": (n[s] / n[prev] if prev and n[prev] else None),
                      "survival": (n[s] / first if first else None)}
            prev = s
        out[str(gv)] = tab
    return out


def per_gate_rates(rows: list[dict], gate_ids: Iterable[str]) -> dict:
    """Conditional pass rate (denominator = rows reaching the gate) and cumulative survival."""
    total = len(rows)
    out, alive = {}, total
    for gid in gate_ids:
        reach = [r for r in rows if any(x["gate_id"] == gid for x in r.get("gates", []))]
        passed = [r for r in reach if next(x for x in r["gates"] if x["gate_id"] == gid)["passed"]]
        out[gid] = {"reached": len(reach), "passed": len(passed),
                    "conditional_pass_rate": len(passed) / len(reach) if reach else None,
                    "cumulative_survival": len(passed) / total if total else None}
    return out


def survival_by_book(units: list[dict], from_stage: str = "S2", to_stage: str = "S8") -> dict:
    per: dict[str, list] = defaultdict(lambda: [0, 0])
    for u in units:
        if from_stage in u["stages"]:
            per[u["book_id"]][0] += 1
            per[u["book_id"]][1] += int(to_stage in u["stages"])
    surv = {b: (k / n if n else None) for b, (n, k) in per.items()}
    vals = [v for v in surv.values() if v is not None]
    med = float(np.median(vals)) if vals else None
    flags = {b: bool(med is not None and v is not None and med > 0 and (v > 2 * med or v < med / 2))
             for b, v in surv.items()}
    if med == 0:
        flags = {b: bool(v and v > 0) for b, v in surv.items()}
    return {"survival": surv, "median": med, "flag_2x": flags}


def s_adj(scores: dict[str, float], item_stratum: dict[str, str], n_s4_by_stratum: dict[str, int]) -> dict:
    """S_adj = sum_g w_g H_g with w_g = n_S4,g / n_S4 (assumes ignorable missingness within stratum)."""
    tot = sum(n_s4_by_stratum.values())
    if not tot:
        return {"S_adj": None, "reason": "n_S4 = 0"}
    by: dict[str, list[float]] = defaultdict(list)
    for i, s in scores.items():
        by[item_stratum[i]].append(s)
    missing = sorted(g for g, n in n_s4_by_stratum.items() if n > 0 and not by.get(g))
    if missing:
        return {"S_adj": None, "reason": "strata with S4 mass but no eligible scored item", "missing_strata": missing}
    val = sum((n / tot) * float(np.mean(by[g])) for g, n in n_s4_by_stratum.items() if n > 0)
    return {"S_adj": val, "weights": {g: n / tot for g, n in n_s4_by_stratum.items()},
            "H_g": {g: float(np.mean(v)) for g, v in by.items()}, "assumption": "ignorable missingness within stratum"}


def envelope(H: float, e: float) -> list[float]:
    """Assumption-free bounds for scores in [0, 1]: [H e, H e + (1 - e)]."""
    if not (0 <= H <= 1 and 0 <= e <= 1):
        raise ValueError("H and e must lie in [0, 1]")
    return [H * e, H * e + (1 - e)]


def smd(included: list[float], reference: list[float]) -> float | None:
    a = np.array([x for x in included if x is not None], float)
    b = np.array([x for x in reference if x is not None], float)
    if len(a) < 2 or len(b) < 2:
        return None
    pooled = math.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return None if pooled == 0 else float((a.mean() - b.mean()) / pooled)


def smd_table(included_rows: list[dict], g1_rows: list[dict], fields: Iterable[str], limit: float = 0.2) -> dict:
    out = {}
    for f in fields:
        v = smd([r["img_stats"].get(f) for r in included_rows], [r["img_stats"].get(f) for r in g1_rows])
        inc = [r["img_stats"].get(f) for r in included_rows if r["img_stats"].get(f) is not None]
        out[f] = {"smd": v, "flag": v is not None and abs(v) > limit,
                  "included_range": [min(inc), max(inc)] if inc else None}
    return out


def decile_table(g1_rows: list[dict], field: str) -> list[dict]:
    """Exclusion rate by decile of one statistic over all rows that have it (G1 rows)."""
    pairs = [(r["img_stats"].get(field), r.get("outcome") == "excluded") for r in g1_rows
             if r.get("img_stats", {}).get(field) is not None]
    if not pairs:
        return []
    x = np.array([p[0] for p in pairs], float)
    y = np.array([p[1] for p in pairs], bool)
    edges = np.quantile(x, np.linspace(0, 1, 11))
    idx = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    return [{"decile": d + 1, "lo": float(edges[d]), "hi": float(edges[d + 1]), "n": int((idx == d).sum()),
             "exclusion_rate": float(y[idx == d].mean()) if (idx == d).any() else None} for d in range(10)]


def composition_table(rows: list[dict], fields: Iterable[str]) -> dict:
    """Eligible vs excluded on pre-gate metadata. A bias indicator from unvalidated metadata, not a correction."""
    grp = {"eligible": [r for r in rows if r.get("outcome") == "eligible"],
           "excluded": [r for r in rows if r.get("outcome") == "excluded"]}
    out: dict[str, Any] = {"label": "bias indicator computed from unvalidated metadata, not a correction"}
    for name, rs in grp.items():
        stats = {}
        for f in fields:
            v = np.array([r["img_stats"][f] for r in rs if r.get("img_stats", {}).get(f) is not None], float)
            stats[f] = {"n": int(len(v)), "mean": float(v.mean()) if len(v) else None,
                        "median": float(np.median(v)) if len(v) else None}
        reasons: dict[str, int] = defaultdict(int)
        for r in rs:
            for x in r.get("exclusion_reasons", []):
                reasons[x] += 1
        out[name] = {"n": len(rs), "stats": stats, "exclusion_reasons": dict(reasons)}
    return out


# --------------------------------------------------------------------------- logistic model
def logistic_irls(X: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Plain IRLS for logit P(y=1) = X b. Reports convergence and quasi-separation."""
    n, p = X.shape
    b = np.zeros(p)
    converged = False
    for it in range(max_iter):
        eta = np.clip(X @ b, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        w = np.maximum(mu * (1 - mu), 1e-12)
        z = eta + (y - mu) / w
        XtW = X.T * w
        try:
            b_new = np.linalg.solve(XtW @ X, XtW @ z)
        except np.linalg.LinAlgError:
            b_new = np.linalg.lstsq(XtW @ X, XtW @ z, rcond=None)[0]
        if np.max(np.abs(b_new - b)) < tol:
            b = b_new
            converged = True
            break
        b = b_new
    eta = X @ b
    mu = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
    separation = bool(np.any((mu < 1e-8) | (mu > 1 - 1e-8)))
    return {"coef": b, "mu": mu, "converged": converged, "iterations": it + 1, "separation_suspected": separation}


def cluster_robust_se(X: np.ndarray, y: np.ndarray, mu: np.ndarray, clusters: np.ndarray) -> np.ndarray:
    """CR1 sandwich: bread = (X'WX)^-1, meat = sum_g s_g s_g', scaled by G/(G-1)."""
    w = mu * (1 - mu)
    bread = np.linalg.pinv((X.T * w) @ X)
    meat = np.zeros((X.shape[1], X.shape[1]))
    ug = np.unique(clusters)
    for g in ug:
        m = clusters == g
        s = X[m].T @ (y[m] - mu[m])
        meat += np.outer(s, s)
    G = len(ug)
    V = bread @ meat @ bread * (G / (G - 1) if G > 1 else 1.0)
    return np.sqrt(np.clip(np.diag(V), 0, None))


def exclusion_logit(g1_rows: list[dict], fields: list[str]) -> dict:
    """logit P(excluded) ~ standardized image statistics + book fixed effects; book-clustered SEs."""
    rows = [r for r in g1_rows if all(r.get("img_stats", {}).get(f) is not None for f in fields)]
    if len(rows) < len(fields) + 2:
        return {"fitted": False, "reason": "too few complete rows", "n": len(rows)}
    Xs = np.array([[r["img_stats"][f] for f in fields] for r in rows], float)
    sd = Xs.std(axis=0)
    keep = sd > 0
    fields_k = [f for f, k in zip(fields, keep) if k]
    Xs = (Xs[:, keep] - Xs[:, keep].mean(axis=0)) / sd[keep]
    books = sorted({r["book_id"] for r in rows})
    B = np.array([[1.0 if r["book_id"] == b else 0.0 for b in books[1:]] for r in rows]).reshape(len(rows), -1)
    X = np.hstack([np.ones((len(rows), 1)), Xs, B])
    y = np.array([r.get("outcome") == "excluded" for r in rows], float)
    fit = logistic_irls(X, y)
    se = cluster_robust_se(X, y, fit["mu"], np.array([r["book_id"] for r in rows]))
    coefs = {f: {"coef_per_sd": float(fit["coef"][1 + i]), "se_cluster_book": float(se[1 + i])}
             for i, f in enumerate(fields_k)}
    return {"fitted": True, "n": len(rows), "n_books": len(books), "coefficients": coefs,
            "dropped_constant_fields": [f for f, k in zip(fields, keep) if not k],
            "converged": fit["converged"], "separation_suspected": fit["separation_suspected"],
            "note": "book fixed effects; standard errors clustered by book (CR1)"}


# --------------------------------------------------------------------------- §13.5 family overlap
def family_overlap_matrix(contestants: dict[str, str], builders: dict[str, str], judges: dict[str, str],
                          judge_configs: dict[str, list[str]] | None = None) -> dict:
    """Rows: contestant families; columns: builder and judge families (True = overlap).
    For each contestant, list judge configurations in which its family is in neither role."""
    bf, jf = sorted(set(builders.values())), sorted(set(judges.values()))
    matrix = {c: {"builder": {f: fam == f for f in bf}, "judge": {f: fam == f for f in jf}}
              for c, fam in sorted(contestants.items())}
    configs = judge_configs or {"all_judges": sorted(judges)}
    clean = {}
    for c, fam in sorted(contestants.items()):
        ok = [cfg for cfg, js in sorted(configs.items())
              if fam not in set(builders.values()) and fam not in {judges[j] for j in js}]
        clean[c] = ok if ok else "none: this contestant's family appears as builder or judge in every configuration"
    return {"builder_families": bf, "judge_families": jf, "matrix": matrix, "clean_configurations": clean}


# --------------------------------------------------------------------------- §13.6 prominence
def prominence_terciles(manifests: dict[str, dict]) -> dict[str, int]:
    """Book-level prominence tercile 1..3 = ceil(3 * rank_pct) of (hosted count, fulltext) ascending;
    ties share the highest rank of their tie block."""
    keys = {}
    for b, m in manifests.items():
        p = m.get("prominence") or {}
        if not isinstance(p.get("hosted_by_major_aggregators"), int) or not isinstance(p.get("public_fulltext_layer"), bool):
            continue
        keys[b] = (p["hosted_by_major_aggregators"], int(p["public_fulltext_layer"]))
    n = len(keys)
    out = {}
    for b, k in keys.items():
        rank = sum(1 for v in keys.values() if v <= k)
        out[b] = max(1, math.ceil(3 * rank / n))
    return out


# --------------------------------------------------------------------------- drivers over stored artifacts
def latest_funnel_rows(layout: Layout) -> dict[str, dict]:
    from .store import iter_jsonl
    out: dict[str, dict] = {}
    for r in iter_jsonl(layout.funnel):
        out[r["proposal_id"]] = r
    return out


def card_status(layout: Layout, item_id: str | None) -> dict | None:
    """{built, n_eligible_facts, exclusion_reasons} from semantic_cards/ and data/card_gates/ (None if no item)."""
    if not item_id:
        return None
    cp = layout.card(item_id)
    if not cp.exists():
        return {"built": False, "n_eligible_facts": 0, "exclusion_reasons": []}
    card = read_json(cp)
    gp = layout.p("data", "card_gates", f"{item_id}.json")
    gates = read_json(gp) if gp.exists() else {}
    return {"built": True, "n_eligible_facts": sum(1 for f in card.get("facts", []) if f.get("eligible")),
            "exclusion_reasons": gates.get("exclusion_reasons", [])}


def _book_strata(layout: Layout) -> dict[str, dict]:
    from .rights import load_manifests
    out = {}
    for b, m in load_manifests(layout.rights_manifest).items():
        y = m.get("publication_year")
        out[b] = {"language": m.get("language"), "decade": (y // 10 * 10) if isinstance(y, int) else None,
                  "source_codec": (m.get("acquisition") or {}).get("source_codec"),
                  "repository": (m.get("acquisition") or {}).get("source_repository")}
    return out


def funnel_report(layout: Layout) -> dict:
    """§6.6/§13.3 funnel report from the stored funnel (latest row per proposal), page rows and cards."""
    from .acquisition.imgstats import STAT_FIELDS
    from .store import iter_jsonl
    rows = list(latest_funnel_rows(layout).values())
    strata = _book_strata(layout)
    units = []
    for r in rows:
        u = {"stages": stages_of_row(r, card_status(layout, r.get("item_id"))), "book_id": r["book_id"],
             "touching_cc": r.get("touching_cc"),
             **strata.get(r["book_id"], {})}
        units.append(u)
    pages = list(iter_jsonl(layout.p("manifests", "funnel_pages.jsonl")))
    books_cleared = sorted({p["book_id"] for p in pages
                            if any(g["gate_id"] == "G0a" and g["passed"] for g in p.get("gates", []))})
    pages_rendered = sum(1 for p in pages if p.get("page_index") is not None and any(
        g["gate_id"] == "G0b" and (g.get("measured_value") or {}).get("rendered") for g in p.get("gates", [])))
    g1_rows = [r for r in rows if r.get("img_stats")]
    included = [r for r in rows if r.get("outcome") == "eligible"]
    fields = [f for f in STAT_FIELDS if f not in ("eff_res_xheight_px", "skew_deg")]
    by = {k: funnel_table(units, k) for k in ("book_id", "language", "decade", "source_codec", "touching_cc")}
    return {
        "note": "counts per stage; retention = n_s/n_(s-1); survival from S2. S0/S1 from page rows.",
        "S0_books_cleared": len(books_cleared), "S1_pages_rendered": pages_rendered,
        "overall": funnel_table(units)["all"] if units else {},
        "by_stratum": by,
        "per_gate": per_gate_rates(rows, ["G2", "G3", "G4", "G5", "G6", "G7a", "G7b", "G7c", "G7d",
                                          "G8b"]),
        "crop_border": crop_border_stats(rows),
        "diagram_type": diagram_type_counts(rows),
        "figure_only_pages": figure_only_page_counts(pages, rows),
        "survival_by_book": survival_by_book(units),
        "smd_included_vs_g1": smd_table(included, g1_rows, fields) if included else
        {"status": "no eligible items yet"},
        "deciles": {f: decile_table(g1_rows, f) for f in fields},
        "composition": composition_table(rows, fields),
        "exclusion_logit": exclusion_logit(g1_rows, fields),
        "exclusion_reasons": dict(sorted(_count(x for r in rows for x in r.get("exclusion_reasons", [])).items())),
        "S_adj_envelope": {"status": "needs scores; computed at scoring time"},
    }


def diagram_type_counts(rows: list[dict]) -> dict:
    """Descriptive diagram-type counts over G8b survivors (owner decision 2026-09-25): the gatekeeper's
    predicted label, never a gate or quota. Survivors whose reply had no valid label are counted apart."""
    from .vocab import DiagramType
    surv = [r for r in rows if any(g["gate_id"] == "G8b" and g["passed"] for g in r.get("gates", []))]
    counts = {t.value: sum(1 for r in surv if r.get("diagram_type") == t.value) for t in DiagramType}
    return {"n_g8b_survivors": len(surv), "counts": counts,
            "unlabelled_invalid_reply": sum(1 for r in surv if r.get("diagram_type") is None),
            "note": "predicted by the G8b gatekeeper; descriptive metadata only"}


def figure_only_page_counts(pages: list[dict], rows: list[dict]) -> dict:
    """Figure-only pages (no prose, caption or body text) and their crops at each gate (C21: never excluded
    for being figure-only)."""
    fo = {(p["book_id"], p["page_index"]) for p in pages if p.get("figure_only")}
    crops = [r for r in rows if (r["book_id"], r["page_index"]) in fo]
    return {"n_pages": len(fo), "n_crops": len(crops),
            "crops_by_terminal_gate": dict(sorted(_count(r.get("terminal_gate") or "none" for r in crops).items())),
            "n_crops_reaching_g8b": sum(1 for r in crops if any(g["gate_id"] == "G8b" for g in r.get("gates", []))),
            "n_crops_passing_g8b": sum(1 for r in crops if any(g["gate_id"] == "G8b" and g["passed"]
                                                               for g in r.get("gates", [])))}


def crop_border_stats(rows: list[dict]) -> dict:
    """D10: G7a border measures over every row that reached G7a (recorded, never an exclusion)."""
    m = [r for r in rows if r.get("border_ink") is not None]
    worst = np.array([max(r["border_ink"]) for r in m], float)
    g7a = [next((g for g in r.get("gates", []) if g["gate_id"] == "G7a"), None) for r in m]
    return {"n_measured": len(m), "n_touching_cc": sum(1 for r in m if r.get("touching_cc")),
            "n_border_test_fires": sum(1 for g in g7a if g and (g.get("measured_value") or {}).get("offending")),
            "max_edge_border_ink": {"median": float(np.median(worst)), "p90": float(np.quantile(worst, 0.9)),
                                    "max": float(worst.max())} if len(m) else None,
            "note": "border test recorded only (D10); Paddle crops cannot grow"}


def _count(xs) -> dict:
    out: dict = defaultdict(int)
    for x in xs:
        out[x] += 1
    return dict(out)


def split_inputs(layout: Layout, eligible_item_ids: set[str] | None = None) -> tuple[list[dict], list[dict], dict]:
    """(books, items, eligible_by_book) for split_graph. Items are every funnel proposal with a stored
    cluster id (members of a cluster are hard edges whether or not they survive). Eligible counts come
    from `eligible_item_ids`, else from card-gate files with no exclusion reason."""
    from .rights import load_manifests
    rows = latest_funnel_rows(layout)
    mans = load_manifests(layout.rights_manifest)
    book_ids = sorted({r["book_id"] for r in rows.values()} & set(mans))
    books = [mans[b] for b in book_ids]
    items = [{"item_id": pid, "book_id": r["book_id"], "phash_cluster_id": r.get("phash_cluster_id")}
             for pid, r in rows.items() if r["book_id"] in mans]
    if eligible_item_ids is None:
        eligible_item_ids = set()
        gdir = layout.p("data", "card_gates")
        for p in sorted(gdir.glob("*.json")) if gdir.exists() else []:
            g = read_json(p)
            if not g.get("exclusion_reasons"):
                eligible_item_ids.add(g["item_id"])
    elig: dict[str, int] = defaultdict(int)
    for r in rows.values():
        if r.get("item_id") in eligible_item_ids:
            elig[r["book_id"]] += 1
    return books, items, dict(elig)


def _cmd_funnel_report(args) -> int:
    import json
    lay = Layout.from_env(args.root)
    rep = funnel_report(lay)
    out = lay.reports / "funnel_report.json"
    write_mutable(out, (json.dumps(rep, indent=1, ensure_ascii=False, default=str) + "\n").encode())
    print(json.dumps({"report": str(out), "S0": rep["S0_books_cleared"], "S1": rep["S1_pages_rendered"],
                      "overall": {k: v["n"] for k, v in rep["overall"].items()},
                      "exclusion_reasons": rep["exclusion_reasons"]}, indent=1))
    return 0


def register_cli(sub) -> None:
    p = sub.add_parser("funnel-report", help="§6.6/§13.3 funnel report -> reports/funnel_report.json")
    p.set_defaults(func=_cmd_funnel_report)


__all__ = ["fold", "work_key", "split_graph", "collapse_check", "apply_collapse_guard", "COLLAPSE_RULE",
           "assign_splits", "soft_edge_crossings", "residual_cross_split_duplicates",
           "write_split_groups", "canonical_fact_set", "jaccard", "dedup_representatives",
           "select_for_cards", "SELECTION_RULE", "stages_of_row", "funnel_table", "per_gate_rates", "survival_by_book", "s_adj", "envelope", "smd",
           "smd_table", "decile_table", "composition_table", "logistic_irls", "cluster_robust_se",
           "exclusion_logit", "family_overlap_matrix", "prominence_terciles"]
