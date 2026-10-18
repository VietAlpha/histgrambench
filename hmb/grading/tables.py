"""§11.4–11.8 aggregation and §14 inference over score records (historical tracks A and B only; Track C
is never pooled, CONTRACTS.md "Track values"). Simplification 2026-09-25; component 07 §11.3/§14.

Conventions (recorded):
* Common denominator: every eligible item has a cell for every model and track (non-valid outcomes
  score 0, a judge failure leaves facts INDETERMINATE). Nothing is dropped list-wise, so one model's
  numbers never depend on which other models are in the run.
* Headline: H = mean over split groups g of the mean item score in g. Its interval is the
  Ibragimov–Müller t-interval over the G group means (df = G − 1). A paired model difference is the
  same computation on the per-item differences, so it equals the difference of the two headlines.
* Mandatory diagnostic families: (1) validity (outcome codes, Valid@1); (2) per-class recall E/R/L
  with the contradicted and indeterminate shares, Q5 and the evaluator-failure rate. (The material-extra
  item rate was removed 2026-09-26 with the extras channel.)
* Missing inputs return None / an explicit status, never 0.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np

from hmb import stats
from hmb.schemas import ScoreRecord
from hmb.vocab import OUTCOME_GROUPS, REPORTED_TRACKS, Outcome

from .cell import CLASSES

Cells = Mapping[str, ScoreRecord]            # item_id -> record (one model, one track)


def _reported(recs: Iterable[ScoreRecord]) -> list[ScoreRecord]:
    out = list(recs)
    for r in out:
        if str(r.track) not in {t.value for t in REPORTED_TRACKS}:
            raise ValueError(f"track {r.track} is not a reported historical track; never pooled with A/B")
    return out


def _ok(r: ScoreRecord) -> bool:
    return str(r.outcome_code) == Outcome.OK.value


def complete_items(cells_by_model: Mapping[str, Cells], items: Sequence[str] | None = None) -> list[str]:
    """The common item set. Every model must have a cell for every item (non-valid outcomes are
    scored 0, never missing), so the set is the same for every model."""
    universe = sorted(items) if items is not None else sorted(set().union(*[set(c) for c in cells_by_model.values()]))
    for m in sorted(cells_by_model):
        _reported(cells_by_model[m].values())
        miss = [i for i in universe if i not in cells_by_model[m]]
        if miss:
            raise ValueError(f"model {m} has no cell for {len(miss)} items (non-valid outcomes must be scored 0)")
    return universe


# ------------------------------------------------------------------ family 1: validity (§9.5)
def outcome_distribution(recs: Iterable[ScoreRecord]) -> dict:
    rs = _reported(recs)
    n = len(rs)
    if n == 0:
        return {"status": "insufficient_items", "n": 0}
    cnt = {o.value: 0 for o in Outcome}
    for r in rs:
        cnt[str(r.outcome_code)] += 1
    return {"status": "ok", "n": n, "counts": cnt, "fractions": {k: v / n for k, v in cnt.items()}}


def derived_rates(recs: Iterable[ScoreRecord]) -> dict:
    """Valid@1, Compile@1, Unsafe/Invalid@1, Timeout@1, Extraction-failure@1, Protocol-violation@1
    over the same denominator n (eligible items)."""
    rs = _reported(recs)
    n = len(rs)
    if n == 0:
        return {"status": "insufficient_items", "n": 0}
    return {"status": "ok", "n": n, **{name: sum(str(r.outcome_code) in {o.value for o in grp} for r in rs) / n
                                        for name, grp in OUTCOME_GROUPS.items()}}


def valid_rate(cells: Cells, items: Sequence[str]) -> float | None:
    return (sum(_ok(cells[i]) for i in items) / len(items)) if items else None


# ------------------------------------------------------------------ headline (§11.8, §14.2)
def group_means(values: Mapping[str, float], items: Sequence[str],
                group_of: Mapping[str, Hashable]) -> tuple[dict, dict]:
    by: dict[Hashable, list[float]] = defaultdict(list)
    for i in items:
        by[group_of[i]].append(float(values[i]))
    return {g: float(np.mean(by[g])) for g in sorted(by, key=str)}, {g: len(by[g]) for g in sorted(by, key=str)}


def group_estimate(values: Mapping[str, float], items: Sequence[str], group_of: Mapping[str, Hashable],
                   th) -> dict:
    """Mean of the split-group means with the IM t-interval over them (the one inferential path)."""
    gm, n_g = group_means(values, items, group_of)
    if not gm:
        return {"status": "insufficient_items", "estimate": None, "G": 0}
    est = float(np.mean(list(gm.values())))
    im = stats.im_group_t(list(gm.values()), alpha=float(th.value("alpha")), G_min=int(th.value("G_min")))
    return {"status": im["status"], "estimate": est, "G": len(gm), "n_items": len(items), "n_per_group": n_g,
            "ci": im.get("ci"), "p_value": im.get("p_value"), "df": im.get("df"), "se": im.get("se"),
            "inferential_claim_allowed": im.get("inferential_claim_allowed"), "method": "ibragimov_mueller_group_t"}


def headline(cells: Cells, items: Sequence[str], group_of: Mapping[str, Hashable], th) -> dict:
    return group_estimate({i: cells[i].S for i in items}, items, group_of, th)


# ------------------------------------------------------------------ family 2: per-class diagnostics (§11.4)
def per_class_recall(cells: Cells, items: Sequence[str]) -> dict:
    """Per class: R^c = item-macro mean of g·R_i^c over items whose class has n_i^c* >= m_c, and the
    fact-level contradicted / indeterminate shares over valid cells."""
    out = {}
    for c in CLASSES:
        rows = [cells[i].per_class[c] for i in items if cells[i].per_class[c]["reportable"]]
        ok = [cells[i].per_class[c] for i in items if _ok(cells[i])]
        ns = sum(r["n_star"] for r in ok)
        out[c] = {"R": float(np.mean([r["R"] for r in rows])) if rows else None, "n_items": len(rows),
                  "contradicted_share": (sum(r["n_contradicted"] for r in ok) / ns) if ns else None,
                  "indeterminate_share": (sum(r["n_indeterminate"] for r in ok) / ns) if ns else None,
                  "n_items_zero_class": sum(cells[i].per_class[c]["n"] == 0 for i in items)}
    return out


def label_errors(cells: Cells, items: Sequence[str]) -> dict:
    """Label recall is strict string match only (§7.5); transcription / attachment / missing counts."""
    errs = {"attachment": 0, "transcription": 0, "missing": 0}
    for i in items:
        for k in errs:
            errs[k] += cells[i].per_class["L"].get("label_errors", {}).get(k, 0)
    return errs


def q5(cells: Cells, items: Sequence[str], th) -> dict:
    """Q5 = P(item has >= 1 CONTRADICTED relation | o = OK and R^E >= τ_E)."""
    tau = float(th.value("q5_tau_E"))
    cond = [i for i in items if _ok(cells[i]) and cells[i].per_class["E"]["R"] is not None
            and cells[i].per_class["E"]["R"] >= tau]
    if not cond:
        return {"status": "insufficient_items", "n": 0}
    k = sum(cells[i].per_class["R"]["n_contradicted"] >= 1 for i in cond)
    return {"status": "ok", "Q5": k / len(cond), "n": len(cond), "k": k}


def evaluator_failure_rate(cells: Cells, items: Sequence[str]) -> float | None:
    """Share of valid cells in which a judge call failed (their undecided facts count as not recovered)."""
    ok = [i for i in items if _ok(cells[i])]
    return (sum(cells[i].evaluator_failure is not None for i in ok) / len(ok)) if ok else None


# ------------------------------------------------------------------ §11.11 slices
def slice_publishable(n_items: int, n_groups: int, n_facts: int, th) -> dict:
    ok = (n_items >= int(th.value("slice_min_items")) and n_groups >= int(th.value("slice_min_groups"))
          and n_facts >= int(th.value("slice_min_facts")))
    return {"publish_scores": ok, "n_items": n_items, "n_groups": n_groups, "n_facts": n_facts,
            "mode": "scores" if ok else "counts_only"}


# ------------------------------------------------------------------ §14 paired inference
def paired_comparison(c1: Cells, c2: Cells, items: Sequence[str], group_of: Mapping[str, Hashable], th) -> dict:
    """IM group t on the per-item differences S(m1) − S(m2); the estimate equals H(m1) − H(m2)."""
    return group_estimate({i: c1[i].S - c2[i].S for i in items}, items, group_of, th)


def pairwise_partial_order(cells_by_model: Mapping[str, Cells], items: Sequence[str],
                           group_of: Mapping[str, Hashable], th) -> dict:
    """§14.4: all pairs, IM p-values, Benjamini–Hochberg at bh_q; unseparated models share a rank band.
    On Track A it is the G7 input (`separation_evidence`, D21)."""
    models = sorted(cells_by_model)
    pairs = {}
    for x in range(len(models)):
        for y in range(x + 1, len(models)):
            a, b = models[x], models[y]
            pc = paired_comparison(cells_by_model[a], cells_by_model[b], items, group_of, th)
            ok = pc.get("p_value") is not None
            pairs[(a, b)] = {"status": "ok" if ok else pc["status"], "p": pc.get("p_value"), "diff": pc["estimate"]}
    return stats.partial_order(models, pairs, q=float(th.value("bh_q")))


def separation_evidence(cells_by_model: Mapping[str, Cells], order: Mapping[str, Any] | None, items: Sequence[str],
                        group_of: Mapping[str, Hashable], th) -> dict:
    """G7 input (owner decision D21, 2026-09-26): every pair of the all-pairs partial order (IM p-values, BH
    at bh_q) with its separation, its difference and the §14.3 MDD floor from the pair's own item
    differences (ρ̂ = one-way ANOVA ICC by split group, σ̂_d). No second interval or test."""
    if not order:
        return {"status": "fewer_than_two_models"}
    models = sorted(cells_by_model)
    clusters = [group_of[i] for i in items]
    G = len(set(clusters))
    pairs = []
    for x in range(len(models)):
        for y in range(x + 1, len(models)):
            a, b = models[x], models[y]
            r = order["pairs"][f"{a}|{b}"]
            d = [cells_by_model[a][i].S - cells_by_model[b][i].S for i in items]
            icc = stats.icc_oneway(d, clusters) if len(d) > 1 else {"status": "insufficient_items"}
            sd = float(np.std(d, ddof=1)) if len(d) > 1 else None
            floor = None
            if icc.get("status") == "ok" and sd is not None and G >= 2:
                floor = stats.mdd(sd, G, float(icc["rho_hat"]), alpha=float(th.value("alpha")),
                                  power=float(th.value("power_target")))["MDD_floor"]
            pairs.append({"pair": [a, b], "status": r["status"], "diff": r["diff"], "bh_adjusted": r["bh_adjusted"],
                          "separated": r["separated"], "rho_hat": icc.get("rho_hat"), "sigma_d": sd,
                          "mdd_floor": floor})
    return {"status": "ok", "G": G, "q": order["q"], "family_size": order["family_size"], "pairs": pairs}


def ceiling_flag(cells_by_model: Mapping[str, Cells], items: Sequence[str], group_of: Mapping[str, Hashable],
                 th) -> dict:
    """§14.6: the top model has H > ceiling_flag_H and the mean within-group SD of its item scores is
    < ceiling_flag_sd (the sample is too easy to separate models)."""
    if not cells_by_model or not items:
        return {"status": "insufficient_items", "flag": None}
    H = {m: headline(c, items, group_of, th)["estimate"] for m, c in cells_by_model.items()}
    top = max(sorted(H), key=lambda m: H[m])
    by: dict[Hashable, list[float]] = defaultdict(list)
    for i in items:
        by[group_of[i]].append(cells_by_model[top][i].S)
    sds = [float(np.std(v, ddof=1)) for v in by.values() if len(v) > 1]
    if not sds:
        return {"status": "no_within_group_replication", "flag": None, "top_model": top}
    wsd = float(np.mean(sds))
    return {"status": "ok", "flag": bool(H[top] > float(th.value("ceiling_flag_H"))
                                        and wsd < float(th.value("ceiling_flag_sd"))),
            "top_model": top, "H_top": H[top], "within_group_sd": wsd}


# ------------------------------------------------------------------ §11.7 context effect (G8)
def context_effect(A: Mapping[str, Cells], B: Mapping[str, Cells], items: Sequence[str],
                   group_of: Mapping[str, Hashable], th) -> dict:
    """Track B minus Track A on the same items and the same card. Per model: ΔH and ΔValid@1. Pooled
    (the G8 input): the per-item difference averaged over models, IM interval over group means."""
    models = sorted(set(A) & set(B))
    if not models or not items:
        return {"status": "insufficient_items"}
    per = {m: {"delta_H": group_estimate({i: B[m][i].S - A[m][i].S for i in items}, items, group_of, th)["estimate"],
               "delta_valid": (sum(_ok(B[m][i]) for i in items) - sum(_ok(A[m][i]) for i in items)) / len(items)}
           for m in models}
    d = {i: float(np.mean([B[m][i].S - A[m][i].S for m in models])) for i in items}
    return {"status": "ok", "models": models, "per_model": per, "pooled": group_estimate(d, items, group_of, th)}


# ------------------------------------------------------------------ §11.14 table
def model_track_table(cells_by_model: Mapping[str, Cells], group_of: Mapping[str, Hashable], th, *,
                      items: Sequence[str] | None = None) -> dict:
    """The required per-model table for one track: headline with its interval and the two
    mandatory diagnostic families, plus the secondary partial order."""
    its = complete_items(cells_by_model, items)
    out: dict[str, Any] = {"N": len(its), "models": {}}
    for m in sorted(cells_by_model):
        c = cells_by_model[m]
        allrecs = [c[i] for i in its]
        out["models"][m] = {
            "headline": headline(c, its, group_of, th),
            "validity": {"Valid@1": valid_rate(c, its), "outcome_distribution": outcome_distribution(allrecs),
                         "derived_rates": derived_rates(allrecs)},
            "per_class": {"recall": per_class_recall(c, its), "label_errors": label_errors(c, its),
                          "Q5": q5(c, its, th), "evaluator_failure_rate": evaluator_failure_rate(c, its)},
        }
    out["partial_order"] = pairwise_partial_order(cells_by_model, its, group_of, th) if len(cells_by_model) > 1 else None
    return out
