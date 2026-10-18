"""§14 statistical analysis plan and §13.7 sizing simulation.

Plain functions over numpy/scipy. One inferential method for every reported interval and test: the
Ibragimov–Müller group t on split-group means (§14.2). No mixed models, no Bayesian hierarchy, no
pairs cluster bootstrap, no randomization companion. The remaining functions serve the MS3 sizing (§13.7). Missing inputs or too few groups return an explicit status
dict (e.g. ``{"status": "insufficient_groups", "G": 3}``), never an interval or 0.

Every successful result carries ``"status": "ok"``.
"""
from __future__ import annotations

import math
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np
from scipy import stats as _st

OK = "ok"


def _status(status: str, **kw) -> dict:
    return {"status": status, **kw}


def _arr(x: Iterable[float]) -> np.ndarray:
    return np.asarray(list(x), dtype=float)


# ============================================================ §14.2 primary: IM group t-test
def im_group_t(d_g: Sequence[float], alpha: float = 0.05, G_min: int | None = None) -> dict:
    """Ibragimov–Müller group t: one-sample t on split-group paired means with G−1 df.

    CI = d̄ ± t_{1−α/2, G−1} · s_d/√G; two-sided p from the same t distribution.
    ``G_min`` (§13.7 hard floor) only sets ``inferential_claim_allowed``; the arithmetic is shown.
    """
    d = _arr(d_g)
    G = int(d.size)
    if G < 2:
        return _status("insufficient_groups", G=G)
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    df = G - 1
    tcrit = float(_st.t.ppf(1 - alpha / 2, df))
    out = {"status": OK, "method": "ibragimov_mueller_group_t", "G": G, "df": df, "mean": mean,
           "sd": sd, "alpha": alpha, "t_crit": tcrit,
           "inferential_claim_allowed": (G >= G_min) if G_min is not None else None}
    if sd == 0.0:
        out.update(status="zero_variance", se=0.0, ci=None, t=None, p_value=None, reject=None)
        return out
    se = sd / math.sqrt(G)
    t_stat = mean / se
    p = float(2 * _st.t.sf(abs(t_stat), df))
    out.update(se=se, ci=[mean - tcrit * se, mean + tcrit * se], t=t_stat, p_value=p,
               reject=bool(p < alpha))
    return out


# ============================================================ §14.3 MDD (design check for G7, MS3 sizing)
def n_A(n_g: Sequence[int]) -> float:
    """Size-weighted mean group size n_A = Σ n_g² / N."""
    n = _arr(n_g)
    return float((n ** 2).sum() / n.sum())


def mdd(sigma_d: float, G: int, rho: float, *, N: float | None = None,
        n_g: Sequence[int] | None = None, alpha: float = 0.05, power: float = 0.80) -> dict:
    """§14.3 minimum detectable difference with t quantiles on G−1 df, exactly as written:

    MDD       = (t_{1−α/2,G−1} + t_{power,G−1}) · σ_d · √(DEFF/N)
    MDD_floor = (t_{1−α/2,G−1} + t_{power,G−1}) · σ_d · √(ρ/G)
    With ``n_g`` omitted, groups are taken as equal (n_A = N/G).
    """
    if G < 2:
        return _status("insufficient_groups", G=G)
    k = float(_st.t.ppf(1 - alpha / 2, G - 1) + _st.t.ppf(power, G - 1))
    out = {"status": OK, "G": G, "rho": rho, "sigma_d": sigma_d, "multiplier": k,
           "MDD_floor": k * sigma_d * math.sqrt(rho / G),
           "N_eff_ceiling": (G / rho) if rho > 0 else math.inf}
    if n_g is not None:
        N = float(sum(n_g))
        nA = n_A(n_g)
    elif N is not None:
        nA = N / G
    else:
        out.update(MDD=None, DEFF=None, N=None, n_A=None, N_eff=None)
        return out
    deff = 1 + (nA - 1) * rho
    out.update(N=N, n_A=nA, DEFF=deff, N_eff=N / deff, MDD=k * sigma_d * math.sqrt(deff / N))
    return out


# ============================================================ §14.4 BH and the partial order
def benjamini_hochberg(pvals: Sequence[float], q: float = 0.10) -> dict:
    p = _arr(pvals)
    m = int(p.size)
    if m == 0:
        return {"status": OK, "rejected": [], "adjusted": [], "q": q}
    order = np.argsort(p, kind="mergesort")
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adjusted = np.empty(m)
    adjusted[order] = adj_sorted
    below = np.nonzero(p[order] <= q * np.arange(1, m + 1) / m)[0]
    rejected = np.zeros(m, dtype=bool)
    if below.size:
        rejected[order[: below.max() + 1]] = True
    return {"status": OK, "rejected": rejected.tolist(), "adjusted": adjusted.tolist(), "q": q}


def partial_order(models: Sequence[str], pairs: Mapping[tuple[str, str], Mapping[str, Any]],
                  q: float = 0.10) -> dict:
    """§14.4 partial order over ALL M(M−1)/2 model pairs.

    ``pairs[(a, b)]`` holds ``p`` (the IM group-t p-value) and ``diff`` (d̄ = mean a − b), optionally
    ``status``. BH runs over the full family m = M(M−1)/2: a pair whose test is uncomputable (status
    != "ok", p None, or the pair missing) stays in the family with p = 1.0 and is never separated.
    A pair is separated iff BH rejects. Rank band of m = [1 + #models separated above m,
    M − #models separated below m].
    """
    models = list(models)
    family = [(models[x], models[y]) for x in range(len(models)) for y in range(x + 1, len(models))]
    rows: list[dict] = []
    for a, b in family:
        r = pairs.get((a, b))
        if r is None and (b, a) in pairs:           # normalise orientation to (a, b)
            rr = pairs[(b, a)]
            r = {**rr, "diff": -rr["diff"] if rr.get("diff") is not None else None}
        if r is None:
            r = {"status": "missing"}
        computable = r.get("status", OK) == OK and r.get("p") is not None and r.get("diff") not in (None, 0)
        rows.append({"a": a, "b": b, "r": r, "computable": computable,
                     "p": float(r["p"]) if computable else 1.0})
    bh = benjamini_hochberg([row["p"] for row in rows], q)
    edges, detail = [], {}
    for row, rej, adj in zip(rows, bh["rejected"], bh["adjusted"]):
        a, b, r = row["a"], row["b"], row["r"]
        sep = bool(row["computable"] and rej)
        detail[f"{a}|{b}"] = {"status": r.get("status", OK), "diff": r.get("diff"), "p_in_family": row["p"],
                              "bh_adjusted": adj, "separated": sep}
        if sep:
            edges.append((a, b) if r["diff"] > 0 else (b, a))
    M = len(models)
    above = {m: sum(1 for w, l in edges if l == m) for m in models}
    below = {m: sum(1 for w, l in edges if w == m) for m in models}
    return {"status": OK, "family_size": len(family), "separated": edges, "pairs": detail,
            "rank_bands": {m: [1 + above[m], M - below[m]] for m in models}, "q": q}


# ============================================================ G6 ranking stability
def kendall_tau_b(x: Sequence[float], y: Sequence[float]) -> dict:
    """Kendall τ_b with tie correction, O(n²) (orderings of ≤ tens of models)."""
    a, b = _arr(x), _arr(y)
    n = int(a.size)
    if n != b.size:
        raise ValueError("x and y differ in length")
    if n < 2:
        return _status("insufficient_items", n=n)
    conc = disc = tx = ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx, sy = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                tx += 1
            elif sy == 0:
                ty += 1
            elif sx == sy:
                conc += 1
            else:
                disc += 1
    denom = math.sqrt((conc + disc + tx) * (conc + disc + ty))
    if denom == 0:
        return _status("undefined_all_tied", n=n)
    return {"status": OK, "tau_b": (conc - disc) / denom, "n": n, "concordant": conc,
            "discordant": disc}


# ============================================================ §13.7 ICC, ρ_plan, sizing
def _groups_arrays(d: Sequence[float], groups: Sequence[Hashable]) -> list[np.ndarray]:
    buckets: dict[Hashable, list[float]] = {}
    for v, g in zip(d, groups):
        buckets.setdefault(g, []).append(float(v))
    return [np.asarray(buckets[g]) for g in sorted(buckets, key=str)]


def _icc_from_groups(arrs: list[np.ndarray]) -> dict:
    G = len(arrs)
    n_g = np.array([a.size for a in arrs], dtype=float)
    N = float(n_g.sum())
    if G < 2:
        return _status("insufficient_groups", G=G)
    if N - G <= 0:
        return _status("no_within_group_replication", G=G, N=N)
    grand = float(np.concatenate(arrs).mean())
    means = np.array([a.mean() for a in arrs])
    msb = float((n_g * (means - grand) ** 2).sum() / (G - 1))
    msw = float(sum(((a - a.mean()) ** 2).sum() for a in arrs) / (N - G))
    n0 = (N - float((n_g ** 2).sum()) / N) / (G - 1)
    denom = msb + (n0 - 1) * msw
    if denom <= 0:
        return _status("degenerate_zero_variance", G=G, N=N, MSB=msb, MSW=msw, n0=n0)
    rho = max(0.0, (msb - msw) / denom)
    return {"status": OK, "rho_hat": rho, "MSB": msb, "MSW": msw, "n0": n0, "G": G, "N": N}


def icc_oneway(d: Sequence[float], groups: Sequence[Hashable]) -> dict:
    """§13.7 one-way random-effects ANOVA on d_i by book:
    n₀ = (N − Σ n_g²/N)/(G − 1); ρ̂ = max(0, (MSB − MSW)/(MSB + (n₀ − 1)·MSW))."""
    return _icc_from_groups(_groups_arrays(d, groups))


def rho_plan(d: Sequence[float], groups: Sequence[Hashable], reps: int = 2000, seed: int = 0,
             pct: float = 80.0) -> dict:
    """ρ_plan = the 80th percentile of a book-level bootstrap distribution of ρ̂ (books resampled
    with replacement; a book drawn twice counts as two groups). Degenerate replicates are excluded
    and counted."""
    arrs = _groups_arrays(d, groups)
    G = len(arrs)
    if G < 2:
        return _status("insufficient_groups", G=G)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        pick = rng.integers(0, G, size=G)
        r = _icc_from_groups([arrs[i] for i in pick])
        if r["status"] == OK:
            vals.append(r["rho_hat"])
    if not vals:
        return _status("all_replicates_degenerate", G=G, reps=reps)
    return {"status": OK, "rho_plan": float(np.percentile(vals, pct)), "pct": pct, "reps": reps,
            "n_valid": len(vals), "seed": seed, "point": icc_oneway(d, groups)}


def _scale_sizes(draw: np.ndarray, N: int) -> np.ndarray:
    """Scale drawn cluster sizes to sum exactly N (largest remainder; every group ≥ 1)."""
    G = draw.size
    if N < G:
        raise ValueError(f"N={N} < G={G}")
    raw = draw / draw.sum() * N
    base = np.maximum(np.floor(raw).astype(int), 1)
    diff = N - int(base.sum())
    frac = raw - np.floor(raw)
    if diff > 0:
        for i in np.argsort(-frac, kind="mergesort")[:diff]:
            base[i] += 1
    elif diff < 0:
        for i in np.argsort(frac, kind="mergesort"):
            if diff == 0:
                break
            if base[i] > 1:
                base[i] -= 1
                diff += 1
    return base


def sizing_simulation(d: Sequence[float], groups: Sequence[Hashable], delta_star: float,
                      G_values: Sequence[int], grid: Sequence[int] = (150, 250, 350, 450),
                      reps: int = 2000, alpha: float = 0.05, power_target: float = 0.80,
                      seed: int = 0) -> dict:
    """§13.7 simulation-based sizing rule, from development-probe paired differences only.

    For each (N, G) and b = 1..reps: draw G group effects by resampling the dev book means of d
    (centred at the mean of the book means so that the simulated effect is exactly δ*; ASSUMPTION,
    the plan text leaves the centring implicit); draw n_g by resampling the empirical dev cluster
    sizes and scaling them to sum N; draw residuals from the pooled within-book residuals
    d_i − mean_book(d); d_i := δ* + u_g(i) + e_i; reject if the §14.2 IM group t-test rejects at α.
    power(N, G) = mean(reject); N*(G) = min{N : power ≥ power_target}, else "G binding".
    """
    arrs = _groups_arrays(d, groups)
    if len(arrs) < 2:
        return _status("insufficient_groups", G_dev=len(arrs))
    book_means = np.array([a.mean() for a in arrs])
    u_pool = book_means - book_means.mean()
    resid = np.concatenate([a - a.mean() for a in arrs])
    sizes = np.array([a.size for a in arrs], dtype=float)
    rng = np.random.default_rng(seed)
    table, n_star = [], {}
    for G in G_values:
        if G < 2:
            n_star[str(G)] = _status("insufficient_groups", G=G)
            continue
        tcrit = float(_st.t.ppf(1 - alpha / 2, G - 1))
        found = None
        for N in grid:
            rej = 0
            for _ in range(reps):
                n_g = _scale_sizes(sizes[rng.integers(0, sizes.size, size=G)], int(N))
                u = u_pool[rng.integers(0, u_pool.size, size=G)]
                e = resid[rng.integers(0, resid.size, size=int(N))]
                starts = np.concatenate([[0], np.cumsum(n_g)[:-1]])
                gm = delta_star + u + np.add.reduceat(e, starts) / n_g
                sd = gm.std(ddof=1)
                if sd > 0 and abs(gm.mean()) / (sd / math.sqrt(G)) > tcrit:
                    rej += 1
            pw = rej / reps
            table.append({"N": int(N), "G": int(G), "power": pw})
            if found is None and pw >= power_target:
                found = int(N)
        n_star[str(G)] = ({"status": OK, "N_star": found} if found is not None
                          else {"status": "G_binding", "N_star": None})
    return {"status": OK, "delta_star": delta_star, "alpha": alpha, "power_target": power_target,
            "reps": reps, "seed": seed, "grid": list(grid), "power": table, "N_star": n_star,
            "assumptions": ["group effects centred at the mean of dev book means",
                            "cluster sizes resampled then scaled to N by largest remainder, min 1"]}


def sizing_decision(rho_plan_value: float | None, sigma_d: float | None, G: int | None,
                    G_min: int = 6) -> dict:
    """§13.7 decision rules (text form). The 250-item rule of thumb applies only at G >= 12; below
    that the simulation N* decides. The redistribution option is an owner decision, never a
    default; the pilot is never enlarged by default."""
    if rho_plan_value is None or sigma_d is None or G is None:
        return _status("missing_parameter")
    if G < G_min:
        return {"status": OK, "decision": "no_inferential_claim",
                "reason": f"G={G} < G_min={G_min} (hard floor)"}
    if G >= 12 and rho_plan_value <= 0.05 and sigma_d <= 0.15:     # the rule of thumb is stated at G = 12
        return {"status": OK, "decision": "pilot_250_sufficient",
                "reason": "rho_plan <= 0.05 and sigma_d <= 0.15: 250 items suffice for delta* >= 0.05 at G = 12"}
    if rho_plan_value >= 0.15:
        return {"status": OK, "decision": "do_not_enlarge",
                "reason": "rho_plan >= 0.15: no N reaches delta* = 0.05; raise delta* and state it, or "
                          "redistribute N* across more books (explicit owner decision)"}
    return {"status": OK, "decision": "use_simulation_N_star",
            "reason": "between the rule-of-thumb regions; N* from sizing_simulation"}
