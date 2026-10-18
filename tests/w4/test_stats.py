"""§14 / §13.7 statistics with independently derived expectations (hand arithmetic in comments)."""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sst

from hmb import stats


def test_im_group_t_by_hand():
    # d = [0.1, 0.2, 0.3, -0.1]; mean 0.125; squared deviations .000625+.005625+.030625+.050625 = .0875
    # s² = .0875/3 = .0291667, s = .1707825, se = s/2 = .0853913; t_{.975,3} = 3.1824463 (table)
    r = stats.im_group_t([0.1, 0.2, 0.3, -0.1])
    assert r["status"] == "ok" and r["df"] == 3
    assert r["mean"] == pytest.approx(0.125)
    assert r["sd"] == pytest.approx(0.1707825, abs=1e-7)
    assert r["t_crit"] == pytest.approx(3.1824463, abs=1e-6)
    half = 3.1824463 * 0.0853913
    assert r["ci"] == pytest.approx([0.125 - half, 0.125 + half], abs=1e-6)
    assert r["t"] == pytest.approx(0.125 / 0.0853913, abs=1e-5)


def test_im_insufficient_and_g_min_flag():
    assert stats.im_group_t([0.3]) == {"status": "insufficient_groups", "G": 1}
    r = stats.im_group_t([0.1, 0.2, 0.3], G_min=6)
    assert r["status"] == "ok" and r["inferential_claim_allowed"] is False


# ------------------------------------------------------------------ §14.3 illustration table
PLAN_ROWS = [  # G, rho, N_eff ceiling, MDD at N=450, MDD floor (sigma_d = 0.20), as printed
    (10, 0.05, 200, 0.047, 0.040),
    (10, 0.15, 67, 0.073, 0.069),
    (12, 0.10, 120, 0.063, 0.056),
    (12, 0.15, 80, 0.074, 0.069),
    (15, 0.15, 100, 0.066, 0.060),
]


@pytest.mark.parametrize("G,rho,neff,mdd,floor", PLAN_ROWS)
def test_mdd_table_rows(G, rho, neff, mdd, floor):
    r = stats.mdd(0.20, G, rho, N=450)
    assert round(r["N_eff_ceiling"]) == neff
    if G >= 12:
        # the §14.3 formula (t quantiles on G−1 df) reproduces these rows at printed precision
        assert round(r["MDD"], 3) == mdd and round(r["MDD_floor"], 3) == floor
    else:
        # DISCREPANCY (reported, not tuned): the G = 10 rows are reproduced only with normal
        # quantiles (1.960 + 0.842), not with the t quantiles the formula specifies.
        assert round(r["MDD"], 3) != mdd
        z = sst.norm.ppf(0.975) + sst.norm.ppf(0.80)
        nA = 450 / G
        assert round(z * 0.2 * math.sqrt((1 + (nA - 1) * rho) / 450), 3) == mdd
        assert round(z * 0.2 * math.sqrt(rho / G), 3) == floor


# ------------------------------------------------------------------ §14.4 BH + partial order
def test_bh_by_hand():
    # sorted p .01 .03 .04 .20 vs q·k/m = .025 .05 .075 .10 -> reject the first three
    r = stats.benjamini_hochberg([0.01, 0.04, 0.03, 0.2], q=0.10)
    assert r["rejected"] == [True, True, True, False]
    # adjusted: .2; .04·4/3 = .05333; min(.03·2, .05333) = .05333; .01·4 = .04
    assert r["adjusted"] == pytest.approx([0.04, 0.0533333, 0.0533333, 0.2], abs=1e-6)


def test_partial_order_bh_only():
    # separation is the BH decision on the IM p-values, nothing else (simplification 2026-09-25)
    pairs = {("a", "b"): {"p": 0.001, "diff": 0.2},                  # separated, a ahead
             ("a", "c"): {"p": 0.30, "diff": 0.05},                  # not rejected
             ("b", "c"): {"p": 0.004, "diff": -0.2}}                 # separated, c ahead
    r = stats.partial_order(["a", "b", "c"], pairs)
    assert r["separated"] == [("a", "b"), ("c", "b")] and r["family_size"] == 3
    assert r["rank_bands"] == {"a": [1, 2], "b": [3, 3], "c": [1, 2]}
    # an uncomputable pair stays in the family with p = 1 and is never separated
    r2 = stats.partial_order(["a", "b", "c"], {("a", "b"): {"p": 0.001, "diff": 0.2}})
    assert r2["separated"] == [("a", "b")] and r2["pairs"]["a|c"]["p_in_family"] == 1.0


# ------------------------------------------------------------------ §14.5, §13.7
def test_kendall_tau_b_matches_scipy_with_ties():
    x, y = [1, 2, 2, 3, 4, 5], [2, 1, 3, 3, 5, 4]
    assert stats.kendall_tau_b(x, y)["tau_b"] == pytest.approx(sst.kendalltau(x, y).statistic)


def test_icc_oneway_by_hand():
    # groups [1,2,3], [4,5,6]: MSB = 3·1.5²·2/1 = 13.5, MSW = 4/4 = 1, n0 = (6 − 18/6)/1 = 3
    # ρ̂ = (13.5 − 1)/(13.5 + 2·1) = 12.5/15.5
    r = stats.icc_oneway([1, 2, 3, 4, 5, 6], ["a", "a", "a", "b", "b", "b"])
    assert (r["MSB"], r["MSW"], r["n0"]) == pytest.approx((13.5, 1.0, 3.0))
    assert r["rho_hat"] == pytest.approx(12.5 / 15.5)
    # unequal sizes 2 and 4: n0 = (6 − 20/6)/1 = 2.6667
    assert stats.icc_oneway([1, 2, 3, 4, 5, 6], ["a", "a", "b", "b", "b", "b"])["n0"] == pytest.approx(8 / 3)
    # ρ̂ truncated at 0 when MSB < MSW
    assert stats.icc_oneway([1, 5, 1, 5], ["a", "a", "b", "b"])["rho_hat"] == 0.0
    assert stats.icc_oneway([1, 2], ["a", "a"])["status"] == "insufficient_groups"


def test_rho_plan_and_sizing_simulation():
    rng = np.random.default_rng(3)
    groups = sum([[g] * 30 for g in "abc"], [])
    d = rng.normal(0, 0.1, size=90)
    rp = stats.rho_plan(d, groups, reps=200, seed=1)
    assert rp["status"] == "ok" and 0 <= rp["rho_plan"] <= 1 and rp["pct"] == 80
    big = stats.sizing_simulation(d, groups, delta_star=0.2, G_values=[12], reps=200, seed=2)
    assert big["N_star"]["12"] == {"status": "ok", "N_star": 150}
    null = stats.sizing_simulation(d, groups, delta_star=0.0, G_values=[12], grid=(150,), reps=400, seed=2)
    assert null["N_star"]["12"]["status"] == "G_binding"
    assert null["power"][0]["power"] < 0.1               # ≈ α under δ* = 0
    assert stats.sizing_simulation(d, ["a"] * 90, 0.1, [12])["status"] == "insufficient_groups"


def test_sizing_decision_rules():
    assert stats.sizing_decision(0.04, 0.12, 12)["decision"] == "pilot_250_sufficient"
    assert stats.sizing_decision(0.16, 0.20, 12)["decision"] == "do_not_enlarge"
    assert stats.sizing_decision(0.10, 0.20, 5)["decision"] == "no_inferential_claim"
    assert stats.sizing_decision(None, 0.2, 12)["status"] == "missing_parameter"
