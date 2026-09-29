"""Tests for aedrover.analysis.stats, cross-checked against scipy and analytic cases."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats as sps

from aedrover.analysis.stats import (
    BCA_MAX_N,
    bootstrap_ci,
    compare_groups,
    holm_correction,
    mann_whitney,
    n_per_group_for_d,
    paired_t,
    welch_t,
    wilcoxon_signed_rank,
    wilson_ci,
)


@pytest.fixture(scope="module")
def two_groups() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(2024)
    return rng.normal(0.5, 1.0, 40), rng.normal(0.0, 1.6, 55)  # unequal n and variances


@pytest.fixture(scope="module")
def pairs() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(99)
    base = rng.normal(10.0, 2.0, 30)
    return base + rng.normal(0.4, 0.8, 30), base


# ---------------------------------------------------------------------------------------------
# Welch and paired t
# ---------------------------------------------------------------------------------------------
def test_welch_matches_scipy(two_groups):
    a, b = two_groups
    ours = welch_t(a, b)
    ref = sps.ttest_ind(a, b, equal_var=False)
    ci = ref.confidence_interval(0.95)
    assert ours.t == pytest.approx(ref.statistic, rel=1e-12)
    assert ours.p == pytest.approx(ref.pvalue, rel=1e-10)
    assert ours.df == pytest.approx(ref.df, rel=1e-12)
    assert ours.diff_ci_low == pytest.approx(ci.low, rel=1e-10)
    assert ours.diff_ci_high == pytest.approx(ci.high, rel=1e-10)
    assert ours.mean_diff == pytest.approx(a.mean() - b.mean())
    assert (ours.n, ours.n1, ours.n2) == (95, 40, 55)
    assert ours.test == "welch"


def test_paired_matches_scipy(pairs):
    a, b = pairs
    ours = paired_t(a, b)
    ref = sps.ttest_rel(a, b)
    ci = ref.confidence_interval(0.95)
    assert ours.t == pytest.approx(ref.statistic, rel=1e-12)
    assert ours.p == pytest.approx(ref.pvalue, rel=1e-10)
    assert ours.df == ref.df == 29
    assert ours.diff_ci_low == pytest.approx(ci.low, rel=1e-10)
    assert ours.diff_ci_high == pytest.approx(ci.high, rel=1e-10)
    assert ours.n == 30


def test_cohens_d_pooled_analytic():
    a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    res = welch_t(a, b)
    assert res.mean_diff == pytest.approx(-1.0)
    assert res.d == pytest.approx(-1.0 / math.sqrt(2.5))  # pooled SD = sqrt(2.5)
    assert res.df == pytest.approx(8.0)  # equal n and variance: Welch df = 2n - 2
    assert res.t == pytest.approx(-1.0 / math.sqrt(2.5 / 5 + 2.5 / 5))


def test_cohens_dz_paired_analytic():
    a = np.array([3.0, 5.0, 4.0, 8.0, 6.0])
    b = np.array([2.0, 3.0, 3.0, 5.0, 3.0])  # differences 1, 2, 1, 3, 3
    res = paired_t(a, b)
    diffs = a - b
    assert res.mean_diff == pytest.approx(2.0)
    assert res.d == pytest.approx(2.0 / diffs.std(ddof=1))
    assert res.t == pytest.approx(res.d * math.sqrt(5))


def test_d_confidence_interval_is_the_noncentral_t_pivot(two_groups):
    a, b = two_groups
    res = welch_t(a, b)
    n1, n2 = a.size, b.size
    scale = math.sqrt(n1 * n2 / (n1 + n2))
    df = n1 + n2 - 2
    sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / df)
    assert res.d == pytest.approx((a.mean() - b.mean()) / sp)
    t_pooled = res.d * scale
    assert sps.nct.sf(t_pooled, df, res.d_ci_low * scale) == pytest.approx(0.025, abs=1e-7)
    assert sps.nct.cdf(t_pooled, df, res.d_ci_high * scale) == pytest.approx(0.025, abs=1e-7)
    assert res.d_ci_low < res.d < res.d_ci_high


def test_paired_d_ci_pivot_and_normal_approximation(pairs):
    a, b = pairs
    res = paired_t(a, b)
    n = a.size
    assert sps.nct.sf(res.t, n - 1, res.d_ci_low * math.sqrt(n)) == pytest.approx(0.025, abs=1e-7)
    assert sps.nct.cdf(res.t, n - 1, res.d_ci_high * math.sqrt(n)) == pytest.approx(
        0.025, abs=1e-7
    )
    se = math.sqrt(1 / n + res.d**2 / (2 * n))  # Hedges-Olkin approximation, large-n check
    assert res.d_ci_low == pytest.approx(res.d - 1.96 * se, abs=0.08)
    assert res.d_ci_high == pytest.approx(res.d + 1.96 * se, abs=0.08)


def test_null_difference_gives_zero_effect_and_symmetric_intervals():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([4.0, 3.0, 2.0, 1.0])
    res = welch_t(a, b)
    assert res.mean_diff == 0.0 and res.t == 0.0
    assert res.p == pytest.approx(1.0)
    assert res.d == 0.0
    assert res.d_ci_low == pytest.approx(-res.d_ci_high, abs=1e-8)
    assert res.diff_ci_low == pytest.approx(-res.diff_ci_high)


def test_confidence_level_argument_widens_interval(two_groups):
    a, b = two_groups
    r95, r99 = welch_t(a, b, 0.95), welch_t(a, b, 0.99)
    assert r99.diff_ci_low < r95.diff_ci_low and r99.diff_ci_high > r95.diff_ci_high
    assert r99.confidence == 0.99


def test_constant_data_returns_nan_instead_of_raising():
    res = welch_t([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert res.mean_diff == -1.0
    assert math.isnan(res.t) and math.isnan(res.p) and math.isnan(res.d)
    assert res.diff_ci_low == res.diff_ci_high == -1.0
    resp = paired_t([1.0, 2.0, 3.0], [0.0, 1.0, 2.0])  # constant difference of exactly 1
    assert resp.mean_diff == 1.0 and math.isnan(resp.p)


def test_input_validation():
    with pytest.raises(ValueError):
        welch_t([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        welch_t([1.0, math.nan], [1.0, 2.0])
    with pytest.raises(ValueError):
        paired_t([1.0, 2.0, 3.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        welch_t(np.ones((2, 2)), [1.0, 2.0])


def test_type_i_error_of_welch_is_near_nominal():
    rng = np.random.default_rng(0)
    p = [welch_t(rng.normal(0, 1, 15), rng.normal(0, 3, 30)).p for _ in range(1500)]
    assert np.mean(np.array(p) < 0.05) == pytest.approx(0.05, abs=0.02)


# ---------------------------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------------------------
def test_bootstrap_bca_is_seeded_and_brackets_the_estimate():
    x = np.random.default_rng(5).normal(3.0, 2.0, 120)
    r1 = bootstrap_ci(x, n_resamples=2000, seed=11)
    r2 = bootstrap_ci(x, n_resamples=2000, seed=11)
    r3 = bootstrap_ci(x, n_resamples=2000, seed=12)
    assert (r1.low, r1.high) == (r2.low, r2.high)
    assert (r1.low, r1.high) != (r3.low, r3.high)
    assert r1.method == "BCa"
    assert r1.low < x.mean() == r1.estimate < r1.high


def test_bootstrap_mean_ci_close_to_t_interval():
    x = np.random.default_rng(6).normal(0.0, 1.0, 200)
    res = bootstrap_ci(x, n_resamples=4000, seed=1)
    lo, hi = sps.t.interval(0.95, x.size - 1, loc=x.mean(), scale=sps.sem(x))
    assert res.low == pytest.approx(lo, abs=0.03)
    assert res.high == pytest.approx(hi, abs=0.03)


def test_bootstrap_paired_two_sample_statistic():
    rng = np.random.default_rng(8)
    base = rng.normal(0.3, 0.1, 150)
    other = base + rng.normal(0.05, 0.02, 150)

    def mean_diff(x, y, axis=-1):
        return np.mean(x - y, axis=axis)

    res = bootstrap_ci(other, base, statistic=mean_diff, paired=True, n_resamples=2000, seed=2)
    assert res.estimate == pytest.approx(0.05, abs=0.01)
    assert 0.0 < res.low < res.estimate < res.high


def test_bootstrap_constant_data_is_degenerate():
    res = bootstrap_ci(np.full(20, 0.7))
    assert res.low == res.high == res.estimate == pytest.approx(0.7)
    assert res.method == "degenerate"


def test_bootstrap_falls_back_to_percentile_for_large_n():
    x = np.random.default_rng(1).normal(size=BCA_MAX_N + 10)
    res = bootstrap_ci(x, n_resamples=200, seed=0)
    assert res.method.startswith("percentile")
    assert res.low < res.estimate < res.high


def test_bootstrap_percentile_method_and_validation():
    x = np.random.default_rng(3).normal(size=60)
    assert bootstrap_ci(x, method="percentile", n_resamples=500).method == "percentile"
    with pytest.raises(ValueError):
        bootstrap_ci()
    with pytest.raises(ValueError):
        bootstrap_ci([1.0])


# ---------------------------------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------------------------------
def _wilson_by_hand(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


@pytest.mark.parametrize("k,n", [(10, 20), (81, 263), (3, 7), (99, 100), (1, 50)])
def test_wilson_matches_hand_formula_and_scipy(k, n):
    lo, hi = wilson_ci(k, n)
    hlo, hhi = _wilson_by_hand(k, n)
    ref = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert (lo, hi) == (pytest.approx(hlo, abs=1e-12), pytest.approx(hhi, abs=1e-12))
    assert lo == pytest.approx(ref.low, abs=1e-12) and hi == pytest.approx(ref.high, abs=1e-12)


def test_wilson_known_values():
    lo, hi = wilson_ci(10, 20)
    assert (lo, hi) == (pytest.approx(0.29930, abs=1e-5), pytest.approx(0.70070, abs=1e-5))
    lo, hi = wilson_ci(81, 263)  # Newcombe (1998) worked example
    assert (lo, hi) == (pytest.approx(0.2553, abs=1e-4), pytest.approx(0.3662, abs=1e-4))


def test_wilson_boundaries_and_arrays():
    lo, hi = wilson_ci(0, 10)
    z2 = 1.959963984540054**2
    assert lo == 0.0
    assert hi == pytest.approx(z2 / (10 + z2))  # analytic upper limit for k = 0
    lo, hi = wilson_ci(10, 10)
    assert hi == 1.0 and lo == pytest.approx(10 / (10 + z2))
    los, his = wilson_ci(np.array([1, 5, 9]), 10)
    assert los.shape == his.shape == (3,) and np.all(los < his)
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(0, 0)


# ---------------------------------------------------------------------------------------------
# Holm
# ---------------------------------------------------------------------------------------------
def test_holm_hand_example():
    adj = holm_correction([0.01, 0.04, 0.03, 0.005])
    # sorted 0.005, 0.01, 0.03, 0.04 -> x4, x3, x2, x1 = 0.02, 0.03, 0.06, 0.04 -> running max
    assert adj == pytest.approx([0.03, 0.06, 0.06, 0.02])


def test_holm_properties():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 0.2, 12)
    adj = holm_correction(p)
    assert np.all(adj >= p) and np.all(adj <= np.minimum(1.0, p.size * p) + 1e-15)
    assert np.all(adj <= 1.0)
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= -1e-15)  # monotone in the raw ordering
    assert holm_correction([0.2]) == pytest.approx([0.2])
    assert holm_correction([0.9, 0.95]) == pytest.approx([1.0, 1.0])


def test_holm_nan_and_validation():
    adj = holm_correction([0.01, math.nan, 0.02])
    assert adj[0] == pytest.approx(0.02) and adj[2] == pytest.approx(0.02)  # m = 2
    assert math.isnan(adj[1])
    assert holm_correction([math.nan]).shape == (1,)
    with pytest.raises(ValueError):
        holm_correction([1.2])
    with pytest.raises(ValueError):
        holm_correction([[0.1, 0.2]])


# ---------------------------------------------------------------------------------------------
# Mann-Whitney / Wilcoxon
# ---------------------------------------------------------------------------------------------
def test_mann_whitney_matches_scipy(two_groups):
    a, b = two_groups
    ours = mann_whitney(a, b)
    ref = sps.mannwhitneyu(a, b, alternative="two-sided")
    assert ours.u == ref.statistic
    assert ours.p == pytest.approx(ref.pvalue, rel=1e-12)
    assert ours.rank_biserial == pytest.approx(2 * ref.statistic / (a.size * b.size) - 1)
    assert ours.prob_superiority == pytest.approx(ref.statistic / (a.size * b.size))


def test_mann_whitney_fully_separated_exact_case():
    hi = mann_whitney([4, 5, 6], [1, 2, 3])
    assert hi.u == 9.0 and hi.rank_biserial == 1.0 and hi.prob_superiority == 1.0
    assert hi.p == pytest.approx(2 / 20)  # exact two-sided p = 2 / C(6, 3)
    lo = mann_whitney([1, 2, 3], [4, 5, 6])
    assert lo.u == 0.0 and lo.rank_biserial == -1.0


def test_mann_whitney_identical_distributions_zero_effect():
    res = mann_whitney([1, 2, 3, 4], [1, 2, 3, 4])
    assert res.rank_biserial == pytest.approx(0.0)
    assert res.prob_superiority == pytest.approx(0.5)
    assert res.p == pytest.approx(1.0)


def test_wilcoxon_matches_scipy_and_analytic_effect(pairs):
    a, b = pairs
    ours = wilcoxon_signed_rank(a, b)
    ref = sps.wilcoxon(a, b)
    assert ours.p == pytest.approx(ref.pvalue, rel=1e-12)
    assert min(ours.w_plus, ours.w_minus) == ref.statistic
    assert ours.w_plus + ours.w_minus == pytest.approx(a.size * (a.size + 1) / 2)
    small = wilcoxon_signed_rank([1, 2, 3, 4, -5], [0, 0, 0, 0, 0])
    assert (small.w_plus, small.w_minus) == (10.0, 5.0)
    assert small.rank_biserial == pytest.approx(1 / 3)


def test_wilcoxon_drops_zero_differences():
    res = wilcoxon_signed_rank([1, 2, 3, 4], [1, 2, 0, 0])
    assert res.n_pairs == 4 and res.n_nonzero == 2
    same = wilcoxon_signed_rank([1, 2, 3], [1, 2, 3])
    assert same.p == 1.0 and same.rank_biserial == 0.0


# ---------------------------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("d,expected", [(0.2, 394), (0.5, 64), (0.8, 26)])
def test_n_per_group_known_values(d, expected):
    """Cohen's small/medium/large effects at alpha 0.05, power 0.80 (exact non-central t)."""
    assert n_per_group_for_d(d) == expected


def test_n_per_group_close_to_normal_approximation_and_monotone():
    z = sps.norm.ppf(0.975) + sps.norm.ppf(0.80)  # z_(1-alpha/2) + z_power
    for d in (0.3, 0.5, 1.0):
        approx = 2 * (z / d) ** 2
        assert approx <= n_per_group_for_d(d) <= approx + 3
    assert n_per_group_for_d(0.5, power=0.9) > n_per_group_for_d(0.5, power=0.8)
    assert n_per_group_for_d(0.5, alpha=0.01) > n_per_group_for_d(0.5, alpha=0.05)
    assert n_per_group_for_d(-0.5) == n_per_group_for_d(0.5)


def test_n_per_group_achieves_requested_power_and_is_minimal():
    n = n_per_group_for_d(0.5)
    tcrit = sps.t.ppf(0.975, 2 * n - 2)
    ncp = 0.5 * math.sqrt(n / 2)
    assert sps.nct.sf(tcrit, 2 * n - 2, ncp) + sps.nct.cdf(-tcrit, 2 * n - 2, ncp) >= 0.80
    n0 = n - 1
    tcrit0 = sps.t.ppf(0.975, 2 * n0 - 2)
    ncp0 = 0.5 * math.sqrt(n0 / 2)
    assert sps.nct.sf(tcrit0, 2 * n0 - 2, ncp0) + sps.nct.cdf(-tcrit0, 2 * n0 - 2, ncp0) < 0.80


def test_n_per_group_paired_and_validation():
    assert n_per_group_for_d(0.5, paired=True) == 34  # pairs for d_z = 0.5
    assert n_per_group_for_d(0.5, paired=True) < n_per_group_for_d(0.5)
    with pytest.raises(ValueError):
        n_per_group_for_d(0.0)
    with pytest.raises(ValueError):
        n_per_group_for_d(0.5, alpha=1.5)


# ---------------------------------------------------------------------------------------------
# compare_groups
# ---------------------------------------------------------------------------------------------
def test_compare_groups_independent_record(two_groups):
    a, b = two_groups
    rec = compare_groups(a, b)
    ref_t = sps.ttest_ind(a, b, equal_var=False)
    ref_u = sps.mannwhitneyu(a, b)
    assert rec["test"] == "welch" and rec["nonparam_test"] == "mann_whitney_u"
    assert rec["n_a"] == 40 and rec["n_b"] == 55
    assert rec["t"] == pytest.approx(ref_t.statistic) and rec["p_t"] == pytest.approx(ref_t.pvalue)
    assert rec["nonparam_stat"] == ref_u.statistic
    assert rec["p_nonparam"] == pytest.approx(ref_u.pvalue)
    assert rec["mean_diff"] == pytest.approx(a.mean() - b.mean())
    assert rec["median_a"] == pytest.approx(np.median(a))
    assert rec["sd_b"] == pytest.approx(b.std(ddof=1))
    assert -1.0 <= rec["rank_biserial"] <= 1.0
    assert rec["d_ci_low"] < rec["cohen_d"] < rec["d_ci_high"]


def test_compare_groups_paired_record(pairs):
    a, b = pairs
    rec = compare_groups(a, b, paired=True)
    ref_t = sps.ttest_rel(a, b)
    ref_w = sps.wilcoxon(a, b)
    assert rec["test"] == "paired" and rec["nonparam_test"] == "wilcoxon_signed_rank"
    assert rec["p_t"] == pytest.approx(ref_t.pvalue)
    assert rec["p_nonparam"] == pytest.approx(ref_w.pvalue)
    assert rec["nonparam_stat"] == ref_w.statistic
    assert rec["cohen_d"] == pytest.approx((a - b).mean() / (a - b).std(ddof=1))


def test_compare_groups_is_table_ready(two_groups):
    a, b = two_groups
    rec = compare_groups(a, b)
    assert all(isinstance(v, (int, float, str)) for v in rec.values())
    assert list(rec) == list(compare_groups(b, a))  # stable column order
    with pytest.raises(ValueError):
        compare_groups(a[:5], b[:6], paired=True)
