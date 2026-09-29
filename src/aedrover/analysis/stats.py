"""Frequentist comparison utilities used for the results tables.

Everything is a thin, tested layer over numpy/scipy. Conventions:

* Two-sided tests, ``confidence = 0.95`` unless stated.
* ``mean_diff`` is always ``mean(a) - mean(b)``; positive means the first sample is larger.
* Effect sizes: Cohen's ``d`` with the pooled SD for independent samples, ``d_z`` (mean
  difference over SD of the differences) for paired samples. The 95% CI of ``d`` is the exact
  non-central-t pivot interval (Steiger and Fouladi), falling back to the large-sample normal
  approximation of Hedges and Olkin if root finding fails. For Welch tests the CI of ``d`` is
  still the pooled-variance one (an approximation when the variances differ).
* NaN inputs are rejected (``ValueError``); a zero-variance comparison yields NaN t/p/d fields
  instead of raising, so a results table with a constant column still renders.
"""

from __future__ import annotations

import inspect
import math
import warnings
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import optimize, stats

FloatArray = NDArray[np.float64]

# BCa needs a leave-one-out (jackknife) array of size n x (n - 1); above this n it is skipped.
BCA_MAX_N = 4000


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _sample(x: ArrayLike, name: str, min_n: int = 2) -> FloatArray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < min_n:
        raise ValueError(f"{name} needs at least {min_n} observations")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _ncp_bounds(t_obs: float, df: float, confidence: float) -> tuple[float, float]:
    """Non-centrality parameters (lambda_low, lambda_high) whose non-central t brackets t_obs.

    ``P(T >= t_obs; lambda_low) = alpha/2`` and ``P(T <= t_obs; lambda_high) = alpha/2``.
    """
    half = (1.0 - confidence) / 2.0

    def g_low(lam: float) -> float:
        return float(stats.nct.sf(t_obs, df, lam)) - half  # increasing in lambda

    def g_high(lam: float) -> float:
        return float(stats.nct.cdf(t_obs, df, lam)) - half  # decreasing in lambda

    def root(g: Callable[[float], float]) -> float:
        width = 8.0
        for _ in range(8):
            lo, hi = t_obs - width, t_obs + width
            g_lo, g_hi = g(lo), g(hi)
            if math.isfinite(g_lo) and math.isfinite(g_hi) and g_lo * g_hi < 0.0:
                return float(optimize.brentq(g, lo, hi, xtol=1e-10))
            width *= 2.0
        raise RuntimeError("could not bracket the non-centrality parameter")

    return root(g_low), root(g_high)


def _d_ci(
    d: float, t_obs: float, df: float, scale: float, se_normal: float, confidence: float
) -> tuple[float, float]:
    """CI for Cohen's d: non-central-t pivot, normal approximation as fallback.

    ``scale`` converts the non-centrality parameter to d (``sqrt(n1 n2 / (n1 + n2))`` for two
    samples, ``sqrt(n)`` for paired data).
    """
    try:
        lam_lo, lam_hi = _ncp_bounds(t_obs, df, confidence)
        return lam_lo / scale, lam_hi / scale
    except (RuntimeError, ValueError):
        z = float(stats.norm.ppf(0.5 + confidence / 2.0))
        return d - z * se_normal, d + z * se_normal


# ---------------------------------------------------------------------------------------------
# t tests
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TTestResult:
    """Outcome of a Welch or paired t test.

    Attributes:
        test: "welch" or "paired".
        n: total observations (n1 + n2 for Welch, number of pairs for paired).
        n1, n2: group sizes (both equal to the pair count for paired).
        mean_diff: mean(a) - mean(b) (mean of the paired differences for paired).
        se_diff: standard error of ``mean_diff``.
        df: degrees of freedom (Welch-Satterthwaite, or n - 1 for paired).
        t: test statistic.
        p: exact two-sided p value from the t distribution.
        d: Cohen's d (pooled SD) for Welch, d_z for paired.
        d_ci_low, d_ci_high: confidence interval of ``d``.
        diff_ci_low, diff_ci_high: t-based confidence interval of ``mean_diff``.
        confidence: confidence level of both intervals.
    """

    test: str
    n: int
    n1: int
    n2: int
    mean_diff: float
    se_diff: float
    df: float
    t: float
    p: float
    d: float
    d_ci_low: float
    d_ci_high: float
    diff_ci_low: float
    diff_ci_high: float
    confidence: float


def welch_t(a: ArrayLike, b: ArrayLike, confidence: float = 0.95) -> TTestResult:
    """Welch's unequal-variance t test of ``mean(a) - mean(b)`` with effect size and CIs."""
    x, y = _sample(a, "a"), _sample(b, "b")
    n1, n2 = x.size, y.size
    v1, v2 = float(np.var(x, ddof=1)), float(np.var(y, ddof=1))
    diff = float(np.mean(x) - np.mean(y))
    se2 = v1 / n1 + v2 / n2
    if se2 == 0.0:
        nan = math.nan
        return TTestResult("welch", n1 + n2, n1, n2, diff, 0.0, nan, nan, nan, nan, nan, nan,
                           diff, diff, confidence)
    se = math.sqrt(se2)
    df = se2**2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    t = diff / se
    p = float(2.0 * stats.t.sf(abs(t), df))
    tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df))

    sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2)
    if sp2 == 0.0:
        d = d_lo = d_hi = math.nan
    else:
        sp = math.sqrt(sp2)
        d = diff / sp
        scale = math.sqrt(n1 * n2 / (n1 + n2))
        t_pooled = d * scale
        se_norm = math.sqrt((n1 + n2) / (n1 * n2) + d * d / (2.0 * (n1 + n2)))
        d_lo, d_hi = _d_ci(d, t_pooled, n1 + n2 - 2, scale, se_norm, confidence)
    return TTestResult("welch", n1 + n2, n1, n2, diff, se, df, t, p, d, d_lo, d_hi,
                       diff - tcrit * se, diff + tcrit * se, confidence)


def paired_t(a: ArrayLike, b: ArrayLike, confidence: float = 0.95) -> TTestResult:
    """Paired t test on ``a - b`` with d_z and CIs (``a`` and ``b`` must have equal length)."""
    x, y = _sample(a, "a"), _sample(b, "b")
    if x.size != y.size:
        raise ValueError("paired samples must have equal length")
    diffs = x - y
    n = diffs.size
    diff = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    if sd == 0.0:
        nan = math.nan
        return TTestResult("paired", n, n, n, diff, 0.0, n - 1.0, nan, nan, nan, nan, nan,
                           diff, diff, confidence)
    se = sd / math.sqrt(n)
    df = n - 1.0
    t = diff / se
    p = float(2.0 * stats.t.sf(abs(t), df))
    tcrit = float(stats.t.ppf(0.5 + confidence / 2.0, df))
    dz = diff / sd
    se_norm = math.sqrt(1.0 / n + dz * dz / (2.0 * n))
    d_lo, d_hi = _d_ci(dz, t, df, math.sqrt(n), se_norm, confidence)
    return TTestResult("paired", n, n, n, diff, se, df, t, p, dz, d_lo, d_hi,
                       diff - tcrit * se, diff + tcrit * se, confidence)


# ---------------------------------------------------------------------------------------------
# bootstrap, proportions, multiplicity
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BootstrapCI:
    """Bootstrap confidence interval of a statistic.

    Attributes:
        estimate: statistic on the original data.
        low, high: interval limits.
        confidence: confidence level.
        method: interval method actually used ("BCa" or "percentile", with a suffix if BCa
            could not be applied).
        n_resamples: number of bootstrap resamples.
        seed: seed of the numpy generator.
    """

    estimate: float
    low: float
    high: float
    confidence: float
    method: str
    n_resamples: int
    seed: int


def bootstrap_ci(
    *samples: ArrayLike,
    statistic: Callable[..., ArrayLike] = np.mean,
    confidence: float = 0.95,
    n_resamples: int = 9999,
    seed: int = 0,
    method: str = "BCa",
    paired: bool = False,
) -> BootstrapCI:
    """Seeded bootstrap CI (BCa by default) via ``scipy.stats.bootstrap``.

    Args:
        *samples: one array, or several arrays (resampled independently unless ``paired``).
        statistic: vectorised callable ``statistic(*samples, axis=-1)``; default ``np.mean``.
        confidence: confidence level.
        n_resamples: bootstrap resamples.
        seed: seed for ``numpy.random.default_rng``.
        method: "BCa" or "percentile". BCa is skipped (percentile used, reported in the result)
            when a sample has more than :data:`BCA_MAX_N` points, or when BCa is undefined for
            the data.
        paired: resample the samples jointly (same indices); needs equal lengths.

    A constant dataset returns a zero-width interval at the point estimate.
    """
    if not samples:
        raise ValueError("at least one sample is required")
    data = tuple(_sample(s, f"sample {i}", min_n=2) for i, s in enumerate(samples))
    estimate = float(np.asarray(statistic(*data, axis=-1)))
    if all(np.ptp(s) == 0.0 for s in data):
        return BootstrapCI(estimate, estimate, estimate, confidence, "degenerate", 0, seed)

    used = method
    if method == "BCa" and max(s.size for s in data) > BCA_MAX_N:
        used = "percentile (BCa skipped: n > BCA_MAX_N)"
    total = sum(s.size for s in data)
    batch = int(max(1, min(n_resamples, 5_000_000 // max(total, 1))))
    rng_kw = "rng" if "rng" in inspect.signature(stats.bootstrap).parameters else "random_state"

    def run(m: str) -> tuple[float, float]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = stats.bootstrap(
                data,
                statistic,
                n_resamples=n_resamples,
                confidence_level=confidence,
                method=m,
                paired=paired,
                vectorized=True,
                batch=batch,
                **{rng_kw: np.random.default_rng(seed)},
            )
        return float(res.confidence_interval.low), float(res.confidence_interval.high)

    kind = "BCa" if used == "BCa" else "percentile"
    low, high = run(kind)
    if not (math.isfinite(low) and math.isfinite(high)) and kind == "BCa":
        used = "percentile (BCa undefined for these data)"
        low, high = run("percentile")
    return BootstrapCI(estimate, low, high, confidence, used, n_resamples, seed)


def wilson_ci(
    k: ArrayLike, n: ArrayLike, confidence: float = 0.95
) -> tuple[float | FloatArray, float | FloatArray]:
    """Wilson score interval for a binomial proportion ``k / n``.

    ``(low, high)`` with ``centre = (p + z^2/2n) / (1 + z^2/n)`` and half-width
    ``z sqrt(p(1-p)/n + z^2/4n^2) / (1 + z^2/n)``. Accepts scalars or arrays.
    """
    kk = np.asarray(k, dtype=float)
    nn = np.asarray(n, dtype=float)
    if np.any(nn < 1) or np.any(kk < 0) or np.any(kk > nn):
        raise ValueError("need 0 <= k <= n and n >= 1")
    z = float(stats.norm.ppf(0.5 + confidence / 2.0))
    p = kk / nn
    denom = 1.0 + z * z / nn
    centre = (p + z * z / (2.0 * nn)) / denom
    half = z * np.sqrt(p * (1.0 - p) / nn + z * z / (4.0 * nn * nn)) / denom
    # Analytically the limits are exactly 0 (k = 0) and 1 (k = n); remove rounding error there.
    low = np.where(kk == 0, 0.0, np.clip(centre - half, 0.0, 1.0))
    high = np.where(kk == nn, 1.0, np.clip(centre + half, 0.0, 1.0))
    if low.ndim == 0:
        return float(low), float(high)
    return low, high


def holm_correction(pvals: ArrayLike) -> FloatArray:
    """Holm-Bonferroni step-down adjusted p values, in the input order.

    Sorted ascending ``p_(1) <= ... <= p_(m)``: ``adj_(i) = max_{j <= i} min(1, (m - j + 1)
    p_(j))``. NaN entries are kept as NaN and do not count towards ``m``.
    """
    p = np.asarray(pvals, dtype=float)
    if p.ndim != 1:
        raise ValueError("pvals must be one-dimensional")
    valid = ~np.isnan(p)
    if np.any(p[valid] < 0) or np.any(p[valid] > 1):
        raise ValueError("p values must lie in [0, 1]")
    out = np.full(p.shape, np.nan)
    idx = np.flatnonzero(valid)
    m = idx.size
    if m == 0:
        return out
    order = idx[np.argsort(p[idx], kind="stable")]
    scaled = (m - np.arange(m)) * p[order]
    out[order] = np.minimum(1.0, np.maximum.accumulate(scaled))
    return out


# ---------------------------------------------------------------------------------------------
# rank-based tests
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class MannWhitneyResult:
    """Mann-Whitney U test of two independent samples.

    Attributes:
        n1, n2: sample sizes.
        u: U statistic of the first sample (number of (a, b) pairs with a > b, ties count 1/2).
        p: p value from ``scipy.stats.mannwhitneyu`` (exact when feasible, else asymptotic).
        rank_biserial: ``2 U / (n1 n2) - 1`` in [-1, 1]; positive when ``a`` tends to be larger.
        prob_superiority: ``U / (n1 n2)``, probability that a random ``a`` exceeds a random ``b``.
    """

    n1: int
    n2: int
    u: float
    p: float
    rank_biserial: float
    prob_superiority: float


def mann_whitney(a: ArrayLike, b: ArrayLike, alternative: str = "two-sided") -> MannWhitneyResult:
    """Mann-Whitney U test with rank-biserial correlation as the effect size."""
    x, y = _sample(a, "a", min_n=1), _sample(b, "b", min_n=1)
    res = stats.mannwhitneyu(x, y, alternative=alternative, method="auto")
    n1, n2 = x.size, y.size
    u = float(res.statistic)
    return MannWhitneyResult(n1, n2, u, float(res.pvalue), 2.0 * u / (n1 * n2) - 1.0,
                             u / (n1 * n2))


@dataclass(frozen=True)
class WilcoxonResult:
    """Wilcoxon signed-rank test of paired samples (the paired analogue of Mann-Whitney).

    Attributes:
        n_pairs: number of pairs supplied.
        n_nonzero: pairs with a non-zero difference (zero differences are dropped).
        w_plus, w_minus: rank sums of the positive and negative differences.
        p: two-sided p value from ``scipy.stats.wilcoxon``.
        rank_biserial: matched-pairs rank-biserial ``(W+ - W-) / (W+ + W-)``; positive when
            ``a > b`` tends to hold.
    """

    n_pairs: int
    n_nonzero: int
    w_plus: float
    w_minus: float
    p: float
    rank_biserial: float


def wilcoxon_signed_rank(a: ArrayLike, b: ArrayLike) -> WilcoxonResult:
    """Wilcoxon signed-rank test on ``a - b`` with the matched-pairs rank-biserial effect."""
    x, y = _sample(a, "a"), _sample(b, "b")
    if x.size != y.size:
        raise ValueError("paired samples must have equal length")
    diffs = x - y
    nz = diffs[diffs != 0.0]
    if nz.size == 0:
        return WilcoxonResult(x.size, 0, 0.0, 0.0, 1.0, 0.0)
    ranks = stats.rankdata(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    p = float(stats.wilcoxon(nz, zero_method="wilcox", method="auto").pvalue)
    rbc = (w_plus - w_minus) / (w_plus + w_minus)
    return WilcoxonResult(x.size, nz.size, w_plus, w_minus, p, rbc)


# ---------------------------------------------------------------------------------------------
# power
# ---------------------------------------------------------------------------------------------
def _power_t(d: float, n: int, alpha: float, paired: bool) -> float:
    """Exact power of the two-sided t test via the non-central t distribution."""
    if paired:
        df, ncp = n - 1.0, d * math.sqrt(n)
    else:
        df, ncp = 2.0 * n - 2.0, d * math.sqrt(n / 2.0)
    tcrit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    return float(stats.nct.sf(tcrit, df, ncp) + stats.nct.cdf(-tcrit, df, ncp))


def n_per_group_for_d(
    d: float, alpha: float = 0.05, power: float = 0.80, paired: bool = False
) -> int:
    """Smallest per-group n giving at least ``power`` for a two-sided t test of effect size d.

    Exact (non-central t), equal group sizes. With ``paired=True`` returns the number of pairs
    for a paired t test where ``d`` is d_z. Known values (alpha 0.05, power 0.80, independent):
    d = 0.2 -> 394, d = 0.5 -> 64, d = 0.8 -> 26.
    """
    if d == 0.0:
        raise ValueError("d must be non-zero")
    if not 0.0 < alpha < 1.0 or not 0.0 < power < 1.0:
        raise ValueError("alpha and power must be in (0, 1)")
    d = abs(d)
    lo, hi = 2, 2
    while _power_t(d, hi, alpha, paired) < power:
        lo, hi = hi, hi * 2
        if hi > 10_000_000:
            raise ValueError("required sample size is unreasonably large")
    while lo < hi:  # smallest n with power >= target (power is increasing in n)
        mid = (lo + hi) // 2
        if _power_t(d, mid, alpha, paired) >= power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


# ---------------------------------------------------------------------------------------------
# results-table record
# ---------------------------------------------------------------------------------------------
def compare_groups(
    a: ArrayLike, b: ArrayLike, paired: bool = False, confidence: float = 0.95
) -> dict[str, float | int | str]:
    """One flat record comparing samples ``a`` and ``b`` for a results table.

    Runs Welch (or paired) t and its non-parametric companion: Mann-Whitney for independent
    samples, Wilcoxon signed-rank for paired samples (Mann-Whitney assumes independence, which
    paired designs violate). The non-parametric columns share the same names in both cases.
    """
    x, y = _sample(a, "a"), _sample(b, "b")
    if paired:
        tt = paired_t(x, y, confidence)
        wx = wilcoxon_signed_rank(x, y)
        nonparam = {
            "nonparam_test": "wilcoxon_signed_rank",
            "nonparam_stat": min(wx.w_plus, wx.w_minus),
            "p_nonparam": wx.p,
            "rank_biserial": wx.rank_biserial,
        }
    else:
        tt = welch_t(x, y, confidence)
        mw = mann_whitney(x, y)
        nonparam = {
            "nonparam_test": "mann_whitney_u",
            "nonparam_stat": mw.u,
            "p_nonparam": mw.p,
            "rank_biserial": mw.rank_biserial,
        }
    return {
        "test": tt.test,
        "n_a": int(x.size),
        "n_b": int(y.size),
        "mean_a": float(np.mean(x)),
        "mean_b": float(np.mean(y)),
        "sd_a": float(np.std(x, ddof=1)),
        "sd_b": float(np.std(y, ddof=1)),
        "median_a": float(np.median(x)),
        "median_b": float(np.median(y)),
        "mean_diff": tt.mean_diff,
        "diff_ci_low": tt.diff_ci_low,
        "diff_ci_high": tt.diff_ci_high,
        "df": tt.df,
        "t": tt.t,
        "p_t": tt.p,
        "cohen_d": tt.d,
        "d_ci_low": tt.d_ci_low,
        "d_ci_high": tt.d_ci_high,
        **nonparam,
    }
