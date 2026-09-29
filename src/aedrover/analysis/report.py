"""Benchmark summaries and paired statistical comparisons (used by tables, figures and the paper).

Design: every controller is run on the *same* seeds, so comparisons are paired by
``(family, seed)``. Success is a binary outcome (exact McNemar test on discordant pairs, Wilson
intervals per controller); time and payload shock are compared with a paired t test and Wilcoxon
signed-rank on episodes where both controllers reached the goal. p values are Holm-adjusted per
metric across all (family, controller) tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import compare_groups, holm_correction, wilson_ci

BUDGET_G = 3.0


def load_benchmark(*paths: str | Path) -> pd.DataFrame:
    frames = [pd.read_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True)
    df["success"] = df["success"].astype(bool)
    return df


def summarize(df: pd.DataFrame, keys=("controller", "family"), budget_g: float = BUDGET_G) -> pd.DataFrame:
    rows = []
    for k, g in df.groupby(list(keys), sort=False):
        k = k if isinstance(k, tuple) else (k,)
        n, s = len(g), int(g.success.sum())
        lo, hi = wilson_ci(s, n)
        ok = g[g.success]
        over = int((ok.peak_shock_g > budget_g).sum())
        olo, ohi = wilson_ci(over, max(len(ok), 1)) if len(ok) else (np.nan, np.nan)
        rows.append({
            **dict(zip(keys, k, strict=True)), "n": n, "success": s / n, "success_lo": lo, "success_hi": hi,
            "collision": float((g.outcome == "collision").mean()), "stall": float((g.outcome == "stall").mean()),
            "time_med": float(ok.time_s.median()) if len(ok) else np.nan,
            "time_q25": float(ok.time_s.quantile(0.25)) if len(ok) else np.nan,
            "time_q75": float(ok.time_s.quantile(0.75)) if len(ok) else np.nan,
            "shock_med": float(ok.peak_shock_g.median()) if len(ok) else np.nan,
            "shock_p90": float(ok.peak_shock_g.quantile(0.9)) if len(ok) else np.nan,
            "over_budget": over / max(len(ok), 1) if len(ok) else np.nan, "over_lo": olo, "over_hi": ohi,
            "min_clear_med": float(g.min_clearance_m.median()) if g.min_clearance_m.notna().any() else np.nan,
            "energy_wh_med": float(ok.energy_wh.median()) if len(ok) else np.nan,
            "interv_frac": float(g.intervention_frac.mean()) if "intervention_frac" in g else np.nan,
        })
    return pd.DataFrame(rows)


def delivery_success(df: pd.DataFrame, budget_g: float = BUDGET_G) -> pd.Series:
    """An episode is a *safe delivery* if the rover reached the goal without exceeding the shock budget."""
    return df.success & (df.peak_shock_g <= budget_g)


def _mcnemar(a: np.ndarray, b: np.ndarray) -> tuple[float, int, int]:
    """Exact McNemar test (two-sided) for paired binary outcomes; returns (p, b_only, a_only)."""
    a_only, b_only = int(np.sum(a & ~b)), int(np.sum(~a & b))
    n = a_only + b_only
    if n == 0:
        return 1.0, b_only, a_only
    return float(sps.binomtest(a_only, n, 0.5).pvalue), b_only, a_only


def paired_vs_reference(df: pd.DataFrame, reference: str, *, families=None, min_pairs: int = 8,
                        budget_g: float = BUDGET_G) -> pd.DataFrame:
    """Compare every controller with ``reference`` on shared seeds, per family and pooled ('all').

    Returns one row per (family, controller, metric) with the paired test results; ``p_holm`` is
    adjusted within each metric across all rows.
    """
    df = df.copy()
    df["safe"] = delivery_success(df, budget_g)
    fams = list(families) if families is not None else list(df.family.unique()) + ["all"]
    rows = []
    for fam in fams:
        d = df if fam == "all" else df[df.family == fam]
        ref = d[d.controller == reference].set_index(["family", "seed"])
        for ctrl in d.controller.unique():
            if ctrl == reference:
                continue
            cur = d[d.controller == ctrl].set_index(["family", "seed"])
            idx = ref.index.intersection(cur.index)
            if len(idx) < min_pairs:
                continue
            r, c = ref.loc[idx], cur.loc[idx]
            for metric in ("success", "safe"):
                p, b_only, a_only = _mcnemar(c[metric].to_numpy(bool), r[metric].to_numpy(bool))
                rows.append({"family": fam, "controller": ctrl, "reference": reference, "metric": metric,
                             "n_pairs": len(idx), "mean_ctrl": float(c[metric].mean()),
                             "mean_ref": float(r[metric].mean()), "mean_diff": float(c[metric].mean() - r[metric].mean()),
                             "ctrl_only": a_only, "ref_only": b_only, "p_raw": p})
            both = c.success.to_numpy(bool) & r.success.to_numpy(bool)
            if both.sum() >= min_pairs:
                for metric in ("time_s", "peak_shock_g"):
                    res = compare_groups(c[metric].to_numpy(float)[both], r[metric].to_numpy(float)[both], paired=True)
                    rows.append({"family": fam, "controller": ctrl, "reference": reference, "metric": metric,
                                 "n_pairs": int(both.sum()), "mean_ctrl": res["mean_a"], "mean_ref": res["mean_b"],
                                 "mean_diff": res["mean_diff"], "diff_ci_low": res["diff_ci_low"],
                                 "diff_ci_high": res["diff_ci_high"], "df": res["df"], "t": res["t"],
                                 "cohen_d": res["cohen_d"], "d_ci_low": res["d_ci_low"], "d_ci_high": res["d_ci_high"],
                                 "p_raw": res["p_t"], "p_nonparam": res["p_nonparam"]})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for _metric, idx in out.groupby("metric").groups.items():
        out.loc[idx, "p_holm"] = holm_correction(out.loc[idx, "p_raw"].to_numpy())
    return out


def to_markdown(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Small dependency-free Markdown table renderer."""
    cols = list(df.columns)

    def fmt(x) -> str:
        if isinstance(x, (float, np.floating)):
            return "" if np.isnan(x) else float_fmt.format(x)
        return str(x)

    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)
