"""Tests for benchmark summaries and paired comparisons."""

import numpy as np
import pandas as pd

from aedrover.analysis.report import delivery_success, paired_vs_reference, summarize, to_markdown


def _frame(seed=0, n=60):
    rng = np.random.default_rng(seed)
    rows = []
    for fam in ("kerb", "mixed"):
        for s in range(n):
            base_ok = rng.random() < 0.7
            for ctrl, p_ok, shock in (("ref", 0.7, 3.5), ("good", 0.95, 2.0)):
                ok = (rng.random() < p_ok) if ctrl == "good" else base_ok
                rows.append({"controller": ctrl, "family": fam, "seed": s, "success": ok,
                             "outcome": "goal" if ok else "stall", "time_s": rng.normal(25, 3) - (4 if ctrl == "good" else 0),
                             "peak_shock_g": rng.normal(shock, 0.3), "min_clearance_m": 0.5, "energy_wh": 5.0,
                             "intervention_frac": 0.1})
    return pd.DataFrame(rows)


def test_summary_counts_and_wilson_ordering():
    df = _frame()
    sm = summarize(df)
    assert set(sm.controller) == {"ref", "good"}
    assert (sm.success_lo <= sm.success).all() and (sm.success <= sm.success_hi).all()
    assert sm[sm.controller == "good"].success.mean() > sm[sm.controller == "ref"].success.mean()
    assert (sm.n == 60).all()


def test_paired_comparison_detects_a_real_difference():
    df = _frame()
    res = paired_vs_reference(df, "ref")
    assert not res.empty
    t = res[(res.metric == "time_s") & (res.family == "all")].iloc[0]
    assert t.mean_diff < -2.0 and t.p_holm < 0.01 and t.cohen_d < -0.5
    s = res[(res.metric == "success") & (res.family == "all")].iloc[0]
    assert s.mean_diff > 0.1 and s.p_holm < 0.05


def test_holm_adjusted_p_is_never_smaller_than_raw():
    res = paired_vs_reference(_frame(3), "ref")
    assert (res.p_holm >= res.p_raw - 1e-12).all()


def test_delivery_success_requires_budget_compliance():
    df = pd.DataFrame({"success": [True, True, False], "peak_shock_g": [2.0, 4.0, 1.0]})
    assert delivery_success(df, 3.0).tolist() == [True, False, False]


def test_markdown_renders():
    md = to_markdown(summarize(_frame()).head(2))
    assert md.startswith("| controller") and md.count("\n") == 3
