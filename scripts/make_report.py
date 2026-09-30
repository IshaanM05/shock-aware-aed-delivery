"""Turn results/*.csv|json into publication figures and docs/RESULTS.md (never hand-edited).

Every section is generated only if its input exists, so the script can be re-run at any stage.

    python scripts/make_report.py [--tag standard]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aedrover.analysis import figures as F
from aedrover.analysis import figures_results as FR
from aedrover.analysis.report import delivery_success, paired_vs_reference, summarize, to_markdown
from aedrover.analysis.stats import wilson_ci

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "docs" / "figures"


def rel(p: Path) -> str:
    return "figures/" + p.name


def section_design(lines: list[str]) -> None:
    if (RES / "codesign.json").exists():
        F.fig_codesign(FIG / "codesign_shock.png")
        lines += ["## 1. Mechanical co-design (RQ1)", "", f"![co-design]({rel(FIG / 'codesign_shock.png')})", ""]
    if (ROOT / "configs" / "curb_table.json").exists():
        F.fig_climb_window(FIG / "climb_window.png")
        lines += [f"![climb window]({rel(FIG / 'climb_window.png')})", ""]


def section_benchmark(lines: list[str], tag: str) -> pd.DataFrame | None:
    path = RES / f"benchmark_{tag}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["success"] = df["success"].astype(bool)
    n = int(df.groupby(["controller", "family"]).size().min())
    FR.fig_benchmark(df, FIG / "benchmark.png", title=f"Controller benchmark ({n}+ paired episodes per cell, co-designed vehicle, common speed cap)")
    sm = summarize(df)
    sm.to_csv(RES / "tables_benchmark_summary.csv", index=False)
    df["safe"] = delivery_success(df)
    safe = (df.groupby(["controller", "family"]).safe.agg(["sum", "size"]).reset_index())
    safe["safe_delivery"] = safe["sum"] / safe["size"]
    safe[["safe_lo", "safe_hi"]] = [wilson_ci(int(a), int(b)) for a, b in zip(safe["sum"], safe["size"], strict=True)]
    pivot = safe.pivot(index="controller", columns="family", values="safe_delivery")
    order = [c for c in FR.CONTROLLER_ORDER if c in pivot.index]
    cols = [f for f in F.FAMILY_LABELS if f in pivot.columns]
    tab = pivot.loc[order, cols].rename(index=F.CONTROLLER_LABELS, columns=F.FAMILY_LABELS).reset_index()
    tab = tab.rename(columns={"controller": "Controller"})
    lines += ["## 2. Controller benchmark (RQ2)", "",
              f"Every controller ran the same {n} seeds per scenario family (paired), on the co-designed vehicle, with the "
              "same top speed and the same safety filter. **Safe delivery** = goal reached AND peak payload shock within 3 g.", "",
              f"![benchmark]({rel(FIG / 'benchmark.png')})", "", "### Safe-delivery rate", "", to_markdown(tab, "{:.2f}"), ""]
    ref = "dwa" if "dwa" in set(df.controller) else df.controller.iloc[0]
    pv = paired_vs_reference(df, ref)
    if len(pv):
        pv.to_csv(RES / "tables_benchmark_paired.csv", index=False)
        keep = pv[pv.family == "all"]
        cols = [c for c in ("controller", "metric", "n_pairs", "mean_ctrl", "mean_ref", "mean_diff", "df", "t", "cohen_d",
                            "p_raw", "p_holm") if c in keep.columns]
        lines += [f"### Paired comparison with `{ref}` (all families pooled; Holm-adjusted p)", "",
                  to_markdown(keep[cols], "{:.3g}"), ""]
    return df


def section_ood(lines: list[str], tag: str) -> None:
    path = RES / f"ood_{tag}.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    df["success"] = df["success"].astype(bool)
    df["safe"] = df["safe"].astype(bool)
    rows = []
    for (c, cond), g in df.groupby(["controller", "condition"]):
        lo, hi = wilson_ci(int(g.safe.sum()), len(g))
        rows.append({"controller": c, "condition": cond, "n": len(g), "safe_delivery": g.safe.mean(), "safe_lo": lo, "safe_hi": hi})
    tab = pd.DataFrame(rows)
    FR.fig_ood(tab, FIG / "ood.png")
    tab.to_csv(RES / "tables_ood.csv", index=False)
    lines += ["## 3. Out-of-distribution robustness (RQ2)", "", f"![ood]({rel(FIG / 'ood.png')})", "",
              to_markdown(tab.round(3)), ""]


def section_ablations(lines: list[str], tag: str) -> None:
    got = False
    for name in ("vehicle", "shield", "speed_cap"):
        path = RES / f"ablation_{name}_{tag}.csv"
        if not path.exists():
            continue
        if not got:
            lines += ["## 4. Ablations", ""]
            got = True
        df = pd.read_csv(path)
        df["success"] = df["success"].astype(bool)
        df["safe"] = df["safe"].astype(bool)
        if name == "speed_cap":
            FR.fig_speed_cap(df, FIG / "speed_cap.png")
            lines += [f"![speed cap]({rel(FIG / 'speed_cap.png')})", ""]
        g = df.groupby(["controller", "condition", "family"]).agg(n=("seed", "size"), safe=("safe", "mean"), success=("success", "mean"),
                                                                  collision=("outcome", lambda x: (x == "collision").mean()),
                                                                  time_med=("time_s", "median"), shock_med=("peak_shock_g", "median")).reset_index()
        lines += [f"### {name.replace('_', ' ')}", "", to_markdown(g.round(3)), ""]


def section_clinical(lines: list[str]) -> None:
    sp, pp = RES / "clinical_survival_vs_radius.csv", RES / "clinical_dispatch_policies.csv"
    if not sp.exists():
        return
    surv = pd.read_csv(sp)
    FR.fig_survival(surv, FIG / "survival_vs_radius.png", densities=[d for d in ("osm_lower_bound", "assumed_4_per_km", "assumed_8_per_km") if d in set(surv.density)])
    lines += ["## 5. Clinical impact (RQ3)", "", f"![survival]({rel(FIG / 'survival_vs_radius.png')})", ""]
    if pp.exists():
        pol = pd.read_csv(pp)
        best = pol[pol["policy"] == "rover"].groupby("controller").mean_survival.mean().idxmax()
        FR.fig_policies(pol, FIG / "dispatch_policies.png", best)
        lines += [f"![policies]({rel(FIG / 'dispatch_policies.png')})", ""]
    cj = RES / "clinical.json"
    if cj.exists():
        d = json.loads(cj.read_text(encoding="utf-8"))
        be = pd.DataFrame([{"case": k, **v} for k, v in d.get("breakeven", {}).items()])
        if len(be):
            lines += ["### Break-even radius (rover alone versus ambulance alone)", "", to_markdown(be.round(3)), ""]
        if d.get("economics"):
            lines += ["### Dimensionless economics (course model; every input is an assumption)", "",
                      "```json", json.dumps(d["economics"], indent=2), "```", ""]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="standard")
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    lines = ["# Results", "", "Generated by `scripts/make_report.py` from `results/`. Do not edit by hand.", ""]
    section_design(lines)
    section_benchmark(lines, args.tag)
    section_ood(lines, args.tag)
    section_ablations(lines, args.tag)
    section_clinical(lines)
    FR.fig_architecture(FIG / "architecture.png")
    (ROOT / "docs" / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote docs/RESULTS.md and figures in docs/figures")
    _ = np


if __name__ == "__main__":
    main()
