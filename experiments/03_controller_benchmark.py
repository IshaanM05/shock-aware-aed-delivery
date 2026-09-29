"""Experiment 03: controller benchmark (RQ2).

Runs every controller on every scenario family for N seeded episodes (same seeds across
controllers, so the comparison is paired) and writes results/benchmark_<tag>.csv.

    python experiments/03_controller_benchmark.py --n 10 --tag smoke
    python experiments/03_controller_benchmark.py --n 100 --tag standard \
        --controllers pure_pursuit apf_nocurb apf mppi ppo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from aedrover.analysis.experiments import make_jobs, run_job
from aedrover.parallel import default_workers, pmap
from aedrover.sim.scenario import FAMILIES

ROOT = Path(__file__).resolve().parents[1]


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["controller", "family"])
    out = g.agg(
        n=("seed", "size"),
        success=("success", "mean"),
        time_s=("time_s", "median"),
        shock_g=("peak_shock_g", "median"),
        over_budget=("shock_over_budget", "mean"),
        min_clear=("min_clearance_m", "median"),
        interv=("intervention_frac", "mean"),
    )
    return out.round(3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10, help="episodes per controller x family")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--controllers", nargs="+", default=["pure_pursuit", "apf_nocurb", "apf"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    jobs = make_jobs(args.controllers, args.families, range(args.seed0, args.seed0 + args.n),
                     shield=not args.no_shield, tag=args.tag)
    print(f"{len(jobs)} episodes on {args.workers or default_workers()} workers")
    t0 = time.perf_counter()
    rows = pmap(run_job, jobs, workers=args.workers, chunksize=4, desc="bench")
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    sim_s = df.time_s.sum()
    print(f"done in {wall:.0f}s wall; {sim_s:.0f}s simulated ({sim_s / wall:.0f}x real time)")
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"benchmark_{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summarise(df).to_string())
    print("\noutcomes:\n", df.groupby("controller").outcome.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
