"""Experiment 03: controller benchmark (RQ2).

Runs every controller on every scenario family for N seeded episodes (same seeds across
controllers, so the comparison is paired) and writes results/benchmark_<tag>.csv.

    python experiments/03_controller_benchmark.py --n 10 --tag smoke
    python experiments/03_controller_benchmark.py --n 100 --tag standard \
        --controllers pure_pursuit apf_nocurb apf mppi ppo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from aedrover.analysis.experiments import controller_spec, make_jobs, run_job
from aedrover.parallel import default_workers, pmap
from aedrover.sim.scenario import FAMILIES
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]


def tuned_mppi() -> dict:
    """Frozen MPPI settings selected on tuning seeds (configs/mppi_tuned.json), if the tuning was run."""
    path = ROOT / "configs" / "mppi_tuned.json"
    return json.loads(path.read_text(encoding="utf-8"))["kwargs"] if path.exists() else {}


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
    ap.add_argument("--n-mppi", type=int, default=None, help="MPPI episodes per family (first seeds only; default: n)")
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--controllers", nargs="+", default=["pure_pursuit", "apf_nocurb", "apf"])
    ap.add_argument("--families", nargs="+", default=list(FAMILIES))
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--vehicle", choices=("nominal", "optimized"), default="optimized")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--speed-cap", type=float, default=2.0, help="top speed [m/s] given to EVERY controller")
    ap.add_argument("--ppo-path", default="checkpoints/ppo_shielded")
    ap.add_argument("--mppi-samples", type=int, default=128)
    ap.add_argument("--mppi-horizon", type=int, default=20)
    args = ap.parse_args()

    specs = []
    for name in args.controllers:
        extra = {"path": args.ppo_path} if name == "ppo" else {}
        if name == "mppi":
            extra = {"K": args.mppi_samples, "H": args.mppi_horizon, **tuned_mppi()}
        specs.append(controller_spec(name, args.speed_cap, **extra))
    jobs = make_jobs(specs, args.families, range(args.seed0, args.seed0 + args.n),
                     shield=not args.no_shield, tag=args.tag,
                     env_kwargs=(("veh", VehicleParams.by_name(args.vehicle)),))
    if args.n_mppi is not None:      # MPPI is ~100x costlier: fewer seeds, still paired on the shared ones
        jobs = [j for j in jobs if j.controller != "mppi" or j.seed < args.seed0 + args.n_mppi]
    jobs.sort(key=lambda j: j.controller != "mppi")            # slow MPPI episodes first: better load balance
    print(f"{len(jobs)} episodes on {args.workers or default_workers()} workers")
    t0 = time.perf_counter()
    rows = pmap(run_job, jobs, workers=args.workers, chunksize=4, desc="bench")
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df["vehicle"] = args.vehicle
    df["speed_cap"] = args.speed_cap
    sim_s = df.time_s.sum()
    print(f"done in {wall:.0f}s wall; {sim_s:.0f}s simulated ({sim_s / wall:.0f}x real time)")
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"benchmark_{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    print(summarise(df).to_string())
    print("\noutcomes:\n", df.groupby("controller").outcome.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
