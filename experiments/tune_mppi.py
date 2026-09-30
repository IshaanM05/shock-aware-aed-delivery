"""MPPI hyper-parameter selection on tuning seeds that are disjoint from every evaluation seed.

Compares a few variants of the cost weights and sample budget on the kerb-heavy families and
prints safe-delivery rate (goal reached AND payload shock under budget), collision rate and median
time. The winning variant is then frozen for the main benchmark; nothing is tuned on evaluation seeds.

    python experiments/tune_mppi.py --n 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from aedrover.analysis.experiments import Job, run_job
from aedrover.analysis.report import delivery_success
from aedrover.parallel import default_workers, pmap
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]

VARIANTS = {
    "base": {},
    "shock_x2.5": {"w_shock": 10.0},
    "shock_x2.5_clear": {"w_shock": 10.0, "w_clear": 10.0, "comfort_clearance": 0.6},
    "shock_x2.5_K192": {"w_shock": 10.0, "K": 192},
    "smooth": {"w_shock": 10.0, "w_u": 0.3, "sigma_v": 0.4},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=30000)
    ap.add_argument("--families", nargs="+", default=["kerb", "mixed", "crowded"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--speed-cap", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    env_kwargs = (("veh", VehicleParams.optimized()),)
    jobs = []
    for name in args.variants:
        kw = {"v_max": args.speed_cap, **VARIANTS[name]}
        for fam in args.families:
            for seed in range(args.seed0, args.seed0 + args.n):
                jobs.append(Job("mppi", fam, seed, controller_kwargs=tuple(sorted(kw.items())),
                                env_kwargs=env_kwargs, tag=name))
    print(f"{len(jobs)} episodes on {args.workers or default_workers()} workers")
    t0 = time.perf_counter()
    df = pd.DataFrame(pmap(run_job, jobs, workers=args.workers, chunksize=1, desc="tune"))
    print(f"done in {time.perf_counter() - t0:.0f}s")
    df["safe"] = delivery_success(df)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "mppi_tuning.csv", index=False)
    g = df.groupby(["tag", "family"])
    tab = g.agg(n=("seed", "size"), success=("success", "mean"), safe=("safe", "mean"),
                collision=("outcome", lambda x: (x == "collision").mean()),
                time_med=("time_s", "median"), shock_med=("peak_shock_g", "median"))
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string())
    summ = df.groupby("tag").agg(safe=("safe", "mean"), success=("success", "mean"),
                                 collision=("outcome", lambda x: (x == "collision").mean()),
                                 time_med=("time_s", "median"))
    summ = summ.sort_values(["safe", "collision", "time_med"], ascending=[False, True, True])
    print(summ.round(3).to_string())
    best = summ.index[0]                       # highest safe-delivery rate, then fewest collisions, then fastest
    cfg = {"variant": best, "kwargs": VARIANTS[best], "tuning_seeds": [args.seed0, args.seed0 + args.n - 1],
           "families": args.families, "score": summ.loc[best].round(4).to_dict()}
    (ROOT / "configs" / "mppi_tuned.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("selected", best, "-> configs/mppi_tuned.json")


if __name__ == "__main__":
    main()
