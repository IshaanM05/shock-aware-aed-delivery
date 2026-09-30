"""Experiment 04: out-of-distribution generalisation (RQ2).

The learned policy is trained with kerb heights 6-16 cm and tyre friction 0.5-1.2. Here every
controller is evaluated on the same seeds inside that range and on conditions outside it:

  id        : kerb 6-16 cm, friction 0.5-1.2 (the training distribution)
  ood_kerb  : kerb 15-19 cm (taller than anything seen in training)
  ood_mu    : friction 0.3-0.5 (slipperier than anything seen in training)
  ood_both  : both at once

Writes results/ood_<tag>.csv and prints delivery success (reached the goal AND payload shock under
the budget) with Wilson intervals.

    python experiments/04_ood_generalization.py --n 50 --tag standard --controllers dwa mppi ppo
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from aedrover.analysis.experiments import Job, controller_spec, run_job
from aedrover.analysis.report import delivery_success
from aedrover.analysis.stats import wilson_ci
from aedrover.parallel import default_workers, pmap
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]

CONDITIONS = {
    "id": {"kerb_range": (0.06, 0.16), "mu_range": (0.5, 1.2)},
    "ood_kerb": {"kerb_range": (0.15, 0.19), "mu_range": (0.5, 1.2)},
    "ood_mu": {"kerb_range": (0.06, 0.16), "mu_range": (0.3, 0.5)},
    "ood_both": {"kerb_range": (0.15, 0.19), "mu_range": (0.3, 0.5)},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed0", type=int, default=20000)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--controllers", nargs="+", default=["dwa", "ppo"])
    ap.add_argument("--families", nargs="+", default=["kerb", "mixed"])
    ap.add_argument("--speed-cap", type=float, default=2.0)
    ap.add_argument("--ppo-path", default="checkpoints/ppo_shielded")
    ap.add_argument("--vehicle", choices=("nominal", "optimized"), default="optimized")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    env_kwargs = (("veh", VehicleParams.by_name(args.vehicle)),)
    specs = [controller_spec(c, args.speed_cap, **({"path": args.ppo_path} if c == "ppo" else {}))
             for c in args.controllers]
    jobs = []
    for cond, kw in CONDITIONS.items():
        sk = tuple(sorted(kw.items()))
        for name, ck in specs:
            for fam in args.families:
                for seed in range(args.seed0, args.seed0 + args.n):
                    jobs.append(Job(name, fam, seed, controller_kwargs=tuple(sorted(ck.items())),
                                    env_kwargs=env_kwargs, scenario_kwargs=sk, tag=cond))
    jobs.sort(key=lambda j: j.controller != "mppi")
    print(f"{len(jobs)} episodes on {args.workers or default_workers()} workers")
    t0 = time.perf_counter()
    rows = pmap(run_job, jobs, workers=args.workers, chunksize=4, desc="ood")
    df = pd.DataFrame(rows).rename(columns={"tag": "condition"})
    df["safe"] = delivery_success(df)
    print(f"done in {time.perf_counter() - t0:.0f}s")
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / f"ood_{args.tag}.csv", index=False)

    rows = []
    for (ctrl, cond), g in df.groupby(["controller", "condition"]):
        n = len(g)
        lo, hi = wilson_ci(int(g.safe.sum()), n)
        rows.append({"controller": ctrl, "condition": cond, "n": n, "success": g.success.mean(),
                     "safe_delivery": g.safe.mean(), "safe_lo": lo, "safe_hi": hi,
                     "shock_med": g[g.success].peak_shock_g.median()})
    tab = pd.DataFrame(rows)
    order = list(CONDITIONS)
    pd.set_option("display.width", 200)
    print(tab.pivot(index="controller", columns="condition", values="safe_delivery")[order].round(2).to_string())
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
