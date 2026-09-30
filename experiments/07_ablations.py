"""Experiment 07: ablations that explain the main results.

  vehicle    : co-designed vs draft rover, same controllers and seeds
  shield     : safety filter on vs off
  speed_cap  : top speed 0.8 / 1.4 / 2.0 / 2.6 m/s for every controller (the course brief cites 0.8 m/s
               in shared pedestrian zones, so the cost of obeying it is quantified, not assumed)

Writes results/ablation_<name>_<tag>.csv and prints delivery success (goal reached AND shock within the
budget) and median time per condition.

    python experiments/07_ablations.py --n 40 --tag standard --controllers dwa mppi ppo
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from aedrover.analysis.experiments import Job, controller_spec, run_job
from aedrover.analysis.report import delivery_success
from aedrover.analysis.stats import wilson_ci
from aedrover.parallel import default_workers, pmap
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("kerb", "mixed")


def tuned_mppi() -> dict:
    path = ROOT / "configs" / "mppi_tuned.json"
    return json.loads(path.read_text(encoding="utf-8"))["kwargs"] if path.exists() else {}


def spec(name: str, cap: float, ppo_path: str) -> tuple[str, dict]:
    extra = {"path": ppo_path} if name == "ppo" else (tuned_mppi() if name == "mppi" else {})
    return controller_spec(name, cap, **extra)


def build_jobs(ablation: str, controllers, n: int, seed0: int, ppo_path: str) -> list[Job]:
    jobs: list[Job] = []
    seeds = range(seed0, seed0 + n)
    if ablation == "vehicle":
        for design in ("nominal", "optimized"):
            env_kwargs = (("veh", VehicleParams.by_name(design)),)
            for c in controllers:
                if c == "ppo":
                    continue      # the policy was trained on the co-designed vehicle only
                name, kw = spec(c, 2.0, ppo_path)
                jobs += [Job(name, f, s, controller_kwargs=tuple(sorted(kw.items())), env_kwargs=env_kwargs, tag=design)
                         for f in FAMILIES for s in seeds]
    elif ablation == "shield":
        env_kwargs = (("veh", VehicleParams.optimized()),)
        for shield in (True, False):
            for c in controllers:
                name, kw = spec(c, 2.0, ppo_path)
                jobs += [Job(name, f, s, shield=shield, controller_kwargs=tuple(sorted(kw.items())),
                             env_kwargs=env_kwargs, tag="filter_on" if shield else "filter_off")
                         for f in ("mixed", "crowded") for s in seeds]
    elif ablation == "speed_cap":
        env_kwargs = (("veh", VehicleParams.optimized()),)
        for cap in (0.8, 1.4, 2.0, 2.6):
            for c in controllers:
                name, kw = spec(c, cap, ppo_path)
                jobs += [Job(name, "mixed", s, controller_kwargs=tuple(sorted(kw.items())), env_kwargs=env_kwargs,
                             tag=f"cap_{cap}") for s in seeds]
    else:
        raise ValueError(ablation)
    return jobs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablations", nargs="+", default=["vehicle", "shield", "speed_cap"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed0", type=int, default=40000)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--controllers", nargs="+", default=["dwa", "ppo"])
    ap.add_argument("--ppo-path", default="checkpoints/ppo_shielded")
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    (ROOT / "results").mkdir(exist_ok=True)
    for ab in args.ablations:
        jobs = build_jobs(ab, args.controllers, args.n, args.seed0, args.ppo_path)
        jobs.sort(key=lambda j: j.controller != "mppi")
        print(f"[{ab}] {len(jobs)} episodes on {args.workers or default_workers()} workers")
        t0 = time.perf_counter()
        df = pd.DataFrame(pmap(run_job, jobs, workers=args.workers, chunksize=2, desc=ab))
        df = df.rename(columns={"tag": "condition"})
        df["safe"] = delivery_success(df)
        df.to_csv(ROOT / "results" / f"ablation_{ab}_{args.tag}.csv", index=False)
        rows = []
        for (ctrl, cond, fam), g in df.groupby(["controller", "condition", "family"]):
            lo, hi = wilson_ci(int(g.safe.sum()), len(g))
            rows.append({"controller": ctrl, "condition": cond, "family": fam, "n": len(g), "safe": g.safe.mean(),
                         "safe_lo": lo, "safe_hi": hi, "success": g.success.mean(),
                         "collision": (g.outcome == "collision").mean(), "time_med": g[g.success].time_s.median(),
                         "shock_med": g[g.success].peak_shock_g.median()})
        print(f"[{ab}] done in {time.perf_counter() - t0:.0f}s")
        pd.set_option("display.width", 200)
        print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
