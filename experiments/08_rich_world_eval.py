"""Experiment 08: a small evaluation in the rich world (collidable street furniture).

NOT comparable with ``results/benchmark_standard.csv``: the scenarios here are different (different seeds, a street full of
parked vehicles, stalls, lamp posts and tree trunks) and the sample is small. It pairs every scenario with its standard
twin (same seed, same family, same pedestrians) so the effect of the furniture alone can be read off, and it writes its
own files under ``results/rich_world/`` and ``docs/RICH_WORLD.md``.

    python experiments/08_rich_world_eval.py                        # 20 seeds x 3 families x 4 controllers + 8 seeds of MPPI, both worlds
    python experiments/08_rich_world_eval.py --n 5 --n-mppi 0       # a quick smoke run
    python experiments/08_rich_world_eval.py --doc-only             # rewrite the document from the saved episodes
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from aedrover.analysis.experiments import controller_spec, make_jobs, run_job
from aedrover.analysis.report import _mcnemar, delivery_success, to_markdown
from aedrover.analysis.stats import wilson_ci
from aedrover.parallel import default_workers, pmap
from aedrover.sim import furniture as fur
from aedrover.sim.scenario import sample_scenario
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "rich_world"
FAMILIES = ("flat_clear", "crowded", "mixed")
CONTROLLERS = ("pure_pursuit", "apf", "dwa", "ppo", "mppi")


def tuned_mppi() -> dict:
    path = ROOT / "configs" / "mppi_tuned.json"
    return json.loads(path.read_text(encoding="utf-8"))["kwargs"] if path.exists() else {}


def run(args) -> pd.DataFrame:
    specs = []
    for name in CONTROLLERS:
        extra = {"path": str(ROOT / "models" / "ppo_selected")} if name == "ppo" else (tuned_mppi() if name == "mppi" else {})
        specs.append(controller_spec(name, 2.0, **extra))
    veh = VehicleParams.optimized()
    jobs = []
    for world in ("standard", "rich"):
        jobs += make_jobs(specs, FAMILIES, range(args.seed0, args.seed0 + args.n), shield=True, tag=world,
                          env_kwargs=(("veh", veh), ("rich", world == "rich")))
    jobs = [j for j in jobs if j.controller != "mppi" or j.seed < args.seed0 + args.n_mppi]
    jobs.sort(key=lambda j: j.controller != "mppi")                  # the slow planner first, for load balance
    print(f"{len(jobs)} episodes on {args.workers or default_workers()} workers", flush=True)
    t0 = time.perf_counter()
    df = pd.DataFrame(pmap(run_job, jobs, workers=args.workers, chunksize=2, desc="rich"))
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)
    df = df.rename(columns={"tag": "world"})
    df["safe"] = delivery_success(df)
    df["n_furniture"] = [len(fur.place_furniture(sample_scenario(f, s))) if w == "rich" else 0
                         for f, s, w in zip(df.family, df.seed, df.world, strict=True)]
    return df


def summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (ctrl, world), g in df.groupby(["controller", "world"]):
        for fam, h in [("all", g)] + [(f, g[g.family == f]) for f in FAMILIES]:
            n, k = len(h), int(h.safe.sum())
            lo, hi = wilson_ci(k, n)
            rows.append({"controller": ctrl, "world": world, "family": fam, "n": n, "safe_delivery": k / n, "ci_low": lo, "ci_high": hi,
                         "collision": float((h.outcome == "collision").mean()), "stall": float((h.outcome == "stall").mean()),
                         "other_failure": float((~h.safe & ~h.outcome.isin(["collision", "stall"])).mean()),
                         "median_time_s": float(h[h.success].time_s.median()) if h.success.any() else float("nan"),
                         "median_peak_shock_g": float(h.peak_shock_g.median()),
                         "median_min_clearance_m": float(h.min_clearance_m.median())})
    return pd.DataFrame(rows)


def paired(df: pd.DataFrame) -> pd.DataFrame:
    """Per controller, pooled over families: the same scenarios without and with furniture."""
    rows = []
    for ctrl, g in df.groupby("controller"):
        a = g[g.world == "standard"].set_index(["family", "seed"]).safe
        b = g[g.world == "rich"].set_index(["family", "seed"]).safe
        keys = a.index.intersection(b.index)
        a, b = a.loc[keys].to_numpy(bool), b.loc[keys].to_numpy(bool)
        p, rich_only, std_only = _mcnemar(a, b)
        rows.append({"controller": ctrl, "pairs": len(keys), "safe_standard": a.mean(), "safe_rich": b.mean(),
                     "change": b.mean() - a.mean(), "lost_to_furniture": std_only, "gained": rich_only, "mcnemar_p": p})
    return pd.DataFrame(rows)


def write_doc(df: pd.DataFrame, s: pd.DataFrame, p: pd.DataFrame, args) -> None:
    pct = lambda x: f"{100 * x:.0f}%"                                  # noqa: E731
    pooled = s[s.family == "all"].copy()
    table = pooled.assign(safe_delivery=pooled.safe_delivery.map(pct), ci=[f"{100 * lo:.0f}-{100 * hi:.0f}%" for lo, hi in
                                                                         zip(pooled.ci_low, pooled.ci_high, strict=True)],
                          collision=pooled.collision.map(pct), stall=pooled.stall.map(pct),
                          other_failure=pooled.other_failure.map(pct))[
        ["controller", "world", "n", "safe_delivery", "ci", "collision", "stall", "other_failure"]]
    pt = p.assign(safe_standard=p.safe_standard.map(pct), safe_rich=p.safe_rich.map(pct), change=(100 * p.change).map("{:+.0f} points".format),
                  mcnemar_p=p.mcnemar_p.map("{:.3f}".format))
    n_items = df[df.world == "rich"].drop_duplicates(["family", "seed"]).n_furniture
    sig = p[p.mcnemar_p < 0.05]
    pfmt = lambda v: "p < 0.001" if v < 0.0005 else f"p = {v:.3f}"          # noqa: E731
    drops = ", ".join(f"{r.controller} ({100 * r.change:+.0f} points, {pfmt(r.mcnemar_p)})" for r in sig.itertuples())
    noise = ", ".join(p[p.mcnemar_p >= 0.05].controller)
    off = df.groupby(["controller", "world"]).apply(lambda g: int((g.outcome == "off_sidewalk").sum()), include_groups=False)
    n_ppo = int((df.controller == "ppo").sum() // 2)
    shows = (f"Safe delivery changed significantly (exact McNemar, p < 0.05) for {drops or 'no controller'}; "
             f"for {noise or 'none of the controllers'} the change is within noise at this sample size. "
             f"PPO's drop is mostly leaving the sidewalk: {off.get(('ppo', 'rich'), 0)} of {n_ppo} episodes end `off_sidewalk` in the rich world "
             f"against {off.get(('ppo', 'standard'), 0)} in the standard one. The shielded pure-pursuit baseline, which never steers round anything, "
             f"turns most of the extra failures into stalls in front of the furniture rather than collisions, which is what the safety filter is for.")
    text = f"""# Rich-world evaluation

**Not comparable with `results/benchmark_standard.csv` or any number elsewhere in this repository.** This is a separate, small
experiment (`experiments/08_rich_world_eval.py`) in an opt-in environment (`AEDRoverEnv(rich=True)`), on seeds that are not the
benchmark's. Nothing here touches the standard environment, its seeds or its results (`tests/test_benchmark_lock.py`).

## What the rich world is

Parked cars, rows of motorbikes, street stalls, lamp posts and tree trunks, placed by a pure function of the scenario
(`aedrover.sim.furniture.place_furniture`) where a Mumbai footpath really has them: encroaching on the footway. They are static
collision boxes from the road up, seen by the lidar (so by the safety filter, DWA, APF and the PPO observation) and counted in the
episode's clearance and outcome (`collision_kind = "furniture"`). MPPI plans in a world with the same boxes and covers them with
obstacle discs. The same items drive the cinematic dressing, so what is drawn is what collides (`scripts/live_cinematic.py --rich`).

Feasibility is a rule, not luck: an item reaches at most {fur.MAX_INTRUSION_M:g} m past the corridor edge, so a lane of at least
{fur.MIN_CLEAR_LANE_M:g} m stays free at every x (the rover is 0.72 m wide); items on opposite sides keep {fur.SIDE_GAP_M:g} m apart along the
footway; nothing stands in the start zone, the goal zone, the road crossing or within {fur.OBSTACLE_GAP_M:g} m of a bollard or planter
(`tests/test_furniture.py` checks all of this over {5 * 40} scenarios). There are {n_items.mean():.1f} items per scenario on average (range
{int(n_items.min())} to {int(n_items.max())}).

## Protocol

{len(CONTROLLERS)} controllers at the benchmark's 2.0 m/s speed cap with the swept-footprint safety filter, the optimized rover,
families {", ".join(f"`{f}`" for f in FAMILIES)}, seeds {args.seed0} to {args.seed0 + args.n - 1} ({args.n} per controller and family;
MPPI: the first {args.n_mppi}). Every scenario is run twice, without and with furniture (same seed, same pedestrians), so the paired
comparison isolates the furniture. A *safe delivery* is the goal reached with payload shock within the 3 g budget, as in the benchmark.

## Results (pooled over the three families)

{to_markdown(table)}

Failures are split into `collision`, `stall` (the safety filter or planner stopped for more than 15 s without progress) and
other (off the sidewalk, rollover, timeout, shock over budget). Intervals are 95% Wilson.

### The furniture alone (paired, same scenarios)

{to_markdown(pt)}

`lost_to_furniture` counts scenarios delivered safely in the standard world and not in the rich one; `gained` the opposite.
`mcnemar_p` is the exact two-sided McNemar test on those pairs.

## What it shows

{shows}

## Reading it honestly

* **PPO is out of distribution here.** It was trained without any of this furniture; its observation (lidar) shows the boxes but
  it never learned to steer round them. Read its numbers as a robustness check, not as a fair comparison.
* The standard-world numbers in this table come from different seeds than `results/` and are not its numbers.
* The sample is small ({args.n} episodes per controller and family, fewer for MPPI), so differences of a few points are noise;
  `mcnemar_p` and the intervals say which changes are not.
* The furniture is a stylised set of boxes (pedestrians walk round it, nothing moves), and the layout rule keeps every scenario
  solvable, so this measures planners in clutter, not in worst cases.

## Reproduce

```
python experiments/08_rich_world_eval.py
```

writes `results/rich_world/eval.csv` (one row per episode), `results/rich_world/summary.csv` (per controller, world and family)
and `results/rich_world/paired.csv`, and regenerates this file.
"""
    (ROOT / "docs" / "RICH_WORLD.md").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20, help="seeds per controller and family")
    ap.add_argument("--n-mppi", type=int, default=8, help="seeds of MPPI per family (it is about 100x costlier)")
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--doc-only", action="store_true", help="rewrite docs/RICH_WORLD.md and the summaries from results/rich_world/eval.csv")
    args = ap.parse_args()
    df = pd.read_csv(OUT / "eval.csv") if args.doc_only else run(args)
    OUT.mkdir(parents=True, exist_ok=True)
    s, p = summary(df), paired(df)
    df.to_csv(OUT / "eval.csv", index=False)
    s.to_csv(OUT / "summary.csv", index=False)
    p.to_csv(OUT / "paired.csv", index=False)
    write_doc(df, s, p, args)
    pd.set_option("display.width", 200)
    print(p.round(3).to_string(index=False))
    print(df.groupby(["controller", "world"]).outcome.value_counts().unstack(fill_value=0).to_string())
    assert np.isfinite(df.time_s).all()


if __name__ == "__main__":
    main()
