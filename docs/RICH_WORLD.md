# Rich-world evaluation

**Not comparable with `results/benchmark_standard.csv` or any number elsewhere in this repository.** This is a separate, small
experiment (`experiments/08_rich_world_eval.py`) in an opt-in environment (`AEDRoverEnv(rich=True)`), on seeds that are not the
benchmark's. Nothing here touches the standard environment, its seeds or its results (`tests/test_benchmark_lock.py`).

## What the rich world is

Parked cars, rows of motorbikes, street stalls, lamp posts and tree trunks, placed by a pure function of the scenario
(`aedrover.sim.furniture.place_furniture`) where a Mumbai footpath really has them: encroaching on the footway. They are static
collision boxes from the road up, seen by the lidar (so by the safety filter, DWA, APF and the PPO observation) and counted in the
episode's clearance and outcome (`collision_kind = "furniture"`). MPPI plans in a world with the same boxes and covers them with
obstacle discs. The same items drive the cinematic dressing, so what is drawn is what collides (`scripts/live_cinematic.py --rich`).

Feasibility is a rule, not luck: an item reaches at most 1.2 m past the corridor edge, so a lane of at least
1.6 m stays free at every x (the rover is 0.72 m wide); items on opposite sides keep 4 m apart along the
footway; nothing stands in the start zone, the goal zone, the road crossing or within 3 m of a bollard or planter
(`tests/test_furniture.py` checks all of this over 200 scenarios). There are 3.9 items per scenario on average (range
0 to 8).

## Protocol

5 controllers at the benchmark's 2.0 m/s speed cap with the swept-footprint safety filter, the optimized rover,
families `flat_clear`, `crowded`, `mixed`, seeds 7000 to 7019 (20 per controller and family;
MPPI: the first 8). Every scenario is run twice, without and with furniture (same seed, same pedestrians), so the paired
comparison isolates the furniture. A *safe delivery* is the goal reached with payload shock within the 3 g budget, as in the benchmark.

## Results (pooled over the three families)

| controller | world | n | safe_delivery | ci | collision | stall | other_failure |
| --- | --- | --- | --- | --- | --- | --- | --- |
| apf | rich | 60 | 67% | 54-77% | 2% | 22% | 10% |
| apf | standard | 60 | 73% | 61-83% | 0% | 17% | 10% |
| dwa | rich | 60 | 53% | 41-65% | 3% | 37% | 7% |
| dwa | standard | 60 | 67% | 54-77% | 7% | 22% | 5% |
| mppi | rich | 24 | 71% | 51-85% | 0% | 12% | 17% |
| mppi | standard | 24 | 71% | 51-85% | 4% | 17% | 8% |
| ppo | rich | 60 | 60% | 47-71% | 0% | 12% | 28% |
| ppo | standard | 60 | 85% | 74-92% | 2% | 10% | 3% |
| pure_pursuit | rich | 60 | 20% | 12-32% | 3% | 75% | 2% |
| pure_pursuit | standard | 60 | 40% | 29-53% | 7% | 50% | 3% |

Failures are split into `collision`, `stall` (the safety filter or planner stopped for more than 15 s without progress) and
other (off the sidewalk, rollover, timeout, shock over budget). Intervals are 95% Wilson.

### The furniture alone (paired, same scenarios)

| controller | pairs | safe_standard | safe_rich | change | lost_to_furniture | gained | mcnemar_p |
| --- | --- | --- | --- | --- | --- | --- | --- |
| apf | 60 | 73% | 67% | -7 points | 7 | 3 | 0.344 |
| dwa | 60 | 67% | 53% | -13 points | 10 | 2 | 0.039 |
| mppi | 24 | 71% | 71% | +0 points | 2 | 2 | 1.000 |
| ppo | 60 | 85% | 60% | -25 points | 16 | 1 | 0.000 |
| pure_pursuit | 60 | 40% | 20% | -20 points | 12 | 0 | 0.000 |

`lost_to_furniture` counts scenarios delivered safely in the standard world and not in the rich one; `gained` the opposite.
`mcnemar_p` is the exact two-sided McNemar test on those pairs.

## What it shows

Safe delivery changed significantly (exact McNemar, p < 0.05) for dwa (-13 points, p = 0.039), ppo (-25 points, p < 0.001), pure_pursuit (-20 points, p < 0.001); for apf, mppi the change is within noise at this sample size. PPO's drop is mostly leaving the sidewalk: 17 of 60 episodes end `off_sidewalk` in the rich world against 2 in the standard one. The shielded pure-pursuit baseline, which never steers round anything, turns most of the extra failures into stalls in front of the furniture rather than collisions, which is what the safety filter is for.

## Reading it honestly

* **PPO is out of distribution here.** It was trained without any of this furniture; its observation (lidar) shows the boxes but
  it never learned to steer round them. Read its numbers as a robustness check, not as a fair comparison.
* The standard-world numbers in this table come from different seeds than `results/` and are not its numbers.
* The sample is small (20 episodes per controller and family, fewer for MPPI), so differences of a few points are noise;
  `mcnemar_p` and the intervals say which changes are not.
* The furniture is a stylised set of boxes (pedestrians walk round it, nothing moves), and the layout rule keeps every scenario
  solvable, so this measures planners in clutter, not in worst cases.

## Reproduce

```
python experiments/08_rich_world_eval.py
```

writes `results/rich_world/eval.csv` (one row per episode), `results/rich_world/summary.csv` (per controller, world and family)
and `results/rich_world/paired.csv`, and regenerates this file.
