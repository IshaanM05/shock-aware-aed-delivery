# Pretrained policy

`ppo_selected/` is the PPO policy used for every PPO number in the README and in `docs/RESULTS.md`. It is small
(2.4 MB) so the benchmark can be reproduced without retraining.

| File | What it is |
|---|---|
| `final.zip` | Stable-Baselines3 PPO weights (the checkpoint at 3,499,944 steps, see below) |
| `vecnormalize.pkl` | observation and reward normalisation statistics; the policy is meaningless without them |
| `selection.json` | which training run and checkpoint were chosen, and the validation score that chose it |

```python
from aedrover.nav import make_controller
policy = make_controller("ppo", path="models/ppo_selected", v_max=2.0)   # v_max is the speed cap at test time
```

## How it was trained and chosen

* Two training runs of `python -m aedrover.learning.train_ppo` (24 environments, 10 Hz decisions, two hidden layers of
  256 tanh units, generalised advantage estimation, domain randomisation), both on the co-designed vehicle with the
  swept-footprint safety filter in the loop. Details and the learning curve are in `docs/PPO_TRAINING.md`.
* Every reset draws a scenario family from a fixed mixture (flat 5%, kerb 30%, crowded 20%, mixed 30%, slippery 15%)
  with kerb height 6-16 cm, friction 0.5-1.2, and randomised payload mass, ramps, obstacles and pedestrians.
* The reward penalises payload shock above 2.5 g, a margin under the 3 g evaluation budget.
* The policy commands a speed in [0, 2.6] m/s and a steering angle. The speed caps used in the benchmark (2.0 m/s) and
  the ablation are applied as a clip on that command at test time; the policy was never trained under a lower cap.
* The checkpoint was chosen among the saved checkpoints of both runs on validation seeds 50000-50019 (80 episodes,
  86.3% safe delivery). Benchmark, out-of-distribution and ablation seeds are disjoint from the validation seeds.

## What it does not show

It was trained and benchmarked on the same scenario distribution, in simulation only, on this one vehicle design. It
does not transfer to other vehicles or to real hardware, and it has not been trained for a slower speed cap. See
`docs/MODEL_CARD.md` for the full list of limits.
