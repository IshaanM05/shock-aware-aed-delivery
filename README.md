# GoldenMinute

**Shock-aware autonomous sidewalk AED delivery in MuJoCo.** A four-wheel rover that carries an
automated external defibrillator (AED) to a cardiac-arrest patient, studied end to end: the
mechanical design that lets it cross kerbs without damaging its payload, physics-in-the-loop MPC
and learned policies against classical navigation, and a clinical break-even analysis against
ambulance and drone delivery.

> Status: work in progress. Sections marked **pending** are being finalised; every number below
> is measured by code in this repository and can be regenerated. Nothing is quoted from memory.

![Rover crossing a 13.5 cm kerb (MuJoCo render; draft vehicle, DWA controller)](assets/kerb_dwa_stills.png)

![System overview](docs/figures/architecture.png)

## The question

Sudden cardiac arrest survival falls steeply with every minute before the first shock, and
urban ambulances are often slow. Can a ground robot on the sidewalk get an AED to the patient
sooner, and under what conditions does it actually help? The starting point is a course brief
from the NMIMS MPSTME "Modern Day Robotics and Its Industrial Applications" (2026) project list
(autonomous last-mile ground AED delivery). This repository takes that brief much further and
turns it into three testable research questions.

| | Question | Where |
|---|---|---|
| **RQ1** | For a given kerb height, wheel size, friction and suspension, when can the rover cross a kerb and keep peak payload acceleration under a stated budget (3 g by default)? | experiments 01, 02, 06 |
| **RQ2** | Do physics-in-the-loop MPPI or a domain-randomised PPO policy beat classical navigation (pure pursuit, potential field, dynamic window) on time, payload shock and safety, and how do they degrade out of distribution? | experiments 03, 04 |
| **RQ3** | With published survival models and measured ambulance delays, at what response radius does the rover improve expected survival, and how does it compare with a drone and a hybrid dispatch policy? | experiments 05, 07 |

## What has been measured so far

**The simulator is validated against closed-form mechanics** (`experiments/01_validate_suspension.py`).
Free-bounce ring-down of the suspension matches the analytic two-degree-of-freedom quarter-car
eigenvalues: natural frequency within 1% and damping ratio within 4% for both vehicle designs,
straight-line rolling within 1% of the commanded speed, and random-command soak runs stay finite.

**A 12 cm kerb needs momentum.** Over 6,048 simulated trials (`experiments/02_curb_traversability.py`),
the minimum speed at which the draft rover climbs a sharp kerb rises with kerb height and falls
with friction: for a 12 cm kerb it is 1.0 m/s on dry paving (mu = 1.0) and 2.2 m/s on wet paving
(mu = 0.5), and the payload shock at that speed is about 3 g and about 6 g respectively. Speed
gets it over the kerb; speed is also what shakes the payload.

**Mechanical co-design fixes most of it** (`experiments/06_mech_codesign.py`, differential
evolution over wheel radius, suspension, payload isolator and motor size, 1,116 evaluations).

| Condition (kerb up) | Draft design | Co-designed | 
|---|---|---|
| 12 cm, mu = 0.85 | 4.47 g | 2.24 g |
| 14 cm, mu = 0.85 | 5.36 g | 2.97 g |
| 14 cm, mu = 0.60 | 6.98 g | 3.29 g |
| 12 cm kerb-down | 2.09 g | 1.05 g |

Share of the eight test conditions within the 3 g budget: **25% -> 88%**. The search pushes the
wheel radius to its 0.22 m packaging bound and wants a soft suspension and a soft payload
isolator, so the design is wheel-radius-limited, not stiffness-limited.

**Classical navigation on the co-designed vehicle** (40 episodes per cell, paired seeds, safety
filter on): the kerb scenario is solved 40 out of 40 times by the dynamic-window planner with a
kerb negotiator, at a median payload shock of 2.4 g. Reactive planners struggle with dense
pedestrian streams, which is exactly where the sampling-based and learned controllers are being
compared.

**A learned policy trains on a laptop CPU.** PPO with domain randomisation (24 environments, 12 million
decisions, 69 minutes, `docs/PPO_TRAINING.md`) reaches the goal in 82.5% of 200 held-out episodes versus 69.0% for
the dynamic-window planner on the same seeds (paired exact McNemar, Holm-adjusted p = 0.007), and is faster. That
timing gap is partly a speed-cap effect (the policy may command 2.6 m/s, the planner cruises at 1.8 m/s), so the
main benchmark below gives every controller the same cap. Switching the safety filter off barely changes the
policy's results, so it is not leaning on the filter.

**Pending**: the full controller benchmark with equal speed caps (100 paired episodes per controller and scenario
family), MPPI and the second PPO seed in that benchmark, the out-of-distribution study, ablations, and the clinical
analysis. The real Mumbai route geometry from OpenStreetMap is already in (`docs/OSM_ROUTES.md`, data under `data/osm`).

## How it works

* **Vehicle and world** (`src/aedrover/sim`): a parametric MJCF generator. Four independent
  slide-spring-damper suspensions, wheel-spin hinges with torque-limited velocity actuators, front
  Ackermann steering, and a three-degree-of-freedom viscoelastic payload isolator that carries the
  accelerometer. Terrain, obstacles and pedestrians are re-positionable mocap slots, so a new
  scenario is an array write, not a recompile.
* **Scenarios**: five seeded families (flat, kerb, crowded, mixed, slippery) with randomised kerb
  height, friction, payload mass, dropped-kerb ramps, obstacles and social-force pedestrian streams.
* **Controllers** (`src/aedrover/nav`): pure pursuit, artificial potential field, dynamic window
  approach, MPPI that rolls out the real MuJoCo model with `mujoco.rollout`, and a PPO policy. All of
  them can be wrapped by the same swept-footprint stopping-distance safety filter, so comparisons
  hold identical safety constraints.
* **Clinical layer** (`src/aedrover/clinical`): the Larsen (1993) survival model with the published
  coefficients, ambulance response times from Naess et al. (2024), a calibrated quadrotor comparator,
  route-level composition of simulated segments, and dispatch policies.
* **Statistics** (`src/aedrover/analysis`): Welch and paired t tests with degrees of freedom and
  exact p, Cohen's d with confidence intervals, Wilson intervals, exact McNemar for paired success,
  Wilcoxon and Mann-Whitney companions, and Holm correction.

## Documentation

| Document | What it covers |
|---|---|
| [`docs/METHODS.md`](docs/METHODS.md) | the model, controllers, metrics and clinical layer exactly as implemented |
| [`docs/VALIDATION.md`](docs/VALIDATION.md) | physics validation against closed-form mechanics (generated from results) |
| [`docs/RESULTS.md`](docs/RESULTS.md) | generated tables, figures and paired statistics (appears once the pipeline has run) |
| [`docs/PPO_TRAINING.md`](docs/PPO_TRAINING.md) | learning curve and held-out evaluation, regenerated automatically |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | every command in order, run times and seed ranges |
| [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) | what the study supports and what it does not |
| [`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md) | 14 silent MuJoCo pitfalls, each with a regression test |
| [`docs/NMIMS_EXPORT.md`](docs/NMIMS_EXPORT.md) | generating the course-template folder from this repository |

## Engineering notes worth reading

[`docs/ENGINEERING_NOTES.md`](docs/ENGINEERING_NOTES.md) records 14 MuJoCo behaviours that silently
produce wrong physics or wrong data without raising an error (for example, `mujoco.rollout` resets
mocap bodies, which deletes all terrain from planner rollouts; `mj_multiRay`'s cutoff drops infinite
planes; `<map znear>` is in units of model extent). Each has a regression test.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,viz]"       # add ",rl" for PPO training (PyTorch, Stable-Baselines3)
pytest -m "not slow" -n auto      # about 200 tests
python experiments/01_validate_suspension.py
python experiments/03_controller_benchmark.py --n 10 --tag smoke --controllers pure_pursuit apf dwa
python scripts/render_demo.py --controller dwa --family kerb --seed 1003 --name kerb_dwa
```

Everything runs on CPU. A GPU is not required and is not used.

## Repository layout

```
src/aedrover/   sim  nav  control  learning  drone  clinical  analysis  geo
experiments/    01 validation  02 kerb map  03 benchmark  05 clinical  06 co-design
configs/        vehicle and lookup-table configs (vehicle_optimized.yaml is the co-designed rover)
results/        small committed summaries (CSV and JSON); large artifacts are git-ignored
docs/           ENGINEERING_NOTES, CLINICAL_MODEL, DRONE_COMPARATOR, REFERENCES (all DOIs verified)
scripts/        verify_citations.py  render_demo.py  build_curb_table.py
tests/          physics, metrics, sensors, environment, safety filter, MPPI, statistics, clinical
```

## References

Every citation in [`docs/REFERENCES.md`](docs/REFERENCES.md) is checked against Crossref or
DataCite by `scripts/verify_citations.py` (DOI resolves and the registered title matches).

## Limitations

This is a simulation study. Wheels are rigid cylinders with a calibrated contact softness, weather
and sensor faults are not modelled beyond the noisy pedestrian tracker, pedestrians follow a
social-force model rather than recorded trajectories, ambulance delays come from a published study
rather than local data, and survival models come from Western cohorts. Economics are the course's
dimensionless model with assumed ratios and are illustrative only. The drone is a comparator, not
the subject, and is not tuned to lose.

## Licence

MIT. See [`LICENSE`](LICENSE).
