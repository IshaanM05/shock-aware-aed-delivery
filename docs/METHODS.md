# Methods

Everything below describes what the code in `src/aedrover` does. Symbols use SI units; frames
follow REP-103 (x forward, y left, z up). Parameters marked ASSUMPTION are unsourced engineering
choices, exposed in code and varied in the robustness studies.

## 1. Vehicle model (`sim/vehicle_mjcf.py`)

A parametric MJCF generator for a four-wheel rover.

* **Suspension**: each wheel carrier slides vertically (`slide` joint) with spring `k_s` and damper
  `c_s`. The spring reference is set so that static equilibrium is the nominal ride height:
  `springref = -(M_s g / 4) / k_s`, with sprung mass `M_s` = chassis + payload.
* **Payload isolator**: the AED case is a 3-DOF (x, y, z) viscoelastic mount, `k_z, c_z` vertically
  and `k_xy, c_xy` laterally, carrying the accelerometer. Vertical reference `+m_p g / k_z`.
* **Drive**: hinge-y wheel spin driven by MuJoCo `velocity` actuators (P speed loop, gain 8 N m s/rad)
  with a torque clamp. A torque-speed derating scales the clamp by
  `clip(1 - |w| / (1.08 w_max), 0.05, 1)` each control step.
* **Steering**: front axle only, Ackermann geometry from a bicycle steering angle `delta`:
  `tan(d_left) = L tan(delta) / (L - (W/2) tan(delta))`, `tan(d_right) = L tan(delta) / (L + (W/2) tan(delta))`.
  Wheel speeds follow the distance of each wheel from the instantaneous centre of rotation.
* **Tyre contact**: rigid cylinder with soft contact (`solref = (0.03, 1)`), measured effective radial
  stiffness about 200 kN/m (nominal) and 310 kN/m (co-designed) by static penetration under load.
* **Timestep** 2 ms, `implicitfast` integrator, pyramidal friction cones, control at 50 Hz.

Two designs are used throughout: **nominal** (0.15 m wheels, `k_s` = 4500 N/m, `c_s` = 350 N s/m,
25 N m motors; the course draft values) and **optimized** (Section 8).

### Validation against closed-form mechanics

`experiments/01_validate_suspension.py`. Free bounce of the symmetric mode is compared with the
eigenvalues of the analytic 2-DOF quarter-car,
`M q'' + C q' + K q = 0` with `M = diag(m_s, m_u)`, `K = [[k_s, -k_s], [-k_s, k_s + k_t]]`,
`C = [[c_s, -c_s], [-c_s, c_s]]`, using the measured tyre stiffness `k_t`. Also checked: static ride
height, straight-line rolling speed error, and a random-command soak that must stay finite.

## 2. Payload shock metric (`sim/metrics.py`)

MuJoCo accelerometers report *proper* acceleration (about 1 g at rest). The shock signal is the
dynamic part

    a_dyn(t) = a_proper(t) - g * u_body(t)

where `u_body` is world-up expressed in the chassis frame (third row of the chassis rotation
matrix), so pitch and roll do not leak into the reading. `a_dyn` is sampled at 250 Hz, low-passed
with a zero-phase 4th-order Butterworth filter at 80 Hz, and the peak of its magnitude divided by g
is the reported **peak payload shock**. The default budget is **3 g**, a design requirement inherited
from the course brief, not a certified limit.

## 3. Scenarios and pedestrians (`sim/scenario.py`, `sim/pedestrians.py`)

A scenario is a 36 m sidewalk segment. Kerb families add a road crossing:
walk A `[0, 16 m]`, road `[16, 22 m]`, walk B `[22, 36 m]`, with a kerb of height `h` (uniform in
9-15 cm by default) at each transition, optionally replaced by a dropped-kerb ramp of slope 1:12.

| Family | Contents |
|---|---|
| `flat_clear` | flat ground, no pedestrians |
| `kerb` | crossing, friction 0.7-1.1, no pedestrians |
| `crowded` | flat, 3-6 pedestrians, 1-2 obstacles |
| `mixed` | crossing, 2-4 pedestrians, 1-3 obstacles |
| `slippery` | as mixed with friction 0.35-0.6 |

Payload mass is drawn from 3-5 kg. Pedestrians follow the social-force model of Helbing and Molnar
(1995): driving force toward the goal with relaxation time 0.5 s, exponential repulsion between
pedestrians, from the sidewalk edges and, for "aware" pedestrians (80 percent), from the rover.
Every pedestrian, aware or not, also avoids contact at very short range (last-moment sidestep).
The numeric strengths and ranges are ASSUMPTIONS (`SFMParams`). Pedestrians are a *stream*: they
walk the whole sidewalk piece and respawn at the entry end, but never within 6 m of the rover.

## 4. Perception (`sim/sensors.py`)

Shared by every controller. A 240 degree, 73-ray, 10 m 2D lidar; a 7 x 3 forward terrain-height scan
(0.35-4.5 m ahead, +/- 0.5 m, one batched ray cast from a mast point); and a noisy tracker for the
four nearest pedestrians (position sigma 0.05 m, velocity sigma 0.10 m/s).

## 5. Safety filter (`control/safety_filter.py`)

The rover footprint (1.04 x 0.72 m, oriented rectangle) is swept along the circular arc of the
commanded steering angle, `kappa = tan(delta) / L`. Let `d_free` be the first arc length at which the
footprint, inflated by 0.15 m ahead and 0.04 m to the sides (pedestrians: an extra 0.22 + 0.10 m),
overlaps a lidar return or a tracked pedestrian, minus one sampling step (0.1 m). The speed is limited
to the largest `v` satisfying the barrier

    d_free - v * t_react - v^2 / (2 a_brake) >= 0,     t_react = 0.1 s, a_brake = 2 m/s^2.

Within 3 m of a pedestrian the speed is additionally capped (1.4 m/s by default). The filter only
reduces speed.

## 6. Controllers

* **Pure pursuit** (`nav/pure_pursuit.py`): constant cruise speed, lookahead steering. The naive anchor.
* **Artificial potential field** (`nav/apf.py`; Khatib 1986): attraction along the path, repulsion from
  lidar returns and pedestrians inside 2.5 m, virtual sidewalk walls, plus the kerb negotiator.
* **Dynamic window** (`nav/dwa.py`; Fox et al. 1997): 9 speeds x 15 steering angles rolled out on a bicycle
  model over 2.4 s, three-circle footprint, constant-velocity pedestrian prediction, admissibility by
  stopping distance to contact, score = progress + lane centring + speed + clearance + heading, with an
  escape rule after 1.5 s standing still.
* **Kerb negotiator** (`control/curb.py`): detects a full-width step in the terrain scan, then schedules
  the approach speed from a lookup table measured in experiment 02 (minimum climbing speed for the
  detected height and an *assumed* friction 0.8, plus 0.15 m/s). Friction is not observed, so a
  conservative belief costs shock on grippy kerbs and an optimistic one stalls on wet kerbs.
* **MPPI** (`nav/mppi.py`; Williams et al. 2017; Howell et al. 2022): `K` = 128 sequences of (speed, steering)
  over 2 s (20 steps of 0.1 s, 5 spline knots), rolled out **in the MuJoCo model** with `mujoco.rollout`
  at a 10 ms step. Terrain, obstacles and the constant-velocity-predicted pedestrians enter the physics
  as mocap poses through `control_spec`. Cost: forward progress; payload shock above `0.75 x` budget from
  the simulated accelerometer; clearance of the oriented footprint to pedestrians and obstacles;
  sidewalk lane; heading; roll/pitch; control smoothness. The mean sequence is updated by the
  exponentially weighted average of the perturbations (temperature relative to the cost spread). The
  planner's model is deliberately imperfect: coarser timestep (which over-predicts kerb shock by about
  8 percent, conservatively), nominal friction 0.9 and payload 4 kg.
* **PPO** (`learning/`; Schulman et al. 2017): Stable-Baselines3 on CPU, 24 parallel environments,
  decisions at 10 Hz (action repeat 5), MLP 2 x 256 tanh, observation normalisation. The 123-dimensional
  observation contains goal-relative pose, velocity and attitude, the lidar, the terrain scan, the
  instantaneous payload acceleration, four pedestrian tracks and the previous action. Reward per decision:
  `+1.0 x progress - 0.05 x time - 4 x max(0, shock - 2.5 g)^2 - 0.5 x max(0, 0.6 m - clearance)
  - 0.02 x |delta action|^2`, `+25` at the goal, `-25` for collision, rollover or leaving the sidewalk, `-5`
  for a stall or timeout. **Domain randomisation** every episode (Tobin et al. 2017): scenario family
  mixture, kerb 6-16 cm, friction 0.5-1.2, payload, ramps, obstacles, pedestrians. Trained with the safety
  filter in the loop.

All controllers can be wrapped by the same safety filter, and the benchmark gives every controller the same
top speed.

## 7. Episodes and metrics (`sim/env.py`)

An episode ends on: goal (x >= 36 m), collision (footprint overlap with a pedestrian or obstacle), rollover
(roll or pitch beyond 1 rad), leaving the sidewalk, a stall (no 0.5 m progress for 15 s) or the time limit
(90 s). Reported per episode: outcome, time, path length, peak payload shock, minimum clearance, energy
(actuator power plus a 25 W ASSUMPTION baseline), and the scenario parameters. **Delivery success** means
the goal was reached *and* the peak shock stayed within the budget.

## 8. Mechanical co-design (`sim/codesign.py`, experiment 06)

Differential evolution over wheel radius (0.13-0.22 m), suspension stiffness (2500-8000 N/m) and damping
(150-700 N s/m), isolator stiffness (600-4000 N/m) and damping (40-400 N s/m), motor peak torque (15-45 N m).
Wheel mass scales as `1.65 (r / 0.15)^2 + 0.04 (tau - 25)` kg (ASSUMPTION), so bigger is not free. The
objective is the mean payload shock at the lowest climbing speed plus 0.2 m/s over six kerb-up conditions
(heights 10, 12, 14 cm; friction 0.6, 0.85), plus half the mean kerb-down shock at 12 and 16 cm; a condition
the design cannot climb is charged 8 g.

## 9. Clinical model (`clinical/`, `analysis/route_model.py`)

* **Survival** (Larsen et al. 1993): `S = 0.67 - 0.023 t_CPR - 0.011 t_defib - 0.021 t_ACLS`, times in minutes
  from collapse, coefficients read from the primary abstract. (The course draft used 0.046 for the
  defibrillation term, which is not the published value.) A rule-of-thumb model (7-10 percent per minute
  without CPR) is used only as a sensitivity check.
* **Ambulance**: response time lognormal with median and p90 from Naess et al. (2024) (urban: 10.0 and 17.7 min;
  rural: 14.8 and 33.3 min).
* **Time to first shock** for a device: collapse-to-call + call-to-alert + travel + hand-off. CPR and ACLS
  times are identical for every mode, so a device changes survival only through the defibrillation term.
  Dispatch is in parallel with the ambulance: the first shock is the earlier arrival.
* **Rover travel time** is not a constant speed. Per-segment simulated episodes are composed into a route
  (`route_model.py`) by bootstrap: a segment failure makes the route fail, and a segment with payload shock
  over the budget makes the delivery unsafe (treated like a failure).
* **Route geometry** from OpenStreetMap (Vile Parle, 2 km disc, 500 origin-destination pairs): median walking
  route factor 1.54, median driving route factor 1.71. OSM tags almost none of the crossings here, so the tagged
  crossing density (0.24 per km) is a lower bound; the analysis sweeps crossings per km.
* **Drone comparator** (`drone/`): a quadrotor with cascaded control, wind, an energy model and a
  weather-grounding probability, calibrated against its own full simulation and cross-checked against the
  published drone-AED median flight statistics. It is a comparator, not tuned to lose.

## 10. Statistics (`analysis/stats.py`, `analysis/report.py`)

All controllers run on the same seeds, so comparisons are paired by (family, seed). Success rates carry
Wilson intervals; paired success uses the exact McNemar test; time and shock use a paired t test (degrees of
freedom, t, exact p, Cohen's d_z with its interval, 95 percent interval of the mean difference) with a
Wilcoxon signed-rank companion, on episodes where both controllers succeeded. p values are Holm-adjusted
per metric.

## References

See [`REFERENCES.md`](REFERENCES.md); every DOI there is resolved and title-matched by
`scripts/verify_citations.py`.
