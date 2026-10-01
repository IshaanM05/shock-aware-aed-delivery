# Drone comparator ("Track D")

The subject of `aedrover` is a sidewalk AED rover. The drone here is a deliberately small
comparator (an AED quadrotor) so that results can be reported as ambulance vs rover vs drone.
It is a physics-in-the-loop MuJoCo model plus a fast analytic model calibrated against it.

Frame convention: x forward (= track direction), y left, z up. SI units (m, s, kg, N, rad,
W, Wh). Minutes appear only in fields named `*_min`. Code: `src/aedrover/drone/`; tests:
`tests/test_drone.py`; calibration: `experiments/drone_calibration.py` (writes
`results/drone_calibration.json`).

**The drone is NOT tuned to lose.** Every performance-relevant default is either a sourced
number, the midpoint of a sourced range, or an explicit ASSUMPTION with a stated sensitivity
range. Section 6 lists the direction in which each assumption biases the drone-vs-rover
comparison (some against the drone, some in its favour), so the result can be reported with
both kinds of bias visible. Nothing here is fitted to make the drone look slow or fast.

## 1. Modules

| Module | Role |
| --- | --- |
| `quadrotor_mjcf.py` | `QuadParams` (frozen dataclass), `build_mjcf`, `QuadSim` (reset / step / state / imu). X-configuration, free joint, 4 site-based rotor actuators (first-order motor lag, gear `0 0 1 0 0 +/-c_q`), accelerometer + gyro + quaternion sensors at the centre of mass, flat ground. `integrator="implicitfast"`, timestep 0.002 s. |
| `flight_ctrl.py` | `CascadedController` with `ControllerGains`; `mixer_matrix`. |
| `wind.py` | `WindParams`, `WindField` (mean + Ornstein-Uhlenbeck gust, seeded), `drag_force`. |
| `energy.py` | `EnergyParams`, `rotor_power_w`, `hover_power_w`, `cruise_power_w`, `max_range_m`, `EnergyMeter`. |
| `mission.py` | `simulate_mission` (full simulation), `mission_time_s` / `time_to_scene_s` / `energy_wh` / `can_reach` (analytic), `p_available`, `max_hold_wind_mps`. |

## 2. Design and equations

### 2.1 Vehicle

Total mass m = airframe (6.0 kg) + payload (2.0 kg) = 8.0 kg; four rotors at
(+/-a, +/-a), a = L / sqrt(2), L = 0.45 m. Each rotor is commanded in thrust [N]; the applied
thrust follows a first-order lag with time constant 0.03 s and produces a yaw reaction torque
s_i c_q T_i with alternating signs s_i. Thrust-to-weight is 2.04.

### 2.2 Controller (all loops at 125 Hz)

```
v_cmd = v_ref + Kp_pos (p_ref - p)                 (correction magnitude-limited)
a_cmd = a_ref + Kp_vel (v_cmd - v) + Ki_vel * integral(v_cmd - v) dt
F_des = m (a_cmd + g e_z)                          (tilt-limited to 35 deg)
T     = F_des . z_body
R_d   = [x_d y_d z_d], z_d = F_des/|F_des|, y_d = z_d x x_c/|.|, x_d = y_d x z_d
e_R   = 0.5 (R_d^T R - R^T R_d)^vee
w_cmd = -Kp_att e_R
tau   = J (Kp_rate (w_cmd - w) - Kd_rate dw_filt/dt) + w x J w
f     = M^-1 [T, tau_x, tau_y, tau_z]              (M from rotor geometry; see mixer_matrix)
```

The SO(3) attitude error is the geometric formulation of `lee2010`; the cascaded
position/velocity/attitude/rate architecture is the standard multirotor structure
(`mahony2012`). No numbers are taken from either. Motor saturation policy: shed yaw torque
first, then scale the differential roll/pitch demand, then shift the collective, so attitude
authority is preserved over altitude. Gains are hand-tuned (ASSUMPTION).

### 2.3 Wind and drag

Wind = constant mean vector + gust; the gust is an Ornstein-Uhlenbeck process per axis
(the simplest Dryden-family model),

```
g[k+1] = g[k] exp(-dt/tau) + sigma sqrt(1 - exp(-2 dt/tau)) N(0,1)
```

initialised from its stationary distribution, with vertical sigma equal to 0.5 times the
horizontal sigma. The vehicle feels only body drag at the centre of mass,
`F = -0.5 rho Cd A |v_rel| v_rel`, `v_rel = v - v_wind`, applied through `xfrc_applied`.
`test_wind_is_seeded_and_gust_has_requested_statistics` checks the seeded reproducibility
and the mean / standard deviation.

### 2.4 Energy

```
P_i     = T_i^1.5 / sqrt(2 rho A) / FM        A = pi R^2 per rotor
P_total = sum_i P_i + P_avionics
E_usable = m_batt * e_spec * (1 - reserve_fraction)
```

The simulation integrates `P_total` from the applied (motor-lagged) rotor thrusts every
control tick. Steady cruise balances weight and drag, |T| = sqrt((m g)^2 + D^2). Default
numbers: hover power 860 W, battery 300 Wh nominal / 240 Wh usable, hover endurance 16.7 min,
steady-cruise range at 15 m/s of 14.6 km before climb and descent overhead.

### 2.5 Mission and the analytic model

Profile: launch, vertical climb to cruise altitude, straight cruise along +x, vertical
descent to release height, release. Every leg has a rest-to-rest trapezoidal velocity profile
(acceleration limited). The plan commands a fixed AIRSPEED: with wind component
`w` along the track (positive = tailwind, negative = headwind) the ground-speed reference is
`v_ground = clip(v_air + w, ., v_max)`, i.e. the planner is assumed to know the mean wind.
Headwind therefore lengthens the flight at unchanged drag power; strong headwind
(`v_air + w < 2 m/s`) or a tilt / thrust envelope violation makes the mission infeasible.
Release is declared when the plan has ended and the vehicle is within 0.5 m and 0.5 m/s of
the release point.

* `mission_time_s(d, w)`: closed-form sum of the three trapezoid durations plus the fitted
  offset (flight time, liftoff to release; `inf` if infeasible).
* `time_to_scene_s(d, w)`: launch latency (60 s, ASSUMPTION) + flight time.
* `energy_wh(d, w)`: quadrature of the momentum-theory power along the reference plan with
  the thrust vector `m (a + g e_z) - F_drag(v - v_wind)`, mean wind only, times the fitted scale.
* `p_available(wind_samples, wind_limit, rain_prob, p_night_grounded)`:
  `P(wind <= limit) * (1 - rain_prob) * (1 - p_night_grounded)` with the wind term the
  empirical CDF of the supplied samples. Non-decreasing in the wind limit. The grounding
  causes are treated as independent (see the assumption table). No site climate statistics
  are built in or claimed; the caller supplies samples.

## 3. Real numbers read from primary sources

Only what could actually be read is listed. Nothing else in the code is presented as
literature.

### 3.1 `schierbeck2023` (Lancet Digit Health; abstract via PubMed, full text NOT accessible)

Prospective observational study of AED drones dispatched in addition to EMS in Sweden:

* five AED-equipped drones in two controlled airspaces covering about 200,000 inhabitants;
  flights autonomous;
* study period 21 April 2021 to 31 May 2022: 211 suspected OHCA alerts, a drone was deployed
  in 72 (34 percent), AED delivery succeeded in 58 of 72 (81 percent); the major reason for
  non-delivery was cancellation by the dispatch centre because the case was not an OHCA;
* where both drone and ambulance arrival times were available (n = 55), the drone arrived
  first in 37 cases (67 percent) with a median time benefit of 3 min 14 s;
* alerts about children under 8 years, trauma and EMS-witnessed cases were not included;
  exclusion criteria were air-traffic-control non-approval of the flight, unfavourable
  weather, no-delivery zones and darkness;
* among the 37 early arrivals, 18 (49 percent) were true OHCA and a drone AED was attached
  in six (33 percent); two had a shockable first rhythm and were defibrillated, one person
  survived to 30 days; no adverse events; delivery (not landing) was within 15 m of the
  patient or building in 91 percent of cases.

NOT verified (publisher full text returned HTTP 403 and there is no open PubMed Central
copy): flight distances, flight times, dispatch-to-launch latency, and the numeric weather
limits. These are therefore ASSUMPTIONS below and are not attributed to the paper. Note the
34 percent deployment fraction mixes weather, airspace, darkness and the study's own clinical
inclusion rules, so it is NOT used as a weather-availability estimate.

### 3.2 `claesson2017` (JAMA; PubMed Central full text)

Eighteen consecutive autonomous flights of an 8-rotor drone to simulated OHCA locations:

| Item | Value |
| --- | --- |
| Drone mass | 5.7 kg |
| Maximum cruising speed | 75 km/h (20.8 m/s) |
| AED payload | 763 g |
| Flight distance | median 3.2 km, range 15 m to 8927 m |
| Dispatch to launch | median 3 s |
| Dispatch to arrival | median 5 min 21 s (IQR 3 min 03 s to 8 min 33 s) |
| EMS dispatch to arrival | median 22 min 00 s |
| Weather | small number of flights over short distances in good weather (a stated limitation) |

### 3.3 `stolaroff2018` (Nat Commun; PubMed Central full text)

* overall power efficiency fitted to measured multicopter data of about 50 percent at low
  speed and about 70 percent at higher speed;
* body drag coefficient C_D,body = 1.5;
* battery specific energy 150 Wh/kg (540 kJ/kg) and depth of discharge 80 percent.

`lee2010` and `mahony2012` are cited for controller structure only.

## 4. ASSUMPTIONS table

"Range" is the sensitivity range documented for sweeps. Sourced rows say which key; everything
else is an ASSUMPTION with no primary source.

| Parameter | Value | Range | Source |
| --- | --- | --- | --- |
| Airframe + battery mass | 6.0 kg | 5-7 kg | ASSUMPTION (class of 5.7 kg in `claesson2017`) |
| AED payload (AED + case + release) | 2.0 kg | 0.763-2.5 kg | ASSUMPTION; `claesson2017` reports a 0.763 kg AED, so 2.0 kg is conservative |
| Arm length | 0.45 m | 0.35-0.55 m | ASSUMPTION |
| Inertia (Ixx, Iyy, Izz) | 0.25, 0.25, 0.45 kg m^2 | +/-30 percent | ASSUMPTION |
| Rotor radius | 0.25 m | 0.20-0.30 m | ASSUMPTION |
| Thrust coefficient k_thrust | 3.0e-4 N s^2/rad^2 | n/a (reporting only) | ASSUMPTION |
| Yaw torque ratio c_q | 0.02 m | 0.01-0.03 m | ASSUMPTION |
| Max thrust per rotor | 40 N (thrust-to-weight 2.04) | 30-50 N | ASSUMPTION |
| Motor time constant | 0.03 s | 0.02-0.06 s | ASSUMPTION |
| Body drag coefficient Cd | 1.5 | fixed | `stolaroff2018` |
| Frontal area A | 0.08 m^2 | 0.05-0.12 m^2 | ASSUMPTION |
| Air density | 1.225 kg/m^3 | 1.10-1.25 kg/m^3 | ASSUMPTION (ISA sea level; site not specified) |
| Figure of merit / overall efficiency | 0.6 | 0.5-0.7 | Range from `stolaroff2018`; mapping overall efficiency onto FM and the midpoint are ASSUMPTIONS |
| Avionics load | 25 W | 10-60 W | ASSUMPTION |
| Battery specific energy | 150 Wh/kg | fixed | `stolaroff2018` |
| Battery mass | 2.0 kg (300 Wh) | 1.5-3.0 kg | ASSUMPTION (part of airframe mass) |
| Reserve fraction | 0.20 | fixed | `stolaroff2018` (depth of discharge 80 percent) |
| Cruise airspeed | 15 m/s | 10-20.8 m/s | ASSUMPTION; upper end is the `claesson2017` maximum cruise, lower end the roughly 10 m/s mean speed implied by its medians |
| Cruise altitude | 50 m | 30-120 m | ASSUMPTION (no airspace rules modelled) |
| Release height | 2 m | 1-5 m | ASSUMPTION |
| Climb / descent speed | 4 / 3 m/s | 2-6 m/s | ASSUMPTION |
| Horizontal / vertical acceleration limit | 2 / 2 m/s^2 | 1-3 m/s^2 | ASSUMPTION |
| Ground-speed cap / minimum | 25 / 2 m/s | n/a | ASSUMPTION |
| Launch latency (alert to liftoff) | 60 s | 3-180 s | ASSUMPTION; `claesson2017` gives 3 s for a research drone without an air-traffic-control step, `schierbeck2023` lists air-traffic-control approval as a real-service step but its latency is not accessible |
| Operational wind limit | 10 m/s | 6-15 m/s | ASSUMPTION; `schierbeck2023` lists unfavourable weather as an exclusion without an accessible threshold. Physical hold limit of this vehicle (tilt 35 deg) is 27.3 m/s, an upper bound only |
| Rain grounding probability | caller supplied | 0-0.20 | ASSUMPTION; no site statistics claimed |
| Night grounding probability | 0 | 0-0.5 | ASSUMPTION; `schierbeck2023` excluded darkness, so sweep it |
| Grounding causes independent | yes | n/a | ASSUMPTION |
| Gust sigma / correlation time / vertical scale | swept 0-3 m/s / 5 s / 0.5 | tau 2-15 s | ASSUMPTION |
| Planner knows mean wind | yes | n/a | ASSUMPTION |
| Controller gains, tilt limit 35 deg | see `ControllerGains` | n/a | ASSUMPTION (hand tuned) |
| Calibration: time offset | 0.023 s | n/a | FITTED by `drone_calibration.py` (0.0228 s) |
| Calibration: energy scale | 1.0 | n/a | FITTED (1.00002), no correction applied |

## 5. Calibration and validation results

`experiments/drone_calibration.py` (12 gust-free fit runs, 12 gusty out-of-sample runs;
distances 500, 1500, 3000 m; along-track wind +5, 0, -5, -9 m/s; gust sigma 1.5 m/s, seed 11):

| Set | Max abs time error | Max abs energy error |
| --- | --- | --- |
| Fit (gust-free) | 0.004 percent | 0.012 percent |
| Validation (gusts) | 0.004 percent | 0.73 percent |

These are errors with the shipped constants (time offset 0.023 s, energy scale 1.0). Peak
3-D tracking error 0.17-0.28 m and peak tilt 21-25 deg over the sweep, inside the 35 deg
limit. The tests require 10 percent (time) and 15 percent (energy); the margin is large.

Read this agreement honestly. The analytic time is the closed-form duration of the same
reference plan the simulation flies, and the analytic energy applies the same rotor-power
formula to the same thrust vector, so close agreement is largely by construction. What the
simulation independently establishes is that the cascaded controller actually tracks that plan
under gusts and wind (bounded tracking error, no saturation), that thrust imbalance, motor lag
and gusts add at most about 0.7 percent to energy, and where the feasible envelope ends. The
agreement is NOT evidence that the vehicle matches a real drone.

The only independent real-world check is a plausibility comparison with `claesson2017`. For
a 3.2 km flight the model gives 253 s of flight time at the default 15 m/s airspeed (256 s
with that study's 3 s launch), against a reported median dispatch-to-arrival of 321 s (IQR
183-513 s). The default is therefore about 20 percent faster than that research drone's median;
at 12 m/s the model gives 305 s and at 10 m/s 357 s, bracketing the reported median. Medians of
different distributions do not compose exactly, so this is a plausibility check, not a fit.

Sensitivity of the analytic model (3000 m, still air; baseline 239.5 s and 58.9 Wh):
figure of merit 0.5 / 0.7 gives energy +19 / -14 percent; a 0.763 kg AED instead of 2.0 kg gives
-21 percent energy; airspeed 10 / 20 m/s gives time +41 / -20 percent and energy +38 / -16
percent; cruise altitude 30 / 120 m gives time -5 / +17 percent; avionics 10 / 60 W gives
energy -2 / +4 percent. The full table is in `results/drone_calibration.json`.

Illustrative outputs (still air, default parameters): 1000 m takes 106 s of flight (166 s with
the 60 s launch latency) and 26 Wh; 3000 m takes 240 s (300 s) and 59 Wh; the energy budget
limits the one-way range to about 14 km in still air, 9.4 km in a 5 m/s headwind and 5.6 km in
a 9 m/s headwind.

## 6. Direction of each bias

Against the drone: 2 kg payload versus the 0.763 kg AED of `claesson2017` (about 21 percent
more energy); hover-form momentum theory at all speeds (no translational-lift saving in
forward flight); independence of the grounding causes (positive wind-rain correlation would
raise availability); a 60 s launch latency that is well above the 3 s reported for a research
drone.

In favour of the drone: straight-line flight with no airspace, obstacle or no-delivery-zone
routing; true-state feedback (no estimator error or sensor noise); the planner knows the mean
wind; no battery voltage sag; default cruise speed about 20 percent faster than the median of
`claesson2017`; night flying allowed by default.

Neither direction is known: launch latency and the operational wind limit of a real service
are not in the accessible text of `schierbeck2023`, so sweep them across the documented ranges.

## 7. Limitations

* No rotor aerodynamics beyond quadratic body drag: no inflow, blade flapping, ground effect
  or vortex-ring state, and drag acts at the centre of mass (no drag torque).
* No battery voltage sag, temperature or Peukert effect; energy is a constant-efficiency
  momentum-theory estimate with a fixed reserve.
* No regulatory or airspace constraints: no air-traffic approval, no-delivery zones, altitude
  ceilings, beyond-visual-line-of-sight rules or obstacle avoidance; flight is a straight line.
* True-state feedback and exact mass knowledge; there is no estimator, sensor noise, latency
  or GPS error.
* Wind is a mean plus stationary OU gust; no shear, thermals, urban canyon effects or
  turbulence spectra from measured data, and no site-specific climate statistics are used or
  claimed (in particular nothing for Mumbai). `p_available` needs samples from the caller.
* Release is an event, not a payload-drop model; the drone stops at 2 m and mass is not changed.
* The analytic models are valid inside the tilt/thrust envelope reported by `is_feasible`
  and return `inf` outside it, rather than an extrapolated number.

## 8. Reproduce

```
python -m pytest tests/test_drone.py -q
python experiments/drone_calibration.py            # about 1.5 min; --quick for a smoke run
ruff check src/aedrover/drone tests/test_drone.py experiments/drone_calibration.py
python scripts/verify_citations.py
```

Usage from other tracks: `time_to_scene_s(distance_m, wind_along_track_mps)` for
drone time-to-scene (launch latency + flight), `energy_wh` and `can_reach` for the battery
check, and `p_available(wind_samples, wind_limit_mps, rain_prob)` for the weather factor.
Wind along the track is positive for a tailwind.

## 9. How this model relates to the clinical comparison

The survival numbers in `results/clinical_*.csv` do **not** use the `aedrover.drone` package. The clinical decision model
(`src/aedrover/clinical/decision.py`, `ScenarioParams`) gives the drone a simpler timing: `drone_launch_min = 0.5` (30 s)
plus a straight flight at `drone_speed_mps = 15`, with no climb, no descent and no acceleration limit. Nothing under
`aedrover.clinical` imports `aedrover.drone`.

| Radius | Clinical model (since dispatch) | This simulation (`time_to_scene_s`) | Difference |
| --- | --- | --- | --- |
| 500 m | 63.3 s | 132.9 s | 69.5 s |
| 1000 m | 96.7 s | 166.2 s | 69.5 s |

The difference is the 30 s of extra launch latency (60 s here, 30 s there) plus about 40 s for the 50 m climb, the 48 m
descent and the acceleration limits. Using the simulated timing in the clinical model (same arrests, same seed,
`drone_launch_min` raised by the difference, 1.16 min) moves the drone's survival at 1 km from 29.7% to 28.5% (95%
interval 28.0 to 29.0), a gain over the ambulance of +5.2 points instead of +6.4. The conclusion (a drone that can always
fly helps at 1 km, a rover does not) is unchanged, but the published drone numbers are slightly optimistic with respect
to the flight simulated in this document. The sensitivity is reproduced by:

```python
import numpy as np
from aedrover.clinical.decision import ScenarioParams, evaluate_modes, mode_times
from aedrover.clinical.survival import get_model
from aedrover.drone.mission import time_to_scene_s

gap_min = (time_to_scene_s(1000.0) - (30.0 + 1000.0 / 15.0)) / 60
p = ScenarioParams(radius_m=1000.0, route_factor=1.5352, drone_launch_min=0.5 + gap_min)
samples = mode_times(2000, np.random.default_rng(0), p, modes=("ambulance", "drone"), parallel=True)
print(evaluate_modes(samples, get_model("larsen1993"), seed=0, n_resamples=500))
```

The film's rover-versus-drone shot (`docs/RENDERING.md`) uses the clocks of this document (60 s latency plus the flown
mission) and states the difference next to the survival figures it quotes from the CSV.
