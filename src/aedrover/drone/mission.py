"""AED delivery mission for the comparator quadrotor: full simulation plus a fast analytic model.

Mission profile: launch -> vertical climb to cruise altitude -> straight-line cruise along +x
-> vertical descent to release height -> release. Each leg follows a trapezoidal velocity
profile (acceleration-limited), so the reference is smooth and the vehicle starts and ends every
leg at rest.

Wind convention: the track runs along +x; ``wind_along_track_mps`` is the x component of the
wind velocity vector (POSITIVE = TAILWIND, negative = headwind) and ``wind_cross_mps`` the y
component (m/s). The flight plan commands a fixed AIRSPEED ``v_air_mps``: the planner knows the
mean wind (forecast / estimator; ASSUMPTION) and sets the ground-speed reference to
``v_ground = clip(v_air + wind_along_track, ., max_ground_speed)``. Headwind therefore lengthens
the flight at unchanged drag power, and a tailwind shortens it.

Total drone time-to-scene = launch latency + flight time
(``time_to_scene_s``); ``mission_time_s`` is the FLIGHT time only (liftoff to release).

Sources and assumptions:
    * Real-world flight-time context: ``claesson2017`` (18 simulated-OHCA flights with an 8-rotor
      5.7 kg drone, median distance 3.2 km, range 15 m to 8927 m, maximum cruise speed 75 km/h,
      median dispatch-to-launch 3 s, median dispatch-to-arrival 5 min 21 s). Used ONLY as an
      order-of-magnitude cross-check, never fitted.
    * ``schierbeck2023``: real service exclusions (air-traffic-control non-approval, unfavourable
      weather, no-delivery zones, darkness). Its weather thresholds and launch latency are not in
      the accessible abstract, so they are ASSUMPTIONS below.
    * Launch latency 60 s, cruise airspeed 15 m/s, cruise altitude 50 m, release height 2 m,
      climb/descent speeds, accelerations, ground-speed caps and the weather limit are all
      ASSUMPTIONS; the sensitivity ranges are in docs/DRONE_COMPARATOR.md.
    * ``time_offset_s`` and ``energy_scale`` are FITTED to this simulator by
      experiments/drone_calibration.py (see that script for the residuals).

Limitations: straight-line flight only (no obstacle or airspace routing), true-state feedback,
no rotor aerodynamics beyond drag, no battery voltage sag.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from aedrover.drone.energy import EnergyMeter, EnergyParams, hover_power_w, thrust_power_w
from aedrover.drone.flight_ctrl import CascadedController, ControllerGains, Reference
from aedrover.drone.quadrotor_mjcf import G, QuadParams, QuadSim
from aedrover.drone.wind import WindField, WindParams, drag_force

CONTROL_DECIMATION = 4  # physics steps per control update (dt_ctrl = 4 * 2 ms = 8 ms)


@dataclass(frozen=True)
class MissionParams:
    """Mission plan and calibration constants (units in the field names).

    All plan values are ASSUMPTIONS unless noted.

    Attributes:
        v_air_mps: commanded cruise airspeed [m/s]. ASSUMPTION (between the ~10 m/s mean
            speed implied by claesson2017 medians and its 20.8 m/s maximum cruise).
        cruise_alt_m: cruise altitude above the start [m]. ASSUMPTION.
        release_height_m: release height above ground [m]. ASSUMPTION.
        climb_speed_mps: maximum climb speed [m/s]. ASSUMPTION.
        descent_speed_mps: maximum descent speed [m/s]. ASSUMPTION.
        accel_h_mps2: horizontal acceleration limit [m/s^2]. ASSUMPTION.
        accel_v_mps2: vertical acceleration limit [m/s^2]. ASSUMPTION.
        max_ground_speed_mps: ground-speed cap [m/s]. ASSUMPTION.
        min_ground_speed_mps: below this ground speed the mission is declared infeasible [m/s].
        launch_latency_s: alert-to-launch latency [s]. ASSUMPTION (claesson2017 reports a 3 s
            median for a research drone without an air-traffic-control step).
        release_tol_m: position tolerance for release [m].
        release_speed_tol_mps: speed tolerance for release [m/s].
        settle_timeout_s: time allowed after the plan ends to satisfy the release tolerances [s].
        time_offset_s: fitted extra flight time beyond the reference plan [s]; the calibration
            sweep gave 0.0228 s (control-tick quantisation plus tracking lag).
        energy_scale: fitted multiplicative correction of the analytic energy [-]; the sweep
            gave 1.00002, so no correction is applied (1.0).
    """

    v_air_mps: float = 15.0
    cruise_alt_m: float = 50.0
    release_height_m: float = 2.0
    climb_speed_mps: float = 4.0
    descent_speed_mps: float = 3.0
    accel_h_mps2: float = 2.0
    accel_v_mps2: float = 2.0
    max_ground_speed_mps: float = 25.0
    min_ground_speed_mps: float = 2.0
    launch_latency_s: float = 60.0
    release_tol_m: float = 0.5
    release_speed_tol_mps: float = 0.5
    settle_timeout_s: float = 30.0
    time_offset_s: float = 0.023
    energy_scale: float = 1.0

    def with_(self, **kw: float) -> MissionParams:
        return replace(self, **kw)


@dataclass(frozen=True)
class MissionResult:
    """Outcome of one simulated mission.

    Time fields are infinite when the mission did not complete.

    Attributes:
        completed: True if the vehicle reached the release point within tolerance and the
            battery budget was respected.
        flight_time_s: liftoff to release [s].
        launch_latency_s: alert-to-launch latency [s] (a parameter, not simulated).
        total_time_s: launch latency + flight time [s].
        energy_wh: electrical energy used by the flight [Wh].
        peak_tracking_error_m: maximum 3-D distance between the vehicle and the reference [m].
        max_tilt_deg: maximum body tilt [deg].
        reason: empty on success, otherwise the failure cause.
    """

    completed: bool
    flight_time_s: float
    launch_latency_s: float
    total_time_s: float
    energy_wh: float
    peak_tracking_error_m: float
    max_tilt_deg: float
    reason: str = ""


class Trapezoid:
    """1-D rest-to-rest trapezoidal (or triangular) velocity profile.

    Args:
        dist: distance to travel [m] (>= 0).
        vmax: speed limit [m/s].
        amax: acceleration limit [m/s^2].
    """

    def __init__(self, dist: float, vmax: float, amax: float) -> None:
        self.dist = max(float(dist), 0.0)
        self.amax = float(amax)
        if self.dist * self.amax < 1e-12:
            self.vpeak, self.t_acc, self.t_cruise = 0.0, 0.0, 0.0
        elif self.dist >= vmax**2 / amax:
            self.vpeak = float(vmax)
            self.t_acc = vmax / amax
            self.t_cruise = (self.dist - vmax**2 / amax) / vmax
        else:
            self.vpeak = math.sqrt(self.dist * amax)
            self.t_acc = self.vpeak / amax
            self.t_cruise = 0.0
        self.duration = 2.0 * self.t_acc + self.t_cruise

    def at(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Distance, speed and acceleration [m, m/s, m/s^2] at times ``t`` [s]."""
        t = np.asarray(t, dtype=float)
        tc = np.clip(t, 0.0, self.duration)
        ta, tcr, a, vp = self.t_acc, self.t_cruise, self.amax, self.vpeak
        s = np.where(
            tc < ta,
            0.5 * a * tc**2,
            np.where(
                tc < ta + tcr,
                0.5 * a * ta**2 + vp * (tc - ta),
                self.dist - 0.5 * a * (self.duration - tc) ** 2,
            ),
        )
        v = np.where(tc < ta, a * tc, np.where(tc < ta + tcr, vp, a * (self.duration - tc)))
        acc = np.where(tc < ta, a, np.where(tc < ta + tcr, 0.0, -a))
        acc = np.where((t < 0.0) | (t > self.duration) | (self.duration == 0.0), 0.0, acc)
        return s, v, acc


class TrajectoryPlan:
    """Climb -> cruise -> descend reference along +x with start at (0, 0, z_start)."""

    def __init__(self, distance_m: float, ground_speed_mps: float, mp: MissionParams, z0: float):
        self.z0 = z0
        self.distance = distance_m
        self.z_cruise = z0 + mp.cruise_alt_m
        self.z_end = mp.release_height_m
        self.climb = Trapezoid(mp.cruise_alt_m, mp.climb_speed_mps, mp.accel_v_mps2)
        self.cruise = Trapezoid(distance_m, ground_speed_mps, mp.accel_h_mps2)
        self.descent = Trapezoid(self.z_cruise - self.z_end, mp.descent_speed_mps, mp.accel_v_mps2)
        self.t1 = self.climb.duration
        self.t2 = self.t1 + self.cruise.duration
        self.duration = self.t2 + self.descent.duration

    def sample(self, t: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Reference position, velocity, acceleration, each of shape (n, 3)."""
        t = np.atleast_1d(np.asarray(t, dtype=float))
        n = t.size
        pos, vel, acc = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros((n, 3))
        s, v, a = self.climb.at(t)
        pos[:, 2], vel[:, 2], acc[:, 2] = self.z0 + s, v, a
        s, v, a = self.cruise.at(t - self.t1)
        pos[:, 0], vel[:, 0], acc[:, 0] = s, v, a
        pos[:, 2] = np.where(t >= self.t1, self.z_cruise, pos[:, 2])
        vel[:, 2] = np.where(t >= self.t1, 0.0, vel[:, 2])
        acc[:, 2] = np.where(t >= self.t1, 0.0, acc[:, 2])
        s, v, a = self.descent.at(t - self.t2)
        late = t >= self.t2
        pos[:, 2] = np.where(late, self.z_cruise - s, pos[:, 2])
        vel[:, 2] = np.where(late, -v, vel[:, 2])
        acc[:, 2] = np.where(late, -a, acc[:, 2])
        return pos, vel, acc


def cruise_ground_speed_mps(wind_along_track_mps: float, mp: MissionParams) -> float:
    """Commanded ground speed [m/s] for a fixed airspeed plan (positive wind = tailwind)."""
    return min(mp.v_air_mps + wind_along_track_mps, mp.max_ground_speed_mps)


def peak_tilt_deg_bound(
    wind_along_track_mps: float, wind_cross_mps: float, quad: QuadParams, mp: MissionParams
) -> float:
    """Conservative bound on the tilt [deg] needed during the horizontal acceleration leg.

    Horizontal force = m a_h + full cruise drag (drag evaluated at the largest relative speed
    seen on the leg). The controller saturates its tilt command, so a mission whose bound
    exceeds the controller tilt limit is outside the validated envelope.
    """
    vg = cruise_ground_speed_mps(wind_along_track_mps, mp)
    v_rel_x = max(abs(vg - wind_along_track_mps), abs(wind_along_track_mps))
    d = 0.5 * quad.air_density * quad.cd_area_m2
    fx = quad.total_mass_kg * mp.accel_h_mps2 + d * v_rel_x * math.hypot(v_rel_x, wind_cross_mps)
    fy = d * abs(wind_cross_mps) * math.hypot(v_rel_x, wind_cross_mps)
    return math.degrees(math.atan2(math.hypot(fx, fy), quad.weight_n))


def is_feasible(
    wind_along_track_mps: float,
    wind_cross_mps: float = 0.0,
    quad: QuadParams | None = None,
    mp: MissionParams | None = None,
    gains: ControllerGains | None = None,
) -> bool:
    """True if the mission lies inside the controller's tilt and thrust envelope."""
    quad = quad or QuadParams()
    mp = mp or MissionParams()
    gains = gains or ControllerGains()
    if cruise_ground_speed_mps(wind_along_track_mps, mp) < mp.min_ground_speed_mps:
        return False
    tilt = peak_tilt_deg_bound(wind_along_track_mps, wind_cross_mps, quad, mp)
    if tilt > gains.max_tilt_deg:
        return False
    thrust = quad.weight_n / math.cos(math.radians(tilt))
    return thrust <= 0.95 * 4.0 * quad.max_thrust_per_rotor_n


def max_hold_wind_mps(quad: QuadParams | None = None, max_tilt_deg: float = 35.0) -> float:
    """Wind speed [m/s] at which a hovering quad reaches its tilt or thrust limit.

    A physical UPPER BOUND on operable wind; the operational weather limit used for
    availability is far lower (an ASSUMPTION driven by gust tolerance and safety margin).
    """
    quad = quad or QuadParams()
    f_tilt = quad.weight_n * math.tan(math.radians(max_tilt_deg))
    f_thrust = math.sqrt(max((0.95 * 4.0 * quad.max_thrust_per_rotor_n) ** 2 - quad.weight_n**2, 0))
    f = min(f_tilt, f_thrust)
    return math.sqrt(2.0 * f / (quad.air_density * quad.cd_area_m2))


# ---------------------------------------------------------------------------- analytic model
def mission_time_s(
    distance_m: float,
    wind_along_track_mps: float = 0.0,
    quad: QuadParams | None = None,
    mp: MissionParams | None = None,
    wind_cross_mps: float = 0.0,
) -> float:
    """Analytic FLIGHT time, liftoff to release [s]; ``math.inf`` if infeasible.

    Sum of the closed-form durations of the climb, cruise and descent trapezoids plus the
    calibrated ``time_offset_s`` (tracking lag before the release tolerances are met).

    Args:
        distance_m: horizontal distance [m].
        wind_along_track_mps: wind along the track [m/s]; positive = tailwind.
        quad: vehicle parameters.
        mp: mission parameters (carries the fitted offset).
        wind_cross_mps: crosswind [m/s] (affects feasibility and energy, not time).
    """
    quad = quad or QuadParams()
    mp = mp or MissionParams()
    if not is_feasible(wind_along_track_mps, wind_cross_mps, quad, mp):
        return math.inf
    plan = TrajectoryPlan(distance_m, cruise_ground_speed_mps(wind_along_track_mps, mp), mp, 0.0)
    return plan.duration + mp.time_offset_s


def time_to_scene_s(
    distance_m: float,
    wind_along_track_mps: float = 0.0,
    quad: QuadParams | None = None,
    mp: MissionParams | None = None,
    wind_cross_mps: float = 0.0,
) -> float:
    """Drone time-to-scene [s]: launch latency + flight time (``math.inf`` if infeasible)."""
    mp = mp or MissionParams()
    return mp.launch_latency_s + mission_time_s(
        distance_m, wind_along_track_mps, quad, mp, wind_cross_mps
    )


def energy_wh(
    distance_m: float,
    wind_along_track_mps: float = 0.0,
    quad: QuadParams | None = None,
    mp: MissionParams | None = None,
    ep: EnergyParams | None = None,
    wind_cross_mps: float = 0.0,
    dt: float = 0.1,
) -> float:
    """Analytic flight energy [Wh]; ``math.inf`` if infeasible.

    Integrates the momentum-theory rotor power along the reference plan, with the thrust
    vector ``m (a + g e_z) - F_drag(v - v_wind)`` shared equally over the rotors (mean wind
    only, no gust), plus hover power over the calibrated offset time. Scaled by the fitted
    ``energy_scale``.
    """
    quad = quad or QuadParams()
    mp = mp or MissionParams()
    ep = ep or EnergyParams.from_quad(quad)
    if not is_feasible(wind_along_track_mps, wind_cross_mps, quad, mp):
        return math.inf
    plan = TrajectoryPlan(distance_m, cruise_ground_speed_mps(wind_along_track_mps, mp), mp, 0.0)
    n = max(int(math.ceil(plan.duration / dt)), 1)
    t = (np.arange(n) + 0.5) * (plan.duration / n)
    _, vel, acc = plan.sample(t)
    wind = np.array([wind_along_track_mps, wind_cross_mps, 0.0])
    v_rel = vel - wind
    drag = (
        -0.5 * quad.air_density * quad.cd_area_m2 * np.linalg.norm(v_rel, axis=1)[:, None] * v_rel
    )
    f = quad.total_mass_kg * (acc + np.array([0.0, 0.0, G])) - drag
    powers = np.array([thrust_power_w(x, ep) for x in np.linalg.norm(f, axis=1)])
    joule = float(np.sum(powers) * (plan.duration / n))
    joule += hover_power_w(quad.total_mass_kg, ep) * mp.time_offset_s
    return mp.energy_scale * joule / 3600.0


def can_reach(
    distance_m: float,
    wind_along_track_mps: float = 0.0,
    quad: QuadParams | None = None,
    mp: MissionParams | None = None,
    ep: EnergyParams | None = None,
    wind_cross_mps: float = 0.0,
) -> bool:
    """True if the mission is feasible and fits in the usable battery energy."""
    quad = quad or QuadParams()
    ep = ep or EnergyParams.from_quad(quad)
    e = energy_wh(distance_m, wind_along_track_mps, quad, mp, ep, wind_cross_mps)
    return math.isfinite(e) and e <= ep.usable_wh


def p_available(
    wind_speed_samples: np.ndarray,
    wind_limit_mps: float,
    rain_prob: float,
    p_night_grounded: float = 0.0,
) -> float:
    """Probability that the drone is allowed to fly at a random alert time [-].

    ``P = P(wind <= limit) * (1 - rain_prob) * (1 - p_night_grounded)`` with the wind term
    the empirical CDF of the supplied samples, and the three grounding causes treated as
    independent (ASSUMPTION; positive correlation between wind and rain would make this
    conservative, i.e. a bias against the drone).

    Args:
        wind_speed_samples: wind speed samples at the operating height [m/s], any shape.
        wind_limit_mps: operational wind limit [m/s]. ASSUMPTION parameter (documented
            sensitivity range 6-15 m/s); ``schierbeck2023`` lists unfavourable weather as an
            exclusion without a threshold in its abstract.
        rain_prob: probability that precipitation grounds the drone [-] (0..1).
        p_night_grounded: probability the alert falls in a grounded (dark) period [-] (0..1).
            Default 0: the drone is assumed able to fly at night. ``schierbeck2023`` excluded
            darkness, so this should be swept.

    Returns:
        Availability probability in [0, 1]; non-decreasing in ``wind_limit_mps``.
    """
    w = np.asarray(wind_speed_samples, dtype=float).ravel()
    w = w[np.isfinite(w)]
    if w.size == 0:
        raise ValueError("wind_speed_samples has no finite values")
    if not (0.0 <= rain_prob <= 1.0 and 0.0 <= p_night_grounded <= 1.0):
        raise ValueError("probabilities must lie in [0, 1]")
    p_wind = float(np.mean(w <= wind_limit_mps))
    return p_wind * (1.0 - rain_prob) * (1.0 - p_night_grounded)


# ---------------------------------------------------------------------------- full simulation
def simulate_mission(
    distance_m: float,
    wind_mean_mps: float = 0.0,
    gust_sigma_mps: float = 0.0,
    seed: int = 0,
    quad: QuadParams | None = None,
    gains: ControllerGains | None = None,
    mp: MissionParams | None = None,
    ep: EnergyParams | None = None,
    wind_cross_mps: float = 0.0,
    gust_tau_s: float = 5.0,
) -> MissionResult:
    """Run the full MuJoCo mission: launch, climb, cruise, descend, release.

    Args:
        distance_m: horizontal distance to the release point [m].
        wind_mean_mps: mean wind along the track [m/s]; positive = tailwind.
        gust_sigma_mps: OU gust standard deviation per horizontal axis [m/s].
        seed: RNG seed of the gust process (deterministic result for a given seed).
        quad: vehicle parameters.
        gains: controller gains.
        mp: mission parameters.
        ep: energy parameters.
        wind_cross_mps: mean crosswind [m/s].
        gust_tau_s: gust correlation time [s].

    Returns:
        A :class:`MissionResult`; time fields are ``math.inf`` if the mission did not complete.
    """
    quad = quad or QuadParams()
    mp = mp or MissionParams()
    ep = ep or EnergyParams.from_quad(quad)
    gains = gains or ControllerGains()
    inf = math.inf
    if not is_feasible(wind_mean_mps, wind_cross_mps, quad, mp, gains):
        return MissionResult(False, inf, mp.launch_latency_s, inf, 0.0, 0.0, 0.0, "infeasible")

    sim = QuadSim(quad)
    dt_c = CONTROL_DECIMATION * sim.dt
    ctrl = CascadedController(quad, gains, dt_c)
    wind = WindField(
        WindParams((wind_mean_mps, wind_cross_mps, 0.0), gust_sigma_mps, gust_tau_s), dt_c, seed
    )
    meter = EnergyMeter(ep)
    z0 = quad.rest_height_m + 1e-3
    plan = TrajectoryPlan(distance_m, cruise_ground_speed_mps(wind_mean_mps, mp), mp, z0)
    n_steps = int(math.ceil((plan.duration + mp.settle_timeout_s) / dt_c))
    t_grid = np.arange(n_steps + 1) * dt_c
    p_ref, v_ref, a_ref = plan.sample(t_grid)
    goal = np.array([distance_m, 0.0, mp.release_height_m])

    sim.reset(position=(0.0, 0.0, z0))
    peak_err = 0.0
    max_tilt = 0.0
    completed = False
    reason = "settle_timeout"
    t_done = inf
    for k in range(n_steps):
        s = sim.state()
        err = float(np.linalg.norm(s.pos - p_ref[k]))
        peak_err = max(peak_err, err)
        max_tilt = max(max_tilt, s.tilt_rad)
        t = k * dt_c
        if not (np.all(np.isfinite(s.pos)) and np.all(np.isfinite(s.vel))):
            reason = "nan"
            break
        if t > 3.0 and s.pos[2] < 0.5 * quad.rest_height_m:
            reason = "crash"
            break
        if err > 25.0 or s.tilt_rad > math.radians(80.0):
            reason = "lost_tracking"
            break
        if (
            t >= plan.duration
            and np.linalg.norm(s.pos - goal) < mp.release_tol_m
            and np.linalg.norm(s.vel) < mp.release_speed_tol_mps
        ):
            completed, reason, t_done = True, "", t
            break
        w = wind.step()
        force = drag_force(s.vel, w, quad.cd_body, quad.frontal_area_m2, quad.air_density)
        thrusts = ctrl.control(s, Reference(p_ref[k], v_ref[k], a_ref[k]))
        sim.step(thrusts, nstep=CONTROL_DECIMATION, ext_force=force)
        meter.update(sim.rotor_thrusts(), dt_c)

    if completed and meter.energy_wh > ep.usable_wh:
        completed, reason = False, "battery"
    flight = t_done if completed else inf
    return MissionResult(
        completed=completed,
        flight_time_s=flight,
        launch_latency_s=mp.launch_latency_s,
        total_time_s=mp.launch_latency_s + flight,
        energy_wh=meter.energy_wh,
        peak_tracking_error_m=peak_err,
        max_tilt_deg=math.degrees(max_tilt),
        reason=reason,
    )
