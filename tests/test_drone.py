"""Fast deterministic tests for the AED quadrotor comparator (Track D)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aedrover.drone.energy import (
    EnergyMeter,
    EnergyParams,
    hover_power_w,
    max_range_m,
    rotor_power_w,
    thrust_power_w,
)
from aedrover.drone.flight_ctrl import (
    CascadedController,
    ControllerGains,
    Reference,
    mixer_matrix,
)
from aedrover.drone.mission import (
    CONTROL_DECIMATION,
    MissionParams,
    Trapezoid,
    energy_wh,
    is_feasible,
    max_hold_wind_mps,
    mission_time_s,
    p_available,
    simulate_mission,
    time_to_scene_s,
)
from aedrover.drone.quadrotor_mjcf import G, QuadParams, QuadSim, euler_to_quat
from aedrover.drone.wind import WindField, WindParams, drag_force

QUAD = QuadParams()
DT_CTRL = CONTROL_DECIMATION * QUAD.timestep_s


def _hover_run(
    sim: QuadSim, ctrl: CascadedController, target: np.ndarray, seconds: float
) -> np.ndarray:
    """Run the controller against a fixed setpoint; return the distance-to-target history."""
    ref = Reference(pos=target)
    err = []
    for _ in range(int(seconds / DT_CTRL)):
        s = sim.state()
        sim.step(ctrl.control(s, ref), nstep=CONTROL_DECIMATION)
        err.append(np.linalg.norm(s.pos - target))
    return np.array(err)


# ------------------------------------------------------------------------------ vehicle
def test_model_builds_and_static_thrust_balances_weight():
    sim = QuadSim(QUAD)
    assert sim.model.nu == 4
    assert abs(sim.model.body_mass[1] - QUAD.total_mass_kg) < 1e-9
    hover = QUAD.hover_thrust_per_rotor_n
    sim.reset(position=(0.0, 0.0, 5.0), spooled=True)
    for _ in range(100):
        sim.step(np.full(4, hover))
    assert abs(sim.state().vel[2]) < 1e-6  # rotors at hover thrust cancel gravity exactly
    assert QUAD.thrust_to_weight > 1.5


def test_mixer_is_invertible_and_torque_signs_match_physics():
    m = mixer_matrix(QUAD)
    assert abs(np.linalg.det(m)) > 1e-6
    sim = QuadSim(QUAD)
    hover = QUAD.hover_thrust_per_rotor_n
    sim.reset(position=(0.0, 0.0, 5.0), spooled=True)
    f = np.full(4, hover)
    f[0] += 3.0  # front-left rotor: rolls positive about x, pitches negative about y
    for _ in range(20):
        sim.step(f)
    w = sim.state().omega
    expected = m @ (f - hover)
    assert np.sign(w[0]) == np.sign(expected[1]) and np.sign(w[1]) == np.sign(expected[2])
    assert np.sign(w[2]) == np.sign(expected[3])


# ------------------------------------------------------------------------------ hover
def test_hover_holds_position_in_still_air_for_20_s():
    sim = QuadSim(QUAD)
    ctrl = CascadedController(QUAD, dt=DT_CTRL)
    target = np.array([0.0, 0.0, 5.0])
    # start with rotors off, a small position offset and a tilt: the controller must recover
    sim.reset(
        position=(0.03, -0.02, 5.05),
        quat=euler_to_quat(math.radians(1.0), math.radians(-1.0), 0.1),
    )
    err = _hover_run(sim, ctrl, target, 20.0)
    assert err.max() < 0.10
    assert err[-int(5 / DT_CTRL) :].max() < 0.01  # converged to the setpoint


def test_hover_recovers_from_unmodelled_mass():
    """Controller believes nominal mass; the vehicle is 10 percent heavier (integral action)."""
    heavy = QUAD.with_(payload_mass_kg=QUAD.payload_mass_kg + 0.1 * QUAD.total_mass_kg)
    sim = QuadSim(heavy)
    ctrl = CascadedController(QUAD, dt=DT_CTRL)
    sim.reset(position=(0.0, 0.0, 5.0))
    err = _hover_run(sim, ctrl, np.array([0.0, 0.0, 5.0]), 20.0)
    assert err.max() < 0.25
    assert err[-int(5 / DT_CTRL) :].max() < 0.01


# ------------------------------------------------------------------------------ analytic pieces
def test_trapezoid_profile_is_consistent():
    for dist, vmax in ((100.0, 5.0), (3.0, 5.0), (0.0, 5.0)):
        tp = Trapezoid(dist, vmax, 2.0)
        t = np.linspace(0.0, tp.duration + 1.0, 4001)
        s, v, a = tp.at(t)
        assert abs(s[-1] - dist) < 1e-9
        assert v.max() <= vmax + 1e-9
        assert np.all(np.diff(s) >= -1e-12)
        if dist > 0:  # integral of speed equals distance, integral of accel equals zero
            assert abs(np.trapezoid(v, t) - dist) < 0.02 * dist
            assert abs(np.trapezoid(a, t)) < 0.02


def test_energy_model_closed_forms():
    ep = EnergyParams.from_quad(QUAD)
    m = QUAD.total_mass_kg
    per = m * G / 4.0
    manual = (
        4.0 * per**1.5 / math.sqrt(2.0 * ep.air_density * ep.disc_area_m2) / (ep.figure_of_merit)
        + ep.avionics_w
    )
    assert abs(hover_power_w(m, ep) - manual) < 1e-9
    assert abs(thrust_power_w(m * G, ep) - manual) < 1e-9
    assert abs(ep.usable_wh - ep.capacity_wh * (1.0 - ep.reserve_fraction)) < 1e-12
    meter = EnergyMeter(ep)
    meter.update(np.full(4, per), 3600.0)
    assert abs(meter.energy_wh - manual) < 1e-6  # one hour at hover power
    # convexity: uneven thrust costs more than equal thrust for the same total
    uneven = np.array([per * 1.5, per * 0.5, per * 1.5, per * 0.5])
    assert EnergyMeter(ep).energy_wh == 0.0
    assert rotor_power_w(uneven, ep) > rotor_power_w(np.full(4, per), ep)


def test_max_range_monotone_in_wind():
    ep = EnergyParams.from_quad(QUAD)
    args = (QUAD.total_mass_kg, QUAD.cd_area_m2, ep)
    ranges = [max_range_m(15.0, w, *args) for w in (-10.0, -5.0, 0.0, 5.0)]
    assert all(b > a for a, b in zip(ranges, ranges[1:], strict=False))
    assert max_range_m(15.0, -15.0, *args) == 0.0  # headwind equals airspeed: no progress
    assert max_range_m(15.0, 0.0, *args, overhead_wh=1e9) == 0.0


# ------------------------------------------------------------------------------ wind
def test_wind_is_seeded_and_gust_has_requested_statistics():
    params = WindParams((2.0, 0.0, 0.0), gust_sigma_mps=1.5, gust_tau_s=5.0)
    dt, n = 1.0, 30000  # 6000 correlation times: sample statistics are tight
    a = WindField(params, dt, seed=11)
    b = WindField(params, dt, seed=11)
    c = WindField(params, dt, seed=12)
    xs = np.array([a.step() for _ in range(n)])
    ys = np.array([b.step() for _ in range(n)])
    zs = np.array([c.step() for _ in range(n)])
    assert np.array_equal(xs, ys) and not np.array_equal(xs, zs)
    assert abs(xs[:, 0].mean() - 2.0) < 0.15
    assert abs(xs[:, 0].std() - 1.5) < 0.15
    assert abs(xs[:, 2].std() - 0.75) < 0.10  # vertical scale 0.5


def test_drag_force_opposes_relative_velocity_and_is_quadratic():
    f1 = drag_force(np.array([10.0, 0.0, 0.0]), np.zeros(3), 1.5, 0.08, 1.225)
    f2 = drag_force(np.array([20.0, 0.0, 0.0]), np.zeros(3), 1.5, 0.08, 1.225)
    assert f1[0] < 0 and abs(f2[0] / f1[0] - 4.0) < 1e-9
    assert np.allclose(
        drag_force(np.array([5.0, 1.0, 0.0]), np.array([5.0, 1.0, 0.0]), 1.5, 0.08, 1.2), 0
    )


# ------------------------------------------------------------------------------ missions
@pytest.fixture(scope="module")
def calm_missions():
    return {d: simulate_mission(d, 0.0, 0.0, seed=0) for d in (500.0, 1500.0, 3000.0)}


def test_mission_tracks_and_completes_under_gusts():
    r = simulate_mission(500.0, wind_mean_mps=-3.0, gust_sigma_mps=1.5, seed=3)
    assert r.completed, r.reason
    assert r.peak_tracking_error_m < 1.0
    assert r.max_tilt_deg < 35.5  # inside the controller tilt limit
    assert math.isfinite(r.flight_time_s) and r.energy_wh > 0
    assert r.total_time_s == pytest.approx(r.launch_latency_s + r.flight_time_s)


def test_mission_is_deterministic_for_a_seed():
    a = simulate_mission(300.0, -2.0, 1.5, seed=5)
    b = simulate_mission(300.0, -2.0, 1.5, seed=5)
    c = simulate_mission(300.0, -2.0, 1.5, seed=6)
    assert a == b
    assert a.energy_wh != c.energy_wh


def test_analytic_mission_time_matches_simulation(calm_missions):
    for d, r in calm_missions.items():
        assert r.completed, (d, r.reason)
        t = mission_time_s(d, 0.0)
        assert abs(t - r.flight_time_s) / r.flight_time_s < 0.10, (d, t, r.flight_time_s)


def test_analytic_energy_matches_simulation_calm_and_gusty(calm_missions):
    for d, r in calm_missions.items():
        e = energy_wh(d, 0.0)
        assert abs(e - r.energy_wh) / r.energy_wh < 0.15, (d, e, r.energy_wh)
    gusty = simulate_mission(1500.0, wind_mean_mps=-4.0, gust_sigma_mps=1.5, seed=2)
    assert gusty.completed, gusty.reason
    e = energy_wh(1500.0, -4.0)
    assert abs(e - gusty.energy_wh) / gusty.energy_wh < 0.15, (e, gusty.energy_wh)


def test_wind_dependent_time_is_monotone():
    winds = [8.0, 4.0, 0.0, -3.0, -6.0, -9.0, -10.0]  # tailwind to strong headwind
    for d in (500.0, 3000.0):
        t = [mission_time_s(d, w) for w in winds]
        assert all(math.isfinite(x) for x in t)
        assert all(b > a for a, b in zip(t, t[1:], strict=False)), t
        e = [energy_wh(d, w) for w in winds]
        assert all(b > a for a, b in zip(e, e[1:], strict=False)), e
    # a headwind stronger than the airspeed cannot be flown: infinite time, not a wrong number
    assert math.isinf(mission_time_s(500.0, -15.0))
    assert not is_feasible(-15.0)
    # time also grows with distance
    assert mission_time_s(1000.0) < mission_time_s(2000.0)


def test_simulated_headwind_flight_is_slower_than_tailwind():
    head = simulate_mission(500.0, wind_mean_mps=-6.0, seed=0)
    tail = simulate_mission(500.0, wind_mean_mps=6.0, seed=0)
    assert head.completed and tail.completed
    assert head.flight_time_s > tail.flight_time_s


def test_time_to_scene_adds_launch_latency():
    mp = MissionParams(launch_latency_s=90.0)
    assert time_to_scene_s(1000.0, 0.0, mp=mp) == pytest.approx(
        90.0 + mission_time_s(1000.0, mp=mp)
    )
    assert math.isinf(time_to_scene_s(1000.0, -20.0, mp=mp))
    assert max_hold_wind_mps() > 15.0  # physical hold limit exceeds any operational limit used


# ------------------------------------------------------------------------------ soak
def test_randomised_soak_60_s_has_no_nan_and_stays_airborne():
    rng = np.random.default_rng(2024)
    sim = QuadSim(QUAD)
    ctrl = CascadedController(QUAD, gains=ControllerGains(), dt=DT_CTRL)
    roll, pitch = np.radians(rng.uniform(-20.0, 20.0, 2))
    sim.reset(
        position=(0.0, 0.0, 10.0),
        quat=euler_to_quat(roll, pitch, rng.uniform(-3.0, 3.0)),
        velocity=tuple(rng.uniform(-2.0, 2.0, 3)),
        omega=tuple(rng.uniform(-1.0, 1.0, 3)),
        spooled=True,
    )
    wind = WindField(WindParams((3.0, -2.0, 0.0), gust_sigma_mps=2.0), DT_CTRL, seed=5)
    target = np.array([0.0, 0.0, 10.0])
    min_z = np.inf
    for k in range(int(60.0 / DT_CTRL)):
        if k % 375 == 0:  # new random setpoint every 3 s
            target = np.array([rng.uniform(-15, 15), rng.uniform(-15, 15), rng.uniform(5, 15)])
        s = sim.state()
        force = drag_force(s.vel, wind.step(), QUAD.cd_body, QUAD.frontal_area_m2, QUAD.air_density)
        thrusts = ctrl.control(s, Reference(pos=target))
        assert np.all(np.isfinite(thrusts))
        sim.step(thrusts, nstep=CONTROL_DECIMATION, ext_force=force)
        assert np.all(np.isfinite(sim.data.qpos)) and np.all(np.isfinite(sim.data.qvel))
        min_z = min(min_z, float(s.pos[2]))
        assert s.tilt_rad < math.radians(80.0)
    assert min_z > 1.0


# ------------------------------------------------------------------------------ weather
def test_p_available_monotone_in_wind_limit_and_bounded():
    samples = np.random.default_rng(0).weibull(2.0, 5000) * 5.0
    limits = np.linspace(0.0, 25.0, 26)
    p = [p_available(samples, lim, 0.05) for lim in limits]
    assert all(b >= a - 1e-15 for a, b in zip(p, p[1:], strict=False))
    assert p[0] <= 0.05 and abs(p[-1] - 0.95) < 1e-12  # limit above all samples: 1 - rain
    assert p_available(samples, 10.0, 0.0) > p_available(samples, 10.0, 0.2)
    assert p_available(samples, 10.0, 0.0, p_night_grounded=0.5) == pytest.approx(
        0.5 * p_available(samples, 10.0, 0.0)
    )
    with pytest.raises(ValueError):
        p_available(samples, 10.0, 1.5)
    with pytest.raises(ValueError):
        p_available(np.array([np.nan]), 10.0, 0.0)
