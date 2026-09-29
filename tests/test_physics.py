"""Physics regression tests: analytic checks and the MuJoCo pitfalls we hit while building this."""

import mujoco
import numpy as np
import pytest

from aedrover.sim import Rover, VehicleParams, World
from aedrover.sim.metrics import rover_shock
from aedrover.sim.validate import (
    analytic_quarter_car,
    ringdown,
    rolling_slip,
    stability_soak,
    static_equilibrium,
)

VEH = VehicleParams()


def test_vehicle_derived_quantities():
    assert VEH.total_mass == pytest.approx(35.0)
    assert VEH.sprung_per_wheel == pytest.approx(6.5)
    assert VEH.susp_zeta == pytest.approx(1.023, abs=1e-3)   # nominal design is slightly overdamped
    assert VEH.ground_clearance > 0.12                          # must clear a 12 cm kerb belly-wise


def test_static_equilibrium_matches_design():
    res = static_equilibrium(VEH)
    assert abs(res["ride_height_err_mm"]) < 1.0
    assert res["payload_acc_z"] == pytest.approx(9.81, abs=0.02)
    assert max(abs(x) for x in res["susp_deflection_mm"]) < 0.5


@pytest.mark.parametrize("zeta", [0.10, 0.20])
def test_ringdown_matches_quarter_car_analytics(zeta):
    res = ringdown(VEH, zeta)
    assert abs(res["omega_n_err_pct"]) < 10.0
    assert abs(res["zeta_err_pct"]) < 10.0


def test_analytic_quarter_car_limits():
    # stiff tyre -> body mode approaches the sprung-mass 1-DOF values
    a = analytic_quarter_car(6.5, 2.25, 4500.0, 100.0, 1e9)
    assert a["omega_n"] == pytest.approx(np.sqrt(4500 / 6.5), rel=1e-3)
    assert a["zeta"] == pytest.approx(100 / (2 * np.sqrt(4500 * 6.5)), rel=1e-3)


@pytest.mark.parametrize("v", [0.5, 2.0])
def test_rolls_without_slip_or_drift(v):
    res = rolling_slip(VEH, v, t_end=6.0)
    assert abs(res["speed_err_pct"]) < 3.0
    assert abs(res["lateral_drift_m"]) < 0.02


def test_kerb_is_actually_solid():
    """Regression: static geoms edited at runtime and mocap slots compiled with contype=0 are
    silently ignored by MuJoCo's collision pipeline (stale BVH / body-level bit masks)."""
    w = World(VEH)
    w.set_crossing(0.12, x_down=-9.0, x_up=4.0)
    r = Rover(w)
    r.reset(0.0, 0.0, 0.0, settle_s=0.5)
    for _ in range(int(5.0 / r.control_dt)):
        r.step(0.4, 0.0)      # too slow to climb: must be stopped by the kerb, not pass through it
        assert r.pos[2] < 0.35
    assert r.pos[0] < 4.0


def test_climbs_12cm_kerb_with_momentum_and_shock_grows_with_speed():
    peaks = {}
    for v in (1.0, 2.0):
        w = World(VEH)
        w.set_crossing(0.12, x_down=-9.0, x_up=6.0)
        r = Rover(w)
        r.reset(0.0, 0.0, 0.0, settle_s=1.0)
        for _ in range(int(10.0 / r.control_dt)):
            r.step(v, float(np.clip(-1.5 * r.yaw_pitch_roll()[0], -0.5, 0.5)))
            if r.pos[0] > 7.6:
                break
        assert r.pos[2] > 0.12 + 0.15, f"did not climb at v={v}"
        peaks[v] = rover_shock(r).peak_g
    assert peaks[2.0] > peaks[1.0] > 1.0


def test_ramp_orientation_is_walkable():
    """A dropped-kerb ramp must present a smooth surface: climbing it at low speed must work."""
    w = World(VEH)
    w.set_crossing(0.12, x_down=-9.0, x_up=6.0, ramp_up=True)
    r = Rover(w)
    r.reset(0.0, 0.0, 0.0, settle_s=1.0)
    for _ in range(int(20.0 / r.control_dt)):
        r.step(0.5, float(np.clip(-1.5 * r.yaw_pitch_roll()[0], -0.5, 0.5)))
        if r.pos[0] > 7.6:
            break
    assert r.pos[2] > 0.12 + 0.15


def test_mocap_state_survives_reset():
    w = World(VEH)
    w.set_crossing(0.12, x_down=-9.0, x_up=6.0)
    r = Rover(w)
    before = w.data.mocap_pos.copy()
    r.reset(0.0, 0.0, 0.0)
    r.reset(0.0, 0.0, 0.0)
    np.testing.assert_allclose(w.data.mocap_pos, before)


def test_payload_mass_change_updates_model():
    w = World(VEH)
    w.set_flat()
    w.set_payload_mass(6.0)
    assert w.model.body_mass[w.b_payload] == pytest.approx(6.0)
    r = Rover(w)
    r.reset(0.0, 0.0, 0.0, settle_s=2.0)
    assert np.isfinite(r.d.qpos).all()


@pytest.mark.slow
def test_no_nan_soak():
    res = stability_soak(VEH, n_steps=200_000)
    assert res["finite"]
    assert res["max_speed_mps"] < 10.0


def test_model_compiles_from_scratch():
    w = World(VEH)
    assert isinstance(w.model, mujoco.MjModel)
    assert w.model.nu == 6
