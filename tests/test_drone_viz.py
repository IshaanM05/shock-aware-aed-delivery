"""The drone's recording, its visual parts and the dispatch timeline: data and math tests that run anywhere,
plus a GPU smoke test that skips without a GPU."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2", reason="the renderer tests need opencv (pip install -e '.[viz]')")

from aedrover.drone.mission import MissionParams, mission_time_s
from aedrover.viz.drone_recording import DroneRecording, record_drone_mission


def _synthetic(n: int = 11, dt: float = 0.5) -> DroneRecording:
    """A straight climb with a slow roll: easy to check by hand, no simulation needed."""
    t = np.arange(n) * dt
    pos = np.stack([t * 2.0, np.zeros(n), t * 1.0 + 0.05], axis=1)
    roll = 0.05 * t
    quat = np.stack([np.cos(roll / 2), np.sin(roll / 2), np.zeros(n), np.zeros(n)], axis=1)
    thrust = np.tile(np.array([20.0, 20.0, 20.0, 20.0]), (n, 1)) + t[:, None]
    return DroneRecording(dt=dt, t=t, pos=pos, quat=quat, thrust=thrust, meta={"distance_m": 20.0})


@pytest.fixture(scope="module")
def mission() -> DroneRecording:
    return record_drone_mission(300.0)


# ----------------------------------------------------------------------------------- recording
def test_pose_at_hits_samples_interpolates_and_clamps():
    r = _synthetic()
    for i in (0, 4, 10):
        p, q = r.pose_at(float(r.t[i]))
        assert np.allclose(p, r.pos[i]) and np.allclose(q, r.quat[i])
    p, q = r.pose_at(0.75)                                  # halfway between samples 1 and 2
    assert np.allclose(p, 0.5 * (r.pos[1] + r.pos[2]))
    assert np.isclose(np.linalg.norm(q), 1.0)
    assert np.allclose(r.pose_at(-3.0)[0], r.pos[0])        # on the ground before liftoff
    assert np.allclose(r.pose_at(1e6)[0], r.pos[-1])        # hovering at the release point afterwards
    assert np.allclose(r.thrust_at(0.75), 0.5 * (r.thrust[1] + r.thrust[2]))


def test_recording_round_trip(tmp_path):
    r = _synthetic()
    r2 = DroneRecording.load(r.save(tmp_path / "d.npz"))
    for name in ("t", "pos", "quat", "thrust"):
        assert np.array_equal(getattr(r, name), getattr(r2, name))
    assert r2.dt == r.dt and r2.meta == r.meta and len(r2) == len(r)


def test_recorded_mission_matches_the_analytic_model_and_the_plan(mission):
    m = mission.meta
    assert m["completed"] and m["distance_m"] == 300.0
    assert abs(m["flight_time_s"] - mission_time_s(300.0)) < 0.1
    assert mission.duration == pytest.approx(m["flight_time_s"], abs=mission.dt)
    assert m["total_time_s"] == pytest.approx(MissionParams().launch_latency_s + m["flight_time_s"])
    assert np.allclose(mission.pos[0, :2], 0.0) and np.allclose(mission.pos[-1, :2], [300.0, 0.0], atol=0.5)
    assert mission.pos[:, 2].max() > MissionParams().cruise_alt_m - 1.0
    assert (mission.thrust >= 0).all() and mission.thrust.max() < m["quad"]["max_thrust_per_rotor_n"] + 1e-6


def test_infeasible_mission_is_refused():
    with pytest.raises(ValueError, match="infeasible"):
        record_drone_mission(500.0, wind_mean_mps=-14.0)       # headwind within 1 m/s of the airspeed
