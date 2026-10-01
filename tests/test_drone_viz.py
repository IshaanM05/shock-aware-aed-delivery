"""The drone's recording, its visual parts and the dispatch timeline: data and math tests that run anywhere,
plus a GPU smoke test that skips without a GPU."""

from __future__ import annotations

import hashlib

import mujoco
import numpy as np
import pytest

pytest.importorskip("cv2", reason="the renderer tests need opencv (pip install -e '.[viz]')")

from aedrover.drone.mission import MissionParams, mission_time_s
from aedrover.drone.quadrotor_mjcf import quat_to_rot
from aedrover.viz.dressing import dress_scene
from aedrover.viz.drone_recording import DroneRecording, record_drone_mission
from aedrover.viz.drone_visuals import (
    BODY,
    LEDS,
    MOUNT_Z,
    ROTOR,
    DroneAnimator,
    DroneSpec,
    Placement,
)
from aedrover.viz.look import load_look
from aedrover.viz.overlays import OverlayConfig
from aedrover.viz.recording import Recording, record_episode
from aedrover.viz.render_model import RenderXml


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


# ---------------------------------------------------------------------------------- placement
def test_placement_puts_the_release_point_on_the_goal():
    pl = Placement.arriving_at((36.0, 0.5), 1000.0, yaw=0.4)
    assert np.allclose(pl.point(np.array([1000.0, 0.0, 2.0])), [36.0, 0.5, 2.0])
    assert np.allclose(Placement().point(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, 3.0])
    assert np.isclose(np.linalg.norm(pl.quat), 1.0)


# ------------------------------------------------------------------------- visuals and animator
@pytest.fixture(scope="module")
def rover_rec() -> Recording:
    return record_episode("dwa", "mixed", 5010)


def _dressed(rover_rec, spec, **kw):
    look = load_look()
    rx = RenderXml(rover_rec.xml, look, (320, 180))
    rx.add_base()
    animators = dress_scene(rx, rover_rec, look, overlays=None, drone=spec, **kw)
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)
    for a in animators:
        a.bind(model, rover_rec)
    return rx, model, mujoco.MjData(model), animators


def _bid(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def _quat_close(a, b):
    return np.allclose(a, b, atol=1e-9) or np.allclose(a, -b, atol=1e-9)


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def test_animator_follows_the_recorded_pose_exactly(rover_rec):
    rec = _synthetic()
    pl = Placement((12.0, -1.5, 0.0), yaw=0.35)
    rx, model, data, animators = _dressed(rover_rec, DroneSpec(rec, pl))
    anim = animators[-1]
    for t in (0.0, 0.3, 2.2, 4.9, 7.0):                         # includes a time after the last sample (hover)
        anim.t = t
        anim.apply(data, 0.0)
        mujoco.mj_kinematics(model, data)
        p_rec, q_rec = rec.pose_at(t)
        want_p = pl.point(p_rec)
        want_q = _qmul(np.array([np.cos(0.175), 0.0, 0.0, np.sin(0.175)]), q_rec)
        b = _bid(model, BODY)
        assert np.allclose(data.xpos[b], want_p, atol=1e-9), t
        assert _quat_close(data.xquat[b], want_q), t
        rot = quat_to_rot(want_q)
        for k in range(4):                                       # each rotor hub sits at its arm tip, on the airframe
            hub = want_p + rot @ np.array([*anim.spec.quad.rotor_xy()[k], MOUNT_Z])
            assert np.allclose(data.xpos[_bid(model, ROTOR.format(k=k))], hub, atol=1e-9), (t, k)


def test_status_light_switches_at_release_and_everything_parks_when_inactive(rover_rec):
    rec = _synthetic()
    rx, model, data, animators = _dressed(rover_rec, DroneSpec(rec, Placement()))
    anim = animators[-1]
    amber, green = (_bid(model, n) for n in LEDS)
    anim.t = rec.duration - 1.0
    anim.apply(data, 0.0)
    mujoco.mj_kinematics(model, data)
    assert data.xpos[amber][2] > 0 and data.xpos[green][2] < -10
    anim.t = rec.duration + 0.5
    anim.apply(data, 0.0)
    mujoco.mj_kinematics(model, data)
    assert data.xpos[green][2] > 0 and data.xpos[amber][2] < -10
    anim.active = False
    anim.apply(data, 0.0)
    mujoco.mj_kinematics(model, data)
    for name in (BODY, LEDS[0], LEDS[1], *(ROTOR.format(k=k) for k in range(4))):
        assert data.xpos[_bid(model, name)][2] < -10


def test_rotors_turn_with_the_recorded_thrust_and_neighbours_counter_rotate(rover_rec):
    rec = _synthetic()
    rec.thrust[:] = 0.0
    rec.thrust[3:] = 20.0                                         # motors idle for the first samples, then spin up
    rx, model, data, animators = _dressed(rover_rec, DroneSpec(rec, Placement()))
    anim = animators[-1]
    assert np.allclose(anim.rotor_angles(rec.t[2]), 0.0)
    a1, a2 = anim.rotor_angles(rec.t[6]), anim.rotor_angles(rec.t[9])
    assert np.all(np.abs(a2) > np.abs(a1)) and np.all(np.abs(a1) > 0)
    assert a1[0] * a1[1] < 0 and a1[0] * a1[2] > 0                # adjacent rotors turn opposite ways, diagonal ones alike
    after = np.abs(anim.rotor_angles(rec.t[-1] + 2.0)) - np.abs(anim.rotor_angles(rec.t[-1]))
    assert np.allclose(after, np.abs(anim._omega_end) * 2.0)      # keeps turning at its last speed after the last sample


def test_bind_names_the_missing_part_instead_of_posing_the_wrong_one(rover_rec):
    look = load_look()
    rx = RenderXml(rover_rec.xml, look, (320, 180))
    rx.add_base()
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)       # never dressed with the drone
    with pytest.raises(ValueError, match="dr_body"):
        DroneAnimator(DroneSpec(_synthetic(), Placement())).bind(model, rover_rec)


def test_drone_dressing_leaves_every_physics_body_where_it_was(rover_rec):
    rec = _synthetic()
    rx, model, data, animators = _dressed(rover_rec, DroneSpec(rec, Placement.arriving_at((36.0, 0.0), 20.0)))
    phys = mujoco.MjModel.from_xml_string(rover_rec.xml)
    assert model.nq == phys.nq and model.nmocap >= phys.nmocap
    pd = mujoco.MjData(phys)
    for i in (0, len(rover_rec) // 2, len(rover_rec) - 1):
        for d in (pd, data):
            d.qpos[:] = rover_rec.qpos[i]
            d.mocap_pos[: phys.nmocap] = rover_rec.mocap_pos[i]
            d.mocap_quat[: phys.nmocap] = rover_rec.mocap_quat[i]
        for a in animators:
            a.apply(data, float(i))
        mujoco.mj_kinematics(phys, pd)
        mujoco.mj_kinematics(model, data)
        for b in range(phys.nbody):
            nm = mujoco.mj_id2name(phys, mujoco.mjtObj.mjOBJ_BODY, b)
            assert np.allclose(pd.xpos[b], data.xpos[_bid(model, nm) if nm else b], atol=1e-9), nm


# ------------------------------------------------------------------ the default scenes are untouched
DRESSED_SHA256 = {                  # of RenderXml.build() for dwa, mixed, seed 5010, measured before the drone options existed
    "no overlays": "e9a3a8c6ccd98cd00bc7fd9302191006dc54c347824eeec9a627e9a5c88596d9",
    "default overlays": "a72cd8f29247518afaf5bab7e7e1f7e15da2f114ad4f502765b911e8ed5dee93",
}


@pytest.mark.parametrize("name", sorted(DRESSED_SHA256))
def test_default_dressing_is_byte_identical(rover_rec, name):
    look = load_look()
    rx = RenderXml(rover_rec.xml, look, (640, 360))
    rx.add_base(wet=False)
    dress_scene(rx, rover_rec, look, overlays=OverlayConfig() if name == "default overlays" else None)
    assert hashlib.sha256(rx.build().encode()).hexdigest() == DRESSED_SHA256[name]


def test_clearing_the_flight_corridor_only_removes_towers(rover_rec):
    look = load_look()

    def lines(**kw):
        rx = RenderXml(rover_rec.xml, look, (320, 180))
        rx.add_base()
        dress_scene(rx, rover_rec, look, overlays=None, **kw)
        return rx.build().splitlines()

    base = lines()
    cleared = lines(clear_flight_corridor=((-400.0, 0.0), (36.0, 0.0), 30.0))
    assert len(cleared) < len(base) and set(cleared) <= set(base)
    removed = [ln for ln in base if ln not in set(cleared)]
    assert removed and all("skyline_" in ln for ln in removed)
    ped = [ln for ln in base if "ped0_shirt" in ln or "ped3_skin" in ln]
    assert ped and all(ln in set(cleared) for ln in ped)          # pedestrians keep their colours: the random stream is unchanged
