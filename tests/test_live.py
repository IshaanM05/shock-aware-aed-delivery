"""The live (streaming) render path: causal pedestrian gait, live overlays, and a scene bound to a running environment."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

pytest.importorskip("cv2", reason="the renderer tests need opencv (pip install -e '.[viz]')")

from aedrover.sim import AEDRoverEnv, VehicleParams
from aedrover.viz.camera import chase
from aedrover.viz.live import (
    LiveOverlayAnimator,
    LiveScene,
    LiveState,
    StreamingPedAnimator,
    header_recording,
    live_dressing,
)
from aedrover.viz.look import load_look
from aedrover.viz.overlays import PARK, OverlayConfig
from aedrover.viz.people import N_PED_SLOTS, STRIDE_M, PedAnimator
from aedrover.viz.recording import record_episode
from aedrover.viz.render_model import RenderXml

VEH = VehicleParams.optimized()


@pytest.fixture(scope="module")
def env() -> AEDRoverEnv:
    e = AEDRoverEnv(obs_mode="dict", veh=VEH, rich=True)
    e.reset(seed=5021, options={"family": "crowded"})
    return e


def _live_model(env, cfg=None, rollout_nodes=None):
    cfg = cfg or OverlayConfig(ribbon=False, rollouts=bool(rollout_nodes))
    rec = header_recording(env, "optimized", rollout_nodes=rollout_nodes)
    rx = RenderXml(rec.xml, load_look(), (320, 180))
    rx.add_base()
    animators = live_dressing(rx, rec, load_look(), cfg)
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)
    for a in animators:
        a.bind(model, rec)
    return rec, model, mujoco.MjData(model), animators


def _ped(animators):
    return next(a for a in animators if isinstance(a, StreamingPedAnimator))


def _live(t=0.0, dt=0.02, lidar=None, shock=0.0, rollouts=None):
    return LiveState(t, dt, np.full(73, 10.0) if lidar is None else lidar, shock, rollouts)


# -------------------------------------------------------------------------------------- header
def test_header_recording_describes_the_environment_right_after_reset(env):
    rec = header_recording(env, "optimized")
    assert len(rec) == 1 and rec.xml == env.world.xml and rec.dt == env.dt
    assert rec.qpos.shape == (1, env.world.model.nq) and rec.mocap_pos.shape == (1, env.world.model.nmocap, 3)
    assert np.allclose(rec.lidar[0], env.obs["lidar"], atol=1e-5) and rec.meta["scenario"] == env.scenario.to_dict()
    assert len(rec.meta["furniture"]) == len(env.furniture) > 0
    assert header_recording(env, rollout_nodes=21).rollouts[0]["xy"].shape == (24, 21, 2)


def test_the_live_render_model_shares_the_physics_models_indices(env):
    rec, model, data, animators = _live_model(env)
    phys = env.world.model
    assert model.nq == phys.nq and model.nmocap >= phys.nmocap
    assert [type(a).__name__ for a in animators] == ["StreamingPedAnimator", "LiveOverlayAnimator"]
    d = env.world.data
    data.qpos[:] = d.qpos
    data.mocap_pos[: phys.nmocap] = d.mocap_pos
    data.mocap_quat[: phys.nmocap] = d.mocap_quat
    mujoco.mj_kinematics(model, data)
    for b in range(phys.nbody):
        nm = mujoco.mj_id2name(phys, mujoco.mjtObj.mjOBJ_BODY, b)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) if nm else b
        assert np.allclose(d.xpos[b], data.xpos[bid], atol=1e-9) or d.xpos[b][2] < -10, nm      # (parked slots are re-posed)


def test_a_live_scene_refuses_the_future_ribbon(env):
    with pytest.raises(ValueError, match="future"):
        LiveScene(env, overlays=OverlayConfig(ribbon=True))


# ------------------------------------------------------------------------------ streaming gait
def _walk(anim, data, slot, positions, dt=0.02):
    root = anim.root[slot]
    for p in positions:
        data.mocap_pos[root] = (p[0], p[1], 0.77)
        anim.apply_live(data, _live(dt=dt))


def test_streaming_pedestrians_head_where_they_walk_and_stand_still_when_still(env):
    _, model, data, animators = _live_model(env)
    anim = _ped(animators)
    for k in range(N_PED_SLOTS):
        data.mocap_pos[anim.root[k], 2] = -50.0
    t = np.arange(0, 3.0, 0.02)
    _walk(anim, data, 0, np.stack([1.3 * t, 0.3 * t], axis=1))                       # walking along (1.3, 0.3) m/s
    speed = float(np.linalg.norm(anim._vel[0]))
    assert speed == pytest.approx(math.hypot(1.3, 0.3), rel=0.05)
    assert anim._heading[0] == pytest.approx(math.atan2(0.3, 1.3), abs=0.03)
    walked = float(np.hypot(1.3 * t[-1], 0.3 * t[-1]))
    assert anim._phase[0] == pytest.approx(2 * math.pi * walked / STRIDE_M, rel=0.01)    # the stride phase is the distance walked
    h, ph = anim._heading[0], anim._phase[0]
    still = np.tile([1.3 * t[-1], 0.3 * t[-1]], (100, 1))
    _walk(anim, data, 0, still)
    assert float(np.linalg.norm(anim._vel[0])) < 0.05 and anim._phase[0] == ph and anim._heading[0] == pytest.approx(h, abs=1e-9)   # stands, facing the same way
    limbs = data.mocap_pos[anim.limb[0]]
    assert np.isfinite(limbs).all() and (limbs[:, 2] > -0.5).all()                       # posed on the ground, not parked
    assert (data.mocap_pos[anim.limb[1], 2] < -10).all()                              # an unused slot stays parked


def test_a_respawn_is_not_a_step(env):
    _, model, data, animators = _live_model(env)
    anim = _ped(animators)
    _walk(anim, data, 0, np.stack([np.linspace(0, 2.0, 80), np.zeros(80)], axis=1))
    ph = anim._phase[0]
    _walk(anim, data, 0, [(30.0, 1.0)])                                                  # teleported across the street
    assert anim._phase[0] == ph and float(np.linalg.norm(anim._vel[0])) == 0.0
    data.mocap_pos[anim.root[0], 2] = -50.0
    anim.apply_live(data, _live())
    assert anim._prev[0] is None and (data.mocap_pos[anim.limb[0], 2] < -10).all()


def test_streaming_gait_agrees_with_the_recorded_one_while_pedestrians_walk():
    rec = record_episode("dwa", "crowded", 5021)
    look = load_look()
    rx = RenderXml(rec.xml, look, (320, 180))
    rx.add_base()
    from aedrover.viz.dressing import dress_scene
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)
    recorded = next(a for a in dress_scene(rx, rec, look, overlays=None) if isinstance(a, PedAnimator))
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)
    recorded.bind(model, rec)
    streaming = StreamingPedAnimator()
    streaming.bind(model, rec)
    data = mujoco.MjData(model)
    diffs = []
    for i in range(len(rec)):
        data.mocap_pos[: rec.mocap_pos.shape[1]] = rec.mocap_pos[i]
        streaming.apply_live(data, _live(dt=rec.dt))
        for k in range(N_PED_SLOTS):
            if recorded.alive[i, k] and recorded.speed[i, k] > 0.6 and i > 60:
                d = (streaming._heading[k] - recorded.heading[i, k] + math.pi) % (2 * math.pi) - math.pi
                diffs.append(abs(d))
    assert len(diffs) > 200 and np.median(diffs) < 0.15                                  # same facing, causal smoothing aside


# ---------------------------------------------------------------------------------------- overlays
def _beads(model, data, animator, name):
    return data.mocap_pos[animator.ids[name]]


def test_live_overlays_draw_lidar_halo_and_rollouts_from_the_live_state(env):
    _, model, data, animators = _live_model(env, rollout_nodes=21)
    ov = next(a for a in animators if isinstance(a, LiveOverlayAnimator))
    lidar = np.full(73, 10.0)
    lidar[30:36] = 2.0                                                                    # six returns
    ov.apply_live(data, _live(lidar=lidar, shock=0.2))
    assert (_beads(model, data, ov, "lhit")[:, 2] > -10).sum() == 6 and (_beads(model, data, ov, "lring")[:, 2] > -10).sum() == 67
    assert (_beads(model, data, ov, "h0")[:, 2] > -10).all() and (_beads(model, data, ov, "h2")[:, 2] < -10).all()
    ov.apply_live(data, _live(lidar=lidar, shock=3.0))                                    # over budget: the halo turns red
    assert (_beads(model, data, ov, "h2")[:, 2] > -10).all() and (_beads(model, data, ov, "h0")[:, 2] < -10).all()
    assert (_beads(model, data, ov, "roll")[:, 2] < -10).all()                            # no planner snapshot: parked
    snap = {"xy": np.cumsum(np.full((24, 21, 2), 0.1, np.float32), axis=1), "cost": np.arange(24, dtype=np.float32),
            "best": np.cumsum(np.full((21, 2), 0.1, np.float32), axis=0)}
    ov.apply_live(data, _live(lidar=lidar, shock=0.2, rollouts=snap))
    assert (_beads(model, data, ov, "roll")[:, 2] > -10).any() and (_beads(model, data, ov, "best")[:, 2] > -10).any()
    assert np.isfinite(data.mocap_pos).all() and not np.allclose(PARK, 0)


# ------------------------------------------------------------------------------------------- GPU
@pytest.mark.gpu
def test_a_live_scene_follows_the_running_simulation_and_renders_it():
    from aedrover.viz.backend import filament_importable
    if not filament_importable():
        pytest.skip("MuJoCo has no Filament renderer")
    e = AEDRoverEnv(obs_mode="dict", veh=VEH, rich=True)
    e.reset(seed=5021, options={"family": "crowded"})
    try:
        scene = LiveScene(e, size=(320, 180))
    except Exception as exc:                                    # no GPU / no OpenGL
        pytest.skip(f"cannot build a GPU scene here: {exc}")
    try:
        frames = []
        for k in range(60):
            e.step(np.array([1.5, 0.0]))
            if k in (0, 59):
                scene.update(LiveState(e._t, 0.02, np.asarray(e.obs["lidar"]), 0.2))
                p, yaw = scene.rover_pose()
                assert np.allclose(p, e.world.data.qpos[:3], atol=1e-12) and np.allclose(scene.scene.data.qpos, e.world.data.qpos)
                frames.append(scene.render(chase(p, yaw)))
    finally:
        scene.close()
    assert frames[0].shape == (180, 320, 3) and 20 < frames[0].mean() < 235
    assert np.abs(frames[0].astype(int) - frames[1].astype(int)).mean() > 1.0                # the rover and its surroundings moved
