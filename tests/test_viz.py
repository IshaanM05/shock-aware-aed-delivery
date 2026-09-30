"""Cinematic renderer: math and data tests that run anywhere, plus GPU smoke tests that skip when no GPU."""

from __future__ import annotations

import hashlib
from pathlib import Path

import mujoco
import numpy as np
import pytest

pytest.importorskip("cv2", reason="the renderer tests need opencv (pip install -e '.[viz]')")

from aedrover.viz import assets as A
from aedrover.viz import hud, post
from aedrover.viz.camera import CameraPose, chase
from aedrover.viz.dressing import dress_scene
from aedrover.viz.look import Look, load_look
from aedrover.viz.overlays import OverlayConfig, _densify
from aedrover.viz.people import _quat_z_to
from aedrover.viz.recording import Recording, record_episode
from aedrover.viz.render_model import RenderXml, _slerp, free_joint_quat_slices, interpolate_state
from aedrover.viz.shots import rig_chase, smooth_poses, time_map

ROOT = Path(__file__).resolve().parents[1]
POSE = CameraPose((0.0, 0.0, 1.5), (10.0, 0.0, 0.3), 50.0)


@pytest.fixture(scope="module")
def rec() -> Recording:
    return record_episode("dwa", "mixed", 5010)


# ---------------------------------------------------------------------------------------- camera
def test_camera_basis_is_orthonormal_and_looks_at_target():
    f, up = POSE.forward, POSE.up
    assert np.isclose(np.linalg.norm(f), 1) and np.isclose(np.linalg.norm(up), 1)
    assert abs(float(f @ up)) < 1e-9 and up[2] > 0
    straight_down = CameraPose((0, 0, 5), (0, 0, 0))
    assert abs(float(straight_down.forward @ straight_down.up)) < 1e-9          # no gimbal flip looking straight down


def test_gl_camera_frustum_matches_field_of_view():
    gl = POSE.to_gl(near=0.1, far=100.0)
    assert np.isclose(gl.frustum_top / gl.frustum_near, np.tan(np.radians(25.0)))
    assert np.isclose(gl.frustum_top, -gl.frustum_bottom)


def test_pose_lerp_and_chase_geometry():
    a, b = CameraPose((0, 0, 1), (1, 0, 0), 40), CameraPose((2, 2, 3), (3, 3, 0), 60)
    mid = a.lerp(b, 0.5)
    assert np.allclose(mid.pos, (1, 1, 2)) and mid.fovy == 50
    c = chase((5.0, 1.0, 0.4), 0.0, back=4.0, height=1.5, swing_deg=0.0)
    assert np.allclose(c.pos, (1.0, 1.0, 1.9)) and c.target[0] > 5.0


# ----------------------------------------------------------------------------------- time mapping
def test_time_map_is_monotone_clamped_and_slows_at_the_event():
    s = time_map(300, 60, 100.0, 1000, 0.02, slow=(300.0, 0.4, 0.2))
    assert np.all(np.diff(s) >= 0) and s[0] == 100.0
    steps_near = np.diff(s)[np.abs(s[:-1] - 300.0) < 10]
    steps_far = np.diff(s)[np.abs(s[:-1] - 300.0) > 120]
    assert steps_near.mean() < 0.4 * steps_far.mean()
    assert time_map(50, 60, 990.0, 1000, 0.02).max() <= 999


def test_smoothing_keeps_length_and_reduces_jitter():
    rng = np.random.default_rng(0)
    poses = [CameraPose((i * 0.1 + rng.normal(0, 0.05), 0.0, 1.5), (i * 0.1 + 3.0, 0.0, 0.3), 50.0) for i in range(120)]
    sm = smooth_poses(poses, 6.0)
    assert len(sm) == len(poses)
    roughness = lambda ps: float(np.std(np.diff([p.pos[0] for p in ps], 2)))   # noqa: E731
    assert roughness(sm) < 0.3 * roughness(poses)


# ---------------------------------------------------------------------------------- interpolation
def test_slerp_returns_unit_quaternions_and_takes_the_short_way():
    q0 = np.array([1.0, 0, 0, 0])
    q1 = np.array([np.cos(0.4), 0, 0, np.sin(0.4)])
    mid = _slerp(q0, q1, 0.5)
    assert np.isclose(np.linalg.norm(mid), 1.0) and np.allclose(mid, [np.cos(0.2), 0, 0, np.sin(0.2)])
    assert np.allclose(_slerp(q0, -q1, 0.5), mid)                                # q and -q are the same rotation


def test_interpolation_hits_the_recorded_states(rec):
    m = mujoco.MjModel.from_xml_string(rec.xml)
    sl = free_joint_quat_slices(m)
    q, mp, mq = interpolate_state(rec, 10.0, sl)
    assert np.allclose(q, rec.qpos[10]) and np.allclose(mp, rec.mocap_pos[10])
    q, _, _ = interpolate_state(rec, 10.5, sl)
    assert np.allclose(q[:3], 0.5 * (rec.qpos[10, :3] + rec.qpos[11, :3]))
    assert np.isclose(np.linalg.norm(q[3:7]), 1.0)
    assert np.allclose(interpolate_state(rec, 1e9, sl)[0], rec.qpos[-1])          # clamps at the end


# ---------------------------------------------------------------------------------- recording
def test_recording_round_trip_and_follows_the_benchmark_protocol(rec, tmp_path):
    back = Recording.load(rec.save(tmp_path / "r.npz"))
    assert back.xml == rec.xml and np.array_equal(back.qpos, rec.qpos) and back.meta["seed"] == 5010
    from aedrover.analysis.experiments import Job, controller_spec, run_job
    from aedrover.sim.vehicle_mjcf import VehicleParams
    name, kw = controller_spec("dwa", 2.0)
    row = run_job(Job(name, "mixed", 5010, controller_kwargs=tuple(sorted(kw.items())),
                      env_kwargs=(("veh", VehicleParams.optimized()),)))
    assert rec.meta["outcome"] == row["outcome"]
    assert rec.meta["time_s"] == row["time_s"] and rec.meta["peak_shock_g"] == row["peak_shock_g"]


def test_recording_is_deterministic():
    a, b = record_episode("dwa", "kerb", 5001), record_episode("dwa", "kerb", 5001)
    assert np.array_equal(a.qpos, b.qpos) and np.array_equal(a.mocap_pos, b.mocap_pos)


# ------------------------------------------------------------------ render model == physics model
def test_render_model_reproduces_physics_poses(rec):
    """Dressing adds visual-only content after the physics elements: every original body and geom keeps its pose."""
    look = load_look()
    rx = RenderXml(rec.xml, look, (640, 360))
    rx.add_base()
    animators = dress_scene(rx, rec, look, overlays=OverlayConfig())
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)
    phys = mujoco.MjModel.from_xml_string(rec.xml)
    assert model.nq == phys.nq and model.nmocap >= phys.nmocap and model.ngeom > phys.ngeom
    names = [mujoco.mj_id2name(phys, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(phys.njnt)]
    assert names == [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(phys.njnt)]
    pd_, rd_ = mujoco.MjData(phys), mujoco.MjData(model)
    for a in animators:
        a.bind(model, rec)
    for i in (0, len(rec) // 3, len(rec) - 1):
        pd_.qpos[:] = rec.qpos[i]
        pd_.mocap_pos[:] = rec.mocap_pos[i]
        pd_.mocap_quat[:] = rec.mocap_quat[i]
        mujoco.mj_kinematics(phys, pd_)
        rd_.qpos[:] = rec.qpos[i]
        rd_.mocap_pos[:phys.nmocap] = rec.mocap_pos[i]
        rd_.mocap_quat[:phys.nmocap] = rec.mocap_quat[i]
        for a in animators:
            a.apply(rd_, float(i))
        mujoco.mj_kinematics(model, rd_)
        for b in range(phys.nbody):
            nm = mujoco.mj_id2name(phys, mujoco.mjtObj.mjOBJ_BODY, b)
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, nm) if nm else b
            assert np.allclose(pd_.xpos[b], rd_.xpos[bid], atol=1e-9), nm
        for k in ("chassis", "payload", "wheel_fl", "wheel_rr", "knuckle_fr"):
            bp, br = (mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, k) for m in (phys, model))
            assert np.allclose(pd_.xpos[bp], rd_.xpos[br], atol=1e-9) and np.allclose(pd_.xquat[bp], rd_.xquat[br], atol=1e-9)


def test_overlay_pools_are_parked_or_finite(rec):
    look = load_look()
    rx = RenderXml(rec.xml, look, (320, 180))
    rx.add_base()
    animators = dress_scene(rx, rec, look, overlays=OverlayConfig(rollouts=False))
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)
    d = mujoco.MjData(model)
    for a in animators:
        a.bind(model, rec)
    for i in (0, len(rec) // 2, len(rec) - 1):
        d.qpos[:] = rec.qpos[i]
        for a in animators:
            a.apply(d, float(i))
        assert np.isfinite(d.mocap_pos).all() and np.isfinite(d.mocap_quat).all()


# --------------------------------------------------------------------------------------- assets
def test_sky_has_six_deterministic_faces_with_the_sun_brightest():
    look = Look()
    faces = A.sky_faces(look, 64)
    assert sorted(faces) == sorted(f"{n}.png" for n in A.FACES)
    digest = hashlib.sha256(b"".join(faces[k] for k in sorted(faces))).hexdigest()
    assert digest == hashlib.sha256(b"".join(A.sky_faces.__wrapped__(look, 64)[k] for k in sorted(faces))).hexdigest()
    d = np.array([[look.sun_dir, [0, 0, 1.0], [-look.sun_dir[0], -look.sun_dir[1], 0.2]]], np.float32)
    d = d / np.linalg.norm(d, axis=-1, keepdims=True)
    lum = A.sky_linear(d, look).sum(-1)[0]
    assert lum[0] > lum[1] and lum[0] > lum[2]                                    # sun direction is the brightest of the three


def test_cube_face_directions_follow_the_measured_convention():
    n = 8
    for name, axis in (("right", (1, 0, 0)), ("left", (-1, 0, 0)), ("front", (0, -1, 0)), ("back", (0, 1, 0)),
                       ("up", (0, 0, 1)), ("down", (0, 0, -1))):
        mean = A._face_dirs(name, n).reshape(-1, 3).mean(0)
        assert np.dot(mean / np.linalg.norm(mean), axis) > 0.999, name


def test_procedural_textures_are_deterministic():
    a, b = A.asphalt_maps.__wrapped__(256, 3), A.asphalt_maps.__wrapped__(256, 3)
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)
    assert A.asphalt_maps.__wrapped__(256, 3, wet=True)["asphalt_orm.png"] != a["asphalt_orm.png"]


# ------------------------------------------------------------------------------ post-processing
def test_ground_distance_is_geometric():
    d = post.ground_distance(POSE, (64, 36), ground_z=0.0)
    assert d[0, 32] >= 1e4 - 1                                                    # sky: far
    col = d[20:, 32]
    assert np.all(np.diff(col) <= 1e-3)                                           # lower rows are nearer
    rd = post.ray_directions(POSE, (64, 36))
    assert np.isclose(d[35, 32], 1.5 / -rd[35, 32, 2], rtol=1e-5)                 # ray length to the plane z = 0


def test_depth_decoder_is_monotone_and_decodes_the_table():
    dist = np.geomspace(1, 200, 20)
    vals = 4.5 * 0.5 / dist
    dec = post.DepthDecoder(dist, vals)
    out = dec(vals)
    assert np.allclose(out[:-1], dist[:-1], rtol=1e-3)                            # the farthest calibrated value means "nothing drawn"
    assert dec(np.array([0.0]))[0] >= 1e3                                         # nothing drawn = infinitely far
    assert dec(np.array([vals.max() * 2]))[0] <= dist[0] + 1e-6                   # nearer than calibrated stays near, not far


def test_scene_distance_prefers_nearer_objects_only_when_clearly_nearer():
    ground = np.full((4, 4), 50.0, np.float32)
    buf = np.full((4, 4), 48.0, np.float32)
    assert np.array_equal(post.scene_distance(ground, buf), ground)               # within quantisation noise: keep the exact ground value
    buf[0, 0] = 5.0
    assert post.scene_distance(ground, buf)[0, 0] == 5.0


def test_finish_is_deterministic_uint8_and_seeded():
    rng = np.random.default_rng(1)
    frame = (rng.random((180, 320, 3)) * 255).astype(np.uint8)
    dl = post.low_distance(POSE, (320, 180))
    a = post.finish(frame, look=Look(), pose=POSE, dist_low=dl, seed=2)
    b = post.finish(frame, look=Look(), pose=POSE, dist_low=dl, seed=2)
    c = post.finish(frame, look=Look(), pose=POSE, dist_low=dl, seed=3)
    assert a.dtype == np.uint8 and a.shape == frame.shape
    assert np.array_equal(a, b) and not np.array_equal(a, c)                      # grain differs per seed, nothing else is random


def test_haze_pulls_far_pixels_toward_the_horizon_colour():
    frame = np.full((36, 64, 3), 20, np.uint8)
    near, far = np.full((9, 16), 2.0, np.float32), np.full((9, 16), 900.0, np.float32)
    g, look = post.Grade(), Look()
    assert post.haze(frame, far, look, POSE, g).mean() > post.haze(frame, near, look, POSE, g).mean() + 40


def test_depth_of_field_keeps_the_focal_plane_sharp():
    rng = np.random.default_rng(0)
    frame = (rng.random((90, 160, 3)) * 255).astype(np.uint8)
    at_focus = np.full((22, 40), 5.0, np.float32)
    assert np.abs(post.depth_of_field(frame, at_focus, 5.0).astype(int) - frame.astype(int)).mean() < 1.0
    far = np.full((22, 40), 60.0, np.float32)
    assert np.abs(post.depth_of_field(frame, far, 5.0).astype(int) - frame.astype(int)).mean() > 20.0


def test_grade_luts_are_monotone():
    lut, m = post._grade_tables(post.Grade())
    assert lut.shape == (1, 256, 3) and m.shape == (3, 3)
    assert np.all(np.diff(lut[0, :, 1].astype(int)) >= 0)


# ------------------------------------------------------------------------------------------ HUD
def test_hud_and_cards_render_at_any_size():
    frame = np.zeros((360, 640, 3), np.uint8)
    st = hud.HudState(t=1.0, speed=1.2, shock=3.4, peak=3.4, controller="X", caption="c", history=np.linspace(0, 3.4, 50))
    out = hud.draw_hud(frame, st)
    assert out.shape == frame.shape and out.sum() > 0
    card = hud.results_card((640, 360), "h", [("a", 0.9, 0.03, "n"), ("b", 0.5, 0.05, "m")], "f", progress=1.0, note="x")
    assert card.shape == (360, 640, 3) and hud.end_card((640, 360), "t", ["a"]).shape == (360, 640, 3)


# ---------------------------------------------------------------------------- gait and overlays
def test_quaternion_from_z_maps_the_z_axis_onto_the_direction():
    for v in ([1, 0, 0], [0, 1, 0], [0.3, -0.4, -0.8], [0, 0, -1], [0, 0, 1]):
        v = np.array(v, float) / np.linalg.norm(v)
        q = _quat_z_to(v)
        r = np.zeros(9)
        mujoco.mju_quat2Mat(r, q)
        assert np.allclose(r.reshape(3, 3) @ np.array([0, 0, 1.0]), v, atol=1e-9)


def test_densify_inserts_midpoints():
    xy = np.array([[[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]])
    out = _densify(xy)
    assert out.shape == (1, 5, 2) and np.allclose(out[0, 1], [1.0, 0.0]) and np.allclose(out[0, 3], [2.0, 1.0])


def test_rig_chase_follows_the_rover_with_the_requested_offset(rec):
    class Fake:
        def __init__(self, r):
            self.rec = r

        def rover_pose(self, s):
            return np.array([s, 0.0, 0.4]), 0.0
    poses = rig_chase(back=4.0, height=1.5, swing_deg=0.0, ahead=2.0)(Fake(rec), np.array([10.0, 20.0]), np.array([0.0, 1.0]))
    assert np.allclose(poses[0].pos, (6.0, 0.0, 1.9)) and np.allclose(poses[1].target[:2], (22.0, 0.0))


# ------------------------------------------------------------------------------------------- GPU
def _gpu_backend(model, size=(320, 180), **kw):
    from aedrover.viz.backend import FilamentBackend, filament_importable
    if not filament_importable():
        pytest.skip("MuJoCo has no Filament renderer")
    try:
        return FilamentBackend(model, size, **kw)
    except Exception as exc:                                    # no GPU / no OpenGL
        pytest.skip(f"Filament cannot start here: {exc}")


@pytest.mark.gpu
def test_filament_renders_a_lit_scene_with_sky_above_ground():
    xml = ("<mujoco><visual><global offwidth='320' offheight='180'/></visual><asset>"
           "<material name='g' rgba='.4 .4 .4 1'/></asset><worldbody><light type='directional' dir='-.3 .2 -.9'/>"
           "<geom type='plane' size='50 50 .1' material='g'/></worldbody></mujoco>")
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    bk = _gpu_backend(m)
    try:
        img = bk.render(d, CameraPose((0, 0, 1.5), (10, 0, 0.5), 50))
    finally:
        bk.close()
    assert img.shape == (180, 320, 3) and img.dtype == np.uint8
    assert img[-30:].mean() > 20, "the ground is lit, not black"


@pytest.mark.gpu
def test_dressed_scene_renders_finished_frames(rec):
    from aedrover.viz.render_model import RenderScene
    try:
        scene = RenderScene(rec, size=(320, 180), builder=dress_scene, depth=True)
    except Exception as exc:
        pytest.skip(f"cannot build a GPU scene here: {exc}")
    if scene.backend.name != "filament":
        scene.close()
        pytest.skip("Filament unavailable; the classic fallback has no depth pass")
    try:
        p, yaw = scene.rover_pose(40.0)
        frame = scene.render_finished(40.0, chase(p, yaw), dof=0.3, seed=1)
    finally:
        scene.close()
    assert frame.shape == (180, 320, 3) and 30 < frame.mean() < 235
    assert frame[:60].mean() > 25, "the sky/upper scene is lit"
