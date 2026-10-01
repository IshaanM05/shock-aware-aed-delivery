"""Rich-world mode: collidable street furniture in the physics, perception and episode outcome.

Everything here is opt-in. The default world and environment are locked by ``tests/test_benchmark_lock.py``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from aedrover.control import SafetyFilter
from aedrover.nav import make_controller
from aedrover.nav.base import run_episode
from aedrover.sim import AEDRoverEnv, VehicleParams, World
from aedrover.sim.furniture import Box, FurnitureItem, cover_discs, footprint_gap, place_furniture
from aedrover.sim.rover import Rover
from aedrover.sim.scenario import sample_scenario
from aedrover.sim.sensors import Perception
from aedrover.sim.world import build_xml, build_xml_rich

VEH = VehicleParams.optimized()
FACE_X = 5.0                      # the front face of the car below


def _car(x: float = 6.0, y: float = 0.0) -> FurnitureItem:
    """A 2 m long, 1.6 m wide, 1 m high block whose front face is at x = 5 when placed at x = 6."""
    return FurnitureItem("car", x, y, 0.0, 0.0, (Box(0.0, 0.0, 1.0, 0.8, 1.0),))


# ------------------------------------------------------------------------------------- the model
def test_no_furniture_gives_the_benchmark_world_byte_for_byte():
    assert build_xml_rich(VEH) == build_xml(VEH)
    assert build_xml_rich(VEH, furniture=[]) == build_xml(VEH)
    assert World(VEH, furniture=[]).xml == build_xml(VEH)


def test_furniture_adds_static_collidable_geoms_and_nothing_else():
    base, rich = World(VEH), World(VEH, furniture=[_car(), _car(20.0, 2.5)])
    m, b = rich.model, base.model
    assert m.ngeom == b.ngeom + 2 and m.nbody == b.nbody and m.nq == b.nq and m.nmocap == b.nmocap and m.nu == b.nu
    assert len(rich.g_furniture) == 2 and len(base.g_furniture) == 0
    for g in rich.g_furniture:
        assert m.geom_bodyid[g] == 0                                              # static: part of the world body
        assert m.geom_contype[g] == 1 and m.geom_conaffinity[g] == 1 and m.geom_group[g] == 0
    g = rich.g_furniture[0]
    assert np.allclose(m.geom_pos[g], [6.0, 0.0, 0.5]) and np.allclose(m.geom_size[g], [1.0, 0.8, 0.5])


def test_a_proxy_is_posed_by_its_yaw_and_stands_on_the_road_even_on_a_raised_footway():
    item = FurnitureItem("bikes", 10.0, 2.0, np.pi / 2, 0.12, (Box(0.0, 0.0, 0.9, 0.3, 1.05),))
    w = World(VEH, furniture=[item])
    g = w.g_furniture[0]
    assert np.allclose(w.model.geom_size[g], [0.9, 0.3, (0.12 + 1.05) / 2])         # from z = 0 to the top above the kerb height
    mujoco.mj_forward(w.model, w.data)
    corners = w.data.geom_xmat[g].reshape(3, 3) @ np.array([0.9, 0.0, 0.0])
    assert np.allclose(corners, [0.0, 0.9, 0.0], atol=1e-6)                          # the long axis now runs along y


# --------------------------------------------------------------------- physics and perception
def _drive(world: World, seconds: float, v: float = 1.5) -> Rover:
    rover = Rover(world)
    rover.reset(0.0, 0.0, 0.0, yaw=0.0, settle_s=0.4)
    for _ in range(int(seconds / rover.control_dt)):
        rover.step(v, 0.0)
    return rover


def test_the_rover_is_physically_stopped_by_a_car_in_its_lane():
    free = _drive(World(VEH), 6.0)
    assert free.pos[0] > FACE_X + 2.0                                                # nothing in the way: it drives straight past
    world = World(VEH, furniture=[_car()])
    rover = Rover(world)
    rover.reset(0.0, 0.0, 0.0, yaw=0.0, settle_s=0.4)
    furniture = set(world.g_furniture.tolist())
    farthest, touched = 0.0, False
    for _ in range(int(6.0 / rover.control_dt)):
        rover.step(1.0, 0.0)
        farthest = max(farthest, float(rover.pos[0]))
        d = world.data
        touched |= any({int(c.geom1), int(c.geom2)} & furniture for c in d.contact[: d.ncon])
    assert rover.is_finite() and touched                                             # it hit the car ...
    assert FACE_X - 1.2 < farthest < FACE_X                                          # ... and its centre never got past the front face


def test_lidar_returns_hit_the_furniture_at_the_exact_face_distance():
    world = World(VEH, furniture=[_car()])
    rover = Rover(world)
    rover.reset(0.0, 0.0, 0.0, yaw=0.0, settle_s=0.4)
    perc = Perception(world.model, world.data, world.b_chassis, VEH.nominal_height)
    ranges = perc.lidar(rover.yaw_pitch_roll()[0])
    centre = len(ranges) // 2
    assert ranges[centre] == pytest.approx(FACE_X - float(world.data.xpos[world.b_chassis][0]), abs=0.02)
    free_world = World(VEH)
    Rover(free_world).reset(0.0, 0.0, 0.0, yaw=0.0, settle_s=0.4)
    free = Perception(free_world.model, free_world.data, free_world.b_chassis, VEH.nominal_height).lidar(0.0)
    assert free[centre] > FACE_X + 3.0                                               # the same ray sees nothing without the car


# ------------------------------------------------------------------------------ environment
def test_footprint_gap_is_a_signed_separating_axis_distance():
    box = np.array([[2.0, 0.0, 0.0, 1.0, 1.0]])                                      # x from 1 to 3
    assert footprint_gap(0.0, 0.0, 0.0, box, 0.5, 0.4)[0] == pytest.approx(0.5)       # 1.0 - 0.5
    assert footprint_gap(0.8, 0.0, 0.0, box, 0.5, 0.4)[0] == pytest.approx(-0.3)      # overlapping by 0.3
    assert footprint_gap(0.0, 3.0, 0.0, box, 0.5, 0.4)[0] == pytest.approx(1.6)       # 3.0 - 1.0 - 0.4
    turned = np.array([[2.0, 0.0, np.pi / 4, 0.5, 0.5]])                              # a diamond with corner reach 0.7071
    assert footprint_gap(0.0, 0.0, 0.0, turned, 0.5, 0.4)[0] == pytest.approx(2.0 - 0.5 - 0.5 * np.sqrt(2) * 1.0, abs=0.01)
    many = np.vstack([box, turned])
    assert footprint_gap(0.0, 0.0, 0.0, many, 0.5, 0.4).shape == (2,)


def test_the_default_environment_is_not_rich_and_keeps_its_world():
    env = AEDRoverEnv(veh=VEH, obs_mode="dict")
    env.reset(seed=1, options={"family": "crowded"})
    world = env.world
    env.reset(seed=2, options={"family": "crowded"})
    assert not env.rich and env.furniture == () and env.world is world and len(world.g_furniture) == 0


def test_a_rich_environment_compiles_each_scenarios_furniture_and_reuses_a_world_that_still_fits():
    env = AEDRoverEnv(veh=VEH, obs_mode="dict", rich=True)
    env.reset(seed=5021, options={"family": "crowded"})
    items = tuple(place_furniture(env.scenario, density=env.rich_density))
    assert items and env.furniture == items and len(env.world.g_furniture) == sum(len(i.boxes) for i in items)
    first = env.world
    env.reset(seed=5021, options={"family": "crowded"})
    assert env.world is first                                                        # same scenario: the compiled world is reused
    env.reset(seed=5022, options={"family": "crowded"})
    assert env.world is not first and env.furniture == tuple(place_furniture(env.scenario, density=env.rich_density))
    assert env.perc.m is env.world.model and env.rover.m is env.world.model            # rover and perception follow the new model


def _blind_run(env: AEDRoverEnv, seed: int, furniture) -> tuple[dict, int | None, float | None]:
    """Pure pursuit without a safety filter, straight at the furniture. Returns the episode, the step of the first physical
    contact and the analytic clearance at that step."""
    ctrl = make_controller("pure_pursuit", v_cruise=1.2)
    env.reset(seed=seed, options={"family": "flat_clear", "furniture": furniture})
    ctrl.reset(env)
    names = set(env.world.g_furniture.tolist())
    first_touch, clear_at_touch = None, None
    for step in range(3000):
        v, d = ctrl.act(env.obs)
        _, _, term, trunc, info = env.step(np.array([v, d]))
        d_ = env.world.data
        if first_touch is None and any({int(c.geom1), int(c.geom2)} & names for c in d_.contact[: d_.ncon]):
            first_touch, clear_at_touch = step, info["clearance"]
        if term or trunc:
            return info["episode"], first_touch, clear_at_touch
    raise AssertionError("episode did not end")


def test_driving_into_furniture_ends_in_a_furniture_collision_at_the_moment_of_contact():
    env = AEDRoverEnv(veh=VEH, obs_mode="dict")
    ep, first_touch, clear_at_touch = _blind_run(env, 5300, [_car(8.0, 0.0)])
    assert ep["outcome"] == "collision" and ep["collision_kind"] == "furniture" and ep["min_clearance_m"] < 0
    assert first_touch is not None and abs(clear_at_touch) < 0.03                     # the footprint is within 3 cm of the real contact
    assert 0 <= ep["steps"] - 1 - first_touch <= 5                                    # and the episode ends within 0.1 s of it


def test_the_safety_filter_sees_furniture_through_the_lidar_and_stops_in_front_of_it():
    env = AEDRoverEnv(veh=VEH, obs_mode="dict")
    ctrl, shield = make_controller("pure_pursuit", v_cruise=1.2), SafetyFilter()
    ep = run_episode(env, ctrl, shield, seed=5300, options={"family": "flat_clear", "furniture": [_car(8.0, 0.0)]})
    assert ep["outcome"] != "collision" and ep["outcome"] != "goal"                  # held back (a stall), not through it
    assert shield.n_interventions > 0 and ep["min_clearance_m"] > 0


def test_a_clear_scenario_without_furniture_in_the_way_is_unaffected_by_rich_mode():
    plain = AEDRoverEnv(veh=VEH, obs_mode="dict")
    rich = AEDRoverEnv(veh=VEH, obs_mode="dict", rich=True)
    off_path = [_car(14.0, 4.5)]                                                      # far to the side: can neither be touched nor seen
    a = run_episode(plain, make_controller("dwa", v_cruise=2.0), SafetyFilter(), seed=5301, options={"family": "flat_clear"})
    b = run_episode(rich, make_controller("dwa", v_cruise=2.0), SafetyFilter(), seed=5301,
                    options={"family": "flat_clear", "furniture": off_path})
    for k in ("outcome", "time_s", "peak_shock_g", "path_m"):
        assert a[k] == pytest.approx(b[k], abs=1e-9), k


# ------------------------------------------------------------------------------------- MPPI
def test_cover_discs_cover_every_proxy_without_overshooting_much():
    rng = np.random.default_rng(0)
    for sc in (sample_scenario(f, s) for f in ("flat_clear", "crowded") for s in range(5000, 5012)):
        items = place_furniture(sc)
        discs = cover_discs(items)
        assert len(discs) >= len(items)
        for x, y, yaw, hx, hy, _ in (b for it in items for b in it.world_boxes()):
            local = rng.uniform(-1, 1, size=(200, 2)) * np.array([hx, hy])
            c, s = np.cos(yaw), np.sin(yaw)
            pts = np.stack([x + c * local[:, 0] - s * local[:, 1], y + s * local[:, 0] + c * local[:, 1]], axis=1)
            d = np.hypot(pts[:, None, 0] - discs[None, :, 0], pts[:, None, 1] - discs[None, :, 1]) - discs[None, :, 2]
            assert (d.min(axis=1) <= 1e-9).all()                                    # every point of the footprint is inside a disc
        for x, y, r in discs:                                                      # and no disc is far bigger than what it covers
            near = [b for it in items for b in it.world_boxes() if np.hypot(b[0] - x, b[1] - y) < 3.0]
            assert r <= 1.2 * max(min(b[3], b[4]) for b in near) + 0.2 or r <= 0.25


def test_mppi_plans_in_a_world_with_the_same_furniture_and_default_planning_is_unchanged():
    ctrl = make_controller("mppi", K=4, H=4)
    plain = AEDRoverEnv(veh=VEH, obs_mode="dict")
    plain.reset(seed=5302, options={"family": "crowded"})
    ctrl.reset(plain)
    assert ctrl._veh_key[2] == () and len(ctrl._obs_pts) == len(plain.scenario.obstacles)
    assert ctrl.model.ngeom == plain.world.model.ngeom

    rich = AEDRoverEnv(veh=VEH, obs_mode="dict")
    bike = FurnitureItem("bikes", 12.0, 0.8, np.pi / 2, 0.0, (Box(0.0, 0.0, 0.9, 0.3, 1.05),))
    rich.reset(seed=5302, options={"family": "crowded", "furniture": [bike]})
    ctrl.reset(rich)
    assert ctrl.model.ngeom == rich.world.model.ngeom == plain.world.model.ngeom + 1       # the planner's model has the proxy
    assert len(ctrl._obs_pts) > len(rich.scenario.obstacles)                                # and its cost has the covering discs
    again = ctrl._pw
    ctrl.reset(rich)
    assert ctrl._pw is again                                                              # same furniture: no rebuild
    rich.reset(seed=5302, options={"family": "crowded", "furniture": [bike, _car(24.0, 0.9)]})
    ctrl.reset(rich)
    assert ctrl._pw is not again and ctrl.model.ngeom == plain.world.model.ngeom + 2


@pytest.mark.slow
def test_mppi_steers_round_furniture_that_stops_pure_pursuit_and_never_touches_it():
    stall = FurnitureItem("stall", 12.0, 0.6, 0.0, 0.0, (Box(0.0, 0.0, 0.6, 0.4, 0.9),))          # inner edge at y = 0.2: on the straight line
    env = AEDRoverEnv(veh=VEH, obs_mode="dict")
    blind, _, _ = _blind_run(env, 5303, [stall])
    assert blind["outcome"] == "collision" and blind["collision_kind"] == "furniture"
    ctrl = make_controller("mppi", K=48, H=14, nthread=2)
    env.reset(seed=5303, options={"family": "flat_clear", "furniture": [stall]})
    ctrl.reset(env)
    for _ in range(3000):
        v, d = ctrl.act(env.obs)
        _, _, term, trunc, info = env.step(np.array([v, d]))
        if term or trunc:
            break
    ep = info["episode"]
    assert ep["outcome"] == "goal" and ep["min_clearance_m"] > 0


def test_obstacle_slots_are_analytic_only():
    """Engineering note 16: bollards and planters are never touched by contact, only by the analytic footprint test."""
    w = World(VEH)
    assert (w.model.geom_contype[w.g_obs] == 0).all() and (w.model.geom_conaffinity[w.g_obs] == 0).all()
    assert (w.model.geom_contype[w.g_ped] != 0).all()                                # pedestrians, by contrast, are collidable capsules


# ------------------------------------------------------------------------------- pedestrians
def _walker(solids):
    from aedrover.sim.pedestrians import PedestrianCrowd
    from aedrover.sim.scenario import PedSpec

    crowd = PedestrianCrowd()
    crowd.reset([PedSpec(start=(2.0, 0.0), goal=(12.0, 0.0), v_des=1.3, aware=False, z_surface=0.0, origin=(2.0, 0.0))],
                np.random.default_rng(0))
    crowd.solids = solids
    return crowd


def test_pedestrians_walk_round_furniture_and_only_when_it_is_there():
    from aedrover.sim.pedestrians import PED_R

    box = np.array([[6.0, 0.3, 0.0, 0.7, 0.4]])                                       # a stall on the walker's line (y from -0.1 to 0.7)
    free, blocked = _walker(None), _walker(box)
    nearest, passed_beside, far_free, far_blocked = np.inf, False, 0.0, 0.0
    for _ in range(1200):
        free.step(0.02, np.array([-50.0, 0.0]), np.zeros(2))
        blocked.step(0.02, np.array([-50.0, 0.0]), np.zeros(2))
        far_free, far_blocked = max(far_free, float(free.pos[0, 0])), max(far_blocked, float(blocked.pos[0, 0]))
        _, _, d, depth = blocked._solid_geometry()
        nearest = min(nearest, float((d - depth).min()))
        passed_beside |= bool(abs(blocked.pos[0, 0] - 6.0) < 0.2 and blocked.pos[0, 1] < -0.1 - PED_R + 1e-6)
    assert nearest >= PED_R - 1e-6                                                      # never overlaps the stall
    assert far_blocked > 11.0 and far_free > 11.0                                       # both reach the far end (then respawn) ...
    assert passed_beside                                                                # ... the blocked one by going round it


def test_the_standard_crowd_has_no_solids_and_a_rich_one_keeps_pedestrians_out_of_every_proxy():
    plain = AEDRoverEnv(veh=VEH, obs_mode="dict")
    plain.reset(seed=5021, options={"family": "crowded"})
    assert plain.crowd.solids is None
    rich = AEDRoverEnv(veh=VEH, obs_mode="dict", rich=True)
    rich.reset(seed=5021, options={"family": "crowded"})
    assert rich.crowd.solids is rich._fur and rich._fur is not None
    ctrl = make_controller("dwa", v_cruise=2.0)
    ctrl.reset(rich)
    from aedrover.sim.pedestrians import PED_R
    worst = np.inf
    for _ in range(1500):
        v, d = ctrl.act(rich.obs)
        _, _, term, trunc, _ = rich.step(np.array([v, d]))
        _, _, dist, depth = rich.crowd._solid_geometry()
        worst = min(worst, float((dist - depth).min()))
        if term or trunc:
            break
    assert worst >= PED_R - 1e-6
