"""Rich-world mode: collidable street furniture in the physics, perception and episode outcome.

Everything here is opt-in. The default world and environment are locked by ``tests/test_benchmark_lock.py``.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from aedrover.sim import VehicleParams, World
from aedrover.sim.furniture import Box, FurnitureItem
from aedrover.sim.rover import Rover
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
