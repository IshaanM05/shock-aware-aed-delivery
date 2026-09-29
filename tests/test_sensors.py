"""Perception regression tests."""

import numpy as np

from aedrover.sim import Rover, World
from aedrover.sim.sensors import Perception


def _setup(x0: float, h: float = 0.0, x_up: float = 1e3):
    w = World()
    if h > 0:
        w.set_crossing(h, x_down=-9.0, x_up=x_up)
    else:
        w.set_flat()
    r = Rover(w)
    r.reset(x0, 0.0, 0.0, settle_s=0.3)
    return w, r, Perception(w.model, w.data, w.b_chassis, w.veh.nominal_height)


def test_height_scan_sees_the_road_far_from_the_origin():
    """Regression: mj_multiRay's cutoff dropped the infinite road plane beyond ~8 m from the origin."""
    for x0 in (0.0, 12.0, 45.0, 120.0):
        _, _, p = _setup(x0)
        hs = p.height_scan(0.0)
        assert np.allclose(hs, 0.0, atol=2e-3), f"flat scan wrong at x0={x0}: {hs[:, 1]}"


def test_lidar_range_is_clipped_and_finite_on_open_ground():
    _, _, p = _setup(30.0)
    li = p.lidar(0.0)
    assert np.all(li == p.s.lidar_range)


def test_height_scan_measures_a_kerb_step_up_and_down():
    w, r, p = _setup(0.0, h=0.12, x_up=3.0)
    col = p.height_scan(0.0)[:, 1]
    assert col.max() > 0.10                      # the 4.5 m ray lands on the kerb top
    assert np.all(col[:3] < 0.02)                # near rows still road
    # on the sidewalk looking at a drop
    w2 = World()
    w2.set_crossing(0.12, x_down=3.0, x_up=60.0)
    r2 = Rover(w2)
    r2.reset(0.0, 0.0, 0.12, settle_s=0.3)
    p2 = Perception(w2.model, w2.data, w2.b_chassis, w2.veh.nominal_height)
    col2 = p2.height_scan(0.0)[:, 1]
    assert col2.min() < -0.10


def test_lidar_detects_pedestrian_and_obstacle():
    w = World()
    w.set_flat()
    w.place_pedestrian(0, 4.0, 0.0, 0.0)
    w.set_obstacle(1, 6.0, 1.0, 0.0)   # planter, r = 0.28 m
    r = Rover(w)
    r.reset(0.0, 0.0, 0.0, settle_s=0.2)
    p = Perception(w.model, w.data, w.b_chassis, w.veh.nominal_height)
    li = p.lidar(0.0)
    centre = len(li) // 2
    assert 3.5 < li[centre] < 4.0                # pedestrian front surface at ~3.8 m
    assert li.min() < 4.0 and (li < p.s.lidar_range).sum() >= 2
