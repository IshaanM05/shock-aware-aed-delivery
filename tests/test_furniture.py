"""Rich-world furniture placement: determinism, the feasibility rule, and that the benchmark's scenarios are untouched."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest

from aedrover.sim import FAMILIES, sample_scenario
from aedrover.sim.furniture import (
    CORRIDOR_HALF_WIDTH,
    CROSSING_PAD_M,
    GOAL_CLEAR_M,
    MAX_INTRUSION_M,
    MIN_CLEAR_LANE_M,
    OBSTACLE_GAP_M,
    SAME_SIDE_GAP_M,
    SIDE_GAP_M,
    START_CLEAR_M,
    Box,
    FurnitureItem,
    clear_lane_m,
    place_furniture,
)

SEEDS = range(5000, 5040)


def _scenarios():
    return [sample_scenario(f, s) for f in FAMILIES for s in SEEDS]


def test_placement_is_deterministic_and_depends_on_the_scenario():
    sc = sample_scenario("crowded", 5021)
    assert place_furniture(sc) == place_furniture(sample_scenario("crowded", 5021))
    assert place_furniture(sc) != place_furniture(sample_scenario("crowded", 5022))
    assert place_furniture(sc, density=0) == []
    many, few = place_furniture(sc, density=3.0), place_furniture(sc, density=0.5)
    assert len(many) > len(few)


def test_placement_never_touches_the_scenario_or_its_random_stream():
    for sc in _scenarios()[::17]:
        before = copy.deepcopy(sc)
        place_furniture(sc)
        assert sc == before                                       # nothing mutated
        assert sample_scenario(sc.family, sc.seed) == before      # and sampling the scenario again gives the same one


def test_every_scenario_keeps_a_clear_lane_and_the_keep_out_zones():
    intruded = 0
    for sc in _scenarios():
        items = place_furniture(sc)
        assert clear_lane_m(items, None, x_end=sc.x_goal) >= MIN_CLEAR_LANE_M - 1e-9, (sc.family, sc.seed)
        for it in items:
            lo, hi = it.y_extent()
            inner = min(abs(lo), abs(hi)) if lo * hi > 0 else 0.0
            assert inner >= CORRIDOR_HALF_WIDTH - MAX_INTRUSION_M - 1e-9, (it.kind, sc.family, sc.seed)
            intruded += inner < CORRIDOR_HALF_WIDTH
            x0, x1 = it.x_extent()
            assert x0 >= START_CLEAR_M and x1 <= sc.x_goal - GOAL_CLEAR_M, (it.kind, sc.family, sc.seed)
            if sc.has_kerb:
                assert x1 <= sc.x_down - CROSSING_PAD_M or x0 >= sc.x_up + CROSSING_PAD_M
            assert all(x0 - OBSTACLE_GAP_M >= ob.x or ob.x >= x1 + OBSTACLE_GAP_M for ob in sc.obstacles)
            assert it.z_base == pytest.approx(sc.surface_z(it.x))
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                gap = max(b.x_extent()[0] - a.x_extent()[1], a.x_extent()[0] - b.x_extent()[1])
                same_side = a.y * b.y > 0
                assert gap >= (SAME_SIDE_GAP_M if same_side else SIDE_GAP_M) - 1e-9
    assert intruded > 100                                         # the rule is not met vacuously: items do reach into the corridor


def test_every_kind_appears_and_each_has_collision_boxes_from_the_ground_up():
    kinds = {it.kind for sc in _scenarios() for it in place_furniture(sc)}
    assert kinds == {"car", "bikes", "stall", "lamp", "tree"}
    for it in (it for sc in _scenarios()[:40] for it in place_furniture(sc)):
        assert it.boxes and all(b.top > 0.4 and b.hx > 0 and b.hy > 0 for b in it.boxes)


def test_blocked_interval_geometry():
    car = FurnitureItem("car", 10.0, 2.0, 0.0, 0.0, (Box(0.0, 0.0, 1.0, 0.5, 1.0),))
    assert car.blocked_at(10.0) == [(1.5, 2.5)] and car.blocked_at(12.0) == []
    turned = FurnitureItem("bikes", 10.0, 2.0, math.pi / 2, 0.0, (Box(0.0, 0.0, 1.0, 0.5, 1.0),))     # long axis now along y
    (lo, hi), = turned.blocked_at(10.4)
    assert lo == pytest.approx(1.0) and hi == pytest.approx(3.0) and turned.blocked_at(10.6) == []
    assert np.allclose(turned.x_extent(), (9.5, 10.5)) and np.allclose(turned.y_extent(), (1.0, 3.0))
    skew = FurnitureItem("car", 0.0, 0.0, math.pi / 4, 0.0, (Box(0.0, 0.0, 1.0, 1.0, 1.0),))
    (lo, hi), = skew.blocked_at(0.0)
    assert hi == pytest.approx(math.sqrt(2)) and lo == pytest.approx(-math.sqrt(2))


def test_items_survive_a_round_trip_through_json():
    import json
    for sc in _scenarios()[::23]:
        items = place_furniture(sc)
        again = [FurnitureItem.from_dict(json.loads(json.dumps(it.to_dict()))) for it in items]
        assert again == items
