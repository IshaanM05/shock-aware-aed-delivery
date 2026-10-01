"""Rich-world street furniture: parked cars, motorbikes, stalls, lamp posts and tree trunks as real obstacles.

The standard benchmark world has none of these: the cinematic dressing (``aedrover.viz.dressing``) draws parked
vehicles, stalls, lamps and trees, but only at ``|y| >= 4`` from the corridor, where nothing can touch them.
The *rich-world mode* places a smaller set of the same kinds of objects where a Mumbai footpath really has them,
encroaching on the footway, and makes them collidable.

``place_furniture`` is a pure function of the scenario: same scenario, same items. It draws from its own random
stream (``default_rng([seed, tag, family])``), so it never shifts any draw of ``sample_scenario``, and the
benchmark scenarios, seeds and results are unchanged. The returned items are the single source of truth for the
physics (collision boxes in the MJCF, ``World``), the clearance and outcome checks (``AEDRoverEnv``), MPPI's internal
model and the dressing, so what is drawn is exactly what collides.

Feasibility is guaranteed by construction and checked by tests over many seeds: an item may reach at most
``MAX_INTRUSION_M`` past the corridor edge (``CORRIDOR_HALF_WIDTH``), items on opposite sides keep ``SIDE_GAP_M``
apart along the footway, and nothing stands in the start zone, the goal zone, the road crossing, or within
``OBSTACLE_GAP_M`` of a scenario obstacle. A clear lane of at least ``MIN_CLEAR_LANE_M`` therefore exists at every x.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .scenario import CORRIDOR_HALF_WIDTH, FAMILIES, Scenario

MAX_INTRUSION_M = 1.0          # furthest an item reaches into the corridor from its edge
MIN_CLEAR_LANE_M = 1.6         # lane that must stay free at every x (the rover is 0.72 m wide)
SIDE_GAP_M = 4.0               # along-footway gap between items on opposite sides
SAME_SIDE_GAP_M = 1.2          # along-footway gap between items on the same side
OBSTACLE_GAP_M = 3.0           # along-footway gap to a scenario bollard or planter
START_CLEAR_M = 3.5            # nothing before this x (the rover starts at x = 0)
GOAL_CLEAR_M = 3.5             # nothing within this distance of the goal
CROSSING_PAD_M = 1.5           # keep clear of the road crossing by this much
STEP_M = (2.5, 6.0)            # spacing of candidate positions along the footway
TAG = 0xF0

KINDS = ("car", "bikes", "stall", "lamp", "tree")
_WEIGHTS = (0.15, 0.35, 0.15, 0.20, 0.15)


@dataclass(frozen=True)
class Box:
    """A collision proxy in the item's local frame (x along the item, y across it). It spans from the road surface
    up to ``top`` above the item's support surface, so no gap under it can be climbed or slipped under."""

    cx: float
    cy: float
    hx: float
    hy: float
    top: float


@dataclass(frozen=True)
class FurnitureItem:
    kind: str
    x: float
    y: float
    yaw: float                      # rad, rotation of the local frame about z
    z_base: float                   # height of the support surface under the item [m]
    boxes: tuple[Box, ...]
    variant: int = 0                # colour / detail index for the dressing (drawn from the item's own stream)
    count: int = 1                  # motorbikes in a row

    # -- geometry
    def world_boxes(self) -> list[tuple[float, float, float, float, float, float]]:
        """``(x, y, yaw, hx, hy, top_z)`` of every proxy in world coordinates."""
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return [(self.x + c * b.cx - s * b.cy, self.y + s * b.cx + c * b.cy, self.yaw, b.hx, b.hy, self.z_base + b.top)
                for b in self.boxes]

    def y_extent(self) -> tuple[float, float]:
        """Lowest and highest world y reached by any proxy."""
        lo, hi = math.inf, -math.inf
        for _x, y, yaw, hx, hy, _ in self.world_boxes():
            r = abs(math.cos(yaw)) * hy + abs(math.sin(yaw)) * hx
            lo, hi = min(lo, y - r), max(hi, y + r)
        return lo, hi

    def x_extent(self) -> tuple[float, float]:
        lo, hi = math.inf, -math.inf
        for x, _y, yaw, hx, hy, _ in self.world_boxes():
            r = abs(math.cos(yaw)) * hx + abs(math.sin(yaw)) * hy
            lo, hi = min(lo, x - r), max(hi, x + r)
        return lo, hi

    def blocked_at(self, x: float) -> list[tuple[float, float]]:
        """y-intervals of the proxies' footprints at the along-footway position ``x`` (empty when ``x`` misses them)."""
        out = []
        for bx, by, yaw, hx, hy, _ in self.world_boxes():
            c, s = math.cos(yaw), math.sin(yaw)
            corners = [(bx + c * sx * hx - s * sy * hy, by + s * sx * hx + c * sy * hy)
                       for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
            ys = []
            for i in range(4):
                (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
                if min(x0, x1) <= x <= max(x0, x1):
                    if abs(x1 - x0) < 1e-12:
                        ys += [y0, y1]
                    else:
                        ys.append(y0 + (y1 - y0) * (x - x0) / (x1 - x0))
            if ys:
                out.append((min(ys), max(ys)))
        return out

    # -- (de)serialisation, for recordings
    def to_dict(self) -> dict:
        return {"kind": self.kind, "x": self.x, "y": self.y, "yaw": self.yaw, "z_base": self.z_base, "variant": self.variant,
                "count": self.count, "boxes": [[b.cx, b.cy, b.hx, b.hy, b.top] for b in self.boxes]}

    @classmethod
    def from_dict(cls, d: dict) -> FurnitureItem:
        return cls(d["kind"], float(d["x"]), float(d["y"]), float(d["yaw"]), float(d["z_base"]), tuple(Box(*map(float, b)) for b in d["boxes"]),
                   int(d.get("variant", 0)), int(d.get("count", 1)))


@dataclass(frozen=True)
class _Made:
    boxes: tuple[Box, ...]
    variant: int                    # colour or detail index for the dressing
    count: int                      # motorbikes in a row
    yaw: float                      # rotation of the local frame about z
    along: float                    # half extent along the footway (world x)
    across: float                   # half extent across the footway (world y)


def _car(rng: np.random.Generator) -> _Made:
    return _Made((Box(0.0, 0.0, 2.15, 0.88, 1.45),), int(rng.integers(8)), 1, 0.0, 2.15, 0.88)


def _bikes(rng: np.random.Generator) -> _Made:
    n = int(rng.integers(1, 4))             # a row of n bikes parked head-in: the long axis (local x) runs across the footway
    return _Made((Box(0.0, 0.0, 0.9, 0.3 * n, 1.05),), int(rng.integers(5)), n, math.pi / 2, 0.3 * n, 0.9)


def _stall(rng: np.random.Generator) -> _Made:
    return _Made((Box(0.0, 0.0, 0.7, 0.4, 0.9),), int(rng.integers(4)), 1, 0.0, 0.7, 0.4)


def _lamp(rng: np.random.Generator) -> _Made:
    return _Made((Box(0.0, 0.0, 0.08, 0.08, 3.0),), 0, 1, 0.0, 0.08, 0.08)


def _tree(rng: np.random.Generator) -> _Made:
    return _Made((Box(0.0, 0.0, 0.4, 0.4, 0.44), Box(0.0, 0.0, 0.14, 0.14, 3.0)), int(rng.integers(3)), 1, 0.0, 0.4, 0.4)


_MAKERS = {"car": _car, "bikes": _bikes, "stall": _stall, "lamp": _lamp, "tree": _tree}
# (min, max) of how far the item reaches into the corridor, measured from the corridor edge
_INTRUSION = {"car": (0.3, 1.0), "bikes": (0.25, 1.0), "stall": (0.0, 0.6), "lamp": (0.1, 0.55), "tree": (0.1, 0.6)}


def _x_ranges(sc: Scenario) -> list[tuple[float, float]]:
    lo, hi = START_CLEAR_M, sc.x_goal - GOAL_CLEAR_M
    if not sc.has_kerb:
        return [(lo, hi)]
    return [(lo, sc.x_down - CROSSING_PAD_M), (sc.x_up + CROSSING_PAD_M, hi)]


def place_furniture(sc: Scenario, *, density: float = 1.5) -> list[FurnitureItem]:
    """Deterministic furniture for ``sc`` (``density`` scales how many items there are; 0 gives none)."""
    if density <= 0:
        return []
    rng = np.random.default_rng([int(sc.seed), TAG, FAMILIES.index(sc.family) if sc.family in FAMILIES else len(FAMILIES)])
    items: list[FurnitureItem] = []
    taken: list[tuple[float, float, float]] = []                 # (x_lo, x_hi, side) of every placed item
    for lo, hi in _x_ranges(sc):
        x = lo
        while True:
            x += float(rng.uniform(*STEP_M)) / density
            kind = KINDS[int(rng.choice(len(KINDS), p=_WEIGHTS))]
            side = 1.0 if rng.random() < 0.5 else -1.0
            made = _MAKERS[kind](rng)
            intrusion = float(rng.uniform(*_INTRUSION[kind]))
            x = max(x, lo + made.along)                          # the whole item, not just its centre, stays inside the range
            if x + made.along > hi:
                break
            y = side * (CORRIDOR_HALF_WIDTH - intrusion + made.across)
            x_lo, x_hi = x - made.along, x + made.along
            if any(x_lo < b + (SIDE_GAP_M if s_ != side else SAME_SIDE_GAP_M) and x_hi > a - (SIDE_GAP_M if s_ != side else SAME_SIDE_GAP_M)
                   for a, b, s_ in taken):
                continue
            if any(x_lo - OBSTACLE_GAP_M < ob.x < x_hi + OBSTACLE_GAP_M for ob in sc.obstacles):
                continue
            items.append(FurnitureItem(kind, float(x), float(y), made.yaw, float(sc.surface_z(x)), made.boxes, made.variant, made.count))
            taken.append((x_lo, x_hi, side))
            x = x_hi
    return items


def clear_lane_m(items: list[FurnitureItem], sc: Scenario | None = None, xs: np.ndarray | None = None, *,
                 x_end: float = 36.0) -> float:
    """Smallest, over the footway, of the widest free lane inside the corridor ``|y| < CORRIDOR_HALF_WIDTH``.

    Free means not covered by a furniture proxy or, when a scenario is given, by one of its bollards or planters (as a
    disc). Pedestrians are not counted (they move). Used by the tests to prove the feasibility rule; ``MIN_CLEAR_LANE_M``
    is the contract for the furniture alone, and furniture never comes within ``OBSTACLE_GAP_M`` of an obstacle, so
    the two never combine into something narrower than either.
    """
    from .world import OBS_SIZES

    x_end = sc.x_goal if sc is not None else x_end
    xs = np.arange(0.0, x_end, 0.05) if xs is None else xs
    obstacles = sc.obstacles if sc is not None else []
    worst = 2 * CORRIDOR_HALF_WIDTH
    for x in xs:
        blocked = [iv for it in items for iv in it.blocked_at(float(x))]
        for ob in obstacles:
            r = OBS_SIZES[ob.slot % 2][0]
            if abs(ob.x - x) < r:
                h = math.sqrt(r * r - (ob.x - x) ** 2)
                blocked.append((ob.y - h, ob.y + h))
        edges = sorted((max(lo, -CORRIDOR_HALF_WIDTH), min(hi, CORRIDOR_HALF_WIDTH)) for lo, hi in blocked
                       if hi > -CORRIDOR_HALF_WIDTH and lo < CORRIDOR_HALF_WIDTH)
        cursor, widest = -CORRIDOR_HALF_WIDTH, 0.0
        for lo, hi in edges:
            widest = max(widest, lo - cursor)
            cursor = max(cursor, hi)
        worst = min(worst, max(widest, CORRIDOR_HALF_WIDTH - cursor))
    return float(worst)
