"""Scenario families and seeded sampling.

A *segment* is a straight 36 m stretch of sidewalk. Kerb families add a road crossing:

    walk A [0, x_down]  |  road [x_down, x_up]  |  walk B [x_up, x_goal]

The full mission of 1-2 km is composed hierarchically from many such segments (see
``aedrover.clinical`` and experiment 05): physics is simulated per segment type, the route level
composes the empirical per-segment time / success distributions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FAMILIES = ("flat_clear", "kerb", "crowded", "mixed", "slippery")

SIDEWALK_HALF_WIDTH = 1.5
CORRIDOR_HALF_WIDTH = 1.4      # lateral limit before the rover counts as off the sidewalk


@dataclass
class PedSpec:
    """A pedestrian in a stream: walks ``origin -> goal`` and respawns at the origin on arrival."""

    start: tuple[float, float]
    goal: tuple[float, float]
    v_des: float
    aware: bool                # aware pedestrians yield to the rover, distracted ones do not
    z_surface: float
    origin: tuple[float, float] = (0.0, 0.0)


@dataclass
class ObstacleSpec:
    slot: int
    x: float
    y: float
    z_surface: float


@dataclass
class Scenario:
    family: str
    seed: int
    kerb_h: float = 0.0
    ramp_down: bool = False
    ramp_up: bool = False
    x_down: float = 1e3
    x_up: float = 1e3
    x_goal: float = 36.0
    start_y: float = 0.0
    start_yaw: float = 0.0
    tyre_mu: float = 1.0
    ground_mu: float = 1.0
    payload_mass: float = 4.0
    peds: list[PedSpec] = field(default_factory=list)
    obstacles: list[ObstacleSpec] = field(default_factory=list)

    @property
    def has_kerb(self) -> bool:
        return self.kerb_h > 1e-6

    def surface_z(self, x: float) -> float:
        """Nominal supporting surface height at longitudinal position ``x`` (ignores ramps)."""
        if not self.has_kerb:
            return 0.0
        return self.kerb_h if (x <= self.x_down or x >= self.x_up) else 0.0

    def to_dict(self) -> dict:
        return {
            "family": self.family, "seed": self.seed, "kerb_h": self.kerb_h,
            "ramp_down": self.ramp_down, "ramp_up": self.ramp_up, "x_down": self.x_down,
            "x_up": self.x_up, "x_goal": self.x_goal, "tyre_mu": self.tyre_mu,
            "payload_mass": self.payload_mass, "n_ped": len(self.peds),
            "n_obstacle": len(self.obstacles),
        }


def _walk_ranges(sc: Scenario) -> list[tuple[float, float]]:
    if not sc.has_kerb:
        return [(3.0, sc.x_goal - 3.0)]
    return [(3.0, sc.x_down - 2.5), (sc.x_up + 2.5, sc.x_goal - 3.0)]


def _sample_peds(sc: Scenario, rng: np.random.Generator, n: int, p_aware: float) -> None:
    """Pedestrian streams: each walks the full length of one sidewalk piece, either direction."""
    ranges = _walk_ranges(sc)
    lengths = np.array([b - a for a, b in ranges])
    for _ in range(n):
        k = int(rng.choice(len(ranges), p=lengths / lengths.sum()))
        a, b = ranges[k]
        direction = 1.0 if rng.random() < 0.5 else -1.0
        x_origin, x_goal = (a, b) if direction > 0 else (b, a)
        y0, y1 = rng.uniform(-1.0, 1.0, size=2)
        z = sc.surface_z(0.5 * (a + b))
        sc.peds.append(PedSpec(start=(float(rng.uniform(a, b)), float(y0)), goal=(float(x_goal), float(y1)),
                               v_des=float(rng.uniform(0.9, 1.6)), aware=bool(rng.random() < p_aware),
                               z_surface=z, origin=(float(x_origin), float(y0))))


def _sample_obstacles(sc: Scenario, rng: np.random.Generator, n: int, n_slots: int) -> None:
    ranges = _walk_ranges(sc)
    for i in range(min(n, n_slots)):
        a, b = ranges[int(rng.integers(len(ranges)))]
        sc.obstacles.append(ObstacleSpec(i, float(rng.uniform(a + 1.0, b - 1.0)),
                                         float(rng.uniform(-0.9, 0.9)), sc.surface_z(0.5 * (a + b))))


def sample_scenario(family: str, seed: int, *, n_ped_max: int = 6, n_obs_max: int = 6,
                    kerb_range: tuple[float, float] = (0.09, 0.15),
                    mu_range: tuple[float, float] = (0.7, 1.1)) -> Scenario:
    """Deterministically sample a scenario of ``family`` from ``seed``."""
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
    rng = np.random.default_rng(seed)
    sc = Scenario(family=family, seed=seed)
    sc.payload_mass = float(rng.uniform(3.0, 5.0))
    sc.tyre_mu = sc.ground_mu = float(rng.uniform(*mu_range))
    sc.start_y = float(rng.uniform(-0.3, 0.3))
    sc.start_yaw = float(rng.uniform(-0.12, 0.12))

    if family in ("kerb", "mixed", "slippery"):
        sc.kerb_h = float(rng.uniform(*kerb_range))
        sc.ramp_down = bool(rng.random() < 0.25)
        sc.ramp_up = bool(rng.random() < 0.25)
        sc.x_down = 16.0
        sc.x_up = 22.0
        sc.x_goal = 36.0
    if family == "slippery":
        sc.tyre_mu = sc.ground_mu = float(rng.uniform(0.35, 0.6))

    if family == "crowded":
        _sample_peds(sc, rng, int(rng.integers(3, n_ped_max + 1)), p_aware=0.8)
        _sample_obstacles(sc, rng, int(rng.integers(1, 3)), n_obs_max)
    elif family in ("mixed", "slippery"):
        _sample_peds(sc, rng, int(rng.integers(2, min(4, n_ped_max) + 1)), p_aware=0.8)
        _sample_obstacles(sc, rng, int(rng.integers(1, 4)), n_obs_max)
    return sc
