"""Speed-and-separation safety filter (CBF-style stopping-distance barrier on the swept footprint).

For a commanded steering angle the rover's footprint (a 1.04 x 0.72 m rectangle) is swept along
the circular arc of the bicycle model. The first arc length ``d_free`` at which the (slightly
inflated) footprint overlaps a lidar return or a tracked pedestrian defines the barrier

    h(v) = d_free - v * t_react - v^2 / (2 * a_brake) >= 0

and the filter clips the speed to the largest ``v`` with ``h(v) >= 0``. Because the swept
region follows the commanded arc, an obstacle 0.5 m to the side is not mistaken for one dead
ahead, so the filter is consistent with curved-path planners (DWA, MPPI, learned policies).
Near pedestrians the speed is additionally capped (``ped_speed_cap`` within ``ped_cap_radius``):
this cap is a policy parameter swept in experiment 05 (the course brief cites 0.8 m/s in shared
zones). The filter only ever reduces speed; steering is passed through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..sim.pedestrians import PED_R, ROBOT_HALF_L, ROBOT_HALF_W

WHEELBASE = 0.62


@dataclass(frozen=True)
class SafetyParams:
    a_brake: float = 2.0          # m/s^2 service braking used in the barrier
    t_react: float = 0.10         # s, perception + actuation latency
    front_margin: float = 0.15    # m, extra clearance ahead of the footprint
    side_margin: float = 0.04     # m, extra clearance to the sides of the footprint
    ped_extra: float = 0.10       # m, additional allowance around pedestrians (they move)
    horizon: float = 6.0          # m, arc length considered
    ds: float = 0.1               # m, arc sampling step
    ped_speed_cap: float = 1.4    # m/s, cap inside ped_cap_radius (policy parameter)
    ped_cap_radius: float = 3.0   # m
    enabled: bool = True


class SafetyFilter:
    def __init__(self, params: SafetyParams | None = None):
        self.p = params or SafetyParams()
        self.n_interventions = 0
        self.n_calls = 0
        self._sigma = np.arange(0.0, self.p.horizon + 1e-9, self.p.ds)

    def reset(self) -> None:
        self.n_interventions = 0
        self.n_calls = 0

    # ------------------------------------------------------------------ geometry
    def _arc(self, delta: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Centre-of-footprint poses (x, y, theta) at the sampled arc lengths, body frame."""
        s = self._sigma
        kappa = np.tan(np.clip(delta, -0.6, 0.6)) / WHEELBASE
        if abs(kappa) < 1e-4:
            return s, np.zeros_like(s), np.zeros_like(s)
        th = kappa * s
        return np.sin(th) / kappa, (1.0 - np.cos(th)) / kappa, th

    def _first_overlap(self, arc, px: np.ndarray, py: np.ndarray, half_l: float, half_w: float) -> float:
        """Smallest arc length at which any point lies inside the footprint rectangle."""
        if px.size == 0:
            return np.inf
        ax, ay, th = arc
        dx = px[None, :] - ax[:, None]
        dy = py[None, :] - ay[:, None]
        c, s = np.cos(th)[:, None], np.sin(th)[:, None]
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        inside = (lx > -half_l) & (lx < half_l) & (np.abs(ly) < half_w)
        hit = inside.any(axis=1)
        # the arc is sampled every ``ds``: the true first overlap lies in (sigma - ds, sigma], so
        # subtracting one step keeps the estimate conservative
        return float(self._sigma[np.argmax(hit)]) - self.p.ds if hit.any() else np.inf

    def v_limit(self, obs: dict, delta: float) -> float:
        p = self.p
        arc = self._arc(delta)
        r = obs["lidar"]
        a = obs["lidar_angles"]
        valid = r < 9.99
        d_static = self._first_overlap(arc, (r * np.cos(a))[valid], (r * np.sin(a))[valid],
                                       ROBOT_HALF_L + p.front_margin, ROBOT_HALF_W + p.side_margin)
        d_dyn, v_cap = np.inf, np.inf
        tr = obs["tracks"]
        live = tr[:, 4] > 0.5
        if live.any():
            t = tr[live]
            d_dyn = self._first_overlap(arc, t[:, 0], t[:, 1],
                                        ROBOT_HALF_L + PED_R + p.ped_extra + p.front_margin,
                                        ROBOT_HALF_W + PED_R + p.ped_extra + p.side_margin)
            if np.any(np.hypot(t[:, 0], t[:, 1]) < p.ped_cap_radius):
                v_cap = p.ped_speed_cap
            # a pedestrian closing head-on shortens the usable distance
            closing = np.maximum(0.0, -(t[:, 0] * t[:, 2] + t[:, 1] * t[:, 3]) / np.maximum(np.hypot(t[:, 0], t[:, 1]), 1e-6))
            if np.isfinite(d_dyn):
                d_dyn = max(d_dyn - float(closing.max()) * 1.0, 0.0)
        d_free = min(d_static, d_dyn)
        if d_free <= 0.0:
            return 0.0
        a_q, b_q, c_q = 1.0 / (2.0 * p.a_brake), p.t_react, -d_free       # v t_react + v^2/(2a) = d_free
        v_safe = (-b_q + np.sqrt(b_q * b_q - 4 * a_q * c_q)) / (2 * a_q)
        return float(min(v_safe, v_cap))

    def __call__(self, obs: dict, v: float, delta: float) -> tuple[float, float]:
        self.n_calls += 1
        if not self.p.enabled:
            return v, delta
        lim = self.v_limit(obs, delta)
        if v > lim + 1e-9:
            self.n_interventions += 1
            v = lim
        return v, delta
