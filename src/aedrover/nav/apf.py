"""Baseline B: artificial potential field local planner (Khatib 1986, key khatib1986).

Attractive term toward the path, repulsive terms from lidar returns within the influence radius,
from tracked pedestrians and from the sidewalk edges (virtual walls near +/- ``wall_y``). The
kerb negotiator is an optional overlay that limits the speed on kerb approach.
"""

from __future__ import annotations

import numpy as np

from ..control.curb import CurbNegotiator
from .base import register

WHEELBASE = 0.62


class APFController:
    def __init__(self, v_cruise: float = 1.6, rho0: float = 2.5, eta: float = 0.9, k_att: float = 1.0,
                 wall_y: float = 1.25, wall_gain: float = 2.0, lookahead: float = 1.5,
                 use_curb: bool = True, curb: CurbNegotiator | None = None, name: str = "apf"):
        self.name = name
        self.v_cruise, self.rho0, self.eta, self.k_att = v_cruise, rho0, eta, k_att
        self.wall_y, self.wall_gain, self.ld = wall_y, wall_gain, lookahead
        self.curb = (curb or CurbNegotiator()) if use_curb else None

    def reset(self, env) -> None:
        if self.curb is not None:
            self.curb.reset()

    def _force(self, obs: dict) -> np.ndarray:
        yaw = obs["yaw"]
        c, s = np.cos(yaw), np.sin(yaw)
        f = self.k_att * np.array([1.0, -0.6 * obs["y"] / max(self.ld, 0.5)])   # world frame
        ang = yaw + obs["lidar_angles"]
        rng = obs["lidar"]
        near = rng < self.rho0
        if near.any():
            rho = np.maximum(rng[near], 0.3)
            mag = self.eta * (1.0 / rho - 1.0 / self.rho0) / rho**2
            f -= np.array([np.sum(mag * np.cos(ang[near])), np.sum(mag * np.sin(ang[near]))]) \
                / max(int(near.sum()), 1) * 3.0
        for x, y, _vx, _vy, valid in obs["tracks"]:
            if valid < 0.5:
                continue
            px, py = c * x - s * y, s * x + c * y            # pedestrian position, world-aligned
            d = max(np.hypot(px, py), 0.4)
            if d < self.rho0:
                mag = self.eta * (1.0 / d - 1.0 / self.rho0) / d**2
                f -= mag * np.array([px, py]) / d * 1.5
        yw = obs["y"]
        if abs(yw) > self.wall_y - 0.6:
            f[1] -= np.sign(yw) * self.wall_gain * (abs(yw) - (self.wall_y - 0.6))
        return f

    def act(self, obs: dict) -> tuple[float, float]:
        f = self._force(obs)
        theta_d = np.arctan2(f[1], max(f[0], 0.05))
        err = (theta_d - obs["yaw"] + np.pi) % (2 * np.pi) - np.pi
        delta = float(np.clip(0.9 * np.arctan2(2.0 * WHEELBASE * np.sin(err), self.ld), -0.5, 0.5))
        v = self.v_cruise
        if self.curb is not None:
            v = min(v, self.curb.speed_limit(obs))
        v *= max(0.35, 1.0 - 0.6 * min(abs(err), 1.0))       # slow down when turning hard
        return float(max(v, 0.0)), delta


@register("apf")
def _make_apf(**kw):
    return APFController(**kw)


@register("apf_nocurb")
def _make_apf_nocurb(**kw):
    return APFController(use_curb=False, name="apf_nocurb", **kw)
