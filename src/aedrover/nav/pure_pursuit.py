"""Baseline A: pure-pursuit path following at constant cruise speed (no avoidance, no kerb logic).

This is deliberately the naive controller. With the safety filter it cannot collide but it may
stall or shock the payload at kerbs; it anchors the comparison.
"""

from __future__ import annotations

import numpy as np

from .base import register

WHEELBASE = 0.62


class PurePursuit:
    name = "pure_pursuit"

    def __init__(self, v_cruise: float = 1.4, lookahead: float = 1.2, y_ref: float = 0.0):
        self.v_cruise, self.ld, self.y_ref = v_cruise, lookahead, y_ref

    def reset(self, env) -> None:
        pass

    def act(self, obs: dict) -> tuple[float, float]:
        dx, dy = self.ld, self.y_ref - obs["y"]
        alpha = np.arctan2(dy, dx) - obs["yaw"]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        delta = np.arctan2(2.0 * WHEELBASE * np.sin(alpha), self.ld)
        return self.v_cruise, float(np.clip(delta, -0.5, 0.5))


@register("pure_pursuit")
def _make(**kw):
    return PurePursuit(**kw)
