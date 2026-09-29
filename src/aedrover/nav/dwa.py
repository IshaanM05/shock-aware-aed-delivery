"""Baseline C: Dynamic Window Approach local planner (Fox, Burgard, Thrun 1997, key fox1997).

Samples constant (speed, steering) pairs inside the kinematic dynamic window, rolls each out on
a bicycle model over a short horizon, rejects trajectories that cannot stop before contact, and
scores the rest on progress, lane centring, clearance to lidar points and constant-velocity
pedestrian predictions, and speed. Fully vectorised (about 0.3 ms per plan).

This is the strong classical baseline: unlike a potential field it plans against predicted
motion and cannot be trapped by symmetric obstacle forces. The kerb negotiator is an overlay
that caps the speed on kerb approach.
"""

from __future__ import annotations

import numpy as np

from ..control.curb import CurbNegotiator
from ..sim.pedestrians import PED_R
from .base import register

L = 0.62


class DWAController:
    def __init__(self, v_cruise: float = 1.8, horizon: float = 2.4, dt: float = 0.15,
                 n_v: int = 9, n_delta: int = 15, a_max: float = 1.6, a_brake: float = 2.0,
                 lane_y: float = 1.2, w_progress: float = 1.0, w_center: float = 0.5,
                 w_speed: float = 1.0, w_clear: float = 1.2, w_head: float = 0.2, margin: float = 0.06, ped_inflate: float = 0.12,
                 use_curb: bool = True, curb: CurbNegotiator | None = None, replan_every: int = 2,
                 name: str = "dwa"):
        self.name = name
        self.v_cruise, self.T, self.dt = v_cruise, horizon, dt
        self.a_max, self.a_brake = a_max, a_brake
        self.lane_y, self.margin, self.ped_inflate = lane_y, margin, ped_inflate
        self.w = (w_progress, w_center, w_speed, w_clear, w_head)
        self.n_steps = int(round(horizon / dt))
        self.deltas = np.linspace(-0.45, 0.45, n_delta)
        self.n_v = n_v
        self.curb = (curb or CurbNegotiator()) if use_curb else None
        self.replan_every = replan_every
        self._k = 0
        self._cmd = (0.0, 0.0)
        self._still = 0
        self.last: dict = {}
        # footprint approximated by 3 circles along the axis
        self._circ_x = np.array([-0.16, 0.0, 0.16])   # 3 circles, r = 0.37: covers the 1.04 x 0.72 m footprint
        self._circ_r = 0.37

    def reset(self, env) -> None:
        self._k = 0
        self._cmd = (0.0, 0.0)
        self._still = 0
        if self.curb is not None:
            self.curb.reset()
            self.curb.set_vehicle(env.veh0)

    def act(self, obs: dict) -> tuple[float, float]:
        if self._k % self.replan_every == 0:
            self._cmd = self._plan(obs)
        self._k += 1
        return self._cmd

    # ------------------------------------------------------------------
    def _plan(self, obs: dict) -> tuple[float, float]:
        vmax = self.v_cruise
        if self.curb is not None:
            vmax = min(vmax, self.curb.speed_limit(obs))
        v0 = max(obs["vx"], 0.0)
        lo, hi = max(0.0, v0 - self.a_brake * self.T * 0.5), min(vmax, v0 + self.a_max * self.T * 0.5)
        if hi < lo:
            lo = hi
        vs = np.linspace(lo, hi, self.n_v)
        V, D = np.meshgrid(vs, self.deltas, indexing="ij")
        V, D = V.ravel(), D.ravel()
        S, N = V.size, self.n_steps
        t = (np.arange(1, N + 1) * self.dt)[None, :]
        # bicycle rollout in the world frame (constant command)
        yaw0, x0, y0 = obs["yaw"], obs["x"], obs["y"]
        w = V * np.tan(D) / L
        th = yaw0 + w[:, None] * t
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = np.where(np.abs(w)[:, None] > 1e-4,
                          x0 + (V / w)[:, None] * (np.sin(th) - np.sin(yaw0)), x0 + V[:, None] * np.cos(yaw0) * t)
            ys = np.where(np.abs(w)[:, None] > 1e-4,
                          y0 - (V / w)[:, None] * (np.cos(th) - np.cos(yaw0)), y0 + V[:, None] * np.sin(yaw0) * t)
        # ---- clearance --------------------------------------------------------------------
        c, s = np.cos(yaw0), np.sin(yaw0)
        rng = obs["lidar"]
        keep = rng < 6.0
        if keep.any():
            a = yaw0 + obs["lidar_angles"][keep]
            px, py = x0 + rng[keep] * np.cos(a), y0 + rng[keep] * np.sin(a)
        else:
            px = py = np.zeros(0)
        clear = np.full((S, N), np.inf)
        ccx = np.cos(th)[..., None] * self._circ_x
        ccy = np.sin(th)[..., None] * self._circ_x
        cx, cy = xs[..., None] + ccx, ys[..., None] + ccy                    # (S, N, 3)
        if px.size:
            d = np.sqrt((cx[..., None] - px) ** 2 + (cy[..., None] - py) ** 2).min(axis=(2, 3)) - self._circ_r
            clear = np.minimum(clear, d)
        for tx, ty, vx, vy, valid in obs["tracks"]:
            if valid < 0.5:
                continue
            wx, wy = x0 + c * tx - s * ty, y0 + s * tx + c * ty                # ped world position
            wvx, wvy = c * vx - s * vy, s * vx + c * vy
            pxt, pyt = wx + wvx * t, wy + wvy * t                               # (1, N)
            d = np.sqrt((cx - pxt[..., None]) ** 2 + (cy - pyt[..., None]) ** 2).min(axis=2) - self._circ_r - PED_R - self.ped_inflate
            clear = np.minimum(clear, d)
        min_clear = clear.min(axis=1)
        # admissible: can stop before actual contact (the comfort margin lives in the score, not here)
        first_hit = np.where((clear < self.margin).any(axis=1), (clear < self.margin).argmax(axis=1), N)
        dist_free = V * first_hit * self.dt
        admissible = V <= np.sqrt(2.0 * self.a_brake * np.maximum(dist_free, 0.0)) + 0.05
        admissible |= first_hit >= N
        # ---- score ------------------------------------------------------------------------
        wp, wc, ws, wcl, wh = self.w
        progress = (xs[:, -1] - x0) / (self.v_cruise * self.T)
        center = -np.abs(ys[:, -1]) / self.lane_y - 2.0 * np.maximum(np.abs(ys).max(axis=1) - self.lane_y, 0.0)
        speed = V / self.v_cruise
        head = -np.abs(np.arctan2(np.sin(th[:, -1]), np.cos(th[:, -1])))
        clear_term = np.minimum(min_clear, 1.0)
        score = wp * progress + wc * center + ws * speed + wcl * clear_term + wh * head
        score = np.where(admissible & (min_clear > 0.0), score, -1e9)
        # local-minimum escape: after standing still for ~1.5 s, force a moving candidate if one exists
        self._still = self._still + 1 if v0 < 0.05 else 0
        if self._still * self.dt * self.replan_every / 0.15 > 1.5:
            moving = np.where(V >= 0.25, score, -1e9)
            if moving.max() > -1e8:
                score = moving
        self.last = {"n_adm": int((score > -1e8).sum()), "n_total": int(S), "vmax": float(vmax), "lo": float(lo), "hi": float(hi),
                     "min_clear_now": float(min_clear.max())}
        if not np.isfinite(score).any() or score.max() < -1e8:
            return 0.0, float(np.clip(-1.0 * obs["yaw"], -0.45, 0.45))         # nothing admissible: stop, steer straight
        i = int(np.argmax(score))
        return float(V[i]), float(D[i])


@register("dwa")
def _make_dwa(**kw):
    return DWAController(**kw)


@register("dwa_nocurb")
def _make_dwa_nocurb(**kw):
    return DWAController(use_curb=False, name="dwa_nocurb", **kw)
