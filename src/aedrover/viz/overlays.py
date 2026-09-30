"""3D data overlays drawn inside the scene: the rover's upcoming path, lidar returns, MPPI's sampled
rollouts, and a halo that shifts from green to red with the live payload shock.

The PBR renderer cannot take per-frame extra geometry from Python, and resizing a geom per frame does
not propagate, so every overlay is a *pool of glowing beads* (small emissive spheres on mocap bodies)
that is repositioned each frame; unused beads are parked below the floor. Beads are fixed-size, so a
dense row of them reads as a continuous glowing line, and bloom does the rest. Colours come from a
small set of materials (ranks and fade levels), because material colour is what updates reliably.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .dressing import Materials, geom
from .recording import Recording
from .render_model import RenderXml

PARK = np.array([0.0, 0.0, -50.0])


@dataclass(frozen=True)
class OverlayConfig:
    ribbon: bool = True
    lidar: bool = True
    rollouts: bool = True
    halo: bool = True
    ribbon_beads: int = 80
    ribbon_spacing: float = 0.085
    ribbon_start: float = 0.65          # metres ahead of the chassis centre where the trail begins
    ribbon_radius: float = 0.03
    lidar_show_radius: float = 2.3      # where a ray that hit nothing is drawn (a faint sensing bubble)
    n_rollouts: int = 24
    halo_beads: int = 20


def _densify(xy: np.ndarray) -> np.ndarray:
    """Insert the midpoint between consecutive nodes along the second-to-last axis: (..., n, 2) -> (..., 2n - 1, 2)."""
    mid = 0.5 * (xy[..., :-1, :] + xy[..., 1:, :])
    out = np.empty(xy.shape[:-2] + (2 * xy.shape[-2] - 1, 2), dtype=xy.dtype)
    out[..., 0::2, :] = xy
    out[..., 1::2, :] = mid
    return out


def _lerp(a, b, w):
    return tuple(float(x + (y - x) * w) for x, y in zip(a, b, strict=True))


def dress_overlays(rx: RenderXml, rec: Recording, cfg: OverlayConfig, mats: Materials) -> OverlayAnimator:
    """Add the bead pools to ``rx`` and return the animator that moves them."""
    has_rollouts = bool(rec.rollouts)
    pools: dict[str, int] = {}

    def pool(name: str, n: int, radius: float, material_of) -> None:
        pools[name] = n
        for i in range(n):
            rx.world_xml.append(f'<body name="ov_{name}_{i}" mocap="true" pos="0 0 -50">'
                                + geom("sphere", (0, 0, 0), (radius,), material_of(i)) + "</body>")

    if cfg.ribbon:
        levels = 7
        for k in range(levels):
            f = k / (levels - 1)
            mats.neon(f"ov_rib_{k}", _lerp((0.15, 0.90, 1.0), (0.05, 0.35, 0.75), f), 3.0 - 1.6 * f)
        pool("rib", cfg.ribbon_beads, cfg.ribbon_radius, lambda i: f"ov_rib_{min(i * levels // cfg.ribbon_beads, levels - 1)}")
    if cfg.lidar:
        mats.neon("ov_lidar_hit", (1.0, 0.75, 0.15), 4.0)
        mats.neon("ov_lidar_ring", (0.10, 0.55, 1.0), 1.2)
        n = len(rec.lidar_angles)
        pool("lhit", n, 0.045, lambda i: "ov_lidar_hit")
        pool("lring", n, 0.02, lambda i: "ov_lidar_ring")
    if cfg.rollouts and has_rollouts:
        levels = 8
        for k in range(levels):
            f = k / (levels - 1)
            mats.neon(f"ov_roll_{k}", _lerp((0.15, 1.0, 0.55), (1.0, 0.25, 0.70), f), 2.6 - 0.9 * f)
        h1 = 2 * rec.rollouts[next(iter(rec.rollouts))]["xy"].shape[1] - 1     # nodes plus midpoints
        pool("roll", cfg.n_rollouts * h1, 0.022, lambda i: f"ov_roll_{min((i // h1) * levels // cfg.n_rollouts, levels - 1)}")
        mats.neon("ov_best", (1.0, 0.90, 0.35), 4.5)
        pool("best", h1, 0.034, lambda i: "ov_best")
    if cfg.halo:
        mats.neon("ov_halo_ok", (0.10, 1.0, 0.35), 3.0)
        mats.neon("ov_halo_mid", (1.0, 0.55, 0.05), 3.4)
        mats.neon("ov_halo_hot", (1.0, 0.07, 0.05), 4.2)
        for lvl, name in enumerate(("ok", "mid", "hot")):
            pool(f"h{lvl}", cfg.halo_beads, 0.03, lambda i, name=name: f"ov_halo_{name}")
    return OverlayAnimator(cfg, pools)


class OverlayAnimator:
    """Positions the bead pools every frame from the recording."""

    def __init__(self, cfg: OverlayConfig, pools: dict[str, int]) -> None:
        self.cfg, self.pools = cfg, pools

    # ----------------------------------------------------------------------------------- setup
    def bind(self, model: mujoco.MjModel, rec: Recording) -> None:
        self.rec = rec
        sc = rec.meta["scenario"]
        self._kerb = (float(sc["kerb_h"]), float(sc["x_down"]), float(sc["x_up"]))
        self.ids = {}
        for name, n in self.pools.items():
            self.ids[name] = np.array([int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ov_{name}_{i}")])
                                       for i in range(n)])
        xy = rec.qpos[:, :2]
        self._path = uniform_filter1d(xy, size=5, axis=0, mode="nearest")
        self._seglen = np.r_[0.0, np.linalg.norm(np.diff(self._path, axis=0), axis=1)]
        self._cum = np.cumsum(self._seglen)
        win = max(int(0.4 / rec.dt), 1)
        self._shock = maximum_filter1d(rec.shock_g, size=win, mode="nearest")
        self._roll_keys = np.array(sorted(rec.rollouts)) if rec.rollouts else np.array([], dtype=int)
        self._lidar_range = float(rec.meta.get("lidar_range", 8.0))
        self._budget = float(rec.meta.get("budget_g", 3.0))

    def _surface(self, x):
        h, xd, xu = self._kerb
        x = np.asarray(x)
        return np.where((x <= xd) | (x >= xu), h, 0.0) if h > 1e-6 else np.zeros_like(x, dtype=float)

    def _park(self, name: str) -> None:
        self._d.mocap_pos[self.ids[name], :] = PARK

    def _put(self, name: str, pts: np.ndarray) -> None:
        ids = self.ids[name]
        n = min(len(pts), len(ids))
        self._d.mocap_pos[ids[:n]] = pts[:n]
        if n < len(ids):
            self._d.mocap_pos[ids[n:]] = PARK

    # ------------------------------------------------------------------------------------ frame
    def apply(self, data: mujoco.MjData, s: float) -> None:
        self._d = data
        T = len(self.rec)
        s = float(np.clip(s, 0, T - 1))
        i0 = int(round(s))
        pos = data.qpos[:3].copy()
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        f = np.array([math.cos(yaw), math.sin(yaw)])
        if "rib" in self.ids:
            self._ribbon(pos, f, s)
        if "lhit" in self.ids:
            self._lidar(pos, yaw, i0)
        if "roll" in self.ids:
            self._rollouts(i0, pos)
        if "h0" in self.ids:
            self._halo(pos, yaw, i0, s)

    def _ribbon(self, pos: np.ndarray, f: np.ndarray, s: float) -> None:
        cfg = self.cfg
        i0 = min(int(s), len(self._cum) - 2)
        base = self._cum[i0] + (s - i0) * self._seglen[i0 + 1]
        targets = base + cfg.ribbon_start + cfg.ribbon_spacing * np.arange(cfg.ribbon_beads)
        valid = targets <= self._cum[-1]
        px = np.interp(targets[valid], self._cum, self._path[:, 0])
        py = np.interp(targets[valid], self._cum, self._path[:, 1])
        pts = np.stack([px, py, self._surface(px) + 0.03], -1)
        self._put("rib", pts)

    def _lidar(self, pos: np.ndarray, yaw: float, i0: int) -> None:
        rng_m = self.rec.lidar[i0]
        ang = yaw + self.rec.lidar_angles
        hit = rng_m < self._lidar_range - 1e-3
        r_show = np.where(hit, rng_m, self.cfg.lidar_show_radius)
        r_show = np.minimum(r_show, self.cfg.lidar_show_radius * 2.0)
        ox, oy = pos[0] + 0.45 * math.cos(yaw), pos[1] + 0.45 * math.sin(yaw)
        px, py = ox + r_show * np.cos(ang), oy + r_show * np.sin(ang)
        pts = np.stack([px, py, np.full_like(px, pos[2] - 0.26)], -1)
        pts[:, 2] = self._surface(px) + 0.15
        hit_pts = np.where(hit[:, None], pts, PARK[None])
        ring_pts = np.where(hit[:, None], PARK[None], pts)
        self._d.mocap_pos[self.ids["lhit"]] = hit_pts
        self._d.mocap_pos[self.ids["lring"]] = ring_pts

    def _rollouts(self, i0: int, pos: np.ndarray) -> None:
        k = np.searchsorted(self._roll_keys, i0, side="right") - 1
        if k < 0 or i0 - self._roll_keys[k] > 12:
            self._park("roll")
            self._park("best")
            return
        r = self.rec.rollouts[int(self._roll_keys[k])]
        xy = _densify(r["xy"])                                # (K, 2H+1, 2)
        z = self._surface(xy[..., 0]) + 0.04
        self._put("roll", np.concatenate([xy, z[..., None]], -1).reshape(-1, 3))
        b = _densify(r["best"][None])[0]
        self._put("best", np.stack([b[:, 0], b[:, 1], self._surface(b[:, 0]) + 0.06], -1))

    def _halo(self, pos: np.ndarray, yaw: float, i0: int, s: float) -> None:
        frac = float(self._shock[i0]) / self._budget
        level = 0 if frac < 0.45 else (1 if frac < 0.8 else 2)
        n = self.cfg.halo_beads
        t = np.arange(n) / n * 2 * math.pi
        pulse = 1.0 + 0.04 * math.sin(s * 0.5)
        lx, ly = 0.80 * pulse * np.cos(t), 0.56 * pulse * np.sin(t)
        c, sn = math.cos(yaw), math.sin(yaw)
        px, py = pos[0] + c * lx - sn * ly, pos[1] + sn * lx + c * ly
        pts = np.stack([px, py, self._surface(px) + 0.035], -1)
        for lvl in range(3):
            if lvl == level:
                self._d.mocap_pos[self.ids[f"h{lvl}"]] = pts
            else:
                self._park(f"h{lvl}")
