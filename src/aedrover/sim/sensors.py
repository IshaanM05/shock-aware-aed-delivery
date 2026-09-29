"""Perception: 2D lidar, forward terrain-height scan and a noisy dynamic-obstacle tracker.

All controllers (classical, MPPI, PPO) consume the same observations, so comparisons are fair.
Rays are batched with ``mj_multiRay`` from a single origin (lidar) or one call per lateral
offset (height scan), which keeps perception around 30 us per control step.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# geom groups: 0 world (slabs, road, obstacles), 1 pedestrians, 2 robot (excluded from rays)
_GROUP_MASK = np.array([1, 1, 0, 0, 0, 0], dtype=np.uint8)

# ``mj_multiRay(cutoff=...)`` pre-filters on the distance from the ray origin to each geom's
# *centre*. An infinite plane has its centre at the world origin (rbound 0), so a small cutoff
# silently drops the road once the rover is a few metres from the origin. Use a huge cutoff and
# clip the returned distances ourselves. (Regression test: tests/test_sensors.py.)
_NO_CUTOFF = 1.0e4


@dataclass(frozen=True)
class PerceptionSpec:
    n_lidar: int = 73
    fov_deg: float = 240.0
    lidar_range: float = 10.0
    lidar_height: float = 0.12               # above the chassis centre
    scan_fwd: tuple[float, ...] = (0.35, 0.7, 1.1, 1.6, 2.3, 3.2, 4.5)
    scan_lat: tuple[float, ...] = (-0.5, 0.0, 0.5)
    scan_mast_height: float = 1.2            # ray origin above the rover's support plane
    track_pos_sigma: float = 0.05            # m (detector/tracker abstraction)
    track_vel_sigma: float = 0.10            # m/s
    track_range: float = 10.0


class Perception:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, chassis_body: int,
                 nominal_height: float, spec: PerceptionSpec | None = None):
        self.m, self.d = model, data
        self.body = chassis_body
        self.h0 = nominal_height
        self.s = spec or PerceptionSpec()
        s = self.s
        self.angles = np.radians(np.linspace(-s.fov_deg / 2, s.fov_deg / 2, s.n_lidar))
        self._ldir = np.zeros((s.n_lidar, 3))
        self._lgeom = np.zeros(s.n_lidar, dtype=np.int32)
        self._ldist = np.zeros(s.n_lidar)
        nf, nl = len(s.scan_fwd), len(s.scan_lat)
        self._nf, self._nl = nf, nl
        fwd, lat = np.meshgrid(np.array(s.scan_fwd), np.array(s.scan_lat), indexing="ij")
        self._F, self._L = fwd.ravel(), lat.ravel()
        self._sgeom = np.zeros(nf * nl, dtype=np.int32)
        self._sdist = np.zeros(nf * nl)

    # ------------------------------------------------------------------ lidar
    def lidar(self, yaw: float) -> np.ndarray:
        s = self.s
        ang = yaw + self.angles
        self._ldir[:, 0] = np.cos(ang)
        self._ldir[:, 1] = np.sin(ang)
        origin = self.d.xpos[self.body].copy()
        origin[2] += s.lidar_height
        self._lgeom[:] = -1
        self._ldist[:] = -1.0
        mujoco.mj_multiRay(self.m, self.d, origin, self._ldir.ravel(), _GROUP_MASK, 1, self.body,
                           self._lgeom, self._ldist, None, s.n_lidar, _NO_CUTOFF)
        return np.where((self._ldist < 0.0) | (self._ldist > s.lidar_range), s.lidar_range, self._ldist)

    # ---------------------------------------------------------- terrain scan
    def height_scan(self, yaw: float) -> np.ndarray:
        """Surface height ahead relative to the rover's support plane, shape (n_fwd, n_lat).

        One batched ray cast: rays fan out from a mast point above the rover toward a grid of
        target points on the support plane; the height of the first surface hit is reported
        (-1.0 where nothing is hit, e.g. a void).
        """
        s = self.s
        c, sn = np.cos(yaw), np.sin(yaw)
        base = self.d.xpos[self.body]
        support_z = base[2] - self.h0
        origin = np.array([base[0], base[1], support_z + s.scan_mast_height])
        vec = np.empty((len(self._F), 3))
        vec[:, 0] = c * self._F - sn * self._L
        vec[:, 1] = sn * self._F + c * self._L
        vec[:, 2] = -s.scan_mast_height
        vec /= np.linalg.norm(vec, axis=1, keepdims=True)
        self._sgeom[:] = -1
        self._sdist[:] = -1.0
        mujoco.mj_multiRay(self.m, self.d, origin, vec.ravel(), _GROUP_MASK, 1, self.body,
                           self._sgeom, self._sdist, None, len(self._F), _NO_CUTOFF)
        hit = (self._sdist >= 0.0) & (self._sdist < 8.0)
        rel = np.where(hit, origin[2] + vec[:, 2] * self._sdist - support_z, -1.0)
        return rel.reshape(self._nf, self._nl)

    # --------------------------------------------------------------- tracker
    def track(self, robot_xy: np.ndarray, yaw: float, ped_pos: np.ndarray, ped_vel: np.ndarray,
              rng: np.random.Generator, k: int = 4) -> np.ndarray:
        """Noisy nearest-``k`` pedestrian tracks in the body frame: rows (x, y, vx, vy, valid)."""
        out = np.zeros((k, 5))
        if len(ped_pos) == 0:
            return out
        rel = ped_pos - robot_xy
        dist = np.linalg.norm(rel, axis=1)
        idx = np.argsort(dist)[:k]
        idx = idx[dist[idx] < self.s.track_range]
        c, sn = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, sn], [-sn, c]])   # world -> body
        for r, i in enumerate(idx):
            p_b = rot @ rel[i] + self.s.track_pos_sigma * rng.standard_normal(2)
            v_b = rot @ ped_vel[i] + self.s.track_vel_sigma * rng.standard_normal(2)
            out[r] = (p_b[0], p_b[1], v_b[0], v_b[1], 1.0)
        return out
