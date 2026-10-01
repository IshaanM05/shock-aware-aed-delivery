"""Vectorised social-force pedestrian crowd (Helbing & Molnar 1995, key helbing1995).

Each pedestrian is part of a stream: it walks its sidewalk piece end to end and respawns at the origin end on arrival. Forces: driving toward the
current goal, exponential repulsion from other pedestrians, from the sidewalk edges and from
the rover (aware pedestrians only; distracted ones ignore it).

The functional form follows Helbing & Molnar (1995). The numeric strengths and ranges below
are ASSUMPTIONS chosen to give plausible passing behaviour; they are exposed in ``SFMParams``
and varied in the robustness study, not claimed as fitted to any dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scenario import SIDEWALK_HALF_WIDTH, PedSpec

PED_R = 0.22
ROBOT_R = 0.50            # bounding radius used by pedestrian repulsion
ROBOT_HALF_L = 0.52       # m, footprint half-length (chassis + wheels)
ROBOT_HALF_W = 0.36       # m, footprint half-width (outer wheel faces)


@dataclass(frozen=True)
class SFMParams:
    tau: float = 0.5            # s, relaxation time
    a_ped: float = 2.0          # m/s^2, ped-ped repulsion strength (ASSUMPTION)
    b_ped: float = 0.30         # m, ped-ped range (ASSUMPTION)
    a_robot: float = 4.0        # m/s^2, ped-robot repulsion for aware pedestrians (ASSUMPTION)
    b_robot: float = 0.6        # m (ASSUMPTION)
    a_last: float = 8.0         # m/s^2, last-moment avoidance of ANY pedestrian type (ASSUMPTION)
    b_last: float = 0.25        # m, short range: distracted people still sidestep near contact
    a_wall: float = 3.0         # m/s^2 (ASSUMPTION)
    b_wall: float = 0.25        # m (ASSUMPTION)
    v_cap: float = 1.3          # x v_des speed cap (fast walking tops out near 2 m/s)
    wall_margin: float = 0.30   # m, distance of the soft wall from the sidewalk edge


class PedestrianCrowd:
    def __init__(self, params: SFMParams | None = None):
        self.p = params or SFMParams()
        self.n = 0
        self.pos = np.zeros((0, 2))
        self.vel = np.zeros((0, 2))
        self._a = np.zeros((0, 2))
        self._b = np.zeros((0, 2))
        self._origin = np.zeros((0, 2))
        self.v_des = np.zeros(0)
        self.aware = np.zeros(0, dtype=bool)
        self.z_surface = np.zeros(0)
        # Rich-world furniture as ``(M, 5)`` boxes ``(x, y, yaw, hx, hy)``: pedestrians walk round them. ``None`` in the
        # standard world, where none of the code below runs, so benchmark episodes are untouched.
        self.solids: np.ndarray | None = None

    def reset(self, specs: list[PedSpec], rng: np.random.Generator) -> None:
        self.n = len(specs)
        self._a = np.array([s.start for s in specs], dtype=float).reshape(-1, 2)
        self._b = np.array([s.goal for s in specs], dtype=float).reshape(-1, 2)
        self._origin = np.array([s.origin for s in specs], dtype=float).reshape(-1, 2)
        self.pos = self._a.copy()
        self.v_des = np.array([s.v_des for s in specs], dtype=float)
        self.aware = np.array([s.aware for s in specs], dtype=bool)
        self.z_surface = np.array([s.z_surface for s in specs], dtype=float)
        direction = self._b - self._a
        norm = np.linalg.norm(direction, axis=1, keepdims=True)
        self.vel = self.v_des[:, None] * direction / np.maximum(norm, 1e-6)
        # small random heading jitter so identical specs do not stay perfectly aligned
        self.vel += 0.05 * rng.standard_normal(self.vel.shape)

    def step(self, dt: float, robot_xy: np.ndarray, robot_vel_xy: np.ndarray) -> None:
        if self.n == 0:
            return
        p = self.p
        to_goal = self._b - self.pos
        dist = np.linalg.norm(to_goal, axis=1)
        arrived = dist < 0.6
        if arrived.any():                       # stream: leave and re-enter at the origin end
            # never materialise a pedestrian next to the rover: wait at the goal until the entry is clear
            clear = np.linalg.norm(self._origin - robot_xy, axis=1) > 6.0
            arrived = arrived & clear
            hold = (dist < 0.6) & ~clear
            self.vel[hold] = 0.0
            self.pos[arrived] = self._origin[arrived]
            self.vel[arrived] *= 0.5
            to_goal = self._b - self.pos
            dist = np.linalg.norm(to_goal, axis=1)
        e = to_goal / np.maximum(dist, 1e-6)[:, None]
        force = (self.v_des[:, None] * e - self.vel) / p.tau

        if self.n > 1:
            diff = self.pos[:, None, :] - self.pos[None, :, :]
            d = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(d, np.inf)
            mag = p.a_ped * np.exp((2 * PED_R - d) / p.b_ped)
            force += np.sum(mag[:, :, None] * diff / np.maximum(d, 1e-6)[:, :, None], axis=1)

        rdiff = self.pos - robot_xy[None, :]
        rd = np.linalg.norm(rdiff, axis=1)
        rmag = p.a_robot * np.exp((PED_R + ROBOT_R - rd) / p.b_robot) * self.aware
        rmag = rmag + p.a_last * np.exp((PED_R + ROBOT_R - rd) / p.b_last)   # everyone, at contact range
        force += rmag[:, None] * rdiff / np.maximum(rd, 1e-6)[:, None]

        wall_y = SIDEWALK_HALF_WIDTH - p.wall_margin
        force[:, 1] += p.a_wall * np.exp((-(wall_y + self.pos[:, 1]) + PED_R) / p.b_wall)
        force[:, 1] -= p.a_wall * np.exp((-(wall_y - self.pos[:, 1]) + PED_R) / p.b_wall)

        if self.solids is not None:
            force += self._solid_force(e)
        self.vel += dt * force
        if 'hold' in locals() and hold.any():
            self.vel[hold] = 0.0
        speed = np.linalg.norm(self.vel, axis=1)
        cap = p.v_cap * self.v_des
        scale = np.minimum(1.0, cap / np.maximum(speed, 1e-9))
        self.vel *= scale[:, None]
        self.pos += dt * self.vel
        np.clip(self.pos[:, 1], -SIDEWALK_HALF_WIDTH + PED_R, SIDEWALK_HALF_WIDTH - PED_R, out=self.pos[:, 1])
        if self.solids is not None:
            self._push_out_of_solids()
        _ = robot_vel_xy  # reserved for a velocity-dependent robot repulsion term


    # -------------------------------------------------------------------- rich-world furniture
    def _solid_geometry(self):
        """For every pedestrian and box: the world-frame unit vector pointing from the box to the pedestrian, its distance
        to the box (0 when inside) and how deep inside it is (0 when outside); arrays of shape ``(n, M)``."""
        bx, by, byaw, bhx, bhy = self.solids.T
        c, s = np.cos(byaw), np.sin(byaw)
        dx, dy = self.pos[:, None, 0] - bx[None], self.pos[:, None, 1] - by[None]
        lx, ly = c * dx + s * dy, -s * dx + c * dy                       # the pedestrian in each box's frame
        qx, qy = np.clip(lx, -bhx, bhx), np.clip(ly, -bhy, bhy)          # the closest point of the box
        vx, vy = lx - qx, ly - qy
        d = np.hypot(vx, vy)
        inside = d < 1e-9
        px, py = bhx - np.abs(lx), bhy - np.abs(ly)                      # penetration along each axis when inside
        along_x = px < py
        vx = np.where(inside, np.where(along_x, np.where(lx >= 0, 1.0, -1.0), 0.0), vx)
        vy = np.where(inside, np.where(along_x, 0.0, np.where(ly >= 0, 1.0, -1.0)), vy)
        depth = np.where(inside, np.minimum(px, py), 0.0)
        norm = np.maximum(np.hypot(vx, vy), 1e-9)
        ux, uy = vx / norm, vy / norm
        return c * ux - s * uy, s * ux + c * uy, d, depth

    def _solid_force(self, e_goal: np.ndarray) -> np.ndarray:
        """Repulsion from every box plus a sidestep along its surface, toward the way the pedestrian is heading and,
        failing that, toward the side it already leans to; without the sidestep a pedestrian walking straight at a
        box stalls against it (the usual social-force local minimum)."""
        wx, wy, d, _ = self._solid_geometry()
        mag = 3.0 * self.p.a_wall * np.exp((PED_R - d) / self.p.b_wall)
        centre = self.pos[:, None, :] - self.solids[None, :, :2]
        lean = centre / np.maximum(np.linalg.norm(centre, axis=2, keepdims=True), 1e-9)
        toward = e_goal[:, None, :] + 0.5 * lean
        tx, ty = -wy, wx                                                 # a unit tangent; its opposite is the other way round
        flip = np.where(tx * toward[..., 0] + ty * toward[..., 1] >= 0.0, 1.0, -1.0)
        return np.stack([np.sum(mag * (wx + 0.8 * flip * tx), axis=1), np.sum(mag * (wy + 0.8 * flip * ty), axis=1)], axis=1)

    def _push_out_of_solids(self) -> None:
        """Hard constraint behind the soft repulsion: nobody ends a step overlapping a box."""
        wx, wy, d, depth = self._solid_geometry()
        push = np.maximum(PED_R + depth - d, 0.0) * (d < PED_R + 1e-9)
        self.pos[:, 0] += np.sum(push * wx, axis=1)
        self.pos[:, 1] += np.sum(push * wy, axis=1)
