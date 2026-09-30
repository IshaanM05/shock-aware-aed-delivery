"""Camera rigs and time mapping for the film.

A rig turns the recorded rover motion into a desired camera path (one pose per output frame); the film
smooths that path so the camera has inertia instead of being bolted to the rover. The time map decides
which simulation step each output frame shows, which is how slow motion around the kerb strike is made.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .camera import CameraPose

Rig = Callable[..., list[CameraPose]]      # (scene, s, u) -> poses


def time_map(frames: int, fps: int, start_step: float, n_steps: int, dt: float, *, base_rate: float = 1.0,
             slow: tuple[float, float, float] | None = None) -> np.ndarray:
    """Simulation step shown by each output frame.

    ``slow = (centre_step, width_seconds, rate)`` dips the playback speed to ``rate`` (x real time) with a
    Gaussian profile around ``centre_step``; everywhere else it runs at ``base_rate``.
    """
    s = np.empty(frames)
    cur = float(start_step)
    for k in range(frames):
        s[k] = min(cur, n_steps - 1)
        v = base_rate
        if slow is not None:
            c, width, rate = slow
            dip = math.exp(-0.5 * ((cur - c) * dt / width) ** 2)
            v = base_rate * (1.0 - (1.0 - rate) * dip)
        cur += v / (dt * fps)
    return s


def smooth_poses(poses: list[CameraPose], sigma_frames: float) -> list[CameraPose]:
    """Zero-phase Gaussian smoothing of camera positions, targets and field of view."""
    if sigma_frames <= 0 or len(poses) < 3:
        return poses
    a = np.array([[*p.pos, *p.target, p.fovy] for p in poses])
    a = gaussian_filter1d(a, sigma_frames, axis=0, mode="nearest")
    return [CameraPose(tuple(r[:3]), tuple(r[3:6]), float(r[6])) for r in a]


def ease(u: np.ndarray | float) -> np.ndarray | float:
    return u * u * (3 - 2 * u)


def _rover(scene, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ps, yaws = [], []
    for si in s:
        p, yaw = scene.rover_pose(float(si))
        ps.append(p)
        yaws.append(yaw)
    return np.array(ps), np.unwrap(np.array(yaws))


def _frame(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    return f, np.array([-math.sin(yaw), math.cos(yaw), 0.0])


def rig_chase(back: float = 4.2, height: float = 1.5, swing_deg: float = 20.0, ahead: float = 2.5,
              target_z: float = 0.15, fovy: float = 50.0) -> Rig:
    def rig(scene, s, u):
        ps, yaws = _rover(scene, s)
        out = []
        for p, yaw in zip(ps, yaws, strict=True):
            a = yaw + math.radians(swing_deg)
            cam = p + np.array([-back * math.cos(a), -back * math.sin(a), height])
            tgt = p + np.array([ahead * math.cos(yaw), ahead * math.sin(yaw), target_z])
            out.append(CameraPose(tuple(cam), tuple(tgt), fovy))
        return out
    return rig


def rig_crane(off0: tuple[float, float, float], off1: tuple[float, float, float], *, fov0: float = 55.0,
              fov1: float = 48.0, ahead: float = 3.0, target_z: float = 0.3) -> Rig:
    """Camera offset from the rover in its own (forward, left, up) frame, moving from ``off0`` to ``off1``."""
    def rig(scene, s, u):
        ps, yaws = _rover(scene, s)
        out = []
        for p, yaw, uk in zip(ps, yaws, u, strict=True):
            e = ease(uk)
            o = np.array(off0) * (1 - e) + np.array(off1) * e
            f, le = _frame(yaw)
            cam = p + f * o[0] + le * o[1] + np.array([0, 0, o[2]])
            tgt = p + f * ahead + np.array([0, 0, target_z])
            out.append(CameraPose(tuple(cam), tuple(tgt), fov0 + (fov1 - fov0) * e))
        return out
    return rig


def rig_orbit(radius: float = 3.4, height: float = 1.3, az0_deg: float = 215.0, deg_per_frame: float = 0.35,
              ahead: float = 0.0, fovy: float = 42.0) -> Rig:
    def rig(scene, s, u):
        ps, yaws = _rover(scene, s)
        out = []
        for k, (p, yaw) in enumerate(zip(ps, yaws, strict=True)):
            a = yaw + math.radians(az0_deg + deg_per_frame * k)
            cam = p + np.array([radius * math.cos(a), radius * math.sin(a), height])
            tgt = p + np.array([ahead * math.cos(yaw), ahead * math.sin(yaw), 0.2])
            out.append(CameraPose(tuple(cam), tuple(tgt), fovy))
        return out
    return rig


def rig_kerb(side: float = -3.7, before: float = 2.0, height: float = 0.42, drift: float = 0.6,
             fovy: float = 44.0, edge: str = "up") -> Rig:
    """A low camera at the roadside of the kerb, panning to follow the rover through the climb."""
    def rig(scene, s, u):
        sc = scene.rec.meta["scenario"]
        xk = float(sc["x_up"] if edge == "up" else sc["x_down"])
        ps, _ = _rover(scene, s)
        out = []
        for p, uk in zip(ps, u, strict=True):
            cam = (xk - before + drift * uk, side, height)
            tgt = (p[0] + 0.15, p[1], p[2] - 0.12)
            out.append(CameraPose(cam, tgt, fovy))
        return out
    return rig


def rig_top(height: float = 24.0, back: float = 5.0, side: float = 0.0, fovy: float = 42.0) -> Rig:
    """High angle following the rover; the planner's rollouts and lidar read clearly from here."""
    def rig(scene, s, u):
        ps, yaws = _rover(scene, s)
        out = []
        for p, yaw in zip(ps, yaws, strict=True):
            f, le = _frame(yaw)
            cam = p - f * back + le * side + np.array([0, 0, height])
            out.append(CameraPose(tuple(cam), tuple(p + f * 1.5), fovy))
        return out
    return rig
