"""Camera poses and small helpers shared by the renderers and the shot designer."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else v


@dataclass(frozen=True)
class CameraPose:
    """A look-at camera: position, target point and vertical field of view (degrees)."""

    pos: tuple[float, float, float]
    target: tuple[float, float, float]
    fovy: float = 50.0

    @property
    def forward(self) -> np.ndarray:
        return _unit(np.asarray(self.target, float) - np.asarray(self.pos, float))

    @property
    def up(self) -> np.ndarray:
        f = self.forward
        if abs(float(f @ WORLD_UP)) > 0.999:           # looking straight up or down
            return np.array([1.0, 0.0, 0.0])
        right = _unit(np.cross(f, WORLD_UP))
        return np.cross(right, f)

    def lerp(self, other: CameraPose, w: float) -> CameraPose:
        a, b = np.asarray(self.pos, float), np.asarray(other.pos, float)
        ta, tb = np.asarray(self.target, float), np.asarray(other.target, float)
        return CameraPose(tuple(a + (b - a) * w), tuple(ta + (tb - ta) * w), self.fovy + (other.fovy - self.fovy) * w)

    def to_gl(self, near: float = 0.05, far: float = 400.0) -> mujoco.MjvGLCamera:
        """Camera in the form the PBR renderer takes (the horizontal extent follows the viewport)."""
        gl = mujoco.MjvGLCamera()
        gl.pos[:] = self.pos
        gl.forward[:] = self.forward
        gl.up[:] = self.up
        top = near * np.tan(np.radians(self.fovy) / 2.0)
        gl.frustum_near, gl.frustum_far = near, far
        gl.frustum_top, gl.frustum_bottom = top, -top
        return gl

    def to_free(self) -> mujoco.MjvCamera:
        """The same viewpoint as a free ``MjvCamera`` (used by the classic renderer and by decorations)."""
        f = self.forward
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = self.target
        cam.distance = float(np.linalg.norm(np.asarray(self.target, float) - np.asarray(self.pos, float)))
        cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
        cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1.0, 1.0))))
        return cam


def chase(pos, yaw: float, *, back: float = 4.2, height: float = 1.5, swing_deg: float = 20.0, ahead: float = 2.5,
          target_z: float = 0.15, fovy: float = 50.0) -> CameraPose:
    """A camera behind the rover (``swing_deg`` off its axis) looking ahead of it."""
    p = np.asarray(pos, float)
    a = yaw + np.radians(swing_deg)
    cam = p + np.array([-back * np.cos(a), -back * np.sin(a), height])
    tgt = p + np.array([ahead * np.cos(yaw), ahead * np.sin(yaw), target_z])
    return CameraPose(tuple(cam), tuple(tgt), fovy)
