"""The AED quadrotor as visual parts, posed every frame from a ``DroneRecording``.

The drone is built from the same dimensions as the simulated vehicle (``QuadParams``: 0.45 m arms, 0.25 m
rotors, a 0.14 m half-width hull). One mocap body carries the airframe, the AED case and the skids; four more
carry the rotors so they can turn; two parked-or-shown glowing spheres are the status light (amber in flight,
green once the mission has released), because material colour is not updated per frame by this renderer.
Nothing here is simulated. The airframe pose is the recorded pose, moved into the scene by a rigid
``Placement``; rotor angles are the integral of a cosmetic fraction of the recorded rotor speed (a real rotor at
about 40 rev/s would strobe at 60 frames per second).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from ..drone.quadrotor_mjcf import YAW_SIGNS, QuadParams, quat_to_rot
from .dressing import Materials, geom
from .drone_recording import DroneRecording
from .people import _quat_z_to
from .render_model import RenderXml

PARK_Z = -50.0
SPIN_FRACTION = 0.05            # drawn rotor speed as a fraction of the simulated one (cosmetic, see module docstring)
MOUNT_Z = 0.045                 # rotor hub height above the centre of mass [m]
LED_OFFSET = np.array([0.0, 0.0, 0.098])
CASE_HALF = (0.16, 0.12, 0.075)
CASE_Z = -0.135
SKID_Z = -0.215
BODY = "dr_body"
ROTOR = "dr_rotor{k}"
LEDS = ("dr_led_amber", "dr_led_green")


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qz(angle: float) -> np.ndarray:
    return np.array([math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)])


@dataclass(frozen=True)
class Placement:
    """Rigid transform from the drone's own frame (flight along +x from the origin) into the scene."""

    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    yaw: float = 0.0

    @property
    def quat(self) -> np.ndarray:
        return _qz(self.yaw)

    def point(self, p: np.ndarray) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        p = np.asarray(p, float)
        return np.array([c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]) + np.asarray(self.origin, float)

    @classmethod
    def arriving_at(cls, goal_xy: tuple[float, float], distance_m: float, yaw: float = 0.0) -> Placement:
        """Place the flight so that its release point (``distance_m`` along +x) lands on ``goal_xy`` in the scene."""
        c, s = math.cos(yaw), math.sin(yaw)
        return cls((goal_xy[0] - c * distance_m, goal_xy[1] - s * distance_m, 0.0), yaw)


@dataclass(frozen=True)
class DroneSpec:
    """What ``dress_scene`` needs to draw the drone: its recording and where the flight sits in the scene."""

    rec: DroneRecording
    placement: Placement
    quad: QuadParams = QuadParams()


def _capsule_between(a, b, radius: float, mat: str) -> str:
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    length = float(np.linalg.norm(d))
    return geom("capsule", tuple((a + b) / 2), (radius, max(length / 2 - radius, 1e-3)), mat, quat=tuple(_quat_z_to(d)))


def dress_drone(rx: RenderXml, mats: Materials, spec: DroneSpec) -> DroneAnimator:
    """Append the drone's mocap bodies to ``rx`` and return the animator that poses them."""
    quad = spec.quad
    white = mats("dr_white", (0.93, 0.94, 0.96, 1.0), metallic=0.05, roughness=0.28)
    carbon = mats("dr_carbon", (0.045, 0.047, 0.055, 1.0), metallic=0.3, roughness=0.45)
    metal = mats("dr_metal", (0.55, 0.56, 0.6, 1.0), metallic=0.9, roughness=0.3)
    arm_mat = mats("dr_arm", (0.80, 0.81, 0.84, 1.0), metallic=0.15, roughness=0.35)
    glass = mats("dr_glass", (0.02, 0.03, 0.05, 1.0), metallic=0.0, roughness=0.05)
    green = mats("dr_case", (0.04, 0.50, 0.27, 1.0), metallic=0.0, roughness=0.5)
    cross = mats("dr_cross", (0.96, 0.96, 0.94, 1.0), roughness=0.5)
    blade = mats("dr_blade", (0.06, 0.06, 0.07, 1.0), roughness=0.5)
    tip_front = mats("dr_tip_front", (0.97, 0.52, 0.06, 1.0), roughness=0.45)
    tip_rear = mats("dr_tip_rear", (0.90, 0.90, 0.88, 1.0), roughness=0.45)
    amber = mats.neon("dr_led_amber", (1.0, 0.62, 0.08), 6.0)
    lime = mats.neon("dr_led_green", (0.15, 1.0, 0.35), 6.0)

    mounts = np.array([[x, y, MOUNT_Z] for x, y in quad.rotor_xy()])
    hx, hy, hz = quad.hull_half_m
    parts = [geom("box", (0, 0, 0), (hx - 0.01, hy - 0.01, hz - 0.01), carbon),
             geom("ellipsoid", (0, 0, 0.035), (0.16, 0.125, 0.06), white),
             geom("sphere", (0.15, 0.0, 0.035), (0.02,), glass),
             geom("cylinder", (-0.07, 0.0, 0.098), (0.03, 0.006), arm_mat)]
    for m in mounts:
        parts.append(_capsule_between((0, 0, 0.015), (m[0], m[1], 0.015), 0.013, arm_mat))
        parts.append(geom("cylinder", (m[0], m[1], 0.0), (0.034, 0.028), metal))
        parts.append(geom("cylinder", (m[0], m[1], 0.03), (0.036, 0.004), carbon))
    # AED case slung under the hull, with straps, and a white cross on the underside and both flanks
    parts.append(geom("box", (0, 0, CASE_Z), CASE_HALF, green))
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(geom("cylinder", (sx * 0.10, sy * 0.07, -0.05), (0.01, 0.03), carbon))
    zc = CASE_Z - CASE_HALF[2] - 0.001
    parts += [geom("box", (0, 0, zc), (0.05, 0.014, 0.0015), cross), geom("box", (0, 0, zc), (0.014, 0.05, 0.0015), cross)]
    for sy in (-1, 1):
        yc = sy * (CASE_HALF[1] + 0.001)
        parts += [geom("box", (0, yc, CASE_Z), (0.045, 0.0015, 0.013), cross), geom("box", (0, yc, CASE_Z), (0.013, 0.0015, 0.045), cross)]
    # landing skids
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(_capsule_between((sx * 0.12, sy * 0.10, -0.03), (sx * 0.12, sy * 0.19, SKID_Z), 0.008, carbon))
    for sy in (-1, 1):
        parts.append(geom("capsule", (0, sy * 0.19, SKID_Z), (0.011, 0.19), carbon, euler=(0.0, math.pi / 2, 0.0)))
    rx.world_xml.append(f'<body name="{BODY}" mocap="true" pos="0 0 {PARK_Z}">' + "".join(parts) + "</body>")

    for k in range(4):
        tip = tip_front if mounts[k][0] > 0 else tip_rear
        rotor = [geom("cylinder", (0, 0, 0), (0.02, 0.012), metal), geom("box", (0, 0, 0.012), (quad.rotor_radius_m - 0.005, 0.02, 0.003), blade),
                 geom("box", (quad.rotor_radius_m - 0.035, 0, 0.012), (0.03, 0.0205, 0.0035), tip)]
        rx.world_xml.append(f'<body name="{ROTOR.format(k=k)}" mocap="true" pos="0 0 {PARK_Z}">' + "".join(rotor) + "</body>")
    for name, mat in zip(LEDS, (amber, lime), strict=True):
        rx.world_xml.append(f'<body name="{name}" mocap="true" pos="0 0 {PARK_Z}">' + geom("sphere", (0, 0, 0), (0.02,), mat) + "</body>")
    return DroneAnimator(spec)


class DroneAnimator:
    """Poses the drone's parts from its recording at the time held in ``t`` (seconds after liftoff).

    ``active = False`` parks everything below the floor (the drone is not in the shot). The clock is a plain
    attribute so the drone's timeline is independent of the rover recording's step index that ``apply`` receives.
    """

    def __init__(self, spec: DroneSpec) -> None:
        self.spec = spec
        self.t = 0.0
        self.active = True

    def bind(self, model: mujoco.MjModel, scene_rec=None) -> None:
        def mocap_id(name: str) -> int:
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            mid = int(model.body_mocapid[bid]) if bid >= 0 else -1
            if mid < 0:
                raise ValueError(f"render model has no mocap body {name!r} (was the drone dressed into this scene?)")
            return mid

        self.m_body = mocap_id(BODY)
        self.m_rotor = np.array([mocap_id(ROTOR.format(k=k)) for k in range(4)])
        self.m_led = [mocap_id(n) for n in LEDS]
        quad, rec = self.spec.quad, self.spec.rec
        self.mounts = np.array([[x, y, MOUNT_Z] for x, y in quad.rotor_xy()])
        k_thrust = float(rec.meta.get("quad", {}).get("k_thrust", quad.k_thrust))
        omega = np.sqrt(np.maximum(rec.thrust, 0.0) / k_thrust)                       # (T, 4) rad/s
        self._sign = -np.array(YAW_SIGNS)                                              # a rotor turns against its reaction torque
        self._phase = np.cumsum(omega, axis=0) * rec.dt * SPIN_FRACTION                # (T, 4) drawn angle
        self._omega_end = omega[-1] * SPIN_FRACTION
        self.t_release = rec.duration if rec.meta.get("completed", True) else math.inf

    def rotor_angles(self, t: float) -> np.ndarray:
        """Drawn angle of each rotor [rad] at ``t``; after the last sample the rotors keep turning at their final speed."""
        rec = self.spec.rec
        if t <= rec.t[-1]:
            ang = np.array([np.interp(t, rec.t, self._phase[:, k]) for k in range(4)])
        else:
            ang = self._phase[-1] + self._omega_end * (t - rec.t[-1])
        return self._sign * ang

    def pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Airframe position and quaternion in the scene at ``t``."""
        p, q = self.spec.rec.pose_at(t)
        pl = self.spec.placement
        return pl.point(p), _qmul(pl.quat, q)

    def apply(self, data: mujoco.MjData, s: float = 0.0) -> None:
        parts = [self.m_body, *self.m_rotor, *self.m_led]
        if not self.active:
            data.mocap_pos[parts, 2] = PARK_Z
            return
        p, q = self.pose(self.t)
        rot = quat_to_rot(q)
        data.mocap_pos[self.m_body], data.mocap_quat[self.m_body] = p, q
        angles = self.rotor_angles(self.t)
        for k, m in enumerate(self.m_rotor):
            data.mocap_pos[m] = p + rot @ self.mounts[k]
            data.mocap_quat[m] = _qmul(q, _qz(float(angles[k])))
        released = self.t >= self.t_release
        for j, m in enumerate(self.m_led):
            if (j == 1) == released:
                data.mocap_pos[m] = p + rot @ LED_OFFSET
                data.mocap_quat[m] = q
            else:
                data.mocap_pos[m, 2] = PARK_Z
