"""Runtime wrapper: Ackermann wheel allocation, stepping and state access.

The controller commands a forward speed ``v`` [m/s] (at the geometric centre) and a bicycle
steering angle ``delta`` [rad]. Per-wheel speed targets and the two front steer angles follow
Ackermann geometry; the speed loops run inside MuJoCo (velocity actuators, torque clamped).
"""

from __future__ import annotations

import math
from math import atan, copysign, hypot, tan

import mujoco
import numpy as np

from .vehicle_mjcf import WHEELS
from .world import World

_TQ = mujoco.mjtObj


class Rover:
    def __init__(self, world: World, control_dt: float = 0.02, log_every: int = 2,
                 derate_motors: bool = True):
        self.w = world
        self.m, self.d = world.model, world.data
        self.p = world.veh
        self.dt_phys = float(self.m.opt.timestep)
        self.n_sub = max(1, round(control_dt / self.dt_phys))
        self.control_dt = self.n_sub * self.dt_phys
        self.log_every = max(1, min(log_every, self.n_sub))
        self.derate = derate_motors

        m = self.m
        aid = lambda n: mujoco.mj_name2id(m, _TQ.mjOBJ_ACTUATOR, n)  # noqa: E731
        jid = lambda n: mujoco.mj_name2id(m, _TQ.mjOBJ_JOINT, n)  # noqa: E731
        sid = lambda n: mujoco.mj_name2id(m, _TQ.mjOBJ_SENSOR, n)  # noqa: E731
        self.a_drive = np.array([aid(f"drive_{t}") for t in WHEELS])
        self.a_steer = np.array([aid("steer_fl"), aid("steer_fr")])
        self.j_root = jid("root")
        self.qadr_susp = np.array([m.jnt_qposadr[jid(f"susp_{t}")] for t in WHEELS])
        self.vadr_spin = np.array([m.jnt_dofadr[jid(f"spin_{t}")] for t in WHEELS])
        self.qadr_iso_z = m.jnt_qposadr[jid("iso_z")]
        self.qadr_iso = np.array([m.jnt_qposadr[jid(n)] for n in ("iso_x", "iso_y", "iso_z")])
        s_pay = sid("payload_acc")
        self.s_pay = slice(m.sensor_adr[s_pay], m.sensor_adr[s_pay] + 3)
        s_chs = sid("chassis_acc")
        self.s_chs = slice(m.sensor_adr[s_chs], m.sensor_adr[s_chs] + 3)
        s_gyr = sid("chassis_gyro")
        self.s_gyr = slice(m.sensor_adr[s_gyr], m.sensor_adr[s_gyr] + 3)
        tau = self.p.motor_peak_torque
        self._forcerange0 = np.tile([-tau, tau], (4, 1))   # nominal, NOT read back from the model

        L, W = self.p.wheelbase, self.p.track
        self._L, self._W = L, W
        # wheel positions relative to the rear-axle centre: (x, y) for fl, fr, rl, rr
        self._wx = np.array([L, L, 0.0, 0.0])
        self._wy = np.array([W / 2, -W / 2, W / 2, -W / 2])
        self._att_t, self._att = -1.0, (0.0, 0.0, 0.0)
        self.acc_log: list[np.ndarray] = []    # payload proper acceleration, payload frame [m/s^2]
        self.gdir_log: list[np.ndarray] = []   # world-up direction in the chassis frame (R row 2)
        self.t_log: list[float] = []

    # ------------------------------------------------------------------ reset
    def reset(self, x: float, y: float, z_surface: float, yaw: float = 0.0,
              speed: float = 0.0, settle_s: float = 0.0) -> None:
        d, p = self.d, self.p
        mujoco.mj_resetData(self.m, d)
        adr = self.m.jnt_qposadr[self.j_root]
        d.qpos[adr:adr + 3] = (x, y, z_surface + p.nominal_height)
        d.qpos[adr + 3:adr + 7] = (np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2))
        if speed:
            dof = self.m.jnt_dofadr[self.j_root]
            d.qvel[dof:dof + 3] = (speed * np.cos(yaw), speed * np.sin(yaw), 0.0)
            d.qvel[self.vadr_spin] = speed / p.wheel_radius
        d.ctrl[:] = 0.0
        self.m.actuator_forcerange[self.a_drive] = self._forcerange0   # undo previous-episode derating
        if speed:
            d.ctrl[self.a_drive] = speed / p.wheel_radius
        self.w.apply_mocap()
        mujoco.mj_forward(self.m, d)
        self._att_t = -1.0
        self.clear_logs()
        if settle_s > 0.0:
            mujoco.mj_step(self.m, d, nstep=int(settle_s / self.dt_phys))
            self.clear_logs()

    def clear_logs(self) -> None:
        self.acc_log.clear()
        self.gdir_log.clear()
        self.t_log.clear()

    @property
    def log_rate(self) -> float:
        return 1.0 / (self.log_every * self.dt_phys)

    # ------------------------------------------------------------- allocation
    def allocate(self, v: float, delta: float) -> tuple[np.ndarray, np.ndarray]:
        """Ackermann allocation -> (wheel angular speed targets [fl fr rl rr], steer angles [fl fr])."""
        L, W, r = self._L, self._W, self.p.wheel_radius
        lim = self.p.steer_range * 0.98
        t = tan(max(-1.2, min(1.2, delta)))
        if abs(t) < 1e-6:
            return np.full(4, v / r), np.zeros(2)
        R = L / t
        sfl = atan(L * t / (L - 0.5 * W * t))
        sfr = atan(L * t / (L + 0.5 * W * t))
        dist = np.hypot(self._wx, R - self._wy)
        center = hypot(0.5 * L, R)
        wheel = copysign(1.0, v) * abs(v) * dist / center / r
        return wheel, np.clip([sfl, sfr], -lim, lim)

    # ------------------------------------------------------------------ step
    def step(self, v_cmd: float, delta_cmd: float, log: bool = True) -> None:
        d, m = self.d, self.m
        wheel, steer = self.allocate(v_cmd, delta_cmd)
        wmax = self.p.wheel_speed_max
        d.ctrl[self.a_drive] = np.clip(wheel, -wmax, wmax)
        d.ctrl[self.a_steer] = steer
        if self.derate:
            w = np.abs(d.qvel[self.vadr_spin])
            scale = np.clip(1.0 - w / (1.08 * wmax), 0.05, 1.0)
            m.actuator_forcerange[self.a_drive, 0] = self._forcerange0[:, 0] * scale
            m.actuator_forcerange[self.a_drive, 1] = self._forcerange0[:, 1] * scale
        if not log:
            mujoco.mj_step(m, d, nstep=self.n_sub)
            return
        left = self.n_sub
        while left > 0:
            k = min(self.log_every, left)
            mujoco.mj_step(m, d, nstep=k)
            self.acc_log.append(d.sensordata[self.s_pay].copy())
            self.gdir_log.append(d.xmat[self.w.b_chassis, 6:9].copy())
            self.t_log.append(d.time)
            left -= k

    # ------------------------------------------------------------------ state
    @property
    def pos(self) -> np.ndarray:
        return self.d.xpos[self.w.b_chassis]

    @property
    def quat(self) -> np.ndarray:
        return self.d.xquat[self.w.b_chassis]

    def yaw_pitch_roll(self) -> tuple[float, float, float]:
        """Yaw, pitch, roll [rad]; cached per simulation time (called several times per step)."""
        t = self.d.time
        if t == self._att_t:
            return self._att
        w, x, y, z = self.d.xquat[self.w.b_chassis]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        self._att_t, self._att = t, (yaw, pitch, roll)
        return self._att

    def body_velocity(self) -> np.ndarray:
        """Linear velocity of the chassis in its own frame (vx forward, vy left, vz up)."""
        dof = self.m.jnt_dofadr[self.j_root]
        vw = self.d.qvel[dof:dof + 3]
        R = self.d.xmat[self.w.b_chassis].reshape(3, 3)
        return R.T @ vw

    def yaw_rate(self) -> float:
        return float(self.d.sensordata[self.s_gyr][2])

    def wheel_speeds(self) -> np.ndarray:
        return self.d.qvel[self.vadr_spin]

    def susp_deflection(self) -> np.ndarray:
        return self.d.qpos[self.qadr_susp]

    def payload_acc(self) -> np.ndarray:
        return self.d.sensordata[self.s_pay]

    def is_finite(self) -> bool:
        return bool(np.isfinite(self.d.qpos).all() and np.isfinite(self.d.qvel).all())
