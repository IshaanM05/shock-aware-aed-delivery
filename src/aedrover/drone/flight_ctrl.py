"""Cascaded flight controller for the comparator quadrotor.

Structure (outer to inner), all loops run at the control rate:

    position P  ->  velocity PI  ->  desired thrust vector  ->  attitude P (SO(3) error)
        ->  body-rate PD (with gyroscopic feed-forward)  ->  motor mixing with saturation

Equations (world frame, x forward, y left, z up; SI units):

    v_cmd = v_ref + Kp_pos (p_ref - p)                      (correction magnitude-limited)
    a_cmd = a_ref + Kp_vel (v_cmd - v) + Ki_vel * int(v_cmd - v) dt
    F_des = m (a_cmd + g e_z)                               (tilt-limited to max_tilt)
    T     = F_des . z_b                                     (collective thrust, N)
    R_d   = [x_d y_d z_d],  z_d = F_des/|F_des|, y_d = z_d x x_c / |.|, x_d = y_d x z_d
    e_R   = 1/2 (R_d^T R - R^T R_d)^vee                     (attitude error, body frame)
    w_cmd = -Kp_att * e_R
    tau   = J (Kp_rate (w_cmd - w) - Kd_rate dw_filt/dt) + w x J w
    f     = M^-1 [T, tau_x, tau_y, tau_z]                   (per-rotor thrust, N)

The attitude error follows the SO(3) formulation used for geometric quadrotor control
(``lee2010``); the cascaded position/velocity/attitude/rate architecture is the standard
multirotor structure (``mahony2012``). The controller uses the TRUE simulated state (no
estimator noise) and knows the vehicle mass exactly; both are documented limitations.

All gains are tuned by hand for the default QuadParams. They are ASSUMPTIONS in the sense
that they carry no literature value; the tests demonstrate the resulting tracking.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from aedrover.drone.quadrotor_mjcf import N_ROTORS, YAW_SIGNS, G, QuadParams, QuadState


@dataclass(frozen=True)
class ControllerGains:
    """Tunable gains of the cascaded controller (all ASSUMPTION, hand tuned).

    Attributes:
        kp_pos: position P gain per axis (x, y, z) [1/s].
        kp_vel: velocity P gain per axis [1/s].
        ki_vel: velocity I gain per axis [1/s^2].
        kp_att: attitude P gain (roll, pitch, yaw) [1/s], maps SO(3) error to a rate command.
        kp_rate: body-rate P gain (roll, pitch, yaw) [1/s], applied on inertia-scaled error.
        kd_rate: body-rate D gain (roll, pitch, yaw) [s], on the filtered angular acceleration.
        max_tilt_deg: tilt limit of the commanded thrust vector [deg].
        max_pos_correction_mps: limit of the position-error velocity correction
            (horizontal, vertical) [m/s].
        accel_int_limit_mps2: anti-windup clamp on the velocity-integral acceleration [m/s^2].
        rate_filter_hz: cut-off of the angular-acceleration low-pass filter [Hz].
        min_thrust_n: lower bound on per-rotor thrust so the rotors never stop [N].
    """

    kp_pos: tuple[float, float, float] = (1.2, 1.2, 1.5)
    kp_vel: tuple[float, float, float] = (3.5, 3.5, 5.0)
    ki_vel: tuple[float, float, float] = (1.0, 1.0, 2.0)
    kp_att: tuple[float, float, float] = (8.0, 8.0, 3.0)
    kp_rate: tuple[float, float, float] = (25.0, 25.0, 8.0)
    kd_rate: tuple[float, float, float] = (0.02, 0.02, 0.0)
    max_tilt_deg: float = 35.0
    max_pos_correction_mps: tuple[float, float] = (5.0, 3.0)
    accel_int_limit_mps2: float = 5.0
    rate_filter_hz: float = 40.0
    min_thrust_n: float = 0.5


@dataclass(frozen=True)
class Reference:
    """Position-level setpoint with feed-forward.

    Attributes:
        pos: reference position [m], shape (3,).
        vel: reference velocity [m/s], shape (3,).
        acc: reference acceleration [m/s^2], shape (3,).
        yaw: reference yaw [rad].
    """

    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acc: np.ndarray = field(default_factory=lambda: np.zeros(3))
    yaw: float = 0.0


def mixer_matrix(p: QuadParams) -> np.ndarray:
    """Matrix M with [T, tau_x, tau_y, tau_z] = M @ f for rotor thrusts f (N).

    Rows: total thrust; roll torque sum(y_i f_i); pitch torque -sum(x_i f_i); yaw reaction
    torque sum(s_i c_q f_i). Frame x forward, y left, z up.
    """
    xy = p.rotor_xy()
    return np.array(
        [
            np.ones(N_ROTORS),
            xy[:, 1],
            -xy[:, 0],
            np.array(YAW_SIGNS) * p.yaw_torque_coeff_m,
        ]
    )


def _vee(m: np.ndarray) -> np.ndarray:
    return np.array([m[2, 1], m[0, 2], m[1, 0]])


class CascadedController:
    """Position -> velocity -> attitude -> body-rate -> motor-mixing controller."""

    def __init__(
        self,
        params: QuadParams | None = None,
        gains: ControllerGains | None = None,
        dt: float = 0.008,
    ) -> None:
        self.p = params or QuadParams()
        self.g = gains or ControllerGains()
        self.dt = dt
        self.mass = self.p.total_mass_kg
        self.inertia = np.array(self.p.inertia_kgm2)
        self._minv = np.linalg.inv(mixer_matrix(self.p))
        self._kp_pos = np.array(self.g.kp_pos)
        self._kp_vel = np.array(self.g.kp_vel)
        self._ki_vel = np.array(self.g.ki_vel)
        self._kp_att = np.array(self.g.kp_att)
        self._kp_rate = np.array(self.g.kp_rate)
        self._kd_rate = np.array(self.g.kd_rate)
        self._tan_tilt = math.tan(math.radians(self.g.max_tilt_deg))
        tau_f = 1.0 / (2.0 * math.pi * self.g.rate_filter_hz)
        self._alpha = dt / (dt + tau_f)
        self._t_total_max = N_ROTORS * self.p.max_thrust_per_rotor_n * 0.95
        self.reset()

    def reset(self) -> None:
        """Clear integrator and derivative-filter memory."""
        self._int = np.zeros(3)
        self._omega_prev: np.ndarray | None = None
        self._alpha_filt = np.zeros(3)
        self.last_total_thrust_n = 0.0
        self.last_saturated = False

    # ------------------------------------------------------------------ outer loops
    def _thrust_vector(self, s: QuadState, ref: Reference) -> np.ndarray:
        e_p = ref.pos - s.pos
        corr = self._kp_pos * e_p
        lim_xy, lim_z = self.g.max_pos_correction_mps
        n_xy = math.hypot(corr[0], corr[1])
        if n_xy > lim_xy:
            corr[0:2] *= lim_xy / n_xy
        corr[2] = min(max(corr[2], -lim_z), lim_z)
        v_cmd = ref.vel + corr
        e_v = v_cmd - s.vel
        lim_i = self.g.accel_int_limit_mps2 / np.maximum(self._ki_vel, 1e-9)
        self._int = np.clip(self._int + e_v * self.dt, -lim_i, lim_i)
        a_cmd = ref.acc + self._kp_vel * e_v + self._ki_vel * self._int
        f = self.mass * (a_cmd + np.array([0.0, 0.0, G]))
        f[2] = max(f[2], 0.3 * self.mass * G)  # always point the thrust upward
        n_h = math.hypot(f[0], f[1])
        lim_h = f[2] * self._tan_tilt
        if n_h > lim_h:
            f[0:2] *= lim_h / n_h
        return f

    # ------------------------------------------------------------------ inner loops
    def _attitude_rate_cmd(self, s: QuadState, f_des: np.ndarray, yaw: float) -> np.ndarray:
        z_d = f_des / np.linalg.norm(f_des)
        x_c = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        y_d = np.cross(z_d, x_c)
        y_d /= np.linalg.norm(y_d)
        x_d = np.cross(y_d, z_d)
        r_d = np.column_stack((x_d, y_d, z_d))
        e_r = 0.5 * _vee(r_d.T @ s.rot - s.rot.T @ r_d)
        return -self._kp_att * e_r

    def _torque(self, s: QuadState, w_cmd: np.ndarray) -> np.ndarray:
        if self._omega_prev is None:
            self._omega_prev = s.omega.copy()
        ang_acc = (s.omega - self._omega_prev) / self.dt
        self._alpha_filt += self._alpha * (ang_acc - self._alpha_filt)
        self._omega_prev = s.omega.copy()
        e_w = w_cmd - s.omega
        j = self.inertia
        return j * (self._kp_rate * e_w - self._kd_rate * self._alpha_filt) + np.cross(
            s.omega, j * s.omega
        )

    # ------------------------------------------------------------------ mixing
    def allocate(self, thrust: float, tau: np.ndarray) -> np.ndarray:
        """Map collective thrust [N] and body torque [N m] to per-rotor thrusts [N].

        Saturation policy: shed yaw authority first (yaw is the least critical axis); if the
        differential roll/pitch demand still exceeds the range, scale it, then shift the
        collective to fit. Attitude authority is therefore preserved over altitude.
        """
        tmin = self.g.min_thrust_n
        tmax = self.p.max_thrust_per_rotor_n
        base = self._minv @ np.array([thrust, tau[0], tau[1], 0.0])
        yaw = self._minv[:, 3] * tau[2]
        for a in (1.0, 0.5, 0.25, 0.0):
            f = base + a * yaw
            if f.min() >= tmin and f.max() <= tmax:
                self.last_saturated = a < 1.0
                return f
        self.last_saturated = True
        mean = float(base.mean())
        diff = base - mean
        span = float(diff.max() - diff.min())
        if span > tmax - tmin:
            diff *= (tmax - tmin) / span
        mean = min(max(mean, tmin - float(diff.min())), tmax - float(diff.max()))
        return np.clip(mean + diff, tmin, tmax)

    # ------------------------------------------------------------------ public
    def control(self, s: QuadState, ref: Reference) -> np.ndarray:
        """One control update.

        Args:
            s: current true state.
            ref: position/velocity/acceleration/yaw reference.

        Returns:
            Commanded thrust per rotor [N], shape (4,).
        """
        f_des = self._thrust_vector(s, ref)
        thrust = float(np.dot(f_des, s.rot[:, 2]))
        thrust = min(max(thrust, 0.0), self._t_total_max)
        w_cmd = self._attitude_rate_cmd(s, f_des, ref.yaw)
        tau = self._torque(s, w_cmd)
        self.last_total_thrust_n = thrust
        return self.allocate(thrust, tau)
