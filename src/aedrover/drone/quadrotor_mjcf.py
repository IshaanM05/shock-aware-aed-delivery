"""Parametric MJCF builder and runtime wrapper for the AED comparator quadrotor.

The subject of the aedrover project is a ground rover. This quadrotor is a deliberately
small comparator ("Track D") so that results can be reported as ambulance vs rover vs drone.

Frame convention (REP-103): x forward, y left, z up. SI units (m, s, kg, N, rad) unless a
name says otherwise. The free-joint angular velocity is expressed in the BODY frame (MuJoCo
convention), the linear velocity in the WORLD frame.

Topology
--------
``quad`` (free joint), centre of mass at the body origin, X configuration:

    rotor 0 front-left  (+a, +a), yaw sign +1        a = arm_length / sqrt(2)
    rotor 1 rear-left   (-a, +a), yaw sign -1
    rotor 2 rear-right  (-a, -a), yaw sign +1
    rotor 3 front-right (+a, -a), yaw sign -1

Each rotor is a site-based ``general`` actuator whose control is the commanded thrust [N]:
a first-order lag (time constant ``motor_tau_s``) turns it into the applied thrust, and the
gear vector (0 0 1 0 0 s_i*c_q) gives +z thrust at the rotor site (so roll/pitch torques
arise from the lever arm) plus a yaw reaction torque s_i*c_q*T. An IMU-equivalent sensor
group (accelerometer, gyro, orientation quaternion) sits at the centre of mass.

Sources and assumptions (see docs/DRONE_COMPARATOR.md for the full table):
    * cd_body = 1.5: ``stolaroff2018`` (body drag coefficient at higher speeds).
    * The 5-9 kg vehicle class follows ``claesson2017`` (a 5.7 kg 8-rotor AED drone with a
      763 g AED payload). The 2 kg payload used here is a deliberately conservative
      ASSUMPTION (AED + case + release mechanism), not a reported number.
    * Every other numeric default below is an ASSUMPTION, flagged in the field comments.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

import mujoco
import numpy as np

G = 9.81  # m/s^2, standard gravity
RHO_AIR = 1.225  # kg/m^3, ISA sea level; ASSUMPTION for the site (sensitivity 1.10-1.25)

N_ROTORS = 4
YAW_SIGNS = (1.0, -1.0, 1.0, -1.0)  # sign of the yaw reaction torque on the body per rotor


@dataclass(frozen=True)
class QuadParams:
    """Physical parameters of the comparator quadrotor.

    All fields are SI. Fields marked ASSUMPTION have no primary source (see the doc table).

    Attributes:
        mass_kg: airframe + battery + avionics mass, without payload [kg]. ASSUMPTION
            (class-consistent with 5.7 kg in claesson2017).
        payload_mass_kg: AED + case + release mechanism [kg]. ASSUMPTION (claesson2017
            reports a 0.763 kg AED; 2 kg is conservative).
        arm_length_m: centre-to-rotor distance [m]. ASSUMPTION.
        inertia_kgm2: (Ixx, Iyy, Izz) of the fully loaded vehicle about its centre of mass
            [kg m^2]. ASSUMPTION (lumped-mass estimate for the arm length above).
        rotor_radius_m: rotor disc radius [m]. ASSUMPTION.
        k_thrust: rotor thrust coefficient, T = k_thrust * omega^2 [N s^2 / rad^2]. Used only
            to report rotor speed; the MuJoCo actuator commands thrust directly. ASSUMPTION.
        yaw_torque_coeff_m: ratio of rotor drag torque to thrust, c_q = Q / T [m]. ASSUMPTION.
        max_thrust_per_rotor_n: thrust limit per rotor [N]. ASSUMPTION (thrust-to-weight ~2).
        motor_tau_s: first-order motor + ESC + rotor spin-up time constant [s]. ASSUMPTION.
        cd_body: body drag coefficient [-]. stolaroff2018 (C_D,body = 1.5).
        frontal_area_m2: reference frontal area of body + payload [m^2]. ASSUMPTION.
        air_density: air density [kg/m^3]. ASSUMPTION (ISA sea level).
        timestep_s: MuJoCo integration step [s].
        hull_half_m: half extents of the collision hull (x, y, z) [m].
    """

    mass_kg: float = 6.0
    payload_mass_kg: float = 2.0
    arm_length_m: float = 0.45
    inertia_kgm2: tuple[float, float, float] = (0.25, 0.25, 0.45)
    rotor_radius_m: float = 0.25
    k_thrust: float = 3.0e-4
    yaw_torque_coeff_m: float = 0.02
    max_thrust_per_rotor_n: float = 40.0
    motor_tau_s: float = 0.03
    cd_body: float = 1.5
    frontal_area_m2: float = 0.08
    air_density: float = RHO_AIR
    timestep_s: float = 0.002
    hull_half_m: tuple[float, float, float] = (0.14, 0.14, 0.05)

    # ------------------------------------------------------------------ derived
    @property
    def total_mass_kg(self) -> float:
        return self.mass_kg + self.payload_mass_kg

    @property
    def weight_n(self) -> float:
        return self.total_mass_kg * G

    @property
    def hover_thrust_per_rotor_n(self) -> float:
        return self.weight_n / N_ROTORS

    @property
    def thrust_to_weight(self) -> float:
        return N_ROTORS * self.max_thrust_per_rotor_n / self.weight_n

    @property
    def disc_area_m2(self) -> float:
        """Area of one rotor disc [m^2]."""
        return math.pi * self.rotor_radius_m**2

    @property
    def cd_area_m2(self) -> float:
        """Drag area Cd * A [m^2]."""
        return self.cd_body * self.frontal_area_m2

    @property
    def rest_height_m(self) -> float:
        """Centre-of-mass height when resting on the hull [m]."""
        return self.hull_half_m[2]

    def rotor_xy(self) -> np.ndarray:
        """Rotor positions in the body frame, shape (4, 2) [m]."""
        a = self.arm_length_m / math.sqrt(2.0)
        return np.array([[a, a], [-a, a], [-a, -a], [a, -a]])

    def rotor_speed_rad_s(self, thrust_n: float) -> float:
        """Rotor angular speed that produces ``thrust_n`` [rad/s]."""
        return math.sqrt(max(thrust_n, 0.0) / self.k_thrust)

    def with_(self, **kw: object) -> QuadParams:
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)


def build_mjcf(p: QuadParams | None = None) -> str:
    """Return the MJCF XML string of the quadrotor and a flat ground plane."""
    p = p or QuadParams()
    ixx, iyy, izz = p.inertia_kgm2
    hx, hy, hz = p.hull_half_m
    pos = p.rotor_xy()
    z0 = p.rest_height_m + 1e-3
    lines = [
        '<mujoco model="aedrover_quad">',
        f'  <option timestep="{p.timestep_s:g}" integrator="implicitfast" gravity="0 0 -{G}"/>',
        "  <worldbody>",
        '    <geom name="ground" type="plane" size="0 0 0.1" friction="1.0 0.005 0.0001" '
        'rgba="0.55 0.58 0.6 1"/>',
        f'    <body name="quad" pos="0 0 {z0:.6g}">',
        '      <freejoint name="root"/>',
        f'      <inertial pos="0 0 0" mass="{p.total_mass_kg:.6g}" '
        f'diaginertia="{ixx:.6g} {iyy:.6g} {izz:.6g}"/>',
        f'      <geom name="hull" type="box" size="{hx:.6g} {hy:.6g} {hz:.6g}" '
        'rgba="0.15 0.15 0.18 1"/>',
        '      <site name="imu" pos="0 0 0" size="0.01"/>',
    ]
    for i in range(N_ROTORS):
        x, y = pos[i]
        lines += [
            f'      <geom name="arm{i}" type="capsule" fromto="0 0 0 {x:.6g} {y:.6g} 0" '
            'size="0.012" contype="0" conaffinity="0" mass="0" rgba="0.3 0.3 0.3 1"/>',
            f'      <geom name="rotor{i}" type="cylinder" pos="{x:.6g} {y:.6g} 0.02" '
            f'size="{p.rotor_radius_m:.6g} 0.004" contype="0" conaffinity="0" mass="0" '
            'rgba="0.9 0.3 0.2 0.35"/>',
            f'      <site name="motor{i}" pos="{x:.6g} {y:.6g} 0" size="0.01"/>',
        ]
    lines += ["    </body>", "  </worldbody>", "  <actuator>"]
    for i in range(N_ROTORS):
        cq = YAW_SIGNS[i] * p.yaw_torque_coeff_m
        lines.append(
            f'    <general name="motor{i}" site="motor{i}" gear="0 0 1 0 0 {cq:.6g}" '
            f'ctrllimited="true" ctrlrange="0 {p.max_thrust_per_rotor_n:.6g}" '
            f'dyntype="filter" dynprm="{p.motor_tau_s:.6g}" gaintype="fixed" gainprm="1" '
            'biastype="none"/>'
        )
    lines += [
        "  </actuator>",
        "  <sensor>",
        '    <accelerometer name="imu_acc" site="imu"/>',
        '    <gyro name="imu_gyro" site="imu"/>',
        '    <framequat name="imu_quat" objtype="site" objname="imu"/>',
        "  </sensor>",
        "</mujoco>",
    ]
    return "\n".join(lines)


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix (body to world) from a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Quaternion (w, x, y, z) for ZYX Euler angles [rad] (yaw, then pitch, then roll)."""
    cr, sr = math.cos(roll / 2), math.sin(roll / 2)
    cp, sp = math.cos(pitch / 2), math.sin(pitch / 2)
    cy, sy = math.cos(yaw / 2), math.sin(yaw / 2)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


@dataclass(frozen=True)
class QuadState:
    """Rigid-body state of the quadrotor.

    Attributes:
        pos: position in the world frame [m], shape (3,).
        quat: orientation (w, x, y, z), body to world, shape (4,).
        vel: linear velocity in the world frame [m/s], shape (3,).
        omega: angular velocity in the body frame [rad/s], shape (3,).
        rot: rotation matrix body to world, shape (3, 3).
    """

    pos: np.ndarray
    quat: np.ndarray
    vel: np.ndarray
    omega: np.ndarray
    rot: np.ndarray

    @property
    def tilt_rad(self) -> float:
        """Angle between the body z axis and world up [rad]."""
        return math.acos(min(1.0, max(-1.0, float(self.rot[2, 2]))))


class QuadSim:
    """Runtime wrapper: build the model once, then reset / step / read state."""

    def __init__(self, params: QuadParams | None = None) -> None:
        self.params = params or QuadParams()
        self.model = mujoco.MjModel.from_xml_string(build_mjcf(self.params))
        self.data = mujoco.MjData(self.model)
        self.dt = float(self.model.opt.timestep)
        self._body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "quad")
        self._sens = {
            name: (
                int(self.model.sensor_adr[self.model.sensor(name).id]),
                int(self.model.sensor_dim[self.model.sensor(name).id]),
            )
            for name in ("imu_acc", "imu_gyro", "imu_quat")
        }
        self.reset()

    @property
    def time(self) -> float:
        return float(self.data.time)

    def reset(
        self,
        position: tuple[float, float, float] | None = None,
        quat: np.ndarray | None = None,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
        omega: tuple[float, float, float] = (0.0, 0.0, 0.0),
        spooled: bool = False,
    ) -> None:
        """Reset the simulation.

        Args:
            position: initial centre-of-mass position [m]; default resting on the ground.
            quat: initial orientation (w, x, y, z); default level.
            velocity: initial world-frame velocity [m/s].
            omega: initial body-frame angular velocity [rad/s].
            spooled: if True the rotors start at hover thrust (airborne starts); otherwise
                the rotors start at zero thrust (take-off from the ground).
        """
        mujoco.mj_resetData(self.model, self.data)
        if position is None:
            position = (0.0, 0.0, self.params.rest_height_m + 1e-3)
        self.data.qpos[0:3] = position
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else quat
        self.data.qvel[0:3] = velocity
        self.data.qvel[3:6] = omega
        if spooled:
            hover = self.params.hover_thrust_per_rotor_n
            self.data.act[:] = hover
            self.data.ctrl[:] = hover
        self.data.xfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def step(
        self,
        thrusts: np.ndarray,
        nstep: int = 1,
        ext_force: np.ndarray | None = None,
    ) -> None:
        """Advance ``nstep`` physics steps with constant commands.

        Args:
            thrusts: commanded thrust per rotor [N], shape (4,); clipped to the actuator range.
            nstep: number of physics steps of ``dt`` each.
            ext_force: optional external world-frame force on the centre of mass [N], applied
                through ``xfrc_applied`` (used for aerodynamic drag).
        """
        self.data.ctrl[:] = thrusts
        if ext_force is not None:
            self.data.xfrc_applied[self._body, 0:3] = ext_force
        mujoco.mj_step(self.model, self.data, nstep=nstep)

    def state(self) -> QuadState:
        quat = self.data.qpos[3:7].copy()
        quat /= np.linalg.norm(quat)
        return QuadState(
            pos=self.data.qpos[0:3].copy(),
            quat=quat,
            vel=self.data.qvel[0:3].copy(),
            omega=self.data.qvel[3:6].copy(),
            rot=quat_to_rot(quat),
        )

    def rotor_thrusts(self) -> np.ndarray:
        """Applied (motor-lagged) thrust per rotor [N]."""
        return self.data.act.copy()

    def imu(self) -> tuple[np.ndarray, np.ndarray]:
        """IMU-equivalent readings: specific force [m/s^2] and body rates [rad/s]."""
        a0, a1 = self._sens["imu_acc"]
        g0, g1 = self._sens["imu_gyro"]
        return self.data.sensordata[a0 : a0 + a1].copy(), self.data.sensordata[g0 : g0 + g1].copy()
