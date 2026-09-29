"""Parametric MJCF builder for the AED rover.

Frame convention (REP-103): x forward, y left, z up. SI units, angles in radians.

Topology
--------
chassis (free joint)
|-- payload      : 3 slide joints (viscoelastic isolator), carries the accelerometer
|-- knuckle_{fl,fr,rl,rr}  : slide-z (independent suspension) [+ hinge-z steer on the front axle]
    `-- wheel_*  : hinge-y spin, cylinder geom

Low-level wheel control is done natively inside MuJoCo (velocity actuators with a torque
clamp on the spin joints, position actuators on the steering joints) so the Python control
loop only writes 8 numbers per control step.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path

import yaml

WHEELS = ("fl", "fr", "rl", "rr")
G = 9.81


@dataclass(frozen=True)
class VehicleParams:
    # --- chassis / payload -------------------------------------------------
    chassis_half: tuple[float, float, float] = (0.42, 0.24, 0.07)
    chassis_mass: float = 22.0            # kg, sprung
    payload_half: tuple[float, float, float] = (0.18, 0.13, 0.07)
    payload_mass: float = 4.0             # kg, AED + case, on the isolator
    payload_offset_z: float = 0.16        # payload centre above chassis centre at rest [m]
    # --- geometry ----------------------------------------------------------
    wheelbase: float = 0.62               # m
    track: float = 0.62                   # m, lateral distance between wheel centres
    axle_drop: float = 0.06               # m, axle below the chassis centre at nominal ride height
    wheel_radius: float = 0.15            # m
    wheel_half_width: float = 0.035       # m
    wheel_mass: float = 1.65              # kg, wheel + hub motor
    knuckle_mass: float = 0.60            # kg
    # --- suspension (per wheel) --------------------------------------------
    susp_k: float = 4500.0                # N/m
    susp_c: float = 350.0                 # N s/m
    susp_range: tuple[float, float] = (-0.08, 0.06)   # m, droop / bump stops
    # --- payload isolator ---------------------------------------------------
    iso_kz: float = 2200.0
    iso_cz: float = 180.0
    iso_kxy: float = 2200.0
    iso_cxy: float = 180.0
    iso_range_z: float = 0.05
    iso_range_xy: float = 0.04
    # --- drive / steering ---------------------------------------------------
    motor_peak_torque: float = 25.0       # N m per wheel
    motor_kv: float = 8.0                 # N m s/rad speed-loop gain
    wheel_speed_max: float = 30.0         # rad/s (4.5 m/s at r = 0.15)
    wheel_damping: float = 0.05           # N m s/rad (bearing loss)
    wheel_armature: float = 0.02          # kg m^2 (rotor inertia)
    steer_kp: float = 250.0
    steer_kv: float = 12.0
    steer_range: float = 0.65             # rad
    # --- contact ------------------------------------------------------------
    friction: float = 1.0                 # tyre / ground Coulomb coefficient
    tyre_solref: tuple[float, float] = (0.03, 1.0)   # measured k_t ~ 200 kN/m

    # ------------------------------------------------------------------ derived
    @property
    def sprung_mass(self) -> float:
        return self.chassis_mass + self.payload_mass

    @property
    def sprung_per_wheel(self) -> float:
        return self.sprung_mass / 4.0

    @property
    def unsprung_mass(self) -> float:
        return self.wheel_mass + self.knuckle_mass

    @property
    def total_mass(self) -> float:
        return self.sprung_mass + 4.0 * self.unsprung_mass

    @property
    def nominal_height(self) -> float:
        """Chassis-centre height above flat ground at static equilibrium."""
        return self.wheel_radius + self.axle_drop

    @property
    def ground_clearance(self) -> float:
        return self.nominal_height - self.chassis_half[2]

    @property
    def static_sag(self) -> float:
        return self.sprung_per_wheel * G / self.susp_k

    @property
    def susp_zeta(self) -> float:
        return self.susp_c / (2.0 * (self.susp_k * self.sprung_per_wheel) ** 0.5)

    @property
    def susp_omega_n(self) -> float:
        return (self.susp_k / self.sprung_per_wheel) ** 0.5

    def with_(self, **kw) -> VehicleParams:
        return replace(self, **kw)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> VehicleParams:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        for k, v in list(raw.items()):
            if isinstance(v, list):
                raw[k] = tuple(v)
        return cls(**raw)


def _f(x: float) -> str:
    return f"{x:.6g}"


def _v(*xs: float) -> str:
    return " ".join(_f(x) for x in xs)


def vehicle_xml(p: VehicleParams, *, spawn_z: float, spawn_x: float = 0.0, spawn_y: float = 0.0,
                spawn_yaw: float = 0.0) -> str:
    """Return the <body name="chassis"> subtree."""
    from math import cos, sin

    hx, hy, hz = p.chassis_half
    phx, phy, phz = p.payload_half
    x_half, y_half = p.wheelbase / 2.0, p.track / 2.0
    quat = _v(cos(spawn_yaw / 2), 0.0, 0.0, sin(spawn_yaw / 2))
    sag_iso = p.payload_mass * G / p.iso_kz
    spring_ref = -(p.sprung_per_wheel * G) / p.susp_k
    lo, hi = p.susp_range
    wfr = f"{_f(p.friction)} 0.005 0.0001"
    solref = _v(*p.tyre_solref)

    def knuckle(tag: str, sx: int, sy: int, steer: bool) -> str:
        steer_joint = (
            f'<joint name="steer_{tag}" type="hinge" axis="0 0 1" limited="true" '
            f'range="{_v(-p.steer_range, p.steer_range)}" damping="0.5" armature="0.01"/>'
            if steer else ""
        )
        return f"""
      <body name="knuckle_{tag}" pos="{_v(sx * x_half, sy * y_half, -p.axle_drop)}">
        <inertial pos="0 0 0" mass="{_f(p.knuckle_mass)}" diaginertia="0.002 0.002 0.002"/>
        <joint name="susp_{tag}" type="slide" axis="0 0 1" limited="true" range="{_v(lo, hi)}"
               stiffness="{_f(p.susp_k)}" damping="{_f(p.susp_c)}" springref="{_f(spring_ref)}"/>
        {steer_joint}
        <body name="wheel_{tag}">
          <joint name="spin_{tag}" type="hinge" axis="0 1 0" damping="{_f(p.wheel_damping)}"
                 armature="{_f(p.wheel_armature)}"/>
          <geom name="tyre_{tag}" type="cylinder" size="{_v(p.wheel_radius, p.wheel_half_width)}"
                euler="1.5707963 0 0" mass="{_f(p.wheel_mass)}" friction="{wfr}" solref="{solref}"
                rgba="0.08 0.08 0.09 1" group="2"/>
        </body>
      </body>"""

    wheels = "".join(
        knuckle(t, sx, sy, steer=(sx > 0))
        for t, sx, sy in (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1))
    )

    return f"""
    <body name="chassis" pos="{_v(spawn_x, spawn_y, spawn_z)}" quat="{quat}">
      <freejoint name="root"/>
      <geom name="chassis_geom" type="box" size="{_v(hx, hy, hz)}" mass="{_f(p.chassis_mass)}"
            rgba="0.85 0.12 0.12 1" friction="0.4 0.005 0.0001" group="2"/>
      <geom name="chassis_nose" type="box" size="0.04 {_f(hy)} 0.05" pos="{_v(hx + 0.04, 0, -0.01)}"
            mass="0.001" rgba="0.15 0.15 0.15 1" group="2" contype="0" conaffinity="0"/>
      <site name="imu" pos="0 0 0" size="0.01"/>
      <site name="lidar" pos="{_v(hx, 0, 0.10)}" size="0.01"/>
      <body name="payload" pos="{_v(0, 0, p.payload_offset_z)}">
        <joint name="iso_x" type="slide" axis="1 0 0" limited="true" range="{_v(-p.iso_range_xy, p.iso_range_xy)}"
               stiffness="{_f(p.iso_kxy)}" damping="{_f(p.iso_cxy)}"/>
        <joint name="iso_y" type="slide" axis="0 1 0" limited="true" range="{_v(-p.iso_range_xy, p.iso_range_xy)}"
               stiffness="{_f(p.iso_kxy)}" damping="{_f(p.iso_cxy)}"/>
        <joint name="iso_z" type="slide" axis="0 0 1" limited="true" range="{_v(-p.iso_range_z, p.iso_range_z)}"
               stiffness="{_f(p.iso_kz)}" damping="{_f(p.iso_cz)}" springref="{_f(sag_iso)}"/>
        <geom name="payload_geom" type="box" size="{_v(phx, phy, phz)}" mass="{_f(p.payload_mass)}"
              rgba="0.98 0.98 0.98 1" contype="0" conaffinity="0" group="2"/>
        <geom name="payload_cross" type="box" size="0.05 0.012 0.075" pos="0 0 0.001"
              mass="0.001" rgba="0.1 0.7 0.2 1" contype="0" conaffinity="0" group="2"/>
        <site name="payload_accel" pos="0 0 0" size="0.01"/>
      </body>{wheels}
    </body>"""


def vehicle_actuators_xml(p: VehicleParams) -> str:
    tau = p.motor_peak_torque
    w = p.wheel_speed_max
    drive = "\n    ".join(
        f'<velocity name="drive_{t}" joint="spin_{t}" kv="{_f(p.motor_kv)}" '
        f'ctrllimited="true" ctrlrange="{_v(-w, w)}" forcelimited="true" forcerange="{_v(-tau, tau)}"/>'
        for t in WHEELS
    )
    steer = "\n    ".join(
        f'<position name="steer_{t}" joint="steer_{t}" kp="{_f(p.steer_kp)}" kv="{_f(p.steer_kv)}" '
        f'ctrllimited="true" ctrlrange="{_v(-p.steer_range, p.steer_range)}"/>'
        for t in ("fl", "fr")
    )
    return f"""
  <actuator>
    {drive}
    {steer}
  </actuator>"""


def vehicle_sensors_xml() -> str:
    return """
  <sensor>
    <accelerometer name="payload_acc" site="payload_accel"/>
    <accelerometer name="chassis_acc" site="imu"/>
    <gyro name="chassis_gyro" site="imu"/>
  </sensor>"""


def vehicle_contact_excludes_xml() -> str:
    return "\n    ".join(f'<exclude body1="chassis" body2="wheel_{t}"/>' for t in WHEELS)


DEFAULT_PARAMS = VehicleParams()
