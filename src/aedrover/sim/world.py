"""Procedural sidewalk-crossing world with pre-allocated, re-positionable slots.

The MJCF is compiled once. Per-episode variation (kerb height, ramp type, obstacle and
pedestrian placement, tyre friction, payload mass ...) is applied by writing into model or
data arrays, so a scenario reset costs microseconds instead of a recompile.

Engineering note (verified, see docs/ENGINEERING_NOTES.md): MuJoCo builds the collision
midphase BVH of *static* bodies at compile time, so editing ``model.geom_pos`` of a static
geom at runtime silently has no effect on collisions. Every re-positionable piece of terrain
(slabs, ramps, obstacles) is therefore a **mocap body**, positioned through ``data.mocap_pos``.
Because ``mj_resetData`` restores mocap poses to their compiled values, the authoritative
mocap state lives on this class and ``apply_mocap`` re-applies it after every reset.

The per-body BVH (``bvh_aabb``) is also compile-time, so *resizing* a geom at runtime leaves a
stale collision volume. Every slot therefore has a fixed compile-time size (slabs are 200 m
long, ramps 3 m, obstacles come in two fixed kinds) and is only ever moved or rotated.

Body-level contype/conaffinity are also aggregated at compile time, so a slot compiled with
``contype=0`` is filtered out at the body level no matter what is written into ``geom_contype``
later. All slots are compiled collidable and "deactivated" by parking them 50 m below the road.

Layout along +x (heights above the road surface at z = 0):

    walk A (top z = h) | kerb-down | road | kerb-up | walk B (top z = h)

With h = 0 the world degenerates to flat ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin

import mujoco
import numpy as np

from .vehicle_mjcf import (
    VehicleParams,
    vehicle_actuators_xml,
    vehicle_contact_excludes_xml,
    vehicle_sensors_xml,
    vehicle_xml,
)

PARK = np.array([0.0, 0.0, -50.0])
PED_RADIUS = 0.22
PED_HALF_LEN = 0.55                       # capsule half length -> 1.54 m pedestrian
PED_CENTER_Z = PED_HALF_LEN + PED_RADIUS  # capsule centre above its footing surface
GROUP_WORLD, GROUP_PED, GROUP_ROBOT = 0, 1, 2
_IDENT_QUAT = np.array([1.0, 0.0, 0.0, 0.0])
# fixed obstacle kinds by slot parity: (radius, half height): bollard, planter
OBS_SIZES = ((0.12, 0.45), (0.28, 0.35))
OBS_RGBA = ("0.95 0.75 0.1 1", "0.35 0.6 0.3 1")


@dataclass(frozen=True)
class WorldSpec:
    """Compile-time structure (changing it requires a recompile)."""

    n_ped: int = 6
    n_obstacle: int = 6
    walk_halfwidth: float = 1.5
    slab_len: float = 200.0     # fixed slab length (only moved, never resized)
    ramp_len: float = 3.0       # fixed dropped-kerb ramp length
    timestep: float = 0.002


def _mocap_box(name: str, gname: str, half: tuple, material: str, friction: float,
               contype: int = 1) -> str:
    return (
        f'    <body name="{name}" mocap="true" pos="0 0 {PARK[2]}">\n'
        f'      <geom name="{gname}" type="box" size="{half[0]} {half[1]} {half[2]}" material="{material}" '
        f'friction="{friction} 0.005 0.0001" group="{GROUP_WORLD}" contype="{contype}" conaffinity="{contype}"/>\n'
        f'    </body>'
    )


def build_xml(veh: VehicleParams, spec: WorldSpec | None = None, *, start_x: float = 0.0,
              start_y: float = 0.0, start_h: float = 0.0, start_yaw: float = 0.0) -> str:
    spec = spec or WorldSpec()
    hw = spec.walk_halfwidth
    half_slab = (spec.slab_len / 2, hw, 0.5)
    half_ramp = (spec.ramp_len / 2, hw, 0.02)
    mu = veh.friction
    slabs = "\n".join([
        _mocap_box("kerb_a", "walk_a", half_slab, "walk", mu),
        _mocap_box("kerb_b", "walk_b", half_slab, "walk", mu),
        _mocap_box("kerb_ra", "ramp_a", half_ramp, "walk", mu),
        _mocap_box("kerb_rb", "ramp_b", half_ramp, "walk", mu),
    ])
    obst = "\n".join(
        f'''    <body name="obs_{i}" mocap="true" pos="0 0 {PARK[2]}">
      <geom name="obs_geom_{i}" type="cylinder" size="0.15 0.4" rgba="0.9 0.55 0.1 1"
            group="{GROUP_WORLD}" contype="0" conaffinity="0"/>
    </body>'''
        for i in range(spec.n_obstacle)
    )
    peds = "\n".join(
        f'''    <body name="ped_{i}" mocap="true" pos="0 {2 * i} {PARK[2]}">
      <geom name="ped_geom_{i}" type="capsule" size="{PED_RADIUS} {PED_HALF_LEN}" rgba="0.2 0.45 0.85 1"
            group="{GROUP_PED}" contype="2" conaffinity="1"/>
    </body>'''
        for i in range(spec.n_ped)
    )
    spawn_z = start_h + veh.nominal_height
    return f"""<mujoco model="aed_rover_world">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{spec.timestep}" integrator="implicitfast" gravity="0 0 -9.81" cone="pyramidal"
          iterations="30" ls_iterations="12" noslip_iterations="0"/>
  <size memory="8M"/>
  <!-- znear/zfar are in units of the model extent; the 200-400 m slabs and road plane would inflate
       the automatic extent and clip everything near the camera, so it is pinned explicitly. -->
  <statistic extent="10" center="15 0 0"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="4096" offsamples="4"/>
    <headlight ambient="0.32 0.32 0.32" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05"/>
    <map znear="0.004" zfar="40" shadowclip="1.0" shadowscale="0.6"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.62 0.75 0.9" rgb2="0.94 0.95 0.97" width="512" height="512"/>
    <texture name="tex_road" type="2d" builtin="checker" rgb1="0.30 0.30 0.32" rgb2="0.33 0.33 0.35"
             width="256" height="256" mark="edge" markrgb="0.42 0.42 0.44"/>
    <texture name="tex_walk" type="2d" builtin="checker" rgb1="0.72 0.7 0.66" rgb2="0.66 0.64 0.6"
             width="256" height="256" mark="edge" markrgb="0.5 0.49 0.46"/>
    <material name="road" texture="tex_road" texrepeat="60 60" texuniform="true" reflectance="0.0"/>
    <material name="walk" texture="tex_walk" texrepeat="1 1" texuniform="false" reflectance="0.0"/>
  </asset>
  <contact>
    {vehicle_contact_excludes_xml()}
  </contact>
  <worldbody>
    <light name="sun" directional="true" pos="0 0 30" dir="-0.35 0.25 -1" diffuse="0.55 0.55 0.52"
           specular="0.1 0.1 0.1" castshadow="true"/>
    <geom name="road" type="plane" size="400 400 0.1" material="road" friction="{mu} 0.005 0.0001"
          group="{GROUP_WORLD}"/>
{slabs}
{obst}
{peds}
    <camera name="chase" mode="trackcom" target="chassis" pos="-2.4 -2.0 1.5" xyaxes="0.64 -0.77 0 0.33 0.27 0.9"/>
    <camera name="side" mode="trackcom" target="chassis" pos="0 -3.2 0.9" xyaxes="1 0 0 0 0.2 1"/>
    <camera name="top" mode="trackcom" target="chassis" pos="0 0 9" xyaxes="1 0 0 0 1 0"/>
{vehicle_xml(veh, spawn_z=spawn_z, spawn_x=start_x, spawn_y=start_y, spawn_yaw=start_yaw)}
  </worldbody>
{vehicle_actuators_xml(veh)}
{vehicle_sensors_xml()}
</mujoco>
"""


class World:
    """Compiled model + named-id caches + slot editing helpers."""

    def __init__(self, veh: VehicleParams | None = None, spec: WorldSpec | None = None, **spawn):
        self.veh = veh or VehicleParams()
        self.spec = spec or WorldSpec()
        self.xml = build_xml(self.veh, self.spec, **spawn)
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        m = self.model
        gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)  # noqa: E731
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        mid = lambda n: int(m.body_mocapid[bid(n)])  # noqa: E731
        self.g_walk_a, self.g_walk_b = gid("walk_a"), gid("walk_b")
        self.g_ramp_a, self.g_ramp_b = gid("ramp_a"), gid("ramp_b")
        self.m_walk_a, self.m_walk_b = mid("kerb_a"), mid("kerb_b")
        self.m_ramp_a, self.m_ramp_b = mid("kerb_ra"), mid("kerb_rb")
        self.g_obs = np.array([gid(f"obs_geom_{i}") for i in range(self.spec.n_obstacle)], dtype=int)
        self.m_obs = np.array([mid(f"obs_{i}") for i in range(self.spec.n_obstacle)], dtype=int)
        self.g_tyre = np.array([gid(f"tyre_{t}") for t in ("fl", "fr", "rl", "rr")], dtype=int)
        self.g_ped = np.array([gid(f"ped_geom_{i}") for i in range(self.spec.n_ped)], dtype=int)
        self.m_ped = np.array([mid(f"ped_{i}") for i in range(self.spec.n_ped)], dtype=int)
        self.g_road = gid("road")
        self.b_chassis = bid("chassis")
        self.b_payload = bid("payload")
        self._payload_mass0 = float(m.body_mass[self.b_payload])
        self._payload_inertia0 = m.body_inertia[self.b_payload].copy()
        self._mpos = self.data.mocap_pos.copy()
        self._mquat = self.data.mocap_quat.copy()
        self.clear_all()

    # ----------------------------------------------------------- mocap state
    def apply_mocap(self) -> None:
        """Re-apply the authoritative mocap poses (call after ``mj_resetData``)."""
        self.data.mocap_pos[:] = self._mpos
        self.data.mocap_quat[:] = self._mquat

    def _place(self, mocap_id: int, pos, quat=_IDENT_QUAT) -> None:
        self._mpos[mocap_id] = pos
        self._mquat[mocap_id] = quat
        self.data.mocap_pos[mocap_id] = pos
        self.data.mocap_quat[mocap_id] = quat

    def _park(self, mocap_id: int, gid=None) -> None:
        self._place(mocap_id, PARK)

    def clear_all(self) -> None:
        for mid_, g in ((self.m_walk_a, self.g_walk_a), (self.m_walk_b, self.g_walk_b),
                        (self.m_ramp_a, self.g_ramp_a), (self.m_ramp_b, self.g_ramp_b)):
            self._park(mid_, g)
        for i in range(self.spec.n_obstacle):
            self._park(int(self.m_obs[i]), int(self.g_obs[i]))
        for i in range(self.spec.n_ped):
            self.park_pedestrian(i)

    # --------------------------------------------------------------- terrain
    def set_flat(self) -> None:
        """No kerb: ground is the road plane (slabs parked)."""
        for mid_, g in ((self.m_walk_a, self.g_walk_a), (self.m_walk_b, self.g_walk_b),
                        (self.m_ramp_a, self.g_ramp_a), (self.m_ramp_b, self.g_ramp_b)):
            self._park(mid_, g)

    def set_crossing(self, kerb_h: float, x_down: float, x_up: float, *, ramp_down: bool = False,
                     ramp_up: bool = False, ramp_slope: float = 1.0 / 12.0) -> None:
        """Walk A ends at ``x_down``; road until ``x_up``; walk B starts at ``x_up`` (top at ``kerb_h``).

        Slabs have fixed length (see ``WorldSpec.slab_len``) and extend away from the road.
        ``ramp_*`` replaces the vertical kerb by a dropped-kerb ramp of the given rise/run whose
        upper edge meets the kerb lip and which extends over the road side.
        """
        h = float(kerb_h)
        if h <= 1e-6:
            self.set_flat()
            return
        half = 0.5 * self.spec.slab_len
        self._place(self.m_walk_a, (x_down - half, 0.0, h - 0.5))
        self._place(self.m_walk_b, (x_up + half, 0.0, h - 0.5))
        th = atan2(ramp_slope, 1.0)
        c, s_, t, hl = cos(th), sin(th), 0.02, 0.5 * self.spec.ramp_len
        # ramp A descends toward +x from the lip at x_down (rotation about +y by +th)
        if ramp_down:
            self._place(self.m_ramp_a, (x_down + hl * c - t * s_, 0.0, h - hl * s_ - t * c),
                        (cos(th / 2), 0.0, sin(th / 2), 0.0))
        else:
            self._park(self.m_ramp_a, self.g_ramp_a)
        # ramp B rises toward +x to the lip at x_up (rotation about +y by -th)
        if ramp_up:
            self._place(self.m_ramp_b, (x_up - hl * c + t * s_, 0.0, h - hl * s_ - t * c),
                        (cos(th / 2), 0.0, -sin(th / 2), 0.0))
        else:
            self._park(self.m_ramp_b, self.g_ramp_b)

    def set_obstacle(self, slot: int, x: float, y: float, z_surface: float) -> None:
        """Activate obstacle ``slot`` (fixed kind: even = bollard, odd = planter) on the surface."""
        half_h = OBS_SIZES[slot % 2][1]
        self._place(int(self.m_obs[slot]), (x, y, z_surface + half_h))

    def clear_obstacles(self) -> None:
        for i in range(self.spec.n_obstacle):
            self._park(int(self.m_obs[i]), int(self.g_obs[i]))

    # ------------------------------------------------------------ pedestrians
    def place_pedestrian(self, slot: int, x: float, y: float, z_surface: float) -> None:
        self._place(int(self.m_ped[slot]), (x, y, z_surface + PED_CENTER_Z))

    def move_pedestrian(self, slot: int, x: float, y: float, z_surface: float) -> None:
        pos = (x, y, z_surface + PED_CENTER_Z)
        mid_ = int(self.m_ped[slot])
        self._mpos[mid_] = pos
        self.data.mocap_pos[mid_] = pos

    def park_pedestrian(self, slot: int) -> None:
        self._place(int(self.m_ped[slot]), (0.0, 4.0 + 2.0 * slot, PARK[2]))

    # -------------------------------------------------------------- scenarios
    def apply_scenario(self, sc, *, tyre_mu: float | None = None, ground_mu: float | None = None,
                       payload_mass: float | None = None, with_pedestrians: bool = True) -> None:
        """Load a ``Scenario``'s terrain, obstacles and physical parameters into this world.

        ``tyre_mu`` / ``ground_mu`` / ``payload_mass`` override the scenario's true values, which is
        how a planner builds its (deliberately mismatched) internal model.
        """
        self.clear_all()
        if sc.has_kerb:
            self.set_crossing(sc.kerb_h, sc.x_down, sc.x_up, ramp_down=sc.ramp_down, ramp_up=sc.ramp_up)
        else:
            self.set_flat()
        self.set_tyre_friction(sc.tyre_mu if tyre_mu is None else tyre_mu)
        self.set_ground_friction(sc.ground_mu if ground_mu is None else ground_mu)
        mass = sc.payload_mass if payload_mass is None else payload_mass
        if mass != float(self.model.body_mass[self.b_payload]):
            self.set_payload_mass(mass)
        for ob in sc.obstacles:
            self.set_obstacle(ob.slot, ob.x, ob.y, ob.z_surface)
        _ = with_pedestrians   # pedestrians are placed by the environment's crowd model

    # --------------------------------------------------------------- physics
    def set_tyre_friction(self, mu: float) -> None:
        self.model.geom_friction[self.g_tyre, 0] = mu

    def set_ground_friction(self, mu: float) -> None:
        for g in (self.g_road, self.g_walk_a, self.g_walk_b, self.g_ramp_a, self.g_ramp_b):
            self.model.geom_friction[g, 0] = mu

    def set_payload_mass(self, mass: float) -> None:
        """Set the payload mass; inertia is recomputed from the *compiled* values, never rescaled in place.

        Rescaling the current inertia by a ratio leaves a rounding residue that depends on every mass
        this model has had before. Contact dynamics amplify it, so the same seed gave different episodes
        depending on which jobs a worker had run earlier (`tests/test_env.py`, engineering note 15).
        """
        m = self.model
        m.body_mass[self.b_payload] = mass
        m.body_inertia[self.b_payload] = self._payload_inertia0 * (mass / self._payload_mass0)
        mujoco.mj_setConst(m, self.data)
