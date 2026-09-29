"""Kerb traversability study: success and payload shock versus kerb height, approach speed,
approach angle, wheel radius and friction, for kerb-up and kerb-down.

The trial function is module-level and picklable so grids run under ``aedrover.parallel.pmap``.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

from .metrics import rover_shock
from .rover import Rover
from .vehicle_mjcf import VehicleParams
from .world import World


@dataclass(frozen=True)
class KerbTrial:
    direction: str = "up"          # "up" or "down"
    kerb_h: float = 0.12
    speed: float = 1.0
    angle_deg: float = 0.0
    wheel_radius: float = 0.15
    mu: float = 1.0
    ramp: bool = False
    susp_c: float = 350.0
    susp_k: float = 4500.0
    iso_kz: float = 2200.0
    iso_cz: float = 180.0
    motor_peak_torque: float = 25.0
    payload_mass: float = 4.0


@functools.lru_cache(maxsize=32)
def _world_for(veh: VehicleParams) -> World:
    return World(veh)


def run_kerb_trial(t: KerbTrial, x_kerb: float = 7.0) -> dict:
    veh = VehicleParams(wheel_radius=t.wheel_radius, susp_c=t.susp_c, susp_k=t.susp_k,
                        iso_kz=t.iso_kz, iso_cz=t.iso_cz, motor_peak_torque=t.motor_peak_torque)
    w = _world_for(veh)
    w.clear_all()
    if t.payload_mass != w.model.body_mass[w.b_payload]:
        w.set_payload_mass(t.payload_mass)
    up = t.direction == "up"
    if up:
        w.set_crossing(t.kerb_h, x_down=-9.0, x_up=x_kerb, ramp_up=t.ramp)
        z0, x_goal = 0.0, x_kerb + 2.5
    else:
        w.set_crossing(t.kerb_h, x_down=x_kerb, x_up=x_kerb + 60.0, ramp_down=t.ramp)
        z0, x_goal = t.kerb_h, x_kerb + 4.0
    w.set_tyre_friction(t.mu)
    w.set_ground_friction(t.mu)
    r = Rover(w)
    ang = np.radians(t.angle_deg)
    # start far enough back that the rover is at speed and (for oblique runs) on the right line
    x_start = 0.0
    y_start = -np.tan(ang) * (x_kerb - x_start)
    r.reset(x_start, y_start, z0, yaw=ang, speed=t.speed, settle_s=0.3)
    outcome, t_stall0, x_prev = "timeout", None, r.pos[0]
    max_t = 8.0 + (x_kerb + 6.0) / max(t.speed, 0.2)
    for _ in range(int(max_t / r.control_dt)):
        yaw = r.yaw_pitch_roll()[0]
        delta = float(np.clip(-2.0 * (yaw - ang), -0.5, 0.5))
        r.step(t.speed, delta)
        if not r.is_finite():
            outcome = "nan"
            break
        _, pitch, roll = r.yaw_pitch_roll()
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            outcome = "rollover"
            break
        if r.pos[0] >= x_goal and (r.pos[2] > t.kerb_h + 0.1 if up else True):
            outcome = "ok"
            break
        if r.pos[0] > x_kerb - 1.0:      # near the kerb: watch for a stall
            if abs(r.pos[0] - x_prev) < 0.002:
                t_stall0 = r.d.time if t_stall0 is None else t_stall0
                if r.d.time - t_stall0 > 2.0:
                    outcome = "stall"
                    break
            else:
                t_stall0 = None
        x_prev = r.pos[0]
    sh = rover_shock(r)
    return {
        "direction": t.direction, "kerb_h": t.kerb_h, "speed": t.speed, "angle_deg": t.angle_deg,
        "wheel_radius": t.wheel_radius, "mu": t.mu, "ramp": t.ramp, "outcome": outcome,
        "success": outcome == "ok", "peak_g": sh.peak_g, "peak_vertical_g": sh.peak_vertical_g,
        "max_pitch_deg": float(np.degrees(abs(r.yaw_pitch_roll()[1]))),
    }
