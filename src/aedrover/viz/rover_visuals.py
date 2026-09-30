"""The rover's look: shell, AED case, knobby wheels with visible spokes, suspension arms, lidar, beacon and lights.

All parts are visual-only geoms injected into the physics bodies (chassis, payload, knuckles, wheels), so
they move with the real suspension and isolator; the original box/cylinder physics geoms are shrunk to a
point in the render model (which is never simulated).
"""

from __future__ import annotations

import math

from ..sim.vehicle_mjcf import VehicleParams
from .dressing import Materials, geom
from .render_model import RenderXml, hide_physics_geoms

TAGS = (("fl", 1, 1), ("fr", 1, -1), ("rl", -1, 1), ("rr", -1, -1))     # tag, sx (front +), sy (left +)


def dress_rover(rx: RenderXml, veh: VehicleParams, mats: Materials) -> None:
    paint = mats("rv_paint", (0.74, 0.07, 0.05), metallic=0.35, roughness=0.30)
    graphite = mats("rv_graphite", (0.06, 0.065, 0.07), metallic=0.6, roughness=0.5)
    rubber = mats("rv_rubber", (0.025, 0.025, 0.028), roughness=0.92)
    case = mats("rv_case", (0.93, 0.94, 0.92), roughness=0.28)
    green = mats("rv_green", (0.05, 0.62, 0.22), roughness=0.4, emission=0.7)
    silver = mats("rv_silver", (0.78, 0.79, 0.81), metallic=1.0, roughness=0.22)
    amber = mats("rv_amber", (1.0, 0.52, 0.03), roughness=0.4, emission=1.6)
    lamp = mats("rv_lamp", (1.0, 0.95, 0.80), roughness=0.3, emission=4.0)
    tail = mats("rv_tail", (0.9, 0.04, 0.03), roughness=0.3, emission=3.0)
    lidar = mats("rv_lidar", (0.05, 0.05, 0.06), metallic=0.3, roughness=0.3)
    ring = mats("rv_ring", (0.1, 0.8, 1.0), roughness=0.3, emission=4.0)
    white = mats("rv_stripe", (0.92, 0.92, 0.9), roughness=0.4)

    hx, hy, hz = veh.chassis_half
    # ---- chassis: frame, red shell, stripes, bumpers, lamps, sensor mast, beacon
    ch = [geom("box", (0, 0, -0.012), (hx, hy - 0.03, 0.045), graphite),
          geom("box", (0, 0, 0.035), (hx - 0.01, hy, 0.030), paint),
          geom("box", (0.27, 0, 0.078), (0.15, hy - 0.02, 0.022), paint),
          geom("box", (-0.26, 0, 0.078), (0.15, hy - 0.02, 0.022), paint),
          geom("box", (0.27, 0.07, 0.101), (0.14, 0.011, 0.002), white),
          geom("box", (0.27, -0.07, 0.101), (0.14, 0.011, 0.002), white),
          geom("box", (-0.26, 0.07, 0.101), (0.14, 0.011, 0.002), white),
          geom("box", (-0.26, -0.07, 0.101), (0.14, 0.011, 0.002), white),
          geom("box", (hx + 0.025, 0, 0.0), (0.03, hy + 0.01, 0.04), rubber),
          geom("box", (-hx - 0.025, 0, 0.0), (0.03, hy + 0.01, 0.04), rubber),
          geom("box", (hx + 0.052, 0, 0.012), (0.004, hy - 0.05, 0.012), lamp),
          geom("box", (-hx - 0.052, hy - 0.06, 0.012), (0.004, 0.04, 0.012), tail),
          geom("box", (-hx - 0.052, -hy + 0.06, 0.012), (0.004, 0.04, 0.012), tail),
          # lidar: mast, puck and a glowing ring
          geom("cylinder", (0.30, 0, 0.125), (0.012, 0.03), graphite),
          geom("cylinder", (0.30, 0, 0.175), (0.052, 0.028), lidar),
          geom("cylinder", (0.30, 0, 0.176), (0.054, 0.004), ring),
          # beacon mast and amber light
          geom("cylinder", (-0.30, 0, 0.20), (0.012, 0.12), graphite),
          geom("sphere", (-0.30, 0, 0.335), (0.04,), amber)]
    rx.body_xml.setdefault("chassis", []).extend(ch)

    # ---- payload case (rides on the isolator): white shell, rubber rim, latches, handle, green crosses
    phx, phy, phz = veh.payload_half
    pl = [geom("box", (0, 0, -0.015), (phx, phy, phz - 0.02), case),
          geom("box", (0, 0, phz - 0.030), (phx - 0.008, phy - 0.008, 0.022), case),
          geom("box", (0, 0, 0.004), (phx + 0.006, phy + 0.006, 0.007), rubber),
          geom("box", (phx + 0.001, 0.06, -0.01), (0.004, 0.018, 0.014), graphite),
          geom("box", (phx + 0.001, -0.06, -0.01), (0.004, 0.018, 0.014), graphite),
          geom("box", (0, 0.06, phz + 0.004), (0.012, 0.008, 0.012), graphite),
          geom("box", (0, -0.06, phz + 0.004), (0.012, 0.008, 0.012), graphite),
          geom("box", (0, 0, phz + 0.017), (0.012, 0.068, 0.006), graphite),
          geom("box", (0, 0, phz - 0.004), (0.075, 0.020, 0.002), green),
          geom("box", (0, 0, phz - 0.004), (0.020, 0.075, 0.002), green)]
    for sgn in (-1, 1):                                   # crosses on both side faces and on front/back faces
        pl += [geom("box", (0, sgn * (phy + 0.0015), 0.0), (0.05, 0.002, 0.016), green),
               geom("box", (0, sgn * (phy + 0.0015), 0.0), (0.016, 0.002, 0.05), green),
               geom("box", (sgn * (phx + 0.0015), 0, 0.0), (0.002, 0.04, 0.014), green),
               geom("box", (sgn * (phx + 0.0015), 0, 0.0), (0.002, 0.014, 0.04), green)]
    rx.body_xml.setdefault("payload", []).extend(pl)

    # ---- wheels (spin with the wheel body) and suspension arms (move with the knuckle)
    r, hw = veh.wheel_radius, veh.wheel_half_width
    for tag, _sx, sy in TAGS:
        wheel = [geom("cylinder", (0, 0, 0), (r, hw), rubber, euler=(math.pi / 2, 0, 0)),
                 geom("cylinder", (0, sy * 0.004, 0), (r * 0.56, hw + 0.003), silver, euler=(math.pi / 2, 0, 0)),
                 geom("cylinder", (0, sy * (hw + 0.006), 0), (0.05, 0.007), graphite, euler=(math.pi / 2, 0, 0))]
        for k in range(4):                                # an 8-spoke star on the outer face
            wheel.append(geom("box", (0, sy * (hw + 0.002), 0), (r * 0.52, 0.006, 0.013), graphite, euler=(0, k * math.pi / 4, 0)))
        for k in range(18):                               # tread lugs around the circumference
            a = 2 * math.pi * k / 18
            wheel.append(geom("box", (math.cos(a) * (r + 0.004), 0, math.sin(a) * (r + 0.004)), (0.011, hw * 0.92, 0.014),
                              rubber, euler=(0, -a, 0)))
        rx.body_xml.setdefault(f"wheel_{tag}", []).extend(wheel)
        rx.body_xml.setdefault(f"knuckle_{tag}", []).extend([
            geom("box", (0, -sy * 0.12, 0.02), (0.022, 0.12, 0.018), graphite),
            geom("cylinder", (0, -sy * 0.14, 0.09), (0.014, 0.075), silver),
            geom("cylinder", (0, -sy * 0.14, 0.09), (0.021, 0.04), graphite)])

    hide_physics_geoms(rx, {"chassis_geom": 3, "chassis_nose": 3, "payload_geom": 3, "payload_cross": 3,
                            **{f"tyre_{t}": 2 for t, _, _ in TAGS}})
