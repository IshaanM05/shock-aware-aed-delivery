"""Visual-only set dressing baked per episode from its scenario (nothing here is ever simulated).

The physics world is a raised 3 m sidewalk strip and a road plane. Dressing makes it read as a city
street crossing a main road: the pavement is widened with extra geoms on the *same* mocap slabs (so kerb
height and ramps match the physics exactly), kerbstones and road paint are baked at the scenario's
positions, and building rows, street lamps, trees and parked cars frame the route. Nothing is placed
inside the lateral corridor the rover may use (``|y| < CORRIDOR_HALF_WIDTH``).
"""

from __future__ import annotations

import math

import numpy as np

from . import assets as A
from .look import Look
from .recording import Recording
from .render_model import RenderXml, _pbr_material

PAVE_HALF_Y = 5.0          # the widened pavement reaches this lateral distance on each side
BUILDING_FRONT_Y = 5.0     # building fronts stand on the pavement edge
STREET_X0, STREET_X1_PAD = -45.0, 40.0
FACADE_TILE_M = 12.0       # one facade texture covers 4 bays x 4 floors = 12 m x 12 m

# tints multiplied into the facade albedo: cream, terracotta, pale blue, sand, sage, grey
FACADE_TINTS = ((1.0, 0.93, 0.80), (0.95, 0.72, 0.60), (0.80, 0.86, 0.92), (0.96, 0.86, 0.68),
                (0.78, 0.86, 0.74), (0.86, 0.85, 0.84))
CAR_COLOURS = ((0.92, 0.92, 0.90), (0.72, 0.74, 0.78), (0.10, 0.16, 0.30), (0.45, 0.08, 0.10), (0.25, 0.27, 0.29),
               (0.85, 0.72, 0.30))


def _v(*xs: float) -> str:
    return " ".join(f"{x:.5g}" for x in xs)


def geom(kind: str, pos, size, mat: str, *, euler=None, quat=None, extra: str = "") -> str:
    """A visual-only geom (no contacts) for the worldbody or injected into a body."""
    rot = f' euler="{_v(*euler)}"' if euler is not None else (f' quat="{_v(*quat)}"' if quat is not None else "")
    return (f'<geom type="{kind}" pos="{_v(*pos)}" size="{_v(*size)}"{rot} material="{mat}" '
            f'contype="0" conaffinity="0" group="1"{extra}/>')


class Materials:
    """Registers plain PBR materials once and hands back their names."""

    def __init__(self, rx: RenderXml) -> None:
        self.rx, self._have = rx, set()

    def __call__(self, name: str, rgba, *, metallic: float = 0.0, roughness: float = 0.7, emission: float = 0.0,
                 reflectance: float = 0.0) -> str:
        if name not in self._have:
            self._have.add(name)
            r = _v(*rgba) if len(rgba) == 4 else _v(*rgba, 1.0)
            self.rx.asset_xml.append(f'<material name="{name}" rgba="{r}" metallic="{metallic}" roughness="{roughness}" '
                                     f'emission="{emission}" reflectance="{reflectance}"/>')
        return name


def _scenario(rec: Recording) -> dict:
    sc = dict(rec.meta["scenario"])
    sc["has_kerb"] = sc["kerb_h"] > 1e-6
    return sc


def dress_scene(rx: RenderXml, rec: Recording, look: Look) -> list:
    """Add the full street dressing to ``rx`` for the recorded scenario; returns the per-frame animators."""
    sc = _scenario(rec)
    mats = Materials(rx)
    rng = np.random.default_rng(look.seed * 1009 + int(sc["seed"]))
    _street(rx, sc, mats)
    x_end = sc["x_goal"] + STREET_X1_PAD
    if sc["has_kerb"]:
        ranges = [(STREET_X0, sc["x_down"] - 1.0), (sc["x_up"] + 1.0, x_end)]
    else:
        ranges = [(STREET_X0, x_end)]
    _buildings(rx, ranges, rng, mats)
    _street_furniture(rx, ranges, rng, mats)
    if sc["has_kerb"]:
        _road_paint(rx, sc, mats)
        _parked_cars(rx, sc, rng, mats)
    _skyline(rx, sc, rng, mats)
    from ..sim.vehicle_mjcf import VehicleParams
    from .people import dress_obstacles, dress_people
    from .rover_visuals import dress_rover
    dress_rover(rx, VehicleParams.by_name(rec.meta["vehicle"]), mats)
    dress_obstacles(rx, mats)
    return [dress_people(rx, rec, rng, mats)]


# ------------------------------------------------------------------------------------- pavement
def _street(rx: RenderXml, sc: dict, mats: Materials) -> None:
    """Widen the pavement on the physics slabs, add kerbstones, and re-skin the ground when there is no kerb."""
    if not sc["has_kerb"]:
        # flat scenario: the whole ground is pavement
        rx.asset_xml[:] = [x for x in rx.asset_xml if not x.startswith('<material name="road"')]
        rx.asset_xml.append(_pbr_material("road", "pave", 2.4))
        return
    side = PAVE_HALF_Y - 1.5                     # the physical slab already covers |y| < 1.5
    for body, hx in (("kerb_a", 100.0), ("kerb_b", 100.0)):
        rx.body_xml.setdefault(body, []).extend(
            geom("box", (0, s * (1.5 + side / 2), 0), (hx, side / 2, 0.5), "walk") for s in (-1, 1))
    for body in ("kerb_ra", "kerb_rb"):
        rx.body_xml.setdefault(body, []).extend(
            geom("box", (0, s * (1.5 + side / 2), 0), (1.5, side / 2, 0.02), "walk") for s in (-1, 1))
    kerb = mats("kerbstone", (0.66, 0.65, 0.62), roughness=0.92)
    h = sc["kerb_h"]
    for x, dirn, ramp in ((sc["x_down"], 1.0, sc["ramp_down"]), (sc["x_up"], -1.0, sc["ramp_up"])):
        if ramp:
            continue                              # the dropped-kerb ramp replaces the vertical face in the corridor
        # a proud facing strip so the vertical face reads as concrete, not as pavement texture
        rx.world_xml.append(geom("box", (x + dirn * 0.012, 0, h / 2), (0.014, PAVE_HALF_Y, h / 2), kerb))
    # tactile warning strip (yellow, raised dots suggested by a darker lip) on the pavement side of each kerb
    tact = mats("tactile", (0.86, 0.66, 0.10), roughness=0.75)
    for x, dirn in ((sc["x_down"], -1.0), (sc["x_up"], 1.0)):
        rx.world_xml.append(geom("box", (x + dirn * 0.45, 0, sc["kerb_h"] + 0.004), (0.4, 1.7, 0.004), tact))


# ------------------------------------------------------------------------------------- buildings
def _facade_materials(rx: RenderXml, mats: Materials) -> list[str]:
    rx.files.update(A.facade_maps())
    rx.asset_xml += ['<texture name="t_fac_a" type="2d" file="facade_albedo.png"/>',
                     '<texture name="t_fac_n" type="2d" file="facade_normal.png"/>',
                     '<texture name="t_fac_o" type="2d" file="facade_orm.png"/>',
                     '<texture name="t_fac_e" type="2d" file="facade_emissive.png"/>']
    names = []
    rep = 2.0 / FACADE_TILE_M
    for k, tint in enumerate(FACADE_TINTS):
        name = f"facade_{k}"
        rx.asset_xml.append(
            f'<material name="{name}" rgba="{_v(*tint, 1.0)}" texrepeat="{rep:.5g} {rep:.5g}" texuniform="true" '
            f'metallic="0" roughness="1" emission="1"><layer texture="t_fac_a" role="rgb"/>'
            f'<layer texture="t_fac_n" role="normal"/><layer texture="t_fac_o" role="orm"/>'
            f'<layer texture="t_fac_e" role="emissive"/></material>')
        names.append(name)
    return names


def _buildings(rx: RenderXml, ranges, rng: np.random.Generator, mats: Materials) -> None:
    facades = _facade_materials(rx, mats)
    roof = mats("roof", (0.22, 0.21, 0.21), roughness=0.95)
    for side in (-1.0, 1.0):
        for x0, x1 in ranges:
            x = x0
            while x < x1 - 4.0:
                w = float(rng.uniform(6.0, 12.0))
                w = min(w, x1 - x)
                depth = float(rng.uniform(8.0, 13.0))
                height = float(rng.choice([9.0, 12.0, 15.0, 18.0, 21.0]) + rng.uniform(-1.0, 1.0))
                cx, cy = x + w / 2, side * (BUILDING_FRONT_Y + depth / 2)
                mat = facades[int(rng.integers(len(facades)))]
                rx.world_xml.append(geom("box", (cx, cy, height / 2), (w / 2, depth / 2, height / 2), mat))
                rx.world_xml.append(geom("box", (cx, cy, height + 0.12), (w / 2 + 0.15, depth / 2 + 0.15, 0.12), roof))
                x += w + float(rng.uniform(0.0, 0.6))                 # mostly continuous frontage


# -------------------------------------------------------------------------- lamps and trees
def _street_furniture(rx: RenderXml, ranges, rng: np.random.Generator, mats: Materials) -> None:
    metal = mats("lamp_metal", (0.10, 0.11, 0.12), metallic=0.8, roughness=0.45)
    bulb = mats("lamp_bulb", (1.0, 0.78, 0.45), roughness=0.4, emission=8.0)
    trunk = mats("trunk", (0.26, 0.17, 0.10), roughness=0.95)
    leaves = [mats(f"leaf_{k}", c, roughness=0.9) for k, c in enumerate(((0.10, 0.27, 0.08), (0.14, 0.33, 0.10),
                                                                          (0.08, 0.22, 0.09)))]
    pot = mats("planter_pot", (0.45, 0.42, 0.38), roughness=0.9)
    for x0, x1 in ranges:
        for side in (-1.0, 1.0):
            x = x0 + 6.0
            k = 0
            while x < x1 - 3.0:
                y = side * 4.4
                if k % 2 == 0:                                             # lamp post with an arm and a glowing head
                    rx.world_xml += [geom("cylinder", (x, y, 3.2), (0.06, 3.2), metal),
                                     geom("box", (x, y - side * 0.7, 6.3), (0.05, 0.75, 0.05), metal),
                                     geom("box", (x, y - side * 1.4, 6.22), (0.32, 0.14, 0.05), metal),
                                     geom("sphere", (x, y - side * 1.4, 6.12), (0.2,), bulb)]
                    rx.world_xml.append(
                        f'<light type="point" pos="{_v(x, y - side * 1.4, 6.0)}" diffuse="1 0.72 0.42" '
                        f'intensity="900000" castshadow="false"/>')
                else:                                                      # tree
                    th = float(rng.uniform(2.6, 3.4))
                    rx.world_xml.append(geom("cylinder", (x, y, th / 2), (0.14, th / 2), trunk))
                    rx.world_xml.append(geom("cylinder", (x, y, 0.22), (0.4, 0.22), pot))
                    for _ in range(int(rng.integers(4, 7))):
                        off = rng.normal(0, 0.55, size=2)
                        r = float(rng.uniform(0.9, 1.45))
                        rx.world_xml.append(geom("ellipsoid", (x + off[0], y + off[1], th + float(rng.uniform(0.3, 1.6))),
                                                 (r, r, r * 0.85), leaves[int(rng.integers(len(leaves)))]))
                x += 7.0
                k += 1


# ------------------------------------------------------------------------- road paint and cars
def _road_paint(rx: RenderXml, sc: dict, mats: Materials) -> None:
    white = mats("paint_white", (0.88, 0.88, 0.84), roughness=0.55)
    yellow = mats("paint_yellow", (0.82, 0.62, 0.08), roughness=0.55)
    x0, x1 = sc["x_down"], sc["x_up"]
    xc = 0.5 * (x0 + x1)
    z = 0.004
    # edge lines (continuous) and a dashed centre line along the main road, which runs along y
    for xe in (x0 + 0.35, x1 - 0.35):
        rx.world_xml.append(geom("box", (xe, 0, z), (0.07, 70.0, 0.003), white))
    for y in np.arange(-66.0, 66.0, 4.0):
        if abs(y) < 2.4:
            continue                               # the crossing interrupts the centre line
        rx.world_xml.append(geom("box", (xc, y + 1.0, z), (0.07, 1.0, 0.003), yellow))
    # zebra crossing: bars parallel to the traffic (y), spaced along the walking direction (x)
    for x in np.arange(x0 + 0.7, x1 - 0.5, 0.9):
        rx.world_xml.append(geom("box", (x, 0, z + 0.001), (0.22, 1.75, 0.003), white))
    # stop lines either side of the crossing
    for y in (-2.6, 2.6):
        rx.world_xml.append(geom("box", (xc, y, z), ((x1 - x0) / 2 - 0.4, 0.12, 0.003), white))


def _car(rx: RenderXml, x: float, y: float, yaw_deg: float, colour, mats: Materials, tag: str) -> None:
    body = mats(f"car_{tag}", (*colour, 1.0), metallic=0.55, roughness=0.32, reflectance=0.0)
    glass = mats("car_glass", (0.03, 0.05, 0.08), roughness=0.05)
    tyre = mats("car_tyre", (0.03, 0.03, 0.035), roughness=0.85)
    head = mats("car_head", (1.0, 0.92, 0.75), roughness=0.3, emission=3.0)
    tail = mats("car_tail", (0.9, 0.05, 0.04), roughness=0.3, emission=2.5)
    c, s = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))

    def at(lx: float, ly: float, lz: float) -> tuple[float, float, float]:
        return (x + c * lx - s * ly, y + s * lx + c * ly, lz)

    q = (math.cos(math.radians(yaw_deg) / 2), 0.0, 0.0, math.sin(math.radians(yaw_deg) / 2))
    parts = [("box", at(0, 0, 0.62), (2.15, 0.88, 0.36), body), ("box", at(-0.15, 0, 1.08), (1.15, 0.80, 0.30), glass),
             ("box", at(-0.15, 0, 1.36), (1.12, 0.78, 0.03), body), ("box", at(2.12, 0.55, 0.66), (0.04, 0.2, 0.08), head),
             ("box", at(2.12, -0.55, 0.66), (0.04, 0.2, 0.08), head), ("box", at(-2.12, 0.55, 0.70), (0.04, 0.2, 0.08), tail),
             ("box", at(-2.12, -0.55, 0.70), (0.04, 0.2, 0.08), tail)]
    for kind, pos, size, m in parts:
        rx.world_xml.append(geom(kind, pos, size, m, quat=q))
    for lx in (1.3, -1.3):
        for ly in (0.85, -0.85):
            rx.world_xml.append(geom("cylinder", at(lx, ly, 0.32), (0.32, 0.11), tyre,
                                     euler=(math.pi / 2, 0.0, math.radians(yaw_deg))))


def _parked_cars(rx: RenderXml, sc: dict, rng: np.random.Generator, mats: Materials) -> None:
    lanes = (sc["x_down"] + 1.3, sc["x_up"] - 1.3)
    k = 0
    for lane, yaw in zip(lanes, (90.0, -90.0), strict=True):
        for y in np.arange(9.0, 60.0, 7.2):
            for sign in (-1.0, 1.0):
                if rng.random() < 0.35:
                    continue
                col = CAR_COLOURS[int(rng.integers(len(CAR_COLOURS)))]
                _car(rx, lane + float(rng.uniform(-0.2, 0.2)), sign * (y + float(rng.uniform(-0.6, 0.6))), yaw, col, mats, f"{k}")
                k += 1


# ----------------------------------------------------------------------------------- skyline
def _skyline(rx: RenderXml, sc: dict, rng: np.random.Generator, mats: Materials) -> None:
    """Hazy silhouettes of distant towers so the horizon is a city, not a flat line."""
    cx = 0.5 * (sc["x_goal"] + STREET_X0)
    for ring, (radius, haze, hmin, hmax) in enumerate(((260.0, 0.42, 25.0, 70.0), (420.0, 0.62, 35.0, 110.0))):
        mat = mats(f"skyline_{ring}", (0.50 + 0.3 * haze, 0.38 + 0.28 * haze, 0.36 + 0.26 * haze), roughness=1.0)
        n = 46
        for k in range(n):
            ang = 2 * math.pi * (k + rng.uniform(-0.3, 0.3)) / n
            r = radius * float(rng.uniform(0.92, 1.12))
            w = float(rng.uniform(14.0, 34.0))
            h = float(rng.uniform(hmin, hmax))
            rx.world_xml.append(geom("box", (cx + r * math.cos(ang), r * math.sin(ang), h / 2),
                                     (w / 2, w / 2 * float(rng.uniform(0.7, 1.3)), h / 2), mat,
                                     euler=(0.0, 0.0, ang + float(rng.uniform(-0.4, 0.4)))))
