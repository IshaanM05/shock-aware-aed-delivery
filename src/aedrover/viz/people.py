"""Pedestrians with a walking gait, and the bollards and planters that re-skin the obstacle slots.

Each physics pedestrian (a mocap capsule) is replaced, visually, by a figure made of separate mocap
segments appended after every physics body (thigh, shin, shoe, upper arm, forearm per side, torso, head,
hair). ``PedAnimator`` poses them every frame from the pedestrian's recorded position: heading from the
smoothed velocity, stride phase from the distance walked, so feet never slide and a standing pedestrian
stands. Nothing here is simulated.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy.ndimage import uniform_filter1d

from ..sim.world import OBS_SIZES, PED_CENTER_Z
from .dressing import Materials, geom
from .recording import Recording
from .render_model import RenderXml, hide_physics_geoms

N_PED_SLOTS = 6
PARK_Z = -10.0
STRIDE_M = 1.45                   # one full gait cycle (two steps) covers this distance

THIGH, SHIN, UPPER, FORE = 0.43, 0.43, 0.29, 0.27
HIP_H, SHOULDER_H, HEAD_H = 0.88, 1.27, 1.43
HIP_Y, SHOULDER_Y = 0.095, 0.205

SHIRTS = ((0.80, 0.16, 0.12), (0.12, 0.40, 0.72), (0.93, 0.78, 0.18), (0.18, 0.55, 0.32), (0.92, 0.92, 0.88),
          (0.55, 0.16, 0.45), (0.95, 0.52, 0.12), (0.15, 0.17, 0.20))
TROUSERS = ((0.10, 0.12, 0.20), (0.18, 0.17, 0.16), (0.30, 0.25, 0.18), (0.08, 0.08, 0.09), (0.36, 0.40, 0.46))
SKIN = ((0.56, 0.36, 0.24), (0.47, 0.29, 0.19), (0.64, 0.43, 0.30), (0.38, 0.23, 0.15), (0.72, 0.52, 0.38))
HAIR = ((0.04, 0.03, 0.03), (0.09, 0.06, 0.04), (0.14, 0.12, 0.11))

PARTS = ("thl", "thr", "shl", "shr", "fel", "fer", "ful", "fur", "uul", "uur", "torso", "head", "hair")


def _quat_z_to(v: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) rotating +z onto the direction ``v``."""
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    v = v / n
    w = 1.0 + v[2]
    if w < 1e-9:
        return np.array([0.0, 1.0, 0.0, 0.0])
    q = np.array([w, -v[1], v[0], 0.0])        # cross(z, v) = (-vy, vx, 0)
    return q / np.linalg.norm(q)


def dress_people(rx: RenderXml, rec: Recording, rng: np.random.Generator, mats: Materials) -> PedAnimator:
    n_ped = int(rec.meta["scenario"].get("n_ped", N_PED_SLOTS))
    del n_ped                                  # all six slots are dressed; parked ones stay out of sight
    shoe = mats("ped_shoe", (0.06, 0.05, 0.05), roughness=0.6)
    hide_physics_geoms(rx, {f"ped_geom_{i}": 2 for i in range(N_PED_SLOTS)})
    for i in range(N_PED_SLOTS):
        shirt = mats(f"ped{i}_shirt", (*SHIRTS[int(rng.integers(len(SHIRTS)))], 1.0), roughness=0.85)
        trou = mats(f"ped{i}_trousers", (*TROUSERS[int(rng.integers(len(TROUSERS)))], 1.0), roughness=0.85)
        skin = mats(f"ped{i}_skin", (*SKIN[int(rng.integers(len(SKIN)))], 1.0), roughness=0.55)
        hair = mats(f"ped{i}_hair", (*HAIR[int(rng.integers(len(HAIR)))], 1.0), roughness=0.6)
        spec = {"thl": ("capsule", (0.072, THIGH / 2 - 0.072), trou), "thr": ("capsule", (0.072, THIGH / 2 - 0.072), trou),
                "shl": ("capsule", (0.058, SHIN / 2 - 0.058), trou), "shr": ("capsule", (0.058, SHIN / 2 - 0.058), trou),
                "fel": ("box", (0.05, 0.035, 0.022), shoe), "fer": ("box", (0.05, 0.035, 0.022), shoe),
                "ful": ("capsule", (0.042, FORE / 2 - 0.042), skin), "fur": ("capsule", (0.042, FORE / 2 - 0.042), skin),
                "uul": ("capsule", (0.05, UPPER / 2 - 0.05), shirt), "uur": ("capsule", (0.05, UPPER / 2 - 0.05), shirt),
                "torso": ("ellipsoid", (0.12, 0.185, 0.27), shirt), "head": ("sphere", (0.105,), skin),
                "hair": ("ellipsoid", (0.112, 0.112, 0.085), hair)}
        for part in PARTS:
            kind, size, mat = spec[part]
            rx.world_xml.append(f'<body name="pl{i}_{part}" mocap="true" pos="0 0 -50">'
                                + geom(kind, (0, 0, 0), size, mat) + "</body>")
    return PedAnimator()


class PedAnimator:
    """Poses the pedestrian segments from the recorded root positions (heading, speed, stride phase)."""

    def bind(self, model: mujoco.MjModel, rec: Recording) -> None:
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        self.root = np.array([int(model.body_mocapid[bid(f"ped_{i}")]) for i in range(N_PED_SLOTS)])
        self.limb = np.array([[int(model.body_mocapid[bid(f"pl{i}_{p}")]) for p in PARTS] for i in range(N_PED_SLOTS)])
        pos = rec.mocap_pos[:, self.root, :2]                                  # (T, n, 2)
        alive = rec.mocap_pos[:, self.root, 2] > PARK_Z
        dt = rec.dt
        vel = np.gradient(pos, dt, axis=0)
        vel = uniform_filter1d(vel, size=max(int(0.6 / dt), 1), axis=0, mode="nearest")
        speed = np.linalg.norm(vel, axis=-1) * alive
        heading = np.arctan2(vel[..., 1], vel[..., 0])
        for k in range(N_PED_SLOTS):                                           # hold the last heading while (nearly) still
            h, last = heading[:, k].copy(), 0.0
            for t in range(len(h)):
                if speed[t, k] > 0.12:
                    last = h[t]
                h[t] = last
            heading[:, k] = h
        step = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=-1) * alive
        self.speed, self.heading = speed, heading
        self.phase = 2 * np.pi * np.cumsum(step, axis=0) / STRIDE_M            # (T, n)
        self.alive = alive

    def _at(self, arr: np.ndarray, s: float) -> np.ndarray:
        s = float(np.clip(s, 0, len(arr) - 1))
        i = min(int(s), len(arr) - 2)
        w = s - i
        return arr[i] * (1 - w) + arr[i + 1] * w

    def apply(self, data: mujoco.MjData, s: float) -> None:
        speed, phase = self._at(self.speed, s), self._at(self.phase, s)
        i0 = int(np.clip(round(s), 0, len(self.heading) - 1))
        heading = self.heading[i0]
        for k in range(N_PED_SLOTS):
            root = data.mocap_pos[self.root[k]].copy()
            if root[2] < PARK_Z:
                data.mocap_pos[self.limb[k], 2] = -50.0
                continue
            self._pose(data, k, root, float(heading[k]), float(speed[k]), float(phase[k]))

    def _pose(self, data: mujoco.MjData, k: int, root: np.ndarray, h: float, speed: float, phase: float) -> None:
        feet = root[2] - PED_CENTER_Z
        f = np.array([math.cos(h), math.sin(h), 0.0])
        left = np.array([-math.sin(h), math.cos(h), 0.0])
        up = np.array([0.0, 0.0, 1.0])
        base = np.array([root[0], root[1], feet])
        amp = float(np.clip(speed / 1.3, 0.0, 1.0))
        a_hip, bob = 0.55 * amp, 0.018 * amp * abs(math.sin(phase))
        pelvis = base + up * (HIP_H + bob)
        out: dict[str, tuple[np.ndarray, np.ndarray]] = {}

        def seg(name: str, a: np.ndarray, b: np.ndarray) -> None:
            out[name] = ((a + b) / 2, _quat_z_to(b - a))

        for side, sgn, ph in (("l", 1.0, phase), ("r", -1.0, phase + math.pi)):
            hip = pelvis + left * (sgn * HIP_Y)
            ang = a_hip * math.sin(ph)
            flex = amp * 0.85 * max(0.0, math.cos(ph)) + 0.03
            knee = hip + THIGH * (f * math.sin(ang) - up * math.cos(ang))
            sa = ang - flex
            ankle = knee + SHIN * (f * math.sin(sa) - up * math.cos(sa))
            seg(f"th{side}", hip, knee)
            seg(f"sh{side}", knee, ankle)
            out[f"fe{side}"] = (ankle + f * 0.045 - up * 0.028, _quat_yaw(h, 0.0))
            sh = pelvis + up * (SHOULDER_H - HIP_H) + left * (sgn * SHOULDER_Y)
            arm = -0.85 * a_hip * math.sin(ph) + 0.04
            elbow_pt = sh + UPPER * (f * math.sin(arm) - up * math.cos(arm)) + left * (sgn * 0.03)
            bend = 0.22 + 0.55 * amp * max(0.0, -math.sin(ph) * 0.6 + 0.4)
            ea = arm + bend
            hand = elbow_pt + FORE * (f * math.sin(ea) - up * math.cos(ea)) + left * (sgn * 0.02)
            seg(f"uu{side}", sh, elbow_pt)
            seg(f"fu{side}", elbow_pt, hand)
        lean = 0.05 + 0.10 * amp
        out["torso"] = (pelvis + up * 0.215 + f * (0.02 * amp), _quat_yaw(h, lean))
        out["head"] = (base + up * (HEAD_H + bob) + f * (0.02 * amp), np.array([1.0, 0, 0, 0]))
        out["hair"] = (base + up * (HEAD_H + bob + 0.032) - f * 0.008, _quat_yaw(h, 0.0))
        for j, part in enumerate(PARTS):
            pos, quat = out[part]
            data.mocap_pos[self.limb[k, j]] = pos
            data.mocap_quat[self.limb[k, j]] = quat


def _quat_yaw(yaw: float, pitch: float) -> np.ndarray:
    """Rotation by ``yaw`` about z, preceded by a forward lean ``pitch`` about the body's left axis."""
    qz = np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])
    qy = np.array([math.cos(pitch / 2), 0.0, math.sin(pitch / 2), 0.0])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qy
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def dress_obstacles(rx: RenderXml, mats: Materials) -> None:
    """Even slots become striped bollards, odd slots planters (the physics kinds, same size)."""
    n_slots = 6
    hide_physics_geoms(rx, {f"obs_geom_{i}": 2 for i in range(n_slots)})
    yellow = mats("bol_yellow", (0.95, 0.72, 0.06), roughness=0.5)
    black = mats("bol_black", (0.05, 0.05, 0.05), roughness=0.6)
    reflect = mats("bol_reflect", (0.92, 0.92, 0.9), roughness=0.25, emission=0.6)
    pot = mats("planter_clay", (0.62, 0.30, 0.18), roughness=0.9)
    soil = mats("planter_soil", (0.12, 0.08, 0.05), roughness=1.0)
    leaf = [mats(f"plant_{k}", c, roughness=0.85) for k, c in enumerate(((0.10, 0.36, 0.10), (0.16, 0.44, 0.12), (0.08, 0.28, 0.10)))]
    bloom = [mats(f"bloom_{k}", c, roughness=0.6) for k, c in enumerate(((0.95, 0.32, 0.40), (0.98, 0.78, 0.15), (0.95, 0.95, 0.95)))]
    rng = np.random.default_rng(3)
    for i in range(n_slots):
        r, hh = OBS_SIZES[i % 2]
        parts: list[str] = []
        if i % 2 == 0:                                                   # bollard: radius r, total height 2 * hh
            parts += [geom("cylinder", (0, 0, 0), (r, hh), yellow)]
            for z in (-0.22, -0.02, 0.18):
                parts.append(geom("cylinder", (0, 0, z), (r + 0.004, 0.045), black))
            parts += [geom("cylinder", (0, 0, hh - 0.11), (r + 0.004, 0.028), reflect), geom("sphere", (0, 0, hh), (r,), yellow)]
        else:                                                            # planter: terracotta pot and a bushy plant
            parts += [geom("cylinder", (0, 0, -hh + 0.17), (r, 0.17), pot), geom("cylinder", (0, 0, -hh + 0.35), (r + 0.03, 0.03), pot),
                      geom("cylinder", (0, 0, -hh + 0.375), (r - 0.02, 0.008), soil)]
            for _ in range(7):
                o = rng.normal(0, r * 0.45, size=2)
                rr = float(rng.uniform(0.11, 0.17))
                parts.append(geom("ellipsoid", (float(o[0]), float(o[1]), -hh + 0.50 + float(rng.uniform(0.0, 0.14))),
                                  (rr, rr, rr * 0.9), leaf[int(rng.integers(3))]))
            for _ in range(5):
                o = rng.normal(0, r * 0.5, size=2)
                parts.append(geom("sphere", (float(o[0]), float(o[1]), -hh + 0.66 + float(rng.uniform(0.0, 0.08))),
                                  (0.035,), bloom[int(rng.integers(3))]))
        rx.body_xml.setdefault(f"obs_{i}", []).extend(parts)
