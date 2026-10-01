"""Truly live rendering: the simulation steps and the renderer draws that same state, with no recording in between.

``RenderScene`` replays a whole ``Recording``. For a live run there is no future to look at, so this module supplies the
pieces that only ever look at the present and the past:

* ``header_recording``: a one-frame ``Recording`` of the environment, enough to build and dress the render model;
* ``StreamingPedAnimator``: pedestrian gait with the heading from a causal low-pass of velocity and the stride phase
  accumulated as the pedestrians walk (``PedAnimator`` computes both from the whole episode);
* ``LiveOverlayAnimator``: halo, lidar bubble and the planner's rollouts from the live state (the recorded "actual path
  ahead" ribbon needs the future, so it is not drawn live);
* ``LiveScene``: one scene bound to an environment; ``update`` copies the physics state across, ``render`` draws it.

Nothing here steps or changes the simulation: it only reads ``env.world.data``. A rich-world environment rebuilds its
world on ``reset``, so a ``LiveScene`` is made after ``reset`` and used for that one episode.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from .camera import CameraPose
from .dressing import dress_scene
from .look import Look
from .overlays import PARK, OverlayAnimator, OverlayConfig
from .people import N_PED_SLOTS, PARK_Z, STRIDE_M, PedAnimator
from .recording import Recording
from .render_model import RenderScene

VELOCITY_TAU_S = 0.3              # time constant of the causal velocity low-pass that gives the pedestrians' heading
WALKING_SPEED = 0.12              # m/s below which a pedestrian keeps its last heading
TELEPORT_M = 1.5                  # a jump this large in one update is a respawn, not a step


@dataclass
class LiveState:
    t: float                              # simulated time [s]
    dt: float                             # simulated seconds since the previous update
    lidar: np.ndarray                     # (n,) lidar ranges [m]
    shock_g: float                        # recent peak payload shock [g]
    rollouts: dict | None = None          # planner snapshot: ``xy`` (K, H+1, 2) and ``best`` (H+1, 2)


def header_recording(env, vehicle: str = "optimized", *, rollout_nodes: int | None = None) -> Recording:
    """A one-frame recording of ``env`` right after ``reset``: the physics MJCF, the scenario and the initial state.

    ``rollout_nodes`` (planner horizon plus one) reserves overlay beads for a planner's sampled rollouts.
    """
    d, sc = env.world.data, env.scenario
    meta = {"controller": "live", "family": sc.family, "seed": sc.seed, "vehicle": vehicle, "speed_cap": None, "outcome": "",
            "time_s": 0.0, "peak_shock_g": 0.0, "scenario": sc.to_dict(), "lidar_range": float(env.perc.s.lidar_range),
            "budget_g": env.budget_g}
    if env.furniture:
        meta["furniture"] = [it.to_dict() for it in env.furniture]
    rollouts = None
    if rollout_nodes:
        rollouts = {0: {"xy": np.zeros((24, rollout_nodes, 2), np.float32), "cost": np.zeros(24, np.float32),
                        "best": np.zeros((rollout_nodes, 2), np.float32)}}
    return Recording(xml=env.world.xml, dt=env.dt, t=np.zeros(1), qpos=d.qpos[None].copy(), mocap_pos=d.mocap_pos[None].copy(),
                     mocap_quat=d.mocap_quat[None].copy(), shock_g=np.zeros(1), cmd=np.zeros((1, 2)),
                     lidar=np.asarray(env.obs["lidar"], dtype=np.float32)[None],
                     lidar_angles=np.asarray(env.obs["lidar_angles"], dtype=np.float32), meta=meta, rollouts=rollouts)


class StreamingPedAnimator(PedAnimator):
    """Pedestrian gait from the present and the past only (``PedAnimator`` looks at the whole episode)."""

    def bind(self, model: mujoco.MjModel, rec=None) -> None:
        bid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)  # noqa: E731
        from .people import PARTS
        self.root = np.array([int(model.body_mocapid[bid(f"ped_{i}")]) for i in range(N_PED_SLOTS)])
        self.limb = np.array([[int(model.body_mocapid[bid(f"pl{i}_{p}")]) for p in PARTS] for i in range(N_PED_SLOTS)])
        self._prev: list[np.ndarray | None] = [None] * N_PED_SLOTS
        self._vel = np.zeros((N_PED_SLOTS, 2))
        self._heading = np.zeros(N_PED_SLOTS)
        self._phase = np.zeros(N_PED_SLOTS)

    def apply(self, data: mujoco.MjData, s: float) -> None:
        """The recorded-path entry point (``RenderScene`` calls it once while building): nothing to draw before the first
        live update, so the figures stay parked."""

    def apply_live(self, data: mujoco.MjData, ls: LiveState) -> None:
        dt = max(ls.dt, 1e-3)
        alpha = 1.0 - math.exp(-dt / VELOCITY_TAU_S)
        for k in range(N_PED_SLOTS):
            root = data.mocap_pos[self.root[k]].copy()
            if root[2] < PARK_Z:
                data.mocap_pos[self.limb[k], 2] = -50.0
                self._prev[k], self._vel[k] = None, 0.0
                continue
            xy, prev = root[:2].copy(), self._prev[k]
            if prev is None or float(np.linalg.norm(xy - prev)) > TELEPORT_M:        # first sight or a respawn
                self._vel[k], step = 0.0, 0.0
            else:
                step = float(np.linalg.norm(xy - prev))
                self._vel[k] += ((xy - prev) / dt - self._vel[k]) * alpha
            self._prev[k] = xy
            speed = float(np.linalg.norm(self._vel[k]))
            if speed > WALKING_SPEED:
                self._heading[k] = math.atan2(self._vel[k][1], self._vel[k][0])
            self._phase[k] += 2 * math.pi * step / STRIDE_M
            self._pose(data, k, root, float(self._heading[k]), speed, float(self._phase[k]))


class LiveOverlayAnimator(OverlayAnimator):
    """Halo, lidar bubble and planner rollouts from the live state."""

    def bind(self, model: mujoco.MjModel, rec: Recording) -> None:
        self.rec = rec
        sc = rec.meta["scenario"]
        self._kerb = (float(sc["kerb_h"]), float(sc["x_down"]), float(sc["x_up"]))
        self.ids = {name: np.array([int(model.body_mocapid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"ov_{name}_{i}")])
                                    for i in range(n)]) for name, n in self.pools.items()}
        self._lidar_range = float(rec.meta.get("lidar_range", 8.0))
        self._budget = float(rec.meta.get("budget_g", 3.0))

    def apply(self, data: mujoco.MjData, s: float) -> None:
        """The recorded-path entry point: nothing to draw before the first live update."""

    def apply_live(self, data: mujoco.MjData, ls: LiveState) -> None:
        self._d = data
        pos = data.qpos[:3].copy()
        w, x, y, z = data.qpos[3:7]
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        if "lhit" in self.ids:
            self._lidar_ranges(pos, yaw, np.asarray(ls.lidar))
        if "roll" in self.ids:
            if ls.rollouts is None:
                self._park("roll")
                self._park("best")
            else:
                self._put_rollouts(ls.rollouts)
        if "h0" in self.ids:
            self._halo_at(pos, yaw, ls.shock_g / self._budget, ls.t / max(self.rec.dt, 1e-6))


def live_dressing(rx, rec: Recording, look: Look, cfg: OverlayConfig) -> list:
    """The usual dressing (street, rover, people, overlays) with the animators swapped for their streaming versions."""
    out = []
    for a in dress_scene(rx, rec, look, overlays=cfg):
        if type(a) is PedAnimator:
            a = StreamingPedAnimator()
        elif type(a) is OverlayAnimator:
            a = LiveOverlayAnimator(a.cfg, a.pools)
        out.append(a)
    return out


class LiveScene:
    """One render scene bound to a live environment (build it after ``env.reset``, use it for that episode)."""

    def __init__(self, env, *, vehicle: str = "optimized", size: tuple[int, int] = (1280, 720), look: Look | None = None,
                 backend: str = "filament", overlays: OverlayConfig | None = None, rollout_nodes: int | None = None,
                 depth: bool = False) -> None:
        self.env = env
        self.cfg = overlays or OverlayConfig(ribbon=False, rollouts=bool(rollout_nodes))
        if self.cfg.ribbon:
            raise ValueError("the recorded-path ribbon needs the future; turn it off for a live scene")
        header = header_recording(env, vehicle, rollout_nodes=rollout_nodes)
        self.scene = RenderScene(header, look=look, size=size, backend=backend, depth=depth,
                                 builder=lambda rx, r, lk: live_dressing(rx, r, lk, self.cfg))
        self._t_prev = 0.0

    def update(self, ls: LiveState) -> None:
        """Copy the simulation's current state into the render model and pose everything that is drawn from it."""
        d = self.env.world.data
        self.scene.set_live(d.qpos, d.mocap_pos, d.mocap_quat, ls)
        self._t_prev = ls.t

    def rover_pose(self) -> tuple[np.ndarray, float]:
        q = self.scene.data.qpos
        w, x, y, z = q[3:7]
        return q[:3].copy(), float(math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def render(self, pose: CameraPose) -> np.ndarray:
        return self.scene.backend.render(self.scene.data, pose)

    def close(self) -> None:
        self.scene.close()


__all__ = ["LiveScene", "LiveState", "LiveOverlayAnimator", "StreamingPedAnimator", "header_recording", "live_dressing", "PARK"]
