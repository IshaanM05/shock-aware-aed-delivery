"""The rover-versus-drone shot: one dispatch clock, two vehicles, numbers read from ``results/``.

The rover scene is a single 36 m street segment, but the clinical comparison is about a 1 km radius, so the
film cannot show the real distances in frame. Instead each shot is one *beat* on a shared **dispatch clock**
(simulated seconds since the alert): a HUD shows the clock, the distance each vehicle still has to cover at its
true scale, and when each one arrives, while the 3D view shows what happens at the patient.

* Beat 1 follows the drone's last approach (the recorded flight of ``aedrover.drone``) down to its 2 m hover.
* Beat 2, after a time skip, replays the end of the rover's recorded segment arriving beneath the hovering drone.

Every number on screen comes from code (``MissionParams``, ``mission_time_s``, ``ScenarioParams``) or from the
clinical result files; nothing is typed in. The drone's flight is simulated in its own model and placed in the
scene by a rigid transform; the rover is the benchmark recording, hidden while it is hundreds of metres away.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import pandas as pd

from ..clinical.decision import ScenarioParams
from ..drone.mission import MissionParams, mission_time_s
from . import hud
from .camera import CameraPose
from .dressing import dress_scene
from .drone_visuals import DroneSpec
from .overlays import OverlayConfig
from .post import Grade
from .render_model import RenderScene
from .shots import ease, smooth_poses, time_map

PARK_Z = -50.0


# -------------------------------------------------------------------------------------- numbers
@dataclass(frozen=True)
class DispatchNumbers:
    """Everything the shot states, gathered once from code and result files."""

    radius_m: float
    drone_launch_s: float                   # alert to liftoff (an assumption, ``MissionParams.launch_latency_s``)
    drone_flight_s: float                   # liftoff to release, ``mission_time_s`` (still air)
    clinical_drone_s: float                 # the clinical model's drone: launch plus straight flight, no climb or descent
    rover_route_m: float                    # ``clinical_routes.csv``
    rover_arrival_s: float                  # median travel time of the routes the rover delivers safely
    rover_p_safe: float                     # fraction of routes delivered safely
    survival: dict[str, tuple[float, float, float]]    # mode -> (mean, 95% low, 95% high), parallel dispatch with the ambulance
    controller: str
    density: str
    model: str

    @property
    def drone_arrival_s(self) -> float:
        return self.drone_launch_s + self.drone_flight_s

    @property
    def timing_gap_s(self) -> float:
        """How much faster the clinical model's drone is than the simulated one (positive = clinical is faster)."""
        return self.drone_arrival_s - self.clinical_drone_s


def load_numbers(results_dir: str | Path, *, controller: str = "ppo", radius_m: float = 1000.0,
                 density: str = "assumed_4_per_km", model: str = "larsen1993", mp: MissionParams | None = None) -> DispatchNumbers:
    results_dir, mp = Path(results_dir), mp or MissionParams()
    routes = pd.read_csv(results_dir / "clinical_routes.csv")
    r = routes[(routes.controller == controller) & (routes.radius_m == radius_m) & (routes.density == density)]
    surv = pd.read_csv(results_dir / "clinical_survival_vs_radius.csv")
    s = surv[(surv.controller == controller) & (surv.radius_m == radius_m) & (surv.density == density) & (surv.model == model)]
    if len(r) != 1 or set(s["mode"]) != {"ambulance", "rover", "drone"} or len(s) != 3:
        raise ValueError(f"results do not hold one route row and three survival rows for {controller}, {radius_m:g} m, {density}, {model}")
    flight = mission_time_s(radius_m, 0.0, None, mp)
    if not math.isfinite(flight):
        raise ValueError(f"the drone cannot fly {radius_m:g} m in still air with these parameters")
    sp = ScenarioParams(radius_m=radius_m)
    row = r.iloc[0]
    return DispatchNumbers(
        radius_m=float(radius_m), drone_launch_s=float(mp.launch_latency_s), drone_flight_s=float(flight),
        clinical_drone_s=60.0 * sp.drone_launch_min + radius_m / sp.drone_speed_mps,
        rover_route_m=float(row.route_m), rover_arrival_s=60.0 * float(row.time_min_median), rover_p_safe=float(row.p_safe_delivery),
        survival={m.mode: (float(m.mean_survival), float(m.survival_ci_low), float(m.survival_ci_high)) for m in s.itertuples()},
        controller=controller, density=density, model=model)


# ---------------------------------------------------------------------------------- scene helpers
class RoverHider:
    """Parks the rover's chassis below the floor (all its dressed parts follow) while it is far from the scene."""

    def bind(self, model: mujoco.MjModel, rec) -> None:
        j = next(i for i in range(model.njnt) if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE)
        self._z = int(model.jnt_qposadr[j]) + 2

    def apply(self, data: mujoco.MjData, s: float) -> None:
        data.qpos[self._z] = PARK_Z


@dataclass
class DispatchContext:
    """Per-frame arrays handed to a camera rig."""

    u: np.ndarray                      # (F,) progress through the shot, 0..1
    clock: np.ndarray                  # (F,) dispatch clock [s]
    drone: np.ndarray                  # (F, 3) drone position in the scene
    heading: float                     # flight direction in the scene [rad]
    rover: np.ndarray | None           # (F, 3) rover position, None while it is not in the scene
    rover_yaw: np.ndarray | None


DispatchRig = Callable[[DispatchContext], list[CameraPose]]


def rig_aerial_follow(offset0=(7.5, -3.0, 7.0), offset1=(6.0, -1.8, 1.4), fov0: float = 42.0, fov1: float = 46.0) -> DispatchRig:
    """A camera pacing the drone from a little ahead and to its right (forward, left, up offsets in the flight frame).

    The lateral distance stays inside the street (the building fronts stand at ``|y| = 5``), so the camera never ends
    up in a wall when the drone comes down. The sun is ahead of the drone, so this side shows its lit face.
    """
    def rig(ctx: DispatchContext) -> list[CameraPose]:
        c, s = math.cos(ctx.heading), math.sin(ctx.heading)
        out = []
        for p, u in zip(ctx.drone, ctx.u, strict=True):
            e = float(ease(u))
            f, le, z = (np.array(offset0) * (1 - e) + np.array(offset1) * e)
            cam = p + np.array([c * f - s * le, s * f + c * le, z])
            out.append(CameraPose(tuple(cam), tuple(p), fov0 + (fov1 - fov0) * e))
        return out
    return rig


def rig_arrival(back: float = 4.2, height: float = 1.3, swing_deg: float = 36.0, bias: float = 0.55, fovy: float = 56.0) -> DispatchRig:
    """Behind the rover, looking at a point between it and the drone hovering ahead."""
    def rig(ctx: DispatchContext) -> list[CameraPose]:
        assert ctx.rover is not None and ctx.rover_yaw is not None
        out = []
        for r, yaw, d in zip(ctx.rover, ctx.rover_yaw, ctx.drone, strict=True):
            a = yaw + math.radians(swing_deg)
            cam = r + np.array([-back * math.cos(a), -back * math.sin(a), height])
            tgt = r + bias * (d - r) + np.array([0.0, 0.0, 0.2])
            out.append(CameraPose(tuple(cam), tuple(tgt), fovy))
        return out
    return rig


def fit_rate(frames: int, fps: int, clock_start: float, clock_end: float,
             slow: tuple[float, float, float] | None = None) -> float:
    """The playback rate (simulated seconds per film second) that takes the clock from ``clock_start`` to ``clock_end``
    in ``frames`` frames, given an optional slow-down; found by bisection on the same time map the shot uses."""
    def end(rate: float) -> float:
        return float(time_map(frames, fps, clock_start, 1e9, 1.0, base_rate=rate, slow=slow)[-1])

    lo, hi = 1e-3, 1e3
    for _ in range(80):
        mid = math.sqrt(lo * hi)
        lo, hi = (mid, hi) if end(mid) < clock_end else (lo, mid)
    return math.sqrt(lo * hi)


# ------------------------------------------------------------------------------------------ shot
@dataclass
class DispatchShot:
    """One beat of the dispatch story, rendered from the rover recording ``rover`` and a drone flight.

    ``clock_start`` is the dispatch clock at the first frame and ``rate`` the simulated seconds per film second;
    ``slow = (clock, width_s, rate)`` dips the rate around one moment (used to settle into the hover).
    With ``rover_visible`` the rover plays the end of its recorded segment, arriving exactly at the clock value
    ``numbers.rover_arrival_s``; otherwise it is hidden.
    """

    name: str
    rover: str                                   # key into the film's recordings
    drone: DroneSpec
    numbers: DispatchNumbers
    rig: DispatchRig
    frames: int
    clock_start: float
    rate: float = 1.0
    slow: tuple[float, float, float] | None = None
    rover_visible: bool = False
    overlays: OverlayConfig | None = None
    title: str = ""
    caption: str = ""
    footnote: str = ""
    show_result: bool = False
    grade: Grade | None = None
    dof: float = 0.3
    smooth_s: float = 0.35
    fade_in: float = 0.25
    fade_out: float = 0.25
    corridor_margin_m: float = 30.0

    # -- timeline (pure numpy, tested without a GPU)
    def clock(self, fps: int) -> np.ndarray:
        """Dispatch clock [s] at each frame."""
        return time_map(self.frames, fps, self.clock_start, 1e9, 1.0, base_rate=self.rate, slow=self.slow)

    def rover_steps(self, clock: np.ndarray, n_steps: int, dt: float, fps: int) -> np.ndarray:
        """Rover recording step at each frame: aligned to the clock when visible, otherwise walking time for the pedestrians."""
        if self.rover_visible:
            return np.clip((n_steps - 1) - (self.numbers.rover_arrival_s - clock) / dt, 0.0, n_steps - 1.0)
        return np.minimum(np.arange(self.frames) / fps / dt, n_steps - 1.0)

    def hud_state(self, k: int, fps: int, clock_k: float, drone_to_go_m: float, drone_released: bool, rover_to_go_m: float) -> hud.DispatchHud:
        n = self.numbers
        flown = float(np.clip(1.0 - drone_to_go_m / n.radius_m, 0.0, 1.0))
        rover_frac = float(np.clip(1.0 - rover_to_go_m / n.rover_route_m, 0.0, 1.0))
        rows: tuple = ()
        alpha = 0.0
        if self.show_result:
            alpha = float(np.clip((k - (self.frames - 3.5 * fps)) / (1.0 * fps), 0.0, 1.0))
            sv = n.survival
            rows = (("ambulance alone", f"{100 * sv['ambulance'][0]:.1f}%", hud.INK),
                    ("ambulance + rover", f"{100 * sv['rover'][0]:.1f}%", hud.GREEN),
                    ("ambulance + drone", f"{100 * sv['drone'][0]:.1f}%", hud.AMBER))
        return hud.DispatchHud(
            clock_s=clock_k, drone_frac=flown, rover_frac=rover_frac,
            drone_text="arrived" if drone_released else f"{drone_to_go_m:,.0f} m to go",
            rover_text="arrived" if rover_to_go_m <= 0.5 else f"{rover_to_go_m:,.0f} m to go",
            drone_arrival_s=n.drone_arrival_s, rover_arrival_s=n.rover_arrival_s,
            title=self.title, caption=self.caption, footnote=self.footnote,
            result_title=f"survival at {n.radius_m / 1000:g} km ({n.model.replace('1993', ' 1993').title()})", result_rows=rows,
            result_note=(f"the clinical model's drone is {n.timing_gap_s / 60:.1f} min faster than the flight above"
                         if rows else ""),
            result_alpha=alpha)

    # -- rendering
    def frames_from(self, film):
        """Yield the finished frames of this beat (called by ``Film.render``)."""
        rec = film.recs[self.rover]
        n, fps = self.numbers, film.fps
        start = self.drone.placement.point(self.drone.rec.pos[0])
        goal = self.drone.placement.point(np.array([n.radius_m, 0.0, 0.0]))
        corridor = ((float(start[0]), float(start[1])), (float(goal[0]), float(goal[1])), self.corridor_margin_m)
        hider = [] if self.rover_visible else [RoverHider()]
        scene = RenderScene(rec, look=film.look, size=film.size, backend=film.backend, depth=True,
                            builder=lambda rx, r, lk: dress_scene(rx, r, lk, overlays=self.overlays, drone=self.drone,
                                                                  clear_flight_corridor=corridor) + hider)
        anim = next(a for a in scene.animators if type(a).__name__ == "DroneAnimator")
        try:
            scene.backend.depth_decoder()
            clock = self.clock(fps)
            t_drone = clock - n.drone_launch_s
            s = self.rover_steps(clock, len(rec), rec.dt, fps)
            drone_pos = np.array([anim.pose(float(t))[0] for t in t_drone])
            rover = rover_yaw = None
            if self.rover_visible:
                rp = [scene.rover_pose(float(si)) for si in s]
                rover, rover_yaw = np.array([p for p, _ in rp]), np.unwrap(np.array([y for _, y in rp]))
            ctx = DispatchContext(np.linspace(0.0, 1.0, self.frames), clock, drone_pos, self.drone.placement.yaw, rover, rover_yaw)
            poses = smooth_poses(self.rig(ctx), self.smooth_s * fps)
            x_goal = float(rec.meta["scenario"]["x_goal"])
            release = np.array([n.radius_m, 0.0, float(self.drone.rec.meta.get("mission", {}).get("release_height_m", 0.0))])
            for k in range(self.frames):
                anim.t, anim.active = float(t_drone[k]), True
                pose = poses[k]
                focus = float(np.linalg.norm(np.asarray(pose.pos) - np.asarray(pose.target)))
                frame = scene.render_finished(float(s[k]), pose, grade=self.grade, seed=k, dof=self.dof, focus=focus)
                to_go = max(x_goal - float(rover[k, 0]), 0.0) if rover is not None else n.rover_route_m * max(1.0 - clock[k] / n.rover_arrival_s, 0.0)
                drone_to_go = float(np.linalg.norm(self.drone.rec.pose_at(float(t_drone[k]))[0] - release))
                state = self.hud_state(k, fps, float(clock[k]), drone_to_go, bool(t_drone[k] >= anim.t_release), to_go)
                frame = hud.draw_dispatch_hud(frame, state)
                f = film._fade(k, self.frames, fps, self.fade_in, self.fade_out)
                if f < 1.0:
                    frame = (frame.astype(np.float32) * f).astype(np.uint8)
                yield frame
        finally:
            scene.close()
