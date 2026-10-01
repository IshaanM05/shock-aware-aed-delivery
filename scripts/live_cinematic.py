"""Watch the simulation live in the cinematic renderer, in a window, until you close it.

    python scripts/live_cinematic.py                                   # PPO through a crowd
    python scripts/live_cinematic.py --drone                           # the rover AND the AED drone, both simulated live
    python scripts/live_cinematic.py --rich                            # a street with collidable parked vehicles, stalls, lamps, trees
    python scripts/live_cinematic.py --controller dwa --family mixed --seed 5010
    python scripts/live_cinematic.py --replay                          # simulate each episode first, then play it back (old behaviour)
    python scripts/live_cinematic.py --drone --record dispatch.mp4     # no window: the same live run, fixed time step, written to a video

The simulation and the renderer run together: each frame the simulation is stepped until its clock catches up with the
wall clock, and the very state it has reached is drawn through the dressed PBR scene with a chase camera, halo, lidar
bubble (and MPPI's sampled rollouts) and a HUD. There is no pre-simulation pass. Planners that cannot keep up in real time
(MPPI is the usual one) simply run slower than real time; the HUD says by how much. ``--rich`` compiles each scenario's street
furniture into the physics (``aedrover.sim.furniture``), so the vehicles, stalls and lamp posts you see are what the rover
can hit; it is a separate mode, never the standard benchmark.

``--drone`` adds the AED quadrotor, simulated live in its own MuJoCo model with its cascaded flight controller
(``aedrover.drone.mission.MissionStepper``), dispatched at the same moment as the rover and heading for the same patient. The 36 m
street cannot hold the 1 km clinical comparison, so the drone leaves from the rover's start and flies to the same goal (``--drone-distance``,
``--drone-altitude``): the physics and the controller are the same, the distances are not, and the HUD says so. Esc or closing
the window quits.
"""

from __future__ import annotations

import argparse
import sys
import time
import tkinter as tk
from pathlib import Path

import numpy as np
from PIL import Image, ImageTk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aedrover.analysis.experiments import controller_spec  # noqa: E402
from aedrover.control.safety_filter import SafetyFilter  # noqa: E402
from aedrover.drone.mission import MissionParams, MissionStepper, mission_time_s  # noqa: E402
from aedrover.nav.base import make_controller  # noqa: E402
from aedrover.sim.env import AEDRoverEnv  # noqa: E402
from aedrover.sim.vehicle_mjcf import VehicleParams  # noqa: E402
from aedrover.viz import hud, post  # noqa: E402
from aedrover.viz.camera import chase, watch_both  # noqa: E402
from aedrover.viz.dressing import dress_scene  # noqa: E402
from aedrover.viz.drone_visuals import Placement  # noqa: E402
from aedrover.viz.live import DroneLive, LiveScene, LiveState  # noqa: E402
from aedrover.viz.overlays import OverlayConfig  # noqa: E402
from aedrover.viz.recording import REPO, _tuned_mppi, record_episode  # noqa: E402
from aedrover.viz.render_model import RenderScene  # noqa: E402

MAX_STEPS_PER_FRAME = 12          # keeps the window responsive when a planner is slower than real time
HOLD_S = 2.5                      # how long the last frame stays after an episode ends
ROLLOUT_SHOW_STEPS = 12           # a planner snapshot is drawn for this many control steps
DRONE_LANE_Y = -1.0               # the drone flies 1 m to the rover's right, so the two never overlap


def build_controller(name: str):
    extra: dict = {}
    if name == "ppo":
        extra["path"] = str(REPO / "models" / "ppo_selected")
    elif name == "mppi":
        extra = {**_tuned_mppi(), "nthread": 8}
    cname, kw = controller_spec(name, 2.0, **extra)
    return make_controller(cname, **kw)


class Window:
    def __init__(self, w: int, h: int, title: str) -> None:
        self.root = tk.Tk()
        self.root.title(title)
        self.label = tk.Label(self.root, bg="black")
        self.label.pack()
        self.ok = True
        self.root.bind("<Escape>", lambda e: self.quit())
        self.root.protocol("WM_DELETE_WINDOW", self.quit)
        self.size = (w, h)

    def quit(self) -> None:
        self.ok = False

    def show(self, frame: np.ndarray) -> None:
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.label.configure(image=img)
        self.label.image = img
        self.root.update()

    def splash(self, title: str, line: str) -> None:
        self.show(hud.end_card(self.size, title, [line]))


class Session:
    """One live episode: the rover environment, its controller and (with ``--drone``) the live drone, drawn by one scene.

    ``advance(t)`` brings the simulation to the demo clock ``t``; ``frame()`` draws what it has reached. The rover episode
    ends at the goal, the drone keeps flying until it has released and hovers; ``finished`` is true when both are done.
    """

    def __init__(self, a, env: AEDRoverEnv, seed: int, size: tuple[int, int]) -> None:
        self.a, self.env, self.seed = a, env, seed
        env.reset(seed=seed, options={"family": a.family})
        self.ctrl, self.shield = build_controller(a.controller), SafetyFilter()
        self.ctrl.reset(env)
        self.shield.reset()
        self.is_mppi = a.controller == "mppi"
        if self.is_mppi:
            self.ctrl.capture = True
        self.x_goal = float(env.scenario.x_goal)
        self.stepper = placement = None
        if a.drone:
            mp = MissionParams(cruise_alt_m=a.drone_altitude)
            self.drone_distance = a.drone_distance or self.x_goal              # by default: from the rover's own start to the goal
            self.stepper = MissionStepper(self.drone_distance, mp=mp)
            self.drone_plan_s = mission_time_s(self.drone_distance, 0.0, None, mp)
            placement = Placement.arriving_at((self.x_goal, DRONE_LANE_Y), self.drone_distance)
        self.scene = LiveScene(env, size=size, rollout_nodes=self.ctrl.H + 1 if self.is_mppi else None, drone_placement=placement)
        self.placement = placement
        self.shock_hist: list[float] = []
        self.peak, self.cam, self.done, self.info = 0.0, None, False, {}
        self.snapshot, self.snap_step = None, -10**9
        self.t, self.t_prev, self.t_drone_prev, self.t_rover_done, self.t_all_done = 0.0, 0.0, 0.0, None, None
        self.caption = f"{a.family}, seed {seed}" + (", rich world" if a.rich else "")

    @property
    def finished(self) -> bool:
        drone_done = self.stepper is None or self.stepper.released or bool(self.stepper.reason)
        return self.done and drone_done

    def advance(self, t_target: float, max_steps: int | None = None) -> int:
        """Step the rover to the demo clock ``t_target`` (at most ``max_steps`` control steps) and the drone with it."""
        env, steps = self.env, 0
        while not self.done and env._t < t_target and (max_steps is None or steps < max_steps):
            obs = env.obs
            v, d = self.shield(obs, *self.ctrl.act(obs))
            if self.is_mppi and self.ctrl.last_rollouts is not None:
                self.snapshot, self.snap_step = self.ctrl.last_rollouts, env._steps
                self.ctrl.last_rollouts = None
            _, _, term, trunc, self.info = env.step(np.array([v, d]))
            self.shock_hist.append(float(self.info["shock_g"]))
            self.done, steps = term or trunc, steps + 1
        self.t = min(t_target, env._t) if not self.done else t_target
        if self.done and self.t_rover_done is None:
            self.t_rover_done = env._t
        if self.stepper is not None:
            self.stepper.advance_to(self.t)
        if self.finished and self.t_all_done is None:
            self.t_all_done = self.t
        return steps

    def frame(self, speed_note: str = "") -> np.ndarray:
        env, a = self.env, self.a
        self.peak = max(self.peak, self.shock_hist[-1] if self.shock_hist else 0.0)
        drone = None
        if self.stepper is not None:
            st = self.stepper
            s = st.state()
            drone = DroneLive(s.pos, s.quat, st.rotor_thrusts(), max(st.t - self.t_drone_prev, 1e-3), st.released)
            self.t_drone_prev = st.t
        ls = LiveState(t=env._t, dt=max(env._t - self.t_prev, 1e-3), lidar=np.asarray(env.obs["lidar"]),
                       shock_g=max(self.shock_hist[-20:], default=0.0),
                       rollouts=self.snapshot if env._steps - self.snap_step <= ROLLOUT_SHOW_STEPS else None, drone=drone)
        self.t_prev = env._t
        self.scene.update(ls)
        p, yaw = self.scene.rover_pose()
        if self.stepper is not None:
            want = watch_both(p, self.placement.point(drone.pos), (self.x_goal, 0.0), self.placement.yaw,
                              float(env.obs["x"]) / self.x_goal)
        else:
            want = chase(p, yaw, back=4.2, height=1.6, swing_deg=22, fovy=52)
        self.cam = want if self.cam is None else self.cam.lerp(want, 0.18)
        img = post.grade_colour(self.scene.render(self.cam), post.Grade())
        if self.stepper is None:
            return hud.draw_hud(img, hud.HudState(t=env._t, speed=float(env.obs["vx"]), shock=ls.shock_g, peak=self.peak,
                                                  controller=a.controller, caption=self.caption + speed_note,
                                                  history=np.array(self.shock_hist[-200:])))
        st = self.stepper
        to_go = float(np.linalg.norm(st.state().pos - st.goal))
        x = float(env.obs["x"])
        return hud.draw_dispatch_hud(img, hud.DispatchHud(
            clock_s=self.t, drone_frac=float(np.clip(1.0 - to_go / self.drone_distance, 0.0, 1.0)),
            rover_frac=float(np.clip(x / self.x_goal, 0.0, 1.0)),
            drone_text="arrived" if st.released else (st.reason or f"{to_go:,.0f} m to go"),
            rover_text="arrived" if self.done and self.info.get("episode", {}).get("outcome") == "goal"
            else (self.info["episode"]["outcome"] if self.done else f"{max(self.x_goal - x, 0.0):,.0f} m to go"),
            drone_arrival_s=st.t_release if st.released else self.drone_plan_s,
            rover_arrival_s=self.t_rover_done if self.done else float("nan"),
            title=f"Rover + drone, live ({a.controller})", caption=f"{self.caption}{speed_note}",
            footnote=(f"both simulated live; the drone (its own quadrotor model and flight controller) leaves beside the rover and flies "
                      f"{self.drone_distance:g} m at {a.drone_altitude:g} m altitude, scaled to the street, not the 1 km clinical case"),
            shock=ls.shock_g, peak=self.peak, budget=env.budget_g))

    def summary(self) -> str:
        env, a = self.env, self.a
        if not self.info:
            return ""
        ep = self.info["episode"]
        s = (f"{a.controller} {a.family} seed {self.seed}: {ep['outcome']}"
             + (f" ({ep['collision_kind']})" if ep["collision_kind"] else "")
             + f" in {ep['time_s']:.1f} s, peak payload shock {ep['peak_shock_g']:.2f} g")
        if self.stepper is not None:
            s += (f"; drone released at {self.stepper.t_release:.1f} s" if self.stepper.released
                  else f"; drone flight ended: {self.stepper.reason or 'not released'}")
        return s + (f", {len(env.furniture)} pieces of furniture" if a.rich else "")

    def close(self) -> None:
        self.scene.close()


def run_live(a, win: Window) -> None:
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name("optimized"), rich=a.rich)
    seed = a.seed
    while win.ok:
        win.splash(f"{a.controller}  |  {a.family}  |  seed {seed}", "building the scene ...")
        ses = Session(a, env, seed, win.size)
        t0 = time.perf_counter()
        lag_t0, lag_s0, speed_factor = t0, 0.0, 1.0
        while win.ok:
            wall = (time.perf_counter() - t0) * a.speed
            steps = ses.advance(wall, MAX_STEPS_PER_FRAME)
            if steps == MAX_STEPS_PER_FRAME and env._t < wall - 0.25:          # the planner is the bottleneck: do not run away
                t0 += (wall - env._t - 0.25) / a.speed
            now = time.perf_counter()
            if now - lag_t0 > 1.0:
                speed_factor, lag_t0, lag_s0 = (env._t - lag_s0) / (now - lag_t0), now, env._t
            note = f"  |  x{speed_factor:.1f} real time" if speed_factor < 0.9 and not ses.done else ""
            win.show(ses.frame(note))
            if ses.t_all_done is not None and ses.t - ses.t_all_done > HOLD_S:
                break
        if win.ok:
            print(ses.summary(), flush=True)
        ses.close()
        seed += 1
    env.close()


def run_record(a) -> None:
    """The same live run with no window: a fixed time step, so the video is smooth and reproducible whatever the machine."""
    import imageio.v2 as imageio

    w, h = (int(x) for x in a.size.split("x"))
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name("optimized"), rich=a.rich)
    ses = Session(a, env, a.seed, (w, h))
    out = Path(a.record)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(out), fps=a.fps, codec="libx264", quality=None, macro_block_size=1,
                                ffmpeg_params=["-crf", "22", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    k, t0 = 0, time.perf_counter()
    try:
        while True:
            ses.advance(k / a.fps)
            writer.append_data(ses.frame())
            k += 1
            if ses.t_all_done is not None and ses.t - ses.t_all_done > HOLD_S or k > 90 * a.fps:
                break
    finally:
        writer.close()
    print(ses.summary(), flush=True)
    print(f"wrote {out} ({k} frames, {k / a.fps:.1f} s of simulated time, {time.perf_counter() - t0:.0f} s to run)", flush=True)
    ses.close()
    env.close()


def run_replay(a, win: Window) -> None:
    w, h = win.size
    seed = a.seed
    while win.ok:
        win.splash(f"{a.controller}  |  {a.family}  |  seed {seed}", "simulating, then building the scene ...")
        kw = {"controller_kwargs": {"nthread": 8}} if a.controller == "mppi" else {}
        rec = record_episode(a.controller, a.family, seed, rich=a.rich, **kw)
        scene = RenderScene(rec, size=(w, h), builder=lambda rx, r, lk: dress_scene(rx, r, lk, overlays=OverlayConfig(rollouts=False)))
        speed = np.r_[0.0, np.linalg.norm(np.diff(rec.qpos[:, :2], axis=0), axis=1) / rec.dt]
        peak = np.maximum.accumulate(rec.shock_g)
        cam, t0 = None, time.perf_counter()
        while win.ok:
            s = (time.perf_counter() - t0) / rec.dt
            if s >= len(rec) - 1:
                break
            p, yaw = scene.rover_pose(s)
            want = chase(p, yaw, back=4.2, height=1.6, swing_deg=22, fovy=52)
            cam = want if cam is None else cam.lerp(want, 0.18)
            frame = post.grade_colour(scene.render(s, cam), post.Grade())
            i = int(s)
            st = hud.HudState(t=float(rec.t[i]), speed=float(speed[i]), shock=float(rec.shock_g[i]), peak=float(peak[i]),
                              controller=a.controller, caption=f"{a.family}, seed {seed}" + (", rich world" if a.rich else ""),
                              history=rec.shock_g[max(i - 200, 0):i + 1])
            win.show(hud.draw_hud(frame, st))
        if win.ok:
            print(f"{a.controller} {a.family} seed {seed}: {rec.meta['outcome']} in {rec.meta['time_s']:.1f} s, "
                  f"peak payload shock {rec.meta['peak_shock_g']:.2f} g", flush=True)
        scene.close()
        seed += 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="ppo", choices=("ppo", "mppi", "dwa", "apf", "pure_pursuit"))
    ap.add_argument("--family", default="crowded")
    ap.add_argument("--seed", type=int, default=5021)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--rich", action="store_true", help="collidable street furniture (a separate mode, not the benchmark)")
    ap.add_argument("--drone", action="store_true", help="also fly the AED drone, simulated live, to the same patient")
    ap.add_argument("--drone-distance", type=float, default=None,
                    help="length of the drone's mission [m] (default: the street's length, from the rover's start to the goal)")
    ap.add_argument("--drone-altitude", type=float, default=6.0, help="the drone's cruise altitude [m] (scaled to the street)")
    ap.add_argument("--replay", action="store_true", help="simulate each episode first and play it back, instead of live")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed relative to the wall clock (live mode)")
    ap.add_argument("--record", default=None, help="write one live episode to this video instead of opening a window")
    ap.add_argument("--fps", type=int, default=30, help="frame rate of --record")
    a = ap.parse_args()
    if a.replay and (a.drone or a.record):
        ap.error("--replay plays back a recording; --drone and --record need the live path")
    if a.record:
        run_record(a)
    else:
        w, h = (int(x) for x in a.size.split("x"))
        title = "AED rover" + (" + drone" if a.drone else "") + ": live cinematic view (Esc quits)" + (" - rich world" if a.rich else "")
        win = Window(w, h, title)
        (run_replay if a.replay else run_live)(a, win)
        win.root.destroy()
    import os
    sys.stdout.flush()
    os._exit(0)                      # skip the renderer's interpreter-exit teardown


if __name__ == "__main__":
    main()
