"""Watch the simulation live in the cinematic renderer, in a window, until you close it.

    python scripts/live_cinematic.py                                   # PPO through a crowd
    python scripts/live_cinematic.py --rich                            # a street with collidable parked vehicles, stalls, lamps, trees
    python scripts/live_cinematic.py --controller dwa --family mixed --seed 5010
    python scripts/live_cinematic.py --replay                          # simulate each episode first, then play it back (old behaviour)

The simulation and the renderer run together: each frame the simulation is stepped until its clock catches up with the
wall clock, and the very state it has reached is drawn through the dressed PBR scene with a chase camera, halo, lidar
bubble (and MPPI's sampled rollouts) and a HUD. There is no pre-simulation pass. Planners that cannot keep up in real time
(MPPI is the usual one) simply run slower than real time; the HUD says by how much. ``--rich`` compiles each scenario's street
furniture into the physics (``aedrover.sim.furniture``), so the vehicles, stalls and lamp posts you see are what the rover
can hit; it is a separate mode, never the standard benchmark. Esc or closing the window quits.
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
from aedrover.nav.base import make_controller  # noqa: E402
from aedrover.sim.env import AEDRoverEnv  # noqa: E402
from aedrover.sim.vehicle_mjcf import VehicleParams  # noqa: E402
from aedrover.viz import hud, post  # noqa: E402
from aedrover.viz.camera import chase  # noqa: E402
from aedrover.viz.dressing import dress_scene  # noqa: E402
from aedrover.viz.live import LiveScene, LiveState  # noqa: E402
from aedrover.viz.overlays import OverlayConfig  # noqa: E402
from aedrover.viz.recording import REPO, _tuned_mppi, record_episode  # noqa: E402
from aedrover.viz.render_model import RenderScene  # noqa: E402

MAX_STEPS_PER_FRAME = 12          # keeps the window responsive when a planner is slower than real time
HOLD_S = 2.5                      # how long the last frame stays after an episode ends
ROLLOUT_SHOW_STEPS = 12           # a planner snapshot is drawn for this many control steps


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


def run_live(a, win: Window) -> None:
    w, h = win.size
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name("optimized"), rich=a.rich)
    seed = a.seed
    while win.ok:
        win.splash(f"{a.controller}  |  {a.family}  |  seed {seed}", "building the scene ...")
        env.reset(seed=seed, options={"family": a.family})
        ctrl, shield = build_controller(a.controller), SafetyFilter()
        ctrl.reset(env)
        shield.reset()
        is_mppi = a.controller == "mppi"
        if is_mppi:
            ctrl.capture = True
        scene = LiveScene(env, size=(w, h), rollout_nodes=ctrl.H + 1 if is_mppi else None)
        shock_hist: list[float] = []
        peak, cam, done, info = 0.0, None, False, {}
        snapshot, snap_step = None, -10**9
        t0, t_end, t_prev, lag_t0, lag_s0, speed_factor = time.perf_counter(), None, 0.0, time.perf_counter(), 0.0, 1.0
        caption0 = f"{a.family}, seed {seed}" + (", rich world" if a.rich else "")
        while win.ok:
            wall = (time.perf_counter() - t0) * a.speed
            steps = 0
            while not done and env._t < wall and steps < MAX_STEPS_PER_FRAME:
                obs = env.obs
                v, d = shield(obs, *ctrl.act(obs))
                if is_mppi and ctrl.last_rollouts is not None:
                    snapshot, snap_step = ctrl.last_rollouts, env._steps
                    ctrl.last_rollouts = None
                _, _, term, trunc, info = env.step(np.array([v, d]))
                shock_hist.append(float(info["shock_g"]))
                done, steps = term or trunc, steps + 1
            if steps == MAX_STEPS_PER_FRAME and env._t < wall - 0.25:          # the planner is the bottleneck: do not run away
                t0 += (wall - env._t - 0.25) / a.speed
            if done and t_end is None:
                t_end = time.perf_counter()
            peak = max(peak, shock_hist[-1] if shock_hist else 0.0)
            now = time.perf_counter()
            if now - lag_t0 > 1.0:
                speed_factor, lag_t0, lag_s0 = (env._t - lag_s0) / (now - lag_t0), now, env._t
            ls = LiveState(t=env._t, dt=max(env._t - t_prev, 1e-3), lidar=np.asarray(env.obs["lidar"]),
                           shock_g=max(shock_hist[-20:], default=0.0),
                           rollouts=snapshot if env._steps - snap_step <= ROLLOUT_SHOW_STEPS else None)
            t_prev = env._t
            scene.update(ls)
            p, yaw = scene.rover_pose()
            want = chase(p, yaw, back=4.2, height=1.6, swing_deg=22, fovy=52)
            cam = want if cam is None else cam.lerp(want, 0.18)
            frame = post.grade_colour(scene.render(cam), post.Grade())
            note = f"  |  x{speed_factor:.1f} real time" if speed_factor < 0.9 and not done else ""
            frame = hud.draw_hud(frame, hud.HudState(t=env._t, speed=float(env.obs["vx"]), shock=ls.shock_g, peak=peak,
                                                     controller=a.controller, caption=caption0 + note,
                                                     history=np.array(shock_hist[-200:])))
            win.show(frame)
            if t_end is not None and time.perf_counter() - t_end > HOLD_S:
                break
        if win.ok and info:
            ep = info["episode"]
            print(f"{a.controller} {a.family} seed {seed}: {ep['outcome']}"
                  + (f" ({ep['collision_kind']})" if ep["collision_kind"] else "")
                  + f" in {ep['time_s']:.1f} s, peak payload shock {ep['peak_shock_g']:.2f} g"
                  + (f", {len(env.furniture)} pieces of furniture" if a.rich else ""), flush=True)
        scene.close()
        seed += 1
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
    ap.add_argument("--replay", action="store_true", help="simulate each episode first and play it back, instead of live")
    ap.add_argument("--speed", type=float, default=1.0, help="simulation speed relative to the wall clock (live mode)")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    win = Window(w, h, "AED rover: live cinematic view (Esc quits)" + (" - rich world" if a.rich else ""))
    (run_replay if a.replay else run_live)(a, win)
    win.root.destroy()
    import os
    os._exit(0)                      # skip the renderer's interpreter-exit teardown


if __name__ == "__main__":
    main()
