"""Watch a controller drive live in MuJoCo's interactive viewer (real time, episodes repeat until you close it).

    python scripts/live_viewer.py                                  # PPO through a crowd
    python scripts/live_viewer.py --controller dwa --family mixed --seed 5010
    python scripts/live_viewer.py --rich                           # collidable street furniture (a window per episode)

Mouse: left-drag rotates, right-drag pans, wheel zooms, space pauses. Needs a display; no GPU beyond OpenGL.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aedrover.analysis.experiments import controller_spec  # noqa: E402
from aedrover.control.safety_filter import SafetyFilter  # noqa: E402
from aedrover.nav.base import make_controller  # noqa: E402
from aedrover.sim.env import AEDRoverEnv  # noqa: E402
from aedrover.sim.vehicle_mjcf import VehicleParams  # noqa: E402


def run_episode(a, env, ctrl, shield, v, seed: int) -> bool:
    """One episode in the viewer ``v``; False when the window was closed."""
    print(f"episode: {a.controller}, {a.family}, seed {seed}" + (f", {len(env.furniture)} pieces of furniture" if a.rich else ""), flush=True)
    while v.is_running():
        t0 = time.perf_counter()
        obs = env.obs
        vv, dl = shield(obs, *ctrl.act(obs))
        _, _, term, trunc, info = env.step(np.array([vv, dl]))
        v.sync()
        time.sleep(max(0.0, env.dt / a.speed - (time.perf_counter() - t0)))
        if term or trunc:
            ep = info["episode"]
            print(f"  {ep['outcome']} in {ep['time_s']:.1f} s, peak payload shock {ep['peak_shock_g']:.2f} g", flush=True)
            time.sleep(1.0)
            return True
    return False


def track(v, env) -> None:
    v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    v.cam.trackbodyid = env.world.b_chassis
    v.cam.distance, v.cam.elevation, v.cam.azimuth = 7.0, -22.0, 200.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="ppo", choices=("ppo", "mppi", "dwa", "apf", "pure_pursuit"))
    ap.add_argument("--family", default="crowded")
    ap.add_argument("--seed", type=int, default=5021)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--rich", action="store_true",
                    help="collidable street furniture (a separate mode); the world is rebuilt per episode, so each gets its own window")
    a = ap.parse_args()

    extra = {"path": str(ROOT / "models" / "ppo_selected")} if a.controller == "ppo" else (
        {"nthread": 8} if a.controller == "mppi" else {})
    name, kw = controller_spec(a.controller, 2.0, **extra)
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.optimized(), rich=a.rich)
    ctrl, shield = make_controller(name, **kw), SafetyFilter()
    seed = a.seed
    if a.rich:
        # a rich environment compiles a new model on every reset, and a viewer is bound to one model: one window per episode
        while True:
            env.reset(seed=seed, options={"family": a.family})
            ctrl.reset(env)
            shield.reset()
            with mujoco.viewer.launch_passive(env.world.model, env.world.data) as v:
                track(v, env)
                if not run_episode(a, env, ctrl, shield, v, seed):
                    return
            seed += 1
    m, d = env.world.model, env.world.data
    with mujoco.viewer.launch_passive(m, d) as v:
        track(v, env)
        while v.is_running():
            env.reset(seed=seed, options={"family": a.family})
            ctrl.reset(env)
            shield.reset()
            if not run_episode(a, env, ctrl, shield, v, seed):
                break
            seed += 1


if __name__ == "__main__":
    main()
