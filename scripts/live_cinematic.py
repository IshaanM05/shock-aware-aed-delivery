"""Watch episodes in the cinematic renderer at real-time pace, in a window, until you close it.

    python scripts/live_cinematic.py                                   # PPO through a crowd
    python scripts/live_cinematic.py --controller dwa --family mixed --seed 5010

Each episode is simulated first (a few seconds on the CPU, the same deterministic physics as the benchmark) and
then played back at real-time pace through the dressed PBR scene with a chase camera, path trail, halo and HUD.
Esc or closing the window quits. Playback drops frames to keep real time if the GPU cannot keep up.
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

from aedrover.viz import hud, post  # noqa: E402
from aedrover.viz.camera import chase  # noqa: E402
from aedrover.viz.dressing import dress_scene  # noqa: E402
from aedrover.viz.overlays import OverlayConfig  # noqa: E402
from aedrover.viz.recording import record_episode  # noqa: E402
from aedrover.viz.render_model import RenderScene  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="ppo", choices=("ppo", "mppi", "dwa", "apf", "pure_pursuit"))
    ap.add_argument("--family", default="crowded")
    ap.add_argument("--seed", type=int, default=5021)
    ap.add_argument("--size", default="1280x720")
    a = ap.parse_args()
    w, h = (int(x) for x in a.size.split("x"))
    root = tk.Tk()
    root.title("AED rover: live cinematic view (Esc quits)")
    label = tk.Label(root, bg="black")
    label.pack()
    alive = {"ok": True}
    root.bind("<Escape>", lambda e: alive.update(ok=False))
    root.protocol("WM_DELETE_WINDOW", lambda: alive.update(ok=False))

    def show(frame: np.ndarray) -> None:
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        label.configure(image=img)
        label.image = img
        root.update()

    seed = a.seed
    while alive["ok"]:
        splash = np.zeros((h, w, 3), np.uint8)
        show(hud.end_card((w, h), f"{a.controller}  |  {a.family}  |  seed {seed}", ["simulating, then building the scene ..."]))
        kw = {"controller_kwargs": {"nthread": 8}} if a.controller == "mppi" else {}
        rec = record_episode(a.controller, a.family, seed, **kw)
        del splash
        scene = RenderScene(rec, size=(w, h), builder=lambda rx, r, lk: dress_scene(rx, r, lk, overlays=OverlayConfig(rollouts=False)))
        speed = np.r_[0.0, np.linalg.norm(np.diff(rec.qpos[:, :2], axis=0), axis=1) / rec.dt]
        peak = np.maximum.accumulate(rec.shock_g)
        cam = None
        t0 = time.perf_counter()
        while alive["ok"]:
            s = (time.perf_counter() - t0) / rec.dt
            if s >= len(rec) - 1:
                break
            p, yaw = scene.rover_pose(s)
            want = chase(p, yaw, back=4.2, height=1.6, swing_deg=22, fovy=52)
            cam = want if cam is None else cam.lerp(want, 0.18)
            frame = post.grade_colour(scene.render(s, cam), post.Grade())
            i = int(s)
            st = hud.HudState(t=float(rec.t[i]), speed=float(speed[i]), shock=float(rec.shock_g[i]), peak=float(peak[i]),
                              controller=a.controller, caption=f"{a.family}, seed {seed}", history=rec.shock_g[max(i - 200, 0):i + 1])
            show(hud.draw_hud(frame, st))
        if alive["ok"]:
            print(f"{a.controller} {a.family} seed {seed}: {rec.meta['outcome']} in {rec.meta['time_s']:.1f} s, "
                  f"peak payload shock {rec.meta['peak_shock_g']:.2f} g", flush=True)
        scene.close()
        seed += 1
    root.destroy()
    import os
    os._exit(0)                      # skip the renderer's interpreter-exit teardown


if __name__ == "__main__":
    main()
