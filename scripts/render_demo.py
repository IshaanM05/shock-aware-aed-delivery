"""Render an episode to GIF / MP4 (offscreen MuJoCo) plus a telemetry plot.

    python scripts/render_demo.py --controller dwa --family kerb --seed 1003 --name kerb_dwa
    python scripts/render_demo.py --controller dwa --family crowded --seed 1001 --name crowd_dwa

Outputs go to assets/<name>.{gif,mp4} and assets/<name>_telemetry.png.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from aedrover.control.safety_filter import SafetyFilter  # noqa: E402
from aedrover.nav.base import make_controller  # noqa: E402
from aedrover.sim.env import AEDRoverEnv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", default="dwa")
    ap.add_argument("--family", default="kerb")
    ap.add_argument("--seed", type=int, default=1003)
    ap.add_argument("--name", default="demo")
    ap.add_argument("--camera", default="chase")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=400)
    ap.add_argument("--every", type=int, default=2, help="render every k-th control step (50 Hz / k fps)")
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--max-frames", type=int, default=900)
    args = ap.parse_args()

    env = AEDRoverEnv(obs_mode="dict", render_size=(args.height, args.width))
    ctrl = make_controller(args.controller)
    shield = None if args.no_shield else SafetyFilter()
    env.reset(seed=args.seed, options={"family": args.family})
    ctrl.reset(env)
    frames, tele = [], []
    k = 0
    while True:
        o = env.obs
        v, d = ctrl.act(o)
        if shield is not None:
            v, d = shield(o, v, d)
        _, _, term, trunc, info = env.step(np.array([v, d]))
        tele.append((o["t"], o["x"], o["vx"], v, info["shock_g"], info["clearance"] if np.isfinite(info["clearance"]) else np.nan))
        if k % args.every == 0 and len(frames) < args.max_frames:
            frames.append(env.render(args.camera))
        k += 1
        if term or trunc:
            break
    ep = info["episode"]
    print(f"{args.controller}/{args.family}/seed {args.seed}: {ep['outcome']}  t={ep['time_s']:.1f}s  "
          f"peak shock {ep['peak_shock_g']:.2f} g  ({len(frames)} frames)")

    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    fps = int(round(50 / args.every))
    imageio.mimsave(out / f"{args.name}.mp4", frames, fps=fps, macro_block_size=1, quality=7)
    # compact GIF for READMEs: half resolution, every other frame
    from PIL import Image

    small = [np.asarray(Image.fromarray(f).resize((480, int(480 * f.shape[0] / f.shape[1])), Image.LANCZOS))
             for f in frames[::2]]
    imageio.mimsave(out / f"{args.name}.gif", small, duration=1000 / (fps / 2), loop=0)

    t = np.array(tele)
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(t[:, 0], t[:, 2], label="measured v")
    ax[0].plot(t[:, 0], t[:, 3], "--", label="commanded v")
    ax[0].set_ylabel("speed [m/s]")
    ax[0].legend(loc="upper right")
    ax[1].plot(t[:, 0], t[:, 4], color="tab:red")
    ax[1].axhline(3.0, color="k", ls=":", lw=1, label="3 g budget")
    ax[1].set_ylabel("payload shock [g]")
    ax[1].legend(loc="upper right")
    ax[2].plot(t[:, 0], t[:, 5], color="tab:green")
    ax[2].axhline(0.0, color="k", lw=0.8)
    ax[2].set_ylabel("clearance [m]")
    ax[2].set_xlabel("time [s]")
    fig.suptitle(f"{args.controller} on {args.family} (seed {args.seed}): {ep['outcome']}, "
                 f"peak {ep['peak_shock_g']:.2f} g", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"{args.name}_telemetry.png", dpi=130)
    print("wrote", out / f"{args.name}.gif", out / f"{args.name}.mp4", out / f"{args.name}_telemetry.png")
    env.close()


if __name__ == "__main__":
    main()
