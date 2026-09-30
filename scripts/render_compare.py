"""Side-by-side render of several controllers on the SAME scenario seed (the README hero video).

Each panel shows one controller with a live payload-shock meter (turns red above the budget), the
episode clock and the outcome. Shorter episodes hold their last frame so the panels stay aligned.

    python scripts/render_compare.py --family mixed --seed 5003 --controllers dwa mppi ppo --name hero
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from aedrover.analysis.experiments import controller_spec
from aedrover.control.safety_filter import SafetyFilter
from aedrover.nav.base import make_controller
from aedrover.sim.env import AEDRoverEnv
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]
LABEL = {"dwa": "Dynamic window", "mppi": "MPPI (physics rollouts)", "ppo": "PPO (learned)", "apf": "Potential field",
         "pure_pursuit": "Pure pursuit"}
BUDGET = 3.0


def font(size: int):
    for name in ("segoeui.ttf", "arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def tuned_mppi() -> dict:
    path = ROOT / "configs" / "mppi_tuned.json"
    return json.loads(path.read_text(encoding="utf-8"))["kwargs"] if path.exists() else {}


def record(name: str, family: str, seed: int, cap: float, size: tuple[int, int], every: int, ppo_path: str,
           camera: str, max_frames: int) -> tuple[list[np.ndarray], list[dict], dict]:
    extra = {"path": ppo_path} if name == "ppo" else (tuned_mppi() if name == "mppi" else {})
    _, kw = controller_spec(name, cap, **extra)
    env = AEDRoverEnv(veh=VehicleParams.optimized(), obs_mode="dict", render_size=(size[1], size[0]))
    ctrl, shield = make_controller(name, **kw), SafetyFilter()
    env.reset(seed=seed, options={"family": family})
    ctrl.reset(env)
    frames, tele, k = [], [], 0
    while True:
        o = env.obs
        v, d = ctrl.act(o)
        v, d = shield(o, v, d)
        _, _, term, trunc, info = env.step(np.array([v, d]))
        tele.append({"t": env._t, "shock": info["shock_g"]})
        if k % every == 0 and len(frames) < max_frames:
            frames.append(env.render(camera))
        k += 1
        if term or trunc:
            break
    ep = info["episode"]
    env.close()
    return frames, tele, ep


def overlay(frame: np.ndarray, title: str, t: float, shock: float, peak: float, outcome: str | None) -> np.ndarray:
    img = Image.fromarray(frame)
    dr = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    dr.rectangle([0, 0, w, 34], fill=(252, 252, 251, 235))
    dr.text((10, 6), title, fill=(11, 11, 11), font=font(19))
    dr.text((w - 92, 8), f"t = {t:5.1f} s", fill=(82, 81, 78), font=font(15))
    bar_w = int((w - 24) * min(shock / (BUDGET * 1.6), 1.0))
    dr.rectangle([12, h - 30, w - 12, h - 14], fill=(252, 252, 251, 210))
    dr.rectangle([12, h - 30, 12 + bar_w, h - 14], fill=(208, 59, 59, 255) if shock > BUDGET else (42, 120, 214, 255))
    x_b = 12 + int((w - 24) / 1.6 * 1.0)
    dr.line([x_b, h - 34, x_b, h - 10], fill=(208, 59, 59, 255), width=2)
    dr.text((14, h - 52), f"payload {shock:4.1f} g   peak {peak:4.1f} g   (3 g budget)", fill=(11, 11, 11), font=font(13))
    if outcome:
        ok = outcome == "goal"
        dr.rectangle([0, h // 2 - 20, w, h // 2 + 20], fill=(27, 175, 122, 215) if ok else (208, 59, 59, 215))
        dr.text((w // 2 - 90, h // 2 - 13), "REACHED GOAL" if ok else outcome.upper().replace("_", " "), fill="white", font=font(24))
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", default="mixed")
    ap.add_argument("--seed", type=int, default=5003)
    ap.add_argument("--controllers", nargs="+", default=["dwa", "mppi", "ppo"])
    ap.add_argument("--speed-cap", type=float, default=2.0)
    ap.add_argument("--ppo-path", default="checkpoints/ppo_selected")
    ap.add_argument("--name", default="hero")
    ap.add_argument("--camera", default="chase")
    ap.add_argument("--panel", default="560x340")
    ap.add_argument("--every", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=1100)
    args = ap.parse_args()
    pw, ph = (int(x) for x in args.panel.split("x"))
    runs = {c: record(c, args.family, args.seed, args.speed_cap, (pw, ph), args.every, args.ppo_path, args.camera,
                      args.max_frames) for c in args.controllers}
    n = max(len(f) for f, _, _ in runs.values())
    out_frames = []
    for i in range(n):
        panels = []
        for c in args.controllers:
            frames, tele, ep = runs[c]
            j = min(i, len(frames) - 1)
            idx = min(j * args.every, len(tele) - 1)
            peak = max(x["shock"] for x in tele[: idx + 1])
            done = i >= len(frames) - 1
            panels.append(overlay(frames[j], LABEL.get(c, c), tele[idx]["t"], tele[idx]["shock"], peak,
                                  ep["outcome"] if done else None))
        out_frames.append(np.concatenate(panels, axis=1))
    out = ROOT / "assets"
    out.mkdir(exist_ok=True)
    fps = int(round(50 / args.every))
    imageio.mimsave(out / f"{args.name}.mp4", out_frames, fps=fps, macro_block_size=1, quality=7)
    Image.fromarray(out_frames[len(out_frames) // 3]).save(out / f"{args.name}_still.png")
    for c, (_, _tele, ep) in runs.items():
        print(f"{c:6s} {ep['outcome']:10s} t={ep['time_s']:5.1f}s peak shock {ep['peak_shock_g']:.2f} g")
    print("wrote", out / f"{args.name}.mp4", "and a still")


if __name__ == "__main__":
    main()
