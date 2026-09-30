"""Render the cinematic showcase film (and optionally a README hero GIF and poster) from scratch.

    python scripts/render_showcase.py --quality draft                     # 720p30, a few minutes
    python scripts/render_showcase.py --quality high --hero --poster      # 1080p60 film + GIF + still

Episodes are simulated once on the CPU with the benchmark's exact settings and cached under
``.cache/recordings``; every shot is then rendered from those recordings. Every number on the result
card is read from ``results/benchmark_standard.csv``, never typed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aedrover.analysis.report import delivery_success  # noqa: E402
from aedrover.analysis.stats import wilson_ci  # noqa: E402
from aedrover.viz import hud  # noqa: E402
from aedrover.viz.film import Card, Film, Shot  # noqa: E402
from aedrover.viz.overlays import OverlayConfig  # noqa: E402
from aedrover.viz.recording import Recording, record_episode  # noqa: E402
from aedrover.viz.shots import (  # noqa: E402
    ease,
    rig_chase,
    rig_crane,
    rig_kerb,
    rig_orbit,
    rig_top,
)

CACHE = ROOT / ".cache" / "recordings"
LABEL = {"ppo": "PPO (learned)", "mppi": "MPPI (physics rollouts)", "dwa": "Dynamic window", "apf": "Potential field",
         "pure_pursuit": "Pure pursuit"}
REPO_URL = "github.com/IshaanM05/shock-aware-aed-delivery"

EPISODES = {                      # key -> (controller, family, seed, record rollouts)
    "ppo": ("ppo", "mixed", 5010, False),
    "dwa": ("dwa", "mixed", 5010, False),
    "mppi": ("mppi", "mixed", 5010, True),
    "crowd": ("ppo", "crowded", 5021, False),
}


def get_recording(key: str, refresh: bool) -> Recording:
    ctrl, fam, seed, rollouts = EPISODES[key]
    path = CACHE / f"{ctrl}_{fam}_{seed}{'_rollouts' if rollouts else ''}.npz"
    if path.exists() and not refresh:
        return Recording.load(path)
    t0 = time.perf_counter()
    kw = {"controller_kwargs": {"nthread": 12}} if ctrl == "mppi" else {}
    rec = record_episode(ctrl, fam, seed, capture_rollouts=rollouts, **kw)
    rec.save(path)
    print(f"  recorded {key}: {len(rec)} steps, {rec.meta['outcome']}, peak {rec.meta['peak_shock_g']:.2f} g "
          f"({time.perf_counter() - t0:.0f}s)", flush=True)
    return rec


def result_rows() -> tuple[list[tuple[str, float, float, str]], int]:
    df = pd.read_csv(ROOT / "results" / "benchmark_standard.csv")
    df["safe"] = delivery_success(df)
    rows = []
    for c, g in df.groupby("controller"):
        lo, hi = wilson_ci(int(g.safe.sum()), len(g))
        p = float(g.safe.mean())
        rows.append((LABEL.get(c, c), p, (hi - lo) / 2, f"{len(g)} episodes", c))
    rows.sort(key=lambda r: -r[1])
    return [(a, b, c, d) for a, b, c, d, _ in rows], len(df)


def build_items(recs: dict[str, Recording], fps: int, size: tuple[int, int]) -> list:
    sec = lambda s: int(round(s * fps))                       # noqa: E731
    peak = {k: int(np.argmax(r.shock_g)) for k, r in recs.items()}
    flow = OverlayConfig(rollouts=False)
    clean = OverlayConfig(rollouts=False, lidar=False)      # kerb close-ups: just path and halo
    rows, n_total = result_rows()
    footer = f"{n_total:,} simulated episodes | 5 scenario families | one safety filter and speed cap for every controller"

    def results_card(p: float) -> np.ndarray:
        return hud.results_card(size, "Safe delivery by controller", rows, footer, progress=float(ease(min(p * 2.2, 1.0))),
                                note="in-distribution benchmark; out-of-distribution results are in the repository")

    def end_card(p: float) -> np.ndarray:
        return hud.end_card(size, "Shock-aware sidewalk AED delivery",
                            ["A MuJoCo study of a payload-safe delivery rover", REPO_URL, "Ishaan Mondal"])

    def kerb(key: str, name: str, caption: str) -> Shot:
        return Shot(f"kerb-{key}", key, rig_kerb(), frames=sec(6.0), start_step=peak[key] - 75, slow=(peak[key], 0.55, 0.18),
                    overlays=clean, controller=name, caption=caption, dof=0.30)

    mppi_n = len(recs["mppi"])
    return [
        Shot("title-crane", "ppo", rig_crane((-46, 5, 36), (-5.0, -2.9, 1.9), fov0=62, fov1=48), frames=sec(6.0), start_step=0,
             overlays=flow, controller="PPO", title=("Shock-aware AED delivery", "MuJoCo  |  learned control vs classical navigation"),
             dof=0.25, fade_in=0.6),
        Shot("crowd-chase", "crowd", rig_chase(back=3.9, height=1.35, swing_deg=24), frames=sec(10.0), start_step=120,
             overlays=OverlayConfig(rollouts=False), controller="PPO (learned)", caption="through a crowd, payload-safe", dof=0.30),
        Shot("crowd-orbit", "crowd", rig_orbit(radius=3.7, height=1.5, az0_deg=205, deg_per_frame=15.0 / fps, fovy=44),
             frames=sec(6.0), start_step=520, overlays=OverlayConfig(rollouts=False), controller="PPO (learned)",
             caption="pedestrians, obstacles, lidar", dof=0.35),
        kerb("dwa", "Dynamic window", "classical planner: payload over budget"),
        kerb("mppi", "MPPI", "physics-rollout planner"),
        kerb("ppo", "PPO (learned)", "learned policy: payload within budget"),
        Shot("mppi-rollouts", "mppi", rig_top(height=11.0, back=3.0, fovy=46), frames=sec(10.0), start_step=int(0.12 * mppi_n),
             overlays=OverlayConfig(), controller="MPPI", caption="physics rollouts every 0.1 s (24 of 128 drawn)", dof=0.0),
        Card("results", sec(8.0), results_card),
        Shot("closing-crane", "ppo", rig_crane((-4.6, -2.9, 1.8), (-40, 6, 32), fov0=46, fov1=60), frames=sec(5.0),
             start_step=int(0.72 * len(recs["ppo"])), overlays=flow, controller="PPO", show_hud=False, dof=0.25, fade_out=0.6),
        Card("end-card", sec(5.0), end_card, fade_in=0.6, fade_out=0.8),
    ]


def compress(src: Path, dst: Path, crf: int = 28) -> None:
    """Re-encode the master to a size GitHub accepts (the master stays under .cache, which is ignored)."""
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(src), "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(dst)], check=True)


def make_gif(src: Path, dst: Path, start: float, dur: float, width: int = 840, fps: int = 12, colors: int = 80) -> None:
    import imageio_ffmpeg
    ff = imageio_ffmpeg.get_ffmpeg_exe()
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];"
          f"[b][p]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle")
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{start}", "-t", f"{dur}", "-i", str(src), "-vf", vf, "-loop", "0", str(dst)],
                   check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality", choices=("draft", "high"), default="draft")
    ap.add_argument("--out", default=None, help="output mp4 (default: the master in .cache, compressed to assets/showcase.mp4 for --quality high)")
    ap.add_argument("--hero", action="store_true", help="also write assets/hero.gif from the kerb-strike comparison")
    ap.add_argument("--poster", action="store_true", help="also write assets/showcase_poster.jpg")
    ap.add_argument("--refresh", action="store_true", help="re-simulate the cached episodes")
    a = ap.parse_args()
    size, fps, crf = ((1920, 1080), 60, 18) if a.quality == "high" else ((1280, 720), 30, 24)
    final = ROOT / "assets" / "showcase.mp4"
    out = Path(a.out) if a.out else (CACHE.parent / "showcase_master.mp4" if a.quality == "high" else CACHE.parent / "showcase_draft.mp4")
    print("recording episodes (cached under .cache/recordings) ...", flush=True)
    recs = {k: get_recording(k, a.refresh) for k in EPISODES}
    film = Film(recs, size=size, fps=fps)
    items = build_items(recs, fps, size)
    t0 = time.perf_counter()
    film.render(items, out, crf=crf)
    print(f"total {time.perf_counter() - t0:.0f}s", flush=True)
    if a.quality == "high" and not a.out:
        compress(out, final)
        print(f"wrote {final.relative_to(ROOT)} ({final.stat().st_size / 1e6:.0f} MB; master {out.stat().st_size / 1e6:.0f} MB stays in .cache)", flush=True)
    if a.hero:
        # the dynamic-window kerb strike then the PPO one: the clearest 10 seconds of the film
        starts, t = {}, 0.0
        for it in items:
            starts[it.name] = t
            t += it.frames / fps
        make_gif(out, ROOT / "assets" / "hero.gif", start=starts["kerb-dwa"] + 0.6, dur=4.4)
        print("wrote assets/hero.gif", flush=True)
    if a.poster:
        import imageio.v2 as imageio
        rd = imageio.get_reader(str(out))
        starts, t = {}, 0.0
        for it in items:
            starts[it.name] = t
            t += it.frames / fps
        frame = rd.get_data(int((starts["crowd-orbit"] + 3.0) * fps))
        imageio.imwrite(ROOT / "assets" / "showcase_poster.jpg", frame, quality=92)
        print("wrote assets/showcase_poster.jpg", flush=True)
    import os
    sys.stdout.flush()
    os._exit(0)                      # skip Filament's interpreter-exit teardown (see docs/RENDERING.md)


if __name__ == "__main__":
    main()
