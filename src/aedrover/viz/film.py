"""Shot assembly: render each shot from its recording, add the HUD, fades and cards, encode to video.

A film is a list of ``Shot`` and ``Card`` items. Shots reuse one render scene per recording; cards are
drawn with Pillow. Frames stream straight into an H.264 encoder (``imageio-ffmpeg``), so memory stays flat.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import uniform_filter1d

from . import hud
from .dressing import dress_scene
from .look import Look, load_look
from .overlays import OverlayConfig
from .post import Grade
from .recording import Recording
from .render_model import RenderScene
from .shots import Rig, smooth_poses, time_map


@dataclass
class Shot:
    name: str
    rec: str                                     # key into the recordings dictionary
    rig: Rig
    frames: int
    start_step: float = 0.0
    base_rate: float = 1.0
    slow: tuple[float, float, float] | None = None       # (centre step, width in seconds, playback rate)
    overlays: OverlayConfig = field(default_factory=OverlayConfig)
    controller: str = ""
    caption: str = ""
    title: tuple[str, str] | None = None
    show_hud: bool = True
    smooth_s: float = 0.35
    grade: Grade | None = None
    dof: float = 0.35                            # 0 disables depth of field
    fade_in: float = 0.25
    fade_out: float = 0.25


@dataclass
class Card:
    name: str
    frames: int
    draw: Callable[[float], np.ndarray]          # progress in [0, 1] -> frame
    fade_in: float = 0.3
    fade_out: float = 0.3


class Film:
    def __init__(self, recordings: dict[str, Recording], size: tuple[int, int] = (1920, 1080), fps: int = 60,
                 look: Look | None = None, backend: str = "filament") -> None:
        self.recs, self.size, self.fps = recordings, size, fps
        self.look = look or load_look()
        self.backend = backend

    def _scene(self, shot: Shot) -> RenderScene:
        rec = self.recs[shot.rec]
        return RenderScene(rec, look=self.look, size=self.size, backend=self.backend, depth=True,
                           builder=lambda rx, r, lk: dress_scene(rx, r, lk, overlays=shot.overlays))

    @staticmethod
    def _fade(k: int, n: int, fps: int, fin: float, fout: float) -> float:
        a = 1.0
        if fin > 0:
            a = min(a, k / (fin * fps))
        if fout > 0:
            a = min(a, (n - 1 - k) / (fout * fps))
        return float(np.clip(a, 0.0, 1.0))

    def shot_frames(self, shot: Shot, progress: Callable[[int, int], None] | None = None):
        """Yield the finished frames of one shot."""
        rec = self.recs[shot.rec]
        scene = self._scene(shot)
        try:
            scene.backend.depth_decoder()
            s = time_map(shot.frames, self.fps, shot.start_step, len(rec), rec.dt, base_rate=shot.base_rate, slow=shot.slow)
            u = np.linspace(0.0, 1.0, shot.frames)
            poses = smooth_poses(shot.rig(scene, s, u), shot.smooth_s * self.fps)
            speed = np.r_[0.0, np.linalg.norm(np.diff(rec.qpos[:, :2], axis=0), axis=1) / rec.dt]
            speed = uniform_filter1d(speed, size=max(int(0.3 / rec.dt), 1), mode="nearest")
            w3 = max(int(0.25 / rec.dt), 1)                       # causal max over the last 0.25 s: catches the spike
            shock = np.max(np.lib.stride_tricks.sliding_window_view(np.r_[np.full(w3 - 1, rec.shock_g[0]), rec.shock_g], w3), axis=1)
            peak = np.maximum.accumulate(rec.shock_g)
            budget = float(rec.meta.get("budget_g", 3.0))
            for k in range(shot.frames):
                sk = float(s[k])
                i = int(round(sk))
                frame = scene.render_finished(sk, poses[k], grade=shot.grade, seed=k, dof=shot.dof)
                if shot.show_hud and not shot.title:
                    lo = max(i - int(4.0 / rec.dt), 0)
                    st = hud.HudState(t=float(rec.t[i]), speed=float(speed[i]), shock=float(shock[i]), peak=float(peak[i]),
                                      budget=budget, controller=shot.controller, caption=shot.caption, history=shock[lo:i + 1])
                    frame = hud.draw_hud(frame, st)
                if shot.title:
                    half = shot.frames * 0.5
                    a = float(np.clip(min(k / (0.8 * self.fps), (shot.frames - 1 - k) / (0.8 * self.fps)), 0, 1))
                    del half
                    frame = hud.title_overlay(frame, shot.title[0], shot.title[1], a)
                f = self._fade(k, shot.frames, self.fps, shot.fade_in, shot.fade_out)
                if f < 1.0:
                    frame = (frame.astype(np.float32) * f).astype(np.uint8)
                if progress:
                    progress(k, shot.frames)
                yield frame
        finally:
            scene.close()

    def card_frames(self, card: Card):
        for k in range(card.frames):
            frame = card.draw(k / max(card.frames - 1, 1))
            f = self._fade(k, card.frames, self.fps, card.fade_in, card.fade_out)
            if f < 1.0:
                frame = (frame.astype(np.float32) * f).astype(np.uint8)
            yield frame

    def render(self, items: list[Shot | Card], out: str | Path, *, crf: int = 18, verbose: bool = True) -> Path:
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        writer = imageio.get_writer(str(out), fps=self.fps, codec="libx264", quality=None, macro_block_size=1,
                                    ffmpeg_params=["-crf", str(crf), "-preset", "slow", "-pix_fmt", "yuv420p",
                                                   "-movflags", "+faststart"])
        n = 0
        try:
            for item in items:
                t1 = time.perf_counter()
                frames = self.shot_frames(item) if isinstance(item, Shot) else self.card_frames(item)
                c = 0
                for frame in frames:
                    writer.append_data(frame)
                    c += 1
                n += c
                if verbose:
                    print(f"  {item.name:<22s} {c:4d} frames in {time.perf_counter() - t1:6.1f}s", flush=True)
        finally:
            writer.close()
        if verbose:
            print(f"wrote {out} ({n} frames, {n / self.fps:.1f} s) in {time.perf_counter() - t0:.0f}s", flush=True)
        return out
