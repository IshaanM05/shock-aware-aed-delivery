"""2D heads-up display, title and result cards, drawn with Pillow at the output resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = (250, 246, 238)
MUTED = (206, 190, 170)
AMBER = (255, 176, 66)
GREEN = (80, 224, 140)
RED = (255, 84, 70)
PANEL = (16, 12, 10)

_FONT_FILES = {"regular": ("segoeui.ttf", "SegoeUI.ttf", "arial.ttf", "DejaVuSans.ttf"),
               "semibold": ("seguisb.ttf", "segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"),
               "light": ("segoeuil.ttf", "arial.ttf", "DejaVuSans.ttf"),
               "mono": ("consola.ttf", "cour.ttf", "DejaVuSansMono.ttf")}


@lru_cache(maxsize=64)
def font(weight: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in _FONT_FILES[weight]:
        for base in ("", os.path.join(os.environ.get("WINDIR", "C:/Windows"), "Fonts") + os.sep):
            try:
                return ImageFont.truetype(base + name, size)
            except OSError:
                continue
    return ImageFont.load_default()


@dataclass
class HudState:
    t: float                      # episode time [s]
    speed: float                  # [m/s]
    shock: float                  # current payload shock [g]
    peak: float                   # peak shock so far [g]
    budget: float = 3.0
    controller: str = ""
    caption: str = ""
    history: np.ndarray | None = None     # recent shock samples (oldest first) for the sparkline


def _shock_colour(frac: float) -> tuple[int, int, int]:
    return GREEN if frac < 0.45 else (AMBER if frac < 0.8 else RED)


def draw_hud(frame: np.ndarray, st: HudState, alpha: float = 1.0) -> np.ndarray:
    """Composite the HUD onto an (H, W, 3) uint8 frame."""
    h, w = frame.shape[:2]
    s = h / 1080.0
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a = int(255 * alpha)
    m = int(56 * s)

    # ---- lower-left: controller and caption
    f1, f2 = font("semibold", int(40 * s)), font("regular", int(24 * s))
    ph = int(150 * s)
    x0, y0 = m, h - m - ph + int(14 * s)
    d.rounded_rectangle([x0 - int(18 * s), y0 - int(14 * s), x0 + int(620 * s), y0 + ph - int(14 * s)], int(14 * s), fill=(*PANEL, int(150 * alpha)))
    d.rectangle([x0 - int(18 * s), y0 - int(14 * s), x0 - int(12 * s), y0 + ph - int(14 * s)], fill=(*AMBER, a))
    d.text((x0 + int(10 * s), y0 + int(6 * s)), st.controller, font=f1, fill=(*INK, a))
    if st.caption:
        d.text((x0 + int(10 * s), y0 + int(64 * s)), st.caption, font=f2, fill=(*MUTED, a))

    # ---- lower-right: shock number, speed, gauge against the budget, sparkline
    gw = int(520 * s)
    gx, gy = w - m - gw, y0
    d.rounded_rectangle([gx - int(18 * s), gy - int(14 * s), gx + gw + int(18 * s), gy + ph - int(14 * s)], int(14 * s), fill=(*PANEL, int(150 * alpha)))
    frac = min(st.shock / st.budget, 1.6)
    col = _shock_colour(st.shock / st.budget)
    d.text((gx, gy - int(2 * s)), f"{st.shock:0.1f} g", font=font("semibold", int(46 * s)), fill=(*col, a))
    d.text((gx + gw, gy + int(8 * s)), f"{st.speed:0.1f} m/s", font=font("mono", int(28 * s)), fill=(*INK, a), anchor="ra")
    d.text((gx, gy + int(52 * s)), f"peak {st.peak:0.1f} g   budget {st.budget:g} g", font=font("regular", int(21 * s)), fill=(*MUTED, a))
    by, bh = gy + int(86 * s), int(14 * s)
    d.rounded_rectangle([gx, by, gx + gw, by + bh], int(7 * s), fill=(255, 255, 255, int(40 * alpha)))
    d.rounded_rectangle([gx, by, gx + max(int(gw * min(frac / 1.6, 1.0)), int(14 * s)), by + bh], int(7 * s), fill=(*col, a))
    tick = gx + int(gw / 1.6)
    d.rectangle([tick - 1, by - int(6 * s), tick + 1, by + bh + int(6 * s)], fill=(*INK, a))
    if st.history is not None and len(st.history) > 2:
        hist = np.clip(st.history / (st.budget * 1.6), 0, 1)
        sx0, sy0, sh = gx, by + bh + int(10 * s), int(20 * s)
        pts = [(sx0 + i / (len(hist) - 1) * gw, sy0 + sh - v * sh) for i, v in enumerate(hist)]
        d.line(pts, fill=(*MUTED, int(200 * alpha)), width=max(int(2 * s), 1))
        ty = sy0 + sh - min(st.budget / (st.budget * 1.6), 1) * sh
        d.line([(sx0, ty), (sx0 + gw, ty)], fill=(*RED, int(110 * alpha)), width=1)

    # ---- top-left: clock
    d.text((m, m - int(8 * s)), f"t = {st.t:05.2f} s", font=font("mono", int(26 * s)), fill=(*INK, int(200 * alpha)))
    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))


def _gradient(size: tuple[int, int]) -> Image.Image:
    w, h = size
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    top, bot = np.array([14, 18, 40], np.float32), np.array([255, 150, 70], np.float32)
    g = (top * (1 - y ** 1.6)[..., None] + bot * (y ** 1.6)[..., None]) * np.ones((1, w, 1), np.float32)
    g *= (0.85 - 0.45 * np.abs(np.linspace(-1, 1, w))[None, :, None] ** 2)
    return Image.fromarray(np.clip(g, 0, 255).astype(np.uint8))


def text_center(d: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt, fill) -> None:
    d.text(xy, text, font=fnt, fill=fill, anchor="mm")


def title_overlay(frame: np.ndarray, title: str, subtitle: str, alpha: float) -> np.ndarray:
    """Large title lower third over a live frame (``alpha`` fades it in and out)."""
    h, w = frame.shape[:2]
    s = h / 1080.0
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    band = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bd = ImageDraw.Draw(band)
    for i in range(int(420 * s)):
        bd.line([(0, h - i), (w, h - i)], fill=(10, 8, 14, int(170 * (1 - i / (420 * s)) ** 1.5 * alpha)))
    a = int(255 * alpha)
    d.text((int(96 * s), h - int(300 * s)), title, font=font("semibold", int(92 * s)), fill=(*INK, a))
    d.text((int(100 * s), h - int(180 * s)), subtitle, font=font("light", int(38 * s)), fill=(*AMBER, a))
    return np.asarray(Image.alpha_composite(Image.alpha_composite(img, band), ov).convert("RGB"))


def results_card(size: tuple[int, int], heading: str, rows: list[tuple[str, float, float, str]], footer: str,
                 progress: float = 1.0, note: str = "") -> np.ndarray:
    """Dark golden-hour card with horizontal bars. ``rows`` = (label, value 0..1, ci half-width, note)."""
    w, h = size
    s = h / 1080.0
    base = _gradient(size).convert("RGBA")
    ov = Image.new("RGBA", size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    d.text((int(120 * s), int(110 * s)), heading, font=font("semibold", int(66 * s)), fill=(*INK, 255))
    d.text((int(124 * s), int(200 * s)), "safe delivery: goal reached with payload shock under 3 g", font=font("light", int(32 * s)),
           fill=(*MUTED, 255))
    top, rowh = int(330 * s), int(112 * s)
    bx0, bx1 = int(560 * s), w - int(240 * s)
    for i, (label, val, ci, row_note) in enumerate(rows):
        y = top + i * rowh
        d.text((int(120 * s), y + int(14 * s)), label, font=font("semibold", int(40 * s)), fill=(*INK, 255))
        d.rounded_rectangle([bx0, y + int(10 * s), bx1, y + int(62 * s)], int(10 * s), fill=(255, 255, 255, 30))
        frac = float(np.clip(val * progress, 0, 1))
        colour = AMBER if i == 0 else (236, 222, 200)
        d.rounded_rectangle([bx0, y + int(10 * s), bx0 + max(int((bx1 - bx0) * frac), int(10 * s)), y + int(62 * s)], int(10 * s),
                            fill=(*colour, 255))
        if progress >= 0.98:
            lo, hi = np.clip(val - ci, 0, 1), np.clip(val + ci, 0, 1)          # 95% interval whisker
            cy = y + int(36 * s)
            d.line([(bx0 + (bx1 - bx0) * lo, cy), (bx0 + (bx1 - bx0) * hi, cy)], fill=(20, 14, 10, 230), width=max(int(3 * s), 1))
            for xe in (lo, hi):
                d.line([(bx0 + (bx1 - bx0) * xe, cy - int(9 * s)), (bx0 + (bx1 - bx0) * xe, cy + int(9 * s))], fill=(20, 14, 10, 230),
                       width=max(int(3 * s), 1))
            d.text((bx1 + int(24 * s), y + int(14 * s)), f"{100 * val:0.1f}%", font=font("semibold", int(40 * s)), fill=(*INK, 255))
            d.text((bx0 + int(14 * s), y + int(70 * s)), row_note, font=font("regular", int(22 * s)), fill=(*MUTED, 255))
    d.text((int(124 * s), h - int(150 * s)), footer, font=font("regular", int(26 * s)), fill=(*MUTED, 255))
    if note:
        d.text((int(124 * s), h - int(106 * s)), note, font=font("regular", int(24 * s)), fill=(*AMBER, 255))
    return np.asarray(Image.alpha_composite(base, ov).convert("RGB"))


def end_card(size: tuple[int, int], title: str, lines: list[str], alpha: float = 1.0) -> np.ndarray:
    w, h = size
    s = h / 1080.0
    img = _gradient(size).convert("RGBA")
    d = ImageDraw.Draw(img)
    a = int(255 * alpha)
    text_center(d, (w / 2, h * 0.36), title, font("semibold", int(84 * s)), (*INK, a))
    for i, ln in enumerate(lines):
        text_center(d, (w / 2, h * 0.52 + i * 60 * s), ln, font("light", int(38 * s)), (*(AMBER if i == 0 else MUTED), a))
    return np.asarray(img.convert("RGB"))


# ---------------------------------------------------------------------------------- dispatch HUD
@dataclass
class DispatchHud:
    """What the rover-versus-drone shot shows: one dispatch clock, a progress bar per vehicle, a caption and a result panel."""

    clock_s: float                                # simulated time since dispatch [s]
    drone_frac: float                             # 0..1 of the straight-line distance flown
    rover_frac: float                             # 0..1 of the route driven
    drone_text: str                               # right-hand note on the drone row (e.g. "612 m to go")
    rover_text: str
    drone_arrival_s: float
    rover_arrival_s: float
    title: str = ""
    caption: str = ""
    footnote: str = ""
    result_title: str = ""
    result_rows: tuple[tuple[str, str, tuple[int, int, int]], ...] = ()    # (label, value, colour)
    result_note: str = ""
    result_alpha: float = 0.0
    shock: float | None = None                    # live runs: the rover's payload shock now, its peak and the budget [g]
    peak: float = 0.0
    budget: float = 3.0


def mmss(seconds: float) -> str:
    s = int(round(max(seconds, 0.0)))
    return f"{s // 60:02d}:{s % 60:02d}"


def _bar(d: ImageDraw.ImageDraw, x0: int, y: int, w: int, h: int, frac: float, colour, mark_at: float, a: int, s: float) -> None:
    d.rounded_rectangle([x0, y, x0 + w, y + h], h // 2, fill=(255, 255, 255, int(46 * a / 255)))
    fw = max(int(w * float(np.clip(frac, 0.0, 1.0))), h)
    d.rounded_rectangle([x0, y, x0 + fw, y + h], h // 2, fill=(*colour, a))
    tick = x0 + int(w * mark_at)
    d.rectangle([tick - 1, y - int(5 * s), tick + 1, y + h + int(5 * s)], fill=(*INK, int(a * 0.8)))


def draw_dispatch_hud(frame: np.ndarray, st: DispatchHud, alpha: float = 1.0) -> np.ndarray:
    """Composite the dispatch HUD onto an (H, W, 3) uint8 frame."""
    h, w = frame.shape[:2]
    s = h / 1080.0
    img = Image.fromarray(frame).convert("RGBA")
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(ov)
    a = int(255 * alpha)
    m = int(56 * s)

    # ---- top-left: dispatch clock and one progress bar per vehicle (the tick marks where each one arrives)
    pw, ph = int(1010 * s), int(278 * s)
    d.rounded_rectangle([m - int(18 * s), m - int(14 * s), m + pw, m + ph], int(14 * s), fill=(*PANEL, int(150 * alpha)))
    d.rectangle([m - int(18 * s), m - int(14 * s), m - int(12 * s), m + ph], fill=(*AMBER, a))
    d.text((m + int(10 * s), m - int(6 * s)), "TIME SINCE DISPATCH", font=font("regular", int(22 * s)), fill=(*MUTED, a))
    d.text((m + int(10 * s), m + int(22 * s)), mmss(st.clock_s), font=font("mono", int(78 * s)), fill=(*INK, a))
    bx, bw, bh = m + int(190 * s), int(470 * s), int(16 * s)
    for i, (label, frac, colour, note, arrive) in enumerate((("DRONE", st.drone_frac, AMBER, st.drone_text, st.drone_arrival_s),
                                                             ("ROVER", st.rover_frac, GREEN, st.rover_text, st.rover_arrival_s))):
        y = m + int(128 * s) + i * int(66 * s)
        d.text((m + int(10 * s), y - int(6 * s)), label, font=font("semibold", int(30 * s)), fill=(*INK, a))
        _bar(d, bx, y + int(4 * s), bw, bh, frac, colour, 1.0, a, s)
        d.text((bx + bw + int(22 * s), y - int(10 * s)), note, font=font("regular", int(25 * s)), fill=(*MUTED, a))
        arrives = f"arrives {mmss(arrive)}" if np.isfinite(arrive) else "en route"
        d.text((bx + bw + int(22 * s), y + int(24 * s)), arrives, font=font("mono", int(21 * s)), fill=(*colour, a))

    # ---- lower-left: what is being shown
    f1, f2 = font("semibold", int(40 * s)), font("regular", int(24 * s))
    cph = int(150 * s)
    x0, y0 = m, h - m - cph + int(14 * s)
    d.rounded_rectangle([x0 - int(18 * s), y0 - int(14 * s), x0 + int(900 * s), y0 + cph - int(14 * s)], int(14 * s), fill=(*PANEL, int(150 * alpha)))
    d.rectangle([x0 - int(18 * s), y0 - int(14 * s), x0 - int(12 * s), y0 + cph - int(14 * s)], fill=(*AMBER, a))
    d.text((x0 + int(10 * s), y0 + int(6 * s)), st.title, font=f1, fill=(*INK, a))
    if st.caption:
        d.text((x0 + int(10 * s), y0 + int(64 * s)), st.caption, font=f2, fill=(*MUTED, a))
    if st.footnote:
        d.rectangle([0, h - int(54 * s), w, h], fill=(*PANEL, int(170 * alpha)))
        d.text((x0 - int(12 * s), h - int(40 * s)), st.footnote, font=font("regular", int(19 * s)), fill=(*MUTED, int(235 * alpha)))

    # ---- lower-right: the rover's payload shock, for live runs
    if st.shock is not None:
        gw = int(430 * s)
        gx, gy = w - m - gw, h - m - cph + int(14 * s)
        d.rounded_rectangle([gx - int(18 * s), gy - int(14 * s), gx + gw + int(18 * s), gy + cph - int(14 * s)], int(14 * s),
                            fill=(*PANEL, int(150 * alpha)))
        col = _shock_colour(st.shock / st.budget)
        d.text((gx, gy - int(2 * s)), f"{st.shock:0.1f} g", font=font("semibold", int(46 * s)), fill=(*col, a))
        d.text((gx + gw, gy + int(8 * s)), "rover payload", font=font("regular", int(22 * s)), fill=(*MUTED, a), anchor="ra")
        d.text((gx, gy + int(56 * s)), f"peak {st.peak:0.1f} g   budget {st.budget:g} g", font=font("regular", int(21 * s)), fill=(*MUTED, a))
        by, bh = gy + int(94 * s), int(14 * s)
        d.rounded_rectangle([gx, by, gx + gw, by + bh], int(7 * s), fill=(255, 255, 255, int(40 * alpha)))
        d.rounded_rectangle([gx, by, gx + max(int(gw * min(st.shock / (st.budget * 1.6), 1.0)), int(14 * s)), by + bh], int(7 * s),
                            fill=(*col, a))
        tick = gx + int(gw / 1.6)
        d.rectangle([tick - 1, by - int(6 * s), tick + 1, by + bh + int(6 * s)], fill=(*INK, a))

    # ---- top-right: survival from the clinical rows, fading in at the end of the shot
    if st.result_alpha > 0.01 and st.result_rows:
        ra = int(255 * alpha * st.result_alpha)
        rw, rh = int(700 * s), int((96 + 52 * len(st.result_rows) + (40 if st.result_note else 0)) * s)
        rx0, ry0 = w - m - rw, m
        d.rounded_rectangle([rx0 - int(18 * s), ry0 - int(14 * s), rx0 + rw + int(18 * s), ry0 + rh], int(14 * s),
                            fill=(*PANEL, int(170 * alpha * st.result_alpha)))
        d.text((rx0, ry0), st.result_title, font=font("semibold", int(28 * s)), fill=(*INK, ra))
        for i, (label, value, colour) in enumerate(st.result_rows):
            y = ry0 + int(54 * s) + i * int(52 * s)
            d.text((rx0, y), label, font=font("regular", int(28 * s)), fill=(*MUTED, ra))
            d.text((rx0 + rw, y - int(2 * s)), value, font=font("semibold", int(34 * s)), fill=(*colour, ra), anchor="ra")
        if st.result_note:
            d.text((rx0, ry0 + rh - int(40 * s)), st.result_note, font=font("regular", int(19 * s)), fill=(*MUTED, ra))
    return np.asarray(Image.alpha_composite(img, ov).convert("RGB"))
