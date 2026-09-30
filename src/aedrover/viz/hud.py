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
