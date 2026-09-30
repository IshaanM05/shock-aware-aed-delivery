"""Procedural textures (numpy only, deterministic): golden-hour sky cube faces and PBR surface maps.

Everything is generated in memory and handed to MuJoCo through the ``assets=`` dictionary of
``MjModel.from_xml_string``, so nothing needs to be shipped or cached on disk.

Cube-face convention of the MuJoCo skybox (measured with sign-coded faces, see docs/RENDERING.md):
for texel coordinates ``u`` (image right) and ``v`` (image down) in [-1, 1] the world direction is
``right: (1, u, -v)``, ``left: (-1, -u, -v)``, ``front: (u, -1, -v)``, ``back: (-u, 1, -v)``,
``up: (u, -v, 1)``; ``down`` is assumed to be ``(u, v, -1)``.
"""

from __future__ import annotations

import io
from functools import lru_cache

import numpy as np
from PIL import Image

from .look import Look

FACES = ("right", "left", "up", "down", "front", "back")


def png_bytes(arr: np.ndarray) -> bytes:
    """Encode an (H, W) or (H, W, 3|4) uint8 array as PNG."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def srgb_encode(lin: np.ndarray) -> np.ndarray:
    """Linear [0, 1] float to 8-bit sRGB."""
    x = np.clip(lin, 0.0, 1.0)
    out = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (out * 255.0 + 0.5).astype(np.uint8)


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ----------------------------------------------------------------------------------------- noise
def tile_noise(size: int, cells: int, seed: int) -> np.ndarray:
    """Smooth value noise in [0, 1), periodic over the image (``cells`` lattice cells per side)."""
    lat = np.random.default_rng(seed).random((cells, cells)).astype(np.float32)
    t = np.arange(size, dtype=np.float32) / size * cells
    i0 = np.floor(t).astype(int)
    f = t - i0
    w = f * f * (3.0 - 2.0 * f)
    i0, i1 = i0 % cells, (i0 + 1) % cells
    a = lat[np.ix_(i0, i0)]
    b = lat[np.ix_(i0, i1)]
    c = lat[np.ix_(i1, i0)]
    d = lat[np.ix_(i1, i1)]
    wy, wx = w[:, None], w[None, :]
    return a + (b - a) * wx + (c - a) * wy + (a - b - c + d) * wx * wy


def fbm_tile(size: int, base_cells: int, octaves: int, seed: int, gain: float = 0.5) -> np.ndarray:
    """Periodic fractal noise in [0, 1]."""
    total, amp, norm = np.zeros((size, size), np.float32), 1.0, 0.0
    for o in range(octaves):
        total += amp * tile_noise(size, base_cells * 2**o, seed + 101 * o)
        norm += amp
        amp *= gain
    return total / norm


def _value_noise_xy(x: np.ndarray, y: np.ndarray, lat: np.ndarray) -> np.ndarray:
    n = lat.shape[0]
    xi, yi = np.floor(x).astype(int), np.floor(y).astype(int)
    xf, yf = x - xi, y - yi
    u, v = xf * xf * (3 - 2 * xf), yf * yf * (3 - 2 * yf)
    x0, x1, y0, y1 = xi % n, (xi + 1) % n, yi % n, (yi + 1) % n
    a, b, c, d = lat[y0, x0], lat[y0, x1], lat[y1, x0], lat[y1, x1]
    return a + (b - a) * u + (c - a) * v + (a - b - c + d) * u * v


def fbm_xy(x: np.ndarray, y: np.ndarray, seed: int, octaves: int = 5) -> np.ndarray:
    """Non-periodic fractal noise in [0, 1] at arbitrary coordinates."""
    lat = np.random.default_rng(seed).random((256, 256)).astype(np.float32)
    total, amp, norm, f = np.zeros_like(x, dtype=np.float32), 1.0, 0.0, 1.0
    for _ in range(octaves):
        total += amp * _value_noise_xy(x * f, y * f, lat)
        norm += amp
        amp, f = amp * 0.5, f * 2.03
    return total / norm


def normal_from_height(h: np.ndarray, strength: float) -> np.ndarray:
    """Tangent-space normal map (uint8 RGB) from a periodic height field."""
    gx = (np.roll(h, -1, 1) - np.roll(h, 1, 1)) * 0.5
    gy = (np.roll(h, -1, 0) - np.roll(h, 1, 0)) * 0.5
    n = np.stack([-gx * strength, -gy * strength, np.ones_like(h)], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return ((n * 0.5 + 0.5) * 255.0 + 0.5).astype(np.uint8)


# ----------------------------------------------------------------------------------------- sky
def _face_dirs(name: str, n: int) -> np.ndarray:
    u = (np.arange(n, dtype=np.float32) + 0.5) / n * 2 - 1
    uu, vv = np.meshgrid(u, u)
    one = np.ones_like(uu)
    d = {"right": (one, uu, -vv), "left": (-one, -uu, -vv), "front": (uu, -one, -vv), "back": (-uu, one, -vv),
         "up": (uu, -vv, one), "down": (uu, vv, -one)}[name]
    v = np.stack(d, -1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def sky_linear(d: np.ndarray, look: Look) -> np.ndarray:
    """Golden-hour sky radiance (linear RGB, may exceed 1 at the sun) for unit directions ``d`` (..., 3)."""
    zen, hor = np.asarray(look.sky_zenith, np.float32), np.asarray(look.sky_horizon, np.float32)
    glow_c, bounce = np.asarray(look.sky_glow, np.float32), np.asarray(look.ground_bounce, np.float32)
    sun = look.sun_dir.astype(np.float32)
    el = np.arcsin(np.clip(d[..., 2], -1.0, 1.0))
    ang = np.arccos(np.clip(d @ sun, -1.0, 1.0))
    # vertical gradient: warm near the horizon, deep blue overhead
    t = np.clip(el / (np.pi / 2), 0.0, 1.0)
    rgb = hor + (zen - hor) * (t ** 0.45)[..., None]
    # forward-scattering glow around the sun, strongest near the horizon
    glow = 0.9 * np.exp(-ang * 2.2) + 2.6 * np.exp(-ang * 9.0) + 9.0 * np.exp(-ang * 34.0)
    rgb = rgb + glow_c * glow[..., None] * (1.0 - 0.55 * t)[..., None]
    # clouds: a high layer projected on a plane, lit warm on the sun side, violet on the far side
    above = el > 0.015
    k = 1.0 / np.maximum(np.tan(np.maximum(el, 0.015) + 0.05), 0.03)
    px, py = d[..., 0] * k * 1.2 + 40.0, d[..., 1] * k * 1.2 + 40.0
    dens = fbm_xy(px.astype(np.float32), py.astype(np.float32), look.seed, octaves=6)
    cover = smoothstep(1.0 - look.cloud_cover - 0.12, 1.0 - look.cloud_cover + 0.22, dens) * above
    lit = np.clip(0.35 + 0.65 * np.exp(-ang * 1.4), 0.0, 1.0)
    cloud_c = (np.array([0.42, 0.30, 0.42], np.float32) * (1 - lit)[..., None]
               + np.array([1.0, 0.58, 0.30], np.float32) * lit[..., None])
    fade = smoothstep(0.015, 0.12, el)
    rgb = rgb * (1 - 0.8 * (cover * fade))[..., None] + cloud_c * (0.8 * cover * fade)[..., None]
    # sun disc
    rgb = rgb + np.array([1.0, 0.92, 0.74], np.float32) * (40.0 * smoothstep(0.045, 0.030, ang))[..., None]
    # below the horizon: warm ground bounce, darker with depth
    below = smoothstep(0.0, -0.06, el)
    return rgb * (1 - below)[..., None] + bounce * (0.6 + 0.4 * smoothstep(-0.6, 0.0, el))[..., None] * below[..., None]


@lru_cache(maxsize=4)
def sky_faces(look: Look, n: int = 768) -> dict[str, bytes]:
    """The six cube faces of the golden-hour sky as PNG bytes (file names ``right.png`` ...)."""
    out = {}
    for name in FACES:
        lin = sky_linear(_face_dirs(name, n), look)
        out[f"{name}.png"] = png_bytes(srgb_encode(lin / (1.0 + 0.04 * lin)))   # soft shoulder keeps the sun from clipping flat
    return out


# ------------------------------------------------------------------------------ surface materials
SHOP_TEXTS = ("CHEMIST", "TEA & SNACKS", "BOOKS", "GROCERY", "TAILOR", "CAFE", "MOBILES", "STATIONERY", "BAKERY", "FRUIT", "SALON", "HARDWARE")
SIGN_COLOURS = ((150, 28, 30), (20, 82, 130), (28, 104, 64), (200, 120, 20), (90, 40, 110), (30, 30, 34), (170, 150, 30))


def _font(size: int):
    from PIL import ImageFont
    for name in ("segoeuib.ttf", "arialbd.ttf", "DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


@lru_cache(maxsize=2)
def shopfront_maps(n: int = 8, w: int = 1024, h: int = 400, seed: int = 41) -> dict[str, bytes]:
    """``n`` shopfront textures (albedo + emissive): a signboard with text above a lit glass front and a door.

    Some shops are shuttered (corrugated grey, unlit). Text is generic English, no brands.
    """
    from PIL import Image, ImageDraw
    rng = np.random.default_rng(seed)
    out: dict[str, bytes] = {}
    for k in range(n):
        alb = Image.new("RGB", (w, h), tuple(int(c) for c in rng.integers(205, 235, size=3)))
        emi = Image.new("RGB", (w, h), (0, 0, 0))
        da, de = ImageDraw.Draw(alb), ImageDraw.Draw(emi)
        sign_h = int(h * 0.26)
        col = SIGN_COLOURS[int(rng.integers(len(SIGN_COLOURS)))]
        da.rectangle([24, 14, w - 24, sign_h], fill=col)
        de.rectangle([24, 14, w - 24, sign_h], fill=tuple(int(c * 0.35) for c in col))
        text = SHOP_TEXTS[int(rng.integers(len(SHOP_TEXTS)))]
        font = _font(int(sign_h * 0.62))
        tw = da.textlength(text, font=font)
        pos = ((w - tw) / 2, 14 + (sign_h - 14) / 2 - sign_h * 0.34)
        da.text(pos, text, font=font, fill=(250, 246, 238))
        de.text(pos, text, font=font, fill=(255, 244, 214))
        gy0, gy1 = sign_h + 26, h - 12
        if rng.random() < 0.22:                                        # shuttered: corrugated steel, unlit
            for x in range(40, w - 40, 14):
                shade = 150 + int(30 * np.sin(x * 0.7))
                da.rectangle([x, gy0, x + 10, gy1], fill=(shade, shade, shade + 6))
        else:
            da.rectangle([40, gy0, w - 40, gy1], fill=(16, 20, 24))
            for y in range(gy0, gy1):                                  # faint warm wash, stronger near the ceiling lights
                t = (y - gy0) / max(gy1 - gy0, 1)
                de.line([(44, y), (w - 44, y)], fill=(int(120 - 70 * t), int(80 - 48 * t), int(40 - 24 * t)))
            rows = 4
            for r in range(rows):                                      # shelves: a bright edge and small coloured goods
                ry = gy0 + 26 + r * (gy1 - gy0 - 60) // rows
                de.rectangle([50, ry + 30, w - 50, ry + 34], fill=(255, 226, 170))
                x = 56
                while x < w - 90:
                    bw = int(rng.integers(14, 44))
                    col = (int(rng.integers(120, 255)), int(rng.integers(70, 210)), int(rng.integers(30, 160)))
                    de.rectangle([x, ry + 30 - int(rng.integers(14, 28)), x + bw, ry + 29], fill=col)
                    x += bw + int(rng.integers(3, 16))
            for lx in range(120, w - 120, 210):                        # ceiling strip lights
                de.rectangle([lx, gy0 + 4, lx + 110, gy0 + 12], fill=(255, 246, 226))
            door_x = int(rng.integers(w // 3, 2 * w // 3))
            da.rectangle([door_x - 50, gy0, door_x + 50, gy1], fill=(14, 16, 18))
            de.rectangle([door_x - 44, gy0 + 6, door_x + 44, gy1], fill=(200, 140, 80))
            for fx in range(40, w - 40, 170):                          # steel mullions
                da.rectangle([fx, gy0, fx + 7, gy1], fill=(40, 40, 44))
                de.rectangle([fx, gy0, fx + 7, gy1], fill=(0, 0, 0))
        out[f"shop{k}_albedo.png"] = png_bytes(np.asarray(alb))
        out[f"shop{k}_emissive.png"] = png_bytes(np.asarray(emi))
    return out


@lru_cache(maxsize=4)
def facade_maps(size: int = 1024, seed: int = 31, bays: int = 4, floors: int = 4, lit_fraction: float = 0.38) -> dict[str, bytes]:
    """A tile of building facade (``bays`` x ``floors`` windows) as albedo, normal, ORM and emissive maps.

    Windows are recessed glass (low roughness, so they mirror the sky) in a plaster wall; a random
    share of them is lit warm (emissive). Wall tint is applied per building through the material colour.
    """
    cw, ch = size // bays, size // floors
    yy, xx = np.mgrid[0:size, 0:size]
    cx, cy = (xx % cw) / cw, (yy % ch) / ch                       # position inside the cell, 0..1
    wx0, wx1, wy0, wy1 = 0.22, 0.78, 0.20, 0.80                   # window opening in cell coordinates
    inside = (cx > wx0) & (cx < wx1) & (cy > wy0) & (cy < wy1)
    frame = ((cx > wx0 - 0.035) & (cx < wx1 + 0.035) & (cy > wy0 - 0.03) & (cy < wy1 + 0.03)) & ~inside
    sill = (cx > wx0 - 0.08) & (cx < wx1 + 0.08) & (cy > wy1 + 0.03) & (cy < wy1 + 0.075)
    mullion = inside & (np.abs(cx - 0.5) < 0.012) | inside & (np.abs(cy - 0.5) < 0.010)
    rng = np.random.default_rng(seed)
    lit_cell = (rng.random((floors, bays)) < lit_fraction)
    lit = lit_cell[(yy // ch), (xx // cw)] & inside & ~mullion
    warm = rng.uniform(0.75, 1.0, size=(floors, bays))[(yy // ch), (xx // cw)]
    stain = fbm_tile(size, 6, 4, seed + 1)
    grain = fbm_tile(size, 96, 2, seed + 2)
    wall = 0.62 + 0.10 * (stain - 0.5) + 0.03 * (grain - 0.5)
    wall = wall * (1 - 0.35 * smoothstep(0.65, 1.0, 1 - cy) * (cy > 0.9))                # darker staining under ledges
    rgb = np.stack([wall * 1.0, wall * 0.96, wall * 0.88], -1)
    rgb[frame] = np.array([0.30, 0.28, 0.26])
    rgb[sill] = np.array([0.78, 0.75, 0.70])
    glass = np.array([0.03, 0.06, 0.10])
    rgb[inside] = glass
    rgb[mullion] = np.array([0.22, 0.21, 0.20])
    albedo = srgb_encode(rgb.astype(np.float32))
    height = np.zeros((size, size), np.float32)
    height[frame] = 0.4
    height[sill] = 0.9
    height[inside] = -0.6
    height[mullion] = -0.2
    height += 0.05 * grain
    normal = normal_from_height(height, 1.5)
    occ = np.ones((size, size), np.float32)
    occ[inside] = 0.75
    occ[frame] = 0.85
    rough = np.full((size, size), 0.88, np.float32) + 0.08 * (grain - 0.5)
    rough[inside & ~mullion] = 0.06
    rough[frame | mullion] = 0.45
    orm = np.stack([occ * 255, np.clip(rough, 0, 1) * 255, np.zeros_like(rough)], -1).astype(np.uint8)
    emis = np.zeros((size, size, 3), np.float32)
    emis[lit] = np.array([1.0, 0.68, 0.30]) * warm[lit][:, None] * 0.9
    return {"facade_albedo.png": png_bytes(albedo), "facade_normal.png": png_bytes(normal),
            "facade_orm.png": png_bytes(orm), "facade_emissive.png": png_bytes(srgb_encode(emis))}


@lru_cache(maxsize=4)
def asphalt_maps(size: int = 1024, seed: int = 11) -> dict[str, bytes]:
    """Dark asphalt with aggregate speckle and patchy wetness: albedo, normal and ORM maps."""
    speck = fbm_tile(size, 64, 3, seed)
    mid = fbm_tile(size, 8, 4, seed + 1)
    tone = 0.10 + 0.05 * mid + 0.04 * (speck - 0.5)
    albedo = srgb_encode(np.repeat(tone[..., None], 3, -1) * np.array([1.0, 0.94, 0.86], np.float32))
    normal = normal_from_height(0.6 * speck + 0.4 * fbm_tile(size, 128, 2, seed + 2), 2.2)
    wet = smoothstep(0.50, 0.72, fbm_tile(size, 4, 4, seed + 3))            # faint damp patches
    rough = np.clip(0.80 - 0.22 * wet + 0.12 * (speck - 0.5), 0.45, 1.0)
    orm = np.stack([np.full_like(rough, 255), rough * 255, np.zeros_like(rough)], -1).astype(np.uint8)
    return {"asphalt_albedo.png": png_bytes(albedo), "asphalt_normal.png": png_bytes(normal),
            "asphalt_orm.png": png_bytes(orm)}


@lru_cache(maxsize=4)
def paving_maps(size: int = 1024, seed: int = 21, tiles: int = 4) -> dict[str, bytes]:
    """Sandstone paving slabs with grout lines: albedo, normal and ORM maps (``tiles`` x ``tiles`` per image)."""
    px = size // tiles
    yy, xx = np.mgrid[0:size, 0:size]
    lx, ly = (xx % px) / px, (yy % px) / px
    edge = np.minimum(np.minimum(lx, 1 - lx), np.minimum(ly, 1 - ly))
    groove = smoothstep(0.035, 0.012, edge)                                    # 1 in the grout
    ti, tj = xx // px, yy // px
    tile_id = (ti + tj * tiles).astype(np.int64)
    rng = np.random.default_rng(seed)
    shade = rng.uniform(-0.05, 0.05, size=tiles * tiles)[tile_id]
    hue = rng.uniform(-0.03, 0.03, size=tiles * tiles)[tile_id]
    grain = fbm_tile(size, 48, 3, seed + 1)
    blotch = fbm_tile(size, 6, 3, seed + 2)
    base = 0.30 + shade + 0.06 * (blotch - 0.5) + 0.04 * (grain - 0.5)
    rgb = np.stack([base * (1.06 + hue), base * 0.90, base * (0.70 - hue)], -1)
    rgb = rgb * (1 - 0.55 * groove[..., None]) + 0.10 * groove[..., None]
    height = (1 - groove) * (0.7 + 0.3 * grain) - 0.0
    rough = np.clip(0.72 + 0.2 * (grain - 0.5) + 0.15 * groove, 0.3, 1.0)
    orm = np.stack([np.clip(255 * (1 - 0.6 * groove), 0, 255), rough * 255, np.zeros_like(rough)], -1).astype(np.uint8)
    return {"paving_albedo.png": png_bytes(srgb_encode(rgb)), "paving_normal.png": png_bytes(normal_from_height(height, 6.0)),
            "paving_orm.png": png_bytes(orm)}
