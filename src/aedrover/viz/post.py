"""Image-space finishing: atmospheric haze, bloom, colour grade, vignette, grain.

Frames arrive display-referred (8-bit sRGB) from the renderer. The chain works on uint8 with OpenCV
(lookup tables for the grade, saturating integer blends, blurs at quarter resolution), which keeps a
1080p frame in the tens of milliseconds; plain float numpy on 25 MB temporaries took seconds.

Haze needs depth. The renderer's depth pass is 8-bit and non-linear, so it would band across a
receding road; ground pixels therefore use the *exact* distance to the ground plane, and the depth
buffer is used only where something clearly stands in front of the ground behind it. Distances are
computed at reduced resolution because the haze weight is smooth.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import cv2
import numpy as np

from .camera import CameraPose
from .look import Look

LOW = 4                                     # haze and bloom are computed at 1/LOW resolution


def resize(img: np.ndarray, size: tuple[int, int], *, area: bool = False) -> np.ndarray:
    """Resize to (width, height)."""
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA if area else cv2.INTER_LINEAR)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT)


# -------------------------------------------------------------------------------------- depth
def ray_directions(pose: CameraPose, size: tuple[int, int]) -> np.ndarray:
    """Unit world-space ray direction of every pixel centre, shape (H, W, 3)."""
    w, h = size
    f = pose.forward
    up = pose.up
    right = np.cross(f, up)
    t = np.tan(np.radians(pose.fovy) / 2.0)
    xs = (np.arange(w, dtype=np.float32) + 0.5) / w * 2 - 1
    ys = 1 - (np.arange(h, dtype=np.float32) + 0.5) / h * 2
    sx, sy = np.meshgrid(xs * t * (w / h), ys * t)
    d = f[None, None, :] + sx[..., None] * right[None, None, :] + sy[..., None] * up[None, None, :]
    return (d / np.linalg.norm(d, axis=-1, keepdims=True)).astype(np.float32)


def ground_distance(pose: CameraPose, size: tuple[int, int], ground_z: float = 0.0, far: float = 1e4) -> np.ndarray:
    """Distance along each pixel ray to the plane ``z = ground_z`` (``far`` above the horizon)."""
    dz = ray_directions(pose, size)[..., 2]
    cam_z = pose.pos[2] - ground_z
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(dz < -1e-4, -cam_z / dz, far)
    return np.clip(t, 0.0, far).astype(np.float32)


class DepthDecoder:
    """Turns the renderer's 8-bit non-linear depth values into metres via a measured table."""

    def __init__(self, distances: np.ndarray, values: np.ndarray) -> None:
        order = np.argsort(values)                       # the encoding falls with distance
        v, d = values[order], distances[order]
        v, keep = np.unique(v, return_index=True)
        self.v, self.d = v, d[keep]

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        # values are sorted ascending, so d[0] is the farthest distance and d[-1] the nearest
        out = np.interp(raw, self.v, self.d, left=self.d[0], right=self.d[-1])
        return np.where(raw <= self.v[0] + 1e-6, 1e4, out).astype(np.float32)


def scene_distance(ground: np.ndarray, buffer_m: np.ndarray | None) -> np.ndarray:
    """Per-pixel distance: exact ground distance, except where the buffer shows something clearly nearer."""
    if buffer_m is None:
        return ground
    return np.where(buffer_m < 0.8 * ground, buffer_m, ground)


def low_distance(pose: CameraPose, size: tuple[int, int], raw_depth_low: np.ndarray | None = None,
                 decoder: DepthDecoder | None = None, ground_z: float = 0.0) -> np.ndarray:
    """Smoothed scene distance at reduced resolution (the size of ``raw_depth_low``, or 1/LOW of ``size``)."""
    if raw_depth_low is not None:
        lh, lw = raw_depth_low.shape[:2]
    else:
        lw, lh = max(size[0] // LOW, 1), max(size[1] // LOW, 1)
    ground = ground_distance(pose, (lw, lh), ground_z)
    buf = None
    if raw_depth_low is not None and decoder is not None:
        buf = decoder(cv2.medianBlur(np.ascontiguousarray(raw_depth_low, dtype=np.float32), 5))   # removes 8-bit dither specks
    return blur(scene_distance(ground, buf), 1.2)


# -------------------------------------------------------------------------------------- effects
@dataclass(frozen=True)
class Grade:
    haze_density: float = 1.0 / 170.0      # 1/metres
    haze_strength: float = 0.85            # cap on the haze blend
    bloom_threshold: float = 0.70
    bloom_strength: float = 0.30
    contrast: float = 1.20
    saturation: float = 1.22
    warm: float = 0.050                    # split tone: highlights toward orange
    cool: float = 0.040                    # split tone: shadows toward blue
    vignette: float = 0.38
    grain: float = 0.012


@lru_cache(maxsize=8)
def _grade_tables(g: Grade) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel tone curve (contrast + split tone) as a LUT, and a saturation matrix."""
    x = np.arange(256, dtype=np.float32) / 255.0
    s = x + (0.5 - x) * 0.0
    s = 0.5 + (s - 0.5) * g.contrast
    hi = np.clip((x - 0.45) / 0.55, 0, 1) ** 1.5
    lo = np.clip((0.55 - x) / 0.55, 0, 1) ** 1.5
    r = s + g.warm * hi - g.cool * 0.6 * lo
    gch = s + g.warm * 0.30 * hi + g.cool * 0.15 * lo
    b = s - g.warm * hi + g.cool * lo
    lut = np.stack([b, gch, r], -1)                       # OpenCV channel order is BGR; we feed RGB, so swap below
    lut = np.stack([r, gch, b], -1)
    lut = (np.clip(lut, 0, 1) * 255 + 0.5).astype(np.uint8).reshape(1, 256, 3)
    k = g.saturation
    lum = np.array([0.2126, 0.7152, 0.0722], np.float32)
    m = (1 - k) * np.outer(np.ones(3, np.float32), lum) + k * np.eye(3, dtype=np.float32)
    return lut, m.astype(np.float32)


@lru_cache(maxsize=4)
def _vignette_mask(size: tuple[int, int], strength: float) -> np.ndarray:
    w, h = size
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x - w / 2) / (w / 2)) ** 2 + ((y - h / 2) / (h / 2)) ** 2
    m = 1.0 - strength * np.clip(r2 / 2.0, 0.0, 1.0) ** 1.3
    return np.repeat((np.clip(m, 0, 1) * 255 + 0.5).astype(np.uint8)[..., None], 3, axis=2)


@lru_cache(maxsize=2)
def _grain_pool(size: tuple[int, int], n: int = 8) -> np.ndarray:
    """``n`` full-size grain frames, uint8 centred on 128 (sigma 32), so any frame can pick one."""
    w, h = size
    rng = np.random.default_rng(1234)
    return np.clip(rng.normal(128.0, 32.0, size=(n, h, w)), 0, 255).astype(np.uint8)


def haze(img: np.ndarray, dist_low: np.ndarray, look: Look, pose: CameraPose, g: Grade) -> np.ndarray:
    """Blend distant pixels toward the warm horizon colour, more so toward the sun."""
    k = 1.0 - np.exp(-dist_low * g.haze_density)
    sun_align = np.clip(float(pose.forward @ look.sun_dir), 0.0, 1.0)
    base = np.array(look.sky_horizon, np.float32)
    base = base / max(float(base.max()), 1e-6)
    tint = np.clip(base * (0.78 + 0.22 * sun_align) + np.array([0.06, 0.08, 0.12], np.float32) * (1 - sun_align), 0, 1)
    tint8 = (np.clip(tint ** (1 / 2.2), 0, 1) * 255 + 0.5).astype(np.uint8)
    w8 = (np.clip(k * g.haze_strength, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)
    h, w = img.shape[:2]
    w8 = cv2.cvtColor(resize(w8, (w, h)), cv2.COLOR_GRAY2RGB)
    keep = cv2.multiply(img, 255 - w8, scale=1 / 255.0)
    add = cv2.multiply(np.broadcast_to(tint8, img.shape).copy(), w8, scale=1 / 255.0)
    return cv2.add(keep, add)


def _blend(a: np.ndarray, b: np.ndarray, w8: np.ndarray) -> np.ndarray:
    """``a * (1 - w) + b * w`` for uint8 images and a uint8 single-channel weight, in saturating integer math."""
    w3 = cv2.merge([w8, w8, w8])
    return cv2.add(cv2.multiply(a, cv2.bitwise_not(w3), scale=1 / 255.0), cv2.multiply(b, w3, scale=1 / 255.0))


def depth_of_field(img: np.ndarray, dist_low: np.ndarray, focus_m: float, strength: float = 0.35) -> np.ndarray:
    """Blur by distance from the focal plane: sharp at ``focus_m``, creamy far away and very close.

    ``strength`` scales the circle of confusion (0.35 is a subtle portrait-lens look). Two blur layers are
    blended by a mask computed at reduced resolution.
    """
    h, w = img.shape[:2]
    d = np.maximum(dist_low, 0.3)
    coc = np.clip(np.abs(1.0 / d - 1.0 / focus_m) * focus_m * strength, 0.0, 1.0)
    coc = np.where(dist_low >= 1e3, min(strength * 2.0, 1.0), coc)                  # sky: as blurred as the far distance
    coc = blur(coc.astype(np.float32), 1.5)
    m1 = resize(np.clip(coc * 2.0, 0.0, 1.0), (w, h))
    m2 = resize(np.clip(coc * 2.0 - 1.0, 0.0, 1.0), (w, h))
    m1_8 = (m1 * 255 + 0.5).astype(np.uint8)
    m2_8 = (m2 * 255 + 0.5).astype(np.uint8)
    soft = cv2.GaussianBlur(img, (0, 0), 2.6)
    wide = resize(cv2.GaussianBlur(resize(img, (w // 2, h // 2), area=True), (0, 0), 5.5), (w, h))
    return _blend(_blend(img, soft, m1_8), wide, m2_8)


def bloom(img: np.ndarray, g: Grade) -> np.ndarray:
    """Soft glow from bright pixels (lamps, lit windows, emissive overlays, the sun)."""
    h, w = img.shape[:2]
    lw, lh = max(w // LOW, 1), max(h // LOW, 1)
    small = resize(img, (lw, lh), area=True).astype(np.float32) / 255.0
    lum = small @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    m = np.clip((lum - g.bloom_threshold) / (1.0 - g.bloom_threshold + 1e-6), 0.0, 1.0) ** 1.5
    src = small * m[..., None]
    acc = blur(src, 1.5) * 0.40 + blur(src, 5.0) * 0.35 + blur(src, 16.0) * 0.25
    up = resize(acc.astype(np.float32), (w, h))
    return cv2.add(img, cv2.convertScaleAbs(up, alpha=255.0 * g.bloom_strength))


def grade_colour(img: np.ndarray, g: Grade) -> np.ndarray:
    lut, m = _grade_tables(g)
    out = cv2.LUT(img, lut)
    return cv2.transform(out, m)


def finish(frame: np.ndarray, *, look: Look, pose: CameraPose, dist_low: np.ndarray | None = None,
           grade: Grade | None = None, seed: int = 0, focus_m: float | None = None, dof: float = 0.0) -> np.ndarray:
    """The full finishing chain on an (H, W, 3) uint8 RGB frame; returns uint8."""
    g = grade or Grade()
    img = np.ascontiguousarray(frame)
    if dist_low is not None:
        img = haze(img, dist_low, look, pose, g)
        if dof > 0 and focus_m:
            img = depth_of_field(img, dist_low, focus_m, dof)
    img = bloom(img, g)
    img = grade_colour(img, g)
    h, w = img.shape[:2]
    img = cv2.multiply(img, _vignette_mask((w, h), g.vignette), scale=1 / 255.0)
    if g.grain > 0:
        pool = _grain_pool((w, h))
        n = pool[seed % len(pool)]
        if (seed // len(pool)) % 2:
            n = n[::-1]
        noise = cv2.cvtColor(np.ascontiguousarray(n), cv2.COLOR_GRAY2RGB)
        amp = g.grain * 255.0 / 32.0
        img = cv2.addWeighted(img, 1.0, noise, amp, -amp * 128.0)
    return img
