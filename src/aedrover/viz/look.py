"""The visual identity of a render (time of day, light, grade) as plain data.

The defaults are the calibrated golden-hour look; ``configs/render/golden_hour.yaml`` overrides any
field. Light intensities are in the physical units the PBR renderer uses once ``intensity`` is set:
a sun of about 1e5 lux with an image light of about 3e4 is well exposed; 1 to 100 render black.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = REPO / "configs" / "render" / "golden_hour.yaml"


@dataclass(frozen=True)
class Look:
    name: str = "golden_hour"
    # sun: azimuth from +x (the rover's heading) toward +y, elevation above the horizon
    sun_azimuth_deg: float = 4.0
    sun_elevation_deg: float = 15.0
    sun_color: tuple[float, float, float] = (1.0, 0.72, 0.46)
    sun_lux: float = 60000.0
    ibl_intensity: float = 25000.0
    # sky (linear RGB)
    sky_zenith: tuple[float, float, float] = (0.07, 0.17, 0.44)
    sky_horizon: tuple[float, float, float] = (1.0, 0.52, 0.24)
    sky_glow: tuple[float, float, float] = (1.0, 0.46, 0.16)
    ground_bounce: tuple[float, float, float] = (0.22, 0.14, 0.09)
    cloud_cover: float = 0.45
    seed: int = 7

    @property
    def sun_dir(self) -> np.ndarray:
        """Unit vector pointing from the scene toward the sun."""
        az, el = np.radians(self.sun_azimuth_deg), np.radians(self.sun_elevation_deg)
        return np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    @property
    def light_dir(self) -> np.ndarray:
        """Direction the sunlight travels (the MJCF light ``dir``)."""
        return -self.sun_dir


def load_look(path: Path | str | None = None, **overrides) -> Look:
    """Defaults, overridden by the YAML file (if present) and then by keyword arguments."""
    look = Look()
    p = Path(path) if path else DEFAULT_CONFIG
    if p.exists():
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        known = {f.name for f in fields(Look)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"{p.name}: unknown look fields {sorted(unknown)}")
        look = replace(look, **{k: (tuple(v) if isinstance(v, list) else v) for k, v in raw.items()})
    return replace(look, **overrides) if overrides else look
