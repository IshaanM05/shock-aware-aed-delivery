"""Wind and aerodynamic drag for the comparator quadrotor.

Wind velocity (world frame, x forward = track direction, y left, z up) is the sum of

    * a constant mean wind vector, and
    * a gust modelled as an Ornstein-Uhlenbeck (first-order Gauss-Markov) process per axis,
      the simplest member of the Dryden family of turbulence models:

          g[k+1] = g[k] exp(-dt/tau) + sigma sqrt(1 - exp(-2 dt/tau)) N(0, 1)

      with stationary standard deviation sigma [m/s] and correlation time tau [s]. The
      process is initialised from its stationary distribution so the statistics are correct
      from t = 0. All randomness comes from a seeded ``numpy.random.Generator``.

The wind acts on the vehicle only as a quadratic body drag force applied at the centre of
mass through ``data.xfrc_applied``:

    F_drag = -0.5 rho Cd A |v_rel| v_rel,   v_rel = v - v_wind

No rotor inflow / blade-flapping effects and no drag-induced torque are modelled (documented
limitation). No numeric gust statistics are taken from the literature: gust sigma and
correlation time are ASSUMPTION inputs that experiments sweep explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WindParams:
    """Wind description.

    Attributes:
        mean_mps: mean wind velocity vector (x, y, z) in the world frame [m/s].
        gust_sigma_mps: stationary standard deviation of the horizontal gust per axis [m/s].
        gust_tau_s: gust correlation time [s]. ASSUMPTION (5 s; sensitivity 2-15 s).
        vertical_scale: ratio of vertical to horizontal gust sigma [-]. ASSUMPTION (0.5).
    """

    mean_mps: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gust_sigma_mps: float = 0.0
    gust_tau_s: float = 5.0
    vertical_scale: float = 0.5


class WindField:
    """Seeded mean-plus-OU wind, advanced once per control step."""

    def __init__(self, params: WindParams, dt: float, seed: int) -> None:
        if dt <= 0.0 or params.gust_tau_s <= 0.0:
            raise ValueError("dt and gust_tau_s must be positive")
        self.params = params
        self.dt = dt
        self._rng = np.random.default_rng(seed)
        self._scale = params.gust_sigma_mps * np.array([1.0, 1.0, params.vertical_scale])
        self._decay = math.exp(-dt / params.gust_tau_s)
        self._drive = math.sqrt(1.0 - self._decay**2)
        self._mean = np.array(params.mean_mps, dtype=float)
        self._gust = self._scale * self._rng.standard_normal(3)

    @property
    def velocity(self) -> np.ndarray:
        """Current wind velocity (mean + gust) [m/s]."""
        return self._mean + self._gust

    def step(self) -> np.ndarray:
        """Advance the gust by one step and return the wind velocity [m/s]."""
        noise = self._rng.standard_normal(3)
        self._gust = self._decay * self._gust + self._drive * self._scale * noise
        return self.velocity


def drag_force(
    vel: np.ndarray, wind: np.ndarray, cd: float, area_m2: float, rho: float
) -> np.ndarray:
    """Quadratic drag force on the vehicle [N] for ground velocity ``vel`` and wind ``wind``.

    Args:
        vel: vehicle velocity in the world frame [m/s].
        wind: wind velocity in the world frame [m/s].
        cd: drag coefficient [-].
        area_m2: reference area [m^2].
        rho: air density [kg/m^3].
    """
    v_rel = vel - wind
    return -0.5 * rho * cd * area_m2 * float(np.linalg.norm(v_rel)) * v_rel
