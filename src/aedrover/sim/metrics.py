"""Episode metrics. Definitions are explicit and unit-tested (tests/test_metrics.py).

Payload shock
-------------
MuJoCo accelerometers report *proper* acceleration: at rest they read +g along the local
"up". The payload shock is the low-pass-filtered magnitude of the *dynamic* part

    a_dyn(t) = a_proper(t) - g * u_body(t)

where ``u_body`` is the world-up direction expressed in the chassis frame (so pitch and roll
do not leak into the reading). The signal is sampled at ``fs`` (250 Hz by default) and passed
through a zero-phase 4th-order Butterworth low-pass (80 Hz) before taking the peak, which
mimics a band-limited shock sensor and removes solver-frequency contact chatter.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

G = 9.81
DEFAULT_CUTOFF_HZ = 80.0


@dataclass(frozen=True)
class ShockResult:
    peak_g: float          # peak magnitude of the dynamic payload acceleration [g]
    peak_vertical_g: float
    peak_lateral_g: float  # peak of the in-plane (x, y) magnitude
    rms_g: float
    t_peak: float          # sample index of the peak divided by fs [s]
    frac_over: float       # fraction of samples above ``budget_g`` (nan if no budget)


def dynamic_acceleration(acc: np.ndarray, up_dir: np.ndarray) -> np.ndarray:
    """a_dyn = a_proper - g * u_body, elementwise over samples (N, 3)."""
    return np.asarray(acc, dtype=float) - G * np.asarray(up_dir, dtype=float)


def lowpass(x: np.ndarray, fs: float, cutoff: float = DEFAULT_CUTOFF_HZ, order: int = 4) -> np.ndarray:
    nyq = 0.5 * fs
    wn = min(cutoff, 0.95 * nyq) / nyq
    sos = butter(order, wn, output="sos")
    # sosfiltfilt needs len > 3 * (2 * sections + 1); fall back to unfiltered for tiny signals
    if len(x) <= 3 * (2 * sos.shape[0] + 1) + 1:
        return np.asarray(x, dtype=float)
    return sosfiltfilt(sos, x, axis=0)


def shock_metrics(acc: np.ndarray, up_dir: np.ndarray, fs: float, *,
                  cutoff: float = DEFAULT_CUTOFF_HZ, budget_g: float | None = None) -> ShockResult:
    acc = np.asarray(acc, dtype=float)
    if acc.ndim != 2 or acc.shape[0] < 2:
        return ShockResult(0.0, 0.0, 0.0, 0.0, 0.0, float("nan"))
    dyn = lowpass(dynamic_acceleration(acc, up_dir), fs, cutoff) / G
    mag = np.linalg.norm(dyn, axis=1)
    i = int(np.argmax(mag))
    frac = float(np.mean(mag > budget_g)) if budget_g is not None else float("nan")
    return ShockResult(
        peak_g=float(mag[i]),
        peak_vertical_g=float(np.max(np.abs(dyn[:, 2]))),
        peak_lateral_g=float(np.max(np.linalg.norm(dyn[:, :2], axis=1))),
        rms_g=float(np.sqrt(np.mean(mag**2))),
        t_peak=i / fs,
        frac_over=frac,
    )


def rover_shock(rover, *, cutoff: float = DEFAULT_CUTOFF_HZ, budget_g: float | None = None) -> ShockResult:
    """Shock metrics from a Rover's logged samples."""
    if not rover.acc_log:
        return shock_metrics(np.zeros((0, 3)), np.zeros((0, 3)), rover.log_rate)
    return shock_metrics(np.asarray(rover.acc_log), np.asarray(rover.gdir_log), rover.log_rate,
                         cutoff=cutoff, budget_g=budget_g)
