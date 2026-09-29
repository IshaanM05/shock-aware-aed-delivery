"""Classical kerb negotiation from the measured kerb study (experiment 02).

``CurbTable`` interpolates the distilled lookup (configs/curb_table.json): minimum successful
climb speed ``v_min(h, mu)`` and the payload shock at that speed. ``CurbNegotiator`` detects a
step from the forward terrain scan, then schedules the approach speed:

* step up  : decelerate to ``v_min(h, mu_assumed) + margin`` (enough momentum to climb, no more,
             because shock grows with speed), hold it until the front wheels are over the lip;
* step down: cap the speed (shock is nearly speed-independent for a drop, so this is mild).

The friction ``mu_assumed`` is a *belief*: the controller does not observe friction. A
conservative belief costs time on grippy kerbs, an optimistic one stalls on wet kerbs. That
trade-off is one of the things the learned and sampling-based controllers can improve on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_DEFAULT_TABLE = Path(__file__).resolve().parents[3] / "configs" / "curb_table.json"
_SCAN_FWD = np.array([0.35, 0.7, 1.1, 1.6, 2.3, 3.2, 4.5])   # must match PerceptionSpec.scan_fwd


def _fill(a: list, fill: float) -> np.ndarray:
    return np.array([[fill if x is None else x for x in row] for row in a], dtype=float)


@dataclass
class CurbTable:
    heights: np.ndarray
    mus: np.ndarray
    v_min: np.ndarray            # (n_h, n_mu): minimum successful climb speed (v_max_table if none)
    peak_g: np.ndarray           # shock at that speed
    v_max_table: float = 3.0

    @classmethod
    def load(cls, path: str | Path = _DEFAULT_TABLE, wheel_radius: float = 0.15) -> CurbTable:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))["up"]
        key = min(raw, key=lambda k: abs(float(k) - wheel_radius))
        t = raw[key]
        return cls(np.array(t["heights"]), np.array(t["mus"]), _fill(t["v_min"], 3.0),
                   _fill(t["peak_g_at_v_min"], 9.0))

    def _interp(self, grid: np.ndarray, h: float, mu: float) -> float:
        hi = float(np.clip(h, self.heights[0], self.heights[-1]))
        mi = float(np.clip(mu, self.mus[0], self.mus[-1]))
        col = np.array([np.interp(mi, self.mus, row) for row in grid])
        return float(np.interp(hi, self.heights, col))

    def v_climb(self, h: float, mu: float) -> float:
        """Minimum climbing speed [m/s] for kerb height ``h`` at friction ``mu`` (bilinear)."""
        return self._interp(self.v_min, h, mu)

    def shock_at_climb(self, h: float, mu: float) -> float:
        return self._interp(self.peak_g, h, mu)


@dataclass
class KerbState:
    kind: str = "none"           # "up", "down", "none"
    h: float = 0.0
    x_face: float = np.inf       # estimated world x of the kerb face


@dataclass
class CurbNegotiator:
    table: CurbTable = field(default_factory=CurbTable.load)
    mu_assumed: float = 0.8
    margin: float = 0.15          # m/s added to the tabulated minimum climb speed
    v_down: float = 0.8           # m/s, descent speed
    a_dec: float = 1.2            # m/s^2 planned deceleration into the kerb
    hold_after: float = 1.3       # m past the face during which the climb speed is held
    detect_thresh: float = 0.03   # m
    max_step: float = 0.25        # m, taller returns are obstacles, not kerbs
    width_tol: float = 0.06       # m, a kerb is uniform across the scan width

    def __post_init__(self) -> None:
        self.state = KerbState()

    def reset(self) -> None:
        self.state = KerbState()

    def observe(self, obs: dict) -> KerbState:
        """Update the kerb estimate from the forward terrain scan (rows = forward distance)."""
        hs = obs["hscan"]
        col = hs[:, hs.shape[1] // 2]
        prev = self.state
        # A kerb is a step that spans the whole scan width and is lower than ``max_step``.
        # Localised raised returns (bollards, planters, pedestrians) are handled by the obstacle logic.
        uniform = (hs.max(axis=1) - hs.min(axis=1)) < self.width_tol
        step_up = (col > self.detect_thresh) & (col < self.max_step) & uniform
        step_dn = (col < -self.detect_thresh) & (col > -self.max_step) & uniform
        det = KerbState()
        up_idx = np.flatnonzero(step_up)
        dn_idx = np.flatnonzero(step_dn)
        if up_idx.size:
            i = int(up_idx[0])
            det = KerbState("up", float(col[i:][step_up[i:]].max()),
                            obs["x"] + 0.5 * (_SCAN_FWD[max(i - 1, 0)] + _SCAN_FWD[i]))
        elif dn_idx.size:
            i = int(dn_idx[0])
            det = KerbState("down", float(-col[i:][step_dn[i:]].min()),
                            obs["x"] + 0.5 * (_SCAN_FWD[max(i - 1, 0)] + _SCAN_FWD[i]))
        if prev.kind != "none" and obs["x"] < prev.x_face + self.hold_after:
            # committed: keep refining while the same kerb is still visible (early far rays only
            # graze the face and underestimate its height), then hold through the crossing
            if det.kind == prev.kind:
                self.state = KerbState(prev.kind, max(prev.h, det.h), det.x_face)
            return self.state
        self.state = det
        return det

    def speed_limit(self, obs: dict) -> float:
        st = self.observe(obs)
        if st.kind == "none":
            return float("inf")
        d = max(st.x_face - obs["x"] - 0.45, 0.0)      # distance from the front axle to the face
        v_t = (self.table.v_climb(st.h, self.mu_assumed) + self.margin) if st.kind == "up" else self.v_down
        if obs["x"] < st.x_face:                       # planned deceleration into the kerb
            return float(np.sqrt(v_t**2 + 2.0 * self.a_dec * d))
        return float(v_t)
