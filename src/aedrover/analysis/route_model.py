"""Hierarchical route model: compose per-segment simulation results into route-level outcomes.

Physics is simulated per 36 m *segment type* (flat sidewalk, crowded sidewalk, kerb crossing,
crossing with pedestrians). A route of ``distance_m`` metres with ``n_crossings`` street
crossings is a sequence of segments; each segment is drawn (bootstrap) from the empirical
episodes of the matching family, so the composite inherits the measured distribution of times,
failures and payload shocks instead of a single mean speed.

* travel time = sum of the sampled segment times;
* the route *fails* (the rover never arrives) if any sampled segment episode failed;
* the delivery is *unsafe* if any segment's payload peak shock exceeded the budget
  (the AED may be damaged); unsafe deliveries are treated like failures for survival purposes.

Assumptions (stated, not hidden): segments are independent draws; the segment length is the
simulated 36 m; a kerb-crossing segment contains one down/up kerb pair (a full street crossing).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SEGMENT_M = 36.0
BUDGET_G = 3.0


@dataclass(frozen=True)
class RouteSamples:
    time_min: np.ndarray      # travel time of every simulated route [min]; inf where the route failed
    arrived: np.ndarray       # bool: every segment succeeded
    safe: np.ndarray          # bool: arrived and no segment exceeded the shock budget
    n_segments: int
    n_crossing_segments: int


def compose_routes(episodes: pd.DataFrame, distance_m: float, n_crossings: int, *, crowd_prob: float = 0.3,
                   n: int = 2000, seed: int = 0, budget_g: float = BUDGET_G) -> RouteSamples:
    """Bootstrap ``n`` routes from one controller's per-segment episodes.

    ``episodes`` must contain the columns ``family, success, time_s, peak_shock_g``. Crossing
    segments are drawn from ``kerb`` (with probability ``1 - crowd_prob``) or ``mixed``; plain
    segments from ``flat_clear`` or ``crowded``.
    """
    rng = np.random.default_rng(seed)
    n_seg = max(1, int(np.ceil(distance_m / SEGMENT_M)))
    n_cross = int(min(max(n_crossings, 0), n_seg))
    pools = {f: g[["success", "time_s", "peak_shock_g"]].to_numpy() for f, g in episodes.groupby("family")}
    need = ("kerb", "mixed", "flat_clear", "crowded")
    missing = [f for f in need if f not in pools]
    if missing:
        raise ValueError(f"episodes lack families {missing}")

    def draw(fams: list[str], count: int) -> np.ndarray:
        """(n, count, 3) array of success, time, shock drawn per segment."""
        which = rng.random((n, count)) < crowd_prob
        out = np.empty((n, count, 3))
        for j, fam in enumerate(fams):
            sel = which == bool(j)
            pool = pools[fam]
            out[sel] = pool[rng.integers(0, len(pool), size=int(sel.sum()))]
        return out

    parts = []
    if n_cross:
        parts.append(draw(["kerb", "mixed"], n_cross))
    if n_seg - n_cross:
        parts.append(draw(["flat_clear", "crowded"], n_seg - n_cross))
    seg = np.concatenate(parts, axis=1)
    ok = seg[..., 0].astype(bool)
    arrived = ok.all(axis=1)
    time_s = np.where(ok, seg[..., 1], 0.0).sum(axis=1)
    shock_ok = (seg[..., 2] <= budget_g) | ~ok
    safe = arrived & shock_ok.all(axis=1)
    # scale from the simulated segment length to the requested distance (last segment may be partial)
    time_min = time_s / 60.0 * (distance_m / (n_seg * SEGMENT_M))
    return RouteSamples(np.where(safe, time_min, np.inf), arrived, safe, n_seg, n_cross)


def summary(rs: RouteSamples) -> dict:
    finite = rs.time_min[np.isfinite(rs.time_min)]
    return {
        "p_arrive": float(rs.arrived.mean()), "p_safe_delivery": float(rs.safe.mean()),
        "time_min_median": float(np.median(finite)) if finite.size else float("nan"),
        "time_min_p90": float(np.quantile(finite, 0.9)) if finite.size else float("nan"),
        "n_segments": rs.n_segments, "n_crossing_segments": rs.n_crossing_segments,
    }
