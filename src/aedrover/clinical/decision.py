"""Compose delay, delivery-mode and survival models into a clinical decision analysis.

Units: all clinical times are MINUTES measured from collapse; distances are metres; speeds are
metres per second; survival is a probability in [0, 1].

Time-to-first-shock decomposition
---------------------------------
Ambulance (sampled response time ``R`` runs from the emergency call to arrival at the scene)::

    t_scene  = collapse_to_call + R (+ busy delay)
    t_defib  = t_scene + ems_arrival_to_shock          (default 0: shock on arrival)

Rover (dispatched in addition to the ambulance)::

    t_rover  = collapse_to_call + call_to_alert + travel + handoff
    travel   = radius * route_factor / speed / 60       (radius = straight-line metres)

Drone (straight-line flight, no route factor)::

    t_drone  = collapse_to_call + call_to_alert + launch + radius / speed / 60 + handoff

Only the first-shock time depends on the mode. CPR time (bystander delay, capped by ambulance
arrival) and ACLS time (ambulance arrival ``t_scene``) are identical for every mode, so a
delivery device changes survival solely through the ``t_defib`` term of the survival model.

Two comparisons are provided and must not be confused:

* ``parallel=True`` (default in :func:`mode_times`): the device is dispatched alongside the
  ambulance, so the first shock is ``min(t_mode, t_ambulance)``. The gain over the ambulance
  alone is the *marginal benefit of adding the device* and is never negative.
* ``parallel=False`` (used by :func:`breakeven_radius`): the device alone versus the ambulance
  alone. The radius where the two expected survivals are equal is the break-even radius.

Defaults for the mode parameters below are ASSUMPTIONS (see docs/CLINICAL_MODEL.md); the
ambulance response-time distribution defaults to the Naess urban statistics (``naess2024``).
Pure functions, no plotting.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq

from aedrover.analysis.stats import bootstrap_ci
from aedrover.clinical.ems_delay import (
    DISPATCH_RANGES,
    NAESS_RURAL,
    NAESS_URBAN,
    DispatchTimes,
    ResponseTimeSpec,
    busy_delay_min,
)
from aedrover.clinical.survival import SurvivalModel

FloatArray = NDArray[np.float64]

# Validation targets from the Swedish drone-AED study (citation key: schierbeck2023): a drone AED
# arrived before the ambulance in 37 of 55 cases (67%), median time benefit 3 min 14 s. These are
# reported for comparison only; they do not feed any computation.
SCHIERBECK_FRACTION_DRONE_FIRST = 37.0 / 55.0
SCHIERBECK_MEDIAN_BENEFIT_MIN = 3.0 + 14.0 / 60.0

MODES = ("ambulance", "rover", "drone")


@dataclass(frozen=True)
class ScenarioParams:
    """All inputs of one decision scenario. Defaults marked ASSUMPTION are unsourced.

    Attributes:
        radius_m: straight-line distance from the device to the patient [m]. ASSUMPTION 1000.
        rover_speed_mps: mean sidewalk speed [m/s]. ASSUMPTION 2.0, range 1.0 to 3.0.
        route_factor: path length / straight-line distance (>= 1). ASSUMPTION 1.3, range 1.1
            to 1.6.
        drone_speed_mps: mean cruise speed [m/s]. ASSUMPTION 15.0, range 10 to 25.
        drone_launch_min: drone preparation and take-off delay [min]. ASSUMPTION 0.5.
        ems_median_min: median ambulance response time from the call [min]; default is the
            Naess urban median 10.0 (``naess2024``).
        ems_p90_over_median: ratio of p90 to median response time (shape); default is the
            Naess urban ratio 17.7 / 10.0 (``naess2024``).
        busy_increase: increase in P(nearest ambulance busy), probability difference. Default 0.
        busy_setting: Naess busy-slope set, "all", "rural" or "urban".
        dispatch: non-travel delays (:class:`DispatchTimes`, all ASSUMPTION defaults).
    """

    radius_m: float = 1000.0
    rover_speed_mps: float = 2.0
    route_factor: float = 1.3
    drone_speed_mps: float = 15.0
    drone_launch_min: float = 0.5
    ems_median_min: float = NAESS_URBAN.median_min
    ems_p90_over_median: float = NAESS_URBAN.p90_min / NAESS_URBAN.median_min
    busy_increase: float = 0.0
    busy_setting: str = "urban"
    dispatch: DispatchTimes = field(default_factory=DispatchTimes)

    def __post_init__(self) -> None:
        if self.radius_m < 0:
            raise ValueError("radius_m must be >= 0")
        if self.rover_speed_mps <= 0 or self.drone_speed_mps <= 0:
            raise ValueError("speeds must be > 0 m/s")
        if self.route_factor < 1.0:
            raise ValueError("route_factor must be >= 1")
        if self.drone_launch_min < 0:
            raise ValueError("drone_launch_min must be >= 0")
        if self.ems_median_min <= 0 or self.ems_p90_over_median <= 1.0:
            raise ValueError("need ems_median_min > 0 and ems_p90_over_median > 1")

    @property
    def ems_spec(self) -> ResponseTimeSpec:
        """Ambulance response-time distribution implied by the median and the p90/median ratio."""
        return ResponseTimeSpec(
            self.ems_median_min, self.ems_median_min * self.ems_p90_over_median
        )


@dataclass(frozen=True)
class ModeSamples:
    """Per-scenario intervention times [min from collapse] for one delivery mode.

    All three arrays share one shape; index ``i`` refers to the same simulated arrest in every
    mode, which makes mode comparisons paired.

    Attributes:
        t_defib: time to first shock.
        t_cpr: time to CPR.
        t_acls: time to ACLS (ambulance arrival).
    """

    t_defib: FloatArray
    t_cpr: FloatArray
    t_acls: FloatArray

    def __post_init__(self) -> None:
        if not (self.t_defib.shape == self.t_cpr.shape == self.t_acls.shape):
            raise ValueError("t_defib, t_cpr and t_acls must share one shape")

    def survival(self, model: SurvivalModel) -> FloatArray:
        """Survival probability of every scenario under ``model``."""
        return np.asarray(model(self.t_cpr, self.t_defib, self.t_acls), dtype=float)


# ---------------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------------
def _scene_times(params: ScenarioParams, n: int, rng: np.random.Generator | None) -> FloatArray:
    """Ambulance arrival at the scene [min from collapse]; quantile grid if ``rng`` is None."""
    spec = params.ems_spec
    resp = spec.quantile_grid(n) if rng is None else spec.sample(n, rng)
    shift = params.dispatch.collapse_to_call_min + busy_delay_min(
        params.busy_increase, params.busy_setting
    )
    return resp + shift


def _cpr_times(t_scene: FloatArray, delay_min: float | None) -> FloatArray:
    """CPR start: bystander delay capped by ambulance arrival (or arrival if no bystander CPR)."""
    return t_scene.copy() if delay_min is None else np.minimum(delay_min, t_scene)


def _rover_first_shock(
    radius_m: ArrayLike,
    speed_mps: ArrayLike,
    route_factor: float,
    collapse_to_call_min: float,
    call_to_alert_min: ArrayLike,
    handoff_min: float,
) -> FloatArray:
    """Rover-alone first-shock time [min from collapse]; broadcasts over radius/speed/alert."""
    travel = np.asarray(radius_m, dtype=float) * route_factor / np.asarray(speed_mps) / 60.0
    return collapse_to_call_min + np.asarray(call_to_alert_min, dtype=float) + travel + handoff_min


def mode_times(
    n: int,
    rng: np.random.Generator | None,
    params: ScenarioParams,
    *,
    modes: tuple[str, ...] = MODES,
    parallel: bool = True,
    rover_travel_min: ArrayLike | None = None,
) -> dict[str, ModeSamples]:
    """Build paired time-to-intervention samples for each delivery mode.

    Args:
        n: number of simulated arrests.
        rng: random generator for the ambulance response times; ``None`` uses the deterministic
            midpoint-quantile grid (noise-free, for sensitivity and root finding).
        params: scenario parameters.
        modes: subset of ``("ambulance", "rover", "drone")``.
        parallel: if True the device is dispatched together with the ambulance and the first
            shock is ``min(t_device, t_ambulance)``; if False the device stands alone.
        rover_travel_min: optional simulated rover travel times [min] (scalar or length ``n``)
            replacing the deterministic ``radius * route_factor / speed`` formula, so measured
            travel times from the MuJoCo simulation plug straight in.

    Returns:
        ``{mode: ModeSamples}``. CPR and ACLS times are shared by all modes.
    """
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes {sorted(unknown)}; choose from {MODES}")
    d = params.dispatch
    t_scene = _scene_times(params, n, rng)
    t_cpr = _cpr_times(t_scene, d.bystander_cpr_delay_min)
    t_amb = t_scene + d.ems_arrival_to_shock_min

    out: dict[str, ModeSamples] = {}
    for mode in modes:
        if mode == "ambulance":
            t_defib = t_amb.copy()
        else:
            if mode == "rover":
                if rover_travel_min is None:
                    alone = _rover_first_shock(
                        params.radius_m, params.rover_speed_mps, params.route_factor,
                        d.collapse_to_call_min, d.call_to_alert_min, d.handoff_min,
                    )
                else:
                    travel = np.asarray(rover_travel_min, dtype=float)
                    if np.any(travel < 0):
                        raise ValueError("rover_travel_min must be >= 0")
                    alone = (d.collapse_to_call_min + d.call_to_alert_min + travel
                             + d.handoff_min)
            else:  # drone
                flight = params.radius_m / params.drone_speed_mps / 60.0
                alone = (d.collapse_to_call_min + d.call_to_alert_min + params.drone_launch_min
                         + flight + d.handoff_min)
            alone = np.broadcast_to(np.asarray(alone, dtype=float), t_amb.shape)
            t_defib = np.minimum(alone, t_amb) if parallel else alone.copy()
        out[mode] = ModeSamples(t_defib=np.asarray(t_defib, dtype=float), t_cpr=t_cpr.copy(),
                                t_acls=t_scene.copy())
    return out


# ---------------------------------------------------------------------------------------------
# expected survival and gain with bootstrap CI
# ---------------------------------------------------------------------------------------------
def evaluate_modes(
    samples: dict[str, ModeSamples],
    model: SurvivalModel,
    *,
    baseline: str = "ambulance",
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
    method: str = "percentile",
) -> pd.DataFrame:
    """Expected survival of every mode with paired-bootstrap CIs and gains over ``baseline``.

    Scenarios are resampled jointly across modes (paired bootstrap), so the CI of a gain
    reflects the per-arrest difference. Use random samples from :func:`mode_times` (a quantile
    grid has no sampling error to bootstrap).

    Returns one row per mode with columns: ``mode, model, n, mean_survival, survival_ci_low,
    survival_ci_high, abs_gain, abs_gain_ci_low, abs_gain_ci_high, rel_gain, rel_gain_ci_low,
    rel_gain_ci_high, p_faster, median_saving_when_faster_min``. ``abs_gain`` is a probability
    difference vs the baseline, ``rel_gain = mean / baseline_mean - 1`` (NaN if the baseline mean
    is 0), ``p_faster`` the fraction of arrests where the mode shocks strictly earlier than the
    baseline, and the last column the median time saved [min] over those arrests.
    """
    if baseline not in samples:
        raise KeyError(f"baseline {baseline!r} not in samples {sorted(samples)}")
    surv = {k: v.survival(model) for k, v in samples.items()}
    base_s = surv[baseline]
    base_t = samples[baseline].t_defib
    kw = {"confidence": confidence, "n_resamples": n_resamples, "seed": seed, "method": method}

    def _diff(x: FloatArray, y: FloatArray, axis: int = -1) -> FloatArray:
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    def _rel(x: FloatArray, y: FloatArray, axis: int = -1) -> FloatArray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.mean(x, axis=axis) / np.mean(y, axis=axis) - 1.0

    rows = []
    for mode, s in surv.items():
        m_ci = bootstrap_ci(s, **kw)
        if mode == baseline:
            g = (0.0, 0.0, 0.0)
            r = (0.0, 0.0, 0.0)
            p_faster, saving = 0.0, math.nan
        else:
            gi = bootstrap_ci(s, base_s, statistic=_diff, paired=True, **kw)
            g = (gi.estimate, gi.low, gi.high)
            if float(np.mean(base_s)) > 0.0:
                ri = bootstrap_ci(s, base_s, statistic=_rel, paired=True, **kw)
                r = (ri.estimate, ri.low, ri.high)
            else:
                r = (math.nan, math.nan, math.nan)
            faster = samples[mode].t_defib < base_t
            p_faster = float(np.mean(faster))
            saving = math.nan
            if faster.any():
                saving = float(np.median((base_t - samples[mode].t_defib)[faster]))
        rows.append({
            "mode": mode, "model": model.name, "n": int(s.size),
            "mean_survival": m_ci.estimate, "survival_ci_low": m_ci.low,
            "survival_ci_high": m_ci.high,
            "abs_gain": g[0], "abs_gain_ci_low": g[1], "abs_gain_ci_high": g[2],
            "rel_gain": r[0], "rel_gain_ci_low": r[1], "rel_gain_ci_high": r[2],
            "p_faster": p_faster, "median_saving_when_faster_min": saving,
        })
    return pd.DataFrame(rows)


def compare_modes(
    model: SurvivalModel,
    params: ScenarioParams | None = None,
    *,
    n: int = 4000,
    seed: int = 0,
    parallel: bool = True,
    modes: tuple[str, ...] = MODES,
    rover_travel_min: ArrayLike | None = None,
    n_resamples: int = 2000,
    method: str = "percentile",
) -> pd.DataFrame:
    """Sample ``n`` arrests with ``seed`` and return :func:`evaluate_modes` for the scenario."""
    p = params if params is not None else ScenarioParams()
    rng = np.random.default_rng(seed)
    samples = mode_times(n, rng, p, modes=modes, parallel=parallel,
                         rover_travel_min=rover_travel_min)
    return evaluate_modes(samples, model, seed=seed, n_resamples=n_resamples, method=method)


def expected_gain(
    model: SurvivalModel,
    params: ScenarioParams,
    *,
    mode: str = "rover",
    metric: str = "parallel",
    n_grid: int = 2000,
) -> float:
    """Noise-free expected survival gain of ``mode`` over the ambulance (probability difference).

    Uses the deterministic quantile grid of the ambulance response time. ``metric="parallel"``
    is the marginal benefit of adding the device to the EMS response; ``metric="alone"``
    compares the device alone with the ambulance alone.
    """
    if metric not in ("parallel", "alone"):
        raise ValueError("metric must be 'parallel' or 'alone'")
    if mode == "ambulance":
        return 0.0
    s = mode_times(n_grid, None, params, modes=("ambulance", mode), parallel=metric == "parallel")
    return float(np.mean(s[mode].survival(model)) - np.mean(s["ambulance"].survival(model)))


# ---------------------------------------------------------------------------------------------
# break-even radius and surface
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Breakeven:
    """Result of :func:`breakeven_radius`.

    Attributes:
        radius_m: straight-line break-even radius [m]; ``0.0`` if the rover is never better,
            ``inf`` if it is still at least as good at ``r_max_m``.
        flag: "crossing" (a finite root was found), "rover_never_better" (even at zero distance
            the rover's expected survival is below the ambulance's) or "rover_always_better"
            (no crossing up to ``r_max_m``).
        ambulance_survival: expected survival of the ambulance alone.
        rover_survival_at_zero: expected survival of the rover alone at zero distance.
    """

    radius_m: float
    flag: str
    ambulance_survival: float
    rover_survival_at_zero: float


def breakeven_radius(
    speed_mps: float,
    route_factor: float,
    dispatch_min: float,
    handoff_min: float,
    ems_median_min: float,
    model: SurvivalModel,
    *,
    ems_p90_over_median: float = NAESS_URBAN.p90_min / NAESS_URBAN.median_min,
    collapse_to_call_min: float = DispatchTimes().collapse_to_call_min,
    bystander_cpr_delay_min: float | None = DispatchTimes().bystander_cpr_delay_min,
    ems_arrival_to_shock_min: float = DispatchTimes().ems_arrival_to_shock_min,
    busy_increase: float = 0.0,
    r_max_m: float = 50_000.0,
    n_grid: int = 2000,
) -> Breakeven:
    """Straight-line radius [m] where the rover ALONE matches the ambulance ALONE in survival.

    Solves ``E[S(rover)] = E[S(ambulance)]`` with :func:`scipy.optimize.brentq`. Expected
    survivals are averaged over a deterministic quantile grid of the ambulance response time,
    so the objective is smooth and monotone in the radius (rover survival falls as the radius
    grows). The rover uses ``collapse_to_call + dispatch + radius * route_factor / speed / 60 +
    handoff``.

    Args:
        speed_mps: rover speed [m/s], > 0.
        route_factor: path / straight-line distance, >= 1.
        dispatch_min: call-taker plus dispatch latency until the rover is alerted [min]
            (``DispatchTimes.call_to_alert_min``).
        handoff_min: bystander retrieves the AED and applies pads [min].
        ems_median_min: median ambulance response time from the call [min].
        model: survival model.
        ems_p90_over_median: shape of the response-time distribution (p90 / median).
        collapse_to_call_min, bystander_cpr_delay_min, ems_arrival_to_shock_min, busy_increase:
            as in :class:`DispatchTimes` / :class:`ScenarioParams`.
        r_max_m: search upper bound [m].
        n_grid: quantile-grid size.

    Returns:
        :class:`Breakeven` with the radius and a flag describing the no-crossing cases.
    """
    d = DispatchTimes(
        collapse_to_call_min=collapse_to_call_min, call_to_alert_min=dispatch_min,
        handoff_min=handoff_min, bystander_cpr_delay_min=bystander_cpr_delay_min,
        ems_arrival_to_shock_min=ems_arrival_to_shock_min,
    )
    params = ScenarioParams(
        rover_speed_mps=speed_mps, route_factor=route_factor, ems_median_min=ems_median_min,
        ems_p90_over_median=ems_p90_over_median, busy_increase=busy_increase, dispatch=d,
    )
    t_scene = _scene_times(params, n_grid, None)
    t_cpr = _cpr_times(t_scene, d.bystander_cpr_delay_min)
    s_amb = float(np.mean(model(t_cpr, t_scene + d.ems_arrival_to_shock_min, t_scene)))

    def rover_survival(radius: float) -> float:
        t = _rover_first_shock(radius, speed_mps, route_factor, d.collapse_to_call_min,
                               d.call_to_alert_min, d.handoff_min)
        return float(np.mean(model(t_cpr, t, t_scene)))

    f0 = rover_survival(0.0) - s_amb
    if f0 < 0.0:
        return Breakeven(0.0, "rover_never_better", s_amb, f0 + s_amb)
    if rover_survival(r_max_m) - s_amb >= 0.0:
        return Breakeven(math.inf, "rover_always_better", s_amb, f0 + s_amb)
    root = brentq(lambda r: rover_survival(r) - s_amb, 0.0, r_max_m, xtol=1e-6, rtol=1e-12)
    return Breakeven(float(root), "crossing", s_amb, f0 + s_amb)


def breakeven_surface(
    model: SurvivalModel,
    radii_m: ArrayLike,
    speeds_mps: ArrayLike,
    dispatch_min_values: ArrayLike,
    *,
    base: ScenarioParams | None = None,
    n_grid: int = 1000,
) -> pd.DataFrame:
    """Expected survival of ambulance and rover over a radius x speed x dispatch-latency grid.

    Args:
        model: survival model.
        radii_m: straight-line radii [m].
        speeds_mps: rover speeds [m/s], > 0.
        dispatch_min_values: rover alert latencies [min] (``call_to_alert_min``).
        base: scenario supplying every other parameter (default :class:`ScenarioParams`).
        n_grid: quantile-grid size for the ambulance response time.

    Returns:
        Long-format DataFrame, one row per grid point: ``model, radius_m, speed_mps,
        dispatch_min, s_ambulance, s_rover_alone, s_rover_parallel, gain_alone, gain_parallel,
        rover_better``. ``gain_alone = s_rover_alone - s_ambulance`` (the break-even surface is
        its zero level set); ``gain_parallel = s_rover_parallel - s_ambulance >= 0``;
        ``rover_better = gain_alone > 0``.
    """
    p = base if base is not None else ScenarioParams()
    d = p.dispatch
    radii = np.asarray(radii_m, dtype=float)
    speeds = np.asarray(speeds_mps, dtype=float)
    alerts = np.asarray(dispatch_min_values, dtype=float)
    if np.any(speeds <= 0):
        raise ValueError("speeds_mps must be > 0")

    t_scene = _scene_times(p, n_grid, None)
    t_cpr = _cpr_times(t_scene, d.bystander_cpr_delay_min)
    t_amb = t_scene + d.ems_arrival_to_shock_min
    s_amb = float(np.mean(model(t_cpr, t_amb, t_scene)))

    rover = _rover_first_shock(
        radii[:, None, None], speeds[None, :, None], p.route_factor, d.collapse_to_call_min,
        alerts[None, None, :], d.handoff_min,
    )  # (R, V, D)
    rover_n = rover[..., None]  # (R, V, D, 1) broadcasts against the (N,) ambulance arrays
    s_alone = np.mean(model(t_cpr, rover_n, t_scene), axis=-1)
    s_par = np.mean(model(t_cpr, np.minimum(rover_n, t_amb), t_scene), axis=-1)

    rr, vv, dd = np.meshgrid(radii, speeds, alerts, indexing="ij")
    return pd.DataFrame({
        "model": model.name,
        "radius_m": rr.ravel(), "speed_mps": vv.ravel(), "dispatch_min": dd.ravel(),
        "s_ambulance": s_amb,
        "s_rover_alone": np.asarray(s_alone).ravel(),
        "s_rover_parallel": np.asarray(s_par).ravel(),
        "gain_alone": np.asarray(s_alone).ravel() - s_amb,
        "gain_parallel": np.asarray(s_par).ravel() - s_amb,
        "rover_better": (np.asarray(s_alone).ravel() - s_amb) > 0.0,
    })


# ---------------------------------------------------------------------------------------------
# one-at-a-time sensitivity
# ---------------------------------------------------------------------------------------------
# Sensitivity ranges (low, high). ems_median_min spans the sourced Naess urban and rural medians
# (naess2024); the dispatch ranges come from ems_delay.DISPATCH_RANGES; the rest are ASSUMPTIONS.
DEFAULT_TORNADO_RANGES: dict[str, tuple[float, float]] = {
    "radius_m": (250.0, 2000.0),
    "rover_speed_mps": (1.0, 3.0),
    "route_factor": (1.1, 1.6),
    "ems_median_min": (NAESS_URBAN.median_min, NAESS_RURAL.median_min),
    "busy_increase": (0.0, 0.20),
    **{f"dispatch.{k}": v for k, v in DISPATCH_RANGES.items()},
}


def _with_param(base: ScenarioParams, name: str, value: float) -> ScenarioParams:
    """Copy of ``base`` with one (possibly nested ``dispatch.``) parameter replaced."""
    if name.startswith("dispatch."):
        field_name = name.split(".", 1)[1]
        if not hasattr(base.dispatch, field_name):
            raise ValueError(f"unknown dispatch parameter {field_name!r}")
        return replace(base, dispatch=replace(base.dispatch, **{field_name: value}))
    if not hasattr(base, name) or name == "dispatch":
        raise ValueError(f"unknown scenario parameter {name!r}")
    return replace(base, **{name: value})


def tornado(
    model: SurvivalModel,
    base: ScenarioParams | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    *,
    mode: str = "rover",
    metric: str = "parallel",
    n_grid: int = 2000,
) -> pd.DataFrame:
    """One-at-a-time sensitivity of the expected survival gain of ``mode`` over the ambulance.

    Each parameter in ``ranges`` is moved to its low and high value with all others at ``base``.
    The gain is :func:`expected_gain` (absolute survival probability, noise-free).

    Args:
        model: survival model.
        base: base scenario (default :class:`ScenarioParams`).
        ranges: ``{name: (low, high)}``; nested delays use the ``dispatch.`` prefix, e.g.
            ``"dispatch.handoff_min"``. Default :data:`DEFAULT_TORNADO_RANGES`.
        mode: "rover" or "drone".
        metric: "parallel" (marginal benefit of adding the device) or "alone".
        n_grid: quantile-grid size.

    Returns:
        DataFrame sorted by decreasing ``swing``: ``parameter, low, high, gain_at_low,
        gain_at_high, gain_base, swing`` where ``swing = |gain_at_high - gain_at_low|``.
    """
    b = base if base is not None else ScenarioParams()
    rng_map = DEFAULT_TORNADO_RANGES if ranges is None else ranges
    g0 = expected_gain(model, b, mode=mode, metric=metric, n_grid=n_grid)
    rows = []
    for name, (lo, hi) in rng_map.items():
        g_lo = expected_gain(model, _with_param(b, name, lo), mode=mode, metric=metric,
                             n_grid=n_grid)
        g_hi = expected_gain(model, _with_param(b, name, hi), mode=mode, metric=metric,
                             n_grid=n_grid)
        rows.append({"parameter": name, "low": lo, "high": hi, "gain_at_low": g_lo,
                     "gain_at_high": g_hi, "gain_base": g0, "swing": abs(g_hi - g_lo)})
    cols = ["parameter", "low", "high", "gain_at_low", "gain_at_high", "gain_base", "swing"]
    df = pd.DataFrame(rows, columns=cols)
    return df.sort_values("swing", ascending=False, kind="stable").reset_index(drop=True)
