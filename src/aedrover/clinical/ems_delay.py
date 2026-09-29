"""Ambulance response-time model and dispatch-time defaults.

Units: every time is in MINUTES. The ambulance response time is measured from the emergency
call to the first ambulance arriving at the scene, the definition used by Naess et al.
(citation key ``naess2024``: "the difference between the time of the incident and the arrival of
the first responding ambulance on the scene", where the incident time is the first call to the
emergency medical communication centre).

Source data (``naess2024``, PLOS ONE 19:e0296308, Table 1; 216,787 acute incidents with an
ambulance response in Central Norway, 2013-2022; urban = neighbourhoods in a densely populated
area of more than 10,000 inhabitants)::

    response time (min)   all (n=216,787)   rural (n=134,116)   urban (n=82,671)
    mean (SD)             15.8 (12.6)       18.3 (14.0)         11.6 (8.5)
    median (IQR)          12.2 (10.4)       14.8 (13.2)         10.0 (5.6)
    90th percentile       29.1              33.3                17.7

Busy-ambulance effect (same paper): each 10 percentage-point increase in the probability that
the nearest ambulance is busy is associated with +0.60 min overall (95% CI 0.58 to 0.62), +0.81
rural, +0.30 urban; the mean additional response time attributable to busy ambulances is 1.61
min overall, 1.76 rural and 1.04 urban.

A lognormal distribution is fitted to (median, p90); its implied mean and IQR are compared with
the reported values in tests/test_survival.py (fit-quality check, not a re-derivation).

Caveat: these are Norwegian data. Nothing here is a claim about any other health system.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

FloatArray = NDArray[np.float64]

Z90 = float(norm.ppf(0.90))  # standard-normal 90th percentile, 1.2816


@dataclass(frozen=True)
class ResponseTimeSpec:
    """Lognormal response-time distribution parameterised by its median and 90th percentile.

    Attributes:
        median_min: median response time [min], > 0.
        p90_min: 90th percentile of the response time [min], > median.
        label: human-readable description of the population.
        citation_key: key in docs/references.json (empty for a user-defined spec).
    """

    median_min: float
    p90_min: float
    label: str = ""
    citation_key: str = ""

    def __post_init__(self) -> None:
        if not (math.isfinite(self.median_min) and math.isfinite(self.p90_min)):
            raise ValueError("median_min and p90_min must be finite")
        if not 0.0 < self.median_min < self.p90_min:
            raise ValueError("need 0 < median_min < p90_min")

    @property
    def mu(self) -> float:
        """Mean of log(response time in min): ln(median)."""
        return math.log(self.median_min)

    @property
    def sigma(self) -> float:
        """Standard deviation of log(response time): ln(p90 / median) / z_0.90."""
        return math.log(self.p90_min / self.median_min) / Z90

    @property
    def mean_min(self) -> float:
        """Mean response time [min] of the fitted lognormal: median * exp(sigma^2 / 2)."""
        return self.median_min * math.exp(0.5 * self.sigma**2)

    @property
    def iqr_min(self) -> float:
        """Interquartile range [min] of the fitted lognormal."""
        z = float(norm.ppf(0.75))
        return self.median_min * (math.exp(z * self.sigma) - math.exp(-z * self.sigma))

    def quantile(self, q: float | FloatArray) -> FloatArray:
        """Response time [min] at cumulative probability ``q`` in (0, 1)."""
        return np.exp(self.mu + self.sigma * norm.ppf(q))

    def quantile_grid(self, n: int) -> FloatArray:
        """Deterministic midpoint-quantile sample of size ``n`` [min], ascending.

        ``quantile((i + 0.5) / n)`` for ``i = 0..n-1``. A noise-free stand-in for Monte Carlo
        draws, used where a smooth, monotone objective is needed (root finding, sensitivity).
        """
        if n < 1:
            raise ValueError("n must be >= 1")
        return self.quantile((np.arange(n) + 0.5) / n)

    def sample(self, n: int, rng: np.random.Generator) -> FloatArray:
        """``n`` i.i.d. response times [min]."""
        return rng.lognormal(self.mu, self.sigma, size=n)

    def with_median(self, median_min: float) -> ResponseTimeSpec:
        """Same shape (p90 / median ratio) rescaled to a new median [min]."""
        ratio = self.p90_min / self.median_min
        return replace(self, median_min=median_min, p90_min=median_min * ratio)


# ---- Naess et al. 2024, Table 1 (citation key: naess2024) -----------------------------------
NAESS_ALL = ResponseTimeSpec(12.2, 29.1, "Central Norway, all incidents", "naess2024")
NAESS_RURAL = ResponseTimeSpec(14.8, 33.3, "Central Norway, rural incidents", "naess2024")
NAESS_URBAN = ResponseTimeSpec(10.0, 17.7, "Central Norway, urban incidents", "naess2024")

RESPONSE_TIME_PRESETS: dict[str, ResponseTimeSpec] = {
    "all": NAESS_ALL,
    "rural": NAESS_RURAL,
    "urban": NAESS_URBAN,
}

# Reported (not fitted) summary statistics from the same table, for validation of the fit [min].
NAESS_REPORTED_MEAN_MIN = {"all": 15.8, "rural": 18.3, "urban": 11.6}
NAESS_REPORTED_IQR_MIN = {"all": 10.4, "rural": 13.2, "urban": 5.6}

# Busy-ambulance effect (naess2024): probability that the nearest ambulance is busy, delay [min]
# per 10 percentage-point increase in that probability, and mean per-incident extra time [min].
BUSY_PROBABILITY = {"all": 0.267, "rural": 0.216, "urban": 0.350}
BUSY_DELAY_MIN_PER_10PP = {"all": 0.60, "rural": 0.81, "urban": 0.30}
BUSY_MEAN_EXTRA_MIN = {"all": 1.61, "rural": 1.76, "urban": 1.04}


def busy_delay_min(delta_p_busy: float, setting: str = "urban") -> float:
    """Extra response time [min] for an increase of ``delta_p_busy`` in P(ambulance busy).

    ``delta_p_busy`` is a probability difference (0.10 = ten percentage points). Linear, using
    the ``naess2024`` within-neighbourhood slope for ``setting`` in {"all", "rural", "urban"}.
    The reference level is the regional average busyness already embedded in the response-time
    distribution, so a positive value models a busier-than-usual system.
    """
    if setting not in BUSY_DELAY_MIN_PER_10PP:
        raise KeyError(f"setting must be one of {sorted(BUSY_DELAY_MIN_PER_10PP)}")
    return BUSY_DELAY_MIN_PER_10PP[setting] * delta_p_busy / 0.10


@dataclass(frozen=True)
class DispatchTimes:
    """Non-travel delays of the alert-and-deliver chain [min]. ALL DEFAULTS ARE ASSUMPTIONS.

    No primary source for these delays was verified for this project, so every default is an
    ASSUMPTION (see docs/CLINICAL_MODEL.md, assumptions table; ranges below are used for the
    tornado sensitivity analysis).

    Attributes:
        collapse_to_call_min: collapse until the emergency call connects (recognition and
            dialling). ASSUMPTION 1.0, range 0.5 to 3.0. Common to every mode.
        call_to_alert_min: call-taker triage plus dispatch latency until the rover/drone is
            alerted. ASSUMPTION 1.5, range 0.5 to 3.0. Not added to ambulances, whose sampled
            response time already runs from the call.
        handoff_min: bystander retrieves the AED from the rover/drone and applies the pads.
            ASSUMPTION 1.0, range 0.5 to 2.5.
        bystander_cpr_delay_min: collapse until bystander CPR starts (dispatcher-assisted);
            ``None`` means no bystander CPR, so CPR starts at ambulance arrival. ASSUMPTION 3.0,
            range 1.0 to 6.0.
        ems_arrival_to_shock_min: ambulance arrival at the scene until its first shock.
            ASSUMPTION 0.0 (conservative for the rover comparison: the crew shocks on arrival),
            range 0.0 to 3.0.
    """

    collapse_to_call_min: float = 1.0
    call_to_alert_min: float = 1.5
    handoff_min: float = 1.0
    bystander_cpr_delay_min: float | None = 3.0
    ems_arrival_to_shock_min: float = 0.0

    def __post_init__(self) -> None:
        for name in ("collapse_to_call_min", "call_to_alert_min", "handoff_min",
                     "ems_arrival_to_shock_min"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0 minutes")
        if self.bystander_cpr_delay_min is not None and self.bystander_cpr_delay_min < 0:
            raise ValueError("bystander_cpr_delay_min must be >= 0 minutes or None")


# Sensitivity ranges [min] (low, high) for the ASSUMPTION defaults above.
DISPATCH_RANGES: dict[str, tuple[float, float]] = {
    "collapse_to_call_min": (0.5, 3.0),
    "call_to_alert_min": (0.5, 3.0),
    "handoff_min": (0.5, 2.5),
    "bystander_cpr_delay_min": (1.0, 6.0),
    "ems_arrival_to_shock_min": (0.0, 3.0),
}


def sample_ambulance_arrival(
    n: int,
    rng: np.random.Generator,
    median_min: float = NAESS_URBAN.median_min,
    p90_min: float = NAESS_URBAN.p90_min,
    busy_increase: float = 0.0,
    busy_setting: str = "urban",
) -> FloatArray:
    """Sample ``n`` ambulance response times [min from call to arrival at the scene].

    Lognormal with the given median and 90th percentile (defaults: Naess urban, 10.0 and 17.7
    min, citation key ``naess2024``), plus a deterministic busy-ambulance delay from
    :func:`busy_delay_min` when ``busy_increase`` (probability difference) is non-zero.

    Args:
        n: number of samples (>= 1).
        rng: numpy random generator (caller controls the seed).
        median_min: median response time [min].
        p90_min: 90th-percentile response time [min].
        busy_increase: increase in P(nearest ambulance busy) vs the regional average, as a
            probability difference in [0, 1].
        busy_setting: which Naess slope to use ("all", "rural", "urban").
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    spec = ResponseTimeSpec(median_min, p90_min)
    return spec.sample(n, rng) + busy_delay_min(busy_increase, busy_setting)
