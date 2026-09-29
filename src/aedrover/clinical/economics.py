"""Dimensionless deployment economics for the course model (no currency anywhere).

Every quantity is either a pure ratio or a count of hours/years, so the model needs no unit of
account. Cost components are expressed relative to one common reference unit (for example the
annualised value of the labour the system reclaims); only their ratios enter the results.

Definitions (course model)
--------------------------
``kappa``            annualised operating burden / annualised value of reclaimed labour hours.
                     Course target: ``kappa <= 0.25`` (:data:`KAPPA_TARGET`).
``capex_parity_factor``  up-front capital / annualised value of reclaimed labour, in years.
``payback_months``   ``capex_parity_factor / (1 - kappa) * 12``. The net annual benefit is
                     ``(1 - kappa)`` reference units per year, so payback is finite only when
                     ``kappa < 1``; otherwise the system never pays back and ``inf`` is returned.
``delta_fte``        ``annual_reclaimed_task_hours / 2080`` (full-time-equivalent staff freed).
                     2080 = 40 h/week x 52 weeks is the course convention, not a measured value.
``qaly_gain_per_encounter``  ``delta_survival * life_expectancy_years * quality_of_life``
                     (dimensionless life-years, undiscounted).

Every default in :class:`EconomicsInputs` is an ASSUMPTION (placeholder values for the course
template, none sourced from the literature); see docs/CLINICAL_MODEL.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

KAPPA_TARGET = 0.25  # course-specified target for kappa (operating burden / labour value)
HOURS_PER_FTE = 2080.0  # course convention: 40 h/week x 52 weeks


def _scalar_or_array(x: FloatArray) -> float | FloatArray:
    return float(x) if x.ndim == 0 else x


def kappa(annual_operating_burden: ArrayLike, annual_labour_value: ArrayLike) -> float | FloatArray:
    """Operating-burden ratio ``kappa = burden / labour value`` (dimensionless).

    Both arguments must be expressed in the same (arbitrary) reference unit per year. The
    labour value must be strictly positive; the burden must be non-negative.
    """
    burden = np.asarray(annual_operating_burden, dtype=float)
    value = np.asarray(annual_labour_value, dtype=float)
    if np.any(burden < 0):
        raise ValueError("annual_operating_burden must be >= 0")
    if np.any(value <= 0):
        raise ValueError("annual_labour_value must be > 0")
    return _scalar_or_array(burden / value)


def kappa_from_ratios(
    energy: ArrayLike,
    maintenance: ArrayLike,
    infrastructure: ArrayLike,
    labour_value: ArrayLike = 1.0,
) -> float | FloatArray:
    """Derive kappa from operating-cost components given in one common reference unit.

    ``kappa = (energy + maintenance + infrastructure) / labour_value``. With the default
    ``labour_value = 1`` the reference unit *is* the annualised value of reclaimed labour and
    the three components are already fractions of it.
    """
    burden = (
        np.asarray(energy, dtype=float)
        + np.asarray(maintenance, dtype=float)
        + np.asarray(infrastructure, dtype=float)
    )
    return kappa(burden, labour_value)


def payback_months(
    kappa_value: ArrayLike, capex_parity_factor: ArrayLike, strict: bool = False
) -> float | FloatArray:
    """Payback period in months: ``capex_parity_factor / (1 - kappa) * 12``.

    Args:
        kappa_value: operating-burden ratio, >= 0.
        capex_parity_factor: capital / annual reclaimed-labour value [years], >= 0.
        strict: if True, raise ``ValueError`` when ``kappa >= 1`` instead of returning ``inf``.

    Returns:
        Months to recover the capital. ``math.inf`` where ``kappa >= 1`` (operating burden eats
        the whole benefit, so the system never pays back) and ``strict`` is False.
    """
    k = np.asarray(kappa_value, dtype=float)
    c = np.asarray(capex_parity_factor, dtype=float)
    if np.any(k < 0):
        raise ValueError("kappa must be >= 0")
    if np.any(c < 0):
        raise ValueError("capex_parity_factor must be >= 0")
    never = k >= 1.0
    if strict and np.any(never):
        raise ValueError("kappa >= 1: operating burden consumes the entire benefit; no payback")
    with np.errstate(divide="ignore", invalid="ignore"):
        months = np.where(never, np.inf, c / (1.0 - np.where(never, 0.0, k)) * 12.0)
    return _scalar_or_array(np.asarray(months, dtype=float))


def delta_fte(
    annual_reclaimed_task_hours: ArrayLike, hours_per_fte: float = HOURS_PER_FTE
) -> float | FloatArray:
    """Full-time-equivalent staff freed: ``hours / 2080`` (course convention, dimensionless)."""
    hours = np.asarray(annual_reclaimed_task_hours, dtype=float)
    if np.any(hours < 0):
        raise ValueError("annual_reclaimed_task_hours must be >= 0")
    if hours_per_fte <= 0:
        raise ValueError("hours_per_fte must be > 0")
    return _scalar_or_array(hours / hours_per_fte)


def qaly_gain_per_encounter(
    delta_survival: ArrayLike, life_expectancy_years: ArrayLike, quality_of_life: ArrayLike
) -> float | FloatArray:
    """Expected quality-adjusted life-years gained per OHCA encounter (dimensionless).

    ``delta_survival * life_expectancy_years * quality_of_life``, undiscounted.

    Args:
        delta_survival: change in survival probability, in [-1, 1].
        life_expectancy_years: remaining life expectancy of a survivor [years], >= 0.
        quality_of_life: utility weight in [0, 1] (1 = full health).
    """
    ds = np.asarray(delta_survival, dtype=float)
    le = np.asarray(life_expectancy_years, dtype=float)
    q = np.asarray(quality_of_life, dtype=float)
    if np.any(np.abs(ds) > 1.0):
        raise ValueError("delta_survival must be in [-1, 1]")
    if np.any(le < 0):
        raise ValueError("life_expectancy_years must be >= 0")
    if np.any((q < 0) | (q > 1)):
        raise ValueError("quality_of_life must be in [0, 1]")
    return _scalar_or_array(ds * le * q)


@dataclass(frozen=True)
class EconomicsInputs:
    """Dimensionless economics inputs. ALL DEFAULTS ARE ASSUMPTIONS (placeholders).

    Cost components are annual quantities expressed as a fraction of the annualised value of the
    labour hours the system reclaims (that value is the reference unit, equal to 1).

    Attributes:
        energy_ratio: annual energy burden / labour value. ASSUMPTION 0.04, range 0.02-0.08.
        maintenance_ratio: annual maintenance burden / labour value. ASSUMPTION 0.12, range
            0.05-0.25.
        infrastructure_ratio: annual infrastructure (docking, connectivity) burden / labour
            value. ASSUMPTION 0.06, range 0.02-0.15.
        capex_parity_factor: up-front capital / annual labour value [years]. ASSUMPTION 0.60,
            range 0.3-1.5.
        annual_reclaimed_task_hours: task hours per year handed back to staff [h].
            ASSUMPTION 1040 (half of one FTE), range 500-2080.
    """

    energy_ratio: float = 0.04
    maintenance_ratio: float = 0.12
    infrastructure_ratio: float = 0.06
    capex_parity_factor: float = 0.60
    annual_reclaimed_task_hours: float = 1040.0

    @property
    def kappa(self) -> float:
        """Operating-burden ratio implied by the three cost components."""
        return float(
            kappa_from_ratios(
                self.energy_ratio, self.maintenance_ratio, self.infrastructure_ratio, 1.0
            )
        )

    @property
    def meets_target(self) -> bool:
        """True when ``kappa <= KAPPA_TARGET`` (0.25)."""
        return self.kappa <= KAPPA_TARGET

    @property
    def payback_months(self) -> float:
        """Payback [months]; ``inf`` when ``kappa >= 1``."""
        return float(payback_months(self.kappa, self.capex_parity_factor))

    @property
    def delta_fte(self) -> float:
        """Staff freed [FTE]."""
        return float(delta_fte(self.annual_reclaimed_task_hours))

    def summary(self) -> dict[str, float | bool]:
        """Headline dimensionless outputs as a flat record."""
        return {
            "kappa": self.kappa,
            "meets_kappa_target": self.meets_target,
            "payback_months": self.payback_months,
            "delta_fte": self.delta_fte,
        }
