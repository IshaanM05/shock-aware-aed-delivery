"""Survival-versus-delay models for out-of-hospital cardiac arrest (OHCA).

All clinical times in this module are in MINUTES measured from the moment of collapse, and
every model returns a survival PROBABILITY in [0, 1] (not a percentage).

Models
------
``larsen1993``
    The linear graphic model of Larsen et al. (citation key ``larsen1993``, Ann Emerg Med
    22:1652-1658). Published equation (abstract, PMID 8214853)::

        survival = 67% - 2.3% * t_cpr - 1.1% * t_defib - 2.1% * t_acls

    where each ``t`` is the interval (minutes) from collapse to CPR, to the first defibrillatory
    shock and to the start of advanced cardiac life support (ACLS). The 67% intercept is the
    survival if all three interventions happened at the instant of collapse; with no treatment
    the decline is the sum of the three slopes, 5.5% per minute. The fitting cohort was 1,667
    King County (Washington) patients with underlying heart disease, ventricular fibrillation
    and collapse before EMS arrival, i.e. a HIGH-survival-likelihood subgroup, so absolute
    levels are optimistic for an all-comers OHCA population; the model is used here for the
    *difference* between delivery modes.

    Correction note: an earlier student draft used 0.046 per minute for defibrillation. The
    published slope is 0.011 per minute; 0.046 is not a coefficient of this model.

``rule_of_thumb`` (+ ``_low`` / ``_high``)
    A widely quoted guideline statement, NOT a fitted model: without CPR, survival after
    defibrillation falls by approximately 7% to 10% per minute (2025 Korean CPR guidelines,
    Part 4 adult ALS, citation key ``kim2026``; the same guideline gives 3% to 4% per minute
    when bystander CPR is performed). Implemented as a linear decline from an initial survival,
    read as PERCENTAGE POINTS per minute. Only the defibrillation time is used; CPR and ACLS
    times are ignored. Used only as a model-form sensitivity check.

``valenzuela1997`` (deliberately NOT implemented)
    Valenzuela et al. (citation key ``valenzuela1997``, Circulation 96:3308-3313) fit a logistic
    regression with collapse-to-CPR, collapse-to-defibrillation and interaction terms. The
    PubMed abstract (PMID 9396421) reports the predictors but NOT the coefficients, and the full
    text is behind a paywall with no open-access copy (Unpaywall: closed; publisher returns HTTP
    403 to scripted access). Coefficients are never cited from memory in this repository, so the
    model is omitted. It remains in docs/references.json as context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
ArrayOrFloat = float | FloatArray

# ---- Larsen et al. 1993 coefficients (citation key: larsen1993) ---------------------------
LARSEN_INTERCEPT = 0.67  # survival if CPR, defibrillation and ACLS occur at collapse
LARSEN_SLOPE_CPR = 0.023  # survival lost per minute of delay to CPR
LARSEN_SLOPE_DEFIB = 0.011  # survival lost per minute of delay to first shock
LARSEN_SLOPE_ACLS = 0.021  # survival lost per minute of delay to ACLS

# ---- Rule of thumb (citation key: kim2026) --------------------------------------------------
RULE_RATE_LOW = 0.07  # lower end of the quoted 7% to 10% per minute range
RULE_RATE_HIGH = 0.10  # upper end of the quoted range
# ASSUMPTION: the midpoint 8.5% per minute is a choice; the guideline gives only the range.
RULE_RATE_MID = 0.085
# ASSUMPTION: the guideline gives no survival at t = 0. 0.67 (the Larsen intercept) is used so
# that the two model families share a starting level and differ only in slope and structure.
RULE_INITIAL_SURVIVAL = LARSEN_INTERCEPT


def _prepare(
    t_cpr: ArrayLike, t_defib: ArrayLike, t_acls: ArrayLike | None
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Broadcast the three delay arrays to a common shape and validate them.

    ``t_acls=None`` means ACLS is assumed to start together with defibrillation
    (``t_acls = t_defib``). Times must be >= 0 minutes; ``inf`` is allowed (a mode that never
    arrives) and yields zero survival; NaN is rejected.
    """
    a = np.asarray(t_cpr, dtype=float)
    b = np.asarray(t_defib, dtype=float)
    c = b if t_acls is None else np.asarray(t_acls, dtype=float)
    for name, arr in (("t_cpr", a), ("t_defib", b), ("t_acls", c)):
        if np.any(np.isnan(arr)):
            raise ValueError(f"{name} contains NaN")
        if np.any(arr < 0):
            raise ValueError(f"{name} must be >= 0 minutes")
    a, b, c = np.broadcast_arrays(a, b, c)
    return a, b, c


def _finalise(x: FloatArray) -> ArrayOrFloat:
    """Return a python float for 0-d results, an array otherwise."""
    return float(x) if x.ndim == 0 else x


@dataclass(frozen=True)
class SurvivalModel(ABC):
    """Base class: survival probability as a function of collapse-to-intervention delays.

    Attributes:
        name: registry key of the model.
        citation_key: key in docs/references.json of the primary source.
        notes: one-paragraph description of scope and caveats.
    """

    name: str
    citation_key: str
    notes: str = ""

    @abstractmethod
    def __call__(
        self, t_cpr: ArrayLike, t_defib: ArrayLike, t_acls: ArrayLike | None = None
    ) -> ArrayOrFloat:
        """Survival probability in [0, 1].

        Args:
            t_cpr: minutes from collapse to CPR.
            t_defib: minutes from collapse to first defibrillatory shock.
            t_acls: minutes from collapse to ACLS; ``None`` means ``t_defib``.

        Inputs broadcast against each other (numpy rules).
        """


@dataclass(frozen=True)
class LarsenModel(SurvivalModel):
    """Larsen et al. 1993 linear model (citation key ``larsen1993``).

    ``survival = intercept - c_cpr * t_cpr - c_defib * t_defib - c_acls * t_acls`` clipped to
    ``[0, intercept]``. Slopes are survival probability lost per minute.
    """

    name: str = "larsen1993"
    citation_key: str = "larsen1993"
    notes: str = (
        "Linear multiple-regression model, 1,667 King County witnessed-VF patients with heart "
        "disease (high-survival subgroup). Intercept 0.67; slopes 0.023 (CPR), 0.011 "
        "(defibrillation), 0.021 (ACLS) per minute; sum 0.055 per minute. Western cohort."
    )
    intercept: float = LARSEN_INTERCEPT
    slope_cpr: float = LARSEN_SLOPE_CPR
    slope_defib: float = LARSEN_SLOPE_DEFIB
    slope_acls: float = LARSEN_SLOPE_ACLS

    @property
    def untreated_slope(self) -> float:
        """Survival lost per minute with no intervention at all (0.055 for the published fit)."""
        return self.slope_cpr + self.slope_defib + self.slope_acls

    def __call__(
        self, t_cpr: ArrayLike, t_defib: ArrayLike, t_acls: ArrayLike | None = None
    ) -> ArrayOrFloat:
        a, b, c = _prepare(t_cpr, t_defib, t_acls)
        raw = self.intercept - self.slope_cpr * a - self.slope_defib * b - self.slope_acls * c
        return _finalise(np.clip(raw, 0.0, self.intercept))


@dataclass(frozen=True)
class RuleOfThumbModel(SurvivalModel):
    """Linear decline of survival with the time to first shock (citation key ``kim2026``).

    ``survival = initial - rate * t_defib`` clipped to ``[0, initial]``. ``rate`` is in
    survival probability per minute (0.07 to 0.10 quoted). CPR and ACLS times are ignored, so
    this is a coarse sensitivity model and not a fitted regression.
    """

    name: str = "rule_of_thumb"
    citation_key: str = "kim2026"
    notes: str = (
        "Guideline rule of thumb (7% to 10% per minute), read as percentage points per minute "
        "and applied to the time to first shock only. ASSUMPTION: midpoint rate 0.085 and "
        "initial survival 0.67. Not a fitted model; sensitivity use only."
    )
    rate_per_min: float = RULE_RATE_MID
    initial_survival: float = RULE_INITIAL_SURVIVAL

    def __post_init__(self) -> None:
        if not 0.0 < self.rate_per_min < 1.0:
            raise ValueError("rate_per_min must be in (0, 1) probability per minute")
        if not 0.0 < self.initial_survival <= 1.0:
            raise ValueError("initial_survival must be in (0, 1]")

    def __call__(
        self, t_cpr: ArrayLike, t_defib: ArrayLike, t_acls: ArrayLike | None = None
    ) -> ArrayOrFloat:
        _, b, _ = _prepare(t_cpr, t_defib, t_acls)
        raw = self.initial_survival - self.rate_per_min * b
        return _finalise(np.clip(raw, 0.0, self.initial_survival))


SURVIVAL_MODELS: dict[str, SurvivalModel] = {
    "larsen1993": LarsenModel(),
    "rule_of_thumb": RuleOfThumbModel(name="rule_of_thumb", rate_per_min=RULE_RATE_MID),
    "rule_of_thumb_low": RuleOfThumbModel(name="rule_of_thumb_low", rate_per_min=RULE_RATE_LOW),
    "rule_of_thumb_high": RuleOfThumbModel(name="rule_of_thumb_high", rate_per_min=RULE_RATE_HIGH),
}


def model_names() -> list[str]:
    """Registered survival-model names, in registration order."""
    return list(SURVIVAL_MODELS)


def get_model(name: str) -> SurvivalModel:
    """Look a model up by name; raises ``KeyError`` listing the valid names."""
    try:
        return SURVIVAL_MODELS[name]
    except KeyError:
        raise KeyError(f"unknown survival model {name!r}; choose from {model_names()}") from None


def delta_survival(
    model: SurvivalModel,
    t_defib_a: ArrayLike,
    t_defib_b: ArrayLike,
    t_cpr: ArrayLike,
    t_acls: ArrayLike | None = None,
) -> ArrayOrFloat:
    """Survival gain of arm A over arm B when only the defibrillation time differs.

    Returns ``model(t_cpr, t_defib_a, t_acls) - model(t_cpr, t_defib_b, t_acls)`` (probability
    difference). Positive when arm A shocks earlier. All times in minutes from collapse; CPR and
    ACLS times are shared by both arms (a delivery device changes only the shock time).
    """
    return model(t_cpr, t_defib_a, t_acls) - model(t_cpr, t_defib_b, t_acls)
