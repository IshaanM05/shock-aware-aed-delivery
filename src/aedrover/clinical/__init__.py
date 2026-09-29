"""Clinical, economic and decision layer: survival vs delay, EMS delays, dimensionless economics.

Units: clinical times in minutes from collapse, distances in metres, speeds in metres per second.
See docs/CLINICAL_MODEL.md for the equations, assumptions and limitations.
"""

from aedrover.clinical.decision import (
    Breakeven,
    ModeSamples,
    ScenarioParams,
    breakeven_radius,
    breakeven_surface,
    compare_modes,
    evaluate_modes,
    expected_gain,
    mode_times,
    tornado,
)
from aedrover.clinical.economics import (
    KAPPA_TARGET,
    EconomicsInputs,
    delta_fte,
    kappa,
    kappa_from_ratios,
    payback_months,
    qaly_gain_per_encounter,
)
from aedrover.clinical.ems_delay import (
    NAESS_ALL,
    NAESS_RURAL,
    NAESS_URBAN,
    DispatchTimes,
    ResponseTimeSpec,
    sample_ambulance_arrival,
)
from aedrover.clinical.survival import (
    SURVIVAL_MODELS,
    LarsenModel,
    RuleOfThumbModel,
    SurvivalModel,
    delta_survival,
    get_model,
    model_names,
)

__all__ = [
    "KAPPA_TARGET",
    "NAESS_ALL",
    "NAESS_RURAL",
    "NAESS_URBAN",
    "SURVIVAL_MODELS",
    "Breakeven",
    "DispatchTimes",
    "EconomicsInputs",
    "LarsenModel",
    "ModeSamples",
    "ResponseTimeSpec",
    "RuleOfThumbModel",
    "ScenarioParams",
    "SurvivalModel",
    "breakeven_radius",
    "breakeven_surface",
    "compare_modes",
    "delta_fte",
    "delta_survival",
    "evaluate_modes",
    "expected_gain",
    "get_model",
    "kappa",
    "kappa_from_ratios",
    "mode_times",
    "model_names",
    "payback_months",
    "qaly_gain_per_encounter",
    "sample_ambulance_arrival",
    "tornado",
]
