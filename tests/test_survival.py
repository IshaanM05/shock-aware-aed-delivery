"""Tests for the survival-vs-delay models and the EMS delay model."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from aedrover.clinical.ems_delay import (
    BUSY_MEAN_EXTRA_MIN,
    DISPATCH_RANGES,
    NAESS_ALL,
    NAESS_REPORTED_IQR_MIN,
    NAESS_REPORTED_MEAN_MIN,
    NAESS_RURAL,
    NAESS_URBAN,
    RESPONSE_TIME_PRESETS,
    DispatchTimes,
    ResponseTimeSpec,
    busy_delay_min,
    sample_ambulance_arrival,
)
from aedrover.clinical.survival import (
    SURVIVAL_MODELS,
    LarsenModel,
    RuleOfThumbModel,
    delta_survival,
    get_model,
    model_names,
)

REFS = Path(__file__).resolve().parents[1] / "docs" / "references.json"


@pytest.fixture(scope="module")
def larsen() -> LarsenModel:
    model = SURVIVAL_MODELS["larsen1993"]
    assert isinstance(model, LarsenModel)
    return model


# ---------------------------------------------------------------------------------------------
# Larsen 1993
# ---------------------------------------------------------------------------------------------
def test_larsen_published_coefficients(larsen):
    """Coefficients of Larsen et al. 1993 (PMID 8214853): 67%, 2.3%, 1.1%, 2.1% per minute."""
    assert larsen.intercept == 0.67
    assert larsen.slope_cpr == 0.023
    assert larsen.slope_defib == 0.011  # NOT 0.046 (earlier student draft was wrong)
    assert larsen.slope_acls == 0.021
    assert larsen.untreated_slope == pytest.approx(0.055)


def test_larsen_exact_values(larsen):
    assert larsen(0, 0, 0) == pytest.approx(0.67)
    assert larsen(10, 10, 10) == pytest.approx(0.67 - 0.55)  # 5.5% per minute untreated
    assert larsen(1, 0, 0) == pytest.approx(0.67 - 0.023)
    assert larsen(0, 1, 0) == pytest.approx(0.67 - 0.011)
    assert larsen(0, 0, 1) == pytest.approx(0.67 - 0.021)
    assert larsen(3.0, 5.0, 11.0) == pytest.approx(0.67 - 0.069 - 0.055 - 0.231)


def test_larsen_acls_default_is_defib_time(larsen):
    assert larsen(2.0, 6.0) == pytest.approx(larsen(2.0, 6.0, 6.0))
    assert larsen(2.0, 6.0, 12.0) < larsen(2.0, 6.0)


def test_larsen_monotone_nonincreasing_in_every_delay(larsen):
    grid = np.linspace(0.0, 20.0, 41)
    base = 3.0
    for axis in range(3):
        args = [np.full_like(grid, base)] * 3
        args[axis] = grid
        s = np.asarray(larsen(*args))
        assert np.all(np.diff(s) <= 1e-15)
        assert s[0] >= s[-1]


def test_larsen_clipping(larsen):
    assert larsen(100, 100, 100) == 0.0  # below zero is clipped
    assert larsen(0, 0, 0) <= 0.67
    assert larsen(np.inf, 5.0, 5.0) == 0.0  # a mode that never arrives
    s = np.asarray(larsen(np.linspace(0, 60, 121), np.linspace(0, 60, 121)))
    assert s.min() >= 0.0 and s.max() <= 0.67
    # boundary of the clipped region: 0.55 * t = 0.67 -> t = 12.18 min untreated
    assert larsen(12.0, 12.0, 12.0) == pytest.approx(0.67 - 0.055 * 12.0)
    assert larsen(12.3, 12.3, 12.3) == 0.0


def test_larsen_broadcasting_and_scalar_return(larsen):
    out = larsen(np.array([[1.0], [2.0], [3.0]]), np.array([[0.0, 5.0, 10.0, 15.0]]), 4.0)
    assert np.shape(out) == (3, 4)
    assert out[1, 2] == pytest.approx(0.67 - 0.023 * 2 - 0.011 * 10 - 0.021 * 4)
    assert isinstance(larsen(1.0, 2.0, 3.0), float)
    assert np.shape(larsen(np.ones(5), 2.0)) == (5,)


def test_models_reject_invalid_times(larsen):
    with pytest.raises(ValueError):
        larsen(-1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        larsen(1.0, math.nan, 1.0)
    with pytest.raises(ValueError):
        SURVIVAL_MODELS["rule_of_thumb"](1.0, -0.5)


# ---------------------------------------------------------------------------------------------
# rule of thumb
# ---------------------------------------------------------------------------------------------
def test_rule_of_thumb_values_and_clip():
    rot = RuleOfThumbModel(rate_per_min=0.10, initial_survival=0.6)
    assert rot(0, 0) == pytest.approx(0.6)
    assert rot(0, 3.0) == pytest.approx(0.3)
    assert rot(0, 6.0) == 0.0
    assert rot(0, 100.0) == 0.0
    assert np.asarray(rot(0, np.linspace(0, 20, 50))).min() >= 0.0


def test_rule_of_thumb_uses_defib_time_only():
    rot = SURVIVAL_MODELS["rule_of_thumb"]
    assert rot(0.0, 3.0, 3.0) == pytest.approx(rot(9.0, 3.0, 20.0))


def test_rule_of_thumb_range_ordering():
    low, mid, high = (SURVIVAL_MODELS[k] for k in
                      ("rule_of_thumb_low", "rule_of_thumb", "rule_of_thumb_high"))
    assert (low.rate_per_min, mid.rate_per_min, high.rate_per_min) == (0.07, 0.085, 0.10)
    assert low(0, 4.0) > mid(0, 4.0) > high(0, 4.0)


def test_rule_of_thumb_validates_parameters():
    with pytest.raises(ValueError):
        RuleOfThumbModel(rate_per_min=0.0)
    with pytest.raises(ValueError):
        RuleOfThumbModel(rate_per_min=0.08, initial_survival=1.5)


# ---------------------------------------------------------------------------------------------
# registry and helper
# ---------------------------------------------------------------------------------------------
def test_registry_names_and_lookup():
    names = model_names()
    assert names[0] == "larsen1993"
    assert "valenzuela1997" not in names  # coefficients not obtainable from a primary source
    assert set(names) == set(SURVIVAL_MODELS)
    for name in names:
        assert get_model(name).name == name
    with pytest.raises(KeyError, match="larsen1993"):
        get_model("nope")


def test_registry_citation_keys_exist_in_references():
    keys = {r["key"] for r in json.loads(REFS.read_text(encoding="utf-8"))["references"]}
    for model in SURVIVAL_MODELS.values():
        assert model.citation_key in keys
        assert model.notes


def test_models_are_frozen():
    with pytest.raises(AttributeError):
        SURVIVAL_MODELS["larsen1993"].intercept = 0.9  # type: ignore[misc]


def test_delta_survival_linear_regime(larsen):
    d = delta_survival(larsen, 4.0, 9.0, t_cpr=3.0, t_acls=10.0)
    assert d == pytest.approx(0.011 * 5.0)
    assert delta_survival(larsen, 9.0, 4.0, 3.0, 10.0) == pytest.approx(-0.011 * 5.0)
    assert delta_survival(larsen, 5.0, 5.0, 3.0, 10.0) == 0.0
    arr = delta_survival(larsen, np.array([2.0, 4.0]), 9.0, 3.0, 10.0)
    assert np.allclose(arr, [0.011 * 7.0, 0.011 * 5.0])


def test_delta_survival_shrinks_when_clipped(larsen):
    assert delta_survival(larsen, 1.0, 60.0, 30.0, 30.0) < 0.011 * 59.0


# ---------------------------------------------------------------------------------------------
# EMS delay model
# ---------------------------------------------------------------------------------------------
def test_naess_table_values():
    """Values transcribed from Table 1 of Naess et al. 2024 (PLOS ONE 19:e0296308)."""
    assert (NAESS_ALL.median_min, NAESS_ALL.p90_min) == (12.2, 29.1)
    assert (NAESS_RURAL.median_min, NAESS_RURAL.p90_min) == (14.8, 33.3)
    assert (NAESS_URBAN.median_min, NAESS_URBAN.p90_min) == (10.0, 17.7)
    assert set(RESPONSE_TIME_PRESETS) == {"all", "rural", "urban"}
    assert all(s.citation_key == "naess2024" for s in RESPONSE_TIME_PRESETS.values())
    assert BUSY_MEAN_EXTRA_MIN["all"] == 1.61


@pytest.mark.parametrize("key", ["all", "rural", "urban"])
def test_lognormal_fit_quality_against_reported_moments(key):
    """The (median, p90) lognormal reproduces the reported mean and IQR to within a few %."""
    spec = RESPONSE_TIME_PRESETS[key]
    assert spec.mean_min == pytest.approx(NAESS_REPORTED_MEAN_MIN[key], rel=0.06)
    assert spec.iqr_min == pytest.approx(NAESS_REPORTED_IQR_MIN[key], rel=0.12)


def test_spec_quantiles_are_exact():
    spec = ResponseTimeSpec(10.0, 17.7)
    assert spec.quantile(0.5) == pytest.approx(10.0)
    assert spec.quantile(0.9) == pytest.approx(17.7)
    assert spec.mu == pytest.approx(math.log(10.0))
    assert spec.sigma == pytest.approx(math.log(1.77) / 1.2815515655446004)


def test_spec_validation():
    for bad in ((0.0, 5.0), (5.0, 5.0), (8.0, 5.0), (math.inf, 9.0), (-1.0, 4.0)):
        with pytest.raises(ValueError):
            ResponseTimeSpec(*bad)


def test_sampling_quantiles_converge_to_median_and_p90():
    rng = np.random.default_rng(12345)
    x = sample_ambulance_arrival(400_000, rng, median_min=12.2, p90_min=29.1)
    assert np.median(x) == pytest.approx(12.2, rel=0.01)
    assert np.quantile(x, 0.9) == pytest.approx(29.1, rel=0.015)
    assert x.min() > 0.0


def test_sampling_is_reproducible_and_seed_dependent():
    a = sample_ambulance_arrival(1000, np.random.default_rng(7))
    b = sample_ambulance_arrival(1000, np.random.default_rng(7))
    c = sample_ambulance_arrival(1000, np.random.default_rng(8))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_busy_delay_shifts_samples():
    base = sample_ambulance_arrival(500, np.random.default_rng(3))
    busy = sample_ambulance_arrival(500, np.random.default_rng(3), busy_increase=0.20)
    assert np.allclose(busy - base, 0.60)  # 0.30 min per 10 pp (urban) x 2
    assert busy_delay_min(0.10, "all") == pytest.approx(0.60)
    assert busy_delay_min(0.05, "rural") == pytest.approx(0.405)
    assert busy_delay_min(0.0) == 0.0
    with pytest.raises(KeyError):
        busy_delay_min(0.1, "suburban")


def test_sampler_rejects_bad_arguments():
    with pytest.raises(ValueError):
        sample_ambulance_arrival(0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_ambulance_arrival(10, np.random.default_rng(0), median_min=20.0, p90_min=10.0)


def test_quantile_grid_is_ascending_and_unbiased():
    spec = NAESS_URBAN
    g = spec.quantile_grid(4000)
    assert np.all(np.diff(g) > 0)
    assert g.mean() == pytest.approx(spec.mean_min, rel=0.01)
    assert np.median(g) == pytest.approx(spec.median_min, rel=1e-3)
    with pytest.raises(ValueError):
        spec.quantile_grid(0)


def test_with_median_keeps_shape():
    s = NAESS_URBAN.with_median(20.0)
    assert s.median_min == 20.0
    assert s.p90_min / s.median_min == pytest.approx(NAESS_URBAN.p90_min / NAESS_URBAN.median_min)


def test_dispatch_defaults_are_within_their_sensitivity_ranges():
    d = DispatchTimes()
    for name, (lo, hi) in DISPATCH_RANGES.items():
        assert lo <= getattr(d, name) <= hi


def test_dispatch_validation_and_no_bystander_cpr():
    assert DispatchTimes(bystander_cpr_delay_min=None).bystander_cpr_delay_min is None
    with pytest.raises(ValueError):
        DispatchTimes(handoff_min=-0.1)
    with pytest.raises(ValueError):
        DispatchTimes(bystander_cpr_delay_min=-1.0)
