"""Tests for the dimensionless economics model, plus a repo-policy scan of the new sources."""

from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pytest

from aedrover.clinical.economics import (
    HOURS_PER_FTE,
    KAPPA_TARGET,
    EconomicsInputs,
    delta_fte,
    kappa,
    kappa_from_ratios,
    payback_months,
    qaly_gain_per_encounter,
)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------------------------
# kappa
# ---------------------------------------------------------------------------------------------
def test_kappa_is_a_ratio():
    assert kappa(0.25, 1.0) == pytest.approx(0.25)
    assert kappa(30.0, 120.0) == pytest.approx(0.25)
    assert kappa(0.0, 5.0) == 0.0
    assert kappa(3.0, 1.0) == pytest.approx(3.0)  # burden may exceed the benefit


def test_kappa_is_scale_free():
    """Rescaling the reference unit leaves kappa unchanged (dimensionless)."""
    assert kappa(7.0 * 13.0, 40.0 * 13.0) == pytest.approx(kappa(7.0, 40.0))


def test_kappa_from_component_ratios():
    assert kappa_from_ratios(0.04, 0.12, 0.06) == pytest.approx(0.22)
    assert kappa_from_ratios(2.0, 6.0, 4.0, labour_value=48.0) == pytest.approx(0.25)
    out = kappa_from_ratios(np.array([0.1, 0.2]), 0.1, 0.1)
    assert np.allclose(out, [0.3, 0.4])


def test_kappa_validation():
    with pytest.raises(ValueError):
        kappa(-0.1, 1.0)
    with pytest.raises(ValueError):
        kappa(0.1, 0.0)
    with pytest.raises(ValueError):
        kappa_from_ratios(0.1, 0.1, 0.1, labour_value=-1.0)


# ---------------------------------------------------------------------------------------------
# payback
# ---------------------------------------------------------------------------------------------
def test_payback_formula():
    assert payback_months(0.25, 0.6) == pytest.approx(0.6 / 0.75 * 12)  # 9.6 months
    assert payback_months(0.0, 1.0) == pytest.approx(12.0)  # no operating burden
    assert payback_months(0.5, 1.0) == pytest.approx(24.0)
    assert payback_months(0.25, 0.0) == 0.0


def test_payback_increases_with_kappa_and_capex():
    ks = np.linspace(0.0, 0.95, 20)
    pb = np.asarray(payback_months(ks, 0.6))
    assert np.all(np.diff(pb) > 0)
    assert payback_months(0.2, 1.2) == pytest.approx(2 * payback_months(0.2, 0.6))


def test_payback_kappa_at_or_above_one_is_infinite_or_raises():
    assert payback_months(1.0, 0.6) == math.inf
    assert payback_months(1.7, 0.6) == math.inf
    assert payback_months(0.999999, 0.6) > 1e6
    with pytest.raises(ValueError, match="kappa >= 1"):
        payback_months(1.0, 0.6, strict=True)
    with pytest.raises(ValueError):
        payback_months(np.array([0.2, 1.5]), 0.6, strict=True)


def test_payback_vectorised_with_mixed_finite_and_infinite():
    out = np.asarray(payback_months(np.array([0.0, 0.5, 1.2]), np.array([1.0, 1.0, 1.0])))
    assert out[0] == pytest.approx(12.0) and out[1] == pytest.approx(24.0)
    assert math.isinf(out[2])
    assert not np.any(np.isnan(out))


def test_payback_validation():
    with pytest.raises(ValueError):
        payback_months(-0.1, 0.6)
    with pytest.raises(ValueError):
        payback_months(0.1, -0.6)


# ---------------------------------------------------------------------------------------------
# FTE and QALY
# ---------------------------------------------------------------------------------------------
def test_delta_fte():
    assert HOURS_PER_FTE == 2080.0
    assert delta_fte(2080.0) == pytest.approx(1.0)
    assert delta_fte(1040.0) == pytest.approx(0.5)
    assert delta_fte(0.0) == 0.0
    assert np.allclose(delta_fte(np.array([520.0, 4160.0])), [0.25, 2.0])
    assert delta_fte(1000.0, hours_per_fte=1000.0) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        delta_fte(-1.0)
    with pytest.raises(ValueError):
        delta_fte(10.0, hours_per_fte=0.0)


def test_qaly_gain_per_encounter():
    assert qaly_gain_per_encounter(0.05, 30.0, 0.8) == pytest.approx(1.2)
    assert qaly_gain_per_encounter(0.0, 30.0, 0.8) == 0.0
    assert qaly_gain_per_encounter(-0.02, 20.0, 1.0) == pytest.approx(-0.4)  # harm is negative
    assert qaly_gain_per_encounter(1.0, 10.0, 1.0) == pytest.approx(10.0)
    assert qaly_gain_per_encounter(0.05, 30.0, 0.0) == 0.0
    out = qaly_gain_per_encounter(np.array([0.01, 0.02]), 25.0, 0.5)
    assert np.allclose(out, [0.125, 0.25])


def test_qaly_validation():
    with pytest.raises(ValueError):
        qaly_gain_per_encounter(1.5, 30.0, 0.8)
    with pytest.raises(ValueError):
        qaly_gain_per_encounter(0.1, -1.0, 0.8)
    with pytest.raises(ValueError):
        qaly_gain_per_encounter(0.1, 30.0, 1.2)


# ---------------------------------------------------------------------------------------------
# EconomicsInputs
# ---------------------------------------------------------------------------------------------
def test_economics_inputs_defaults():
    e = EconomicsInputs()
    assert e.kappa == pytest.approx(0.22)
    assert KAPPA_TARGET == 0.25 and e.meets_target
    assert e.payback_months == pytest.approx(0.60 / (1 - 0.22) * 12)
    assert e.delta_fte == pytest.approx(0.5)
    s = e.summary()
    assert set(s) == {"kappa", "meets_kappa_target", "payback_months", "delta_fte"}


def test_economics_inputs_over_target_and_never_pays_back():
    over = EconomicsInputs(energy_ratio=0.1, maintenance_ratio=0.15, infrastructure_ratio=0.1)
    assert over.kappa == pytest.approx(0.35) and not over.meets_target
    assert math.isfinite(over.payback_months)
    dead = EconomicsInputs(energy_ratio=0.5, maintenance_ratio=0.4, infrastructure_ratio=0.2)
    assert dead.kappa >= 1.0
    assert dead.payback_months == math.inf


def test_economics_inputs_is_frozen():
    with pytest.raises(AttributeError):
        EconomicsInputs().energy_ratio = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------------------------------------
# repo policy: no currency, plain ASCII (patterns are assembled so this file stays clean too)
# ---------------------------------------------------------------------------------------------
_BANNED_WORDS = ["U" + "SD", "I" + "NR", "E" + "UR", "R" + "s"]  # whole words, case-sensitive
_BANNED_PARTS = ["doll" + "ar", "rup" + "ee"]  # substrings, case-insensitive
_BANNED_CHARS = [chr(36), chr(0x20AC), chr(0xA3), chr(0xA5), chr(0x20B9)]


def _policy_files() -> list[Path]:
    files = sorted((ROOT / "src" / "aedrover" / "clinical").glob("*.py"))
    files += sorted((ROOT / "src" / "aedrover" / "analysis").glob("*.py"))
    files += [ROOT / "tests" / n for n in
              ("test_survival.py", "test_stats.py", "test_economics.py", "test_decision.py")]
    files.append(ROOT / "docs" / "CLINICAL_MODEL.md")
    return [f for f in files if f.exists()]


def test_policy_files_exist():
    assert len(_policy_files()) >= 10


@pytest.mark.parametrize("path", _policy_files(), ids=lambda p: p.name)
def test_no_currency_and_ascii_only(path: Path):
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:  # emoji, dingbats and typographic symbols are not allowed
        pytest.fail(f"{path.name} contains non-ASCII bytes: {exc}")
    for ch in _BANNED_CHARS:
        assert ch not in text, f"{path.name} contains a currency symbol"
    for word in _BANNED_WORDS:
        assert not re.search(rf"\b{word}\b", text), f"{path.name} contains a currency code"
    lower = text.lower()
    for part in _BANNED_PARTS:
        assert part not in lower, f"{path.name} contains a currency word"
