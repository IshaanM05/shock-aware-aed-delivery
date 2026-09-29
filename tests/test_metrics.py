"""Unit tests for the payload-shock metric definitions."""

import numpy as np

from aedrover.sim.metrics import G, dynamic_acceleration, lowpass, shock_metrics

FS = 250.0


def test_rest_reads_zero_shock_even_when_tilted():
    n = 500
    tilt = np.radians(20.0)
    up = np.tile([np.sin(tilt), 0.0, np.cos(tilt)], (n, 1))  # world-up in the tilted chassis frame
    acc = G * up                                              # accelerometer at rest on a slope
    res = shock_metrics(acc, up, FS)
    assert res.peak_g < 1e-9


def test_gravity_compensation_matters():
    n = 500
    up = np.tile([0.0, 0.0, 1.0], (n, 1))
    acc = G * up
    dyn = dynamic_acceleration(acc, up)
    assert np.allclose(dyn, 0.0)


def test_known_vertical_impulse_peak():
    n = 500
    up = np.tile([0.0, 0.0, 1.0], (n, 1))
    acc = G * up
    t = np.arange(n) / FS
    acc[:, 2] += 2.0 * G * np.exp(-0.5 * ((t - 1.0) / 0.02) ** 2)  # 2 g gaussian bump, sigma 20 ms
    res = shock_metrics(acc, up, FS)
    assert abs(res.peak_g - 2.0) < 0.05
    assert abs(res.peak_vertical_g - 2.0) < 0.05
    assert res.peak_lateral_g < 1e-6
    assert abs(res.t_peak - 1.0) < 0.02


def test_lowpass_removes_high_frequency_chatter_but_keeps_shock():
    n = 1000
    t = np.arange(n) / FS
    slow = np.sin(2 * np.pi * 5.0 * t)
    fast = np.sin(2 * np.pi * 110.0 * t)  # above the 80 Hz cut-off
    out = lowpass(slow + fast, FS)
    assert np.max(np.abs(out - slow)[50:-50]) < 0.15


def test_budget_fraction():
    n = 400
    up = np.tile([0.0, 0.0, 1.0], (n, 1))
    acc = G * up
    acc[100:200, 2] += 4.0 * G  # a 4 g plateau over 100 samples (25 % of the run)
    res = shock_metrics(acc, up, FS, budget_g=3.0)
    assert 0.2 < res.frac_over < 0.3
