"""Tests for the decision layer: mode times, evaluation, break-even radius, surface, tornado."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aedrover.clinical.decision import (
    DEFAULT_TORNADO_RANGES,
    SCHIERBECK_FRACTION_DRONE_FIRST,
    SCHIERBECK_MEDIAN_BENEFIT_MIN,
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
from aedrover.clinical.ems_delay import NAESS_URBAN, DispatchTimes
from aedrover.clinical.survival import SURVIVAL_MODELS, LarsenModel

LARSEN = SURVIVAL_MODELS["larsen1993"]
ROVER_1KM = 1.0 + 1.5 + 1000.0 * 1.3 / 2.0 / 60.0 + 1.0  # collapse->call + alert + travel + handoff
DRONE_1KM = 1.0 + 1.5 + 0.5 + 1000.0 / 15.0 / 60.0 + 1.0


# ---------------------------------------------------------------------------------------------
# mode_times
# ---------------------------------------------------------------------------------------------
def test_mode_times_decomposition_on_quantile_grid():
    p = ScenarioParams()
    s = mode_times(500, None, p, parallel=False)
    scene = NAESS_URBAN.quantile_grid(500) + 1.0
    assert np.allclose(s["ambulance"].t_defib, scene)
    assert np.allclose(s["ambulance"].t_acls, scene)
    assert np.allclose(s["rover"].t_defib, ROVER_1KM)
    assert np.allclose(s["drone"].t_defib, DRONE_1KM)
    assert ROVER_1KM == pytest.approx(14.3333333)
    assert DRONE_1KM == pytest.approx(5.1111111)


def test_rover_changes_only_the_defibrillation_time():
    s = mode_times(300, None, ScenarioParams())
    for mode in ("rover", "drone"):
        assert np.array_equal(s[mode].t_cpr, s["ambulance"].t_cpr)
        assert np.array_equal(s[mode].t_acls, s["ambulance"].t_acls)
    assert np.all(s["rover"].t_defib <= s["ambulance"].t_defib)  # parallel dispatch
    assert np.array_equal(s["rover"].t_defib,
                          np.minimum(ROVER_1KM, s["ambulance"].t_defib))


def test_cpr_time_is_bystander_delay_capped_by_arrival():
    p = ScenarioParams()
    s = mode_times(200, None, p)
    scene = s["ambulance"].t_acls
    assert np.allclose(s["ambulance"].t_cpr, np.minimum(3.0, scene))
    no_cpr = DispatchTimes(bystander_cpr_delay_min=None)
    none = mode_times(200, None, ScenarioParams(dispatch=no_cpr))
    assert np.allclose(none["ambulance"].t_cpr, none["ambulance"].t_acls)
    late_cpr = DispatchTimes(bystander_cpr_delay_min=100.0)
    late = mode_times(200, None, ScenarioParams(dispatch=late_cpr))
    assert np.allclose(late["ambulance"].t_cpr, late["ambulance"].t_acls)  # EMS arrives first


def test_ambulance_pad_delay_only_delays_the_shock():
    p = ScenarioParams(dispatch=DispatchTimes(ems_arrival_to_shock_min=2.0))
    s = mode_times(100, None, p)
    assert np.allclose(s["ambulance"].t_defib - s["ambulance"].t_acls, 2.0)


def test_busy_increase_delays_ambulance():
    base = mode_times(100, None, ScenarioParams())["ambulance"].t_acls
    busy = mode_times(100, None, ScenarioParams(busy_increase=0.2))["ambulance"].t_acls
    assert np.allclose(busy - base, 0.6)


def test_rover_travel_override_from_simulation():
    p = ScenarioParams()
    s = mode_times(50, None, p, modes=("rover",), parallel=False, rover_travel_min=5.0)
    assert np.allclose(s["rover"].t_defib, 1.0 + 1.5 + 5.0 + 1.0)
    trav = np.linspace(2.0, 9.0, 50)
    s2 = mode_times(50, None, p, modes=("rover",), parallel=False, rover_travel_min=trav)
    assert np.allclose(s2["rover"].t_defib, 3.5 + trav)
    with pytest.raises(ValueError):
        mode_times(50, None, p, modes=("rover",), rover_travel_min=-1.0)


def test_mode_times_random_samples_are_seeded_and_paired():
    p = ScenarioParams()
    a = mode_times(1000, np.random.default_rng(4), p)
    b = mode_times(1000, np.random.default_rng(4), p)
    assert np.array_equal(a["ambulance"].t_defib, b["ambulance"].t_defib)
    assert np.median(a["ambulance"].t_acls) == pytest.approx(11.0, rel=0.05)
    with pytest.raises(ValueError):
        mode_times(10, None, p, modes=("hovercraft",))


def test_scenario_and_samples_validation():
    with pytest.raises(ValueError):
        ScenarioParams(rover_speed_mps=0.0)
    with pytest.raises(ValueError):
        ScenarioParams(route_factor=0.9)
    with pytest.raises(ValueError):
        ScenarioParams(radius_m=-1.0)
    with pytest.raises(ValueError):
        ModeSamples(np.zeros(3), np.zeros(4), np.zeros(3))


# ---------------------------------------------------------------------------------------------
# evaluate_modes
# ---------------------------------------------------------------------------------------------
def test_evaluate_modes_exact_case():
    n = 200
    amb = ModeSamples(np.full(n, 10.0), np.full(n, 3.0), np.full(n, 10.0))
    rov = ModeSamples(np.full(n, 5.0), np.full(n, 3.0), np.full(n, 10.0))
    df = evaluate_modes({"ambulance": amb, "rover": rov}, LARSEN)
    s_amb = 0.67 - 0.023 * 3 - 0.011 * 10 - 0.021 * 10
    row = df.set_index("mode").loc["rover"]
    assert row["mean_survival"] == pytest.approx(s_amb + 0.011 * 5)
    assert row["abs_gain"] == pytest.approx(0.055)
    assert row["abs_gain_ci_low"] == pytest.approx(0.055)  # no sampling variability
    assert row["rel_gain"] == pytest.approx(0.055 / s_amb)
    assert row["p_faster"] == 1.0
    assert row["median_saving_when_faster_min"] == pytest.approx(5.0)
    base = df.set_index("mode").loc["ambulance"]
    assert base["abs_gain"] == 0.0 and base["rel_gain"] == 0.0 and base["p_faster"] == 0.0


def test_evaluate_modes_random_samples():
    rng = np.random.default_rng(10)
    samples = mode_times(3000, rng, ScenarioParams())
    df = evaluate_modes(samples, LARSEN, n_resamples=1000, seed=1).set_index("mode")
    assert list(df.index) == ["ambulance", "rover", "drone"]
    for mode in ("rover", "drone"):
        r = df.loc[mode]
        assert r["abs_gain"] > 0.0
        assert r["abs_gain_ci_low"] <= r["abs_gain"] <= r["abs_gain_ci_high"]
        assert r["survival_ci_low"] <= r["mean_survival"] <= r["survival_ci_high"]
        assert 0.0 < r["p_faster"] <= 1.0
        base_mean = df.loc["ambulance", "mean_survival"]
        assert r["rel_gain"] == pytest.approx(r["mean_survival"] / base_mean - 1)
    assert df.loc["drone", "abs_gain"] > df.loc["rover", "abs_gain"]  # 1 km: drone is faster
    assert df.loc["drone", "abs_gain_ci_low"] > 0.0


def test_evaluate_modes_paired_identical_samples_give_zero_gain():
    far = ScenarioParams(radius_m=1e9)
    s = mode_times(500, np.random.default_rng(2), far, modes=("ambulance", "rover"))
    df = evaluate_modes(s, LARSEN, n_resamples=500).set_index("mode")
    assert df.loc["rover", "abs_gain"] == pytest.approx(0.0, abs=1e-12)  # rover never first


def test_evaluate_modes_handles_zero_baseline_and_bad_baseline():
    n = 50
    dead = ModeSamples(np.full(n, 50.0), np.full(n, 50.0), np.full(n, 50.0))
    alive = ModeSamples(np.full(n, 1.0), np.full(n, 50.0), np.full(n, 50.0))
    rule = SURVIVAL_MODELS["rule_of_thumb"]
    df = evaluate_modes({"ambulance": dead, "rover": alive}, rule).set_index("mode")
    assert np.isnan(df.loc["rover", "rel_gain"])
    assert df.loc["rover", "abs_gain"] > 0.0
    with pytest.raises(KeyError):
        evaluate_modes({"rover": alive}, LARSEN)


def test_compare_modes_is_reproducible_and_parallel_flag_matters():
    a = compare_modes(LARSEN, n=800, seed=3, n_resamples=300)
    b = compare_modes(LARSEN, n=800, seed=3, n_resamples=300)
    pd.testing.assert_frame_equal(a, b)
    alone = compare_modes(LARSEN, ScenarioParams(), n=800, seed=3, parallel=False, n_resamples=300)
    gain_alone = alone.set_index("mode").loc["rover", "abs_gain"]
    gain_parallel = a.set_index("mode").loc["rover", "abs_gain"]
    assert gain_alone < gain_parallel
    assert gain_alone < 0.0  # a rover 1 km away, on its own, loses to the ambulance


def test_schierbeck_reference_constants():
    assert SCHIERBECK_FRACTION_DRONE_FIRST == pytest.approx(37 / 55)
    assert SCHIERBECK_MEDIAN_BENEFIT_MIN == pytest.approx(3.2333, abs=1e-4)


# ---------------------------------------------------------------------------------------------
# expected_gain
# ---------------------------------------------------------------------------------------------
def test_expected_gain_signs_and_metrics():
    p = ScenarioParams()
    assert expected_gain(LARSEN, p, mode="ambulance") == 0.0
    par = expected_gain(LARSEN, p)
    alone = expected_gain(LARSEN, p, metric="alone")
    assert par > 0.0 > alone
    assert par >= alone
    near = ScenarioParams(radius_m=100.0)
    assert expected_gain(LARSEN, near, metric="alone") > 0.0
    assert expected_gain(LARSEN, p, mode="drone") > par
    with pytest.raises(ValueError):
        expected_gain(LARSEN, p, metric="both")


# ---------------------------------------------------------------------------------------------
# break-even radius
# ---------------------------------------------------------------------------------------------
def _be(**overrides):
    kw = dict(speed_mps=2.0, route_factor=1.3, dispatch_min=1.5, handoff_min=1.0,
              ems_median_min=10.0, model=LARSEN)
    kw.update(overrides)
    return breakeven_radius(**kw)


def test_breakeven_is_a_root_of_the_expected_survival_difference():
    res = _be()
    assert res.flag == "crossing" and 100.0 < res.radius_m < 5000.0
    p = ScenarioParams(radius_m=res.radius_m, rover_speed_mps=2.0, route_factor=1.3,
                       dispatch=DispatchTimes(call_to_alert_min=1.5, handoff_min=1.0))
    assert expected_gain(LARSEN, p, metric="alone") == pytest.approx(0.0, abs=1e-9)
    assert res.rover_survival_at_zero > res.ambulance_survival


def test_breakeven_matches_closed_form_in_the_linear_regime():
    """With tiny slopes nothing is clipped, so E[S] is linear and the root has a closed form."""
    lin = LarsenModel(slope_cpr=0.001, slope_defib=0.001, slope_acls=0.001)
    res = _be(model=lin)
    mean_ambulance = 1.0 + NAESS_URBAN.quantile_grid(2000).mean()  # collapse->call + response
    travel_min = mean_ambulance - (1.0 + 1.5 + 1.0)
    assert res.radius_m == pytest.approx(travel_min * 60.0 * 2.0 / 1.3, rel=1e-6)


def test_breakeven_comparative_statics():
    base = _be().radius_m
    assert _be(speed_mps=3.0).radius_m > base > _be(speed_mps=1.0).radius_m
    assert _be(dispatch_min=0.5).radius_m > base > _be(dispatch_min=3.0).radius_m
    assert _be(handoff_min=0.5).radius_m > base > _be(handoff_min=2.5).radius_m
    assert _be(route_factor=1.1).radius_m > base > _be(route_factor=1.6).radius_m
    assert _be(ems_median_min=14.8).radius_m > base > _be(ems_median_min=8.0).radius_m
    assert _be(busy_increase=0.2).radius_m > base


def test_breakeven_defined_for_every_registered_model():
    for name, model in SURVIVAL_MODELS.items():
        res = _be(model=model)
        assert res.flag == "crossing", name
        assert 0.0 < res.radius_m < 50_000.0


def test_breakeven_no_crossing_flags():
    never = _be(dispatch_min=60.0)  # alert takes an hour: worse than the ambulance even at 0 m
    assert never.radius_m == 0.0 and never.flag == "rover_never_better"
    always = _be(ems_median_min=1000.0)  # ambulance never arrives in a survivable time
    assert np.isinf(always.radius_m) and always.flag == "rover_always_better"


# ---------------------------------------------------------------------------------------------
# surface
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def surface() -> pd.DataFrame:
    radii = np.arange(0.0, 3001.0, 250.0)
    return breakeven_surface(LARSEN, radii, [1.0, 2.0, 3.0], [0.5, 1.5, 3.0], n_grid=2000)


def test_surface_shape_and_columns(surface):
    assert len(surface) == 13 * 3 * 3
    expected = {"model", "radius_m", "speed_mps", "dispatch_min", "s_ambulance", "s_rover_alone",
                "s_rover_parallel", "gain_alone", "gain_parallel", "rover_better"}
    assert expected <= set(surface.columns)
    assert surface["model"].eq("larsen1993").all()
    assert surface["s_ambulance"].nunique() == 1


def test_surface_monotonicity(surface):
    for (_, _), g in surface.groupby(["speed_mps", "dispatch_min"]):
        assert np.all(np.diff(g.sort_values("radius_m")["gain_alone"]) <= 1e-12)
    for (_, _), g in surface.groupby(["radius_m", "dispatch_min"]):
        assert np.all(np.diff(g.sort_values("speed_mps")["gain_alone"]) >= -1e-12)
    for (_, _), g in surface.groupby(["radius_m", "speed_mps"]):
        assert np.all(np.diff(g.sort_values("dispatch_min")["gain_alone"]) <= 1e-12)


def test_surface_parallel_gain_is_nonnegative_and_dominates(surface):
    assert (surface["gain_parallel"] >= -1e-12).all()
    assert (surface["gain_parallel"] >= surface["gain_alone"] - 1e-12).all()
    assert (surface["rover_better"] == (surface["gain_alone"] > 0)).all()


def test_surface_agrees_with_expected_gain_and_breakeven(surface):
    at_point = (surface.radius_m == 750.0) & (surface.speed_mps == 2.0)
    row = surface[at_point & (surface.dispatch_min == 1.5)].iloc[0]
    p = ScenarioParams(radius_m=750.0, dispatch=DispatchTimes(call_to_alert_min=1.5))
    assert row["gain_alone"] == pytest.approx(expected_gain(LARSEN, p, metric="alone", n_grid=2000))
    par = expected_gain(LARSEN, p, metric="parallel", n_grid=2000)
    assert row["gain_parallel"] == pytest.approx(par)
    root = _be().radius_m
    fine = breakeven_surface(LARSEN, [0.98 * root, 1.02 * root], [2.0], [1.5], n_grid=2000)
    assert fine["gain_alone"].iloc[0] > 0.0 > fine["gain_alone"].iloc[1]


def test_surface_validation():
    with pytest.raises(ValueError):
        breakeven_surface(LARSEN, [100.0], [0.0], [1.0])


# ---------------------------------------------------------------------------------------------
# tornado
# ---------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def torn() -> pd.DataFrame:
    return tornado(LARSEN, ScenarioParams())


def test_tornado_structure_and_ordering(torn):
    assert set(torn["parameter"]) == set(DEFAULT_TORNADO_RANGES)
    assert list(torn.columns) == ["parameter", "low", "high", "gain_at_low", "gain_at_high",
                                  "gain_base", "swing"]
    assert torn["swing"].is_monotonic_decreasing
    assert np.allclose(torn["swing"], (torn["gain_at_high"] - torn["gain_at_low"]).abs())
    assert torn["gain_base"].nunique() == 1
    assert torn["gain_base"].iloc[0] == pytest.approx(expected_gain(LARSEN, ScenarioParams()))
    assert torn["parameter"].iloc[0] == "radius_m"  # distance dominates the rover benefit


def test_tornado_directions(torn):
    t = torn.set_index("parameter")
    assert t.loc["radius_m", "gain_at_high"] < t.loc["radius_m", "gain_at_low"]
    assert t.loc["rover_speed_mps", "gain_at_high"] > t.loc["rover_speed_mps", "gain_at_low"]
    assert t.loc["route_factor", "gain_at_high"] < t.loc["route_factor", "gain_at_low"]
    assert t.loc["ems_median_min", "gain_at_high"] > t.loc["ems_median_min", "gain_at_low"]
    assert t.loc["busy_increase", "gain_at_high"] > t.loc["busy_increase", "gain_at_low"]
    for name in ("dispatch.handoff_min", "dispatch.call_to_alert_min"):
        assert t.loc[name, "gain_at_high"] < t.loc[name, "gain_at_low"]


def test_tornado_custom_ranges_metric_and_mode():
    t = tornado(LARSEN, ScenarioParams(), {"radius_m": (100.0, 400.0)}, metric="alone")
    assert list(t["parameter"]) == ["radius_m"]
    assert t["gain_at_low"].iloc[0] > 0.0  # a rover 100 m away beats the ambulance outright
    d = tornado(LARSEN, ScenarioParams(), {"drone_speed_mps": (10.0, 25.0)}, mode="drone")
    assert d["gain_at_high"].iloc[0] > d["gain_at_low"].iloc[0]


def test_tornado_validation():
    with pytest.raises(ValueError):
        tornado(LARSEN, ScenarioParams(), {"warp_factor": (1.0, 2.0)})
    with pytest.raises(ValueError):
        tornado(LARSEN, ScenarioParams(), {"dispatch.warp": (1.0, 2.0)})
    with pytest.raises(ValueError):
        tornado(LARSEN, ScenarioParams(), {"dispatch": (1.0, 2.0)})
    with pytest.raises(ValueError):
        tornado(LARSEN, ScenarioParams(), metric="sideways")
