"""Route composition and dispatch-policy tests."""

import numpy as np
import pandas as pd
import pytest

from aedrover.analysis.route_model import SEGMENT_M, compose_routes, summary
from aedrover.clinical.decision import ScenarioParams
from aedrover.clinical.dispatch_policy import POLICIES, evaluate_policies, policy_samples
from aedrover.clinical.survival import get_model


def _episodes(p_fail=0.0, shock=1.0):
    rng = np.random.default_rng(0)
    rows = []
    for fam, t0 in (("flat_clear", 20.0), ("crowded", 26.0), ("kerb", 24.0), ("mixed", 30.0)):
        for _ in range(200):
            ok = rng.random() >= p_fail
            rows.append({"family": fam, "success": ok, "time_s": rng.normal(t0, 1.0), "peak_shock_g": shock})
    return pd.DataFrame(rows)


def test_route_time_scales_with_distance_and_matches_segment_means():
    eps = _episodes()
    a = compose_routes(eps, 360.0, 0, crowd_prob=0.0, n=500, seed=1)          # 10 flat segments
    assert a.n_segments == 10 and a.n_crossing_segments == 0
    assert a.arrived.all() and a.safe.all()
    assert np.median(a.time_min) == pytest.approx(10 * 20.0 / 60.0, rel=0.03)
    b = compose_routes(eps, 720.0, 0, crowd_prob=0.0, n=500, seed=1)
    assert np.median(b.time_min) == pytest.approx(2 * np.median(a.time_min), rel=0.05)


def test_failures_and_unsafe_shocks_propagate_to_the_route():
    reliable = compose_routes(_episodes(0.0, 1.0), 360.0, 2, n=1000, seed=2)
    flaky = compose_routes(_episodes(0.10, 1.0), 360.0, 2, n=1000, seed=2)
    assert flaky.arrived.mean() == pytest.approx(0.9**10, abs=0.05)
    assert flaky.arrived.mean() < reliable.arrived.mean() == 1.0
    harsh = compose_routes(_episodes(0.0, 4.0), 360.0, 2, n=200, seed=3)      # every segment breaks the budget
    assert harsh.arrived.all() and not harsh.safe.any() and np.isinf(harsh.time_min).all()
    assert summary(reliable)["p_safe_delivery"] == 1.0


def test_crossings_are_capped_by_segment_count_and_deterministic():
    eps = _episodes()
    r = compose_routes(eps, 2 * SEGMENT_M, 50, n=100, seed=4)
    assert r.n_crossing_segments == r.n_segments == 2
    r1, r2 = compose_routes(eps, 300.0, 3, n=100, seed=5), compose_routes(eps, 300.0, 3, n=100, seed=5)
    np.testing.assert_array_equal(r1.time_min, r2.time_min)


def test_dispatch_policies_are_ordered_sensibly():
    params = ScenarioParams(radius_m=1200.0)
    model = get_model("larsen1993")
    n = 3000
    rover = np.full(n, 9.0)                                   # 9 minutes of rover travel
    df = evaluate_policies(model, params, rover_travel_min=rover, p_drone=0.6, n_resamples=200)
    s = df.set_index("policy").mean_survival
    assert set(df.policy) == set(POLICIES)
    assert s["both"] >= s["hybrid"] - 1e-9 and s["both"] >= s["rover"] - 1e-9 and s["both"] >= s["drone"] - 1e-9
    assert s["rover"] >= s["ambulance"] - 1e-9 and s["drone"] >= s["ambulance"] - 1e-9
    assert s["hybrid"] >= s["rover"] - 1e-9                    # adding the drone when it can fly never hurts


def test_p_drone_extremes_recover_pure_policies_and_failed_rovers_fall_back_to_the_ambulance():
    params = ScenarioParams(radius_m=800.0)
    n = 2000
    ok = policy_samples(params, rover_travel_min=np.full(n, 8.0), p_drone=0.0, n=n, seed=1)
    np.testing.assert_allclose(ok["drone"].t_defib, ok["ambulance"].t_defib)      # never flies
    np.testing.assert_allclose(ok["hybrid"].t_defib, ok["rover"].t_defib)         # falls back to the rover
    dead = policy_samples(params, rover_travel_min=np.full(n, np.inf), p_drone=0.0, n=n, seed=1)
    np.testing.assert_allclose(dead["rover"].t_defib, dead["ambulance"].t_defib)  # rover failed: ambulance shocks
    with pytest.raises(ValueError):
        policy_samples(params, rover_travel_min=np.zeros(3), p_drone=1.5, n=3, seed=0)
