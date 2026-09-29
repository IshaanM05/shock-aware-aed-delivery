"""Environment, scenario, pedestrian and safety-filter tests."""

import numpy as np
import pytest

from aedrover.control import SafetyFilter
from aedrover.nav import make_controller
from aedrover.nav.base import run_episode
from aedrover.sim import FAMILIES, AEDRoverEnv, VehicleParams, sample_scenario
from aedrover.sim.pedestrians import PedestrianCrowd, SFMParams
from aedrover.sim.scenario import PedSpec

VEH = VehicleParams.optimized()


@pytest.fixture(scope="module")
def env():
    return AEDRoverEnv(veh=VEH, obs_mode="dict")


def test_scenarios_are_deterministic_and_cover_all_families():
    for fam in FAMILIES:
        a, b = sample_scenario(fam, 7), sample_scenario(fam, 7)
        assert a.to_dict() == b.to_dict()
        assert sample_scenario(fam, 8).to_dict() != a.to_dict()
    with pytest.raises(ValueError):
        sample_scenario("nonsense", 0)


def test_family_contents():
    assert not sample_scenario("flat_clear", 1).peds
    assert sample_scenario("kerb", 1).has_kerb and not sample_scenario("kerb", 1).peds
    assert sample_scenario("crowded", 1).peds and not sample_scenario("crowded", 1).has_kerb
    m = sample_scenario("mixed", 1)
    assert m.has_kerb and m.peds
    assert sample_scenario("slippery", 1).tyre_mu < 0.6


def test_observation_and_action_spaces(env):
    obs, _ = env.reset(seed=3, options={"family": "mixed"}) if env.obs_mode == "vector" else (None, None)
    env.obs_mode = "vector"
    obs, _ = env.reset(seed=3, options={"family": "mixed"})
    assert obs.shape == env.observation_space.shape and np.isfinite(obs).all()
    assert env.action_space.shape == (2,)
    env.obs_mode = "dict"


def test_episode_is_deterministic(env):
    r1 = run_episode(env, make_controller("dwa"), SafetyFilter(), seed=1003, options={"family": "kerb"})
    r2 = run_episode(env, make_controller("dwa"), SafetyFilter(), seed=1003, options={"family": "kerb"})
    for k in ("time_s", "peak_shock_g", "path_m", "energy_wh", "outcome"):
        assert r1[k] == r2[k], k


def test_flat_clear_is_solved_by_the_classical_stack(env):
    for seed in range(1000, 1004):
        ep = run_episode(env, make_controller("dwa"), SafetyFilter(), seed=seed, options={"family": "flat_clear"})
        assert ep["outcome"] == "goal"
        assert ep["peak_shock_g"] < 1.5


def test_kerb_crossing_with_cooptimised_vehicle_stays_within_budget(env):
    ep = run_episode(env, make_controller("dwa"), SafetyFilter(), seed=1003, options={"family": "kerb"})
    assert ep["outcome"] == "goal" and ep["peak_shock_g"] < 3.5


def test_clearance_is_negative_on_overlap_and_positive_when_apart(env):
    env.reset(seed=5, options={"family": "crowded"})
    x, y, _ = env.rover.pos
    env.crowd.pos[0] = (x + 0.3, y)              # inside the footprint
    assert env._clearance() < 0
    env.crowd.pos[0] = (x + 3.0, y + 1.0)
    for i in range(1, env.crowd.n):
        env.crowd.pos[i] = (x + 6.0, y - 1.2)
    for ob in env.scenario.obstacles:
        ob.x, ob.y = x + 8.0, y
    assert env._clearance() > 0.5


def test_pedestrians_do_not_respawn_next_to_the_rover():
    crowd = PedestrianCrowd(SFMParams())
    spec = PedSpec(start=(10.0, 0.0), goal=(10.5, 0.0), v_des=1.4, aware=True, z_surface=0.0, origin=(2.0, 0.0))
    crowd.reset([spec], np.random.default_rng(0))
    robot = np.array([2.5, 0.0])                   # rover sits right at the respawn point
    for _ in range(400):
        crowd.step(0.02, robot, np.zeros(2))
        assert np.linalg.norm(crowd.pos[0] - robot) > 1.0
    far = np.array([30.0, 0.0])
    for _ in range(400):
        crowd.step(0.02, far, np.zeros(2))
    assert np.linalg.norm(crowd.pos[0] - np.array([2.0, 0.0])) < 4.0   # respawned once the entry was clear


def _obs(lidar=None, tracks=None):
    ang = np.radians(np.linspace(-120, 120, 73))
    return {"lidar": np.full(73, 10.0) if lidar is None else lidar, "lidar_angles": ang,
            "tracks": np.zeros((4, 5)) if tracks is None else tracks}


def test_safety_filter_is_transparent_in_free_space_and_stops_before_an_obstacle():
    sf = SafetyFilter()
    assert sf.v_limit(_obs(), 0.0) > 5.0
    lid = np.full(73, 10.0)
    lid[36] = 1.2                                   # something 1.2 m dead ahead
    lim = sf.v_limit(_obs(lid), 0.0)
    assert 0.5 < lim < 1.4                          # v t_react + v^2/(2 a) ~= 0.5 m of free arc
    lid[36] = 0.5                                   # already inside the front margin
    assert sf.v_limit(_obs(lid), 0.0) == 0.0


def test_safety_filter_ignores_an_obstacle_beside_the_path_but_not_when_steering_into_it():
    sf = SafetyFilter()
    ang = np.radians(np.linspace(-120, 120, 73))
    lid = np.full(73, 10.0)
    k = int(np.argmin(np.abs(ang - np.arctan2(-0.9, 1.5))))   # obstacle 0.9 m to the right, 1.5 m ahead
    lid[k] = float(np.hypot(1.5, 0.9))
    assert sf.v_limit(_obs(lid), 0.0) > 3.0                    # straight ahead clears it
    assert sf.v_limit(_obs(lid), -0.5) < sf.v_limit(_obs(lid), 0.0)   # curving right sweeps into it


def test_safety_filter_never_accelerates_and_counts_interventions():
    sf = SafetyFilter()
    lid = np.full(73, 10.0)
    lid[36] = 1.0
    v, d = sf(_obs(lid), 2.0, 0.1)
    assert v < 2.0 and d == 0.1 and sf.n_interventions == 1
    v2, _ = sf(_obs(), 1.0, 0.0)
    assert v2 == 1.0
