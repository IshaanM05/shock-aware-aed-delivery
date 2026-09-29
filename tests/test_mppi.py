"""MPPI regression tests, including the mujoco.rollout mocap pitfall."""

import mujoco
import numpy as np
import pytest
from mujoco import rollout

from aedrover.nav import make_controller
from aedrover.nav.mppi import _SPEC
from aedrover.sim import AEDRoverEnv, VehicleParams
from aedrover.sim.scenario import sample_scenario

VEH = VehicleParams.optimized()


def _env_with_kerb():
    env = AEDRoverEnv(veh=VEH, obs_mode="dict")
    sc = sample_scenario("kerb", 1003, n_ped_max=env.world.spec.n_ped, n_obs_max=env.world.spec.n_obstacle)
    env.reset(seed=1003, options={"scenario": sc})
    return env, sc


def test_rollout_resets_mocap_unless_passed_as_control():
    """Documenting the pitfall: without control_spec the kerb vanishes from mujoco.rollout."""
    env, sc = _env_with_kerb()
    ctrl = make_controller("mppi", K=4, H=4)
    ctrl.reset(env)
    m = ctrl.model
    w = env.world
    # put the rover 2.5 m before the kerb at 1.6 m/s, in the planner's own model
    d = ctrl.datas[0]
    d.qpos[:] = w.data.qpos
    d.qvel[:] = 0.0
    root = 0
    d.qpos[root:root + 3] = (sc.x_up - 2.5, 0.0, VEH.nominal_height)
    d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    x0 = np.concatenate([[0.0], d.qpos, d.qvel])[None]
    wheel, steer = env.rover.allocate(1.6, 0.0)
    act = np.concatenate([wheel, steer])
    T = 400

    naive = np.tile(act, (1, T, 1))
    st_blind, _ = rollout.rollout(m, ctrl.datas, x0, naive)
    nu, nm = m.nu, ctrl.nm
    full = np.zeros((1, T, nu + 7 * nm))
    full[0, :, :nu] = act
    full[0, :, nu:] = ctrl._mocap_block(env.obs, T)
    st_ok, _ = rollout.rollout(m, ctrl.datas, x0, full, control_spec=_SPEC)
    z_blind, z_ok = st_blind[0, -1, 3], st_ok[0, -1, 3]         # qpos z is state index 1 + 2
    assert z_blind < VEH.nominal_height + 0.03                  # drove "through" the missing kerb
    assert z_ok > VEH.nominal_height + 0.7 * sc.kerb_h          # climbed the real one
    assert mujoco.mj_stateSize(m, _SPEC) == nu + 7 * nm


def test_allocation_matches_rover_allocation():
    env, _ = _env_with_kerb()
    ctrl = make_controller("mppi", K=4, H=4)
    ctrl.reset(env)
    for v, delta in [(1.0, 0.0), (1.5, 0.3), (0.8, -0.4), (0.0, 0.2)]:
        a = ctrl._allocate(np.array([[v]]), np.array([[delta]]))[0, 0]
        wheel, steer = env.rover.allocate(v, delta)
        np.testing.assert_allclose(a[:4], wheel, atol=1e-9)
        np.testing.assert_allclose(a[4:], steer, atol=1e-5)


def test_predicted_pedestrians_enter_the_mocap_block():
    env = AEDRoverEnv(veh=VEH, obs_mode="dict")
    env.reset(seed=1001, options={"family": "crowded"})
    ctrl = make_controller("mppi", K=4, H=4)
    ctrl.reset(env)
    obs = dict(env.obs)
    tracks = np.zeros((4, 5))
    tracks[0] = (3.0, 0.5, -1.0, 0.0, 1.0)             # ped 3 m ahead walking toward the rover
    obs["tracks"] = tracks
    T = 60
    block = ctrl._mocap_block(obs, T)
    nm = ctrl.nm
    pos = block[:, :3 * nm].reshape(T, nm, 3)
    slot = int(ctrl._pw.m_ped[0])
    x_first, x_last = pos[0, slot, 0], pos[-1, slot, 0]
    assert x_last < x_first - 0.4                        # moved toward the rover over 0.6 s


@pytest.mark.slow
def test_mppi_completes_a_flat_episode():
    env = AEDRoverEnv(veh=VEH, obs_mode="dict")
    ctrl = make_controller("mppi", K=48, H=14, nthread=2)
    env.reset(seed=1, options={"family": "flat_clear"})
    ctrl.reset(env)
    for _ in range(3000):
        v, d = ctrl.act(env.obs)
        _, _, term, trunc, info = env.step(np.array([v, d]))
        if term or trunc:
            break
    assert info["episode"]["outcome"] == "goal"
