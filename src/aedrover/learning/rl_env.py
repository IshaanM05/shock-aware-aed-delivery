"""RL wrapper around ``AEDRoverEnv``.

* Domain randomisation: every reset draws a scenario family from a mixture and randomises kerb
  height, tyre friction, payload mass, ramps, pedestrian and obstacle placement (wide ranges, see
  ``TRAIN_RANGES``); the test-time OOD study samples outside these ranges.
* Decision rate: 10 Hz (action repeat 5 of the 50 Hz environment), which shortens the credit
  assignment horizon five-fold.
* Action: two numbers in [-1, 1] mapped to speed and steering. Optionally filtered by the same
  ``SafetyFilter`` that wraps every other controller (shielded RL).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from ..control.safety_filter import SafetyFilter, SafetyParams
from ..sim.env import AEDRoverEnv, RewardConfig
from ..sim.scenario import FAMILIES, sample_scenario
from ..sim.vehicle_mjcf import VehicleParams

TRAIN_RANGES = {"kerb_range": (0.06, 0.16), "mu_range": (0.5, 1.2)}
OOD_RANGES = {"kerb_range": (0.15, 0.19), "mu_range": (0.3, 0.5)}
DEFAULT_FAMILY_PROBS = {"flat_clear": 0.05, "kerb": 0.30, "crowded": 0.20, "mixed": 0.30, "slippery": 0.15}
V_MID, V_SPAN = 1.3, 1.3          # speed = V_MID + V_SPAN * a0, clipped to [0, v_max]
DELTA_SCALE = 0.5


def action_to_command(a: np.ndarray, v_max: float = 2.6) -> tuple[float, float]:
    a = np.clip(a, -1.0, 1.0)
    return float(np.clip(V_MID + V_SPAN * a[0], 0.0, v_max)), float(DELTA_SCALE * a[1])


def command_to_action(v: float, delta: float) -> np.ndarray:
    return np.array([(v - V_MID) / V_SPAN, delta / DELTA_SCALE], dtype=np.float32)


class RLRoverEnv(gym.Wrapper):
    def __init__(self, veh: VehicleParams | None = None, *, repeat: int = 5, shield: bool = True,
                 family_probs: dict | None = None, ranges: dict | None = None, seed: int = 0,
                 max_time: float = 60.0, budget_g: float = 2.5, v_max: float = 2.6,
                 safety: SafetyParams | None = None):
        env = AEDRoverEnv(veh=veh or VehicleParams.optimized(), obs_mode="vector", max_time=max_time,
                          budget_g=budget_g, reward=RewardConfig(), log_every=5, ped_every=2)
        super().__init__(env)
        self.repeat, self.v_max = repeat, v_max
        self.shield = SafetyFilter(safety) if shield else None
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
        probs = family_probs or DEFAULT_FAMILY_PROBS
        self._fams = list(probs)
        self._p = np.array([probs[f] for f in self._fams], dtype=float)
        self._p /= self._p.sum()
        self._ranges = ranges or TRAIN_RANGES
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = dict(options or {})
        if "scenario" not in options:
            fam = options.get("family") or str(self._rng.choice(self._fams, p=self._p))
            sd = int(self._rng.integers(2**31 - 1))
            spec = self.env.world.spec
            options["scenario"] = sample_scenario(fam, sd, n_ped_max=spec.n_ped, n_obs_max=spec.n_obstacle,
                                                  **self._ranges)
        if self.shield is not None:
            self.shield.reset()
        return self.env.reset(options=options)

    def step(self, action):
        v, d = action_to_command(np.asarray(action), self.v_max)
        total, info = 0.0, {}
        # the shield runs once per decision (10 Hz; its 0.1 s reaction allowance covers the hold),
        # perception is rebuilt only at the end of the repeat
        cv, cd = (v, d) if self.shield is None else self.shield(self.env.obs, v, d)
        cmd = np.array([cv, cd])
        for i in range(self.repeat):
            obs, r, term, trunc, info = self.env.step(cmd, want_obs=(i == self.repeat - 1))
            total += r
            if term or trunc:
                break
        if "episode" in info:                      # SB3's VecMonitor overwrites 'episode'
            info["ep_metrics"] = info.pop("episode")
        return obs, total, term, trunc, info


def make_env_fn(rank: int, seed: int, **kw):
    def _init():
        return RLRoverEnv(seed=seed + 1000 * rank, **kw)
    return _init


_ = FAMILIES
