"""Evaluation-time controller wrapping a trained PPO policy (registered as ``ppo``)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from ..nav.base import register
from .rl_env import action_to_command


class PPOController:
    def __init__(self, path: str = "checkpoints/ppo_shielded", weights: str = "final", v_max: float = 2.6,
                 deterministic: bool = True, name: str = "ppo"):
        import torch
        from stable_baselines3 import PPO

        torch.set_num_threads(1)          # evaluation workers run one process per core
        self.name = name
        root = Path(path)
        self.model = PPO.load(str(root / weights), device="cpu")
        stats = root / "vecnormalize.pkl" if weights == "final" else root / (weights.replace("ppo_", "ppo_vecnormalize_", 1) + ".pkl")
        with open(stats, "rb") as fh:
            vn = pickle.load(fh)
        self.mean, self.var = vn.obs_rms.mean.astype(np.float64), vn.obs_rms.var.astype(np.float64)
        self.eps, self.clip = vn.epsilon, vn.clip_obs
        self.v_max, self.det = v_max, deterministic
        self.env = None

    def reset(self, env) -> None:
        self.env = env

    def act(self, obs: dict) -> tuple[float, float]:
        vec = self.env._obs_vector().astype(np.float64)
        vec = np.clip((vec - self.mean) / np.sqrt(self.var + self.eps), -self.clip, self.clip)
        action, _ = self.model.predict(vec.astype(np.float32), deterministic=self.det)
        return action_to_command(action, self.v_max)


@register("ppo")
def _make_ppo(**kw):
    return PPOController(**kw)
