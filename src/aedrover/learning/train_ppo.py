"""PPO training (Stable-Baselines3, CPU) with vectorised domain-randomised environments.

    python -m aedrover.learning.train_ppo --steps 10000000 --n-envs 24 --out checkpoints/ppo_shielded
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize

from .rl_env import make_env_fn


class ProgressLog(BaseCallback):
    """Prints a compact progress line every ~rollout window and logs outcome rates."""

    def __init__(self, every: int = 20):
        super().__init__()
        self.every, self.t0, self._n = every, time.perf_counter(), 0
        self.outcomes: list[str] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            ep = info.get("ep_metrics")
            if isinstance(ep, dict) and "outcome" in ep:
                self.outcomes.append(ep["outcome"])
        return True

    def _on_rollout_end(self) -> None:
        self._n += 1
        if self._n % self.every == 0:
            recent = self.outcomes[-200:]
            rate = {k: recent.count(k) / max(len(recent), 1) for k in ("goal", "collision", "stall", "timeout")}
            el = time.perf_counter() - self.t0
            print(f"[{self.num_timesteps / 1e6:5.2f}M steps  {self.num_timesteps / el:6.0f} fps]  "
                  f"goal {rate['goal']:.2f} collision {rate['collision']:.2f} stall {rate['stall']:.2f}",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5_000_000)
    ap.add_argument("--n-envs", type=int, default=24)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=1536)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-shield", action="store_true")
    ap.add_argument("--torch-threads", type=int, default=4)
    ap.add_argument("--out", default="checkpoints/ppo_shielded")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--ckpt-every", type=int, default=500_000, help="checkpoint period in environment decisions")
    args = ap.parse_args()

    torch.set_num_threads(args.torch_threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    fns = [make_env_fn(i, args.seed, shield=not args.no_shield) for i in range(args.n_envs)]
    venv = VecNormalize(VecMonitor(SubprocVecEnv(fns, start_method="spawn")), norm_obs=True,
                        norm_reward=True, clip_obs=10.0, gamma=0.99)
    policy_kwargs = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh)
    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu")
    else:
        model = PPO("MlpPolicy", venv, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_epochs=args.epochs, learning_rate=lambda p: args.lr * (0.1 + 0.9 * p), gamma=0.99,
                    gae_lambda=0.95, clip_range=0.2, ent_coef=0.003, vf_coef=0.5, max_grad_norm=0.5,
                    policy_kwargs=policy_kwargs, seed=args.seed, device="cpu",
                    tensorboard_log=str(out / "tb"), verbose=0)
    ckpt = CheckpointCallback(save_freq=max(args.ckpt_every // args.n_envs, 1), save_path=str(out), name_prefix="ppo",
                              save_vecnormalize=True)      # normalisation stats are needed to evaluate a checkpoint
    t0 = time.perf_counter()
    model.learn(total_timesteps=args.steps, callback=[ProgressLog(), ckpt], reset_num_timesteps=not args.resume)
    model.save(out / "final")
    venv.save(str(out / "vecnormalize.pkl"))
    meta = {"steps": int(model.num_timesteps), "wall_s": time.perf_counter() - t0, "n_envs": args.n_envs,
            "shield": not args.no_shield, "seed": args.seed, "torch": torch.__version__}
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("saved", out, meta)
    _ = np


if __name__ == "__main__":
    main()
