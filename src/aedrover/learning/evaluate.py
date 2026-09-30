"""Held-out evaluation of a trained policy against reference controllers (paired seeds)."""

from __future__ import annotations

import pandas as pd

from ..analysis.experiments import make_jobs, run_job
from ..parallel import pmap
from ..sim.scenario import FAMILIES
from ..sim.vehicle_mjcf import VehicleParams


def evaluate_policy(path: str, *, n: int = 40, seed0: int = 9000, families=FAMILIES, references=("dwa",),
                    vehicle: str = "optimized", workers: int | None = None, both_shield_modes: bool = True,
                    **controller_kwargs) -> pd.DataFrame:
    """Run ``ppo`` (from ``path``) and the reference controllers on the same held-out seeds.

    The policy was trained with the safety filter in the loop, so it is evaluated with the filter
    on; ``both_shield_modes`` also evaluates it with the filter off (label ``ppo_noshield``) to show
    how much the policy relies on it. Seeds default to 9000+, disjoint from anything used in
    training or in the other experiments.
    """
    env_kwargs = (("veh", VehicleParams.by_name(vehicle)),)
    seeds = range(seed0, seed0 + n)
    kw = tuple(sorted({"path": path, **controller_kwargs}.items()))
    jobs = make_jobs([("ppo", dict(kw))], families, seeds, shield=True, env_kwargs=env_kwargs, tag="ppo_eval")
    if both_shield_modes:
        jobs += make_jobs([("ppo", dict(kw))], families, seeds, shield=False, env_kwargs=env_kwargs,
                          tag="ppo_eval_noshield")
    jobs += make_jobs(list(references), families, seeds, shield=True, env_kwargs=env_kwargs, tag="ppo_eval")
    df = pd.DataFrame(pmap(run_job, jobs, workers=workers, chunksize=4, desc="ppo-eval"))
    df["vehicle"] = vehicle
    df.loc[(df.controller == "ppo") & (~df["shield"]), "controller"] = "ppo_noshield"
    return df
