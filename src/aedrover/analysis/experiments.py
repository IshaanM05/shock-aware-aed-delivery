"""Episode runner shared by all benchmark experiments (parallel, cached envs, deterministic).

``run_job`` is module-level and picklable. Each worker process keeps one compiled environment per
distinct configuration, so a job costs only an episode, not a model compile.
"""

from __future__ import annotations

import functools
import time
from dataclasses import dataclass

from ..control.safety_filter import SafetyFilter, SafetyParams
from ..nav.base import make_controller, run_episode
from ..sim.env import AEDRoverEnv


@dataclass(frozen=True)
class Job:
    controller: str
    family: str
    seed: int
    shield: bool = True
    controller_kwargs: tuple = ()          # tuple of (key, value) pairs (hashable)
    safety_kwargs: tuple = ()
    env_kwargs: tuple = ()
    scenario_kwargs: tuple = ()            # e.g. (('kerb_range', (0.15, 0.19)),) for out-of-distribution runs
    tag: str = ""


@functools.lru_cache(maxsize=8)
def _env(env_kwargs: tuple) -> AEDRoverEnv:
    return AEDRoverEnv(obs_mode="dict", **dict(env_kwargs))


@functools.lru_cache(maxsize=16)
def _controller(name: str, kwargs: tuple):
    return make_controller(name, **dict(kwargs))


def run_job(job: Job) -> dict:
    env = _env(job.env_kwargs)
    ctrl = _controller(job.controller, job.controller_kwargs)
    shield = SafetyFilter(SafetyParams(**dict(job.safety_kwargs))) if job.shield else None
    t0 = time.perf_counter()
    ep = dict(run_episode(env, ctrl, shield, seed=job.seed, options={"family": job.family, "scenario_kwargs": dict(job.scenario_kwargs)}))
    ep.pop("trajectory", None)
    ep.update(controller=job.controller, family=job.family, seed=job.seed, shield=job.shield,
              tag=job.tag, wall_s=time.perf_counter() - t0,
              interventions=(shield.n_interventions if shield else 0),
              intervention_frac=(shield.n_interventions / max(shield.n_calls, 1) if shield else 0.0))
    return ep


def make_jobs(controllers, families, seeds, **kw) -> list[Job]:
    """``controllers`` items are a name or ``(name, kwargs_dict)``."""
    jobs = []
    for c in controllers:
        name, kwargs = (c, {}) if isinstance(c, str) else c
        for f in families:
            for s in seeds:
                jobs.append(Job(name, f, s, controller_kwargs=tuple(sorted(kwargs.items())), **kw))
    return jobs
