"""Controller interface, registry and the shared episode loop.

A controller maps the shared observation dict (``env.obs``) to a command ``(v, delta)``. Every
controller, classical or learned, is optionally wrapped by the same ``SafetyFilter`` so that
comparisons hold identical safety constraints.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np


class Controller(Protocol):
    name: str

    def reset(self, env) -> None: ...
    def act(self, obs: dict) -> tuple[float, float]: ...


_REGISTRY: dict[str, Callable[..., Controller]] = {}


def register(name: str):
    def deco(factory):
        _REGISTRY[name] = factory
        return factory
    return deco


def registered_controllers() -> list[str]:
    _load_builtin()
    return sorted(_REGISTRY)


def make_controller(name: str, **kwargs) -> Controller:
    _load_builtin()
    if name not in _REGISTRY:
        raise KeyError(f"unknown controller {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def _load_builtin() -> None:
    from . import apf, dwa, pure_pursuit  # noqa: F401  (registration side effects)
    try:
        from . import mppi  # noqa: F401
    except ImportError:
        pass
    try:                                   # needs the optional [rl] extra (torch, stable-baselines3)
        from ..learning import ppo_controller  # noqa: F401
    except ImportError:
        pass


def run_episode(env, controller: Controller, shield=None, seed: int | None = None,
                options: dict | None = None, record: bool = False) -> dict:
    """Run one episode; returns ``info['episode']`` (plus a trajectory if ``record``)."""
    env.reset(seed=seed, options=options)
    controller.reset(env)
    if shield is not None:
        shield.reset()
    traj: list[tuple] = []
    info: dict = {}
    while True:
        obs = env.obs
        v, d = controller.act(obs)
        if shield is not None:
            v, d = shield(obs, v, d)
        _, _, term, trunc, info = env.step(np.array([v, d]))
        if record:
            traj.append((env._t, obs["x"], obs["y"], obs["yaw"], v, d, info["shock_g"], info["clearance"]))
        if term or trunc:
            break
    ep = info["episode"]
    if record:
        ep = {**ep, "trajectory": np.array(traj)}
    return ep
