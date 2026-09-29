"""Dispatch policies over the delivery modes (ambulance, rover, drone) and their expected survival.

Every policy dispatches the ambulance; the question is which AED delivery device is sent *in
addition* (the first shock is the earliest arrival among what was sent):

* ``ambulance``  : nothing extra;
* ``rover``      : the ground rover, which may fail or damage the AED (``rover_travel_min = inf``);
* ``drone``      : the drone, which flies only when the weather allows (probability ``p_drone``);
* ``hybrid``     : the drone when it can fly, otherwise the rover;
* ``both``       : every device that can go (upper bound on what parallel dispatch buys).

Arrests are paired across policies (same collapse, same ambulance delay), so differences are
low-variance. ``p_drone`` is a parameter to sweep: no climate claim is made for any city.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decision import ModeSamples, ScenarioParams, evaluate_modes, mode_times
from .survival import SurvivalModel

POLICIES = ("ambulance", "rover", "drone", "hybrid", "both")


def policy_samples(params: ScenarioParams, *, rover_travel_min: np.ndarray, p_drone: float, n: int, seed: int,
                   drone_launch_min: float | None = None) -> dict[str, ModeSamples]:
    """Paired first-shock times for each policy.

    ``rover_travel_min`` (length ``n``) holds simulated rover travel times in minutes, with
    ``inf`` for routes where the rover failed or delivered a damaged AED.
    """
    if not 0.0 <= p_drone <= 1.0:
        raise ValueError("p_drone must lie in [0, 1]")
    rng = np.random.default_rng(seed)
    base = mode_times(n, rng, params, parallel=True, rover_travel_min=rover_travel_min)
    # deterministic-shape parallel draw of the same arrests: drone/rover/ambulance share t_cpr, t_acls
    amb, rov, dro = base["ambulance"], base["rover"], base["drone"]
    can_fly = np.random.default_rng(seed + 1).random(n) < p_drone
    t_amb = amb.t_defib
    t_drone_or_amb = np.where(can_fly, dro.t_defib, t_amb)
    t_rover = rov.t_defib
    hybrid = np.where(can_fly, dro.t_defib, t_rover)
    out = {
        "ambulance": amb,
        "rover": rov,
        "drone": ModeSamples(t_drone_or_amb, amb.t_cpr, amb.t_acls),
        "hybrid": ModeSamples(np.minimum(hybrid, t_amb), amb.t_cpr, amb.t_acls),
        "both": ModeSamples(np.minimum(np.minimum(t_rover, t_drone_or_amb), t_amb), amb.t_cpr, amb.t_acls),
    }
    _ = drone_launch_min
    return out


def evaluate_policies(model: SurvivalModel, params: ScenarioParams, *, rover_travel_min: np.ndarray,
                      p_drone: float, seed: int = 0, n_resamples: int = 1000) -> pd.DataFrame:
    """Expected survival, gain over ambulance-only, CI and how often each policy shocks first."""
    n = int(np.asarray(rover_travel_min).size)
    samples = policy_samples(params, rover_travel_min=np.asarray(rover_travel_min, dtype=float), p_drone=p_drone,
                             n=n, seed=seed)
    df = evaluate_modes(samples, model, baseline="ambulance", seed=seed, n_resamples=n_resamples)
    return df.rename(columns={"mode": "policy"})
