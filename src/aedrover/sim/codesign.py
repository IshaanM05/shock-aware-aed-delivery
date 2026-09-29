"""Mechanical co-design: choose wheel radius, suspension, payload isolator and motor size.

Objective (minimised): the payload shock the rover suffers when climbing kerbs at the *lowest speed
that climbs*, plus a margin, averaged over a set of (height, friction) conditions, with kerb-down
shocks added. A condition the design cannot climb below the top of the speed scan is charged
``FAIL_PENALTY`` g.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curb_study import KerbTrial, run_kerb_trial

BOUNDS = {
    "wheel_radius": (0.13, 0.22),
    "susp_k": (2500.0, 8000.0),
    "susp_c": (150.0, 700.0),
    "iso_kz": (600.0, 4000.0),
    "iso_cz": (40.0, 400.0),
    "motor_peak_torque": (15.0, 45.0),
}
KEYS = tuple(BOUNDS)
NOMINAL = {
    "wheel_radius": 0.15, "susp_k": 4500.0, "susp_c": 350.0, "iso_kz": 2200.0, "iso_cz": 180.0,
    "motor_peak_torque": 25.0,
}

UP_CONDITIONS = tuple((h, mu) for h in (0.10, 0.12, 0.14) for mu in (0.6, 0.85))
DOWN_CONDITIONS = ((0.12, 1.0), (0.16, 1.0))
SPEEDS = tuple(float(v) for v in np.arange(0.6, 2.81, 0.2).round(2))
V_MARGIN = 0.2
FAIL_PENALTY = 8.0


@dataclass
class DesignEval:
    objective: float
    up: list          # per condition: dict(h, mu, v_min, shock)
    down: list
    n_fail: int
    frac_within_budget: float


def _with(t: KerbTrial, **kw) -> KerbTrial:
    d = dict(t.__dict__)
    d.update(kw)
    return KerbTrial(**d)


def evaluate_design(design: dict, budget_g: float = 3.0) -> DesignEval:
    up, fails = [], 0
    for h, mu in UP_CONDITIONS:
        base = KerbTrial(direction="up", kerb_h=h, mu=mu, **design)
        v_min, shock = None, FAIL_PENALTY
        for v in SPEEDS:
            if run_kerb_trial(_with(base, speed=v))["success"]:
                v_min = v
                break
        if v_min is not None:
            r2 = run_kerb_trial(_with(base, speed=v_min + V_MARGIN))
            shock = r2["peak_g"] if r2["success"] else FAIL_PENALTY
            fails += 0 if r2["success"] else 1
        else:
            fails += 1
        up.append({"h": h, "mu": mu, "v_min": v_min, "shock": float(shock)})
    down = []
    for h, mu in DOWN_CONDITIONS:
        r = run_kerb_trial(KerbTrial(direction="down", kerb_h=h, mu=mu, speed=0.8, **design))
        down.append({"h": h, "mu": mu, "shock": float(r["peak_g"]) if r["success"] else FAIL_PENALTY})
    mean_up = float(np.mean([c["shock"] for c in up]))
    mean_down = float(np.mean([c["shock"] for c in down]))
    allc = np.array([c["shock"] for c in up] + [c["shock"] for c in down])
    return DesignEval(objective=mean_up + 0.5 * mean_down, up=up, down=down, n_fail=fails,
                      frac_within_budget=float(np.mean(allc <= budget_g)))


def vec_to_design(x) -> dict:
    return {k: float(v) for k, v in zip(KEYS, x, strict=True)}


def objective(x) -> float:
    """Picklable scalar objective for scipy's differential_evolution(workers=N)."""
    return evaluate_design(vec_to_design(x)).objective
