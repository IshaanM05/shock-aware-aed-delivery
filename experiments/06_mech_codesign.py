"""Experiment 06: mechanical co-design of the rover (RQ1).

Searches wheel radius, suspension spring/damper, payload isolator and motor peak torque with
parallel differential evolution, comparing against the nominal design. Writes
results/codesign.json and configs/vehicle_optimized.yaml.

    python experiments/06_mech_codesign.py [--maxiter 25] [--popsize 6]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml
from scipy.optimize import differential_evolution

from aedrover.parallel import default_workers
from aedrover.sim.codesign import BOUNDS, KEYS, NOMINAL, evaluate_design, objective, vec_to_design

ROOT = Path(__file__).resolve().parents[1]


def show(label: str, ev) -> None:
    print(f"{label}: objective {ev.objective:.2f}  within-budget {ev.frac_within_budget:.0%}  "
          f"climb-fails {ev.n_fail}")
    for c in ev.up:
        v = "none" if c["v_min"] is None else f"{c['v_min']:.1f}"
        print(f"    up   h={c['h']:.2f} mu={c['mu']:.2f}  v_min={v:>4} m/s  shock={c['shock']:.2f} g")
    for c in ev.down:
        print(f"    down h={c['h']:.2f}                         shock={c['shock']:.2f} g")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--maxiter", type=int, default=25)
    ap.add_argument("--popsize", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    nominal = evaluate_design(NOMINAL)
    show("NOMINAL", nominal)

    history: list[float] = []

    def cb(xk, convergence=None):
        history.append(float(objective(xk)))
        print(f"  gen {len(history):2d}: best objective {history[-1]:.3f}", flush=True)

    t0 = time.perf_counter()
    res = differential_evolution(objective, [BOUNDS[k] for k in KEYS], maxiter=args.maxiter,
                                 popsize=args.popsize, seed=args.seed, tol=1e-4, mutation=(0.5, 1.0),
                                 recombination=0.7, workers=default_workers(), updating="deferred",
                                 callback=cb, polish=False)
    print(f"search took {time.perf_counter() - t0:.0f}s, {res.nfev} evaluations")
    best = vec_to_design(res.x)
    ev = evaluate_design(best)
    show("OPTIMISED", ev)
    print("design:", json.dumps({k: round(v, 3) for k, v in best.items()}))

    out = {"nominal": {"design": NOMINAL, "eval": nominal.__dict__},
           "optimised": {"design": best, "eval": ev.__dict__}, "history": history,
           "nfev": int(res.nfev), "bounds": {k: list(v) for k, v in BOUNDS.items()}}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "codesign.json").write_text(json.dumps(out, indent=2, default=float),
                                                    encoding="utf-8")
    cfg = {k: round(float(v), 4) for k, v in best.items()}
    (ROOT / "configs" / "vehicle_optimized.yaml").write_text(
        "# Output of experiments/06_mech_codesign.py (differential evolution, see results/codesign.json)\n"
        + yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print("wrote results/codesign.json, configs/vehicle_optimized.yaml")


if __name__ == "__main__":
    main()
