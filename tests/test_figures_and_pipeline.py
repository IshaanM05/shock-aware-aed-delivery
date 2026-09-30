"""Smoke tests for the figure functions and the pipeline wiring (synthetic data, no simulation)."""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aedrover.analysis import figures as F
from aedrover.analysis import figures_results as FR
from aedrover.analysis.experiments import SPEED_KEY, controller_spec

ROOT = Path(__file__).resolve().parents[1]
CTRLS = ["pure_pursuit", "apf", "dwa", "mppi", "ppo"]
FAMS = ["flat_clear", "kerb", "crowded", "mixed", "slippery"]


def _episodes(n=12, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for c in CTRLS:
        for f in FAMS:
            for s in range(n):
                ok = rng.random() < 0.8
                rows.append({"controller": c, "family": f, "seed": s, "success": ok, "outcome": "goal" if ok else "stall",
                             "time_s": rng.normal(25, 3), "peak_shock_g": rng.normal(2.0, 0.5), "min_clearance_m": 0.4,
                             "energy_wh": 4.0, "intervention_frac": 0.1, "shock_over_budget": False})
    return pd.DataFrame(rows)


def test_fixed_speed_cap_reaches_every_controller():
    for name, key in SPEED_KEY.items():
        _, kw = controller_spec(name, 1.7)
        assert kw[key] == pytest.approx(1.7)
    assert controller_spec("ppo")[1]["path"] == "checkpoints/ppo_shielded"
    assert controller_spec("dwa", None)[1] == {}


def test_benchmark_and_ood_figures_render(tmp_path):
    df = _episodes()
    assert FR.fig_benchmark(df, tmp_path / "b.png", title="t").stat().st_size > 10_000
    rows = [{"controller": c, "condition": k, "safe_delivery": 0.6, "safe_lo": 0.5, "safe_hi": 0.7}
            for c in CTRLS for k in ("id", "ood_kerb", "ood_mu", "ood_both")]
    assert FR.fig_ood(pd.DataFrame(rows), tmp_path / "o.png").exists()


def test_clinical_figures_render(tmp_path):
    rows = []
    for dens in ("osm_lower_bound", "assumed_4_per_km"):
        for r in (250, 500, 1000):
            for mode, ctrl in (("ambulance", "dwa"), ("drone", "dwa"), ("rover", "dwa"), ("rover", "ppo")):
                rows.append({"controller": ctrl, "density": dens, "radius_m": r, "model": "larsen1993", "mode": mode,
                             "mean_survival": 0.2 + 0.1 * (mode != "ambulance") - r / 20000, "survival_ci_low": 0.15,
                             "survival_ci_high": 0.3})
    assert FR.fig_survival(pd.DataFrame(rows), tmp_path / "s.png").exists()
    pol = pd.DataFrame([{"controller": "dwa", "policy": p, "p_drone": q, "mean_survival": 0.2 + 0.05 * q}
                        for p in ("ambulance", "rover", "drone", "hybrid", "both") for q in (0.0, 0.5, 1.0)])
    assert FR.fig_policies(pol, tmp_path / "p.png", "dwa").exists()


def test_speed_cap_and_architecture_figures_render(tmp_path):
    rows = [{"controller": c, "condition": f"cap_{cap}", "success": True, "safe": True, "time_s": 40 / cap}
            for c in ("dwa", "ppo") for cap in (0.8, 1.4, 2.0, 2.6) for _ in range(5)]
    assert FR.fig_speed_cap(pd.DataFrame(rows), tmp_path / "c.png").exists()
    assert FR.fig_architecture(tmp_path / "a.png").exists()


def test_shared_style_and_palette_are_consistent():
    assert F.CONTROLLER_COLOURS["dwa"] == F.SERIES[0]                # colour follows the entity, not its rank
    assert len({F.CONTROLLER_COLOURS[c] for c in CTRLS}) == len(CTRLS)


def test_pipeline_steps_reference_real_scripts_with_unique_markers():
    spec = importlib.util.spec_from_file_location("run_pipeline", ROOT / "scripts" / "run_pipeline.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_pipeline"] = mod
    spec.loader.exec_module(mod)
    args = argparse.Namespace(tag="standard", n_select=1, n_tune=1, n_bench=1, n_ood=1, n_abl=1, mppi_fraction=0.5)
    steps = mod.steps(args)
    markers = [m for _, _, m in steps]
    assert len(set(markers)) == len(markers)
    for _name, cmd, _ in steps:
        assert (ROOT / cmd[1]).exists(), cmd[1]
