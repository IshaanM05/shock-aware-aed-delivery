"""Experiment 05: from simulated rover performance to clinical outcome (RQ3).

Pipeline
--------
1. Per-segment MuJoCo episodes of a controller (results/benchmark_<tag>.csv) are composed into
   *routes* of a given length and crossing count (bootstrap; failures and over-budget payload
   shocks make a route unusable): ``aedrover.analysis.route_model``.
2. Simulated rover travel times feed the clinical time-to-first-shock model
   (``aedrover.clinical.decision``), against the ambulance (Naess response-time statistics) and a
   drone (calibrated quadrotor comparator).
3. Outputs: survival versus radius for every mode and survival model, break-even radii, the
   one-at-a-time sensitivity (tornado), dispatch policies including the hybrid, and the
   dimensionless economics of the course model.

    python experiments/05_clinical_analysis.py --bench results/benchmark_standard.csv --controllers dwa mppi ppo
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aedrover.analysis.report import load_benchmark
from aedrover.analysis.route_model import compose_routes, summary
from aedrover.clinical.decision import (
    ScenarioParams,
    breakeven_radius,
    evaluate_modes,
    mode_times,
    tornado,
)
from aedrover.clinical.dispatch_policy import evaluate_policies
from aedrover.clinical.economics import EconomicsInputs, qaly_gain_per_encounter
from aedrover.clinical.survival import get_model

ROOT = Path(__file__).resolve().parents[1]
RADII = (250, 500, 750, 1000, 1500, 2000, 3000)
DEFAULT_ROUTE_FACTOR = 1.3        # ASSUMPTION unless data/osm/vile_parle_summary.json is present
DEFAULT_CROSSINGS_PER_KM = 0.24   # lower bound; replaced by the OSM pooled estimate when available


def route_geometry() -> dict:
    """Walking route factor from the OSM analysis (median), plus the crossing-density sweep.

    OpenStreetMap tags almost none of the crossings in this area (98 percent of road length has no
    sidewalk tag, 24 crossing nodes in a 2 km disc), so the tagged density is only a LOWER BOUND.
    The analysis therefore sweeps crossings per km from that bound up to assumed urban values.
    """
    path = ROOT / "data" / "osm" / "vile_parle_summary.json"
    densities = {"osm_lower_bound": DEFAULT_CROSSINGS_PER_KM, "assumed_2_per_km": 2.0,
                 "assumed_4_per_km": 4.0, "assumed_8_per_km": 8.0}
    if path.exists():
        s = json.loads(path.read_text(encoding="utf-8"))
        rf = s["routes"]["metrics"]["walk_factor"]["median"]
        lb = s["routes"]["per_km"]["crossings_tagged"]["pooled"]
        densities["osm_lower_bound"] = float(lb)
        return {"route_factor": float(rf), "densities": densities, "source": "OpenStreetMap (Vile Parle, 2 km disc)"}
    return {"route_factor": DEFAULT_ROUTE_FACTOR, "densities": densities,
            "source": "ASSUMPTION (no OSM summary found)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="+", default=["results/benchmark_standard.csv"])
    ap.add_argument("--controllers", nargs="+", default=["dwa", "mppi", "ppo"])
    ap.add_argument("--n-routes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/clinical.json")
    args = ap.parse_args()

    df = load_benchmark(*[ROOT / p for p in args.bench])
    geo = route_geometry()
    print("route geometry:", geo)
    larsen, rot = get_model("larsen1993"), get_model("rule_of_thumb")
    out: dict = {"geometry": geo, "controllers": {}, "radii_m": list(RADII)}
    rows_surv, rows_route = [], []

    for ctrl in args.controllers:
        eps = df[df.controller == ctrl]
        if eps.empty:
            print(f"skip {ctrl}: no episodes")
            continue
        for dens_name, dens in geo["densities"].items():
            for r in RADII:
                dist = r * geo["route_factor"]
                n_cross = int(round(dens * dist / 1000.0))
                rs = compose_routes(eps, dist, n_cross, n=args.n_routes, seed=args.seed)
                sm = summary(rs)
                rows_route.append({"controller": ctrl, "density": dens_name, "crossings_per_km": dens,
                                   "radius_m": r, "route_m": dist, "crossings": n_cross, **sm})
                params = ScenarioParams(radius_m=float(r), route_factor=geo["route_factor"])
                for model in (larsen, rot):
                    rng = np.random.default_rng(args.seed)
                    samples = mode_times(args.n_routes, rng, params, parallel=True, rover_travel_min=rs.time_min)
                    ev = evaluate_modes(samples, model, seed=args.seed, n_resamples=500)
                    for _, row in ev.iterrows():
                        rows_surv.append({"controller": ctrl, "density": dens_name, "radius_m": r,
                                          "model": model.name, **row.to_dict()})
        print(f"{ctrl}: composed routes for {len(RADII)} radii")

    surv = pd.DataFrame(rows_surv)
    route = pd.DataFrame(rows_route)
    out_dir = ROOT / "results"
    surv.to_csv(out_dir / "clinical_survival_vs_radius.csv", index=False)
    route.to_csv(out_dir / "clinical_routes.csv", index=False)

    # analytic break-even at the *measured* sidewalk speed of each controller (rover alone vs ambulance alone)
    be = {}
    for ctrl in args.controllers:
        eps = df[(df.controller == ctrl) & df.success & df.family.isin(["flat_clear", "crowded"])]
        if eps.empty:
            continue
        v = float((eps.path_m / eps.time_s).median())
        d = ScenarioParams().dispatch
        for model in (larsen, rot):
            b = breakeven_radius(v, geo["route_factor"], d.call_to_alert_min, d.handoff_min,
                                 ScenarioParams().ems_median_min, model)
            be[f"{ctrl}|{model.name}"] = {"speed_mps": v, "radius_m": b.radius_m, "flag": b.flag,
                                          "ambulance_survival": b.ambulance_survival}
    out["breakeven"] = be

    tor = tornado(larsen, ScenarioParams(rover_speed_mps=2.0))
    out["tornado"] = tor.to_dict(orient="records")
    tor.to_csv(out_dir / "clinical_tornado.csv", index=False)

    # dispatch policies at 1 km with the best-measured rover, sweeping drone availability
    pol_rows = []
    for ctrl in args.controllers:
        eps = df[df.controller == ctrl]
        if eps.empty:
            continue
        r = 1000
        dist = r * geo["route_factor"]
        rs = compose_routes(eps, dist, int(round(geo["densities"]["assumed_4_per_km"] * dist / 1000.0)),
                            n=args.n_routes, seed=args.seed)
        for pd_ in (0.0, 0.25, 0.5, 0.75, 1.0):
            ev = evaluate_policies(larsen, ScenarioParams(radius_m=float(r), route_factor=geo["route_factor"]),
                                   rover_travel_min=rs.time_min, p_drone=pd_, seed=args.seed, n_resamples=400)
            for _, row in ev.iterrows():
                pol_rows.append({"controller": ctrl, "p_drone": pd_, "radius_m": r, **row.to_dict()})
    pol = pd.DataFrame(pol_rows)
    pol.to_csv(out_dir / "clinical_dispatch_policies.csv", index=False)

    # dimensionless economics (all inputs are assumptions; reported with the course's kappa target)
    econ = EconomicsInputs()
    best = surv[(surv.model == "larsen1993") & (surv["mode"] == "rover") & (surv.radius_m == 1000)
                & (surv.density == "assumed_4_per_km")]
    dS = float(best.abs_gain.max()) if len(best) else float("nan")
    out["economics"] = {"kappa": econ.kappa, "meets_target": econ.meets_target, "payback_months": econ.payback_months,
                        "delta_fte": econ.delta_fte, "qaly_per_encounter_at_1km": float(qaly_gain_per_encounter(dS, 12.0, 0.85)) if np.isfinite(dS) else None,
                        "note": "every ratio is an ASSUMPTION (dimensionless); see docs/CLINICAL_MODEL.md"}
    (ROOT / args.out).write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(route.groupby(["controller", "density", "radius_m"])[["p_arrive", "p_safe_delivery", "time_min_median"]]
          .mean().round(3).to_string())
    print("break-even:", json.dumps(be, indent=1, default=float))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
