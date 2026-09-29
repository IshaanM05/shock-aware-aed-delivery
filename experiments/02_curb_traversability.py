"""Experiment 02: kerb traversability map (RQ1).

Sweeps kerb height x approach speed x approach angle x wheel radius x friction for kerb-up and
kerb-down and writes results/curb_traversability.csv plus results/curb_window.json (per
height/wheel/friction: minimum speed that climbs, maximum speed within the shock budget).

    python experiments/02_curb_traversability.py [--preset smoke|standard]
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path

import pandas as pd

from aedrover.parallel import default_workers, pmap
from aedrover.sim.curb_study import KerbTrial, run_kerb_trial
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]

PRESETS = {
    "smoke": dict(h=(0.08, 0.12), v=(0.6, 1.0, 1.6, 2.4), ang=(0.0, 20.0), r=(0.15, 0.20), mu=(1.0,)),
    "standard": dict(h=(0.06, 0.08, 0.10, 0.12, 0.14, 0.16),
                     v=tuple(round(0.4 + 0.2 * i, 1) for i in range(14)),
                     ang=(0.0, 15.0, 30.0), r=(0.125, 0.15, 0.175, 0.20), mu=(0.5, 0.8, 1.0)),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="standard")
    ap.add_argument("--design", choices=("nominal", "optimized"), default="nominal")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--budget-g", type=float, default=3.0)
    args = ap.parse_args()
    g = dict(PRESETS[args.preset])
    veh = VehicleParams.by_name(args.design)
    if args.design == "optimized":
        g["r"] = (veh.wheel_radius,)            # the design fixes the wheel radius
    design = dict(susp_k=veh.susp_k, susp_c=veh.susp_c, iso_kz=veh.iso_kz, iso_cz=veh.iso_cz,
                  motor_peak_torque=veh.motor_peak_torque)

    trials = [KerbTrial(direction=d, kerb_h=h, speed=v, angle_deg=a, wheel_radius=r, mu=mu, **design)
              for d, h, v, a, r, mu in itertools.product(("up", "down"), g["h"], g["v"], g["ang"], g["r"], g["mu"])]
    print(f"{len(trials)} trials on {args.workers or default_workers()} workers")
    t0 = time.perf_counter()
    rows = pmap(run_kerb_trial, trials, workers=args.workers, chunksize=8, desc="kerb")
    print(f"done in {time.perf_counter() - t0:.0f}s")

    df = pd.DataFrame(rows)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    df.to_csv(out / f"curb_traversability_{args.design}_{args.preset}.csv", index=False)

    # climb window per (direction, kerb_h, wheel_radius, mu) at zero approach angle
    win = []
    z = df[df.angle_deg == 0.0]
    for (d, h, r, mu), grp in z.groupby(["direction", "kerb_h", "wheel_radius", "mu"]):
        grp = grp.sort_values("speed")
        ok = grp[grp.success]
        v_min = float(ok.speed.min()) if len(ok) else None
        within = ok[ok.peak_g <= args.budget_g]
        v_shock_max = float(within.speed.max()) if len(within) else None
        win.append({"direction": d, "kerb_h": h, "wheel_radius": r, "mu": mu,
                    "v_min_success": v_min, "v_max_within_budget": v_shock_max,
                    "feasible_window": bool(v_min is not None and v_shock_max is not None and v_shock_max >= v_min)})
    (out / f"curb_window_{args.design}_{args.preset}.json").write_text(json.dumps(win, indent=2), encoding="utf-8")

    r0 = g["r"][min(1, len(g["r"]) - 1)]
    up = df[(df.direction == "up") & (df.angle_deg == 0.0) & (df.mu == 1.0) & (df.wheel_radius == r0)]
    print(f"\nkerb-up, {args.design}, r={r0}, mu=1.0, straight: success% / peak g (rows: kerb height, cols: speed)")
    print(up.pivot_table(index="kerb_h", columns="speed", values="success", aggfunc="mean").round(2).to_string())
    print(up.pivot_table(index="kerb_h", columns="speed", values="peak_g", aggfunc="mean").round(2).to_string())
    feas = pd.DataFrame(win)
    print("\nfeasible shock-compliant climb window (fraction of cells) by wheel radius, kerb-up:")
    print(feas[feas.direction == "up"].groupby("wheel_radius").feasible_window.mean().round(2).to_string())


if __name__ == "__main__":
    main()
