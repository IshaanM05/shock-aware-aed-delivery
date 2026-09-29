"""Experiment 01: validate the MuJoCo rover against closed-form mechanics, for both vehicle designs.

Checks static equilibrium, effective tyre stiffness, free-bounce ring-down against the analytic
2-DOF quarter-car eigenvalues, straight-line rolling, and a long random-command stability soak.
Writes results/validation_<design>.json and prints a summary table.

    python experiments/01_validate_suspension.py [--soak-steps 1000000]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aedrover.sim.validate import (
    ringdown,
    rolling_slip,
    stability_soak,
    static_equilibrium,
    tyre_stiffness,
)
from aedrover.sim.vehicle_mjcf import VehicleParams

ROOT = Path(__file__).resolve().parents[1]


def validate(veh: VehicleParams, soak_steps: int) -> dict:
    out: dict = {"vehicle": veh.to_dict()}
    out["static"] = static_equilibrium(veh)
    print(f"[{veh.name}] ride height {out['static']['ride_height_m'] * 1000:.2f} mm "
          f"(design {veh.nominal_height * 1000:.2f}), payload static accel "
          f"{out['static']['payload_acc_z']:.3f} m/s^2")
    out["tyre"] = tyre_stiffness(veh)
    print(f"[{veh.name}] effective tyre stiffness {out['tyre']['kt_eff_N_per_m'] / 1000:.0f} kN/m")
    out["ringdown"] = []
    print("  zeta_target  wn_meas  wn_ana  err%   zeta_meas  zeta_ana  err%")
    for z in (0.10, 0.20, 0.30):
        rd = ringdown(veh, z)
        out["ringdown"].append(rd)
        if "measured_zeta" in rd:
            print(f"  {z:11.2f}  {rd['measured_omega_n']:7.2f}  {rd['analytic_omega_n']:6.2f}  "
                  f"{rd['omega_n_err_pct']:5.1f}  {rd['measured_zeta']:9.3f}  {rd['analytic_zeta']:8.3f}  "
                  f"{rd['zeta_err_pct']:5.1f}")
    out["rolling"] = [rolling_slip(veh, v) for v in (0.5, 1.0, 2.0)]
    for r in out["rolling"]:
        print(f"  rolling v={r['v_cmd']:.1f}: measured {r['v_measured']:.3f} m/s "
              f"({r['speed_err_pct']:+.2f}%), lateral drift {r['lateral_drift_m'] * 1000:.1f} mm")
    out["soak"] = stability_soak(veh, n_steps=soak_steps)
    print(f"  soak: {out['soak']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--soak-steps", type=int, default=1_000_000)
    args = ap.parse_args()
    (ROOT / "results").mkdir(exist_ok=True)
    for veh in (VehicleParams.nominal(), VehicleParams.optimized()):
        res = validate(veh, args.soak_steps)
        path = ROOT / "results" / f"validation_{veh.name}.json"
        path.write_text(json.dumps(res, indent=2), encoding="utf-8")
        print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
