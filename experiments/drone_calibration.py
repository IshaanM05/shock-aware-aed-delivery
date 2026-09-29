"""Calibrate and validate the analytic drone mission model against the full simulation.

Sweeps distance x along-track wind (positive = tailwind, negative = headwind) and

  1. FIT set  (gust-free): fits ``time_offset_s`` (mean tracking lag beyond the reference plan)
     and ``energy_scale`` (least-squares ratio of simulated to analytic energy);
  2. VALIDATION set (gusts, out-of-sample): compares the analytic model, using the shipped
     defaults in ``MissionParams``, to gusty simulations.

It also prints a plausibility cross-check against the real-world medians of ``claesson2017``
(order of magnitude only; never fitted), a parameter-sensitivity table of the analytic model,
and the battery/endurance numbers. Writes results/drone_calibration.json.

    python experiments/drone_calibration.py [--quick] [--gust 1.5] [--seed 11]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from aedrover.drone.energy import EnergyParams, hover_power_w, max_range_m
from aedrover.drone.mission import (
    MissionParams,
    energy_wh,
    mission_time_s,
    simulate_mission,
    time_to_scene_s,
)
from aedrover.drone.quadrotor_mjcf import QuadParams

ROOT = Path(__file__).resolve().parents[1]

# Real-world medians read from the PubMed Central full text of claesson2017 (simulated OHCA
# flights, 18 autonomous flights). Used for an order-of-magnitude cross-check only.
CLAESSON = {
    "median_distance_m": 3200.0,
    "median_dispatch_to_launch_s": 3.0,
    "median_dispatch_to_arrival_s": 5 * 60 + 21.0,
    "iqr_dispatch_to_arrival_s": (3 * 60 + 3.0, 8 * 60 + 33.0),
    "max_cruise_mps": 75.0 / 3.6,
    "drone_mass_kg": 5.7,
    "aed_mass_kg": 0.763,
}


def _pct(model: float, sim: float) -> float:
    return 100.0 * (model - sim) / sim


def _sweep(
    distances: list[float], winds: list[float], gust: float, seed: int, quad: QuadParams
) -> list[dict]:
    rows = []
    for d in distances:
        for w in winds:
            r = simulate_mission(d, w, gust, seed, quad=quad)
            rows.append(
                {
                    "distance_m": d,
                    "wind_along_track_mps": w,
                    "gust_sigma_mps": gust,
                    "seed": seed,
                    "completed": r.completed,
                    "reason": r.reason,
                    "sim_flight_time_s": r.flight_time_s,
                    "sim_energy_wh": r.energy_wh,
                    "peak_tracking_error_m": r.peak_tracking_error_m,
                    "max_tilt_deg": r.max_tilt_deg,
                }
            )
    return rows


def _annotate(rows: list[dict], mp: MissionParams, quad: QuadParams) -> None:
    for row in rows:
        d, w = row["distance_m"], row["wind_along_track_mps"]
        row["model_flight_time_s"] = mission_time_s(d, w, quad, mp)
        row["model_energy_wh"] = energy_wh(d, w, quad, mp)
        row["time_err_pct"] = _pct(row["model_flight_time_s"], row["sim_flight_time_s"])
        row["energy_err_pct"] = _pct(row["model_energy_wh"], row["sim_energy_wh"])


def _print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(
        "dist_m  wind_mps  gust  t_sim_s  t_mod_s  t_err%   E_sim_Wh  E_mod_Wh  E_err%"
        "  peak_m  tilt_deg"
    )
    for r in rows:
        print(
            f"{r['distance_m']:6.0f}  {r['wind_along_track_mps']:8.1f}  {r['gust_sigma_mps']:4.1f}"
            f"  {r['sim_flight_time_s']:7.1f}  {r['model_flight_time_s']:7.1f}"
            f"  {r['time_err_pct']:6.2f}"
            f"  {r['sim_energy_wh']:8.2f}  {r['model_energy_wh']:8.2f}  {r['energy_err_pct']:6.2f}"
            f"  {r['peak_tracking_error_m']:6.2f}  {r['max_tilt_deg']:8.1f}"
        )
    tmax = max(abs(r["time_err_pct"]) for r in rows)
    emax = max(abs(r["energy_err_pct"]) for r in rows)
    print(f"max |time err| = {tmax:.2f}%   max |energy err| = {emax:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 distances x 2 winds")
    ap.add_argument("--gust", type=float, default=1.5, help="validation gust sigma [m/s]")
    ap.add_argument("--seed", type=int, default=11, help="validation gust seed")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "drone_calibration.json")
    args = ap.parse_args()

    quad = QuadParams()
    ep = EnergyParams.from_quad(quad)
    shipped = MissionParams()
    distances = [500.0, 1500.0] if args.quick else [500.0, 1500.0, 3000.0]
    winds = [0.0, -5.0] if args.quick else [5.0, 0.0, -5.0, -9.0]

    # ---- fit on the gust-free sweep ------------------------------------------------------
    fit_rows = _sweep(distances, winds, 0.0, 0, quad)
    if not all(r["completed"] for r in fit_rows):
        raise SystemExit("a fit-set simulation did not complete: " + str(fit_rows))
    plan_only = shipped.with_(time_offset_s=0.0, energy_scale=1.0)
    offsets = [
        r["sim_flight_time_s"]
        - mission_time_s(r["distance_m"], r["wind_along_track_mps"], quad, plan_only)
        for r in fit_rows
    ]
    fitted_offset = sum(offsets) / len(offsets)
    unscaled = shipped.with_(time_offset_s=fitted_offset, energy_scale=1.0)
    e_model = [
        energy_wh(r["distance_m"], r["wind_along_track_mps"], quad, unscaled) for r in fit_rows
    ]
    e_sim = [r["sim_energy_wh"] for r in fit_rows]
    fitted_scale = sum(a * b for a, b in zip(e_model, e_sim, strict=True)) / sum(
        a * a for a in e_model
    )
    fitted = shipped.with_(time_offset_s=fitted_offset, energy_scale=fitted_scale)
    print(
        f"fitted time_offset_s = {fitted_offset:.3f} (shipped {shipped.time_offset_s}), "
        f"energy_scale = {fitted_scale:.4f} (shipped {shipped.energy_scale})"
    )

    _annotate(fit_rows, shipped, quad)
    _print_table("FIT set (gust-free), analytic model with SHIPPED constants", fit_rows)

    # ---- out-of-sample validation with gusts ----------------------------------------------
    val_rows = _sweep(distances, winds, args.gust, args.seed, quad)
    val_ok = [r for r in val_rows if r["completed"]]
    _annotate(val_ok, shipped, quad)
    _print_table(f"VALIDATION set (gust sigma {args.gust} m/s, seed {args.seed})", val_ok)
    n_bad = len(val_rows) - len(val_ok)
    if n_bad:
        print(f"WARNING: {n_bad} validation simulations did not complete")

    # ---- plausibility cross-check against real-world medians ------------------------------
    d_c = CLAESSON["median_distance_m"]
    model_flight = mission_time_s(d_c, 0.0, quad, shipped)
    cross = {
        "claesson2017": CLAESSON,
        "model_flight_time_s_at_median_distance": model_flight,
        "model_time_to_scene_s_with_3s_launch": CLAESSON["median_dispatch_to_launch_s"]
        + model_flight,
        "model_time_to_scene_s_with_default_launch": time_to_scene_s(d_c, 0.0, quad, shipped),
        "note": (
            "Medians of different distributions do not compose exactly; the study flew simulated "
            "alerts in good weather. This is a plausibility check, not a validation."
        ),
    }
    print("\nCROSS-CHECK vs claesson2017 (median 3.2 km flight, dispatch-to-arrival median 321 s)")
    print(
        f"  model flight time {model_flight:.0f} s; with the 3 s launch of that study "
        f"{cross['model_time_to_scene_s_with_3s_launch']:.0f} s (real median 321 s, IQR 183-513 s)"
    )

    # ---- sensitivity of the analytic model (3000 m, still air) ----------------------------
    base_t = mission_time_s(3000.0, 0.0, quad, fitted)
    base_e = energy_wh(3000.0, 0.0, quad, fitted, ep)
    sens = []
    cases = [
        ("figure_of_merit 0.5", {"ep": ep.__class__.from_quad(quad, figure_of_merit=0.5)}),
        ("figure_of_merit 0.7", {"ep": ep.__class__.from_quad(quad, figure_of_merit=0.7)}),
        ("avionics 10 W", {"ep": ep.__class__.from_quad(quad, avionics_w=10.0)}),
        ("avionics 60 W", {"ep": ep.__class__.from_quad(quad, avionics_w=60.0)}),
        ("payload 0.763 kg (claesson2017 AED)", {"quad": quad.with_(payload_mass_kg=0.763)}),
        ("v_air 10 m/s", {"mp": fitted.with_(v_air_mps=10.0)}),
        ("v_air 20 m/s", {"mp": fitted.with_(v_air_mps=20.0)}),
        ("cruise altitude 30 m", {"mp": fitted.with_(cruise_alt_m=30.0)}),
        ("cruise altitude 120 m", {"mp": fitted.with_(cruise_alt_m=120.0)}),
    ]
    print("\nSENSITIVITY (3000 m, still air): flight time / energy vs baseline")
    print(f"  baseline: {base_t:.1f} s, {base_e:.2f} Wh")
    for name, kw in cases:
        q = kw.get("quad", quad)
        m = kw.get("mp", fitted)
        e_p = kw.get("ep", EnergyParams.from_quad(q))
        t = mission_time_s(3000.0, 0.0, q, m)
        e = energy_wh(3000.0, 0.0, q, m, e_p)
        sens.append({"case": name, "flight_time_s": t, "energy_wh": e})
        print(
            f"  {name:38s} t={t:7.1f} s ({_pct(t, base_t):+6.1f}%)"
            f"  E={e:6.2f} Wh ({_pct(e, base_e):+6.1f}%)"
        )

    # ---- battery, endurance, range --------------------------------------------------------
    p_hover = hover_power_w(quad.total_mass_kg, ep)
    battery = {
        "hover_power_w": p_hover,
        "capacity_wh": ep.capacity_wh,
        "usable_wh": ep.usable_wh,
        "hover_endurance_min": ep.usable_wh / p_hover * 60.0,
        "max_range_still_air_m_at_v_air": {
            f"{v:g}": max_range_m(v, 0.0, quad.total_mass_kg, quad.cd_area_m2, ep)
            for v in (10.0, 15.0, 20.0)
        },
        "thrust_to_weight": quad.thrust_to_weight,
    }
    print(
        f"\nBATTERY: hover {p_hover:.0f} W, usable {ep.usable_wh:.0f} Wh, hover endurance "
        f"{battery['hover_endurance_min']:.1f} min, steady-cruise range at 15 m/s "
        f"{battery['max_range_still_air_m_at_v_air']['15'] / 1000:.1f} km "
        "(before climb/descent overhead)"
    )

    out = {
        "quad": quad.to_dict(),
        "energy": {
            k: getattr(ep, k)
            for k in ("figure_of_merit", "avionics_w", "battery_mass_kg", "reserve_fraction")
        },
        "shipped_mission_params": shipped.__dict__,
        "fitted": {"time_offset_s": fitted_offset, "energy_scale": fitted_scale},
        "fit_rows": fit_rows,
        "validation_rows": val_ok,
        "validation_incomplete": n_bad,
        "max_abs_time_err_pct_fit": max(abs(r["time_err_pct"]) for r in fit_rows),
        "max_abs_energy_err_pct_fit": max(abs(r["energy_err_pct"]) for r in fit_rows),
        "max_abs_time_err_pct_validation": max(
            (abs(r["time_err_pct"]) for r in val_ok), default=math.nan
        ),
        "max_abs_energy_err_pct_validation": max(
            (abs(r["energy_err_pct"]) for r in val_ok), default=math.nan
        ),
        "crosscheck": cross,
        "sensitivity_3000m_still_air": sens,
        "battery": battery,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
