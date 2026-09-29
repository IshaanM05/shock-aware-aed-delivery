"""Physics validation: compare the MuJoCo rover against closed-form mechanics.

Each function returns plain dicts so tests can assert on them and
``experiments/01_validate_suspension.py`` can write them to ``results/``.
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
from scipy.signal import find_peaks

from .rover import Rover
from .vehicle_mjcf import G, VehicleParams
from .world import World


def analytic_quarter_car(ms: float, mu: float, ks: float, cs: float, kt: float) -> dict:
    """Body-mode natural frequency and damping ratio of a 2-DOF quarter-car (eigen-analysis)."""
    M = np.diag([ms, mu])
    K = np.array([[ks, -ks], [-ks, ks + kt]])
    C = np.array([[cs, -cs], [-cs, cs]])
    Z, eye = np.zeros((2, 2)), np.eye(2)
    A = np.block([[Z, eye], [-np.linalg.solve(M, K), -np.linalg.solve(M, C)]])
    eig = np.linalg.eigvals(A)
    body = eig[np.imag(eig) > 0]
    if len(body) == 0:  # both modes overdamped
        return {"omega_n": float(np.sqrt(ks / ms)), "zeta": float("inf"), "oscillatory": False}
    lam = body[np.argmin(np.abs(body))]
    wn = float(np.abs(lam))
    return {"omega_n": wn, "zeta": float(-lam.real / wn), "oscillatory": True}


def _flat_rover(veh: VehicleParams) -> Rover:
    w = World(veh)
    w.set_flat()
    r = Rover(w)
    r.reset(0.0, 0.0, 0.0, settle_s=2.0)
    return r


def static_equilibrium(veh: VehicleParams) -> dict:
    r = _flat_rover(veh)
    z = float(r.pos[2])
    return {
        "ride_height_m": z,
        "ride_height_expected_m": veh.nominal_height,
        "ride_height_err_mm": 1000.0 * (z - veh.nominal_height),
        "susp_deflection_mm": (1000.0 * r.susp_deflection()).tolist(),
        "payload_acc_z": float(r.payload_acc()[2]),
        "payload_acc_expected": G,
    }


def tyre_stiffness(veh: VehicleParams) -> dict:
    """Effective radial tyre stiffness from static penetration under the measured wheel load."""
    r = _flat_rover(veh)
    d, w = r.d, r.w
    pen = np.zeros(4)
    for c in d.contact[: d.ncon]:
        for k, g in enumerate(w.g_tyre):
            if g in (c.geom1, c.geom2):
                pen[k] = max(pen[k], -c.dist)
    load = veh.total_mass * G / 4.0
    mean_pen = float(pen.mean())
    return {
        "penetration_mm": (1000.0 * pen).tolist(),
        "wheel_load_N": load,
        "kt_eff_N_per_m": load / mean_pen if mean_pen > 1e-9 else float("inf"),
    }


def ringdown(veh: VehicleParams, zeta_target: float, v0: float = 0.25, t_end: float = 2.0) -> dict:
    """Free bounce of the symmetric mode: measured (omega_n, zeta) vs the 2-DOF analytic value."""
    ms_wheel = veh.sprung_per_wheel
    cs = 2.0 * zeta_target * np.sqrt(veh.susp_k * ms_wheel)
    # negligible payload so that the quarter-car model is exact (payload isolator validated apart)
    v = veh.with_(susp_c=float(cs), chassis_mass=veh.sprung_mass - 0.05, payload_mass=0.05)
    kt = tyre_stiffness(v)["kt_eff_N_per_m"]
    r = _flat_rover(v)
    dof = r.m.jnt_dofadr[r.j_root]
    z_eq = float(r.pos[2])
    r.d.qvel[dof + 2] = v0
    n = int(t_end / r.dt_phys)
    zs = np.empty(n)
    ts = np.empty(n)
    for i in range(n):
        mujoco.mj_step(r.m, r.d)
        zs[i] = r.pos[2] - z_eq
        ts[i] = r.d.time
    pk, _ = find_peaks(zs, prominence=2e-5)
    out = {"zeta_target": zeta_target, "kt_eff": kt, "n_peaks": int(len(pk))}
    ana = analytic_quarter_car(ms_wheel, v.unsprung_mass, v.susp_k, float(cs), kt)
    out.update({"analytic_omega_n": ana["omega_n"], "analytic_zeta": ana["zeta"]})
    if len(pk) >= 2:
        T = float(np.mean(np.diff(ts[pk])))
        delta = float(np.mean(np.log(zs[pk[:-1]] / zs[pk[1:]])))
        zeta = delta / np.sqrt(4 * np.pi**2 + delta**2)
        wd = 2 * np.pi / T
        out.update({"measured_omega_n": float(wd / np.sqrt(1 - zeta**2)), "measured_zeta": float(zeta)})
        out["omega_n_err_pct"] = 100 * (out["measured_omega_n"] / ana["omega_n"] - 1)
        out["zeta_err_pct"] = 100 * (out["measured_zeta"] / ana["zeta"] - 1)
    return out


def rolling_slip(veh: VehicleParams, v: float = 1.0, t_end: float = 10.0) -> dict:
    w = World(veh)
    w.set_flat()
    r = Rover(w)
    r.reset(0.0, 0.0, 0.0, settle_s=1.0)
    for _ in range(int(t_end / r.control_dt)):
        r.step(v, 0.0, log=False)
    vx = float(r.body_velocity()[0])
    x = float(r.pos[0])
    return {
        "v_cmd": v,
        "v_measured": vx,
        "speed_err_pct": 100 * (vx / v - 1),
        "mean_speed_err_pct": 100 * (x / t_end / v - 1),
        "lateral_drift_m": float(r.pos[1]),
        "wheel_omega_mean": float(r.wheel_speeds().mean()),
        "wheel_omega_expected": v / veh.wheel_radius,
    }


def stability_soak(veh: VehicleParams, n_steps: int = 1_000_000, seed: int = 0) -> dict:
    """Random commands over a kerb field for ``n_steps`` physics steps: must stay finite."""
    rng = np.random.default_rng(seed)
    w = World(veh)
    w.set_crossing(0.12, x_down=-8.0, x_up=8.0)
    r = Rover(w)
    r.reset(0.0, 0.0, 0.0, settle_s=0.5)
    t0 = time.perf_counter()
    steps, resets = 0, 0
    v_cmd, d_cmd = 1.0, 0.0
    max_speed = 0.0
    while steps < n_steps:
        if steps % (r.n_sub * 25) == 0:
            v_cmd, d_cmd = rng.uniform(-1.5, 3.0), rng.uniform(-0.5, 0.5)
        r.step(v_cmd, d_cmd, log=False)
        steps += r.n_sub
        if not r.is_finite():
            return {"finite": False, "steps": steps, "resets": resets}
        max_speed = max(max_speed, float(np.linalg.norm(r.d.qvel[:3])))
        if abs(r.pos[0]) > 40 or abs(r.pos[1]) > 20 or r.pos[2] > 2.0:
            r.reset(0.0, 0.0, 0.0)
            resets += 1
    wall = time.perf_counter() - t0
    return {"finite": True, "steps": steps, "resets": resets, "max_speed_mps": max_speed,
            "steps_per_s": steps / wall, "wall_s": wall}
