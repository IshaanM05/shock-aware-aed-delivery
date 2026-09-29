"""Physics-in-the-loop MPPI (Williams et al. 2017, key williams2017; sampling MPC with MuJoCo
rollouts as in Howell et al. 2022, key howell2022).

Every planning cycle (10 Hz) the controller samples ``K`` (speed, steering) sequences around the
running mean, rolls all of them out **in the MuJoCo model** with ``mujoco.rollout`` (contacts,
suspension, payload isolator and wheel-speed loops included), scores the trajectories and updates
the mean by the exponentially weighted average of the perturbations.

The planner's internal model is deliberately imperfect:
* a coarser physics step (``dt_phys`` = 10 ms versus 2 ms in the environment; measured to
  over-predict kerb-strike shock by about 8 percent, i.e. conservatively),
* nominal tyre friction and payload mass (the environment randomises both),
* pedestrians follow constant-velocity predictions of the (noisy) tracker output.

Terrain, static obstacles and the predicted pedestrians are fed to ``mujoco.rollout`` as mocap
poses through ``control_spec`` (``mujoco.rollout`` otherwise resets every mocap body to its
compiled position, which silently deletes the kerbs from the rollouts: regression-tested).

Cost terms: forward progress, payload shock above a soft threshold (from the simulated payload
accelerometer, gravity-compensated exactly like the reported metric), pedestrian/obstacle
clearance of the oriented footprint, sidewalk lane, heading, roll/pitch stability and control
smoothness. The physics is what discovers that a kerb needs momentum to climb but that momentum
costs shock.
"""

from __future__ import annotations

import mujoco
import numpy as np
from mujoco import rollout

from ..sim.pedestrians import PED_R, ROBOT_HALF_L, ROBOT_HALF_W
from ..sim.scenario import Scenario
from ..sim.vehicle_mjcf import G, VehicleParams
from ..sim.world import OBS_SIZES, PED_CENTER_Z, World
from .base import register


def _box_gap(x, y, yaw, px, py, r):
    """Gap between the oriented footprint at (x, y, yaw) and discs at (px, py) of radius r.

    Shapes: x, y, yaw are (K, H, 1); px, py are (1, H, M) or (1, 1, M). Returns (K, H, M)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rx, ry = px - x, py - y
    bx, by = c * rx + s * ry, -s * rx + c * ry
    dx = np.maximum(np.abs(bx) - ROBOT_HALF_L, 0.0)
    dy = np.maximum(np.abs(by) - ROBOT_HALF_W, 0.0)
    return np.hypot(dx, dy) - r


_SPEC = mujoco.mjtState.mjSTATE_CTRL | mujoco.mjtState.mjSTATE_MOCAP_POS | mujoco.mjtState.mjSTATE_MOCAP_QUAT


class MPPIController:
    def __init__(self, K: int = 128, H: int = 20, dt_plan: float = 0.1, dt_phys: float = 0.01,
                 sigma_v: float = 0.5, sigma_d: float = 0.22, v_max: float = 2.6, delta_max: float = 0.5,
                 temperature: float = 0.3, replan_every: int = 5, nthread: int = 1,
                 nominal_mu: float = 0.9, nominal_payload: float = 4.0, budget_g: float = 3.0,
                 v_ref: float = 1.8, w_prog: float = 3.0, w_shock: float = 4.0, w_clear: float = 6.0,
                 w_col: float = 100.0, w_lane: float = 4.0, w_head: float = 0.5, w_stab: float = 5.0,
                 w_u: float = 0.05, comfort_clearance: float = 0.5, n_knots: int = 5, name: str = "mppi"):
        self.name = name
        self.K, self.H, self.dt_plan, self.dt_phys = K, H, dt_plan, dt_phys
        self.n_sub = round(dt_plan / dt_phys)
        self.sig = np.array([sigma_v, sigma_d])
        self.vmax, self.dmax = v_max, delta_max
        self.temp, self.replan_every, self.nthread = temperature, replan_every, nthread
        self.nominal_mu, self.nominal_payload, self.budget_g = nominal_mu, nominal_payload, budget_g
        self.v_ref = v_ref
        self.w = dict(prog=w_prog, shock=w_shock, clear=w_clear, col=w_col, lane=w_lane, head=w_head,
                      stab=w_stab, u=w_u)
        self.comfort = comfort_clearance
        self.n_knots = n_knots
        self._pw: World | None = None
        self._veh_key: tuple | None = None
        self.last: dict = {}

    # ---------------------------------------------------------------- setup
    def _build(self, veh: VehicleParams, spec) -> None:
        self._pw = World(veh, spec)
        self.model = self._pw.model
        self.model.opt.timestep = self.dt_phys
        self.datas = [mujoco.MjData(self.model) for _ in range(self.nthread)]
        self._size = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        m = self.model
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "payload_acc")
        self._s_pay = slice(int(m.sensor_adr[sid]), int(m.sensor_adr[sid]) + 3)
        self.nm = m.nmocap
        assert mujoco.mj_stateSize(m, _SPEC) == m.nu + 7 * self.nm
        xk = np.linspace(0.0, self.H - 1.0, self.n_knots)
        interp = np.zeros((self.H, self.n_knots))
        for j in range(self.n_knots):
            e = np.zeros(self.n_knots)
            e[j] = 1.0
            interp[:, j] = np.interp(np.arange(self.H), xk, e)
        self._interp = interp                               # (H, n_knots) linear interpolation weights
        self.veh = veh
        self.L, self.W, self.r = veh.wheelbase, veh.track, veh.wheel_radius
        self._wx = np.array([self.L, self.L, 0.0, 0.0])
        self._wy = np.array([self.W / 2, -self.W / 2, self.W / 2, -self.W / 2])
        self._veh_key = (veh, spec)

    def reset(self, env) -> None:
        key = (env.veh0, env.world.spec)
        if self._veh_key != key:
            self._build(env.veh0, env.world.spec)
        sc: Scenario = env.scenario
        self.env, self.sc = env, sc
        pw = self._pw
        pw.apply_scenario(sc, tyre_mu=self.nominal_mu, ground_mu=self.nominal_mu,
                          payload_mass=self.nominal_payload)
        for i in range(pw.spec.n_ped):
            pw.park_pedestrian(i)
        for d in self.datas:
            mujoco.mj_resetData(self.model, d)
        self._mpos = pw._mpos.copy()
        self._mquat = pw._mquat.copy()
        self._obs_pts = np.array([[ob.x, ob.y, OBS_SIZES[ob.slot % 2][0]] for ob in sc.obstacles]).reshape(-1, 3)
        self.rng = np.random.default_rng(sc.seed + 104729)
        self.U = np.zeros((self.H, 2))
        self.U[:, 0] = 1.0
        self._k = 0
        self._cmd = (0.0, 0.0)
        self._u_prev = np.zeros(2)

    # -------------------------------------------------------------- control
    def act(self, obs: dict) -> tuple[float, float]:
        if self._k % self.replan_every == 0:
            self._cmd = self._plan(obs)
        self._k += 1
        return self._cmd

    def _allocate(self, v: np.ndarray, delta: np.ndarray) -> np.ndarray:
        """Vectorised Ackermann allocation -> actuator controls (..., 6): 4 drive speeds, 2 steer angles."""
        L, W = self.L, self.W
        t = np.tan(np.clip(delta, -1.2, 1.2))
        small = np.abs(t) < 1e-6
        ts = np.where(small, 1e-6, t)
        R = L / ts
        sfl = np.arctan(L * ts / (L - 0.5 * W * ts))
        sfr = np.arctan(L * ts / (L + 0.5 * W * ts))
        dist = np.hypot(self._wx, R[..., None] - self._wy)
        center = np.hypot(0.5 * L, R)[..., None]
        wheel = np.sign(v)[..., None] * np.abs(v)[..., None] * dist / center / self.r
        wheel = np.where(small[..., None], (v / self.r)[..., None], wheel)
        wmax = self.veh.wheel_speed_max
        lim = self.veh.steer_range * 0.98
        return np.concatenate([np.clip(wheel, -wmax, wmax), np.clip(sfl, -lim, lim)[..., None],
                               np.clip(sfr, -lim, lim)[..., None]], axis=-1)

    def _plan(self, obs: dict) -> tuple[float, float]:
        K = self.K
        d0 = self.env.world.data
        x0 = np.concatenate([[d0.time], d0.qpos, d0.qvel])
        # temporally correlated exploration noise + a per-sample speed offset (explores kerb-climb speeds)
        zk = self.rng.standard_normal((K, self.n_knots, 2)) * self.sig
        eps = np.einsum("hk,nkd->nhd", self._interp, zk)                         # (K, H, 2) spline noise
        eps[:, :, 0] += self.rng.standard_normal((K, 1)) * 0.5 * self.sig[0]     # per-sample speed offset
        cand = self.U[None] + eps
        cand[0] = self.U                               # unperturbed mean
        cand[1, :, 0] = 0.0                            # always-available "stop" sample
        cand[1, :, 1] = 0.0
        cand[:, :, 0] = np.clip(cand[:, :, 0], 0.0, self.vmax)
        cand[:, :, 1] = np.clip(cand[:, :, 1], -self.dmax, self.dmax)

        nu, nm, T = self.model.nu, self.nm, self.H * self.n_sub
        ctrl = np.empty((K, T, nu + 7 * nm))
        ctrl[:, :, :nu] = np.repeat(self._allocate(cand[..., 0], cand[..., 1]), self.n_sub, axis=1)
        ctrl[:, :, nu:] = self._mocap_block(obs, T)[None]
        init = np.tile(x0, (K, 1))
        state, sens = rollout.rollout(self.model, self.datas, init, ctrl, control_spec=_SPEC)
        cost = self._cost(state, sens, cand, obs)

        w = np.exp(-(cost - cost.min()) / (self.temp * (cost.std() + 1e-6)))
        w /= w.sum()
        self.U = np.einsum("k,khd->hd", w, cand)
        cmd = (float(self.U[0, 0]), float(self.U[0, 1]))
        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = self.U[-2]
        self._u_prev = np.array(cmd)
        self.last = {"cost_min": float(cost.min()), "cost_mean": float(cost.mean()), "n_eff": float(1.0 / np.sum(w**2))}
        return cmd

    def _mocap_block(self, obs: dict, T: int) -> np.ndarray:
        """Mocap poses for every physics step of the rollout: static terrain/obstacles plus the
        pedestrians propagated at constant velocity from the tracker output. Shape (T, 7 * nm)."""
        pw, nm = self._pw, self.nm
        mp = np.repeat(self._mpos[None], T, axis=0)                          # (T, nm, 3)
        tr = obs["tracks"]
        yaw0, ox, oy = obs["yaw"], obs["x"], obs["y"]
        c0, s0 = np.cos(yaw0), np.sin(yaw0)
        tt = (np.arange(T) + 1) * self.dt_phys
        for j, row in enumerate(tr[tr[:, 4] > 0.5][: pw.spec.n_ped]):
            slot = int(pw.m_ped[j])
            px, py = ox + c0 * row[0] - s0 * row[1], oy + s0 * row[0] + c0 * row[1]
            vx, vy = c0 * row[2] - s0 * row[3], s0 * row[2] + c0 * row[3]
            mp[:, slot, 0] = px + vx * tt
            mp[:, slot, 1] = py + vy * tt
            mp[:, slot, 2] = self.sc.surface_z(px) + PED_CENTER_Z
        mq = np.repeat(self._mquat[None], T, axis=0)
        return np.concatenate([mp.reshape(T, 3 * nm), mq.reshape(T, 4 * nm)], axis=1)

    # ---------------------------------------------------------------- cost
    def _cost(self, state: np.ndarray, sens: np.ndarray, cand: np.ndarray, obs: dict) -> np.ndarray:
        K, T, _ = state.shape
        H, ns = self.H, self.n_sub
        q = state[..., 1:1 + self.model.nq]
        # ---- per-physics-step payload shock (gravity-compensated, as in sim.metrics) -----------
        qw, qx, qy, qz = q[..., 3], q[..., 4], q[..., 5], q[..., 6]
        up = np.stack([2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)], axis=-1)
        dyn = sens[..., self._s_pay] - G * up
        shock = np.linalg.norm(dyn, axis=-1) / G                           # (K, T)
        s0 = 0.75 * self.budget_g
        c_shock = self.w["shock"] * np.sum(np.maximum(shock - s0, 0.0) ** 2, axis=1) * self.dt_phys * 10.0

        # ---- plan-tick quantities -----------------------------------------------------------
        idx = np.arange(ns - 1, T, ns)[:H]
        x, y = q[:, idx, 0], q[:, idx, 1]
        w_, x_, y_, z_ = q[:, idx, 3], q[:, idx, 4], q[:, idx, 5], q[:, idx, 6]
        yaw = np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_**2 + z_**2))
        pitch = np.arcsin(np.clip(2 * (w_ * y_ - z_ * x_), -1, 1))
        roll = np.arctan2(2 * (w_ * x_ + y_ * z_), 1 - 2 * (x_**2 + y_**2))

        prog = (x[:, -1] - float(q[0, 0, 0])) / (self.v_ref * H * self.dt_plan)
        c_prog = -self.w["prog"] * prog
        c_lane = self.w["lane"] * np.sum(np.maximum(np.abs(y) - 1.0, 0.0) ** 2, axis=1)
        c_head = self.w["head"] * np.sum(yaw**2, axis=1) / H
        c_stab = self.w["stab"] * np.sum(np.maximum(np.abs(roll) - 0.3, 0.0) ** 2
                                          + np.maximum(np.abs(pitch) - 0.35, 0.0) ** 2, axis=1)
        dU = np.diff(cand, axis=1, prepend=self._u_prev[None, None, :].repeat(cand.shape[0], 0))
        c_u = self.w["u"] * np.sum((dU / self.sig) ** 2, axis=(1, 2)) / H

        # ---- clearance to pedestrians (constant-velocity prediction) and mapped obstacles ---------
        gaps = []
        yaw0, ox, oy = obs["yaw"], obs["x"], obs["y"]
        c0, s0_ = np.cos(yaw0), np.sin(yaw0)
        tr = obs["tracks"]
        live = tr[:, 4] > 0.5
        t_pred = (np.arange(1, H + 1) * self.dt_plan)[None, :, None]
        if live.any():
            t = tr[live]
            pxw = ox + c0 * t[:, 0] - s0_ * t[:, 1]
            pyw = oy + s0_ * t[:, 0] + c0 * t[:, 1]
            vxw = c0 * t[:, 2] - s0_ * t[:, 3]
            vyw = s0_ * t[:, 2] + c0 * t[:, 3]
            px = pxw[None, None, :] + vxw[None, None, :] * t_pred
            py = pyw[None, None, :] + vyw[None, None, :] * t_pred
            gaps.append(_box_gap(x[..., None], y[..., None], yaw[..., None], px, py, PED_R + 0.08))
        if len(self._obs_pts):
            px = self._obs_pts[:, 0][None, None, :]
            py = self._obs_pts[:, 1][None, None, :]
            gaps.append(_box_gap(x[..., None], y[..., None], yaw[..., None], px, py,
                                 self._obs_pts[:, 2][None, None, :]))
        if gaps:
            gap = np.concatenate(gaps, axis=2).min(axis=2)                   # (K, H)
            c_clear = self.w["clear"] * np.sum(np.maximum(self.comfort - gap, 0.0) ** 2, axis=1)
            c_col = self.w["col"] * np.sum(gap < 0.02, axis=1)
        else:
            c_clear = c_col = np.zeros(K)
        # falling/flipping (non-finite) rollouts are heavily penalised
        bad = ~np.isfinite(state).all(axis=(1, 2))
        total = c_shock + c_prog + c_lane + c_head + c_stab + c_u + c_clear + c_col
        return np.where(bad, 1e6, total)


@register("mppi")
def _make_mppi(**kw):
    return MPPIController(**kw)
