"""Record an episode's full visual state once, replay it from any camera later.

Only the quantities a renderer needs are stored, at the 50 Hz control rate: generalised positions,
mocap poses (kerb slabs, ramps, obstacles, pedestrians), shock, the commanded motion and the lidar
returns. The recording also keeps the exact physics MJCF, so a render model can be built from it
without re-deriving anything. Recording wraps ``env.step`` from outside: the simulator is unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..analysis.experiments import controller_spec
from ..control.safety_filter import SafetyFilter
from ..nav.base import make_controller
from ..sim.env import AEDRoverEnv
from ..sim.vehicle_mjcf import VehicleParams

REPO = Path(__file__).resolve().parents[3]


@dataclass
class Recording:
    xml: str                              # the exact physics MJCF (World.xml)
    dt: float                             # seconds between stored states (the control period)
    t: np.ndarray                         # (T,) time [s]
    qpos: np.ndarray                      # (T, nq)
    mocap_pos: np.ndarray                 # (T, nmocap, 3)
    mocap_quat: np.ndarray                # (T, nmocap, 4)
    shock_g: np.ndarray                   # (T,) peak payload shock within the step [g]
    cmd: np.ndarray                       # (T, 2) commanded speed [m/s] and steering [rad] after the safety filter
    lidar: np.ndarray                     # (T, n_rays) range returns [m]
    lidar_angles: np.ndarray              # (n_rays,) bearing of each ray in the body frame [rad]
    meta: dict = field(default_factory=dict)   # controller, family, seed, vehicle, outcome, scenario, episode metrics
    rollouts: dict | None = None          # optional planner debug data (see nav.mppi), keyed by step index

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"t": self.t, "qpos": self.qpos, "mocap_pos": self.mocap_pos, "mocap_quat": self.mocap_quat,
                  "shock_g": self.shock_g, "cmd": self.cmd, "lidar": self.lidar, "lidar_angles": self.lidar_angles}
        if self.rollouts:
            keys = sorted(self.rollouts)
            arrays["rollout_steps"] = np.array(keys, dtype=np.int64)
            for name in ("xy", "cost", "best"):
                arrays[f"rollout_{name}"] = np.stack([self.rollouts[k][name] for k in keys])
        np.savez_compressed(path, xml=np.array(self.xml), dt=np.array(self.dt),
                            meta=np.array(json.dumps(self.meta, default=float)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        z = np.load(path, allow_pickle=False)
        rollouts = None
        if "rollout_steps" in z:
            rollouts = {int(k): {n: z[f"rollout_{n}"][i] for n in ("xy", "cost", "best")}
                        for i, k in enumerate(z["rollout_steps"])}
        return cls(xml=str(z["xml"]), dt=float(z["dt"]), t=z["t"], qpos=z["qpos"], mocap_pos=z["mocap_pos"],
                   mocap_quat=z["mocap_quat"], shock_g=z["shock_g"], cmd=z["cmd"], lidar=z["lidar"],
                   lidar_angles=z["lidar_angles"], meta=json.loads(str(z["meta"])), rollouts=rollouts)


def _tuned_mppi() -> dict:
    p = REPO / "configs" / "mppi_tuned.json"
    return json.loads(p.read_text(encoding="utf-8"))["kwargs"] if p.exists() else {}


def record_episode(controller: str, family: str, seed: int, *, vehicle: str = "optimized",
                   speed_cap: float | None = 2.0, ppo_path: str = "models/ppo_selected", shield: bool = True,
                   max_time: float = 90.0, controller_kwargs: dict | None = None, capture_rollouts: bool = False,
                   scenario_kwargs: dict | None = None) -> Recording:
    """Run one episode exactly like the benchmark (same env, controller settings, safety filter) and record it."""
    extra = dict(controller_kwargs or {})
    if controller == "ppo":
        extra.setdefault("path", str((REPO / ppo_path) if not Path(ppo_path).is_absolute() else ppo_path))
    elif controller == "mppi":
        extra = {**_tuned_mppi(), **extra}
    name, kw = controller_spec(controller, speed_cap, **extra)
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name(vehicle), max_time=max_time)
    ctrl = make_controller(name, **kw)
    sf = SafetyFilter() if shield else None
    env.reset(seed=seed, options={"family": family, "scenario_kwargs": dict(scenario_kwargs or {})})
    ctrl.reset(env)
    if sf is not None:
        sf.reset()
    if capture_rollouts and hasattr(ctrl, "capture"):
        ctrl.capture = True
    d = env.world.data
    T, Q, MP, MQ, SH, CM, LD = [], [], [], [], [], [], []

    def snap(shock: float, v: float, dlt: float) -> None:
        T.append(env._t)
        Q.append(d.qpos.copy())
        MP.append(d.mocap_pos.copy())
        MQ.append(d.mocap_quat.copy())
        SH.append(shock)
        CM.append((v, dlt))
        LD.append(np.asarray(env.obs["lidar"], dtype=np.float32))

    snap(0.0, 0.0, 0.0)
    rollouts: dict[int, dict] = {}
    info: dict = {}
    while True:
        obs = env.obs
        v, dlt = ctrl.act(obs)
        if sf is not None:
            v, dlt = sf(obs, v, dlt)
        if capture_rollouts and getattr(ctrl, "last_rollouts", None) is not None:
            rollouts[len(T) - 1] = ctrl.last_rollouts
            ctrl.last_rollouts = None
        _, _, term, trunc, info = env.step(np.array([v, dlt]))
        snap(info["shock_g"], v, dlt)
        if term or trunc:
            break
    ep = info["episode"]
    meta = {"controller": controller, "family": family, "seed": seed, "vehicle": vehicle, "speed_cap": speed_cap,
            "outcome": ep["outcome"], "time_s": ep["time_s"], "peak_shock_g": ep["peak_shock_g"],
            "scenario": env.scenario.to_dict(), "lidar_range": float(env.perc.s.lidar_range), "budget_g": env.budget_g}
    rec = Recording(xml=env.world.xml, dt=env.dt, t=np.array(T), qpos=np.array(Q), mocap_pos=np.array(MP),
                    mocap_quat=np.array(MQ), shock_g=np.array(SH), cmd=np.array(CM), lidar=np.array(LD),
                    lidar_angles=np.asarray(env.obs["lidar_angles"], dtype=np.float32), meta=meta,
                    rollouts=rollouts or None)
    env.close()
    return rec
