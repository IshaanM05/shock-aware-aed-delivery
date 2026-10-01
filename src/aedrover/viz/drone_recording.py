"""Record a simulated AED-drone mission once, replay it from any camera later.

The drone has its own MuJoCo model and simulator (``aedrover.drone``), so it is not merged into the rover's
physics. A mission is simulated once in the drone's own frame (start at the origin, climb, cruise along +x,
descend to the release point) and its pose and rotor thrusts are stored every control tick. The renderer draws
the drone from this recording the way it draws pedestrians: visual parts posed from stored arrays. Nothing is
simulated at render time.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..drone.mission import MissionParams, simulate_mission
from ..drone.quadrotor_mjcf import QuadParams
from .render_model import _slerp


@dataclass
class DroneRecording:
    dt: float                         # seconds between stored samples (the drone's control period)
    t: np.ndarray                     # (T,) time since liftoff [s]
    pos: np.ndarray                   # (T, 3) centre of mass in the drone's own frame [m]
    quat: np.ndarray                  # (T, 4) body orientation, (w, x, y, z)
    thrust: np.ndarray                # (T, 4) applied (motor-lagged) thrust per rotor [N]
    meta: dict = field(default_factory=dict)   # distance, wind, seed, outcome, times, QuadParams, MissionParams

    def __len__(self) -> int:
        return len(self.t)

    @property
    def duration(self) -> float:
        return float(self.t[-1] - self.t[0])

    def _index(self, t: float) -> tuple[int, int, float]:
        """Bracketing samples and blend weight for time ``t``, clamped to the recording."""
        n = len(self.t)
        if n == 1:
            return 0, 0, 0.0
        u = float(np.clip((t - self.t[0]) / self.dt, 0.0, n - 1.0))
        i = min(int(math.floor(u)), n - 2)
        return i, i + 1, u - i

    def pose_at(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """Position and unit quaternion at ``t`` seconds after liftoff.

        Clamped at both ends: before liftoff the drone sits on its first sample, after release it holds the last
        one (hovering at the release point).
        """
        i, j, w = self._index(t)
        pos = (1.0 - w) * self.pos[i] + w * self.pos[j]
        return pos, _slerp(self.quat[i], self.quat[j], w)

    def thrust_at(self, t: float) -> np.ndarray:
        i, j, w = self._index(t)
        return (1.0 - w) * self.thrust[i] + w * self.thrust[j]

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, dt=np.array(self.dt), t=self.t, pos=self.pos, quat=self.quat, thrust=self.thrust,
                            meta=np.array(json.dumps(self.meta, default=float)))
        return path

    @classmethod
    def load(cls, path: str | Path) -> DroneRecording:
        z = np.load(path, allow_pickle=False)
        return cls(dt=float(z["dt"]), t=z["t"], pos=z["pos"], quat=z["quat"], thrust=z["thrust"],
                   meta=json.loads(str(z["meta"])))


def record_drone_mission(distance_m: float, *, wind_mean_mps: float = 0.0, gust_sigma_mps: float = 0.0, seed: int = 0,
                         quad: QuadParams | None = None, mp: MissionParams | None = None) -> DroneRecording:
    """Simulate one mission (``aedrover.drone.mission.simulate_mission``) and keep its trajectory.

    Raises ``ValueError`` if the mission is infeasible (nothing was flown). A mission that starts but fails (lost
    tracking, battery) is still returned, with ``meta["completed"] = False`` and the reason.
    """
    quad, mp = quad or QuadParams(), mp or MissionParams()
    trace: list = []
    res = simulate_mission(distance_m, wind_mean_mps, gust_sigma_mps, seed=seed, quad=quad, mp=mp, trace=trace)
    if not trace:
        raise ValueError(f"mission over {distance_m:g} m with {wind_mean_mps:g} m/s along-track wind is infeasible ({res.reason})")
    dt = float(trace[1][0] - trace[0][0]) if len(trace) > 1 else 0.008
    meta = {"distance_m": float(distance_m), "wind_mean_mps": float(wind_mean_mps), "gust_sigma_mps": float(gust_sigma_mps),
            "seed": int(seed), "completed": bool(res.completed), "reason": res.reason,
            "flight_time_s": float(res.flight_time_s), "launch_latency_s": float(res.launch_latency_s),
            "total_time_s": float(res.total_time_s), "energy_wh": float(res.energy_wh),
            "peak_tracking_error_m": float(res.peak_tracking_error_m), "max_tilt_deg": float(res.max_tilt_deg),
            "quad": quad.to_dict(), "mission": asdict(mp)}
    return DroneRecording(dt=dt, t=np.array([r[0] for r in trace]), pos=np.array([r[1] for r in trace]),
                          quat=np.array([r[2] for r in trace]), thrust=np.array([r[3] for r in trace]), meta=meta)


def load_or_record(path: str | Path, distance_m: float, *, refresh: bool = False, **kw) -> DroneRecording:
    """The cached recording at ``path`` if there is one, otherwise simulate, save and return it."""
    path = Path(path)
    if path.exists() and not refresh:
        return DroneRecording.load(path)
    rec = record_drone_mission(distance_m, **kw)
    rec.save(path)
    return rec
