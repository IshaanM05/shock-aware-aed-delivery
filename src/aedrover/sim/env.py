"""Gymnasium environment: one sidewalk segment with kerbs, pedestrians and obstacles.

Action  : [v_cmd (m/s, forward speed), delta_cmd (rad, bicycle steering angle)] in physical units.
Reward  : shaped for RL (progress, time, payload-shock excess, clearance, smoothness); classical
          controllers ignore it. The scientific results use ``info["episode"]`` only.
Episode : ends on goal reached, collision, rollover, leaving the sidewalk, stall or time limit.

All controllers see the same observation dict (``env.obs``), the flat RL vector is derived from it.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import mujoco
import numpy as np

from .metrics import G, ShockResult, rover_shock
from .pedestrians import PED_R, ROBOT_HALF_L, ROBOT_HALF_W, PedestrianCrowd, SFMParams
from .rover import Rover
from .scenario import CORRIDOR_HALF_WIDTH, Scenario, sample_scenario
from .sensors import Perception, PerceptionSpec
from .vehicle_mjcf import VehicleParams
from .world import OBS_SIZES, World, WorldSpec

BASE_POWER_W = 25.0       # ASSUMPTION: compute + sensors baseline electrical load


@dataclass(frozen=True)
class RewardConfig:
    w_progress: float = 1.0          # per metre of forward progress
    w_time: float = 0.05             # per second
    w_shock: float = 4.0             # per (g over budget)^2 per control step
    w_clear: float = 0.5             # per metre inside the comfort clearance
    comfort_clearance: float = 0.6   # m
    w_smooth: float = 0.02
    bonus_goal: float = 25.0
    pen_fail: float = 25.0
    pen_timeout: float = 5.0


class AEDRoverEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(self, family: str = "mixed", *, veh: VehicleParams | None = None,
                 world_spec: WorldSpec | None = None, control_dt: float = 0.02,
                 max_time: float = 90.0, budget_g: float = 3.0, v_max: float = 3.5,
                 delta_max: float = 0.5, perception: PerceptionSpec | None = None,
                 sfm: SFMParams | None = None, reward: RewardConfig | None = None,
                 obs_mode: str = "vector", render_size: tuple[int, int] = (480, 640),
                 log_every: int = 2, ped_every: int = 1):
        super().__init__()
        self.family = family
        self.veh0 = veh or VehicleParams()
        self.world = World(self.veh0, world_spec)
        self.rover = Rover(self.world, control_dt=control_dt, log_every=log_every)
        self.ped_every = ped_every
        self.perc = Perception(self.world.model, self.world.data, self.world.b_chassis,
                               self.veh0.nominal_height, perception)
        self.crowd = PedestrianCrowd(sfm)
        self.dt = self.rover.control_dt
        self.max_time = max_time
        self.budget_g = budget_g
        self.v_max, self.delta_max = v_max, delta_max
        self.rew = reward or RewardConfig()
        self.obs_mode = obs_mode
        self.render_size = render_size
        self._renderer: mujoco.Renderer | None = None
        self.scenario: Scenario | None = None
        self.obs: dict = {}

        self.action_space = gym.spaces.Box(np.array([0.0, -delta_max], dtype=np.float32),
                                           np.array([v_max, delta_max], dtype=np.float32))
        self._n_lidar = self.perc.s.n_lidar
        self._n_scan = len(self.perc.s.scan_fwd) * len(self.perc.s.scan_lat)
        self._k_tracks = 4
        dim = 3 + 5 + self._n_lidar + self._n_scan + 3 + 4 * self._k_tracks + 2
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)

        self._last_action = np.zeros(2)
        self._t = 0.0

    # ------------------------------------------------------------------ reset
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        options = options or {}
        sc: Scenario = options.get("scenario") or sample_scenario(
            options.get("family", self.family),
            int(seed if seed is not None else self.np_random.integers(2**31 - 1)),
            n_ped_max=self.world.spec.n_ped, n_obs_max=self.world.spec.n_obstacle,
            **options.get("scenario_kwargs", {}))
        self.scenario = sc
        self._configure_world(sc)
        self.rover.reset(0.0, sc.start_y, sc.surface_z(0.0), yaw=sc.start_yaw, settle_s=0.4)
        rng = np.random.default_rng(sc.seed + 7919)
        self._rng = rng
        self.crowd.reset(sc.peds, rng)
        self._sync_peds()
        self._t = 0.0
        self._last_action[:] = 0.0
        self._x_best, self._t_best = self.rover.pos[0], 0.0
        self._path_len = 0.0
        self._prev_xy = self.rover.pos[:2].copy()
        self._min_clear = np.inf
        self._energy_j = 0.0
        self._n_over = 0
        self._steps = 0
        self._speed_sum = 0.0
        self._max_step_g = 0.0
        self._nearest_kind = ""
        self._update_obs()
        return self._obs_out(), {"scenario": sc.to_dict()}

    def _configure_world(self, sc: Scenario) -> None:
        self.world.apply_scenario(sc)

    def _sync_peds(self) -> None:
        for i in range(self.crowd.n):
            x, y = self.crowd.pos[i]
            self.world.place_pedestrian(i, x, y, self.crowd.z_surface[i])

    # ------------------------------------------------------------------- step
    def step(self, action, want_obs: bool = True):
        a = np.asarray(action, dtype=float)
        v_cmd = float(np.clip(a[0], 0.0, self.v_max))
        d_cmd = float(np.clip(a[1], -self.delta_max, self.delta_max))
        rv, x0 = self.rover, self.rover.pos[0]
        n_log0 = len(rv.acc_log)

        rv.step(v_cmd, d_cmd)
        self._t += self.dt
        self._steps += 1

        xy = rv.pos[:2]
        if self._steps % self.ped_every == 0:
            self.crowd.step(self.dt * self.ped_every, xy, rv.body_velocity()[:2])
            for i in range(self.crowd.n):
                p = self.crowd.pos[i]
                self.world.move_pedestrian(i, p[0], p[1], self.crowd.z_surface[i])

        # per-step metrics
        new = np.asarray(rv.acc_log[n_log0:])
        up = np.asarray(rv.gdir_log[n_log0:])
        step_g = float(np.max(np.linalg.norm(new - G * up, axis=1)) / G) if len(new) else 0.0
        self._max_step_g = max(self._max_step_g, step_g)
        self._n_over += step_g > self.budget_g
        clear = self._clearance()
        self._min_clear = min(self._min_clear, clear)
        self._path_len += float(np.linalg.norm(xy - self._prev_xy))
        self._prev_xy = xy.copy()
        self._speed_sum += float(rv.body_velocity()[0])
        power = float(np.abs(rv.d.actuator_force[rv.a_drive] * rv.wheel_speeds()).sum()) + BASE_POWER_W
        self._energy_j += power * self.dt
        if rv.pos[0] > self._x_best + 0.5:
            self._x_best, self._t_best = rv.pos[0], self._t

        outcome = self._check_done(clear)
        terminated = outcome in ("goal", "collision", "rollover", "off_sidewalk", "nan")
        truncated = outcome in ("timeout", "stall")
        reward = self._reward(rv.pos[0] - x0, step_g, clear, a, outcome)
        self._last_action[:] = (v_cmd, d_cmd)
        if want_obs or terminated or truncated:
            self._update_obs()
        info = {"shock_g": step_g, "clearance": clear, "outcome": outcome or ""}
        if terminated or truncated:
            info["episode"] = self.episode_metrics(outcome or "timeout")
        return (self._obs_out() if (want_obs or terminated or truncated) else None), reward, terminated, truncated, info

    # ----------------------------------------------------------- bookkeeping
    def _clearance(self) -> float:
        """Gap between the rover's oriented rectangular footprint and the nearest pedestrian or
        obstacle [m] (negative = overlap = collision)."""
        xy = self.rover.pos[:2]
        yaw = self.rover.yaw_pitch_roll()[0]
        c, sn = np.cos(yaw), np.sin(yaw)

        def gap(px: np.ndarray, py: np.ndarray, r) -> np.ndarray:
            rx, ry = px - xy[0], py - xy[1]
            bx, by = c * rx + sn * ry, -sn * rx + c * ry
            dx = np.maximum(np.abs(bx) - ROBOT_HALF_L, 0.0)
            dy = np.maximum(np.abs(by) - ROBOT_HALF_W, 0.0)
            return np.hypot(dx, dy) - r

        best, kind = np.inf, ""
        if self.crowd.n:
            d = float(gap(self.crowd.pos[:, 0], self.crowd.pos[:, 1], PED_R).min())
            best, kind = d, "pedestrian"
        for ob in self.scenario.obstacles:
            d = float(gap(np.array([ob.x]), np.array([ob.y]), OBS_SIZES[ob.slot % 2][0])[0])
            if d < best:
                best, kind = d, "obstacle"
        self._nearest_kind = kind
        return best

    def _check_done(self, clear: float) -> str | None:
        rv, sc = self.rover, self.scenario
        if not rv.is_finite():
            return "nan"
        yaw, pitch, roll = rv.yaw_pitch_roll()
        if abs(roll) > 1.0 or abs(pitch) > 1.0:
            return "rollover"
        if clear < 0.0:
            return "collision"
        x, y, z = rv.pos
        if sc.has_kerb and (x <= sc.x_down or x >= sc.x_up) and abs(y) > CORRIDOR_HALF_WIDTH:
            return "off_sidewalk"
        if abs(y) > 6.0:
            return "off_sidewalk"
        if x >= sc.x_goal:
            return "goal"
        if self._t >= self.max_time:
            return "timeout"
        if self._t - self._t_best > 15.0:
            return "stall"
        return None

    def _reward(self, dx: float, step_g: float, clear: float, a: np.ndarray, outcome: str | None) -> float:
        c = self.rew
        r = c.w_progress * dx - c.w_time * self.dt
        r -= c.w_shock * max(0.0, step_g - self.budget_g) ** 2
        r -= c.w_clear * max(0.0, c.comfort_clearance - clear)
        r -= c.w_smooth * float(np.sum((a - self._last_action) ** 2))
        if outcome == "goal":
            r += c.bonus_goal
        elif outcome in ("collision", "rollover", "off_sidewalk", "nan"):
            r -= c.pen_fail
        elif outcome in ("timeout", "stall"):
            r -= c.pen_timeout
        return float(r)

    def episode_metrics(self, outcome: str) -> dict:
        sh: ShockResult = rover_shock(self.rover, budget_g=self.budget_g)
        n = max(self._steps, 1)
        return {
            "outcome": outcome,
            "success": outcome == "goal",
            "time_s": self._t,
            "path_m": self._path_len,
            "mean_speed": self._speed_sum / n,
            "peak_shock_g": sh.peak_g,
            "peak_shock_vertical_g": sh.peak_vertical_g,
            "shock_rms_g": sh.rms_g,
            "shock_over_budget": bool(sh.peak_g > self.budget_g),
            "frac_time_over_budget": sh.frac_over,
            "min_clearance_m": float(self._min_clear) if np.isfinite(self._min_clear) else float("nan"),
            "collision_kind": self._nearest_kind if outcome == "collision" else "",
            "energy_wh": self._energy_j / 3600.0,
            "steps": self._steps,
            **{f"sc_{k}": v for k, v in self.scenario.to_dict().items()},
        }

    # ----------------------------------------------------------- observation
    def _update_obs(self) -> None:
        rv, sc = self.rover, self.scenario
        yaw, pitch, roll = rv.yaw_pitch_roll()
        vb = rv.body_velocity()
        pos = rv.pos
        tracks = self.perc.track(pos[:2], yaw, self.crowd.pos, self.crowd.vel, self._rng, self._k_tracks)
        up = rv.d.xmat[self.world.b_chassis, 6:9]
        acc = rv.payload_acc()
        self.obs = {
            "t": self._t, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2]), "yaw": yaw,
            "pitch": pitch, "roll": roll, "vx": float(vb[0]), "vy": float(vb[1]),
            "wz": rv.yaw_rate(), "goal": np.array([sc.x_goal, 0.0]),
            "lidar": self.perc.lidar(yaw), "lidar_angles": self.perc.angles,
            "hscan": self.perc.height_scan(yaw), "tracks": tracks,
            "payload_dyn_g": (acc - G * up) / G,
        }

    def _obs_out(self):
        return self._obs_vector() if self.obs_mode == "vector" else self.obs

    def _obs_vector(self) -> np.ndarray:
        o = self.obs
        c, s = np.cos(o["yaw"]), np.sin(o["yaw"])
        dx, dy = o["goal"][0] - o["x"], o["goal"][1] - o["y"]
        gb = (c * dx + s * dy, -s * dx + c * dy)
        tr = o["tracks"][:, :4] * np.array([0.1, 0.1, 0.5, 0.5])
        vec = np.concatenate([
            [gb[0] / 10.0, gb[1] / 10.0, np.arctan2(gb[1], gb[0]) / np.pi],
            [o["vx"] / 3.5, o["vy"], o["wz"], o["roll"], o["pitch"]],
            o["lidar"] / self.perc.s.lidar_range,
            np.clip(o["hscan"].ravel() / 0.2, -1.5, 1.5),
            np.clip(o["payload_dyn_g"] / 3.0, -3.0, 3.0),
            tr.ravel(),
            self._last_action / np.array([self.v_max, self.delta_max]),
        ])
        return vec.astype(np.float32)

    # ---------------------------------------------------------------- render
    # dynamic cameras: (azimuth offset from heading [deg], elevation [deg], distance [m], look-ahead [m])
    _CAMS = {"chase": (35.0, -17.0, 4.0, 1.2), "side": (90.0, -9.0, 4.6, 0.6),
             "front": (200.0, -14.0, 4.0, 0.6), "top": (0.0, -88.0, 9.0, 1.5)}

    def render(self, camera: str = "chase") -> np.ndarray:
        """RGB frame. ``chase``/``side``/``front``/``top`` follow the rover; other names are MJCF cameras."""
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.world.model, *self.render_size)
        cam: object = camera
        if camera in self._CAMS:
            az, el, dist, ahead = self._CAMS[camera]
            yaw = self.rover.yaw_pitch_roll()[0]
            c = getattr(self, "_cam", None) or mujoco.MjvCamera()
            self._cam = c
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            p = self.rover.pos
            c.lookat[:] = (p[0] + ahead * np.cos(yaw), p[1] + ahead * np.sin(yaw), p[2] - 0.05)
            c.distance, c.elevation, c.azimuth = dist, el, np.degrees(yaw) + az
            cam = c
        self._renderer.update_scene(self.world.data, camera=cam)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
