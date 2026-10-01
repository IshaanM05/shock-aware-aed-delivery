"""The rover-versus-drone shot: numbers read from files, the dispatch timeline, camera rigs and the HUD."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("cv2", reason="the renderer tests need opencv (pip install -e '.[viz]')")

from aedrover.clinical.decision import ScenarioParams
from aedrover.drone.mission import MissionParams, mission_time_s, time_to_scene_s
from aedrover.viz import hud
from aedrover.viz.dispatch import (
    DispatchContext,
    DispatchShot,
    fit_rate,
    load_numbers,
    rig_aerial_follow,
    rig_arrival,
)
from aedrover.viz.drone_recording import DroneRecording, record_drone_mission
from aedrover.viz.drone_visuals import DroneSpec, Placement
from aedrover.viz.film import Film
from aedrover.viz.recording import record_episode

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def _synthetic(n: int = 11, dt: float = 0.5) -> DroneRecording:
    t = np.arange(n) * dt
    pos = np.stack([t * 2.0, np.zeros(n), t + 0.05], axis=1)
    quat = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return DroneRecording(dt=dt, t=t, pos=pos, quat=quat, thrust=np.full((n, 4), 20.0), meta={"distance_m": 20.0})


def _row(df, **kw):
    for k, v in kw.items():
        df = df[df[k] == v]
    assert len(df) == 1
    return df.iloc[0]


def _shot(**kw):
    n = load_numbers(RESULTS)
    args = {"name": "t", "rover": "r", "drone": DroneSpec(_synthetic(), Placement()), "numbers": n,
            "rig": rig_aerial_follow(), "frames": 120, "clock_start": 100.0}
    args.update(kw)
    return DispatchShot(**args), n


# ------------------------------------------------------------------------------------- numbers
def test_dispatch_numbers_equal_the_code_and_the_result_files():
    n = load_numbers(RESULTS)
    routes = pd.read_csv(RESULTS / "clinical_routes.csv")
    surv = pd.read_csv(RESULTS / "clinical_survival_vs_radius.csv")
    r = _row(routes, controller="ppo", radius_m=1000, density="assumed_4_per_km")
    assert n.rover_arrival_s == pytest.approx(60.0 * r.time_min_median) and n.rover_route_m == r.route_m
    assert n.rover_p_safe == r.p_safe_delivery
    for mode in ("ambulance", "rover", "drone"):
        s = _row(surv, controller="ppo", radius_m=1000, density="assumed_4_per_km", model="larsen1993", mode=mode)
        assert n.survival[mode] == (s.mean_survival, s.survival_ci_low, s.survival_ci_high)
    assert n.drone_launch_s == MissionParams().launch_latency_s
    assert n.drone_flight_s == mission_time_s(1000.0)
    assert n.drone_arrival_s == time_to_scene_s(1000.0)
    sp = ScenarioParams()
    assert n.clinical_drone_s == pytest.approx(60 * sp.drone_launch_min + 1000.0 / sp.drone_speed_mps)
    assert n.timing_gap_s == pytest.approx(n.drone_arrival_s - n.clinical_drone_s) and 60 < n.timing_gap_s < 80


def test_load_numbers_refuses_rows_that_are_not_there():
    with pytest.raises(ValueError, match="do not hold"):
        load_numbers(RESULTS, radius_m=123.0)


def test_the_recorded_flight_arrives_when_the_dispatch_clock_says():
    n = load_numbers(RESULTS)
    flight = record_drone_mission(n.radius_m)
    assert flight.meta["completed"]
    assert n.drone_launch_s + flight.duration == pytest.approx(n.drone_arrival_s, abs=0.1)


# ------------------------------------------------------------------------------------ timeline
def test_fit_rate_lands_the_clock_on_the_requested_end_and_the_clock_is_monotone():
    slow = (150.0, 2.0, 0.25)
    rate = fit_rate(120, 30, 100.0, 170.0, slow)
    shot, _ = _shot(rate=rate, slow=slow)
    clock = shot.clock(30)
    assert clock[0] == 100.0 and clock[-1] == pytest.approx(170.0, abs=1e-6) and np.all(np.diff(clock) > 0)
    assert np.diff(clock)[np.argmin(abs(clock - 150.0))] < np.diff(clock)[0]        # slower around the dip


def test_a_visible_rover_arrives_exactly_when_the_clock_reaches_its_arrival_time():
    n = load_numbers(RESULTS)
    fps, frames, rate = 30, 90, 1.6
    shot, _ = _shot(frames=frames, rate=rate, rover_visible=True, clock_start=n.rover_arrival_s - rate * (frames - 1) / fps)
    clock = shot.clock(fps)
    s = shot.rover_steps(clock, 1033, 0.02, fps)
    assert clock[-1] == pytest.approx(n.rover_arrival_s) and s[-1] == pytest.approx(1032.0)
    assert np.all(np.diff(s) > 0) and s[0] == pytest.approx(1032.0 - rate * (frames - 1) / fps / 0.02)
    hidden, _ = _shot(frames=frames)
    walk = hidden.rover_steps(hidden.clock(fps), 1033, 0.02, fps)
    assert walk[0] == 0.0 and np.all(np.diff(walk) > 0) and walk[-1] <= 1032.0


# ------------------------------------------------------------------------------------ captions
def test_hud_state_reads_every_number_from_the_numbers():
    shot, n = _shot(show_result=True, title="T", caption="C", footnote="F")
    early = shot.hud_state(0, 30, 120.0, drone_to_go_m=400.0, drone_released=False, rover_to_go_m=n.rover_route_m / 2)
    assert early.drone_frac == pytest.approx(0.6) and early.rover_frac == pytest.approx(0.5)
    assert early.drone_text == "400 m to go" and early.rover_text == f"{n.rover_route_m / 2:,.0f} m to go"
    assert early.result_alpha == 0.0 and early.drone_arrival_s == n.drone_arrival_s and early.rover_arrival_s == n.rover_arrival_s
    late = shot.hud_state(119, 30, 800.0, drone_to_go_m=0.0, drone_released=True, rover_to_go_m=0.0)
    assert late.drone_text == "arrived" and late.rover_text == "arrived" and late.result_alpha == 1.0
    shown = {label: value for label, value, _ in late.result_rows}
    assert shown == {"ambulance alone": f"{100 * n.survival['ambulance'][0]:.1f}%",
                     "ambulance + rover": f"{100 * n.survival['rover'][0]:.1f}%",
                     "ambulance + drone": f"{100 * n.survival['drone'][0]:.1f}%"}
    assert f"{n.timing_gap_s / 60:.1f} min" in late.result_note


def test_dispatch_hud_draws_at_any_size():
    shot, _ = _shot(show_result=True, title="Drone", caption="c", footnote="f")
    st = shot.hud_state(119, 30, 100.0, 5.0, False, 100.0)
    for size in ((320, 180), (1280, 720)):
        out = hud.draw_dispatch_hud(np.zeros((size[1], size[0], 3), np.uint8), st)
        assert out.shape == (size[1], size[0], 3) and out.dtype == np.uint8 and out.max() > 0
    assert hud.mmss(166.19) == "02:46" and hud.mmss(-3) == "00:00"


# --------------------------------------------------------------------------------------- rigs
def test_aerial_rig_stays_inside_the_street_and_looks_at_the_drone():
    n = 60
    drone = np.stack([np.linspace(0, 40, n), np.zeros(n), np.linspace(50, 2, n)], axis=1)
    ctx = DispatchContext(np.linspace(0, 1, n), np.arange(n, dtype=float), drone, 0.0, None, None)
    for pose, p in zip(rig_aerial_follow()(ctx), drone, strict=True):
        assert np.allclose(pose.target, p) and abs(pose.pos[1]) < 4.5            # building fronts stand at |y| = 5
        assert pose.pos[0] > p[0] and pose.pos[2] >= p[2]                        # ahead of the drone, never below it


def test_arrival_rig_sits_behind_the_rover_and_aims_toward_the_drone():
    rover = np.array([[30.0, 0.0, 0.2], [34.0, 0.0, 0.2]])
    ctx = DispatchContext(np.array([0.0, 1.0]), np.zeros(2), np.tile([36.0, 0.0, 2.0], (2, 1)), 0.0, rover, np.zeros(2))
    for pose, r in zip(rig_arrival(swing_deg=0.0, back=4.0, bias=0.5)(ctx), rover, strict=True):
        assert pose.pos[0] == pytest.approx(r[0] - 4.0)
        assert r[0] < pose.target[0] < 36.0 and pose.target[2] > r[2]


# ----------------------------------------------------------------------------------------- film
def test_film_renders_items_that_draw_themselves(tmp_path):
    class Item:
        name, frames = "self", 6

        def frames_from(self, film):
            for k in range(self.frames):
                yield np.full((36, 64, 3), 40 * k, np.uint8)

    out = Film({}, size=(64, 36), fps=10).render([Item()], tmp_path / "x.mp4", crf=30, verbose=False)
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.gpu
def test_dispatch_shot_renders_finished_frames_with_the_drone_in_view():
    from aedrover.viz.backend import filament_importable
    if not filament_importable():
        pytest.skip("MuJoCo has no Filament renderer")
    n = load_numbers(RESULTS)
    flight = record_drone_mission(n.radius_m)
    spec = DroneSpec(flight, Placement.arriving_at((36.0, 0.0), n.radius_m))
    shot = DispatchShot("gpu", "r", spec, n, rig_aerial_follow(), 3, clock_start=n.drone_launch_s + 100.0, title="t")
    film = Film({"r": record_episode("dwa", "mixed", 5010)}, size=(320, 180), fps=10)
    try:
        frames = list(shot.frames_from(film))
    except Exception as exc:                                    # no GPU / no OpenGL
        pytest.skip(f"cannot render here: {exc}")
    assert len(frames) == 3 and frames[1].shape == (180, 320, 3) and 20 < frames[1].mean() < 235
