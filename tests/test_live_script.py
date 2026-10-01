"""The live demo script (`scripts/live_cinematic.py`): its session loop with the rover and the live drone together."""

from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2", reason="the renderer tests need opencv (pip install -e '.[viz]')")

ROOT = Path(__file__).resolve().parents[1]


def _script():
    try:
        spec = importlib.util.spec_from_file_location("live_cinematic", ROOT / "scripts" / "live_cinematic.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)                                  # imports tkinter, which a headless Linux box may lack
        return mod
    except ImportError as exc:
        pytest.skip(f"the live script cannot be imported here: {exc}")


def _args(**kw):
    base = dict(controller="dwa", family="crowded", seed=5021, rich=False, drone=True, drone_distance=None, drone_altitude=6.0, speed=1.0)
    base.update(kw)
    return types.SimpleNamespace(**base)


@pytest.mark.gpu
def test_a_session_flies_the_drone_and_drives_the_rover_together_and_draws_both():
    from aedrover.viz.backend import filament_importable
    if not filament_importable():
        pytest.skip("MuJoCo has no Filament renderer")
    lc = _script()
    from aedrover.sim.env import AEDRoverEnv
    from aedrover.sim.vehicle_mjcf import VehicleParams
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name("optimized"))
    try:
        ses = lc.Session(_args(), env, 5021, (320, 180))
    except Exception as exc:                                          # no GPU / no OpenGL
        pytest.skip(f"cannot build a GPU scene here: {exc}")
    try:
        frames, rover_x, drone_z, drone_x = [], [], [], []
        for k in range(0, 31):                                        # three seconds at 10 frames per second
            ses.advance(k / 10.0)
            frames.append(ses.frame())
            rover_x.append(float(env.obs["x"]))
            s = ses.stepper.state()
            drone_x.append(float(s.pos[0]))
            drone_z.append(float(s.pos[2]))
    finally:
        ses.close()
    assert frames[0].shape == (180, 320, 3) and frames[0].dtype == np.uint8 and 20 < frames[-1].mean() < 235
    assert rover_x[-1] > rover_x[0] + 1.0                             # the rover is driving
    assert drone_z[-1] > 4.0 and max(drone_x) > drone_x[0]            # and the drone has climbed to its cruise altitude and moved along
    assert np.abs(frames[0].astype(int) - frames[-1].astype(int)).mean() > 1.0
    assert ses.stepper.reason == "" and not ses.finished
