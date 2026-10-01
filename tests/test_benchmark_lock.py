"""Locks the standard benchmark world and one standard job to fixed reference values.

Anything that builds on the simulator (rich-world mode, drone visuals, rendering) must leave the default
environment alone. These tests fail if the default MJCF changes by a single character or if one standard
episode no longer reproduces, so an accidental change cannot slip into ``results/`` unnoticed.

The references were measured on the unmodified ``main`` (MuJoCo 3.14). The XML hash is exact and platform
independent. The episode values come from a contact simulation, so they carry small tolerances.
"""

from __future__ import annotations

import hashlib

import pytest

from aedrover.analysis.experiments import controller_spec
from aedrover.control import SafetyFilter
from aedrover.nav import make_controller
from aedrover.nav.base import run_episode
from aedrover.sim import AEDRoverEnv, VehicleParams
from aedrover.sim.world import build_xml

XML_SHA256 = {
    "optimized": ("130c6894d217ddd16703f4c5e60b50317cb3042a3f88d992ead2d0fc7871fa02", 10952),
    "default": ("f80ac02b6d1d3ea9d1f2c9e3a1ca8921e6aa19d381c6f578c3691bb7f27f5c99", 10847),
}


@pytest.mark.parametrize("name", sorted(XML_SHA256))
def test_default_world_xml_is_unchanged(name):
    veh = VehicleParams.optimized() if name == "optimized" else VehicleParams()
    xml = build_xml(veh)
    digest, length = XML_SHA256[name]
    assert len(xml) == length
    assert hashlib.sha256(xml.encode()).hexdigest() == digest


def test_one_standard_job_reproduces_its_reference_values():
    """Dynamic window at a 2.0 m/s cap, mixed family, seed 5010, optimized vehicle, safety filter on."""
    name, kw = controller_spec("dwa", 2.0)
    env = AEDRoverEnv(veh=VehicleParams.optimized(), obs_mode="dict")
    ep = run_episode(env, make_controller(name, **kw), SafetyFilter(), seed=5010, options={"family": "mixed"})
    assert ep["outcome"] == "goal"
    assert ep["time_s"] == pytest.approx(24.7, abs=0.05)
    assert ep["peak_shock_g"] == pytest.approx(3.0634, abs=0.05)
    assert ep["min_clearance_m"] == pytest.approx(0.2377, abs=0.02)
    assert ep["path_m"] == pytest.approx(36.17, abs=0.1)
