"""AED quadrotor comparator ("Track D") for the aedrover ambulance / rover / drone comparison."""

from aedrover.drone.energy import EnergyMeter, EnergyParams, max_range_m
from aedrover.drone.flight_ctrl import CascadedController, ControllerGains, Reference
from aedrover.drone.mission import (
    MissionParams,
    MissionResult,
    can_reach,
    energy_wh,
    mission_time_s,
    p_available,
    simulate_mission,
    time_to_scene_s,
)
from aedrover.drone.quadrotor_mjcf import QuadParams, QuadSim, QuadState, build_mjcf
from aedrover.drone.wind import WindField, WindParams, drag_force

__all__ = [
    "CascadedController",
    "ControllerGains",
    "EnergyMeter",
    "EnergyParams",
    "MissionParams",
    "MissionResult",
    "QuadParams",
    "QuadSim",
    "QuadState",
    "Reference",
    "WindField",
    "WindParams",
    "build_mjcf",
    "can_reach",
    "drag_force",
    "energy_wh",
    "max_range_m",
    "mission_time_s",
    "p_available",
    "simulate_mission",
    "time_to_scene_s",
]
