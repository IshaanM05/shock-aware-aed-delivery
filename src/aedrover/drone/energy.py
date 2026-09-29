"""Electrical energy model of the comparator quadrotor.

Rotor power comes from actuator-disc (momentum) theory with a figure of merit; a constant
avionics load and a battery model are added:

    P_i      = T_i^1.5 / sqrt(2 rho A) / FM              (electrical power of rotor i, W)
    P_total  = sum_i P_i + P_avionics
    E_usable = m_batt * e_spec * (1 - reserve_fraction)   (Wh)

with thrust T_i in N, disc area A = pi R^2 in m^2, air density rho in kg/m^3.

Sources and assumptions:
    * Battery specific energy 150 Wh/kg and an 80 percent usable depth of discharge (so a
      reserve fraction of 0.20) follow ``stolaroff2018``.
    * Figure of merit: ``stolaroff2018`` fits an OVERALL power efficiency of about 50 percent
      at low speed and about 70 percent at higher speed to measured multicopter data. The
      value 0.6 used here is the midpoint of that range. Mapping that overall efficiency onto
      a rotor figure of merit that also absorbs motor/ESC losses is an ASSUMPTION;
      sensitivity 0.5-0.7.
    * Avionics load 25 W (compute, radio, GPS, camera, release servo): ASSUMPTION,
      sensitivity 10-60 W.
    * Battery mass 2.0 kg (part of the airframe mass): ASSUMPTION, sensitivity 1.5-3.0 kg.

Limitations: hover-form momentum theory is applied at all airspeeds, so translational-lift
savings in forward flight are ignored (conservative, energy over-estimated at speed, which
is a bias AGAINST the drone); there is no battery voltage sag, no temperature effect and no
Peukert loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from aedrover.drone.quadrotor_mjcf import G, QuadParams

J_PER_WH = 3600.0


@dataclass(frozen=True)
class EnergyParams:
    """Parameters of the electrical energy model.

    Attributes:
        figure_of_merit: overall rotor figure of merit [-]. stolaroff2018 range 0.5-0.7;
            midpoint 0.6 (mapping is an ASSUMPTION).
        avionics_w: constant avionics load [W]. ASSUMPTION.
        battery_specific_energy_wh_per_kg: pack specific energy [Wh/kg]. stolaroff2018 (150).
        battery_mass_kg: battery mass [kg]. ASSUMPTION.
        reserve_fraction: fraction of capacity held in reserve [-]. stolaroff2018 (1 - 0.8).
        rotor_radius_m: rotor disc radius [m] (from QuadParams).
        n_rotors: number of rotors [-].
        air_density: air density [kg/m^3] (from QuadParams).
    """

    figure_of_merit: float = 0.6
    avionics_w: float = 25.0
    battery_specific_energy_wh_per_kg: float = 150.0
    battery_mass_kg: float = 2.0
    reserve_fraction: float = 0.20
    rotor_radius_m: float = 0.25
    n_rotors: int = 4
    air_density: float = 1.225

    @classmethod
    def from_quad(cls, quad: QuadParams, **kw: float) -> EnergyParams:
        """Energy parameters consistent with a quadrotor's rotor size and air density."""
        return cls(rotor_radius_m=quad.rotor_radius_m, air_density=quad.air_density, **kw)

    @property
    def disc_area_m2(self) -> float:
        return math.pi * self.rotor_radius_m**2

    @property
    def capacity_wh(self) -> float:
        """Nominal battery energy [Wh]."""
        return self.battery_mass_kg * self.battery_specific_energy_wh_per_kg

    @property
    def usable_wh(self) -> float:
        """Energy available to the mission after the reserve [Wh]."""
        return self.capacity_wh * (1.0 - self.reserve_fraction)

    @property
    def _k(self) -> float:
        """Power per T^1.5 [W / N^1.5]."""
        return 1.0 / (math.sqrt(2.0 * self.air_density * self.disc_area_m2) * self.figure_of_merit)


def rotor_power_w(thrusts_n: np.ndarray, ep: EnergyParams) -> float:
    """Total electrical power for per-rotor thrusts [N], including avionics [W]."""
    t = np.maximum(np.asarray(thrusts_n, dtype=float), 0.0)
    return float(ep._k * np.sum(t**1.5) + ep.avionics_w)


def thrust_power_w(total_thrust_n: float, ep: EnergyParams) -> float:
    """Electrical power for a collective thrust shared equally over all rotors [W]."""
    per = max(total_thrust_n, 0.0) / ep.n_rotors
    return float(ep.n_rotors * ep._k * per**1.5 + ep.avionics_w)


def hover_power_w(total_mass_kg: float, ep: EnergyParams) -> float:
    """Electrical power in hover [W]."""
    return thrust_power_w(total_mass_kg * G, ep)


def cruise_power_w(
    total_mass_kg: float,
    v_air_mps: float,
    cd_area_m2: float,
    ep: EnergyParams,
    rho: float | None = None,
    cross_wind_mps: float = 0.0,
) -> float:
    """Electrical power in steady level cruise [W].

    The thrust vector balances weight and drag, so |T| = sqrt((m g)^2 + D^2) with
    D = 0.5 rho CdA |v_rel|^2, where |v_rel|^2 = v_air^2 + cross_wind^2.
    """
    rho = ep.air_density if rho is None else rho
    d = 0.5 * rho * cd_area_m2 * (v_air_mps**2 + cross_wind_mps**2)
    return thrust_power_w(math.hypot(total_mass_kg * G, d), ep)


def max_range_m(
    v_cruise_mps: float,
    wind_along_track_mps: float,
    total_mass_kg: float,
    cd_area_m2: float,
    ep: EnergyParams,
    overhead_wh: float = 0.0,
    cross_wind_mps: float = 0.0,
) -> float:
    """Maximum one-way ground range at a commanded airspeed [m].

    ``range = (E_usable - overhead) / P_cruise * v_ground`` with
    ``v_ground = v_cruise + wind_along_track`` (positive wind_along_track = tailwind).

    Args:
        v_cruise_mps: commanded airspeed [m/s].
        wind_along_track_mps: wind component along the direction of travel [m/s];
            positive is a tailwind, negative a headwind.
        total_mass_kg: take-off mass including payload [kg].
        cd_area_m2: drag area Cd*A [m^2].
        ep: energy parameters.
        overhead_wh: energy spent on climb, descent and hover [Wh], subtracted from the budget.
        cross_wind_mps: crosswind component [m/s]; raises drag and hence power.

    Returns:
        Range [m]; 0 if the ground speed is not positive or the budget is exhausted.
    """
    v_ground = v_cruise_mps + wind_along_track_mps
    budget_j = (ep.usable_wh - overhead_wh) * J_PER_WH
    if v_ground <= 0.0 or budget_j <= 0.0:
        return 0.0
    p = cruise_power_w(total_mass_kg, v_cruise_mps, cd_area_m2, ep, cross_wind_mps=cross_wind_mps)
    return budget_j / p * v_ground


class EnergyMeter:
    """Integrates electrical energy from applied rotor thrusts during a simulation."""

    def __init__(self, ep: EnergyParams) -> None:
        self.ep = ep
        self._joule = 0.0

    def update(self, thrusts_n: np.ndarray, dt: float) -> None:
        """Add the energy of holding ``thrusts_n`` [N] for ``dt`` [s]."""
        self._joule += rotor_power_w(thrusts_n, self.ep) * dt

    @property
    def energy_wh(self) -> float:
        return self._joule / J_PER_WH

    @property
    def remaining_fraction(self) -> float:
        """Fraction of the usable battery energy still available [-]."""
        return 1.0 - self.energy_wh / self.ep.usable_wh
