"""Renderer-independent 3D overlay primitives (capsules, spheres, boxes) for data visualisation.

Both backends draw them as extra ``mjvGeom`` entries, so the same overlay code serves the PBR and
the classic renderer.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

_TYPES = {"capsule": mujoco.mjtGeom.mjGEOM_CAPSULE, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
          "sphere": mujoco.mjtGeom.mjGEOM_SPHERE, "box": mujoco.mjtGeom.mjGEOM_BOX}
_IDENT = np.eye(3).flatten()


@dataclass(frozen=True)
class Prim:
    """A line-like primitive from ``p0`` to ``p1`` (capsule/cylinder, ``radius`` thick) or a solid at ``p0``.

    ``sphere`` uses ``radius``; ``box`` uses ``half`` (half extents, axis aligned).
    """

    kind: str
    p0: tuple[float, float, float]
    p1: tuple[float, float, float] | None = None
    radius: float = 0.03
    rgba: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    half: tuple[float, float, float] = (0.05, 0.05, 0.05)


def fill_geom(geom: mujoco.MjvGeom, prim: Prim) -> None:
    """Initialise ``geom`` (an ``MjvGeom`` owned by a scene or created on the fly) from ``prim``."""
    rgba = np.asarray(prim.rgba, dtype=np.float32)
    p0 = np.asarray(prim.p0, dtype=np.float64)
    kind = _TYPES[prim.kind]
    if prim.kind in ("capsule", "cylinder"):
        if prim.p1 is None:
            raise ValueError(f"{prim.kind} needs p1")
        mujoco.mjv_initGeom(geom, kind, np.array([prim.radius, prim.radius, prim.radius]), np.zeros(3), _IDENT, rgba)
        mujoco.mjv_connector(geom, kind, prim.radius, p0, np.asarray(prim.p1, dtype=np.float64))
    elif prim.kind == "sphere":
        mujoco.mjv_initGeom(geom, kind, np.array([prim.radius, 0.0, 0.0]), p0, _IDENT, rgba)
    else:
        mujoco.mjv_initGeom(geom, kind, np.asarray(prim.half, dtype=np.float64), p0, _IDENT, rgba)
