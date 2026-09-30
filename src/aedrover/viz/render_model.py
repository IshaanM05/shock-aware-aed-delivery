"""The dressed copy of the physics model that is used for drawing, and the replay of a recording on it.

The render model is built from the *unmodified* physics MJCF (the one stored in the recording) by
replacing its look-only blocks (``<visual>``, ``<asset>``, the sun) and adding visual-only content
(geoms with ``contype=0``, extra bodies appended last). Nothing is ever simulated on it: replay copies
the recorded ``qpos`` and mocap poses into ``MjData`` and runs kinematics, so every original degree of
freedom and mocap slot keeps its index and the dynamics cannot be affected.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import assets as A
from .backend import make_backend
from .camera import CameraPose
from .look import Look, load_look
from .recording import Recording

ASPHALT_TILE_M = 4.8      # one asphalt texture covers this many metres
PAVING_TILE_M = 2.4       # one paving texture (4 x 4 slabs) covers this many metres


def _texrepeat(tile_m: float) -> float:
    """With ``texuniform`` one texture tile spans ``2 / texrepeat`` metres (measured, see docs/RENDERING.md)."""
    return 2.0 / tile_m


def _pbr_material(name: str, prefix: str, tile_m: float, reflectance: float = 0.0) -> str:
    return (f'<material name="{name}" texrepeat="{_texrepeat(tile_m):.5g} {_texrepeat(tile_m):.5g}" texuniform="true" '
            f'metallic="0" roughness="1" reflectance="{reflectance}">'
            f'<layer texture="t_{prefix}_a" role="rgb"/><layer texture="t_{prefix}_n" role="normal"/>'
            f'<layer texture="t_{prefix}_o" role="orm"/></material>')


def _sub_block(xml: str, tag: str, new: str) -> str:
    out, n = re.subn(rf"<{tag}\b.*?</{tag}>", lambda _: new, xml, count=1, flags=re.S)
    if n != 1:
        raise ValueError(f"physics MJCF has no <{tag}> block to restyle")
    return out


@dataclass
class RenderXml:
    """Assembles the render MJCF: replaced look blocks plus injected visual-only content."""

    physics_xml: str
    look: Look
    size: tuple[int, int]
    files: dict[str, bytes] = field(default_factory=dict)
    asset_xml: list[str] = field(default_factory=list)
    world_xml: list[str] = field(default_factory=list)
    body_xml: dict[str, list[str]] = field(default_factory=dict)   # injected right after a named body's opening tag

    def add_base(self) -> None:
        look = self.look
        self.files.update(A.sky_faces(look))
        self.files.update(A.asphalt_maps())
        self.files.update(A.paving_maps())
        tex = ['<texture name="sky" type="skybox" fileright="right.png" fileleft="left.png" fileup="up.png" '
               'filedown="down.png" filefront="front.png" fileback="back.png"/>']
        for prefix, stem in (("asph", "asphalt"), ("pave", "paving")):
            tex += [f'<texture name="t_{prefix}_a" type="2d" file="{stem}_albedo.png"/>',
                    f'<texture name="t_{prefix}_n" type="2d" file="{stem}_normal.png"/>',
                    f'<texture name="t_{prefix}_o" type="2d" file="{stem}_orm.png"/>']
        self.asset_xml += tex
        self.asset_xml += [_pbr_material("road", "asph", ASPHALT_TILE_M),
                           _pbr_material("walk", "pave", PAVING_TILE_M)]

    def build(self) -> str:
        w, h = self.size
        xml = self.physics_xml
        xml = _sub_block(xml, "visual", f"""<visual>
    <global offwidth="{w}" offheight="{h}"/>
    <headlight ambient="0 0 0" diffuse="0 0 0" specular="0 0 0"/>
    <quality shadowsize="4096" offsamples="8"/>
    <map znear="0.004" zfar="40" shadowclip="1.0" shadowscale="0.6"/>
  </visual>""")
        xml = _sub_block(xml, "asset", "<asset>\n    " + "\n    ".join(self.asset_xml) + "\n  </asset>")
        ld = self.look.light_dir
        sun = (f'<light name="sun" type="directional" dir="{ld[0]:.5f} {ld[1]:.5f} {ld[2]:.5f}" '
               f'diffuse="{self.look.sun_color[0]} {self.look.sun_color[1]} {self.look.sun_color[2]}" '
               f'intensity="{self.look.sun_lux}" castshadow="true"/>\n    '
               f'<light name="sky_ibl" type="image" texture="sky" intensity="{self.look.ibl_intensity}"/>')
        xml, n = re.subn(r'<light name="sun"[^>]*/>', lambda _: sun, xml, count=1)
        if n != 1:
            raise ValueError("physics MJCF has no sun light to replace")
        for body, parts in self.body_xml.items():
            xml, n = re.subn(rf'(<body name="{re.escape(body)}"[^>]*>)', lambda m, p=parts: m.group(1) + "\n" + "\n".join(p),
                             xml, count=1)
            if n != 1:
                raise ValueError(f"physics MJCF has no body {body!r} to dress")
        if self.world_xml:
            xml = xml.replace("</worldbody>", "\n".join(self.world_xml) + "\n  </worldbody>", 1)
        return xml


def hide_physics_geoms(rx: RenderXml, sizes: dict[str, int]) -> None:
    """Shrink the named physics geoms to a point so dressed visuals replace them.

    ``sizes`` maps a geom name to how many size numbers its type takes (box 3, cylinder/capsule 2, sphere 1).
    Safe because the render model is never simulated; only kinematics run on it.
    """
    for name, n in sizes.items():
        rx.physics_xml, hit = re.subn(rf'(<geom name="{re.escape(name)}"[^>]*?size=")[^"]*(")',
                                      lambda m, k=n: m.group(1) + " ".join(["0.001"] * k) + m.group(2), rx.physics_xml, count=1)
        if hit != 1:
            raise ValueError(f"physics MJCF has no geom {name!r} with a size to hide")


# ------------------------------------------------------------------------------------ state replay
def _slerp(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.abs(dot)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    s = np.sin(theta)
    small = s < 1e-6
    a = np.where(small, 1.0 - w, np.sin((1.0 - w) * theta) / np.where(small, 1.0, s))
    b = np.where(small, w, np.sin(w * theta) / np.where(small, 1.0, s))
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def free_joint_quat_slices(model: mujoco.MjModel) -> list[slice]:
    """qpos slices holding unit quaternions (the free joints), which need slerp rather than linear blending."""
    out = []
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            a = int(model.jnt_qposadr[j])
            out.append(slice(a + 3, a + 7))
    return out


def interpolate_state(rec: Recording, s: float, quat_slices: list[slice]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``qpos``, ``mocap_pos``, ``mocap_quat`` at fractional step ``s`` (clamped to the recording)."""
    s = float(np.clip(s, 0.0, len(rec) - 1))
    i = min(int(np.floor(s)), len(rec) - 2) if len(rec) > 1 else 0
    w = s - i
    j = min(i + 1, len(rec) - 1)
    q = (1.0 - w) * rec.qpos[i] + w * rec.qpos[j]
    for sl in quat_slices:
        q[sl] = _slerp(rec.qpos[i, sl], rec.qpos[j, sl], w)
    mp = (1.0 - w) * rec.mocap_pos[i] + w * rec.mocap_pos[j]
    mq = _slerp(rec.mocap_quat[i], rec.mocap_quat[j], w)
    return q, mp, mq


class RenderScene:
    """A recording replayed on its render model through a backend."""

    def __init__(self, rec: Recording, look: Look | None = None, size: tuple[int, int] = (1920, 1080), *,
                 backend: str = "filament", builder=None, **backend_kw) -> None:
        self.rec, self.look, self.size = rec, look or load_look(), size
        rx = RenderXml(rec.xml, self.look, size)
        rx.add_base()
        self.animators = list(builder(rx, rec, self.look) or []) if builder is not None else []
        self.xml = rx.build()
        self.model = mujoco.MjModel.from_xml_string(self.xml, rx.files)
        self.data = mujoco.MjData(self.model)
        for a in self.animators:
            a.bind(self.model, rec)
        phys = mujoco.MjModel.from_xml_string(rec.xml)
        self._quat_slices = free_joint_quat_slices(phys)
        if self.model.nq != rec.qpos.shape[1] or self.model.nmocap < rec.mocap_pos.shape[1]:
            raise ValueError("render model does not share the physics model's degrees of freedom or mocap slots")
        self._n_mocap = rec.mocap_pos.shape[1]
        self.backend = make_backend(self.model, size, prefer=backend, **backend_kw)
        self.set_state(0.0)

    def set_state(self, s: float) -> None:
        """Put the model in the recorded state at fractional step ``s`` and run kinematics."""
        q, mp, mq = interpolate_state(self.rec, s, self._quat_slices)
        self.data.qpos[:] = q
        self.data.mocap_pos[: self._n_mocap] = mp
        self.data.mocap_quat[: self._n_mocap] = mq
        for a in self.animators:
            a.apply(self.data, s)
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def rover_pose(self, s: float) -> tuple[np.ndarray, float]:
        """Chassis position and yaw at fractional step ``s``."""
        q, _, _ = interpolate_state(self.rec, s, self._quat_slices)
        w, x, y, z = q[3:7]
        return q[:3].copy(), float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))

    def render(self, s: float, pose: CameraPose) -> np.ndarray:
        self.set_state(s)
        return self.backend.render(self.data, pose)

    def render_finished(self, s: float, pose: CameraPose, grade=None, seed: int = 0, dof: float = 0.0,
                        focus: float | None = None) -> np.ndarray:
        """A frame with haze (from depth), bloom, grade, vignette and grain applied (needs a depth-capable backend)."""
        from . import post

        self.set_state(s)
        frame = self.backend.render(self.data, pose)
        dist_low = None
        if getattr(self.backend, "has_depth", False):
            raw = self.backend.depth(self.data, pose)
            dist_low = post.low_distance(pose, self.size, raw, self.backend.depth_decoder())
        if dof > 0 and focus is None:
            rp, _ = self.rover_pose(s)
            focus = float(np.linalg.norm(np.asarray(pose.pos) - rp))
        return post.finish(frame, look=self.look, pose=pose, dist_low=dist_low, grade=grade, seed=seed, focus_m=focus, dof=dof)

    def close(self) -> None:
        self.backend.close()
