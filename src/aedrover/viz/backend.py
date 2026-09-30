"""Rendering backends behind one small interface.

``FilamentBackend`` uses the physically based renderer that ships with MuJoCo 3.14 (headless OpenGL:
PBR materials, image-based lighting, soft shadows, reflections). ``ClassicBackend`` wraps
``mujoco.Renderer`` and is the fallback where no GPU or no Filament is available.

Engineering notes (verified on MuJoCo 3.14, see docs/RENDERING.md):

* a headless ``Window`` must exist before the Filament context is created (it registers the
  resource providers), and there is one engine per process;
* ``Renderer.get_image(...)`` returns an object whose ``__array_interface__`` is broken, so the raw
  ``.pixels`` bytes are read instead;
* ``ModelDecorations.update`` cannot be called from Python (it wants raw ``mjvOption_`` structs that
  the Python bindings only expose as wrapper classes), so 3D overlays are built into the render model
  as a pool of mocap geoms instead (see ``render_model.py``);
* Filament aborts the process if a model's GPU objects are destroyed while its renderables are
  still in the scene, so ``close()`` releases them in order;
* once a light has an ``intensity`` the scene is in physical units (see ``look.py`` for the
  calibrated constants); values of 1 to 100 render black.
"""

from __future__ import annotations

import itertools
import warnings

import mujoco
import numpy as np

from .camera import CameraPose

_COUNTER = itertools.count()
DEPTH_NEAR = 0.5
DEPTH_LOW = 4               # the depth pass is drawn at 1/4 resolution (it only drives smooth haze)
_DECODER = None              # measured once per process


def filament_importable() -> bool:
    try:
        import mujoco._render_filament  # noqa: F401
        from mujoco.experimental.studio import window  # noqa: F401
        from mujoco.rendering.filament import renderer  # noqa: F401
    except Exception:
        return False
    return True


class _Session:
    """One Filament engine (window, context, renderer) per process."""

    _inst: _Session | None = None

    @classmethod
    def get(cls) -> _Session:
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self) -> None:
        import mujoco._render_filament as mjrf
        from mujoco.experimental.studio import window
        from mujoco.rendering.filament import renderer

        self.mjrf = mjrf
        self.window = window.Window("aedrover", 64, 64, "opengl_headless")
        self.ctx = mjrf.Context(mjrf.ContextConfig(graphics_api=mjrf.GraphicsApi.GRAPHICS_API_OPENGL))
        self.renderer = renderer.Renderer(self.ctx)


class FilamentBackend:
    """PBR renderer. One instance renders one model at a fixed size."""

    name = "filament"

    def __init__(self, model: mujoco.MjModel, size: tuple[int, int] = (1280, 720), *, shadows: bool = True,
                 reflections: bool = True, post: bool = True, depth: bool = False) -> None:
        s = _Session.get()
        mjrf, r = s.mjrf, s.renderer
        self.model, self.size, self._s = model, size, s
        k = next(_COUNTER)
        self._scene, self._view, self._target = f"scn{k}", f"v{k}", f"out{k}"
        self._dview, self._dtarget = f"dv{k}", f"depth{k}"
        self._objects = mjrf.ModelObjects(s.ctx, model)
        scene = r.scene(self._scene)
        scene.configure_from_model(model)
        self._lights = mjrf.ModelLights(scene, self._objects)
        self._renderables = mjrf.ModelRenderables(scene, self._objects)
        r.target(self._target, size)
        r.view(self._view, scene=self._scene, target=self._target)
        req = r._views[self._view].request
        req.enable_shadows, req.enable_reflections, req.enable_post_processing = shadows, reflections, post
        self.has_depth = depth
        self._depth_entries = None
        if depth:
            # Filament reads back one target per render call, so the depth view lives outside the active
            # dictionaries and is swapped in for its own pass (see ``depth``).
            self.depth_size = (max(size[0] // DEPTH_LOW, 1), max(size[1] // DEPTH_LOW, 1))
            r.target(self._dtarget, self.depth_size, mjrf.PixelFormat.PIXEL_FORMAT_R32F)
            r.view(self._dview, scene=self._scene, target=self._dtarget, draw_mode=mjrf.DrawMode.DRAW_MODE_DEPTH)
            self._depth_entries = (r._views.pop(self._dview), r._reads.pop(self._dtarget))

    def _update(self, data: mujoco.MjData) -> None:
        self._lights.update(data)
        self._renderables.update(data)

    def render(self, data: mujoco.MjData, pose: CameraPose) -> np.ndarray:
        """RGB frame (height, width, 3) uint8 of ``data`` seen from ``pose``.

        ``data`` must have its kinematics computed (``mj_kinematics``) for the state to draw.
        """
        r = self._s.renderer
        self._update(data)
        r.update_camera(self._view, pose.to_gl())
        r.render()
        return self._read(self._target, 3)

    def depth_decoder(self):
        """A table mapping this renderer's 8-bit depth values to metres, measured with flat walls.

        The depth pass stores a non-linear, ``near``-dependent value (near 0.5 gives about 12 usable levels
        between 24 and 48 m), so it is calibrated rather than assumed.
        """
        from .post import DepthDecoder

        global _DECODER
        if _DECODER is not None:
            return _DECODER
        dists = np.geomspace(0.6, 300.0, 34)
        vals = []
        for dist in dists:
            xml = (f'<mujoco><visual><global offwidth="64" offheight="64"/></visual><asset><material name="f" rgba=".4 .4 .4 1"/></asset>'
                   f'<worldbody><light dir="0 0 -1"/><geom type="box" pos="{dist} 0 1" size=".1 800 800" material="f"/></worldbody></mujoco>')
            m = mujoco.MjModel.from_xml_string(xml)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            b = FilamentBackend(m, (256, 256), depth=True)
            try:
                v = b.depth(d, CameraPose((0, 0, 1), (10, 0, 1), 55))
                vals.append(float(v[v.shape[0] // 2, v.shape[1] // 2]))
            finally:
                b.close()
        _DECODER = DepthDecoder(dists, np.array(vals))
        return _DECODER

    def depth(self, data: mujoco.MjData, pose: CameraPose) -> np.ndarray:
        """Raw depth image (float32, at ``depth_size``, 0..1) of the same view; needs ``depth=True``.

        The camera near plane is 0.5 m for this pass (it sets the encoding); use ``depth_decoder``.
        """
        if self._depth_entries is None:
            raise RuntimeError("backend was created without depth=True")
        r = self._s.renderer
        saved_views, saved_reads = dict(r._views), dict(r._reads)
        r._views.clear()
        r._reads.clear()
        r._views[self._dview], r._reads[self._dtarget] = self._depth_entries
        try:
            self._update(data)
            r.update_camera(self._dview, pose.to_gl(near=DEPTH_NEAR))
            r.render()
            return self._read(self._dtarget, 1, np.float32)[..., 0]
        finally:
            r._views.clear()
            r._views.update(saved_views)
            r._reads.clear()
            r._reads.update(saved_reads)

    def _read(self, target: str, channels: int, dtype=np.uint8) -> np.ndarray:
        im = self._s.renderer.get_image(target)
        return np.frombuffer(im.pixels, dtype).reshape(im.height, im.width, channels).copy()

    def close(self) -> None:
        r = self._s.renderer
        for d, names in ((r._views, (self._view, self._dview)), (r._reads, (self._target, self._dtarget)),
                         (r._targets, (self._target, self._dtarget))):
            for n in names:
                d.pop(n, None)
        self._renderables = None          # renderables and lights leave the scene before their objects go
        self._lights = None
        r._scenes.pop(self._scene, None)
        self._objects = None


class ClassicBackend:
    """Fallback on ``mujoco.Renderer`` (fixed-function OpenGL; no PBR)."""

    name = "classic"

    def __init__(self, model: mujoco.MjModel, size: tuple[int, int] = (1280, 720), *, shadows: bool = True,
                 reflections: bool = True, post: bool = True, depth: bool = False) -> None:
        del post, depth
        w, h = size
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), w)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), h)
        self.model, self.size = model, size
        self._r = mujoco.Renderer(model, h, w)
        self._opt = mujoco.MjvOption()
        self._shadows, self._reflections = shadows, reflections
        self.has_depth = False

    def render(self, data: mujoco.MjData, pose: CameraPose) -> np.ndarray:
        self.model.vis.global_.fovy = pose.fovy
        self._r.update_scene(data, camera=pose.to_free(), scene_option=self._opt)
        scn = self._r.scene
        scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = self._shadows
        scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = self._reflections
        return self._r.render().copy()

    def close(self) -> None:
        self._r.close()


def make_backend(model: mujoco.MjModel, size: tuple[int, int] = (1280, 720), *, prefer: str = "filament",
                 **kw) -> FilamentBackend | ClassicBackend:
    """The PBR backend when it initialises here, otherwise the classic one (with a warning)."""
    if prefer == "filament":
        if filament_importable():
            try:
                return FilamentBackend(model, size, **kw)
            except Exception as exc:                      # no GPU / no GL: fall back rather than fail
                warnings.warn(f"Filament renderer unavailable ({type(exc).__name__}: {exc}); using the classic one",
                              stacklevel=2)
        else:
            warnings.warn("MuJoCo was built without the Filament renderer; using the classic one", stacklevel=2)
    return ClassicBackend(model, size, **kw)
