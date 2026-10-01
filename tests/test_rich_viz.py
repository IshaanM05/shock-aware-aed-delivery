"""Rich-world dressing: the drawn furniture sits exactly on its collision proxies, and recordings carry the items."""

from __future__ import annotations

import re

import mujoco
import numpy as np
import pytest

pytest.importorskip("cv2", reason="the renderer tests need opencv (pip install -e '.[viz]')")

from aedrover.sim.furniture import Box, FurnitureItem
from aedrover.viz.dressing import dress_scene
from aedrover.viz.look import load_look
from aedrover.viz.recording import Recording, record_episode
from aedrover.viz.render_model import RenderXml

ITEMS = [
    FurnitureItem("car", 10.0, 2.0, 0.0, 0.0, (Box(0.0, 0.0, 2.15, 0.88, 1.45),), variant=3),
    FurnitureItem("bikes", 16.0, -1.5, np.pi / 2, 0.0, (Box(0.0, 0.0, 0.9, 0.45, 1.05),), variant=1, count=2),
    FurnitureItem("stall", 22.0, 1.6, 0.0, 0.12, (Box(0.0, 0.0, 0.7, 0.4, 0.9),), variant=2),
    FurnitureItem("lamp", 26.0, 1.2, 0.0, 0.0, (Box(0.0, 0.0, 0.08, 0.08, 3.0),)),
    FurnitureItem("tree", 30.0, -1.2, 0.0, 0.0, (Box(0.0, 0.0, 0.4, 0.4, 0.44), Box(0.0, 0.0, 0.14, 0.14, 3.0)), variant=1),
]


@pytest.fixture(scope="module")
def rich_rec() -> Recording:
    return record_episode("dwa", "flat_clear", 5001, furniture=ITEMS)


def _xml(rec: Recording, **kw) -> RenderXml:
    look = load_look()
    rx = RenderXml(rec.xml, look, (320, 180))
    rx.add_base()
    dress_scene(rx, rec, look, overlays=None, **kw)
    return rx


def _geoms(lines: list[str]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    out = []
    for ln in lines:
        m = re.search(r'<geom type="(\w+)" pos="([^"]+)" size="([^"]+)"', ln)
        if m:
            out.append((m.group(1), np.array(m.group(2).split(), float), np.array(m.group(3).split(), float)))
    return out


def test_recordings_carry_the_furniture_and_default_ones_do_not(rich_rec, tmp_path):
    assert [FurnitureItem.from_dict(d) for d in rich_rec.meta["furniture"]] == ITEMS
    again = Recording.load(rich_rec.save(tmp_path / "r.npz"))
    assert [FurnitureItem.from_dict(d) for d in again.meta["furniture"]] == ITEMS
    assert "furniture" not in record_episode("dwa", "flat_clear", 5001).meta
    assert rich_rec.xml.count('name="fur') == sum(len(i.boxes) for i in ITEMS)


def test_the_collision_proxies_are_hidden_and_every_item_is_drawn_on_its_proxy(rich_rec):
    plain, rich = _xml(rich_rec, furniture=[]), _xml(rich_rec)
    assert plain.build().count("<geom") < rich.build().count("<geom")
    model = mujoco.MjModel.from_xml_string(rich.build(), rich.files)
    for i, it in enumerate(ITEMS):
        for k in range(len(it.boxes)):
            g = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"fur{i}_{k}")
            assert g >= 0 and model.geom_size[g].max() < 0.01                      # shrunk to a point: physics, not looks
    added = _geoms([ln for ln in rich.build().splitlines() if ln not in set(plain.build().splitlines())])
    assert added

    def has(kind, x, y, z, size=None):
        return any(k == kind and np.allclose(p, (x, y, z), atol=1e-3) and (size is None or np.allclose(sz, size, atol=1e-3))
                   for k, p, sz in added)

    car, bikes, stall, lamp, tree = ITEMS
    assert has("box", car.x, car.y, car.z_base + 0.62, (2.15, 0.88, 0.36))                  # the car body is the proxy's footprint
    for j in range(2):                                                                     # a bike frame per bike, 0.6 m apart in x
        assert has("box", bikes.x + (j - 0.5) * 0.6, bikes.y, bikes.z_base + 0.55)
    assert has("box", stall.x, stall.y, stall.z_base + 0.45, (0.7, 0.4, 0.45))             # the cart, raised by the kerb height
    assert has("cylinder", lamp.x, lamp.y, 3.2, (0.06, 3.2))
    assert has("cylinder", tree.x, tree.y, 1.5, (0.14, 1.5)) and has("cylinder", tree.x, tree.y, 0.22, (0.4, 0.22))


def test_rich_dressing_keeps_the_physics_model_intact_and_is_deterministic(rich_rec):
    rx = _xml(rich_rec)
    model = mujoco.MjModel.from_xml_string(rx.build(), rx.files)
    phys = mujoco.MjModel.from_xml_string(rich_rec.xml)
    assert model.nq == phys.nq and model.nmocap >= phys.nmocap and model.nu == phys.nu
    assert _xml(rich_rec).build() == rx.build()                                           # same recording, same dressed model
