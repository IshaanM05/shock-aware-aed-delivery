"""Offline tests of the OpenStreetMap route layer (synthetic graphs and the committed CSV).

Map data from OpenStreetMap contributors, ODbL 1.0. No test touches the network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("shapely")

from aedrover.geo.routes import (  # noqa: E402
    ATTRIBUTION,
    NMIMS_LAT,
    NMIMS_LON,
    OD_COLUMNS,
    REQUIRED_COLUMNS,
    AreaSpec,
    LocalFrame,
    build_major_road_index,
    build_od_table,
    count_major_road_crossings,
    describe,
    haversine_m,
    is_crossing_edge,
    is_crossing_node,
    is_grade_separated,
    kerb_classes,
    load_od_table,
    network_coverage,
    prepare_network,
    route_tag_features,
    sample_od_pairs,
    summarise_routes,
    tag_values,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "osm" / "vile_parle_od_routes.csv"
JSON_PATH = ROOT / "data" / "osm" / "vile_parle_summary.json"
DOC_PATH = ROOT / "docs" / "OSM_ROUTES.md"

FRAME = LocalFrame(NMIMS_LAT, NMIMS_LON)
AREA = AreaSpec(radius_m=350.0)


# ---------------------------------------------------------------------------------------------
# synthetic city: 9 x 9 grid, 100 m spacing, an east-west primary road at y = +50 m between
# grid rows 4 and 5, mapped with a node at every column. Columns 2 and 6 have mapped crossings.
# ---------------------------------------------------------------------------------------------
def make_graph(nodes: dict[int, tuple[float, float, dict]],
               edges: list[tuple[int, int, dict]]) -> nx.MultiDiGraph:
    """Bidirectional unsimplified MultiDiGraph from local-metre node positions."""
    g = nx.MultiDiGraph()
    for nid, (x, y, attrs) in nodes.items():
        lon, lat = FRAME.to_lonlat(x, y)
        g.add_node(nid, x=float(lon), y=float(lat), **attrs)
    for u, v, attrs in edges:
        a, b = g.nodes[u], g.nodes[v]
        length = float(haversine_m(a["x"], a["y"], b["x"], b["y"]))
        g.add_edge(u, v, length=length, **attrs)
        g.add_edge(v, u, length=length, **attrs)
    return g


def make_city(kind: str) -> nx.MultiDiGraph:
    """Synthetic "walk" or "drive" graph (see the module comment)."""
    walk = kind == "walk"
    nodes: dict[int, tuple[float, float, dict]] = {}
    for i in range(9):
        for j in range(9):
            attrs: dict[str, str] = {}
            if walk and (i, j) in {(2, 4), (2, 5)}:
                attrs["kerb"] = "lowered"
            if walk and (i, j) == (6, 4):
                attrs["kerb"] = "raised"
            if walk and (i, j) == (6, 5):
                attrs["kerb"] = "flush"
            nodes[100 * i + j] = ((i - 4) * 100.0, (j - 4) * 100.0, attrs)
        crossing = {"highway": "crossing"} if (walk and i in (2, 6)) else {}
        nodes[5000 + i] = ((i - 4) * 100.0, 50.0, crossing)
    edges: list[tuple[int, int, dict]] = []
    for i in range(9):
        for j in range(9):
            if i < 8:
                tags = {"highway": "footway", "footway": "sidewalk"} if walk else {
                    "highway": "residential"}
                edges.append((100 * i + j, 100 * (i + 1) + j, {**tags, "osmid": 10000 + j}))
            if j < 8 and j != 4:
                edges.append((100 * i + j, 100 * i + j + 1,
                              {"highway": "residential", "osmid": 20000 + i}))
            if j == 4:
                if walk and i in (2, 6):
                    tags = {"highway": "footway", "footway": "crossing", "osmid": 30000 + i}
                else:
                    tags = {"highway": "residential", "osmid": 20000 + i}
                edges.append((100 * i + 4, 5000 + i, tags))
                edges.append((5000 + i, 100 * i + 5, tags))
        if i < 8:
            edges.append((5000 + i, 5000 + i + 1, {"highway": "primary", "osmid": 999}))
    return make_graph(nodes, edges)


@pytest.fixture(scope="module")
def city():
    g_walk, g_drive = make_city("walk"), make_city("drive")
    walk = prepare_network(g_walk, FRAME, "walk")
    drive = prepare_network(g_drive, FRAME, "drive")
    return walk, drive, build_major_road_index(g_drive, FRAME), g_walk


SAMPLE_KW = {"min_m": 100.0, "max_m": 400.0, "max_snap_m": 120.0}


@pytest.fixture(scope="module")
def table(city):
    walk, drive, major, _ = city
    pairs = sample_od_pairs(walk, drive, AREA, 60, seed=3, **SAMPLE_KW)
    return build_od_table(pairs, walk, drive, major)


# ---------------------------------------------------------------------------------------------
# geometry and tag helpers
# ---------------------------------------------------------------------------------------------
def test_frame_round_trip_and_haversine():
    x, y = np.array([-1500.0, 0.0, 700.0]), np.array([300.0, 0.0, -1800.0])
    lon, lat = FRAME.to_lonlat(x, y)
    x2, y2 = FRAME.to_xy(lon, lat)
    assert np.allclose(x, x2, atol=1e-6) and np.allclose(y, y2, atol=1e-6)
    # local metres agree with the great-circle distance to better than 0.05 percent
    d = haversine_m(lon[0], lat[0], lon[2], lat[2])
    assert d == pytest.approx(np.hypot(x[0] - x[2], y[0] - y[2]), rel=5e-4)
    assert haversine_m(NMIMS_LON, NMIMS_LAT, NMIMS_LON, NMIMS_LAT) == 0.0


def test_tag_helpers():
    assert tag_values(None) == set() and tag_values(float("nan")) == set()
    assert tag_values("a;b") == {"a", "b"}
    assert tag_values("['footway', 'path']") == {"footway", "path"}
    assert tag_values(["x", "y;z"]) == {"x", "y", "z"}
    assert is_crossing_node({"highway": "crossing"})
    assert is_crossing_node({"highway": "traffic_signals", "crossing": "traffic_signals"})
    assert not is_crossing_node({"highway": "traffic_signals"})
    assert not is_crossing_node({"crossing": "no"})
    assert is_crossing_edge({"highway": "footway", "footway": "crossing"})
    assert not is_crossing_edge({"highway": "footway", "footway": "sidewalk"})
    assert kerb_classes({"kerb": "lowered"}) == {"lowered"} and kerb_classes({}) == set()
    assert is_grade_separated({"bridge": "yes"}) and not is_grade_separated({"bridge": "no"})


def test_describe_known_values():
    st = describe([1.0, 2.0, 3.0, 4.0, 5.0])
    assert st == {"mean": 3.0, "median": 3.0, "p10": pytest.approx(1.4), "p90": pytest.approx(4.6)}
    with pytest.raises(ValueError):
        describe([])


def test_prepare_network_rejects_simplified_graph():
    g = make_graph({1: (0.0, 0.0, {}), 2: (10.0, 0.0, {})}, [(1, 2, {"highway": "footway"})])
    for _, _, d in g.edges(data=True):
        d["geometry"] = "LINESTRING (0 0, 1 1)"
    with pytest.raises(ValueError, match="simplified"):
        prepare_network(g, FRAME, "walk")


# ---------------------------------------------------------------------------------------------
# features along a route
# ---------------------------------------------------------------------------------------------
def test_tagged_crossing_is_one_run_and_kerbs_are_counted(city):
    walk = city[0]
    path = [203, 204, 5002, 205, 206]  # across the mapped crossing in column 2
    f = route_tag_features(walk, path)
    assert f.n_crossings == 1  # crossing edges and the tagged node merge into one crossing
    assert (f.n_kerb_tagged, f.n_kerb_lowered, f.n_kerb_raised, f.n_kerb_flush) == (2, 2, 0, 0)
    assert f.graph_m == pytest.approx(300.0, rel=1e-3)
    f6 = route_tag_features(walk, [603, 604, 5006, 605, 606])
    assert (f6.n_crossings, f6.n_kerb_raised, f6.n_kerb_flush) == (1, 1, 1)


def test_untagged_crossing_and_pedestrian_share(city):
    walk = city[0]
    f = route_tag_features(walk, [303, 304, 5003, 305, 306])  # residential, no crossing tags
    assert f.n_crossings == 0 and f.n_kerb_tagged == 0 and f.footway_frac == 0.0
    along = route_tag_features(walk, [2, 102, 202, 302, 402])  # all footway edges (row 2)
    assert along.footway_frac == pytest.approx(1.0)
    assert along.n_crossings == 0


def test_two_crossings_in_a_row_stay_separate(city):
    walk = city[0]
    # column 2 up, along the row-5 footway to column 6, back down column 6
    path = [204, 5002, 205, 305, 405, 505, 605, 5006, 604]
    f = route_tag_features(walk, path)
    assert f.n_crossings == 2


def test_major_road_crossing_geometry(city):
    walk, _, major, _ = city
    assert major.way_ids == frozenset({999}) and len(major.lines) == 8
    # pass-through at a road node: counted, even where no crossing is tagged (column 3)
    assert count_major_road_crossings(walk, [303, 304, 5003, 305, 306], major) == 1
    assert count_major_road_crossings(walk, [203, 204, 5002, 205, 206], major) == 1
    # walking along the primary road itself is not a crossing
    assert count_major_road_crossings(walk, [5001, 5002, 5003, 5004], major) == 0
    # route ends on the road, or joins it and follows it: touching, not crossing
    assert count_major_road_crossings(walk, [303, 304, 5003], major) == 0
    assert count_major_road_crossings(walk, [303, 304, 5003, 5004, 5005], major) == 0
    # parallel to the road on one side
    assert count_major_road_crossings(walk, [3, 103, 203, 303, 403], major) == 0
    # two crossings (there and back) are two
    assert count_major_road_crossings(
        walk, [304, 5003, 305, 405, 5004, 404], major) == 2


def test_major_road_crossing_inside_an_edge_and_bridges():
    walk_g = make_graph({1: (0.0, -50.0, {}), 2: (0.0, 50.0, {})},
                        [(1, 2, {"highway": "residential", "osmid": 7})])
    bridge_g = make_graph({1: (0.0, -50.0, {}), 2: (0.0, 50.0, {})},
                          [(1, 2, {"highway": "footway", "osmid": 8, "bridge": "yes"})])
    road_g = make_graph({10: (-100.0, 0.0, {}), 11: (100.0, 0.0, {})},
                        [(10, 11, {"highway": "secondary", "osmid": 9})])
    flyover_g = make_graph({10: (-100.0, 0.0, {}), 11: (100.0, 0.0, {})},
                           [(10, 11, {"highway": "secondary", "osmid": 9, "bridge": "yes"})])
    major = build_major_road_index(road_g, FRAME)
    walk = prepare_network(walk_g, FRAME, "walk")
    assert count_major_road_crossings(walk, [1, 2], major) == 1  # crossing mid-edge
    assert count_major_road_crossings(walk, [2, 1], major) == 1  # direction does not matter
    bridge = prepare_network(bridge_g, FRAME, "walk")
    assert count_major_road_crossings(bridge, [1, 2], major) == 0  # footbridge is not at grade
    fly = build_major_road_index(flyover_g, FRAME)
    assert len(fly.lines) == 0 and count_major_road_crossings(walk, [1, 2], fly) == 0
    # a minor class is never counted
    minor_g = make_graph({10: (-100.0, 0.0, {}), 11: (100.0, 0.0, {})},
                         [(10, 11, {"highway": "residential", "osmid": 5})])
    assert count_major_road_crossings(walk, [1, 2], build_major_road_index(minor_g, FRAME)) == 0


# ---------------------------------------------------------------------------------------------
# coverage
# ---------------------------------------------------------------------------------------------
def test_network_coverage_on_synthetic_walk_graph():
    g = make_city("walk")
    for _, _, d in g.edges(data=True):
        if d.get("osmid") == 20000:  # column 0 vertical roads carry a sidewalk tag
            d["sidewalk"] = "both"
        if d.get("osmid") == 20008:  # column 8 vertical roads: explicitly none
            d["sidewalk"] = "no"
    cov = network_coverage(g)
    assert cov["n_crossing_nodes"] == 2 and cov["n_crossing_ways"] == 4  # 2 columns x 2 half-edges
    assert cov["n_kerb_tagged_nodes"] == 4
    assert cov["n_kerb_tagged_nodes_by_class"] == {"raised": 1, "lowered": 2, "flush": 1}
    assert cov["largest_component_node_share"] == 1.0

    seen, road, sw, no = set(), 0.0, 0.0, 0.0
    for u, v, d in g.edges(data=True):
        key = (min(u, v), max(u, v))
        if key in seen or d["highway"] == "footway":
            continue
        seen.add(key)
        road += d["length"]
        sw += d["length"] * (d.get("sidewalk") == "both")
        no += d["length"] * (d.get("sidewalk") == "no")
    assert cov["road_with_sidewalk_tag_share"] == pytest.approx(sw / road, abs=1e-4)
    assert cov["road_with_sidewalk_no_tag_share"] == pytest.approx(no / road, abs=1e-4)
    total = (cov["road_with_sidewalk_tag_share"] + cov["road_with_sidewalk_no_tag_share"]
             + cov["road_untagged_for_sidewalk_share"])
    assert total == pytest.approx(1.0, abs=2e-4)


# ---------------------------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------------------------
def test_sampling_is_deterministic_and_seeded(city):
    walk, drive, _, _ = city
    a = sample_od_pairs(walk, drive, AREA, 40, seed=11, **SAMPLE_KW)
    b = sample_od_pairs(walk, drive, AREA, 40, seed=11, **SAMPLE_KW)
    c = sample_od_pairs(walk, drive, AREA, 40, seed=12, **SAMPLE_KW)
    pd.testing.assert_frame_equal(a.frame, b.frame)
    assert a.tally == b.tally
    assert not a.frame[["o_lat", "o_lon"]].equals(c.frame[["o_lat", "o_lon"]])
    # a smaller n with the same seed is a prefix of a larger n
    small = sample_od_pairs(walk, drive, AREA, 10, seed=11, **SAMPLE_KW)
    pd.testing.assert_frame_equal(small.frame, a.frame.iloc[:10].reset_index(drop=True))


def test_sampled_pairs_respect_the_constraints(city):
    walk, drive, _, _ = city
    p = sample_od_pairs(walk, drive, AREA, 80, seed=5, **SAMPLE_KW)
    f = p.frame
    assert len(f) == 80 and list(f["pair_id"]) == list(range(80))
    assert f["straight_m"].between(100.0 * (1 - 1e-3), 400.0 * (1 + 1e-3)).all()
    for col in ("walk_o_snap_m", "walk_d_snap_m", "drive_o_snap_m", "drive_d_snap_m"):
        assert (f[col] <= 120.0).all()
    for prefix in ("o", "d"):  # endpoints stay inside the disc
        x, y = FRAME.to_xy(f[f"{prefix}_lon"], f[f"{prefix}_lat"])
        assert (np.hypot(x, y) <= AREA.radius_m * (1 + 1e-6)).all()
    t = p.tally
    assert t["accepted"] == 80
    assert t["proposals"] == (t["accepted"] + t["rejected_outside_disc"]
                              + t["rejected_walk_snap"] + t["rejected_drive_snap"])


def test_sampling_validation(city):
    walk, drive, _, _ = city
    with pytest.raises(ValueError):
        sample_od_pairs(walk, drive, AREA, 5, min_m=500.0, max_m=100.0)
    with pytest.raises(ValueError):
        sample_od_pairs(walk, drive, AREA, 0)
    with pytest.raises(RuntimeError):  # nothing can be within 1 m of a node at this rate
        sample_od_pairs(walk, drive, AREA, 5, max_snap_m=1e-6, max_batches=2, **{
            "min_m": 100.0, "max_m": 400.0})


# ---------------------------------------------------------------------------------------------
# OD table and summary
# ---------------------------------------------------------------------------------------------
def check_table_invariants(t: pd.DataFrame) -> None:
    assert list(t.columns[: len(REQUIRED_COLUMNS)]) == list(REQUIRED_COLUMNS)
    assert list(t.columns) == list(OD_COLUMNS)
    assert (t["walk_factor"] >= 1.0 - 1e-9).all()
    assert (t["drive_factor"] >= 1.0 - 1e-9).all()
    assert (t["walk_m"] >= t["straight_m"] - 1e-9).all()
    assert (t["drive_m"] >= t["straight_m"] - 1e-9).all()
    for col in ("n_crossings", "n_kerb_tagged", "n_crossings_major", "n_kerb_raised",
                "n_kerb_lowered", "n_kerb_flush"):
        assert pd.api.types.is_integer_dtype(t[col]), col
        assert (t[col] >= 0).all(), col
    assert (t["n_kerb_tagged"] >= t[["n_kerb_raised", "n_kerb_lowered", "n_kerb_flush"]].sum(
        axis=1)).all()
    assert np.allclose(t["walk_factor"], t["walk_m"] / t["straight_m"], rtol=1e-3)
    assert np.allclose(t["drive_factor"], t["drive_m"] / t["straight_m"], rtol=1e-3)
    assert np.allclose(t["walk_m"], t["walk_graph_m"] + t["walk_access_m"], atol=0.02)
    assert np.allclose(t["drive_m"], t["drive_graph_m"] + t["drive_access_m"], atol=0.02)
    assert t["walk_footway_frac"].between(0.0, 1.0).all()
    assert (t["osm_attribution"] == ATTRIBUTION).all()
    assert list(t["pair_id"]) == list(range(len(t)))


def test_synthetic_od_table_invariants(table):
    check_table_invariants(table)
    assert len(table) == 60
    # the synthetic city has a straight grid: walking a Manhattan route is longer than the
    # straight line, so the median factor sits strictly above 1
    assert table["walk_factor"].median() > 1.05
    assert table["n_crossings_major"].max() >= 1


def test_od_table_is_deterministic(city, table):
    walk, drive, major, _ = city
    pairs = sample_od_pairs(walk, drive, AREA, 60, seed=3, **SAMPLE_KW)
    pd.testing.assert_frame_equal(build_od_table(pairs, walk, drive, major), table)


def test_summary_shape(table):
    s = summarise_routes(table)
    assert s["n_pairs"] == 60
    for name in ("straight_m", "walk_m", "walk_factor", "drive_m", "drive_factor",
                 "n_crossings", "n_kerb_tagged", "n_crossings_major"):
        st = s["metrics"][name]
        assert set(st) == {"mean", "median", "p10", "p90"}
        assert st["p10"] <= st["median"] <= st["p90"]
        assert all(np.isfinite(v) for v in st.values())
    for label in ("crossings_tagged", "crossings_major", "kerb_tagged_nodes"):
        assert set(s["per_km"][label]) == {"mean", "median", "p10", "p90", "pooled"}
        assert s["per_km"][label]["pooled"] >= 0.0
    assert set(s["graph_only_factor"]) == {"walk", "drive"}
    for key in ("share_pairs_zero_crossings_tagged", "share_pairs_zero_crossings_major",
                "share_pairs_zero_kerb_tagged"):
        assert 0.0 <= s[key] <= 1.0
    json.dumps(s, allow_nan=False)  # serialisable, no NaN
    with pytest.raises(ValueError):
        summarise_routes(table.iloc[0:0])


def test_summary_pooled_rate_matches_definition(table):
    s = summarise_routes(table)
    pooled = table["n_crossings_major"].sum() / (table["walk_m"].sum() / 1000.0)
    assert s["per_km"]["crossings_major"]["pooled"] == pytest.approx(pooled)


# ---------------------------------------------------------------------------------------------
# the committed real-map outputs
# ---------------------------------------------------------------------------------------------
needs_data = pytest.mark.skipif(not CSV_PATH.exists() or not JSON_PATH.exists(),
                                reason="committed OSM outputs not present")


@needs_data
def test_committed_csv_invariants():
    t = load_od_table(CSV_PATH)
    check_table_invariants(t)
    assert (t["straight_m"].between(199.0, 2001.0)).all()


@needs_data
def test_committed_summary_matches_csv():
    t = load_od_table(CSV_PATH)
    stored = json.loads(JSON_PATH.read_text(encoding="ascii"))
    fresh = summarise_routes(t)
    assert stored["attribution"] == ATTRIBUTION
    assert stored["routes"]["n_pairs"] == len(t) == stored["sampling"]["n_pairs"]
    for name, st in fresh["metrics"].items():
        for k, v in st.items():
            assert stored["routes"]["metrics"][name][k] == pytest.approx(v, rel=1e-9, abs=1e-12)
    for name, st in fresh["per_km"].items():
        for k, v in st.items():
            assert stored["routes"]["per_km"][name][k] == pytest.approx(v, rel=1e-9, abs=1e-12)


@needs_data
def test_docs_quote_the_committed_numbers():
    """Headline numbers in docs/OSM_ROUTES.md are the ones in the committed summary."""
    if not DOC_PATH.exists():
        pytest.skip("docs/OSM_ROUTES.md not present")
    doc = DOC_PATH.read_text(encoding="ascii")
    stored = json.loads(JSON_PATH.read_text(encoding="ascii"))
    m = stored["routes"]["metrics"]
    assert ATTRIBUTION in doc
    for key in ("walk_factor", "drive_factor"):
        for stat in ("median", "p90"):
            assert f"{m[key][stat]:.2f}" in doc, f"{key} {stat} missing from the doc"
    for key in ("crossings_tagged", "crossings_major"):          # the doc quotes the pooled densities
        pooled = stored["routes"]["per_km"][key]["pooled"]
        assert f"pooled {pooled:.2f}" in doc, f"{key} pooled density missing from the doc"


# ---------------------------------------------------------------------------------------------
# repo policy: no currency, plain ASCII (patterns are assembled so this file stays clean too)
# ---------------------------------------------------------------------------------------------
_BANNED_WORDS = ["U" + "SD", "I" + "NR", "E" + "UR", "R" + "s"]
_BANNED_PARTS = ["doll" + "ar", "rup" + "ee"]
_BANNED_CHARS = [chr(36), chr(0x20AC), chr(0xA3), chr(0xA5), chr(0x20B9)]


def _policy_files() -> list[Path]:
    files = [ROOT / "src" / "aedrover" / "geo" / n for n in ("__init__.py", "routes.py")]
    files += [ROOT / "experiments" / "08_osm_routes.py", ROOT / "tests" / "test_geo.py",
              DOC_PATH, JSON_PATH, CSV_PATH]
    return [f for f in files if f.exists()]


@pytest.mark.parametrize("path", _policy_files(), ids=lambda p: p.name)
def test_no_currency_and_ascii_only(path: Path):
    raw = path.read_bytes()
    try:
        text = raw.decode("ascii")
    except UnicodeDecodeError as exc:
        pytest.fail(f"{path.name} contains non-ASCII bytes: {exc}")
    for ch in _BANNED_CHARS:
        assert ch not in text, f"{path.name} contains a currency symbol"
    for word in _BANNED_WORDS:
        assert not re.search(rf"\b{word}\b", text), f"{path.name} contains a currency code"
    lower = text.lower()
    for part in _BANNED_PARTS:
        assert part not in lower, f"{path.name} contains a currency word"
