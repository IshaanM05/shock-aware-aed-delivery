"""Real-map route geometry: walking and driving routes around the NMIMS Mumbai campus.

Purpose: replace the assumed ``route_factor`` (path length / straight-line distance) and the
implicit "how many kerb crossings per kilometre" of the rover model by numbers measured on a
real street network (OpenStreetMap, via OSMnx).

Units: every length is in METRES, every count is dimensionless, every factor is a ratio.
Coordinates are WGS84 longitude / latitude in degrees; a small local frame (metres east and
north of the campus, equirectangular on a sphere of radius 6,371,009 m, the radius OSMnx uses
for edge lengths) is used only for snapping, sampling and crossing geometry.

Method (see docs/OSM_ROUTES.md for the full description):

1. Download the pedestrian ("walk") and drivable ("drive") networks inside a disc of the given
   radius, UNSIMPLIFIED, so that every node tag (highway=crossing, kerb=*) survives. Tags that
   OSMnx drops by default (kerb, crossing, footway, sidewalk, barrier ...) are added to its
   ``useful_tags_*`` settings first. Graphs are cached as graphml.
2. Keep the largest strongly connected component of each network for routing (the size of
   what is discarded is reported: fragmentation is itself a completeness indicator).
3. Sample origin-destination pairs: origin uniform in the disc, straight-line distance uniform
   in [200, 2000] m, bearing uniform, destination inside the disc. Both endpoints must lie
   within ``max_snap_m`` of a node of BOTH networks (this rejects points in the sea and in
   large unmapped compounds; the rejection tally is reported). Endpoints snap to the nearest
   node of each network.
4. Route length is door to door: shortest graph path (edge ``length`` weight) between the
   snapped nodes PLUS the straight access legs from each sampled point to its snapped node.
   By the triangle inequality this makes every route factor >= 1; the graph-only length is kept
   in a separate column.
5. Kerb-relevant features on the walking route:
   * ``n_crossings`` (estimate A, tag based): number of separate street crossings, where a
     crossing element is a node tagged highway=crossing (or any crossing=* other than "no")
     or an edge tagged footway=crossing, and consecutive elements merge into one crossing.
   * ``n_crossings_major`` (estimate B, geometric fallback): number of times the walking route
     crosses transversally a primary / secondary / tertiary / trunk road of the drive network.
     Edges on the same OSM way as the major road (walking along it) and bridge / tunnel edges
     (footbridges, flyovers) are not crossings. Dual carriageways count once per carriageway.
   * ``n_kerb_tagged`` and per-class counts: route nodes carrying a kerb=* tag.
6. Coverage of the walking network (sidewalk, crossing and kerb tagging) is computed from the
   full downloaded graph and reported next to the results.

Map data from OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import ast
import json
import math
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import networkx as nx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial import cKDTree
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree

FloatArray = NDArray[np.float64]
T = TypeVar("T")

ATTRIBUTION = "Map data from OpenStreetMap contributors, ODbL 1.0"
LICENCE_URL = "https://opendatacommons.org/licenses/odbl/1-0/"

EARTH_RADIUS_M = 6_371_009.0  # metres; the radius OSMnx uses for edge lengths

# Campus centre: Nominatim (through OSMnx) geocode of "NMIMS Mumbai", Vile Parle West.
NMIMS_LAT = 19.103442
NMIMS_LON = 72.836459

MAJOR_ROAD_CLASSES = frozenset({"primary", "secondary", "tertiary", "trunk"})
PEDESTRIAN_WAY_CLASSES = frozenset({"footway", "pedestrian", "path", "steps", "corridor"})
KERB_CLASSES = ("raised", "lowered", "flush")
SIDEWALK_KEYS = ("sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both")

# Extra OSM keys kept on nodes / ways (OSMnx drops everything not listed).
EXTRA_NODE_TAGS = ("barrier", "kerb", "crossing", "crossing:markings", "tactile_paving")
EXTRA_WAY_TAGS = (
    "footway", "sidewalk", "sidewalk:left", "sidewalk:right", "sidewalk:both", "kerb",
    "crossing", "crossing:markings", "tactile_paving", "surface", "layer", "foot",
)

# Columns of the OD table, required ones first (order is part of the contract).
REQUIRED_COLUMNS = (
    "pair_id", "straight_m", "walk_m", "walk_factor", "drive_m", "drive_factor",
    "n_crossings", "n_kerb_tagged",
)
EXTRA_COLUMNS = (
    "n_crossings_major", "n_kerb_raised", "n_kerb_lowered", "n_kerb_flush", "walk_graph_m",
    "drive_graph_m", "walk_access_m", "drive_access_m", "walk_footway_frac", "o_lat", "o_lon",
    "d_lat", "d_lon", "osm_attribution",
)
OD_COLUMNS = REQUIRED_COLUMNS + EXTRA_COLUMNS

SUMMARY_METRICS = (
    "straight_m", "walk_m", "walk_factor", "drive_m", "drive_factor", "n_crossings",
    "n_kerb_tagged", "n_crossings_major", "walk_footway_frac",
)


# ---------------------------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------------------------
def haversine_m(lon1: Any, lat1: Any, lon2: Any, lat2: Any) -> FloatArray:
    """Great-circle distance [m] between points given in degrees (haversine, radius 6,371,009 m).

    Args:
        lon1, lat1: first point(s), degrees.
        lon2, lat2: second point(s), degrees.

    Returns:
        Distance(s) in metres, broadcast over the inputs.
    """
    y1, y2 = np.radians(np.asarray(lat1, float)), np.radians(np.asarray(lat2, float))
    dy = y2 - y1
    dx = np.radians(np.asarray(lon2, float)) - np.radians(np.asarray(lon1, float))
    h = np.sin(dy / 2.0) ** 2 + np.cos(y1) * np.cos(y2) * np.sin(dx / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.minimum(1.0, h)))


@dataclass(frozen=True)
class LocalFrame:
    """Local metric frame centred on (lat0, lon0): x metres east, y metres north.

    Equirectangular on a sphere; the distortion within 2.5 km of the centre is of order 1e-4
    relative, far below the snapping tolerance. Route lengths never use this frame (they use
    haversine), it serves only for nearest-node search, sampling and crossing geometry.
    """

    lat0: float
    lon0: float

    def to_xy(self, lon: Any, lat: Any) -> tuple[FloatArray, FloatArray]:
        """Convert degrees to local metres (east, north)."""
        x = np.radians(np.asarray(lon, float) - self.lon0) * EARTH_RADIUS_M
        x = x * math.cos(math.radians(self.lat0))
        y = np.radians(np.asarray(lat, float) - self.lat0) * EARTH_RADIUS_M
        return x, y

    def to_lonlat(self, x: Any, y: Any) -> tuple[FloatArray, FloatArray]:
        """Convert local metres (east, north) back to degrees (lon, lat)."""
        lon = self.lon0 + np.degrees(np.asarray(x, float)
                                     / (EARTH_RADIUS_M * math.cos(math.radians(self.lat0))))
        lat = self.lat0 + np.degrees(np.asarray(y, float) / EARTH_RADIUS_M)
        return lon, lat


@dataclass(frozen=True)
class AreaSpec:
    """Study area: a disc around a centre point.

    Attributes:
        name: short identifier used in cache file names.
        query: free-text place name used to confirm the centre through Nominatim.
        lat, lon: disc centre, degrees.
        radius_m: disc radius [m].
    """

    name: str = "vile_parle"
    query: str = "NMIMS Mumbai"
    lat: float = NMIMS_LAT
    lon: float = NMIMS_LON
    radius_m: float = 2000.0

    @property
    def frame(self) -> LocalFrame:
        """The local metric frame centred on the disc."""
        return LocalFrame(self.lat, self.lon)

    def disc_polygon(self, n_vertices: int = 72):
        """Disc as a shapely polygon in (lon, lat) degrees, ``n_vertices`` vertices."""
        from shapely.geometry import Polygon

        ang = np.linspace(0.0, 2.0 * math.pi, n_vertices, endpoint=False)
        lon, lat = self.frame.to_lonlat(self.radius_m * np.cos(ang), self.radius_m * np.sin(ang))
        return Polygon(list(zip(lon.tolist(), lat.tolist(), strict=True)))


# ---------------------------------------------------------------------------------------------
# tag helpers
# ---------------------------------------------------------------------------------------------
def tag_values(value: Any) -> set[str]:
    """Normalise an OSM tag value to a set of strings.

    Accepts None, NaN, plain strings, ';'-separated strings, lists and stringified lists (the
    forms OSMnx and graphml round trips produce).
    """
    if value is None:
        return set()
    if isinstance(value, float) and math.isnan(value):
        return set()
    if isinstance(value, list | tuple | set | frozenset):
        out: set[str] = set()
        for item in value:
            out |= tag_values(item)
        return out
    text = str(value).strip()
    if text.startswith("[") and text.endswith("]"):
        try:
            return tag_values(ast.literal_eval(text))
        except (SyntaxError, ValueError):
            pass
    return {part.strip() for part in text.split(";") if part.strip()}


def _osm_ids(value: Any) -> set[int]:
    """OSM way ids from an ``osmid`` edge attribute (int, str, list or stringified list)."""
    ids: set[int] = set()
    for item in tag_values(value):
        try:
            ids.add(int(item))
        except ValueError:
            continue
    return ids


def is_crossing_node(attrs: dict[str, Any]) -> bool:
    """True if a node is a mapped street crossing (highway=crossing or crossing=* not "no")."""
    if "crossing" in tag_values(attrs.get("highway")):
        return True
    return bool(tag_values(attrs.get("crossing")) - {"no"})


def is_crossing_edge(attrs: dict[str, Any]) -> bool:
    """True if an edge is a mapped crossing way (footway=crossing, or legacy highway=crossing)."""
    return "crossing" in tag_values(attrs.get("footway")) | tag_values(attrs.get("highway"))


def kerb_classes(attrs: dict[str, Any]) -> set[str]:
    """Kerb tag values (raised / lowered / flush / rolled ...) carried by a node or edge."""
    return tag_values(attrs.get("kerb")) - {"no"}


def is_grade_separated(attrs: dict[str, Any]) -> bool:
    """True for bridge / tunnel edges (footbridges, flyovers): never an at-grade crossing."""
    return bool((tag_values(attrs.get("bridge")) | tag_values(attrs.get("tunnel"))) - {"no"})


def is_pedestrian_way(attrs: dict[str, Any]) -> bool:
    """True if the edge is a dedicated pedestrian way (footway, path, pedestrian, steps)."""
    return bool(tag_values(attrs.get("highway")) & PEDESTRIAN_WAY_CLASSES)


# ---------------------------------------------------------------------------------------------
# download and cache
# ---------------------------------------------------------------------------------------------
def _retry(fn: Callable[[], T], *, what: str, retries: int, backoff_s: float,
           log: Callable[[str], None]) -> T:
    """Call ``fn``; on failure wait ``backoff_s * 2**attempt`` seconds and retry."""
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as exc:  # network / Overpass errors come in many types
            if attempt == retries:
                raise
            wait = backoff_s * (2.0**attempt)
            log(f"  {what}: {type(exc).__name__}: {exc}; retry {attempt + 1}/{retries} in "
                f"{wait:.0f} s")
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def configure_osmnx(cache_dir: Path, *, use_http_cache: bool = True) -> Any:
    """Configure OSMnx (extra tags, cache folder, timeout, polite user agent) and return it.

    Args:
        cache_dir: directory for the OSMnx HTTP cache (under .cache, git-ignored).
        use_http_cache: False forces a fresh Overpass request (used by ``--refresh``).
    """
    import osmnx as ox

    ox.settings.use_cache = use_http_cache
    ox.settings.cache_folder = str(cache_dir / "http")
    ox.settings.log_console = False
    ox.settings.requests_timeout = 180
    ox.settings.http_user_agent = "aedrover-osm-routes/0.1 (academic research; uses OSMnx)"
    ox.settings.useful_tags_node = sorted(set(ox.settings.useful_tags_node) | set(EXTRA_NODE_TAGS))
    ox.settings.useful_tags_way = sorted(set(ox.settings.useful_tags_way) | set(EXTRA_WAY_TAGS))
    return ox


def graphml_path(area: AreaSpec, network_type: str, cache_dir: Path) -> Path:
    """Cache location of one network."""
    return cache_dir / f"{area.name}_r{int(area.radius_m)}_{network_type}.graphml"


def load_or_download_network(area: AreaSpec, network_type: str, cache_dir: Path, *,
                             refresh: bool = False, allow_download: bool = True,
                             retries: int = 4, backoff_s: float = 15.0,
                             log: Callable[[str], None] = print) -> nx.MultiDiGraph:
    """Load a cached OSM network or download it once (with retry and exponential backoff).

    The graph is unsimplified (every OSM node kept, edges are straight segments) and
    unprojected (x = longitude, y = latitude in degrees, ``length`` in metres).

    Args:
        area: study disc.
        network_type: "walk" or "drive" (OSMnx network types).
        cache_dir: directory holding the graphml cache (git-ignored).
        refresh: re-download even if a cache file exists (the old file is replaced only on
            success).
        allow_download: if False and no cache exists, raise ``FileNotFoundError``.
        retries: number of retries after the first failed download attempt.
        backoff_s: first wait [s]; doubled after each failure.
        log: progress sink.

    Returns:
        The MultiDiGraph. ``G.graph["created_date"]`` is the retrieval timestamp.
    """
    path = graphml_path(area, network_type, cache_dir)
    ox = configure_osmnx(cache_dir, use_http_cache=not refresh)
    if path.exists() and not refresh:
        log(f"  loading cached {network_type} network: {path.name}")
        return ox.load_graphml(path)
    if not allow_download:
        raise FileNotFoundError(f"no cached {network_type} graph at {path} and downloads are off")
    log(f"  downloading {network_type} network (one Overpass request, radius "
        f"{area.radius_m:.0f} m) ...")
    polygon = area.disc_polygon()
    graph = _retry(
        lambda: ox.graph_from_polygon(polygon, network_type=network_type, simplify=False,
                                      retain_all=True, truncate_by_edge=True),
        what=f"{network_type} download", retries=retries, backoff_s=backoff_s, log=log)
    path.parent.mkdir(parents=True, exist_ok=True)
    ox.save_graphml(graph, path)
    return graph


def confirm_centre(area: AreaSpec, cache_dir: Path, *, refresh: bool = False,
                   allow_network: bool = True,
                   log: Callable[[str], None] = print) -> dict[str, Any]:
    """Confirm the campus centre through Nominatim (cached; never fails the pipeline).

    Args:
        area: study disc (its ``query`` is geocoded).
        cache_dir: directory of the cached geocode result.
        refresh: geocode again even if a cached result exists.
        allow_network: if False, never contact Nominatim.
        log: progress sink.

    Returns:
        dict with the query, the geocoded lat / lon [deg] and the offset [m] to the configured
        centre; ``{"status": "unavailable"}`` if the geocoder cannot be reached and nothing is
        cached.
    """
    cache = cache_dir / f"{area.name}_geocode.json"
    if cache.exists() and not refresh:
        return json.loads(cache.read_text(encoding="ascii"))
    if not allow_network:
        return {"status": "unavailable", "query": area.query}
    try:
        ox = configure_osmnx(cache_dir, use_http_cache=not refresh)
        lat, lon = ox.geocode(area.query)
    except Exception as exc:  # geocoding is a confirmation only
        log(f"  geocode check unavailable: {type(exc).__name__}: {exc}")
        return {"status": "unavailable", "query": area.query}
    offset = float(haversine_m(area.lon, area.lat, lon, lat))
    out = {"status": "ok", "query": area.query, "lat": round(float(lat), 6),
           "lon": round(float(lon), 6), "offset_from_centre_m": round(offset, 1)}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=2), encoding="ascii")
    return out


# ---------------------------------------------------------------------------------------------
# prepared networks
# ---------------------------------------------------------------------------------------------
@dataclass
class RoutingNetwork:
    """A network prepared for routing and nearest-node snapping.

    Attributes:
        name: "walk" or "drive".
        full: the full downloaded MultiDiGraph (used for coverage statistics).
        core: simple DiGraph of the largest strongly connected component; each edge keeps the
            attributes of the shortest parallel edge; ``length`` in metres.
        node_ids: node ids of ``core`` (row order of ``lonlat`` and of the search tree).
        lonlat: (n, 2) node longitude, latitude [deg].
        xy: node -> (x, y) local metres.
        frame: local metric frame.
        tree: nearest-node search tree in local metres.
    """

    name: str
    full: nx.MultiDiGraph
    core: nx.DiGraph
    node_ids: NDArray[np.int64]
    lonlat: FloatArray
    xy: dict[int, tuple[float, float]]
    frame: LocalFrame
    tree: cKDTree = field(repr=False)

    def snap(self, lon: Any, lat: Any) -> tuple[NDArray[np.int64], FloatArray]:
        """Nearest core node of each point.

        Args:
            lon, lat: point coordinates [deg].

        Returns:
            (node ids, great-circle distance [m] from each point to its node).
        """
        x, y = self.frame.to_xy(lon, lat)
        _, idx = self.tree.query(np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]))
        nodes = self.node_ids[idx]
        dist = haversine_m(np.atleast_1d(lon), np.atleast_1d(lat), self.lonlat[idx, 0],
                           self.lonlat[idx, 1])
        return nodes, dist


def prepare_network(graph: nx.MultiDiGraph, frame: LocalFrame, name: str) -> RoutingNetwork:
    """Build a :class:`RoutingNetwork` from an unsimplified, unprojected OSMnx graph.

    Raises:
        ValueError: if the graph is empty or carries edge ``geometry`` (i.e. is simplified).
    """
    if graph.number_of_nodes() == 0:
        raise ValueError(f"{name}: empty graph")
    if any("geometry" in d for _, _, d in graph.edges(data=True)):
        raise ValueError(f"{name}: simplified graphs are not supported (use simplify=False)")
    keep = max(nx.strongly_connected_components(graph), key=len)
    core = nx.DiGraph()
    for node in keep:
        core.add_node(node, **graph.nodes[node])
    best: dict[tuple[int, int], dict[str, Any]] = {}
    for u, v, data in graph.edges(data=True):
        if u in keep and v in keep and u != v:
            cur = best.get((u, v))
            if cur is None or float(data["length"]) < float(cur["length"]):
                best[(u, v)] = data
    for (u, v), data in best.items():
        core.add_edge(u, v, **{**data, "length": float(data["length"])})
    ids = np.array(sorted(core.nodes), dtype=np.int64)
    lonlat = np.array([[float(core.nodes[n]["x"]), float(core.nodes[n]["y"])] for n in ids])
    x, y = frame.to_xy(lonlat[:, 0], lonlat[:, 1])
    xy = {int(n): (float(a), float(b)) for n, a, b in zip(ids, x, y, strict=True)}
    return RoutingNetwork(name, graph, core, ids, lonlat, xy, frame,
                          cKDTree(np.column_stack([x, y])))


# ---------------------------------------------------------------------------------------------
# major-road index (fallback crossing estimate)
# ---------------------------------------------------------------------------------------------
@dataclass
class MajorRoadIndex:
    """Major-road segments of the drive network for the geometric crossing estimate.

    Attributes:
        lines: segment geometries in local metres (at-grade segments only).
        tree: spatial index over ``lines``.
        way_ids: OSM way ids of every major-class way (walking along one is not a crossing).
        classes: highway classes counted as major.
        length_km: total at-grade major-road length [km], each segment counted once.
    """

    lines: list[LineString]
    tree: STRtree | None
    way_ids: frozenset[int]
    classes: frozenset[str]
    length_km: float


def build_major_road_index(drive_full: nx.MultiDiGraph, frame: LocalFrame,
                           classes: Iterable[str] = MAJOR_ROAD_CLASSES) -> MajorRoadIndex:
    """Index the primary / secondary / tertiary / trunk segments of the drive graph.

    Bridge and tunnel segments are left out of the geometry (a flyover does not cross the
    footpath beneath it at grade) but their way ids stay in ``way_ids``.
    """
    wanted = frozenset(classes)
    way_ids: set[int] = set()
    seen: set[tuple[int, int, int]] = set()
    lines: list[LineString] = []
    for u, v, data in drive_full.edges(data=True):
        if not (tag_values(data.get("highway")) & wanted):
            continue
        way_ids |= _osm_ids(data.get("osmid"))
        if is_grade_separated(data):
            continue
        key = (min(u, v), max(u, v), min(_osm_ids(data.get("osmid")) or {0}))
        if key in seen:
            continue
        seen.add(key)
        nu, nv = drive_full.nodes[u], drive_full.nodes[v]
        x, y = frame.to_xy([nu["x"], nv["x"]], [nu["y"], nv["y"]])
        lines.append(LineString(list(zip(x.tolist(), y.tolist(), strict=True))))
    tree = STRtree(lines) if lines else None
    return MajorRoadIndex(lines, tree, frozenset(way_ids), wanted,
                          sum(line.length for line in lines) / 1000.0)


def _points_of(geom: Any) -> list[tuple[float, float]]:
    """Coordinates of the point components of an intersection geometry (lines are ignored)."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Point":
        return [(geom.x, geom.y)]
    if hasattr(geom, "geoms"):
        return [p for g in geom.geoms for p in _points_of(g)]
    return []


def count_major_road_crossings(net: RoutingNetwork, path: Sequence[int], index: MajorRoadIndex,
                               *, offset_m: float = 5.0, merge_m: float = 3.0) -> int:
    """Estimate B: transversal crossings of a walking route with major roads.

    The route is split into runs of consecutive edges that are neither on a major-road way nor
    grade separated. Every point where a run meets a major-road segment is a crossing if the run
    passes from one side of the road to the other there (sign of the cross product of the road
    tangent with the route direction ``offset_m`` before and after the point). Touching
    points (route ends there, joins the road, or turns back) are not crossings; points closer
    than ``merge_m`` count once.

    Args:
        net: walking network (its ``core`` edges carry ``osmid``, ``bridge`` ...).
        path: node ids of the route.
        index: major-road index of the drive network.
        offset_m: half-window along the route used for the side test [m].
        merge_m: crossings closer than this are merged [m].

    Returns:
        Number of crossings (non-negative integer).
    """
    if index.tree is None or len(path) < 2:
        return 0
    runs: list[list[int]] = []
    current: list[int] = []
    for a, b in zip(path[:-1], path[1:], strict=True):
        data = net.core.edges[a, b]
        if is_grade_separated(data) or (_osm_ids(data.get("osmid")) & index.way_ids):
            if len(current) >= 2:
                runs.append(current)
            current = []
            continue
        if not current:
            current = [a]
        current.append(b)
    if len(current) >= 2:
        runs.append(current)

    count = 0
    for run in runs:
        line = LineString([net.xy[n] for n in run])
        length = line.length
        if length <= 2.0 * merge_m / 3.0:
            continue
        accepted: list[tuple[float, float]] = []
        for i in index.tree.query(line, predicate="intersects"):
            road = index.lines[int(i)]
            for px, py in _points_of(line.intersection(road)):
                if any(math.hypot(px - ax, py - ay) < merge_m for ax, ay in accepted):
                    continue
                s = line.project(Point(px, py))
                if s < 0.5 or s > length - 0.5:
                    continue
                d = min(offset_m, s, length - s)
                pb, pa = line.interpolate(s - d), line.interpolate(s + d)
                t0 = road.project(Point(px, py))
                r0 = road.interpolate(max(0.0, t0 - offset_m))
                r1 = road.interpolate(min(road.length, t0 + offset_m))
                tx, ty = r1.x - r0.x, r1.y - r0.y
                if math.hypot(tx, ty) < 1e-6:
                    continue
                c_before = tx * (pb.y - py) - ty * (pb.x - px)
                c_after = tx * (pa.y - py) - ty * (pa.x - px)
                if c_before * c_after < 0.0:
                    accepted.append((px, py))
                    count += 1
    return count


# ---------------------------------------------------------------------------------------------
# tag-based features along a route
# ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RouteFeatures:
    """Tag-based features of one walking route (estimate A).

    Attributes:
        n_crossings: separate mapped street crossings (merged runs of crossing elements).
        n_kerb_tagged: route nodes with any kerb=* tag other than "no".
        n_kerb_raised, n_kerb_lowered, n_kerb_flush: per-class node counts.
        footway_frac: share of the route's graph length on dedicated pedestrian ways [-].
        graph_m: graph length of the route [m].
    """

    n_crossings: int
    n_kerb_tagged: int
    n_kerb_raised: int
    n_kerb_lowered: int
    n_kerb_flush: int
    footway_frac: float
    graph_m: float


def route_tag_features(net: RoutingNetwork, path: Sequence[int]) -> RouteFeatures:
    """Count tagged crossings, kerb nodes and pedestrian-way share along a walking route.

    Elements are visited in route order (node, edge, node, ...). A maximal run of consecutive
    crossing elements (node with highway=crossing / crossing=*, edge with footway=crossing) is
    ONE crossing, so a crossing way plus its tagged nodes is not counted several times.
    """
    core = net.core
    flags: list[bool] = []
    kerb_counts = {"raised": 0, "lowered": 0, "flush": 0}
    n_kerb = 0
    total = ped = 0.0
    for i, node in enumerate(path):
        attrs = core.nodes[node]
        flags.append(is_crossing_node(attrs))
        classes = kerb_classes(attrs)
        if classes:
            n_kerb += 1
            for name in kerb_counts:
                kerb_counts[name] += int(name in classes)
        if i + 1 < len(path):
            data = core.edges[node, path[i + 1]]
            flags.append(is_crossing_edge(data))
            length = float(data["length"])
            total += length
            if is_pedestrian_way(data):
                ped += length
    runs = sum(1 for k, f in enumerate(flags) if f and (k == 0 or not flags[k - 1]))
    return RouteFeatures(runs, n_kerb, kerb_counts["raised"], kerb_counts["lowered"],
                         kerb_counts["flush"], ped / total if total > 0.0 else 0.0, total)


# ---------------------------------------------------------------------------------------------
# network coverage statistics
# ---------------------------------------------------------------------------------------------
def _unique_edge_table(graph: nx.MultiDiGraph) -> pd.DataFrame:
    """One row per undirected edge (deduplicated), with the tag columns used for coverage."""
    rows: dict[tuple[int, int, str], dict[str, Any]] = {}
    for u, v, data in graph.edges(data=True):
        key = (min(u, v), max(u, v), str(sorted(_osm_ids(data.get("osmid")))))
        if key not in rows:
            rows[key] = data
    records = []
    for data in rows.values():
        hw = sorted(tag_values(data.get("highway")))
        records.append({
            "length": float(data["length"]),
            "highway": hw[0] if hw else "",
            "pedestrian": is_pedestrian_way(data),
            "footway": ",".join(sorted(tag_values(data.get("footway")))),
            "crossing_edge": is_crossing_edge(data),
            "sidewalk_present": any(tag_values(data.get(k)) - {"no", "none"}
                                    for k in SIDEWALK_KEYS),
            "sidewalk_absent": (not any(tag_values(data.get(k)) - {"no", "none"}
                                        for k in SIDEWALK_KEYS))
            and any(tag_values(data.get(k)) & {"no", "none"} for k in SIDEWALK_KEYS),
            "kerb_edge": bool(kerb_classes(data)),
        })
    return pd.DataFrame.from_records(records)


def network_coverage(graph: nx.MultiDiGraph) -> dict[str, Any]:
    """Completeness of pedestrian-relevant tagging in the FULL walking graph.

    Lengths are per undirected edge (each street segment counted once), in km. Shares are of
    length unless stated. "Road" edges are those that are not dedicated pedestrian ways.

    Returns:
        dict with node / edge counts, largest-component shares, length by class, sidewalk tag
        shares on road edges, share of pedestrian ways mapped as sidewalk or crossing, and
        counts of crossing and kerb-tagged nodes.
    """
    edges = _unique_edge_table(graph)
    total = float(edges["length"].sum())
    road = edges[~edges["pedestrian"]]
    ped = edges[edges["pedestrian"]]
    road_len = float(road["length"].sum())
    ped_len = float(ped["length"].sum())
    comp = max(nx.weakly_connected_components(graph), key=len)
    comp_len = float(_unique_edge_table(graph.subgraph(comp).copy())["length"].sum())

    def share(mask_len: float, denom: float) -> float | None:
        return round(mask_len / denom, 4) if denom > 0 else None

    nodes = graph.nodes(data=True)
    n_cross = sum(1 for _, a in nodes if is_crossing_node(a))
    kerb_counts = {c: sum(1 for _, a in nodes if c in kerb_classes(a)) for c in KERB_CLASSES}
    n_kerb = sum(1 for _, a in nodes if kerb_classes(a))
    n_tactile = sum(1 for _, a in nodes if tag_values(a.get("tactile_paving")) - {"no"})
    by_class = edges.groupby("highway")["length"].sum().sort_values(ascending=False) / 1000.0
    return {
        "n_nodes": int(graph.number_of_nodes()),
        "n_edges_directed": int(graph.number_of_edges()),
        "n_edges_undirected": int(len(edges)),
        "total_length_km": round(total / 1000.0, 2),
        "largest_component_node_share": round(len(comp) / graph.number_of_nodes(), 4),
        "largest_component_length_share": share(comp_len, total),
        "length_km_by_highway_class": {k: round(float(v), 2) for k, v in by_class.items()},
        "pedestrian_way_length_share": share(ped_len, total),
        "road_length_km": round(road_len / 1000.0, 2),
        "road_with_sidewalk_tag_share": share(
            float(road.loc[road["sidewalk_present"], "length"].sum()), road_len),
        "road_with_sidewalk_no_tag_share": share(
            float(road.loc[road["sidewalk_absent"], "length"].sum()), road_len),
        "road_untagged_for_sidewalk_share": share(
            float(road.loc[~road["sidewalk_present"] & ~road["sidewalk_absent"], "length"].sum()),
            road_len),
        "pedestrian_ways_tagged_footway_key_share": share(
            float(ped.loc[ped["footway"] != "", "length"].sum()), ped_len),
        "crossing_way_length_km": round(float(edges.loc[edges["crossing_edge"], "length"].sum())
                                        / 1000.0, 3),
        "n_crossing_ways": int(edges["crossing_edge"].sum()),
        "n_crossing_nodes": int(n_cross),
        "n_kerb_tagged_nodes": int(n_kerb),
        "n_kerb_tagged_nodes_by_class": {k: int(v) for k, v in kerb_counts.items()},
        "n_kerb_tagged_edges": int(edges["kerb_edge"].sum()),
        "n_tactile_paving_nodes": int(n_tactile),
        "crossing_nodes_per_km_of_network": (round(n_cross / (total / 1000.0), 3)
                                             if total > 0 else None),
    }


# ---------------------------------------------------------------------------------------------
# OD sampling
# ---------------------------------------------------------------------------------------------
@dataclass
class OdPairs:
    """Accepted origin-destination pairs.

    Attributes:
        frame: DataFrame with one row per pair: sampled origin / destination lon and lat [deg],
            straight-line distance [m], snapped node ids and snap distances [m] per network.
        tally: proposals seen and rejection reasons (counts) for the consumed proposals.
    """

    frame: pd.DataFrame
    tally: dict[str, int]


def sample_od_pairs(walk: RoutingNetwork, drive: RoutingNetwork, area: AreaSpec, n: int, *,
                    seed: int = 0, min_m: float = 200.0, max_m: float = 2000.0,
                    max_snap_m: float = 200.0, batch: int = 1024,
                    max_batches: int = 200) -> OdPairs:
    """Draw ``n`` seeded origin-destination pairs and snap them to both networks.

    Origin: uniform over the disc area. Straight-line distance: uniform in [min_m, max_m].
    Bearing: uniform. A proposal is rejected if the destination falls outside the disc, or if
    any endpoint is farther than ``max_snap_m`` from the nearest walk node or drive node. The
    random stream is consumed in fixed batches, so the pairs for a given seed are a prefix of
    the pairs for a larger ``n`` with the same seed.

    Args:
        walk, drive: prepared networks.
        area: study disc (radius and centre).
        n: number of pairs to return.
        seed: seed of ``numpy.random.default_rng``.
        min_m, max_m: straight-line distance bounds [m], 0 < min_m < max_m.
        max_snap_m: maximum distance from an endpoint to the nearest node [m], per network.
        batch: proposals per random batch.
        max_batches: give up (``RuntimeError``) after this many batches.

    Returns:
        :class:`OdPairs`; ``frame`` has exactly ``n`` rows, ``pair_id`` 0..n-1.
    """
    if not 0.0 < min_m < max_m:
        raise ValueError("need 0 < min_m < max_m")
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    frame = area.frame
    R = area.radius_m
    tally = {"proposals": 0, "rejected_outside_disc": 0, "rejected_walk_snap": 0,
             "rejected_drive_snap": 0, "accepted": 0}
    kept: list[pd.DataFrame] = []
    got = 0
    for _ in range(max_batches):
        r = R * np.sqrt(rng.random(batch))
        th = 2.0 * math.pi * rng.random(batch)
        ox_, oy_ = r * np.cos(th), r * np.sin(th)
        dist = rng.uniform(min_m, max_m, batch)
        ph = 2.0 * math.pi * rng.random(batch)
        dx_, dy_ = ox_ + dist * np.cos(ph), oy_ + dist * np.sin(ph)
        inside = np.hypot(dx_, dy_) <= R
        o_lon, o_lat = frame.to_lonlat(ox_, oy_)
        d_lon, d_lat = frame.to_lonlat(dx_, dy_)
        w_o, w_os = walk.snap(o_lon, o_lat)
        w_d, w_ds = walk.snap(d_lon, d_lat)
        v_o, v_os = drive.snap(o_lon, o_lat)
        v_d, v_ds = drive.snap(d_lon, d_lat)
        walk_ok = (w_os <= max_snap_m) & (w_ds <= max_snap_m)
        drive_ok = (v_os <= max_snap_m) & (v_ds <= max_snap_m)
        ok = inside & walk_ok & drive_ok
        need = n - got
        cum = np.cumsum(ok)
        cut = int(np.searchsorted(cum, need)) + 1 if cum[-1] >= need else batch
        sl = slice(0, cut)
        tally["proposals"] += cut
        tally["rejected_outside_disc"] += int((~inside[sl]).sum())
        tally["rejected_walk_snap"] += int((inside & ~walk_ok)[sl].sum())
        tally["rejected_drive_snap"] += int((inside & walk_ok & ~drive_ok)[sl].sum())
        sel = np.flatnonzero(ok[sl])
        kept.append(pd.DataFrame({
            "o_lat": o_lat[sel], "o_lon": o_lon[sel], "d_lat": d_lat[sel], "d_lon": d_lon[sel],
            "straight_m": haversine_m(o_lon[sel], o_lat[sel], d_lon[sel], d_lat[sel]),
            "walk_o": w_o[sel], "walk_d": w_d[sel], "walk_o_snap_m": w_os[sel],
            "walk_d_snap_m": w_ds[sel], "drive_o": v_o[sel], "drive_d": v_d[sel],
            "drive_o_snap_m": v_os[sel], "drive_d_snap_m": v_ds[sel],
        }))
        got += len(sel)
        if got >= n:
            break
    else:
        raise RuntimeError(f"only {got} of {n} pairs accepted after {max_batches} batches; "
                           "the networks are too sparse for this snap limit")
    tally["accepted"] = got
    out = pd.concat(kept, ignore_index=True).iloc[:n].copy()
    out.insert(0, "pair_id", np.arange(n))
    return OdPairs(out, tally)


# ---------------------------------------------------------------------------------------------
# OD table
# ---------------------------------------------------------------------------------------------
def build_od_table(pairs: OdPairs, walk: RoutingNetwork, drive: RoutingNetwork,
                   major: MajorRoadIndex) -> pd.DataFrame:
    """Route every pair on both networks and count kerb-relevant features on the walk route.

    Route length is door to door: graph shortest path (edge ``length`` weight) plus the
    haversine access legs from each sampled point to its snapped node, so every factor is
    >= 1 up to floating-point rounding.

    Args:
        pairs: output of :func:`sample_od_pairs`.
        walk, drive: prepared networks.
        major: major-road index built from the drive graph.

    Returns:
        DataFrame with the columns :data:`OD_COLUMNS` (required columns first).
    """
    rows: list[dict[str, Any]] = []
    for rec in pairs.frame.itertuples(index=False):
        w_len, w_path = nx.bidirectional_dijkstra(walk.core, int(rec.walk_o), int(rec.walk_d),
                                                  weight="length")
        d_len, _ = nx.bidirectional_dijkstra(drive.core, int(rec.drive_o), int(rec.drive_d),
                                             weight="length")
        feats = route_tag_features(walk, w_path)
        w_acc = float(rec.walk_o_snap_m + rec.walk_d_snap_m)
        d_acc = float(rec.drive_o_snap_m + rec.drive_d_snap_m)
        walk_m = float(w_len) + w_acc
        drive_m = float(d_len) + d_acc
        rows.append({
            "pair_id": int(rec.pair_id),
            "straight_m": float(rec.straight_m),
            "walk_m": walk_m,
            "walk_factor": walk_m / float(rec.straight_m),
            "drive_m": drive_m,
            "drive_factor": drive_m / float(rec.straight_m),
            "n_crossings": feats.n_crossings,
            "n_kerb_tagged": feats.n_kerb_tagged,
            "n_crossings_major": count_major_road_crossings(walk, w_path, major),
            "n_kerb_raised": feats.n_kerb_raised,
            "n_kerb_lowered": feats.n_kerb_lowered,
            "n_kerb_flush": feats.n_kerb_flush,
            "walk_graph_m": float(w_len),
            "drive_graph_m": float(d_len),
            "walk_access_m": w_acc,
            "drive_access_m": d_acc,
            "walk_footway_frac": feats.footway_frac,
            "o_lat": float(rec.o_lat), "o_lon": float(rec.o_lon),
            "d_lat": float(rec.d_lat), "d_lon": float(rec.d_lon),
            "osm_attribution": ATTRIBUTION,
        })
    return pd.DataFrame(rows, columns=list(OD_COLUMNS))


# ---------------------------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------------------------
def describe(values: Any) -> dict[str, float]:
    """Mean, median, 10th and 90th percentile (linear interpolation) of a sample.

    Raises:
        ValueError: on an empty sample.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        raise ValueError("cannot summarise an empty sample")
    return {"mean": float(arr.mean()), "median": float(np.median(arr)),
            "p10": float(np.percentile(arr, 10)), "p90": float(np.percentile(arr, 90))}


def summarise_routes(table: pd.DataFrame) -> dict[str, Any]:
    """Summary statistics of an OD table (see :data:`SUMMARY_METRICS`).

    Returns:
        dict with ``n_pairs``; ``metrics`` (each: mean, median, p10, p90); ``per_km`` (per
        pair ratios per kilometre of walking route for the two crossing estimates and for kerb
        tagged nodes, each with the four statistics plus ``pooled`` = total count / total km);
        ``graph_only_factor`` (route factor without access legs, walk and drive); and shares of
        pairs with zero counted crossings or kerb nodes.
    """
    n = len(table)
    if n == 0:
        raise ValueError("empty OD table")
    km = table["walk_m"].to_numpy(float) / 1000.0
    per_km: dict[str, Any] = {}
    for label, col in (("crossings_tagged", "n_crossings"), ("crossings_major", "n_crossings_major"),
                       ("kerb_tagged_nodes", "n_kerb_tagged")):
        counts = table[col].to_numpy(float)
        per_km[label] = {**describe(counts / km), "pooled": float(counts.sum() / km.sum())}
    straight = table["straight_m"].to_numpy(float)
    return {
        "n_pairs": int(n),
        "metrics": {m: describe(table[m]) for m in SUMMARY_METRICS if m in table},
        "per_km": per_km,
        "graph_only_factor": {
            "walk": describe(table["walk_graph_m"].to_numpy(float) / straight),
            "drive": describe(table["drive_graph_m"].to_numpy(float) / straight),
        },
        "share_pairs_zero_crossings_tagged": float((table["n_crossings"] == 0).mean()),
        "share_pairs_zero_crossings_major": float((table["n_crossings_major"] == 0).mean()),
        "share_pairs_zero_kerb_tagged": float((table["n_kerb_tagged"] == 0).mean()),
    }


def load_od_table(path: Path) -> pd.DataFrame:
    """Read the committed OD table (columns per :data:`OD_COLUMNS`)."""
    return pd.read_csv(path)


# ---------------------------------------------------------------------------------------------
# pipeline
# ---------------------------------------------------------------------------------------------
@dataclass
class RouteStudy:
    """Result of :func:`run_pipeline`: the OD table and the JSON-serialisable summary."""

    table: pd.DataFrame
    summary: dict[str, Any]


def run_pipeline(area: AreaSpec, cache_dir: Path, *, n_pairs: int = 500, seed: int = 0,
                 min_m: float = 200.0, max_m: float = 2000.0, max_snap_m: float = 200.0,
                 refresh: bool = False, allow_download: bool = True, retries: int = 4,
                 backoff_s: float = 15.0, log: Callable[[str], None] = print) -> RouteStudy:
    """Download (or load) both networks, sample pairs, route them and summarise.

    Args:
        area: study disc.
        cache_dir: graphml / HTTP cache directory (git-ignored).
        n_pairs: number of OD pairs.
        seed: RNG seed for the pair sampling.
        min_m, max_m: straight-line distance bounds [m].
        max_snap_m: maximum endpoint-to-node distance [m].
        refresh: re-download the networks.
        allow_download: permit network access when a cache file is missing.
        retries, backoff_s: download retry policy (exponential backoff, seconds).
        log: progress sink.

    Returns:
        :class:`RouteStudy` with the OD table and the summary dict.
    """
    import importlib.metadata as md

    log(f"area: {area.query} ({area.lat:.6f}, {area.lon:.6f}), radius {area.radius_m:.0f} m")
    geocode = confirm_centre(area, cache_dir, refresh=refresh, allow_network=allow_download,
                             log=log)
    g_walk = load_or_download_network(area, "walk", cache_dir, refresh=refresh,
                                      allow_download=allow_download, retries=retries,
                                      backoff_s=backoff_s, log=log)
    g_drive = load_or_download_network(area, "drive", cache_dir, refresh=refresh,
                                       allow_download=allow_download, retries=retries,
                                       backoff_s=backoff_s, log=log)
    frame = area.frame
    log("preparing networks ...")
    walk = prepare_network(g_walk, frame, "walk")
    drive = prepare_network(g_drive, frame, "drive")
    major = build_major_road_index(g_drive, frame)
    log(f"  walk core {walk.core.number_of_nodes()} nodes / {walk.core.number_of_edges()} edges; "
        f"drive core {drive.core.number_of_nodes()} nodes / {drive.core.number_of_edges()} edges; "
        f"major-road segments {len(major.lines)}")
    log(f"sampling {n_pairs} pairs (seed {seed}) and routing ...")
    pairs = sample_od_pairs(walk, drive, area, n_pairs, seed=seed, min_m=min_m, max_m=max_m,
                            max_snap_m=max_snap_m)
    table = build_od_table(pairs, walk, drive, major)
    drive_cov = {
        "n_nodes": int(g_drive.number_of_nodes()),
        "n_edges_directed": int(g_drive.number_of_edges()),
        "core_nodes": int(drive.core.number_of_nodes()),
        "core_edges_directed": int(drive.core.number_of_edges()),
        "major_road_segments_at_grade": len(major.lines),
        "major_road_length_km": round(major.length_km, 2),
    }
    walk_cov = network_coverage(g_walk)
    walk_cov["core_nodes"] = int(walk.core.number_of_nodes())
    walk_cov["core_edges_directed"] = int(walk.core.number_of_edges())
    summary: dict[str, Any] = {
        "attribution": ATTRIBUTION,
        "licence": f"Open Database License 1.0, {LICENCE_URL}",
        "area": {"name": area.name, "query": area.query, "centre_lat": area.lat,
                 "centre_lon": area.lon, "radius_m": area.radius_m,
                 "geocode_check": geocode},
        "osm_retrieved": {"walk": str(g_walk.graph.get("created_date", "unknown")),
                          "drive": str(g_drive.graph.get("created_date", "unknown"))},
        "software": {"osmnx": md.version("osmnx"), "networkx": nx.__version__,
                     "graph_created_with": str(g_walk.graph.get("created_with", "unknown"))},
        "sampling": {"n_pairs": n_pairs, "seed": seed, "straight_min_m": min_m,
                     "straight_max_m": max_m, "max_snap_m": max_snap_m, **pairs.tally,
                     "walk_snap_m": describe(pd.concat([pairs.frame["walk_o_snap_m"],
                                                        pairs.frame["walk_d_snap_m"]])),
                     "drive_snap_m": describe(pd.concat([pairs.frame["drive_o_snap_m"],
                                                         pairs.frame["drive_d_snap_m"]]))},
        "walk_network": walk_cov,
        "drive_network": drive_cov,
        "routes": summarise_routes(table),
        "definitions": {
            "walk_m": "door to door: graph shortest path + straight access legs [m]",
            "walk_factor": "walk_m / straight_m",
            "n_crossings": "estimate A (tags): merged runs of highway=crossing nodes / "
                           "footway=crossing edges on the walking route",
            "n_crossings_major": "estimate B (geometry): transversal crossings of primary, "
                                 "secondary, tertiary, trunk roads by the walking route",
            "n_kerb_tagged": "route nodes carrying a kerb=* tag",
        },
    }
    return RouteStudy(table, summary)
