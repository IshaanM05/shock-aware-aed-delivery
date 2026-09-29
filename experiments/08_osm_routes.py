"""Experiment 08: real-map route geometry around the NMIMS Mumbai campus (Track E).

Downloads (or loads from the git-ignored cache) the OpenStreetMap walking and driving networks
inside a 2 km disc around the campus, samples origin-destination pairs, and measures the walking
and driving route factor and the number of kerb-relevant crossings per walking route.

    python experiments/08_osm_routes.py                 # offline if the committed CSV exists
    python experiments/08_osm_routes.py --refresh       # re-download both networks and redo all
    python experiments/08_osm_routes.py --recompute     # redo from cached graphml, no network

Outputs (small, committed): data/osm/vile_parle_od_routes.csv, data/osm/vile_parle_summary.json.
Raw graphs stay under .cache/osm/ (git-ignored).

Map data from OpenStreetMap contributors, ODbL 1.0.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from aedrover.geo.routes import (
    ATTRIBUTION,
    AreaSpec,
    load_od_table,
    run_pipeline,
    summarise_routes,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "osm"
CACHE_DIR = ROOT / ".cache" / "osm"
CSV_NAME = "vile_parle_od_routes.csv"
JSON_NAME = "vile_parle_summary.json"


def format_summary(summary: dict[str, Any]) -> str:
    """Human-readable console rendering of the summary dict (ASCII)."""
    routes, samp = summary["routes"], summary["sampling"]
    walk = summary["walk_network"]
    lines = [
        "",
        ATTRIBUTION,
        f"area: {summary['area']['query']}, radius {summary['area']['radius_m']:.0f} m, "
        f"retrieved {summary['osm_retrieved']['walk']} (walk) / "
        f"{summary['osm_retrieved']['drive']} (drive)",
        f"pairs: {routes['n_pairs']} (seed {samp['seed']}); proposals {samp['proposals']}, "
        f"rejected outside disc {samp['rejected_outside_disc']}, walk snap "
        f"{samp['rejected_walk_snap']}, drive snap {samp['rejected_drive_snap']} "
        f"(max snap {samp['max_snap_m']:.0f} m)",
        f"walk network: {walk['n_nodes']} nodes, {walk['n_edges_directed']} directed edges, "
        f"{walk['total_length_km']} km; drive network: {summary['drive_network']['n_nodes']} "
        f"nodes, {summary['drive_network']['n_edges_directed']} directed edges",
        "",
        f"{'metric':<22}{'mean':>10}{'median':>10}{'p10':>10}{'p90':>10}",
    ]
    for name, st in routes["metrics"].items():
        lines.append(f"{name:<22}{st['mean']:>10.3f}{st['median']:>10.3f}"
                     f"{st['p10']:>10.3f}{st['p90']:>10.3f}")
    lines.append("")
    lines.append("per km of walking route:")
    for name, st in routes["per_km"].items():
        lines.append(f"{name:<22}{st['mean']:>10.3f}{st['median']:>10.3f}"
                     f"{st['p10']:>10.3f}{st['p90']:>10.3f}   pooled {st['pooled']:.3f}")
    lines += [
        "",
        "walking-network tagging coverage (share of length unless stated):",
        f"  dedicated pedestrian ways      {walk['pedestrian_way_length_share']}",
        f"  road length with sidewalk tag  {walk['road_with_sidewalk_tag_share']}",
        f"  road length sidewalk=no        {walk['road_with_sidewalk_no_tag_share']}",
        f"  road length untagged           {walk['road_untagged_for_sidewalk_share']}",
        f"  crossing nodes / ways          {walk['n_crossing_nodes']} / {walk['n_crossing_ways']}",
        f"  kerb-tagged nodes              {walk['n_kerb_tagged_nodes']} "
        f"{walk['n_kerb_tagged_nodes_by_class']}",
        f"  pairs with zero tagged crossings  {routes['share_pairs_zero_crossings_tagged']:.3f}",
        f"  pairs with zero kerb-tagged nodes {routes['share_pairs_zero_kerb_tagged']:.3f}",
    ]
    return "\n".join(lines)


def _close(a: Any, b: Any, tol: float = 1e-9) -> bool:
    """Recursive numeric comparison of two JSON-like structures."""
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, int | float) and isinstance(b, int | float):
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    return bool(a == b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=500, help="number of OD pairs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--radius", type=float, default=2000.0, help="disc radius [m]")
    ap.add_argument("--max-snap", type=float, default=200.0,
                    help="maximum endpoint-to-node distance [m]")
    ap.add_argument("--refresh", action="store_true",
                    help="re-download both networks (one Overpass request each) and recompute")
    ap.add_argument("--recompute", action="store_true",
                    help="recompute from the cached graphml files without any network access")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    args = ap.parse_args()

    csv_path, json_path = args.out_dir / CSV_NAME, args.out_dir / JSON_NAME
    cached = csv_path.exists() and json_path.exists()
    if cached and not (args.refresh or args.recompute):
        stored = json.loads(json_path.read_text(encoding="ascii"))
        same = (stored["sampling"]["n_pairs"] == args.n and stored["sampling"]["seed"] == args.seed
                and stored["area"]["radius_m"] == args.radius
                and stored["sampling"]["max_snap_m"] == args.max_snap)
        if same:
            print(f"offline: using committed {csv_path.name} and {json_path.name} "
                  "(--refresh re-downloads, --recompute reuses the graph cache)")
            table = load_od_table(csv_path)
            fresh = summarise_routes(table)
            if not _close(fresh, stored["routes"]):
                print("WARNING: summary recomputed from the CSV differs from the stored JSON",
                      file=sys.stderr)
            print(format_summary({**stored, "routes": fresh}))
            return 0
        print("stored outputs were made with other parameters; recomputing")

    area = AreaSpec(radius_m=args.radius)
    study = run_pipeline(area, args.cache_dir, n_pairs=args.n, seed=args.seed,
                         max_snap_m=args.max_snap, refresh=args.refresh,
                         allow_download=not args.recompute)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    study.table.to_csv(csv_path, index=False, lineterminator="\n")
    json_path.write_text(json.dumps(study.summary, indent=2, allow_nan=False) + "\n",
                         encoding="ascii")
    print(f"wrote {csv_path} ({len(study.table)} rows) and {json_path}")
    reread = pd.read_csv(csv_path)
    if not _close(summarise_routes(reread), study.summary["routes"], 1e-9):
        print("WARNING: summary is not reproducible from the written CSV", file=sys.stderr)
    print(format_summary(study.summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
