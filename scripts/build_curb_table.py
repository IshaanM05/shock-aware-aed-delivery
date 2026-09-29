"""Distil results/curb_traversability_standard.csv into the compact controller lookup table
configs/curb_table.json (per wheel radius: minimum successful climb speed and the shock it
produces, indexed by kerb height and friction; plus kerb-down shock).

    python scripts/build_curb_table.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = pd.read_csv(ROOT / "results" / "curb_traversability_standard.csv")
    out: dict = {"source": "results/curb_traversability_standard.csv", "up": {}, "down": {}}
    up = df[(df.direction == "up") & (df.angle_deg == 0.0)]
    for r, gr in up.groupby("wheel_radius"):
        heights = sorted(gr.kerb_h.unique())
        mus = sorted(gr.mu.unique())
        vmin = np.full((len(heights), len(mus)), np.nan)
        gmin = np.full_like(vmin, np.nan)
        for i, h in enumerate(heights):
            for j, mu in enumerate(mus):
                c = gr[(gr.kerb_h == h) & (gr.mu == mu) & gr.success]
                if len(c):
                    v = c.speed.min()
                    vmin[i, j] = v
                    gmin[i, j] = c[c.speed == v].peak_g.mean()
        out["up"][f"{r:.3f}"] = {"heights": heights, "mus": mus,
                                 "v_min": np.where(np.isnan(vmin), None, vmin).tolist(),
                                 "peak_g_at_v_min": np.where(np.isnan(gmin), None, gmin).tolist()}
    dn = df[(df.direction == "down") & (df.angle_deg == 0.0) & (df.mu == 1.0)]
    for r, gr in dn.groupby("wheel_radius"):
        piv = gr.pivot_table(index="kerb_h", columns="speed", values="peak_g", aggfunc="mean")
        out["down"][f"{r:.3f}"] = {"heights": piv.index.tolist(), "speeds": piv.columns.tolist(),
                                   "peak_g": piv.values.round(3).tolist()}
    path = ROOT / "configs" / "curb_table.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", path.relative_to(ROOT))


if __name__ == "__main__":
    main()
