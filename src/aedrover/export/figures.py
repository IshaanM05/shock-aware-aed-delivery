"""Publication figures for MDRIIA Group 03 (300 DPI), generated from real data only.

This module is deliberately standalone (numpy, pandas, matplotlib and the standard library
only). It is used in two ways:

* imported by ``aedrover.export`` (repository tooling), and
* copied verbatim into the course folder as ``analytics/generate_paper_figures.py``, where it
  regenerates ``docs/figures/*.png`` from ``analytics/aed_delivery_benchmark.csv`` and
  ``analytics/kerb_crossing_telemetry.csv``::

      python analytics/generate_paper_figures.py

Figures
-------
1. System architecture block diagram (drawn with matplotlib; the numbers in the blocks come from
   ``ARCH_FACTS``, which the exporter rewrites from the live package).
2. Kinematic telemetry of one simulated kerb-crossing episode (speed command, payload shock,
   lateral position), read from the telemetry CSV written by ``src/aed_navigation_controller.py``.
3. Controller comparison from the benchmark CSV: success rate per scenario family with Wilson 95%
   intervals, peak payload shock and completion time of the successful episodes.
4. ``video_poster.png``: the thumbnail card for the LinkedIn demonstration video.

Colours follow the validated categorical order (blue, orange, aqua, yellow, magenta, green,
violet) and each controller keeps its slot in every figure. Because three of those hues are light
on white, bars also carry a hatch pattern and every panel has direct labels, so identity is never
carried by colour alone. The CSV files behind each panel are the table view of the figures.
"""

from __future__ import annotations

import argparse
import math
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

DPI = 300
plt.rcParams["hatch.linewidth"] = 0.6

TITLE = (
    "Can an autonomous last-mile ground AED delivery vehicle simulated in MuJoCo reduce "
    "time-to-first-shock below urban ambulance congestion delays (15-20 minutes), given that "
    "sudden cardiac arrest survival drops 7-10% for every minute without defibrillation?"
)
TEAM_LINE = "Kashish Praveen Jain (E026) and Vaishnavi Parashar (E046)"
COURSE_LINE = "SVKM'S NMIMS MPSTME | MDRIIA (702CO0E012) | SEMESTER VI | AY 2026-2027"

# --- BEGIN ARCH_FACTS (the exporter rewrites this block from the live package) ---
ARCH_FACTS = {
    "n_families": 5,
    "n_lidar": 73,
    "fov_deg": 240.0,
    "lidar_range_m": 10.0,
    "control_hz": 50,
    "physics_dt_ms": 2.0,
    "cutoff_hz": 80.0,
    "budget_g": 3.0,
    "a_brake": 2.0,
    "t_react_s": 0.1,
    "corridor_half_width_m": 1.4,
    "mppi_samples": 128,
    "mppi_horizon": 20,
}
# --- END ARCH_FACTS ---

# ---- palette: validated categorical order (light surface) -----------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e3e2dd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
HATCHES = ["", "///", "...", "\\\\\\", "xxx", "---", "|||"]
SEQ_LINE = "#184f95"          # sequential blue 600, single-quantity telemetry lines
ACCENT = "#eb6834"            # thresholds and events

CONTROLLER_ORDER = ["pure_pursuit", "apf_nocurb", "apf", "dwa", "mppi", "ppo", "dwa_nocurb"]
CONTROLLER_LABEL = {
    "pure_pursuit": "Pure pursuit",
    "apf_nocurb": "APF (no kerb logic)",
    "apf": "APF + kerb logic",
    "dwa": "DWA + kerb logic",
    "dwa_nocurb": "DWA (no kerb logic)",
    "mppi": "MPPI",
    "ppo": "PPO",
}
FAMILY_ORDER = ["flat_clear", "kerb", "crowded", "mixed", "slippery"]
FAMILY_LABEL = {"flat_clear": "flat", "kerb": "kerb", "crowded": "crowded", "mixed": "mixed",
                "slippery": "slippery"}

BENCH_COLUMNS = ("controller", "family", "seed", "success", "outcome", "time_s", "peak_shock_g",
                 "shock_over_budget", "min_clearance_m")
TELEMETRY_COLUMNS = ("t_s", "x_m", "y_m", "v_cmd_mps", "shock_g", "controller", "family", "seed",
                     "kerb_h_m", "x_down_m", "x_up_m")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def _require(df: pd.DataFrame, columns: tuple[str, ...], where: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing columns {missing}; found {list(df.columns)}")


def load_benchmark(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require(df, BENCH_COLUMNS, str(path))
    df["success"] = df["success"].astype(bool)
    return df


def load_telemetry(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    _require(df, TELEMETRY_COLUMNS, str(path))
    return df


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """Success proportion with its Wilson score interval."""
    if n <= 0:
        return math.nan, math.nan, math.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def controller_style(name: str, present: list[str]) -> tuple[str, str]:
    """(colour, hatch) of a controller: fixed by the entity, never by its rank in a panel."""
    if name in CONTROLLER_ORDER:
        i = CONTROLLER_ORDER.index(name)
        return SERIES[i], HATCHES[i]
    return "#6b6a66", "+++"


def ordered_controllers(df: pd.DataFrame) -> list[str]:
    present = list(df["controller"].unique())
    known = [c for c in CONTROLLER_ORDER if c in present]
    return known + sorted(c for c in present if c not in CONTROLLER_ORDER)


def _style_axes(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=INK2, labelsize=8, length=3, width=0.8, color=GRID)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8, linestyle="-")
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def _save(fig, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# figure 1: architecture
# ---------------------------------------------------------------------------------------------
def make_figure1(out_path: str | Path, facts: dict | None = None) -> None:
    f = dict(ARCH_FACTS if facts is None else facts)
    fig, ax = plt.subplots(figsize=(11.0, 6.9), facecolor=SURFACE)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")
    face = {"sim": "#e9f1fb", "ctl": "#fdeee7", "out": "#e8f6f0", "app": "#f4f3ef"}

    def block(x, y, w, h, title, lines, kind):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3",
                                    facecolor=face[kind], edgecolor=INK2, linewidth=1.0))
        ax.text(x + w / 2, y + h - 1.5, title, ha="center", va="top", fontsize=8.6,
                fontweight="bold", color=INK)
        ax.text(x + w / 2, y + h - 5.0, "\n".join(lines), ha="center", va="top", fontsize=7.3,
                color=INK2, linespacing=1.45)

    def arrow(p0, p1, label=None, style="arc3", label_xy=None, ha="center"):
        ax.annotate("", xy=p1, xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, shrinkA=0, shrinkB=0,
                                    connectionstyle=style))
        if label:
            lx, ly = label_xy if label_xy else ((p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2 + 1.2)
            ax.text(lx, ly, label, ha=ha, va="bottom", fontsize=6.9, color=INK2, style="italic")

    w, h = 21.0, 16.0
    r1, r2, r3 = 45.0, 24.0, 3.0
    x1, x2, x3, x4 = 2.0, 27.0, 52.0, 77.0

    block(x1, r1, w, h, "Scenario generator",
          [f"{f['n_families']} seeded families", "kerb height, friction,", "payload mass, crowd,",
           "static obstacles"], "sim")
    block(x2, r1, w, h, "MuJoCo world + rover",
          ["4-wheel independent", "suspension, isolated payload", f"physics step {f['physics_dt_ms']:g} ms",
           "mocap terrain slots"], "sim")
    block(x3, r1, w, h, "Perception",
          [f"lidar {f['n_lidar']} rays, {f['fov_deg']:g} deg", f"range {f['lidar_range_m']:g} m",
           "forward height scan", "pedestrian tracker"], "sim")
    block(x4, r1, w, h, "Controller",
          ["pure pursuit | APF | DWA", f"MPPI (K={f['mppi_samples']}, H={f['mppi_horizon']})",
           "PPO policy", f"control loop {f['control_hz']:g} Hz"], "ctl")
    block(x4, r2, w, h, "Safety filter",
          ["stopping-distance barrier", f"a_brake {f['a_brake']:g} m/s^2, t_react {f['t_react_s']:g} s",
           "only ever reduces speed", "steering passes through"], "ctl")
    block(x2, r2, w, h, "Ackermann + actuators",
          ["wheel velocity actuators", "torque clamp per wheel", "steering position servos"], "ctl")
    block(x1, r2, w, h, "Episode metrics",
          ["payload shock (g)", f"{f['cutoff_hz']:g} Hz zero-phase low-pass",
           "clearance, time, energy", f"shock budget {f['budget_g']:g} g"], "out")
    block(x1, r3, w, h - 2, "Statistics",
          ["Welch and paired t tests", "Cohen d, 95% CI", "Wilson CI, Holm"], "out")
    block(x2, r3, 46.0, h - 2, "Clinical and economics layer",
          ["Larsen 1993 survival vs delay; Naess 2024 ambulance response times",
           "time-to-first-shock composition; break-even radius",
           "dimensionless economics: kappa, payback, delta FTE"], "app")
    block(x4, r3, w, h - 2, "Deliverables",
          ["300 DPI figures, tables,", "manuscript, viva answers"], "app")

    y_mid1, y_mid2 = r1 + h / 2, r2 + h / 2
    arrow((x1 + w, y_mid1), (x2, y_mid1))
    arrow((x2 + w, y_mid1), (x3, y_mid1), "state")
    arrow((x3 + w, y_mid1), (x4, y_mid1), "obs")
    arrow((x4 + w / 2, r1), (x4 + w / 2, r2 + h), "(v, delta)", label_xy=(x4 + w / 2 + 1.2, r1 - 4.3), ha="left")
    arrow((x4, y_mid2), (x2 + w, y_mid2), "filtered (v, delta)", label_xy=((x2 + w + x4) / 2, y_mid2 + 1.2))
    arrow((x2 + w / 2, r2 + h), (x2 + w / 2, r1), "wheel commands", label_xy=(x2 + w / 2 + 1.2, r1 - 4.3), ha="left")
    arrow((x2 + 2.0, r1), (x1 + w - 2.0, r2 + h), "logs", label_xy=(x1 + w + 3.0, r2 + h + 3.6), ha="left")
    arrow((x1 + w / 2, r2), (x1 + w / 2, r3 + h - 2))
    arrow((x1 + w, r3 + (h - 2) / 2), (x2, r3 + (h - 2) / 2))
    arrow((x2 + 46.0, r3 + (h - 2) / 2), (x4, r3 + (h - 2) / 2))
    ax.set_title("Figure 1. System architecture of the simulated AED delivery rover",
                 fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.text(50, 0.4, "Solid arrows: data or command flow. Values inside the blocks are read from "
            "the vendored package.", ha="center", va="bottom", fontsize=7.2, color=INK2)
    _save(fig, out_path)


# ---------------------------------------------------------------------------------------------
# figure 2: telemetry
# ---------------------------------------------------------------------------------------------
def make_figure2(telemetry_csv: str | Path, out_path: str | Path, facts: dict | None = None) -> None:
    f = dict(ARCH_FACTS if facts is None else facts)
    tel = load_telemetry(telemetry_csv)
    t, x, y = tel["t_s"].to_numpy(), tel["x_m"].to_numpy(), tel["y_m"].to_numpy()
    v, g = tel["v_cmd_mps"].to_numpy(), tel["shock_g"].to_numpy()
    meta = tel.iloc[0]
    x_down, x_up, kerb_h = float(meta["x_down_m"]), float(meta["x_up_m"]), float(meta["kerb_h_m"])
    has_kerb = kerb_h > 1e-6 and x_up < 1e2

    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.0), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.16})
    for ax in axes:
        _style_axes(ax)
        if has_kerb:
            t0 = float(np.interp(x_down, x, t))
            t1 = float(np.interp(x_up, x, t))
            ax.axvspan(t0, t1, color="#eceae4", zorder=0, linewidth=0)
    axes[0].plot(t, v, color=SEQ_LINE, lw=1.6, solid_capstyle="round")
    axes[0].set_ylabel("Speed command (m/s)")
    axes[1].plot(t, g, color=SEQ_LINE, lw=1.3, solid_capstyle="round")
    axes[1].axhline(f["budget_g"], color=ACCENT, lw=1.6)
    axes[1].text(t[-1], f["budget_g"], f" budget {f['budget_g']:g} g", color=INK2, fontsize=8,
                 va="bottom", ha="right")
    ip = int(np.argmax(g))
    axes[1].plot([t[ip]], [g[ip]], "o", ms=6, color=SEQ_LINE, mec=SURFACE, mew=1.5)
    axes[1].annotate(f"peak {g[ip]:.2f} g", (t[ip], g[ip]), xytext=(8, 6), textcoords="offset points",
                     fontsize=8, color=INK)
    axes[1].set_ylabel("Payload shock (g)")
    axes[2].plot(t, y, color=SEQ_LINE, lw=1.4, solid_capstyle="round")
    cw = f["corridor_half_width_m"]
    for s in (-cw, cw):
        axes[2].axhline(s, color=ACCENT, lw=1.2, linestyle=(0, (4, 3)))
    axes[2].set_ylabel("Lateral position (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].text(t[0], cw, f" corridor limit +/-{cw:g} m", color=INK2, fontsize=8, va="bottom")
    if has_kerb:
        lo, hi = axes[0].get_ylim()
        axes[0].text(0.5 * (float(np.interp(x_down, x, t)) + float(np.interp(x_up, x, t))),
                     lo + 0.06 * (hi - lo), "road crossing", ha="center", va="bottom", fontsize=8,
                     color=INK2)
    c, fam, seed = str(meta["controller"]), str(meta["family"]), int(meta["seed"])
    fig.suptitle(
        f"Figure 2. Kinematic telemetry of one simulated episode "
        f"({CONTROLLER_LABEL.get(c, c)}, {fam} family, seed {seed}"
        + (f", {100 * kerb_h:.1f} cm kerb)" if has_kerb else ")"),
        fontsize=10.5, fontweight="bold", color=INK, y=0.93)
    _save(fig, out_path)


# ---------------------------------------------------------------------------------------------
# figure 3: benchmark comparison
# ---------------------------------------------------------------------------------------------
def make_figure3(bench_csv: str | Path, out_path: str | Path, facts: dict | None = None) -> None:
    f = dict(ARCH_FACTS if facts is None else facts)
    df = load_benchmark(bench_csv)
    ctrls = ordered_controllers(df)
    fams = [k for k in FAMILY_ORDER if k in set(df["family"])] + sorted(
        set(df["family"]) - set(FAMILY_ORDER))
    fig = plt.figure(figsize=(14.0, 5.0), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 3, width_ratios=[2.5, 1.0, 1.0], wspace=0.24)
    axa, axb, axc = (fig.add_subplot(gs[0, i]) for i in range(3))

    # (a) success rate per family with Wilson 95% intervals
    _style_axes(axa)
    n_c = len(ctrls)
    width = min(0.8 / max(n_c, 1), 0.16)
    for j, c in enumerate(ctrls):
        col, hatch = controller_style(c, ctrls)
        xs, ps, lo, hi = [], [], [], []
        for i, fam in enumerate(fams):
            sub = df[(df["controller"] == c) & (df["family"] == fam)]
            p, a, b = wilson(int(sub["success"].sum()), len(sub))
            xs.append(i + (j - (n_c - 1) / 2) * width)
            ps.append(100 * p)
            lo.append(100 * (p - a) if not math.isnan(p) else 0.0)
            hi.append(100 * (b - p) if not math.isnan(p) else 0.0)
        axa.bar(xs, ps, width=width * 0.86, color=col, hatch=hatch, edgecolor=SURFACE, linewidth=0.0,
                label=CONTROLLER_LABEL.get(c, c), zorder=3)
        axa.errorbar(xs, ps, yerr=[lo, hi], fmt="none", ecolor=INK, elinewidth=0.9, capsize=1.6,
                     capthick=0.9, zorder=4)
    axa.set_xticks(range(len(fams)), [FAMILY_LABEL.get(k, k) for k in fams])
    axa.set_ylim(0, 108)
    axa.set_ylabel("Success rate (%), Wilson 95% CI")
    axa.set_title("(a) Reached the goal", fontsize=9.5, fontweight="bold", color=INK, loc="left")
    axa.legend(ncol=min(n_c, 4), fontsize=7.5, frameon=False, loc="lower center",
               bbox_to_anchor=(0.5, -0.26), labelcolor=INK2)

    def box_panel(ax, column, ylabel, title, hline=None):
        _style_axes(ax)
        data, pos = [], []
        for j, c in enumerate(ctrls):
            vals = df[(df["controller"] == c) & df["success"]][column].dropna().to_numpy()
            if len(vals):
                data.append(vals)
                pos.append(j)
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                        medianprops=dict(color=INK, linewidth=1.4),
                        whiskerprops=dict(color=INK2, linewidth=0.9),
                        capprops=dict(color=INK2, linewidth=0.9), zorder=3)
        for patch, j in zip(bp["boxes"], pos, strict=True):
            col, hatch = controller_style(ctrls[j], ctrls)
            patch.set_facecolor(col)
            patch.set_hatch(hatch)
            patch.set_edgecolor(SURFACE)
            patch.set_linewidth(0.0)
        ax.set_xticks(range(len(ctrls)), [CONTROLLER_LABEL.get(c, c) for c in ctrls], rotation=35,
                      ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=9.5, fontweight="bold", color=INK, loc="left")
        if hline is not None:
            ax.axhline(hline, color=ACCENT, lw=1.6)
            ax.text(len(ctrls) - 0.5, hline, f" budget {hline:g} g", color=INK2, fontsize=8, ha="right",
                    va="bottom")

    box_panel(axb, "peak_shock_g", "Peak payload shock (g)", "(b) Shock, successful episodes",
              hline=f["budget_g"])
    box_panel(axc, "time_s", "Completion time (s)", "(c) Time, successful episodes")
    fig.suptitle(f"Figure 3. Controller comparison on the seeded benchmark (N = {len(df)} episodes)",
                 fontsize=10.5, fontweight="bold", color=INK, y=1.0)
    _save(fig, out_path)


# ---------------------------------------------------------------------------------------------
# poster
# ---------------------------------------------------------------------------------------------
def headline_metrics(df: pd.DataFrame) -> list[tuple[str, str]]:
    """Three headline tiles computed from the benchmark rows only."""
    ctrls = ordered_controllers(df)
    best, best_rate = None, -1.0
    for c in ctrls:
        sub = df[df["controller"] == c]
        rate = float(sub["success"].mean())
        if rate > best_rate:
            best, best_rate = c, rate
    sub = df[df["controller"] == best]
    _, lo, hi = wilson(int(sub["success"].sum()), len(sub))
    ok = sub[sub["success"]]
    med_g = float(ok["peak_shock_g"].median()) if len(ok) else math.nan
    return [
        (f"{len(df)}", f"simulated episodes, {len(ctrls)} controllers"),
        (f"{100 * best_rate:.1f}%", f"best success rate ({CONTROLLER_LABEL.get(best, best)}), "
                                   f"95% CI {100 * lo:.1f}-{100 * hi:.1f}%"),
        (f"{med_g:.2f} g", "median peak payload shock, same controller"),
    ]


def make_poster(bench_csv: str | Path, out_path: str | Path) -> None:
    df = load_benchmark(bench_csv)
    tiles = headline_metrics(df)
    bg, card, edge, blue, txt, dim = "#080d16", "#0f1729", "#1e2a44", "#38bdf8", "#f5f7fb", "#9aa7bd"
    fig = plt.figure(figsize=(12.8, 7.2), facecolor=bg)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((32, 32), 1216, 656, boxstyle="round,pad=0,rounding_size=14",
                                facecolor=card, edgecolor=edge, linewidth=1.5))
    ax.text(64, 70, COURSE_LINE, color="#cbd5e6", fontsize=13, fontweight="bold", va="center")
    ax.plot([64, 1216], [90, 90], color="#2563eb", lw=2)
    ax.text(64, 124, "MDRIIA GROUP 03  |  AUTONOMOUS GROUND AED DELIVERY ROVER", color=blue,
            fontsize=13.5, fontweight="bold", va="center")
    ax.text(64, 156, "\n".join(textwrap.wrap(TITLE, 84)), color=txt, fontsize=15.5, fontweight="bold",
            va="top", linespacing=1.32)
    tile_w, y0 = 372, 318
    for i, (big, small) in enumerate(tiles):
        x0 = 64 + i * (tile_w + 18)
        ax.add_patch(FancyBboxPatch((x0, y0), tile_w, 120, boxstyle="round,pad=0,rounding_size=8",
                                    facecolor="#111c33", edgecolor=edge, linewidth=1.2))
        ax.text(x0 + 20, y0 + 44, big, color=blue, fontsize=26, fontweight="bold", va="center")
        ax.text(x0 + 20, y0 + 92, "\n".join(textwrap.wrap(small, 38)), color=dim, fontsize=10.5,
                va="center", linespacing=1.3)
    cx, cy = 640, 520
    ax.add_patch(plt.Circle((cx, cy), 44, color="#2563eb"))
    ax.add_patch(plt.Polygon([[cx - 12, cy - 22], [cx - 12, cy + 22], [cx + 24, cy]], color="white"))
    ax.text(cx, cy + 68, "60-90 s simulation walkthrough (video pending publication)", color=dim,
            fontsize=11, ha="center", va="center")
    ax.plot([64, 1216], [626, 626], color=edge, lw=1.2)
    ax.text(64, 655, f"Student researchers: {TEAM_LINE}", color=dim, fontsize=10.5, va="center")
    ax.text(1216, 655, "Watch on LinkedIn (link pending)", color=blue, fontsize=10.5,
            fontweight="bold", ha="right", va="center")
    fig.savefig(out_path, dpi=100, facecolor=bg)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------
# command line (used verbatim as analytics/generate_paper_figures.py)
# ---------------------------------------------------------------------------------------------
def build_all(root: str | Path, bench: str | Path | None = None, telemetry: str | Path | None = None,
              facts: dict | None = None) -> list[Path]:
    root = Path(root)
    bench = Path(bench) if bench else root / "analytics" / "aed_delivery_benchmark.csv"
    telemetry = Path(telemetry) if telemetry else root / "analytics" / "kerb_crossing_telemetry.csv"
    out = root / "docs" / "figures"
    written = [out / "figure1_system_architecture.png", out / "figure2_kinematic_telemetry.png",
               out / "figure3_comparative_performance.png", out / "video_poster.png"]
    make_figure1(written[0], facts)
    make_figure2(telemetry, written[1], facts)
    make_figure3(bench, written[2], facts)
    make_poster(bench, written[3])
    return written


def main(argv: list[str] | None = None) -> int:
    default_root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser(description="Regenerate the 300 DPI publication figures from the CSV data.")
    ap.add_argument("--root", default=str(default_root), help="course folder (default: parent of analytics/)")
    ap.add_argument("--benchmark", default=None, help="benchmark CSV (default analytics/aed_delivery_benchmark.csv)")
    ap.add_argument("--telemetry", default=None, help="telemetry CSV (default analytics/kerb_crossing_telemetry.csv)")
    args = ap.parse_args(argv)
    for p in build_all(args.root, args.benchmark, args.telemetry):
        print(f"[OK] wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
