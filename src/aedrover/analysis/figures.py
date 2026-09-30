"""Publication figures with one shared style.

Colours follow the validated categorical palette (checked with the colour-vision validator; three
light-mode slots sit below 3:1 contrast on the surface, so every chart here carries direct value
labels or a legend, never colour alone). Colour follows the *entity*, never its rank.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
BUDGET_COLOUR = "#d03b3b"

CONTROLLER_COLOURS = {"dwa": SERIES[0], "mppi": SERIES[1], "ppo": SERIES[2], "apf": SERIES[3],
                      "pure_pursuit": SERIES[4], "ppo_noshield": "#86d9bf"}
CONTROLLER_LABELS = {"pure_pursuit": "Pure pursuit", "apf": "Potential field", "dwa": "Dynamic window",
                     "mppi": "MPPI (physics rollouts)", "ppo": "PPO (learned)", "ppo_noshield": "PPO, filter off"}
FAMILY_LABELS = {"flat_clear": "Flat, clear", "kerb": "Kerb crossing", "crowded": "Crowded", "mixed": "Kerb + crowd",
                 "slippery": "Wet kerb + crowd"}
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 9.5, "text.color": INK,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.titlesize": 11,
        "axes.titleweight": "semibold", "axes.titlelocation": "left", "axes.spines.top": False,
        "axes.spines.right": False, "xtick.color": INK2, "ytick.color": INK2, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.grid": True, "axes.axisbelow": True, "legend.frameon": False,
        "figure.dpi": 120, "savefig.dpi": 300,
    })


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------
def fig_codesign(out: Path, results: Path | None = None) -> Path:
    """Payload shock per kerb condition: draft design versus co-designed rover."""
    apply_style()
    d = json.loads((results or ROOT / "results" / "codesign.json").read_text(encoding="utf-8"))
    nom, opt = d["nominal"]["eval"], d["optimised"]["eval"]
    labels, a, b = [], [], []
    for cn, co in zip(nom["up"], opt["up"], strict=True):
        labels.append(f"Kerb up {cn['h'] * 100:.0f} cm, friction {cn['mu']:.2f}")
        a.append(cn["shock"]), b.append(co["shock"])
    for cn, co in zip(nom["down"], opt["down"], strict=True):
        labels.append(f"Kerb down {cn['h'] * 100:.0f} cm")
        a.append(cn["shock"]), b.append(co["shock"])
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    h = 0.36
    ax.barh(y + h / 2, a, height=h - 0.03, color=MUTED, edgecolor=SURFACE, linewidth=1.0, label="Draft design")
    ax.barh(y - h / 2, b, height=h - 0.03, color=SERIES[0], edgecolor=SURFACE, linewidth=1.0, label="Co-designed")
    for yi, va, vb in zip(y, a, b, strict=True):
        pad = dict(facecolor=SURFACE, edgecolor="none", pad=0.8)
        ax.text(va + 0.08, yi + h / 2, f"{va:.2f} g", va="center", fontsize=8.3, color=INK2, bbox=pad, zorder=6)
        ax.text(vb + 0.08, yi - h / 2, f"{vb:.2f} g", va="center", fontsize=8.3, color=INK2, bbox=pad, zorder=6)
    ax.axvline(3.0, color=BUDGET_COLOUR, lw=1.3, ls=(0, (4, 3)), zorder=3)
    ax.text(3.05, y.max() + 0.62, "3 g payload budget", color=BUDGET_COLOUR, fontsize=8.5, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Peak payload acceleration [g]")
    ax.set_xlim(0, max(a) * 1.15)
    ax.grid(axis="y", visible=False)
    ax.set_title("Mechanical co-design cuts payload shock in every kerb condition", pad=16)
    ax.legend(loc="lower right")
    fig.text(0.0, -0.02, f"Within budget: {nom['frac_within_budget']:.0%} of conditions (draft) -> "
             f"{opt['frac_within_budget']:.0%} (co-designed). Differential evolution, {d['nfev']} simulated designs.",
             fontsize=8.3, color=INK2)
    return _save(fig, out)


def fig_climb_window(out: Path, table: Path | None = None) -> Path:
    """Minimum climbing speed and the shock at that speed, by kerb height and friction, for both designs."""
    apply_style()
    t = json.loads((table or ROOT / "configs" / "curb_table.json").read_text(encoding="utf-8"))
    panels = []
    for design, title in (("nominal", "Draft design (0.15 m wheels)"), ("optimized", "Co-designed (0.22 m wheels)")):
        up = t[design]["up"]
        key = min(up, key=lambda k: abs(float(k) - (0.15 if design == "nominal" else 0.219)))
        e = up[key]
        vmin = np.array([[np.nan if x is None else x for x in r] for r in e["v_min"]], float)
        gmin = np.array([[np.nan if x is None else x for x in r] for r in e["peak_g_at_v_min"]], float)
        panels.append((title, e["heights"], e["mus"], vmin, gmin))
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.6))
    for j, (title, hs, mus, vmin, gmin) in enumerate(panels):
        for i, (grid, lab, vmax, fmt) in enumerate(((vmin, "Minimum climbing speed [m/s]", 3.0, "{:.1f}"),
                                                    (gmin, "Payload shock at that speed [g]", 7.0, "{:.1f}"))):
            ax = axes[i, j]
            ax.grid(False)
            im = ax.imshow(np.ma.masked_invalid(grid), cmap=SEQ, vmin=0, vmax=vmax, aspect="auto", origin="lower")
            for r in range(grid.shape[0]):
                for c in range(grid.shape[1]):
                    v = grid[r, c]
                    txt = "no climb" if np.isnan(v) else fmt.format(v)
                    bold = i == 1 and not np.isnan(v) and v > 3.0
                    ax.text(c, r, txt, ha="center", va="center", fontsize=8.2,
                            color="white" if (not np.isnan(v) and v > 0.55 * vmax) else INK,
                            fontweight="bold" if bold else "normal")
            ax.set_xticks(range(len(mus)))
            ax.set_xticklabels([f"{m:.1f}" for m in mus])
            ax.set_yticks(range(len(hs)))
            ax.set_yticklabels([f"{h * 100:.0f} cm" for h in hs])
            if i == 1:
                ax.set_xlabel("Tyre-ground friction")
            if j == 0:
                ax.set_ylabel("Kerb height")
            ax.set_title(title if i == 0 else "", pad=8)
            if j == 1:
                cb = fig.colorbar(im, ax=axes[i, :], fraction=0.035, pad=0.02)
                cb.set_label(lab, color=INK2)
                cb.outline.set_edgecolor(AXIS)
    fig.suptitle("Climbing a kerb needs momentum, and momentum shakes the payload", x=0.05, ha="left",
                 fontsize=11.5, fontweight="semibold", y=1.0)
    fig.text(0.05, -0.005, "Bold shock values exceed the 3 g budget. Straight approach, sharp kerb, 6,048 + 1,512 simulated trials.",
             fontsize=8.3, color=INK2)
    return _save(fig, out)


def fig_training_curve(curve: pd.DataFrame, out: Path, title: str) -> Path:
    apply_style()
    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    s = curve.steps / 1e6
    k = min(5, max(1, len(curve) // 4))
    for col, colour, label in (("goal", SERIES[2], "Reached goal"), ("collision", SERIES[7], "Collision"),
                               ("stall", SERIES[3], "Stalled")):
        ax.plot(s, curve[col], color=colour, alpha=0.22, lw=1)
        ax.plot(s, curve[col].rolling(k, min_periods=1).mean(), color=colour, lw=2.0, label=label)
        ax.text(s.iloc[-1] + 0.05, curve[col].rolling(k, min_periods=1).mean().iloc[-1], label, color=INK2,
                fontsize=8.3, va="center")
    ax.set_xlabel("Decisions [millions] (10 Hz)")
    ax.set_ylabel("Share of recent training episodes")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, s.max() * 1.22)
    ax.set_title(title, pad=10)
    return _save(fig, out)
