"""Result figures. They take DataFrames, so they can be tested on any subset of the data.

Shares the style and palette of ``figures.py`` (validated categorical palette; every chart carries
direct value labels or a legend, never colour alone).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figures import (
    AXIS,
    BUDGET_COLOUR,
    CONTROLLER_COLOURS,
    CONTROLLER_LABELS,
    FAMILY_LABELS,
    INK,
    INK2,
    MUTED,
    SERIES,
    SURFACE,
    _save,
    apply_style,
)

CONTROLLER_ORDER = ["pure_pursuit", "apf", "dwa", "mppi", "ppo", "ppo_noshield"]


def _present(df: pd.DataFrame, col: str = "controller") -> list[str]:
    have = set(df[col].unique())
    return [c for c in CONTROLLER_ORDER if c in have]


def _grouped_bars(ax, cats, series, value, err=None, fmt="{:.0%}", label_size: float = 7.4, labels: bool = True) -> None:
    """Grouped bars with the value written above each bar (above its interval whisker when there is one)."""
    n = len(series)
    w = 0.82 / n
    x = np.arange(len(cats))
    for i, s in enumerate(series):
        vals = np.array([value(c, s) for c in cats], dtype=float)
        pos = x - 0.41 + w * (i + 0.5)
        ax.bar(pos, vals, width=w - 0.04, color=CONTROLLER_COLOURS[s], edgecolor=SURFACE, linewidth=1.0,
               label=CONTROLLER_LABELS[s])
        tops = vals
        if err is not None:
            lo = np.array([err(c, s)[0] for c in cats], dtype=float)
            hi = np.array([err(c, s)[1] for c in cats], dtype=float)
            ax.errorbar(pos, vals, yerr=[np.clip(vals - lo, 0, None), np.clip(hi - vals, 0, None)], fmt="none",
                        ecolor=INK2, elinewidth=1.0, capsize=2)
            tops = np.where(np.isfinite(hi), hi, vals)
        if not labels:
            continue
        for p, v, t in zip(pos, vals, tops, strict=True):
            if np.isfinite(v):
                ax.annotate(fmt.format(v), (p, t), xytext=(0, 2.5), textcoords="offset points", ha="center", va="bottom",
                            fontsize=label_size, color=INK, rotation=90)


def fig_benchmark(df: pd.DataFrame, out: Path, budget_g: float = 3.0, title: str | None = None) -> Path:
    """Success rate (Wilson 95% CI), median payload shock and median time per scenario family."""
    from .report import summarize

    apply_style()
    sm = summarize(df, budget_g=budget_g).set_index(["controller", "family"])
    ctrls = _present(df)
    fams = [f for f in FAMILY_LABELS if f in set(df.family)]

    def get(col):
        return lambda f, c: sm.loc[(c, f), col] if (c, f) in sm.index else np.nan

    def rng(f, c):
        return (sm.loc[(c, f), "success_lo"], sm.loc[(c, f), "success_hi"]) if (c, f) in sm.index else (np.nan, np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.3), layout="constrained")
    _grouped_bars(axes[0], fams, ctrls, get("success"), err=rng)
    axes[0].set_ylim(0, 1.3)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_title("Reached the goal", pad=8)
    axes[0].set_ylabel("Success rate (95% Wilson interval)")
    _grouped_bars(axes[1], fams, ctrls, get("shock_med"), fmt="{:.1f}", labels=False)
    axes[1].axhline(budget_g, color=BUDGET_COLOUR, lw=1.2, ls=(0, (4, 3)))
    axes[1].text(len(fams) - 0.5, budget_g + 0.06, f"{budget_g:g} g budget", color=BUDGET_COLOUR, ha="right", fontsize=8.3)
    axes[1].set_title("Median peak payload shock", pad=8)
    axes[1].set_ylabel("g (successful episodes)")
    _grouped_bars(axes[2], fams, ctrls, get("time_med"), fmt="{:.0f}", labels=False)
    axes[2].set_title("Median time to goal", pad=8)
    axes[2].set_ylabel("seconds (successful episodes)")
    for ax in axes:
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([FAMILY_LABELS[f] for f in fams], rotation=18, ha="right")
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(ctrls), fontsize=8.5)
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    return _save(fig, out)


def fig_ood(tab: pd.DataFrame, out: Path, order=("id", "ood_kerb", "ood_mu", "ood_both")) -> Path:
    """Safe-delivery rate inside and outside the training distribution.

    Columns: controller, condition, safe_delivery, safe_lo, safe_hi.
    """
    apply_style()
    ctrls = _present(tab)
    conds = [c for c in order if c in set(tab.condition)]
    names = {"id": "In distribution", "ood_kerb": "Taller kerbs\n(15-19 cm)", "ood_mu": "Slipperier\n(mu 0.3-0.5)",
             "ood_both": "Both"}
    t = tab.set_index(["controller", "condition"])

    def val(c, s):
        return t.loc[(s, c), "safe_delivery"] if (s, c) in t.index else np.nan

    def err(c, s):
        return (t.loc[(s, c), "safe_lo"], t.loc[(s, c), "safe_hi"]) if (s, c) in t.index else (np.nan, np.nan)

    fig, ax = plt.subplots(figsize=(7.8, 4.2), layout="constrained")
    _grouped_bars(ax, conds, ctrls, val, err=err)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([names[c] for c in conds])
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Safe delivery (goal AND shock within budget)")
    ax.grid(axis="x", visible=False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=len(ctrls), fontsize=8.5)
    fig.suptitle("Robustness outside the training distribution", x=0.01, ha="left", fontsize=11.5, fontweight="semibold")
    return _save(fig, out)


def fig_survival(surv: pd.DataFrame, out: Path, model: str = "larsen1993", densities=None) -> Path:
    """Expected survival versus response radius for ambulance, rover controllers and drone."""
    apply_style()
    s = surv[surv.model == model]
    dens = densities or list(dict.fromkeys(s.density))
    titles = {"osm_lower_bound": "Crossings: OSM-tagged lower bound", "assumed_2_per_km": "2 crossings per km",
              "assumed_4_per_km": "4 crossings per km", "assumed_8_per_km": "8 crossings per km"}
    fig, axes = plt.subplots(1, len(dens), figsize=(4.0 * len(dens) + 0.6, 3.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, dens, strict=True):
        sd = s[s.density == d]
        amb = sd[sd["mode"] == "ambulance"].groupby("radius_m").mean_survival.mean()
        drone = sd[sd["mode"] == "drone"].groupby("radius_m").mean_survival.mean()
        ax.plot(amb.index, amb.values * 100, color=MUTED, lw=2.0)
        ax.plot(drone.index, drone.values * 100, color=SERIES[6], lw=2.0)
        ax.text(amb.index[-1], amb.values[-1] * 100 - 0.25, "Ambulance alone", color=INK2, fontsize=8, ha="right", va="top")
        ax.text(drone.index[0], drone.values[0] * 100 + 0.3, "Drone + ambulance", color=SERIES[6], fontsize=8, ha="left",
                va="bottom")
        ax.set_ylim(amb.values.min() * 100 - 1.3, drone.values.max() * 100 + 1.4)
        for c in _present(sd):
            r = sd[(sd["mode"] == "rover") & (sd.controller == c)].sort_values("radius_m")
            if len(r):
                ax.plot(r.radius_m, r.mean_survival * 100, color=CONTROLLER_COLOURS[c], lw=2.0, marker="o", ms=3.5)
                ax.fill_between(r.radius_m, r.survival_ci_low * 100, r.survival_ci_high * 100,
                                color=CONTROLLER_COLOURS[c], alpha=0.12, lw=0)
        ax.set_title(titles.get(d, d), pad=8, fontsize=9.5)
        ax.set_xlabel("Response radius [m]")
    axes[0].set_ylabel("Expected survival [%]")
    rover_ctrls = _present(s[s["mode"] == "rover"])
    handles = [plt.Line2D([0], [0], color=CONTROLLER_COLOURS[c], lw=2, marker="o", ms=3.5,
                          label=CONTROLLER_LABELS[c] + " + ambulance") for c in rover_ctrls]
    fig.legend(handles=handles, loc="lower center", ncols=3, fontsize=8, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Survival benefit of AED delivery versus response radius (Larsen 1993 model)", x=0.01, ha="left",
                 fontsize=11.5, fontweight="semibold", y=1.04)
    fig.tight_layout()
    return _save(fig, out)


def fig_policies(pol: pd.DataFrame, out: Path, controller: str) -> Path:
    """Expected survival by dispatch policy as drone availability varies."""
    apply_style()
    p = pol[pol.controller == controller]
    names = {"ambulance": "Ambulance only", "rover": "Ambulance + rover", "drone": "Ambulance + drone",
             "hybrid": "Ambulance + hybrid (drone, else rover)", "both": "Ambulance + both"}
    colours = {"ambulance": MUTED, "rover": CONTROLLER_COLOURS.get(controller, SERIES[0]), "drone": SERIES[6],
               "hybrid": SERIES[3], "both": SERIES[7]}
    # Policies that coincide are drawn at different widths (widest first) so every line stays visible.
    widths = {"rover": 5.5, "ambulance": 2.0, "drone": 6.5, "hybrid": 4.0, "both": 1.8}
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    handles, overlap = [], False
    for name in ("rover", "ambulance", "drone", "hybrid", "both"):
        g = p[p.policy == name].sort_values("p_drone")
        if g.empty:
            continue
        ax.plot(g.p_drone * 100, g.mean_survival * 100, color=colours[name], lw=widths[name], alpha=0.95, marker="o", ms=3.5,
                solid_capstyle="round")
        handles.append(plt.Line2D([0], [0], color=colours[name], lw=2.4, marker="o", ms=3.5, label=names[name]))
    amb, rov = p[p.policy == "ambulance"].sort_values("p_drone"), p[p.policy == "rover"].sort_values("p_drone")
    if len(amb) and len(rov) and np.allclose(amb.mean_survival.values, rov.mean_survival.values, atol=1e-3):
        overlap = True
    ax.set_xlabel("Probability the drone can fly [%]")
    ax.set_ylabel("Expected survival [%]")
    ax.set_title(f"Dispatch policy at 1 km ({CONTROLLER_LABELS.get(controller, controller)} rover)", pad=10)
    if overlap:
        ax.text(0.02, 0.97, "At this radius the rover adds nothing over the ambulance alone,\n"
                            "so those lines coincide, as do the three drone policies.",
                transform=ax.transAxes, fontsize=8, color=INK2, va="top")
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.14), fontsize=8, frameon=False)
    return _save(fig, out)


def fig_speed_cap(df: pd.DataFrame, out: Path) -> Path:
    """Safe-delivery rate and median time against the common speed cap (condition column 'cap_<x>')."""
    apply_style()
    d = df.copy()
    d["cap"] = d.condition.str.replace("cap_", "", regex=False).astype(float)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.7))
    for c in _present(d):
        g = d[d.controller == c].groupby("cap").agg(safe=("safe", "mean")).reset_index()
        t = d[(d.controller == c) & d.success].groupby("cap").time_s.median().reset_index()
        axes[0].plot(g.cap, g.safe * 100, color=CONTROLLER_COLOURS[c], lw=2.0, marker="o", ms=4)
        axes[1].plot(t.cap, t.time_s, color=CONTROLLER_COLOURS[c], lw=2.0, marker="o", ms=4)
        axes[0].text(g.cap.iloc[-1] + 0.05, g.safe.iloc[-1] * 100, CONTROLLER_LABELS[c], color=INK2, fontsize=8, va="center")
        axes[1].text(t.cap.iloc[-1] + 0.05, t.time_s.iloc[-1], CONTROLLER_LABELS[c], color=INK2, fontsize=8, va="center")
    for k, ax in enumerate(axes):
        ax.axvline(0.8, color=MUTED, lw=1, ls=(0, (2, 3)))
        ax.set_xlim(0.7, 3.6)
        ax.set_xlabel("Speed cap given to every controller [m/s]")
        # label where no data line runs: top of the left panel, bottom of the right one
        ax.text(0.83, 0.97 if k == 0 else 0.03, "0.8 m/s cap", transform=ax.get_xaxis_transform(), fontsize=7.6,
                color=INK2, rotation=90, va="top" if k == 0 else "bottom")
    axes[0].set_ylabel("Safe delivery [%]")
    axes[0].set_title("Safety versus speed cap (kerb + crowd)", pad=8)
    axes[1].set_ylabel("Median time to goal [s]")
    axes[1].set_title("Time cost of a lower cap", pad=8)
    fig.tight_layout()
    return _save(fig, out)


def fig_architecture(out: Path) -> Path:
    """Block diagram of the simulation, autonomy, learning and evaluation layers."""
    apply_style()
    fig, ax = plt.subplots(figsize=(11.0, 5.0))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.1)
    ax.axis("off")

    def box(x, y, w, h, title, lines, colour):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=SURFACE, edgecolor=colour, linewidth=1.8))
        ax.text(x + 0.12, y + h - 0.16, title, fontsize=9.4, fontweight="semibold", color=INK, va="top")
        ax.text(x + 0.12, y + h - 0.55, "\n".join(lines), fontsize=7.9, color=INK2, va="top", linespacing=1.4)

    top, h_top = 2.55, 1.7
    box(0.1, top, 2.6, h_top, "MuJoCo world", ["4-wheel rover, independent suspension", "payload isolator + accelerometer",
                                                "kerbs, ramps, obstacles (mocap slots)", "social-force pedestrian streams"], SERIES[0])
    box(3.1, top, 2.3, h_top, "Perception", ["73-ray lidar", "terrain height scan", "noisy pedestrian tracks",
                                              "one shared observation"], SERIES[2])
    box(5.8, top, 2.5, h_top, "Controllers", ["pure pursuit, potential field", "dynamic window + kerb table",
                                               "MPPI (physics rollouts)", "PPO (domain randomised)"], SERIES[1])
    box(8.7, top, 2.2, h_top, "Safety filter", ["swept-footprint arc", "stopping-distance barrier", "identical for all",
                                                 "controllers"], SERIES[7])
    box(0.1, 0.1, 3.4, 1.55, "Mechanical co-design", ["differential evolution over wheel,", "suspension, isolator, motor",
                                                       "objective: payload shock at climb"], SERIES[3])
    box(3.9, 0.1, 3.4, 1.55, "Evaluation", ["paired seeds, 5 scenario families", "Wilson, McNemar, paired t, Holm",
                                             "out-of-distribution + ablations"], SERIES[4])
    box(7.7, 0.1, 3.2, 1.55, "Clinical layer", ["route composition from segments", "Larsen survival, Naess EMS delays",
                                                 "ambulance vs rover vs drone"], SERIES[6])
    arrow = dict(arrowstyle="-|>", color=INK2, lw=1.4)
    for x0, x1 in ((2.7, 3.1), (5.4, 5.8), (8.3, 8.7)):
        ax.annotate("", xy=(x1, top + h_top / 2), xytext=(x0, top + h_top / 2), arrowprops=arrow)
    ax.annotate("", xy=(1.4, top), xytext=(1.4, 1.65), arrowprops=arrow)
    ax.annotate("", xy=(4.6, 1.65), xytext=(6.9, top), arrowprops=arrow)
    ax.annotate("", xy=(7.7, 0.88), xytext=(7.3, 0.88), arrowprops=arrow)
    # closed loop: filtered command drives the rover
    ax.annotate("", xy=(1.4, top + h_top), xytext=(9.8, top + h_top),
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4, connectionstyle="arc3,rad=0.16"))
    ax.text(5.6, 4.36, "filtered command (speed, steering) -> wheel and steering actuators", fontsize=8, color=INK2, ha="center")
    ax.text(0.1, 5.02, "System overview", fontsize=11.5, fontweight="semibold", color=INK, va="bottom")
    _ = AXIS
    return _save(fig, out)
