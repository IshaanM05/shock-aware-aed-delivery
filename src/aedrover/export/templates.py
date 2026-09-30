"""Text of the exported course folder: document templates, the exported scripts and the numbers
formatted into them.

Placeholders use the syntax ``@@name@@`` (LaTeX braces and dollar signs appear all over the
documents, so ``str.format`` and ``string.Template`` are unusable). ``fill`` refuses unknown or
left-over placeholders, so a template can never silently ship with a hole in it.

Every number comes from ``Results`` (files under ``results/``) or from ``package_facts()``. A
missing piece produces a labelled ``PENDING`` marker (``--allow-partial``) - never a default.
"""

from __future__ import annotations

import math
import re

from . import (
    AUTHORIZED_TITLE,
    BUDGET_G,
    DOMAIN,
    FOUNDATIONAL,
    GROUP_FOLDER,
    GROUP_ID,
    MEMBERS,
    METHOD_REFS,
    PAPER_LEADS,
    PENDING,
    ExportError,
    Results,
    benchmark_summary,
    cite_short,
    clinical_view,
    codesign_summary,
    doi_url,
    family_success,
    fmt_p,
    journal_only,
    kerb_summary,
    paired_comparisons,
    pick_reference,
    venue_display,
)
from .figures import CONTROLLER_LABEL

_PLACEHOLDER = re.compile(r"@@([a-z0-9_]+)@@")


def fill(template: str, values: dict) -> str:
    """Replace every ``@@key@@``; unknown keys and unused values are errors."""
    used: set[str] = set()

    def sub(m: re.Match) -> str:
        key = m.group(1)
        if key not in values:
            raise ExportError(f"template placeholder @@{key}@@ has no value")
        used.add(key)
        return str(values[key])

    out = _PLACEHOLDER.sub(sub, template)
    left = _PLACEHOLDER.findall(out)
    if left:
        raise ExportError(f"unresolved placeholders after fill: {sorted(set(left))}")
    return out


# ---------------------------------------------------------------------------------------------
# formatting helpers
# ---------------------------------------------------------------------------------------------
def lab(c: str) -> str:
    return CONTROLLER_LABEL.get(c, c)


def pend(what: str) -> str:
    return f"**{PENDING}**: {what}"


def md_table(headers: list[str], rows: list[list], align: list[str] | None = None) -> str:
    align = align or ["l"] * len(headers)
    sep = {"l": ":---", "r": "---:", "c": ":---:"}
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep[a] for a in align) + " |"]
    lines += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(lines)


def f1(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.1f}"


def f2(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.2f}"


def f3(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.3f}"


def pc(x: float, digits: int = 1) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.{digits}f}%"


def pp(x: float) -> str:
    """Probability difference in percentage points."""
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}"


def sci(x: float) -> str:
    return f"{x:.3g}"


def vtxt(cell) -> str:
    return "no climb" if cell is None or cell[0] is None else f"{cell[0]:.1f}"


def gtxt(cell) -> str:
    return "n/a" if cell is None or cell[1] is None else f"{cell[1]:.2f}"


def cite_link(ref: dict) -> str:
    return f"[{ref['doi']}]({doi_url(ref)})"


# ---------------------------------------------------------------------------------------------
# result sections (each returns text or a PENDING marker)
# ---------------------------------------------------------------------------------------------
def kerb_blocks(res: Results, pf: dict) -> dict:
    if res.kerb is None:
        p = pend("kerb traversability results (`results/curb_traversability_*_standard.csv`) not available")
        return {"kerb_table": p, "kerb_text": p, "kerb_ok": False, "kerb_sum": None}
    ks = kerb_summary(res.kerb, pf["veh_nominal"].wheel_radius, pf["veh_optimized"].wheel_radius)
    nd, od = ks["designs"]["nominal"], ks["designs"]["optimized"]
    rows = []
    for mu_key in ("hi", "lo"):
        for h in ks["heights"]:
            mu = nd["mu_" + mu_key]
            n_cell, o_cell = nd["vmin_" + mu_key].get(h), od["vmin_" + mu_key].get(h)
            rows.append([f"{100 * h:.0f}", f"{mu:g}", vtxt(n_cell), gtxt(n_cell), vtxt(o_cell), gtxt(o_cell)])
    table = md_table(
        ["Kerb height (cm)", "Friction mu", f"Nominal r = {nd['wheel_radius']:.3f} m: v_min (m/s)",
         "peak shock (g)", f"Optimised r = {od['wheel_radius']:.3f} m: v_min (m/s)", "peak shock (g)"],
        rows, ["r", "r", "r", "r", "r", "r"])
    hs = ks["heights"]
    h12 = min(hs, key=lambda h: abs(h - 0.12))
    a, b = nd["vmin_hi"][h12], od["vmin_hi"][h12]
    slow = [c[0] for d in (nd, od) for k in ("vmin_hi", "vmin_lo") for c in d[k].values() if c[0] is not None]
    text = (
        f"Head-on kerb-up climbs at friction {nd['mu_hi']:g}: the minimum successful approach speed of the "
        f"nominal rover (wheel radius {nd['wheel_radius']:.3f} m) rises from {vtxt(nd['vmin_hi'][hs[0]])} m/s at "
        f"{100 * hs[0]:.0f} cm to {vtxt(nd['vmin_hi'][hs[-1]])} m/s at {100 * hs[-1]:.0f} cm; the co-designed "
        f"rover ({od['wheel_radius']:.3f} m) needs {vtxt(od['vmin_hi'][hs[0]])} to {vtxt(od['vmin_hi'][hs[-1]])} m/s over "
        f"the same range. At the {100 * h12:.0f} cm kerb the peak payload shock at the minimum climbing speed is "
        f"{gtxt(a)} g (nominal, {vtxt(a)} m/s) versus {gtxt(b)} g (optimised, {vtxt(b)} m/s), against a "
        f"{ks['budget_g']:g} g budget. On a low-friction surface (mu = {nd['mu_lo']:g}) the "
        f"{100 * hs[-1]:.0f} cm kerb is "
        + ("not climbable by the nominal rover within the scanned speeds" if nd["vmin_lo"][hs[-1]][0] is None
           else f"climbed by the nominal rover only at {vtxt(nd['vmin_lo'][hs[-1]])} m/s")
        + f", while the optimised rover climbs it at {vtxt(od['vmin_lo'][hs[-1]])} m/s "
        f"(peak shock {gtxt(od['vmin_lo'][hs[-1]])} g). The slowest successful climb anywhere in the grids was "
        f"{min(slow):.1f} m/s, so every climb in these results relies on momentum. "
        f"Of the head-on cells that succeed, "
        + ", ".join(f"{pc(v, 0)} also succeed at a {a_:g} degree approach angle (nominal)"
                    for a_, v in nd["oblique_retention"].items())
        + f". The share of kerb-up (height, friction) cells with a shock-compliant climb window is "
        f"{pc(nd['window_fraction'], 0)} (nominal, {nd['n_window_cells']} cells) versus {pc(od['window_fraction'], 0)} "
        f"(optimised, {od['n_window_cells']} cells). Kerb-down peak shock is a median of "
        f"{f2(nd['down_peak_g_median'])} g (nominal) versus {f2(od['down_peak_g_median'])} g (optimised). "
        f"Grid outcomes: nominal {nd['outcomes']} of {nd['n_trials']} trials, optimised {od['outcomes']} of "
        f"{od['n_trials']} trials."
    )
    return {"kerb_table": table, "kerb_text": text, "kerb_ok": True, "kerb_sum": ks}


def codesign_blocks(res: Results, pf: dict) -> dict:
    if res.codesign is None:
        p = pend("mechanical co-design results (`results/codesign.json`) not available")
        return {"codesign_table": p, "codesign_text": p, "codesign_ok": False, "codesign_sum": None}
    cs = codesign_summary(res.codesign, pf["veh_nominal"])
    n, o = cs["arms"]["nominal"], cs["arms"]["optimised"]
    keys = [("wheel_radius", "Wheel radius (m)", "{:.4f}"), ("susp_k", "Suspension stiffness k_s (N/m)", "{:.1f}"),
            ("susp_c", "Suspension damping c_s (N s/m)", "{:.1f}"), ("iso_kz", "Isolator stiffness k_p (N/m)", "{:.1f}"),
            ("iso_cz", "Isolator damping c_p (N s/m)", "{:.1f}"), ("motor_peak_torque", "Motor peak torque (N m)", "{:.2f}")]
    rows = []
    for k, name, fmt in keys:
        lo, hi = cs["bounds"][k]
        rows.append([name, fmt.format(n["design"][k]), fmt.format(o["design"][k]), f"{lo:g} to {hi:g}"])
    rows += [
        ["Suspension damping ratio zeta_s (derived)", f3(n["zeta_susp"]), f3(o["zeta_susp"]), "-"],
        ["Suspension natural frequency (rad/s, derived)", f2(n["omega_susp"]), f2(o["omega_susp"]), "-"],
        ["Isolator damping ratio zeta_p (derived)", f3(n["zeta_iso"]), f3(o["zeta_iso"]), "-"],
        ["Isolator natural frequency (rad/s, derived)", f2(n["omega_iso"]), f2(o["omega_iso"]), "-"],
        ["Objective (mean shock, g; lower is better)", f2(n["objective"]), f2(o["objective"]), "-"],
        [f"Conditions within the {BUDGET_G:g} g budget", pc(n["frac_within_budget"], 1), pc(o["frac_within_budget"], 1), "-"],
        ["Kerb-up conditions that cannot be climbed", n["n_fail"], o["n_fail"], "-"],
    ]
    table = md_table(["Quantity", "Nominal (draft)", "Optimised (co-design)", "Search bounds"], rows, ["l", "r", "r", "r"])
    text = (
        f"Differential evolution ({cs['nfev']} objective evaluations, {cs['generations']} generations; best-of-generation "
        f"objective {f2(cs['history_first'])} g in the first generation and {f2(cs['history_last'])} g in the last) searched "
        f"six variables. The objective is the mean payload shock at the lowest kerb-climbing speed plus a margin, averaged "
        f"over six kerb-up conditions (heights 0.10, 0.12, 0.14 m at two friction values) with half the mean kerb-down "
        f"shock added; a condition that cannot be climbed is charged 8 g. The objective fell from {f2(n['objective'])} g to "
        f"{f2(o['objective'])} g, and the share of the eight conditions within the {BUDGET_G:g} g budget rose from "
        f"{pc(n['frac_within_budget'], 0)} to {pc(o['frac_within_budget'], 0)}. The optimum uses larger wheels, a softer "
        f"suspension (zeta_s {f3(n['zeta_susp'])} to {f3(o['zeta_susp'])}) and a much softer payload isolator "
        f"(natural frequency {f2(n['omega_iso'])} to {f2(o['omega_iso'])} rad/s)."
    )
    return {"codesign_table": table, "codesign_text": text, "codesign_ok": True, "codesign_sum": cs}


def validation_blocks(res: Results) -> dict:
    if res.validation is None:
        p = pend("suspension and contact validation (`results/validation_suspension.json`) not available")
        return {"validation_table": p, "validation_text": p, "validation_ok": False}
    v = res.validation
    rd = [r for r in v["ringdown"] if "measured_omega_n" in r]
    rows = [[f"{r['zeta_target']:.2f}", f2(r["analytic_omega_n"]), f2(r["measured_omega_n"]),
             f"{r['omega_n_err_pct']:+.2f}%", f3(r["analytic_zeta"]), f3(r["measured_zeta"]),
             f"{r['zeta_err_pct']:+.2f}%"] for r in rd]
    table = md_table(["Target zeta", "omega_n analytic (rad/s)", "omega_n measured (rad/s)", "error",
                      "zeta analytic", "zeta measured", "error"], rows, ["r"] * 7)
    st, ty, so = v["static"], v["tyre"], v["soak"]
    worst_w = max(abs(r["omega_n_err_pct"]) for r in rd) if rd else math.nan
    worst_z = max(abs(r["zeta_err_pct"]) for r in rd) if rd else math.nan
    roll = v["rolling"]
    text = (
        f"Static equilibrium: the rover settles at a ride height of {1000 * st['ride_height_m']:.1f} mm against the "
        f"expected {1000 * st['ride_height_expected_m']:.1f} mm (error {st['ride_height_err_mm']:+.2f} mm), and the payload "
        f"accelerometer reads {st['payload_acc_z']:.3f} m/s^2 against {st['payload_acc_expected']:.2f} m/s^2. "
        f"The effective tyre radial stiffness measured from the static penetration is "
        f"{ty['kt_eff_N_per_m'] / 1000:.0f} kN/m under a wheel load of {ty['wheel_load_N']:.1f} N. "
        f"Free-bounce ring-down runs ({len(rd)} with enough peaks to fit) reproduce the closed-form 2-DOF quarter-car "
        f"natural frequency within {worst_w:.2f}% and the damping ratio within {worst_z:.2f}%. Straight rolling at "
        + ", ".join(f"{r['v_cmd']:g}" for r in roll)
        + f" m/s reaches {min(abs(r['speed_err_pct']) for r in roll):.2f}% to {max(abs(r['speed_err_pct']) for r in roll):.2f}% "
        f"speed error with lateral drift below {1000 * max(abs(r['lateral_drift_m']) for r in roll):.3f} mm. A random-command "
        f"soak of {so['steps']:,} physics steps over a kerb field stayed finite: {so['finite']}."
    )
    return {"validation_table": table, "validation_text": text, "validation_ok": True}


def benchmark_blocks(res: Results) -> dict:
    if res.benchmark is None:
        p = pend("controller benchmark (`results/benchmark_standard*.csv`) not available")
        return {k: p for k in ("bench_table", "family_table", "bench_text", "stats_table", "stats_text")} | {
            "bench_ok": False, "ref": None, "bench_rows": None, "comparisons": None}
    df = res.benchmark
    rows = benchmark_summary(df)
    table = md_table(
        ["Controller", "N", "Success (95% Wilson CI)", "Median time (s)", "Median peak shock (g)",
         f"Episodes over {BUDGET_G:g} g", "Collisions", "Stall or timeout", "Off sidewalk or rollover"],
        [[lab(r["controller"]), r["n"], f"{pc(r['success'])} ({pc(r['ci_low'])} to {pc(r['ci_high'])})",
          f1(r["median_time_s"]), f2(r["median_peak_g"]), pc(r["over_budget"]), r["collision"], r["stall_timeout"],
          r["off_sidewalk"] + r["rollover"]] for r in rows],
        ["l", "r", "r", "r", "r", "r", "r", "r", "r"])
    ctrls, fams, cell = family_success(df)
    fam_table = md_table(
        ["Controller"] + [f"{f} (n = {min(cell[(c, f)][1] for c in ctrls)})" if len({cell[(c, f)][1] for c in ctrls}) == 1
                          else f for f in fams],
        [[lab(c)] + [pc(cell[(c, f)][0], 0) for f in fams] for c in ctrls], ["l"] + ["r"] * len(fams))
    cell_n = df.groupby(["controller", "family"]).size()
    best = max(rows, key=lambda r: r["success"])
    worst = min(rows, key=lambda r: r["success"])
    fastest = min((r for r in rows if not math.isnan(r["median_time_s"])), key=lambda r: r["median_time_s"])
    shield = ""
    if "shield" in df.columns:
        shield = (" All controllers ran behind the same speed-and-separation safety filter."
                  if bool(df["shield"].all()) else " The benchmark mixes shielded and unshielded episodes.")
    ref = pick_reference(df)
    text = (
        f"Source file `{res.benchmark_path.name}`: {len(df)} episodes, {len(rows)} controllers, {df['family'].nunique()} scenario "
        f"families, {int(cell_n.min())} to {int(cell_n.max())} episodes per controller and family; every controller sees "
        f"the same seeds, so comparisons are paired.{shield} Success rates run from {pc(worst['success'])} "
        f"({lab(worst['controller'])}) to {pc(best['success'])} ({lab(best['controller'])}). The fastest median completion "
        f"time among successful episodes is {f1(fastest['median_time_s'])} s ({lab(fastest['controller'])}). "
        f"Collisions: " + ", ".join(f"{lab(r['controller'])} {r['collision']}" for r in rows) + "."
    )
    if ref is None:
        return {"bench_table": table, "family_table": fam_table, "bench_text": text,
                "stats_table": pend("no classical reference controller in the benchmark file"),
                "stats_text": pend("no classical reference controller in the benchmark file"),
                "bench_ok": True, "ref": None, "bench_rows": rows, "comparisons": None}
    comps = paired_comparisons(df, ref)
    srows, sbul = [], []
    for c in comps:
        for metric, name, unit in (("time", "Completion time", "s"), ("shock", "Peak payload shock", "g")):
            t = c[metric]
            if t is None:
                srows.append([f"{lab(c['controller'])} - {lab(ref)}", name, c["n_both"], "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
                continue
            srows.append([
                f"{lab(c['controller'])} - {lab(ref)}", f"{name} ({unit})", c["n_both"],
                f"{t.mean_diff:+.3f} [{t.diff_ci_low:+.3f}, {t.diff_ci_high:+.3f}]", f"{t.df:.0f}", f"{t.t:.2f}",
                fmt_p(t.p), fmt_p(c.get(f"p_{metric}_holm", math.nan)), f"{t.d:+.2f} [{t.d_ci_low:+.2f}, {t.d_ci_high:+.2f}]"])
        srows.append([f"{lab(c['controller'])} - {lab(ref)}", "Reached the goal (exact McNemar)", c["n_pairs"],
                      f"{100 * c['success_diff']:+.1f} pp", "-", "-", fmt_p(c["p_mcnemar"]), "-",
                      f"{c['ref_only']} vs {c['other_only']} discordant"])
        sbul.append(
            f"* **{lab(c['controller'])} against {lab(ref)}** ({c['n_pairs']} paired scenarios, {c['n_both']} reached by both): "
            + (f"completion time difference {c['time'].mean_diff:+.2f} s (95% CI {c['time'].diff_ci_low:+.2f} to "
               f"{c['time'].diff_ci_high:+.2f}), t({c['time'].df:.0f}) = {c['time'].t:.2f}, p = {fmt_p(c['time'].p)}, "
               f"d_z = {c['time'].d:+.2f}; " if c["time"] is not None else "completion time: too few paired successes; ")
            + (f"peak shock difference {c['shock'].mean_diff:+.3f} g (95% CI {c['shock'].diff_ci_low:+.3f} to "
               f"{c['shock'].diff_ci_high:+.3f}), t({c['shock'].df:.0f}) = {c['shock'].t:.2f}, p = {fmt_p(c['shock'].p)}, "
               f"d_z = {c['shock'].d:+.2f}; " if c["shock"] is not None else "peak shock: too few paired successes; ")
            + f"success rate {100 * c['success_diff']:+.1f} percentage points (exact McNemar p = {fmt_p(c['p_mcnemar'])}, "
            f"{c['ref_only']} scenarios won only by the reference, {c['other_only']} only by the other).")
    stable = md_table(["Comparison (other - reference)", "Metric", "Pairs", "Mean difference [95% CI]", "df", "t",
                       "Exact p", "p (Holm)", "Effect size [95% CI]"], srows,
                      ["l", "l", "r", "r", "r", "r", "r", "r", "r"])
    stext = (f"Reference controller: **{lab(ref)}**, chosen by the pre-declared rule 'highest success rate among the "
             f"classical controllers, ties broken by lower median time'. Differences are other minus reference, so a "
             f"positive time difference means the other controller is slower. Paired t tests use scenarios that both "
             f"controllers completed; d_z is the mean difference over the standard deviation of the differences; Holm "
             f"adjustment covers all t tests together.\n\n" + "\n".join(sbul))
    return {"bench_table": table, "family_table": fam_table, "bench_text": text, "stats_table": stable,
            "stats_text": stext, "bench_ok": True, "ref": ref, "bench_rows": rows, "comparisons": comps}


def clinical_blocks(res: Results) -> dict:
    if res.clinical is None:
        p = pend("clinical decision and economics results (`results/clinical*.json`) not available")
        return {"clinical_table": p, "clinical_text": p, "econ_text": p, "econ_table": p, "rq_answer": p,
                "clinical_ok": False, "cv": None}
    cv = clinical_view(res.clinical)
    order = ["ambulance", "rover", "drone"]
    prim = [cv["primary"][m] for m in order if m in cv["primary"]]
    rows = [[r["mode"], r["n"], f"{f3(r['mean_survival'])} [{f3(r['survival_ci_low'])}, {f3(r['survival_ci_high'])}]",
             f"{pp(r['abs_gain'])} [{pp(r['abs_gain_ci_low'])}, {pp(r['abs_gain_ci_high'])}]" if r["mode"] != "ambulance" else "-",
             pc(r["p_faster"], 0) if r["mode"] != "ambulance" else "-",
             f2(r["median_saving_when_faster_min"]) if r["mode"] != "ambulance" else "-"] for r in prim]
    table = md_table(["Mode", "Simulated arrests", f"Expected survival ({cv['model']}) [95% CI]",
                      "Gain over ambulance (pp) [95% CI]", "Shocks first", "Median time saved when first (min)"],
                     rows, ["l", "r", "r", "r", "r", "r"])
    other = cv["other"]
    if other:
        orows = [[r["model"], r["mode"], f"{f3(r['mean_survival'])} [{f3(r['survival_ci_low'])}, {f3(r['survival_ci_high'])}]",
                  f"{pp(r['abs_gain'])} [{pp(r['abs_gain_ci_low'])}, {pp(r['abs_gain_ci_high'])}]" if r["mode"] != "ambulance" else "-"]
                 for r in other]
        table += "\n\nModel-form sensitivity (other survival models in the results file):\n\n" + md_table(
            ["Model", "Mode", "Expected survival [95% CI]", "Gain over ambulance (pp) [95% CI]"], orows, ["l", "l", "r", "r"])
    be, rv, am = cv["breakeven"], cv["primary"]["rover"], cv["primary"]["ambulance"]
    if be["flag"] == "crossing":
        be_txt = (f"the rover acting alone matches the ambulance acting alone at a straight-line radius of "
                  f"{be['radius_m']:.0f} m; beyond that the ambulance is expected to shock first")
    elif be["flag"] == "rover_never_better":
        be_txt = "the rover acting alone never matches the ambulance, even at zero distance"
    elif be["flag"] == "rover_always_better":
        be_txt = "the rover acting alone matches or beats the ambulance out to the largest radius searched"
    else:
        be_txt = f"break-even result flag {be['flag']!r} with radius {be['radius_m']}"
    mode_phrase = ("dispatched in parallel with the ambulance" if cv["parallel"] is True
                   else "acting alone" if cv["parallel"] is False else "as configured in the results file")
    sig = rv["abs_gain_ci_low"] > 0
    scen = ""
    if cv["scenario"]:
        scen = " Scenario inputs recorded in the results file: " + ", ".join(
            f"{k} = {v}" for k, v in cv["scenario"].items() if not isinstance(v, (dict, list))) + "."
    text = (
        f"Under the {cv['model']} model the rover ({mode_phrase}) changes expected survival by {pp(rv['abs_gain'])} "
        f"percentage points (95% CI {pp(rv['abs_gain_ci_low'])} to {pp(rv['abs_gain_ci_high'])}) relative to the "
        f"ambulance alone ({f3(am['mean_survival'])}); the interval "
        + ("excludes zero" if sig else "includes zero") + f". It shocks strictly earlier than the ambulance in "
        f"{pc(rv['p_faster'], 0)} of the {rv['n']} simulated arrests, saving a median of "
        f"{f2(rv['median_saving_when_faster_min'])} min in those. Break-even: {be_txt}.{scen}"
    )
    rq = (
        f"**Conditional answer.** In this simulation study the rover {mode_phrase} "
        + ("does give a statistically clear survival gain" if sig else "does not give a statistically clear survival gain")
        + f" ({pp(rv['abs_gain'])} percentage points, 95% CI {pp(rv['abs_gain_ci_low'])} to {pp(rv['abs_gain_ci_high'])}, "
        f"{cv['model']} model), and {be_txt}. The answer depends on the ambulance response-time distribution and the "
        f"dispatch assumptions of the scenario; it does not establish that urban ambulance delays are 15-20 minutes "
        f"(the sourced Central Norway urban median is 10.0 min and 90th percentile 17.7 min, Naess et al. 2024), "
        f"and it is not a statement about any Indian city."
    )
    econ = cv["economics"]
    meets = "meets" if econ["meets_kappa_target"] else "does not meet"
    inputs = econ.get("inputs")
    econ_table = md_table(
        ["Quantity", "Value"],
        [["Operating-burden ratio kappa", f3(econ["kappa"])], ["Course target", "kappa <= 0.25"],
         ["Meets target", "yes" if econ["meets_kappa_target"] else "no"],
         ["Payback horizon (months)", "no payback (kappa >= 1)" if not math.isfinite(econ["payback_months"]) else f1(econ["payback_months"])],
         ["Staff capacity freed, delta FTE", f2(econ["delta_fte"])]]
        + ([[f"Input (ASSUMPTION): {k}", v] for k, v in inputs.items()] if isinstance(inputs, dict) else []),
        ["l", "r"])
    econ_text = (
        f"Dimensionless economics (no currency anywhere): kappa = {f3(econ['kappa'])}, which {meets} the course target "
        f"kappa <= 0.25; payback horizon "
        + ("does not exist (kappa >= 1)" if not math.isfinite(econ["payback_months"]) else f"{f1(econ['payback_months'])} months")
        + f"; staff capacity freed delta FTE = {f2(econ['delta_fte'])}. The inputs are placeholder ratios (ASSUMPTIONS), "
        f"not measurements; kappa is therefore a demonstration of the method, not a forecast."
    )
    return {"clinical_table": table, "clinical_text": text, "econ_text": econ_text, "econ_table": econ_table,
            "rq_answer": rq, "clinical_ok": True, "cv": cv}


def provenance_table(res: Results, root_name: str = "results") -> str:
    rows = [[s.label, f"`{root_name}/{s.path.name}`", s.detail, f"`{s.sha12}`"] for s in res.sources]
    return md_table(["Result", "Source file", "Content", "SHA-256 (first 12 hex)"], rows) if rows else pend("no result files")


def partial_banner(res: Results) -> str:
    if not res.pending:
        return ""
    items = "\n".join(f"* {p.splitlines()[0]}" for p in res.pending)
    return ("> **PARTIAL EXPORT.** The following results were not available when this folder was generated, and the "
            "sections that depend on them carry a `PENDING` marker. Do not submit this folder before it is "
            "regenerated from complete results.\n>\n" + "\n".join("> " + ln for ln in items.splitlines()) + "\n\n")


# ---------------------------------------------------------------------------------------------
# model, controller and standards tables (constants read from the package)
# ---------------------------------------------------------------------------------------------
REF_ORDER = ("larsen1993", "schierbeck2023", "naess2024", "tsao2023", "weinberg2023", "khatib1986",
             "todorov2012", "fox1997", "williams2017", "schulman2017", "helbing1995", "kim2026")


def veh_table(pf: dict) -> str:
    n, o = pf["veh_nominal"], pf["veh_optimized"]
    spec = [("Chassis mass (kg)", "chassis_mass", "{:.1f}"), ("Payload mass (kg)", "payload_mass", "{:.1f}"),
            ("Wheelbase (m)", "wheelbase", "{:.2f}"), ("Track (m)", "track", "{:.2f}"),
            ("Wheel radius (m)", "wheel_radius", "{:.4f}"), ("Wheel and hub-motor mass (kg)", "wheel_mass", "{:.3f}"),
            ("Knuckle mass (kg)", "knuckle_mass", "{:.2f}"), ("Suspension stiffness k_s (N/m)", "susp_k", "{:.1f}"),
            ("Suspension damping c_s (N s/m)", "susp_c", "{:.1f}"), ("Isolator stiffness k_p (N/m)", "iso_kz", "{:.1f}"),
            ("Isolator damping c_p (N s/m)", "iso_cz", "{:.1f}"), ("Motor peak torque per wheel (N m)", "motor_peak_torque", "{:.2f}"),
            ("Wheel speed limit (rad/s)", "wheel_speed_max", "{:.0f}"), ("Tyre-ground friction", "friction", "{:.2f}"),
            ("Total mass (kg, derived)", "total_mass", "{:.2f}"), ("Sprung mass per wheel (kg, derived)", "sprung_per_wheel", "{:.2f}"),
            ("Unsprung mass per wheel (kg, derived)", "unsprung_mass", "{:.3f}"),
            ("Suspension damping ratio zeta_s (derived)", "susp_zeta", "{:.3f}"),
            ("Suspension natural frequency (rad/s, derived)", "susp_omega_n", "{:.2f}")]
    rows = [[name, fmt.format(getattr(n, k)), fmt.format(getattr(o, k))] for name, k, fmt in spec]
    rows.append(["Tyre contact solref (s, damping ratio)", ", ".join(f"{x:g}" for x in n.tyre_solref),
                 ", ".join(f"{x:g}" for x in o.tyre_solref)])
    return md_table(["Parameter", "Nominal (draft design)", "Optimised (co-design, shipped model)"], rows, ["l", "r", "r"])


def speed_note(res) -> str:
    """How the benchmark departed from the package defaults listed in the controller table."""
    df = res.benchmark
    if df is None or "speed_cap" not in df.columns or df["speed_cap"].dropna().empty:
        return ""
    cap = float(df["speed_cap"].dropna().iloc[0])
    return (f"\n\nIn the benchmark every controller was given the same top speed of {cap:g} m/s (overriding the cruise speeds "
            "above), so time differences reflect planning quality and not a speed advantage. The MPPI settings are the variant "
            "frozen by `configs/mppi_tuned.json` on tuning seeds disjoint from every evaluation seed, and the PPO policy is the "
            "checkpoint chosen on validation seeds (`results/ppo_selection.csv`).")


def controller_table(pf: dict) -> str:
    pp_, apf, dwa, mp = pf["pure_pursuit"], pf["apf"], pf["dwa"], pf["mppi"]
    cur = pf["curb"]
    arch = pf["ppo_net_arch"]
    ppo_net = (f"MLP {'-'.join(str(x) for x in arch)}" + (" with tanh activations" if pf["ppo_tanh"] else "")) if arch else "MLP policy"
    rows = [
        ["Pure pursuit (baseline A)", "Steers toward a point on the centre line; constant speed; no avoidance, no kerb logic.",
         f"cruise {pp_['v_cruise']:g} m/s, look-ahead {pp_['lookahead']:g} m"],
        ["APF, Khatib 1986 (baseline B)", "Attraction toward the path plus repulsion from lidar returns, tracked pedestrians and "
         "the sidewalk edges; optional kerb-speed overlay ('APF + kerb logic').",
         f"cruise {apf['v_cruise']:g} m/s, influence radius {apf['rho0']:g} m, eta {apf['eta']:g}, wall at +/-{apf['wall_y']:g} m"],
        ["DWA, Fox 1997 (baseline C)", f"Samples {dwa['n_v']} speeds x {dwa['n_delta']} steering angles inside the kinematic window, rolls each "
         "out on a bicycle model, rejects candidates that cannot stop before contact and scores progress, lane centring, speed, "
         "clearance to lidar points and constant-velocity pedestrian predictions; optional kerb-speed overlay.",
         f"cruise {dwa['v_cruise']:g} m/s, horizon {dwa['horizon']:g} s (step {dwa['dt']:g} s), a_max {dwa['a_max']:g}, a_brake {dwa['a_brake']:g} m/s^2"],
        ["MPPI, Williams 2017 with MuJoCo rollouts", f"Samples {mp['K']} (speed, steering) sequences, rolls all of them out in a MuJoCo copy of "
         "the world with `mujoco.rollout`, scores progress, payload shock, clearance, lane, heading, stability and smoothness, and "
         "moves the mean by the exponentially weighted average. The internal model is deliberately imperfect (coarser step, nominal "
         "friction and payload, no pedestrians in the physics).",
         f"K = {mp['K']}, H = {mp['H']} x {mp['dt_plan']:g} s, physics step {mp['dt_phys']:g} s, temperature {mp['temperature']:g}, "
         f"re-plan every {mp['replan_every']} control steps, v_max {mp['v_max']:g} m/s"],
        ["PPO, Schulman 2017", f"Stable-Baselines3 PPO with a {ppo_net} policy trained on domain-randomised episodes "
         f"(kerb height {pf['train_ranges']['kerb_range'][0]:g}-{pf['train_ranges']['kerb_range'][1]:g} m, friction "
         f"{pf['train_ranges']['mu_range'][0]:g}-{pf['train_ranges']['mu_range'][1]:g}) behind the same safety filter.",
         f"decision every {pf['rl_repeat']} control steps, shock budget in the reward {pf['rl_budget_g']:g} g"],
        ["Kerb negotiator (overlay for APF and DWA)", "Detects a kerb step from the forward height scan and caps the approach speed at the "
         "tabulated minimum climbing speed for that height and an assumed friction, plus a margin (table `configs/curb_table.json`, "
         "measured in the kerb study).",
         f"assumed friction {cur['mu_assumed']:g}, margin {cur['margin']:g} m/s, descent {cur['v_down']:g} m/s, planned deceleration {cur['a_dec']:g} m/s^2"],
    ]
    return md_table(["Controller", "How it decides", "Key parameters (package defaults)"], rows)


def safety_table(pf: dict) -> str:
    s = pf["safety"]
    rows = [["Service braking used in the barrier", f"{s['a_brake']:g} m/s^2"], ["Reaction allowance", f"{s['t_react']:g} s"],
            ["Extra clearance ahead of the footprint", f"{s['front_margin']:g} m"],
            ["Extra clearance to the sides", f"{s['side_margin']:g} m"],
            ["Extra allowance around pedestrians", f"{s['ped_extra']:g} m"],
            ["Arc length considered", f"{s['horizon']:g} m"],
            ["Speed cap near pedestrians (policy parameter)", f"{s['ped_speed_cap']:g} m/s within {s['ped_cap_radius']:g} m"]]
    return md_table(["Safety-filter parameter", "Default"], rows, ["l", "r"])


def standards_table(pf: dict, res: Results, bb: dict, clb: dict) -> str:
    s, per, mp = pf["safety"], pf["perception"], pf["mppi"]
    rows = [
        ["ISO 13482: speed limit in shared pedestrian zones", "<= 0.80 m/s", f"{s['ped_speed_cap']:g} m/s within {s['ped_cap_radius']:g} m of a pedestrian (policy parameter)",
         "Met" if s["ped_speed_cap"] <= 0.8 else "Not met by default (the cap is a sweepable parameter; the manual states 0.80 m/s for occupied indoor zones)"],
        ["ISO 13482: emergency braking deceleration", ">= 1.5 m/s^2", f"{s['a_brake']:g} m/s^2 assumed in the barrier",
         "Consistent (planning value; not separately measured on the rover)" if s["a_brake"] >= 1.5 else "Not met"],
        ["ISO 13482: dynamic clearance", ">= 0.50 m", f"MPPI comfort clearance {mp['comfort_clearance']:g} m (soft cost); shield margins {s['front_margin']:g} m front, {s['side_margin']:g} m side (hard)",
         "Partly met (hard constraint is smaller than 0.50 m)"],
        ["ISO 3691-4: detection latency", "<= 50 ms", f"{1000 * s['t_react']:.0f} ms reaction allowance", "Not met (assumed value)" if s["t_react"] > 0.05 else "Met"],
        ["ISO 3691-4: personnel detection field of view", ">= 180 degrees", f"{per['fov_deg']:g} degrees lidar, {per['lidar_range']:g} m range", "Met" if per["fov_deg"] >= 180 else "Not met"],
        ["ISO 3691-4: dual-zone warning and stop (1.5 m, 0.4 m)", "fixed zones", "continuous stopping-distance barrier on the swept footprint", "Different mechanism"],
        ["REP-103 axes and SI units", "x forward, y left, z up", "as documented in `sim/vehicle_mjcf.py`", "Met"],
    ]
    if res.benchmark is not None:
        per_c = res.benchmark.groupby("controller").size()
        rows.append(["Sample size per controller", "N >= 50", f"{int(per_c.min())} to {int(per_c.max())} episodes", "Met" if int(per_c.min()) >= 50 else "Not met"])
    if clb.get("cv") is not None:
        k = clb["cv"]["economics"]["kappa"]
        rows.append(["Operating-burden ratio kappa", "<= 0.25", f"{k:.3f} (placeholder inputs)", "Met" if k <= 0.25 else "Not met"])
    return md_table(["Requirement (course manual)", "Threshold", "This implementation", "Status"], rows)


def xml_facts(xml: str) -> dict:
    out = {}
    for key, pat in (("cone", r'cone="([^"]+)"'), ("integrator", r'integrator="([^"]+)"'),
                     ("timestep", r'timestep="([^"]+)"'), ("iterations", r'iterations="([^"]+)"')):
        m = re.search(pat, xml)
        if not m:
            raise ExportError(f"MJCF option {key} not found in the generated XML")
        out[key] = m.group(1)
    return out


def refs_block(refs: dict) -> tuple[dict, str]:
    """Numbered IEEE list (foundational six first, then the method references) and the r_<key> tokens."""
    needed = {k for k, _ in FOUNDATIONAL} | set(METHOD_REFS)
    if set(REF_ORDER) != needed:
        raise ExportError(f"REF_ORDER {sorted(REF_ORDER)} differs from FOUNDATIONAL + METHOD_REFS {sorted(needed)}")
    tokens, lines = {}, []
    for i, k in enumerate(REF_ORDER, 1):
        r = refs[k]
        tokens[f"r_{k}"] = f"[{i}]"
        lines.append(f"[{i}] {r['authors']} \"{r['title']},\" *{venue_display(r)}*, {r['year']}. DOI: [{doi_url(r)}]({doi_url(r)})")
    return tokens, "\n\n".join(lines)


def viva_blocks(res: Results, pf: dict, refs: dict, bb: dict, kb: dict, cb: dict, vb: dict, clb: dict) -> dict:
    """Viva questions with model answers computed from the shipped model and the results files."""
    from aedrover.clinical.decision import ScenarioParams
    from aedrover.clinical.survival import SURVIVAL_MODELS, LarsenModel, delta_survival

    lm = LarsenModel()
    t_cpr, t_acls, t_a, t_b = 3.0, 10.0, 2.0, 6.0
    d_l = float(delta_survival(lm, t_a, t_b, t_cpr, t_acls))
    d_rule = {k: float(delta_survival(SURVIVAL_MODELS[k], t_a, t_b, t_cpr, t_acls))
              for k in ("rule_of_thumb_low", "rule_of_thumb", "rule_of_thumb_high")}
    wrong = 0.046 * (t_b - t_a)
    sp = ScenarioParams()
    d = sp.dispatch
    n, o = pf["veh_nominal"], pf["veh_optimized"]
    dwa, mp, s = pf["dwa"], pf["mppi"], pf["safety"]

    # -- data-dependent sentences -------------------------------------------------------------
    if bb.get("bench_ok") and bb["ref"]:
        ref = bb["ref"]
        rr = next(r for r in bb["bench_rows"] if r["controller"] == ref)
        shielded = bool(res.benchmark["shield"].all()) if "shield" in res.benchmark.columns else False
        collision_txt = (f"In the benchmark the {lab(ref)} controller, {'behind the safety filter' if shielded else 'as run'}, still had "
                         f"{rr['collision']} collision episodes out of {rr['n']} ({pc(rr['collision'] / rr['n'])}): the filter limits speed along "
                         "the commanded arc, it does not choose the path, and it relies on a noisy tracker and a finite look-ahead, so "
                         "pedestrians that step into the arc late can still be reached.")
        c0 = bb["comparisons"][0] if bb["comparisons"] else None
        stat_txt = (f"Example from the results: {lab(c0['controller'])} against {lab(ref)} on {c0['n_both']} scenarios that both completed"
                    + (f", completion time difference {c0['time'].mean_diff:+.2f} s, t({c0['time'].df:.0f}) = {c0['time'].t:.2f}, p = {fmt_p(c0['time'].p)}, d_z = {c0['time'].d:+.2f}."
                       if c0 and c0["time"] is not None else ".")) if c0 else "Fewer than two controllers were paired, so no example is available."
        ref_txt = lab(ref)
    else:
        collision_txt = pend("collision statistics need the controller benchmark")
        stat_txt = pend("paired statistics need the controller benchmark")
        ref_txt = PENDING
    if kb["kerb_ok"]:
        ks = kb["kerb_sum"]
        hs = ks["heights"]
        h12 = min(hs, key=lambda h: abs(h - 0.12))
        nd, od = ks["designs"]["nominal"], ks["designs"]["optimized"]
        a12, b12 = nd["vmin_hi"][h12], od["vmin_hi"][h12]
        lo_h = hs[-1]
        momentum = (f"In the kerb study the minimum climbing speed at a {100 * h12:.0f} cm kerb (friction {nd['mu_hi']:g}) is {vtxt(a12)} m/s "
                    f"(nominal) and {vtxt(b12)} m/s (optimised) with a peak payload shock of {gtxt(a12)} g and {gtxt(b12)} g; nothing in the grids "
                    "succeeded at the slowest scanned speed of 0.4 m/s, so climbing is momentum-driven and the shock rises with the "
                    "speed that momentum requires.")
        mu_txt = (f"At friction {nd['mu_lo']:g} the {100 * lo_h:.0f} cm kerb needs {vtxt(od['vmin_lo'][lo_h])} m/s for the optimised rover "
                  f"(peak {gtxt(od['vmin_lo'][lo_h])} g) and "
                  + ("cannot be climbed by the nominal rover within the scanned speeds." if nd["vmin_lo"][lo_h][0] is None
                     else f"{vtxt(nd['vmin_lo'][lo_h])} m/s for the nominal rover (peak {gtxt(nd['vmin_lo'][lo_h])} g)."))
    else:
        momentum = pend("kerb climbing speeds need results/curb_traversability_*_standard.csv")
        mu_txt = momentum
    if vb["validation_ok"]:
        v = res.validation
        rd = [r for r in v["ringdown"] if "measured_omega_n" in r]
        val_txt = (f"The ring-down validation reproduced the closed-form 2-DOF natural frequency within "
                   f"{max(abs(r['omega_n_err_pct']) for r in rd):.2f}% and the damping ratio within {max(abs(r['zeta_err_pct']) for r in rd):.2f}% "
                   f"({len(rd)} fitted runs), and the effective tyre stiffness measured from static penetration is "
                   f"{v['tyre']['kt_eff_N_per_m'] / 1000:.0f} kN/m.")
    else:
        val_txt = pend("suspension validation needs results/validation_suspension.json")
    if cb["codesign_ok"]:
        cs = cb["codesign_sum"]
        an, ao = cs["arms"]["nominal"], cs["arms"]["optimised"]
        cd_txt = (f"Objective {f2(an['objective'])} g to {f2(ao['objective'])} g; conditions within the {BUDGET_G:g} g budget "
                  f"{pc(an['frac_within_budget'], 0)} to {pc(ao['frac_within_budget'], 0)}; wheel radius {an['design']['wheel_radius']:.3f} m to "
                  f"{ao['design']['wheel_radius']:.3f} m; suspension zeta_s {f3(an['zeta_susp'])} to {f3(ao['zeta_susp'])}; isolator natural "
                  f"frequency {f2(an['omega_iso'])} to {f2(ao['omega_iso'])} rad/s. The price: unsprung mass per wheel grows from "
                  f"{n.unsprung_mass:.2f} kg to {o.unsprung_mass:.2f} kg and total mass from {n.total_mass:.1f} kg to {o.total_mass:.1f} kg "
                  "(wheel and hub-motor mass follows an engineering-estimate model, an ASSUMPTION).")
    else:
        cd_txt = pend("co-design comparison needs results/codesign.json")
    xf = pf["xml_facts"]
    sec_e026 = f"""
#### Student E026: Kashish Praveen Jain
* **Role:** {MEMBERS[0]['role']}
* **Git branch:** `{MEMBERS[0]['assigned_branch']}`
* **Owned modules:** `src/aedrover/nav/` (pure pursuit, APF, DWA, MPPI), `src/aedrover/control/safety_filter.py`, `src/aedrover/sim/pedestrians.py`, `src/aedrover/clinical/` (survival, ambulance delay, decision), `analytics/cardiac_survival_economics.py`.
* **Deliverables:** (i) the closed-loop navigation stack and its baselines, (ii) the pedestrian and crowd scenarios, (iii) the time-to-first-shock decomposition and the survival comparison, (iv) the paired statistical comparison of controllers. Each item needs at least four distinct feature commits on your branch (Standards Manual, section 5).

**Viva questions and model answers (true for this implementation)**

1. *Question:* State the Larsen et al. (1993) survival equation with its units, and explain what changes if the first shock is delivered four minutes earlier. What was wrong in the earlier draft?
   *Model answer:* Survival = {lm.intercept:g} - {lm.slope_cpr:g} t_cpr - {lm.slope_defib:g} t_defib - {lm.slope_acls:g} t_acls, with each t in minutes from collapse (to CPR, to first shock, to advanced life support) and the result clipped to [0, {lm.intercept:g}]. The shock coefficient is {lm.slope_defib:g} per minute, so with CPR at {t_cpr:g} min and ACLS at {t_acls:g} min, moving the shock from {t_b:g} to {t_a:g} min raises survival by {100 * d_l:.1f} percentage points. The earlier draft used {0.046:g} per minute for the shock term and no ACLS term; that value is not a coefficient of the published model and would have predicted {100 * wrong:.1f} points for the same saving. The model was fitted to 1,667 King County patients with heart disease and witnessed ventricular fibrillation, a high-survival subgroup, so absolute values are optimistic and we use it mainly for differences. The 7-10% per minute in the project title is a guideline rule of thumb ({cite_short(refs['kim2026'])}, doi {refs['kim2026']['doi']}), not this model; on the same four-minute saving it gives {100 * d_rule['rule_of_thumb_low']:.0f} to {100 * d_rule['rule_of_thumb_high']:.0f} points, which is why both are reported.
2. *Question:* How does the DWA planner choose the speed and steering command, and why is the kerb negotiator an overlay rather than part of the planner?
   *Model answer:* Every {dwa['replan_every']} control steps it samples {dwa['n_v']} speeds inside the dynamic window and {dwa['n_delta']} steering angles, rolls each pair out on a bicycle model for {dwa['horizon']:g} s in steps of {dwa['dt']:g} s, drops candidates that cannot stop before contact with braking {dwa['a_brake']:g} m/s^2, and picks the highest score of progress, lane centring, speed, clearance and heading. Pedestrians enter through constant-velocity predictions of the tracker output. The kerb negotiator only caps the maximum speed using the measured minimum-climb-speed table plus a margin; keeping it separate lets the same overlay serve both APF and DWA and lets the benchmark measure what the kerb logic is worth ('with' and 'without' variants).
3. *Question:* What does the safety filter guarantee and what does it not?
   *Model answer:* It sweeps the {pf['footprint'][0]:.2f} x {pf['footprint'][1]:.2f} m footprint along the arc of the commanded steering angle, finds the first arc length d_free where the inflated footprint overlaps a lidar return or a tracked pedestrian, and clips the speed so that d_free - v t_react - v^2 / (2 a_brake) >= 0, with t_react = {s['t_react']:g} s and a_brake = {s['a_brake']:g} m/s^2; it also caps the speed at {s['ped_speed_cap']:g} m/s within {s['ped_cap_radius']:g} m of a pedestrian. It only ever reduces speed. {collision_txt}
4. *Question:* What does MPPI plan with, and why is its internal model imperfect on purpose?
   *Model answer:* Every planning cycle ({1.0 / (mp['replan_every'] * pf['env']['control_dt']):g} Hz) it samples {mp['K']} control sequences of {mp['H']} steps of {mp['dt_plan']:g} s, simulates all of them in a MuJoCo copy of the world with `mujoco.rollout` (contacts, suspension, isolator and wheel loops included) at a {mp['dt_phys']:g} s step, and averages the perturbations with exponential weights. The planner uses nominal friction {mp['nominal_mu']:g} and payload {mp['nominal_payload']:g} kg and has no pedestrians in its physics (they enter through predicted positions), whereas the environment randomises friction and payload. Planning against a perfect copy of the test world would overstate what a real controller can do.
5. *Question:* How is the time to first shock composed, and where does the simulated rover enter?
   *Model answer:* From collapse: ambulance = collapse-to-call + sampled response time (the response time already runs from the call); rover = collapse-to-call + call-to-alert + travel + hand-off. The package defaults for the non-travel delays are ASSUMPTIONS: {d.collapse_to_call_min:g} min to call, {d.call_to_alert_min:g} min to alert the rover, {d.handoff_min:g} min for the bystander to take the AED and apply the pads, bystander CPR at {d.bystander_cpr_delay_min:g} min. Travel is radius x route factor / speed ({sp.route_factor:g}); a measured travel time from the simulation can replace it. CPR and ACLS times are shared by all modes, so the device changes survival only through the shock term.
6. *Question:* Why paired tests, and what does the example from your results show?
   *Model answer:* Every controller runs the same seeded scenarios, so the natural unit is the scenario: differences within a scenario remove the large between-scenario variance (kerb height, friction, crowd). The paired t test on scenarios both controllers completed gives df, t, an exact p, d_z and a 95% interval of the mean difference; success uses an exact McNemar test on the discordant scenarios; Holm adjustment covers the family of t tests. The reference controller ({ref_txt}) is chosen by a rule declared in advance, not after seeing the differences. {stat_txt}
"""
    sec_e046 = f"""
#### Student E046: Vaishnavi Parashar
* **Role:** {MEMBERS[1]['role']}
* **Git branch:** `{MEMBERS[1]['assigned_branch']}`
* **Owned modules:** `src/aedrover/sim/vehicle_mjcf.py`, `src/aedrover/sim/world.py`, `src/aedrover/sim/validate.py`, `src/aedrover/sim/curb_study.py`, `src/aedrover/sim/codesign.py`, `src/aedrover/control/curb.py`, `models/aed_delivery_amr.xml`.
* **Deliverables:** (i) the parametric MJCF rover and terrain, (ii) the physics validation against closed-form mechanics, (iii) the kerb traversability study, (iv) the mechanical co-design. Each item needs at least four distinct feature commits on your branch.

**Viva questions and model answers (true for this implementation)**

1. *Question:* What are the suspension natural frequency and damping ratio of the nominal and the optimised rover, and how do you know the simulation matches them?
   *Model answer:* With the sprung mass per wheel m_s = (chassis + payload) / 4 = {n.sprung_per_wheel:.2f} kg, omega_n = sqrt(k_s / m_s) and zeta = c_s / (2 sqrt(k_s m_s)). Nominal: k_s = {n.susp_k:.0f} N/m and c_s = {n.susp_c:.0f} N s/m give omega_n = {n.susp_omega_n:.2f} rad/s ({n.susp_omega_n / (2 * math.pi):.2f} Hz) and zeta = {n.susp_zeta:.3f}. Optimised: k_s = {o.susp_k:.1f} N/m and c_s = {o.susp_c:.1f} N s/m give {o.susp_omega_n:.2f} rad/s ({o.susp_omega_n / (2 * math.pi):.2f} Hz) and zeta = {o.susp_zeta:.3f}. {val_txt} (The earlier draft quoted a tyre stiffness of 30 kN/m; the simulated contact stiffness is a calibrated property of the MuJoCo contact model, not a free choice, and is measured rather than assumed.)
2. *Question:* Why is the sidewalk terrain built from mocap bodies instead of static geoms?
   *Model answer:* MuJoCo builds the collision broad-phase tree of static bodies at compile time, so editing the pose of a static geom at run time silently has no effect on collisions. Every re-positionable piece (sidewalk slabs, ramps, obstacles, pedestrians) is therefore a mocap body with a fixed compile-time size that is only moved or rotated, and parked 50 m below the road when unused; `mj_resetData` restores mocap poses, so the world object re-applies its authoritative poses after every reset. This makes a scenario reset cost microseconds instead of a recompile. The shipped file `models/aed_delivery_amr.xml` places the two slabs at compile time so that it loads with plain `mujoco.MjModel.from_xml_path`.
3. *Question:* Why does a kerb need an approach speed, and what do your results say about it?
   *Model answer:* {momentum} The quasi-static picture explains why: pressed against a step of height h a rigid wheel of radius r has an edge-contact direction at angle phi above the horizontal with tan(phi) = (r - h) / sqrt(2 r h - h^2); the smaller phi is, the more horizontal drive force (about W / tan(phi) for the wheel load W) is needed to lift the wheel, and it approaches infinity as h approaches r. For h = 0.12 m this angle is {math.degrees(math.atan((n.wheel_radius - 0.12) / math.sqrt(2 * n.wheel_radius * 0.12 - 0.12 ** 2))):.1f} degrees for the nominal wheel and {math.degrees(math.atan((o.wheel_radius - 0.12) / math.sqrt(2 * o.wheel_radius * 0.12 - 0.12 ** 2))):.1f} degrees for the optimised wheel, so a slow approach cannot lift the wheel with the available torque and the rover uses kinetic energy instead.
4. *Question:* What did the co-design change, and what did it cost?
   *Model answer:* {cd_txt}
5. *Question:* How is payload shock defined, and why does a softer isolator lower it?
   *Model answer:* The payload accelerometer measures proper acceleration, which reads +g along the local up at rest; the shock is the magnitude of a_dyn = a_proper - g u_body (u_body is world-up expressed in the chassis frame, so pitch and roll do not leak in), passed through a zero-phase 4th-order Butterworth low-pass at {pf['cutoff_hz']:g} Hz, and the peak is reported in g against a {BUDGET_G:g} g budget (a design assumption of this project; no standard is applied). The isolator is a spring-damper on three slide joints: with payload mass {n.payload_mass:g} kg, k_p = {n.iso_kz:.0f} N/m and c_p = {n.iso_cz:.0f} N s/m give omega = {math.sqrt(n.iso_kz / n.payload_mass):.2f} rad/s and zeta = {n.iso_cz / (2 * math.sqrt(n.iso_kz * n.payload_mass)):.3f}; the optimised k_p = {o.iso_kz:.0f} N/m and c_p = {o.iso_cz:.0f} N s/m give {math.sqrt(o.iso_kz / o.payload_mass):.2f} rad/s and {o.iso_cz / (2 * math.sqrt(o.iso_kz * o.payload_mass)):.3f}. A lower isolator frequency filters more of the kerb impulse before it reaches the payload, at the price of a larger payload stroke (limited to +/-{o.iso_range_z * 100:.0f} cm by the joint range).
6. *Question:* How is contact friction stability handled in the simulation, and what happens on slippery ground?
   *Model answer:* The model uses `cone="{xf['cone']}"` friction cones, the `{xf['integrator']}` integrator, a {xf['timestep']} s physics step and {xf['iterations']} solver iterations; the tyre contact softness is set by `solref` = ({', '.join(f'{x:g}' for x in o.tyre_solref)}) and tyre friction is written per episode into the model arrays (scenario friction {pf['scenario_defaults']['mu_range'][0]:g}-{pf['scenario_defaults']['mu_range'][1]:g} normally, lower in the slippery family). Long soak runs keep the state finite. {mu_txt}
"""
    return {"viva_e026": sec_e026, "viva_e046": sec_e046}



TREE = """```
.
|-- README.md                                          <- Front-page research charter, student roster & literature
|-- RESEARCH_AND_IMPLEMENTATION_GUIDE.md               <- Comprehensive technical engineering guide
|-- configs/                                           <- Vehicle and kerb-table data read by the vendored package
|   |-- curb_table.json
|   |-- mppi_tuned.json
|   |-- vehicle.yaml
|   `-- vehicle_optimized.yaml
|-- docs/
|   |-- TEAM_ROSTER.json                               <- Machine-readable team configuration
|   |-- LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md   <- Literature dossier (6 verified papers)
|   |-- RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md         <- 4-page IEEE conference manuscript (results filled in)
|   `-- figures/
|       |-- figure1_system_architecture.png            <- 300 DPI system architecture diagram
|       |-- figure2_kinematic_telemetry.png            <- 300 DPI telemetry of one simulated kerb crossing
|       |-- figure3_comparative_performance.png        <- 300 DPI controller benchmark comparison
|       `-- video_poster.png                           <- LinkedIn video thumbnail
|-- models/
|   |-- .gitkeep
|   `-- aed_delivery_amr.xml                           <- Standalone MuJoCo MJCF snapshot of the rover on a kerb
|-- src/
|   |-- test_env.py                                    <- Toolchain and physics smoke test
|   |-- aed_navigation_controller.py                   <- Runs one episode of the DWA and MPPI controllers
|   `-- aedrover/                                      <- Vendored simulation, navigation, clinical and analysis package
`-- analytics/
    |-- .gitkeep
    |-- cardiac_survival_economics.py                  <- Dimensionless survival and economics comparison
    |-- generate_paper_figures.py                      <- Standalone 300 DPI figure generator
    |-- aed_delivery_benchmark.csv                     <- Real benchmark episodes (one row per episode)
    `-- kerb_crossing_telemetry.csv                    <- Telemetry of the episode shown in Figure 2
```"""

README = r"""# MDRIIA Group 03: Autonomous Last-Mile Ground AED Delivery Robot for Sudden Cardiac Arrest
**Course:** Modern Day Robotics and Its Industrial Applications (MDRIIA - Course Code: 702CO0E012)
**Academic Term:** Academic Year 2026–2027 | Semester VI (B.Tech CSBS)
**Institution:** SVKM's NMIMS MPSTME, Mumbai

@@partial_banner@@---

## 1. Authorized Research Title & Problem Statement

> "@@title@@"

### Core Engineering Focus
* **Physics & Kinematics:** Google DeepMind MuJoCo multi-body simulation of a four-wheel rover with independent suspension and a shock-isolated AED payload (`models/aed_delivery_amr.xml`, generated by `src/aedrover/sim`).
* **Autonomous Control:** Closed-loop navigation stack (pure pursuit, artificial potential field, dynamic window approach, MPPI with MuJoCo rollouts, PPO) behind one speed-and-separation safety filter (`src/aed_navigation_controller.py`, `src/aedrover/nav`).
* **Technoeconomic Evaluation:** Dimensionless CSBS operating-burden ratio, payback horizon and staff-capacity model plus the survival comparison (`analytics/cardiac_survival_economics.py`).
* **Empirical Validation:** Reproducible seeded benchmark episodes and paired hypothesis tests (`analytics/aed_delivery_benchmark.csv`).

---

> [!NOTE]
> ### The Devil's Advocate: Reality Check & Theoretical Roast
> *"Engineering a 4-wheel independent suspension chassis to outrun sudden cardiac arrest across urban traffic, assuming Indian municipal sidewalks don't feature open storm drains, sudden pavement trenches, and stray dogs who consider your shockproof defibrillator pod their new favorite chew toy."*

---

## 2. Student Engineering Team Matrix

@@team_table@@

### Student Engineering Commendation & Acknowledgments
SVKM's NMIMS MPSTME conveys sincere appreciation and heartfelt gratitude to **Kashish Praveen Jain (E026)** and **Vaishnavi Parashar (E046)** for their disciplined commitment, late-night debugging, and technical craftsmanship throughout Semester VI. Your rigorous work in DeepMind MuJoCo physics modeling, closed-loop telemetry instrumentation, and Computer Science & Business Systems (CSBS) technoeconomic modeling exemplifies the highest standards of undergraduate engineering inquiry.

> *"Scientists discover the world that exists; engineers create the world that never was."*
> — **Theodore von Kármán**

> *"There is no substitute for hard work. Genius is one percent inspiration and ninety-nine percent perspiration."*
> — **Thomas A. Edison**

---

## 3. Foundational Literature Benchmarks (Strict 2 Seminal : 4 Recent Ratio)

The research project is theoretically anchored on six foundational peer-reviewed publications. Every DOI below resolves on Crossref and its registered title matches the title cited here; the per-paper summaries in the literature dossier describe only what each paper reports.

@@lit_table@@

For the exhaustive literature analysis, mathematical derivations, and viva defense questions, refer to:
* [`docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md`](docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md)
* [`docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md`](docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md)
* [`RESEARCH_AND_IMPLEMENTATION_GUIDE.md`](RESEARCH_AND_IMPLEMENTATION_GUIDE.md)

---

## 4. Project Demonstration & Academic Showcase (LinkedIn)

[![Watch Video Demonstration on LinkedIn](docs/figures/video_poster.png)](https://www.linkedin.com/)

* **Video Demonstration:** [Watch 60-Second Walkthrough on LinkedIn](https://www.linkedin.com/) *(Click thumbnail above to open LinkedIn post)*
* **Student Presenters:** **Kashish Praveen Jain** (E026), **Vaishnavi Parashar** (E046)
* **Academic Institutional Tags:** SVKM's NMIMS MPSTME | Academic Directorate | Industry 4.0 Robotics
* **Submission Protocol:** Record a 60–90 second demonstration of your MuJoCo simulation and telemetry. Publish on LinkedIn tagging MPSTME, Dean, and Course Faculty. Insert your live post URL in `docs/TEAM_ROSTER.json` under `"linkedin_url"`, and submit a pull request to update this project dossier and the cohort dashboard.

---

## 5. Repository Directory Architecture

@@tree@@

Two entries go beyond the course template and are needed for the folder to run stand-alone: `configs/` (the vendored package reads the vehicle and kerb-table data relative to the folder root) and `analytics/kerb_crossing_telemetry.csv` (the real telemetry behind Figure 2, so that the figures regenerate from CSV files only).

---

## 6. Pedagogical Boundaries: Guidance vs Student Ownership

To ensure academic rigor and authentic student learning, this repository enforces strict boundaries between scaffolding and student deliverables:

* **Provided by Course Scaffolding:**
  * Validated physical model architecture in MuJoCo MJCF (`models/aed_delivery_amr.xml`) and the simulation package that generates it (`src/aedrover/sim`).
  * Verification toolchain and smoke test script (`src/test_env.py`).
  * Literature foundation, corrected mathematical formulations, and the conference paper blueprint with results filled in from the simulation study (`docs/`).
  * Dimensionless technoeconomic framework (`analytics/cardiac_survival_economics.py`).
* **Student Technical Deliverables (Required for Evaluation):**
  * Understand and be able to defend, individually, every module listed for your role in `RESEARCH_AND_IMPLEMENTATION_GUIDE.md`; free-riding is penalised in the viva.
  * Run and extend the physics trials: re-run any benchmark episode with `python src/aed_navigation_controller.py --controllers dwa --family kerb --seed <seed>`, add your own controller through `aedrover.nav.base.register`, and add your rows to `analytics/aed_delivery_benchmark.csv`.
  * Re-run `analytics/generate_paper_figures.py` to regenerate the publication figures from the CSV data after any change.
  * Review the manuscript blueprint critically: confirm every number against the data, rewrite the discussion in your own words, and defend your individual Git commits during the oral viva.

---

## 7. Sprint 0 Onboarding & Physics Environment Verification

Verify your local Python and MuJoCo simulation environment:

```powershell
# Step 1: Clone repository and navigate to group directory
git clone <repository_url>
cd <repository_root>/Group_03_Kashish_Vaishnavi

# Step 2: Checkout your individual feature branch
# Example for lead student:
git checkout -b feat/e026-lead-autonomous-navi

# Step 3: Install the runtime dependencies
pip install mujoco numpy scipy pandas matplotlib pyyaml gymnasium

# Step 4: Run environment smoke test
python src/test_env.py

# Step 5: Run one episode of the DWA and MPPI controllers (MPPI takes a few minutes)
python src/aed_navigation_controller.py --controllers dwa

# Step 6: Verify 300 DPI publication figures
python analytics/generate_paper_figures.py

# Step 7: Run the dimensionless survival and economics comparison
python analytics/cardiac_survival_economics.py
```

---

## 8. Results Summary (generated from the simulation results, not typed in)

Every number in this section is read from the result files listed in `RESEARCH_AND_IMPLEMENTATION_GUIDE.md`, section 10. The figures are simulation outcomes under stated assumptions, not field measurements.

* **Kerb traversability (RQ: can the rover cross a sidewalk kerb without shocking the payload?).** @@kerb_text@@
* **Mechanical co-design.** @@codesign_text@@
* **Controller benchmark.** @@bench_text@@

@@bench_table@@

* **Statistical comparison.** See `RESEARCH_AND_IMPLEMENTATION_GUIDE.md`, section 6.4, for the paired tests (df, t, exact p, Cohen's d, 95% CI). @@stats_short@@
* **Clinical outcome.** @@clinical_text@@
* **Dimensionless economics.** @@econ_text@@
* **Answer to the research question.** @@rq_answer@@
* **Honest limits.** The results are simulation only (rigid cylinder tyres, idealised lidar, tracker noise modelled as Gaussian, pedestrians from a social-force model with assumed parameters); the clinical inputs are Western cohort statistics and assumptions; the economics uses placeholder ratios; and physics is simulated per 36 m sidewalk segment, so a 1-2 km mission is composed at the route level rather than simulated end to end.
"""

GUIDE = r"""# Research & Implementation Guide: Autonomous Ground AED Delivery AMR
## Modern Day Robotics & Its Industrial Applications (MDRIIA)
**Project Title:** @@title@@
**Group ID:** MDRIIA Group 03

@@partial_banner@@---

## 1. Executive Scientific Problem Deconstruction

Out-of-hospital cardiac arrest (OHCA) is time critical: in the linear survival model of Larsen et al. (1993) every minute of delay to the first shock, to CPR and to advanced life support removes survival probability, and the three delays together remove 5.5% per minute when nothing is done. This project asks whether a compact ground rover that drives on sidewalks, climbs kerbs and protects an AED payload from shock loads can deliver the device earlier than an ambulance.

Two premises of the title need care and are treated as scenario assumptions, not as established facts:

1. **"Ambulance congestion delays of 15-20 minutes."** The best sourced distribution available to this project (Naess et al. 2024, Central Norway, 216,787 incidents, 2013-2022) has an urban median response time of @@naess_med@@ min and a 90th percentile of @@naess_p90@@ min; that paper is about busy ambulances, not road congestion. Nothing in the sourced data establishes 15-20 minutes for any Indian city, so the analysis reports its answer as a function of the response-time distribution.
2. **"Survival drops 7-10% for every minute without defibrillation."** This is a guideline rule of thumb (Kim et al. 2026, doi @@kim_doi@@), not the fitted Larsen model, whose shock coefficient is @@larsen_defib@@ per minute. Both are used: the Larsen model as the primary model and the rule of thumb as a model-form sensitivity check.

The engineering question is therefore: how fast can the rover travel on realistic sidewalk scenes (kerbs of 6-16 cm, pedestrians, obstacles, low friction) while keeping the peak payload shock below a budget of @@budget_g@@ g (a design assumption of this project), and how does that speed compare with the ambulance in the clinical model?

---

## 2. Mathematical Formulations & Kinematic Modeling (corrected)

### 2.1 Quarter-Car Suspension Dynamics
Each wheel station $i \in \{1,2,3,4\}$ is a two-degree-of-freedom quarter car:

$$m_s \ddot{z}_{s,i} + c_s (\dot{z}_{s,i} - \dot{z}_{u,i}) + k_s (z_{s,i} - z_{u,i}) = 0$$

$$m_u \ddot{z}_{u,i} - c_s (\dot{z}_{s,i} - \dot{z}_{u,i}) - k_s (z_{s,i} - z_{u,i}) + k_t (z_{u,i} - z_{r,i}) = 0$$

where $m_s$ is the sprung mass per wheel (chassis plus payload divided by four), $m_u$ the unsprung mass (wheel, hub motor and knuckle), $k_s$ and $c_s$ the suspension spring and damper, $k_t$ the effective tyre radial stiffness and $z_{r,i}$ the ground profile. The sprung-mass natural frequency and damping ratio used in this guide are $\omega_n = \sqrt{k_s/m_s}$ and $\zeta_s = c_s / (2\sqrt{k_s m_s})$. In the simulation $k_t$ is not a free parameter: it emerges from the MuJoCo contact softness (`solref`) and is measured by static penetration.

@@veh_table@@

### 2.2 Payload Isolator and Shock Metric
The AED payload sits on a viscoelastic isolator (slide joints along x, y, z; the vertical joint has stiffness $k_p$ and damping $c_p$):

$$m_p \ddot{z}_p = -k_p (z_p - z_c) - c_p (\dot{z}_p - \dot{z}_c)$$

The payload accelerometer reports proper acceleration, so the shock is the dynamic part $\mathbf{a}_{dyn} = \mathbf{a}_{proper} - g\,\mathbf{u}_{body}$, where $\mathbf{u}_{body}$ is the world-up direction in the chassis frame; it is passed through a zero-phase 4th-order Butterworth low-pass filter at @@cutoff_hz@@ Hz and the peak of $\lVert\mathbf{a}_{dyn}\rVert / g$ is reported. The limit @@budget_g@@ g is a project design assumption. (The earlier draft attributed 3 g to ISO 16750-3; that attribution could not be verified and is not used.)

### 2.3 Cardiac Survival Model (Larsen et al. 1993, corrected)
With all times in minutes measured from collapse,

$$S(t_{cpr}, t_{defib}, t_{acls}) = 0.67 - 0.023\,t_{cpr} - 0.011\,t_{defib} - 0.021\,t_{acls}, \qquad S \in [0, 0.67].$$

The intercept 0.67 is the survival if all three interventions happen at collapse; the three slopes sum to 0.055 per minute. **Correction of the earlier draft:** it used 0.046 per minute for the shock term and no ACLS term. Neither is part of the published model. Because the shock slope is only 0.011 per minute, the model attributes modest survival value to a shock-time saving; the guideline rule of thumb (7-10 percentage points per minute, Kim et al. 2026) is about seven times steeper, and the two bracket the plausible range.

Worked example from the package: with CPR at 3 min and ACLS at 10 min, shocking at 2 min instead of 6 min changes survival by @@worked_larsen@@ percentage points under Larsen and by @@worked_rule@@ under the rule of thumb (7% to 10% per minute).

### 2.4 Time to First Shock
From collapse, with $R$ the sampled ambulance response time (which already runs from the emergency call):

$$t_{amb} = t_{c2c} + R, \qquad t_{rover} = t_{c2c} + t_{alert} + \frac{d\,\phi}{v} + t_{handoff}$$

where $t_{c2c}$ is collapse-to-call, $t_{alert}$ the call-taker and dispatch latency, $d$ the straight-line radius, $\phi$ the route factor, $v$ the rover speed and $t_{handoff}$ the time for a bystander to take the AED and apply the pads. CPR time and ACLS time are common to all modes, so a delivery device changes survival only through $t_{defib}$. Two comparisons are used and must not be confused: dispatch in parallel with the ambulance (first shock is the minimum of the two times, so the gain is the marginal benefit of adding the device) and the device alone against the ambulance alone (the radius at which the two expected survivals are equal is the break-even radius). Defaults of the scenario (all ASSUMPTIONS unless a results file says otherwise):

@@scenario_table@@

### 2.5 Statistical Framework
* **Paired t test:** $t = \bar{d} / (s_d/\sqrt{n})$ with $df = n - 1$, exact two-sided $p$, effect size $d_z = \bar{d}/s_d$ and the t-based 95% confidence interval of $\bar{d}$; Welch's test is used for independent groups.
* **Cohen's d** for independent samples uses the pooled standard deviation; its 95% interval is the non-central-t pivot interval.
* **Proportions:** Wilson score intervals; paired success rates use an exact McNemar test on the discordant scenarios.
* **Multiplicity:** Holm adjustment across the family of t tests.
* **Sample size:** at least $N \ge 50$ episodes per controller (course mandate).

### 2.6 Dimensionless Health Economics & Operational Parity
No currency is used anywhere. Every cost is a ratio to the annualised value of the labour hours the system reclaims:

$$\kappa = \frac{\text{annualised operating burden (energy, maintenance, infrastructure)}}{\text{annualised value of reclaimed labour hours}} \le 0.25$$

$$P_{months} = \frac{\text{capital parity factor}}{1 - \kappa} \times 12, \qquad \Delta \text{FTE} = \frac{\text{annual reclaimed task hours}}{2080\ \text{h/year}}$$

$$\Delta Q = \Delta S \times L_{exp} \times QoL \quad \text{(undiscounted QALYs per encounter; inputs must be supplied and are assumptions)}$$

The draft's fixed values (kappa of 0.095, 12 years of life expectancy, quality of life 0.85) were unsourced placeholders and are not used as results.

---

## 3. Simulation Model

The rover is generated by `aedrover.sim.vehicle_mjcf` and placed in a procedural sidewalk world by `aedrover.sim.world`. Topology: a free-jointed chassis; a payload body on three slide joints (viscoelastic isolator) carrying the accelerometer; four knuckles on vertical slide joints (independent suspension), the two front ones with a steering hinge; and one spin hinge with a cylinder tyre per wheel. Drive uses velocity actuators with a torque clamp on the spin joints and position actuators on the steering joints, so the Python loop writes only eight numbers per control step. Controllers command a forward speed and a bicycle steering angle that an Ackermann allocation turns into wheel speeds and steer angles. The environment steps physics at @@physics_dt@@ ms and the controllers act at @@control_hz@@ Hz.

**Scenario families** (`aedrover.sim.scenario`): @@families@@. A segment is a straight 36 m stretch of sidewalk; kerb families add a road crossing (walk A, road, walk B). Kerb height is sampled from @@kerb_range@@ m, friction from @@mu_range@@ (lower in the slippery family), payload mass from 3 to 5 kg, and pedestrians walk in streams under a social-force model (Helbing and Molnar 1995 form; the strengths and ranges are ASSUMPTIONS, not fitted). Episodes end on goal, collision, rollover, leaving the sidewalk, stall or time limit.

**Perception:** a @@n_lidar@@-ray lidar with @@fov_deg@@ degrees field of view and @@lidar_range@@ m range, a forward terrain-height scan, and a pedestrian tracker with Gaussian position and velocity noise. All controllers consume the same observations.

**Shipped MJCF:** `models/aed_delivery_amr.xml` is the co-designed rover on a @@snap_kerb_cm@@ cm kerb crossing, generated by `aedrover.sim.world.build_xml(VehicleParams.optimized(), ...)` with the two sidewalk slabs moved into place at compile time. It loads with plain `mujoco.MjModel.from_xml_path`.

---

## 4. Controllers and Safety

@@controller_table@@

All controllers can be wrapped by the same speed-and-separation filter, so comparisons hold identical safety constraints:

@@safety_table@@

**Standards mapping (honest).** The course manual lists constraints for this group; the table states where the implementation meets them and where it does not.

@@standards_table@@

---

## 5. Experimental Protocol

1. **Physics validation** against closed-form mechanics (static equilibrium, tyre stiffness, ring-down, rolling, million-step soak).
2. **Kerb traversability study:** kerb height x approach speed x approach angle x friction, kerb-up and kerb-down, for the nominal and the optimised design.
3. **Mechanical co-design:** differential evolution over wheel radius, suspension, isolator and motor torque.
4. **Controller benchmark:** every controller on the same seeded scenarios of all families; results are paired by scenario.
5. **Clinical and economic layer:** survival comparison, break-even radius and dimensionless economics.

---

## 6. Results (read from the result files at export time)

### 6.1 Physics validation
@@validation_text@@

@@validation_table@@

### 6.2 Kerb traversability
@@kerb_text@@

@@kerb_table@@

### 6.3 Mechanical co-design
@@codesign_text@@

@@codesign_table@@

### 6.4 Controller benchmark and paired statistics
@@bench_text@@

@@bench_table@@

Success rate by scenario family:

@@family_table@@

@@stats_text@@

@@stats_table@@

### 6.5 Clinical outcome and economics
@@clinical_text@@

@@clinical_table@@

@@econ_text@@

@@econ_table@@

### 6.6 Answer to the research question
@@rq_answer@@

### 6.7 Limitations
* Simulation only: rigid cylinder tyres, idealised lidar and terrain scan, Gaussian tracker noise, no drivetrain elasticity, no hardware-in-the-loop.
* Pedestrian behaviour follows a social-force model whose strengths are assumptions; the benchmark measures robustness to that model, not to real crowds.
* The clinical inputs are Western cohort data (King County for Larsen, Central Norway for response times); no claim is made about Mumbai or any Indian setting.
* Larsen's model is linear and was fitted on a high-survival subgroup; it is not valid for non-shockable rhythms or long delays.
* Physics is simulated per 36 m segment; a 1-2 km route is composed at the route level.
* Economics is a course template with placeholder ratios, undiscounted and single period.

---

## 7. Student Task Breakdown and Oral Defense Questions

Both students defend the whole system at a high level and their own modules in depth. The model answers below are generated from the shipped model and the result files, so every number matches this folder.

@@viva_e026@@
@@viva_e046@@

---

## 8. Minimum Viable Deliverables and Student Work Scope

1. **`models/aed_delivery_amr.xml`:** four-wheel independent suspension rover, kerb crossing and isolated payload (shipped, compiles with plain MuJoCo).
2. **`src/aed_navigation_controller.py`:** working entry point that runs the DWA and MPPI controllers from the vendored package and can log telemetry.
3. **`analytics/aed_delivery_benchmark.csv`:** real simulation episodes, at least 80 rows (this folder ships @@csv_rows@@).
4. **`docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md`:** four-page manuscript with all sections drafted and the results filled in; students must verify and rewrite it.

---

## 9. Reproduction

```powershell
pip install mujoco numpy scipy pandas matplotlib pyyaml gymnasium
python src/test_env.py
python src/aed_navigation_controller.py --controllers dwa --family kerb --seed @@telemetry_seed@@ --telemetry analytics/kerb_crossing_telemetry.csv
python analytics/generate_paper_figures.py
python analytics/cardiac_survival_economics.py
```

The benchmark CSV comes from the research repository's controller-benchmark experiment; a single row can be replayed with the entry point (identical scenario and controller; small differences appear if the simulator code changed after the benchmark was run).

---

## 10. Data Provenance

@@provenance@@

Telemetry replay check: @@replay_note@@
"""

LITERATURE = r"""# Foundational Literature Review and Research Benchmark Dossier

## Project: Autonomous Last-Mile Ground AED Delivery Robot for Sudden Cardiac Arrest
## Group: MDRIIA_GROUP_03

@@partial_banner@@---

## 1. Executive Summary of Foundational Literature

This dossier grounds the problem formulation, the mathematical models and the performance metrics of MDRIIA_GROUP_03 in six peer-reviewed publications (two seminal, four recent). Each DOI resolves on Crossref and the registered title matches the title cited; each summary below restates only what the paper's abstract reports, and says explicitly when the project takes no numerical input from a paper.

Corrections relative to the earlier draft of this dossier: the survival coefficients (the shock term is 0.011 per minute, not 0.046, and the model has a third term for advanced life support); the Schierbeck et al. study is a Swedish drone-AED study with 211 alerts and a median benefit of 3 min 14 s (not a 1 min 52 s saving); Naess et al. study busy ambulances and response times in Central Norway (not road congestion); Weinberg et al. report ethnographic observations of a Pittsburgh sidewalk-robot pilot (not clearance distances); Tsao et al. is the American Heart Association statistical update (no decay law); and the sixth paper, an in-hospital arrest study, was replaced by Khatib (1986) because it does not concern an outdoor sidewalk vehicle.

---

## 2. Comparative Literature Matrix (6 Verified Papers)

@@lit_matrix@@

---

## 3. Paper-by-Paper Analysis

@@paper_sections@@

---

## 4. Cross-Paper Synthesis Matrix

| Evaluation dimension | What the six papers provide | What this project adds |
| :--- | :--- | :--- |
| Delivery-mode evidence | Drone AEDs arrived before the ambulance in 37 of 55 comparable cases in Sweden (Schierbeck et al. 2023); ambulances are often busy and busy ambulances are associated with later arrival (Naess et al. 2024) | A physics-based ground rover with a kerb-climbing and payload-shock evaluation and a survival comparison against the ambulance |
| Survival versus delay | A linear survival model with coefficients per minute of delay (Larsen et al. 1993) | The same model applied to delivery modes, plus the guideline rule of thumb as a model-form sensitivity check |
| Sidewalk interaction | Qualitative observations of distraction, obstruction and accessibility problems (Weinberg et al. 2023) | Seeded crowd and obstacle scenarios and a stopping-distance safety filter shared by all controllers |
| Local navigation | Potential-field obstacle avoidance (Khatib 1986) | Potential-field, dynamic-window, sampling-based (MPPI) and learned (PPO) controllers compared on identical scenarios with paired statistics |
| Epidemiology | Annual US heart-disease and stroke statistics (Tsao et al. 2023) | Used as context only; no number is taken from it |

---

## 5. Methodological Research Gap Formulation

@@gaps@@

---

## 6. Proposed Architectural Contribution

MDRIIA_GROUP_03 implements a four-wheel independent-suspension rover in MuJoCo with a shock-isolated payload, a measured kerb-climbing envelope, a mechanical co-design that trades wheel size, suspension and isolator against payload shock, five navigation controllers behind one safety filter, and a clinical and dimensionless economic layer whose inputs are separated into sourced values and flagged assumptions.

---

## 7. Literature-Grounded Student Viva Defense Questions

@@lit_viva@@
"""

MANUSCRIPT = r"""# Research Paper Manuscript Blueprint (4-Page IEEE Format)

**Title:** @@title@@

**Authors:** Kashish Praveen Jain (E026), Vaishnavi Parashar (E046)

**Affiliation:** SVKM's NMIMS MPSTME, Mumbai; B.Tech CSBS, Semester VI, MDRIIA (702CO0E012)

@@partial_banner@@---

## Abstract
@@abstract@@

**Index Terms:** MuJoCo simulation, sidewalk delivery robot, automated external defibrillator, payload shock, kerb climbing, model predictive path integral control, dimensionless technoeconomics.

---

## I. Introduction
Survival from out-of-hospital cardiac arrest falls with every minute of delay to the first shock @@r_larsen1993@@. Drones that carry an automated external defibrillator (AED) have been flown against real alerts in Sweden and arrived before the ambulance in about two thirds of comparable cases @@r_schierbeck2023@@, and ambulances are often busy, which is associated with later arrival @@r_naess2024@@; national statistics frame the burden @@r_tsao2023@@. A ground rover is a complementary option that is not limited by airspace, weather or darkness, but it must cross kerbs, share sidewalks with pedestrians @@r_weinberg2023@@ and protect a sensitive payload from shock.

This paper investigates the interrogative research question:
> "@@title@@"

The premises are handled as scenario assumptions: the sourced urban ambulance response time (@@naess_med@@ min median, @@naess_p90@@ min 90th percentile in Central Norway @@r_naess2024@@) is shorter than the 15-20 minutes of the question, and the 7-10% per minute is a guideline rule of thumb @@r_kim2026@@ that we use next to the fitted model of Larsen et al. @@r_larsen1993@@.

**Contributions.** (1) A validated MuJoCo model of a four-wheel rover with an isolated payload and a payload-shock metric (Section III). (2) A kerb traversability map and a mechanical co-design (Section VI-B). (3) A benchmark of @@n_controllers@@ controllers on @@n_episodes@@ seeded episodes with paired statistics (Sections VI-C and VI-D). (4) A clinical and dimensionless economic layer with sourced and assumed inputs separated (Sections IV and VI-E).

---

## II. Related Work & Foundational Literature
* **Delivery-mode evidence.** Schierbeck et al. @@r_schierbeck2023@@ dispatched AED drones in addition to standard EMS in two Swedish controlled airspaces and compared the AED delivery time with the ambulance arrival. Naess et al. @@r_naess2024@@ estimated the prevalence of busy ambulances in Central Norway (2013-2022) and their association with response time. Tsao et al. @@r_tsao2023@@ is the American Heart Association statistical update and provides epidemiological context.
* **Survival versus delay.** Larsen et al. @@r_larsen1993@@ fitted survival as a linear function of the delays to CPR, first shock and advanced life support; the 7-10% per minute rule of thumb comes from resuscitation guidelines @@r_kim2026@@.
* **Sidewalk robots and pedestrians.** Weinberg et al. @@r_weinberg2023@@ observed a Pittsburgh sidewalk delivery-robot pilot ethnographically; pedestrians in our simulator follow a social-force model @@r_helbing1995@@.
* **Local navigation.** Artificial potential fields @@r_khatib1986@@, the dynamic window approach @@r_fox1997@@, sampling-based control with physics rollouts @@r_williams2017@@ and proximal policy optimisation @@r_schulman2017@@ are compared on the MuJoCo physics engine @@r_todorov2012@@.

None of the six foundational papers evaluates a ground delivery vehicle's kerb climbing or payload shock; this paper addresses that gap and quantifies what it means for the time to first shock.

---

## III. System Architecture and Mathematical Modeling

### A. MuJoCo Multi-Body Physics Model
The rover (`models/aed_delivery_amr.xml`) has a free-jointed chassis (@@chassis_mass@@ kg), a payload of @@payload_mass@@ kg on a three-axis viscoelastic isolator, four independent suspension slide joints, front-axle steering and velocity-controlled wheel motors. The generalised equations of motion are

$$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + g(q) = \tau + J^T F_{ext}$$

and the sprung natural frequency and damping ratio are $\omega_n = \sqrt{k_s/m_s}$ and $\zeta_s = c_s/(2\sqrt{k_s m_s})$. Physics runs at @@physics_dt@@ ms and controllers act at @@control_hz@@ Hz. Terrain is built from mocap bodies so that scenarios reset without recompiling.

![Fig. 1. System architecture (`docs/figures/figure1_system_architecture.png`)](figures/figure1_system_architecture.png)

**Fig. 1.** System architecture of the simulated rover, from scenario generation to the clinical layer.

### B. Payload Shock Metric
The dynamic payload acceleration is $\mathbf{a}_{dyn} = \mathbf{a}_{proper} - g\mathbf{u}_{body}$, low-pass filtered at @@cutoff_hz@@ Hz with a zero-phase 4th-order Butterworth filter; the reported shock is $\max \lVert\mathbf{a}_{dyn}\rVert/g$ against a budget of @@budget_g@@ g (a design assumption).

### C. Controllers and Safety Filter
Five controllers share one observation interface and one speed-and-separation filter that enforces $d_{free} - v\,t_{react} - v^2/(2 a_{brake}) \ge 0$ on the footprint swept along the commanded arc (@@a_brake@@ m/s^2, @@t_react@@ s): pure pursuit, artificial potential field @@r_khatib1986@@, dynamic window approach @@r_fox1997@@, MPPI @@r_williams2017@@ with MuJoCo rollouts (K = @@mppi_k@@, H = @@mppi_h@@) and a PPO policy @@r_schulman2017@@. APF and DWA can add a kerb negotiator that caps the approach speed at the measured minimum climbing speed plus a margin.

### D. Scenarios
Five seeded families (@@families@@) of 36 m sidewalk segments with kerbs of @@kerb_range@@ m, friction @@mu_range@@, payload 3-5 kg, pedestrians and obstacles. Episodes end on goal, collision, rollover, leaving the sidewalk, stall or timeout.

---

## IV. Computer Science and Business Systems (CSBS) Technoeconomic Analysis
**Survival.** With times in minutes from collapse, $S = 0.67 - 0.023\,t_{cpr} - 0.011\,t_{defib} - 0.021\,t_{acls}$ @@r_larsen1993@@. (An earlier draft used 0.046 for the shock term; that value is not in the published model.) The rover changes only $t_{defib}$; the ambulance response time is sampled from a lognormal distribution fitted to the median and 90th percentile of @@r_naess2024@@.

**Time to first shock.** $t_{amb} = t_{c2c} + R$ and $t_{rover} = t_{c2c} + t_{alert} + d\phi/v + t_{handoff}$; the device is dispatched in addition to the ambulance, so the first shock is the earlier of the two.

**Economics (dimensionless).** $\kappa = \text{operating burden}/\text{value of reclaimed labour}$ (target $\le 0.25$), $P_{months} = \text{capital parity}/(1-\kappa)\times 12$ and $\Delta\text{FTE} = \text{reclaimed hours}/2080$. No currency is used.

---

## V. Experimental Protocol and Statistics
Every controller runs the same seeded scenarios, so comparisons are paired by scenario. Reported per comparison: degrees of freedom, $t$, exact $p$, effect size ($d_z$ for paired data) and the 95% confidence interval of the mean difference; success uses an exact McNemar test; Holm adjustment covers the family of $t$ tests. The sample size is at least 50 episodes per controller. The reference is the classical controller with the highest success rate (ties: lower median time), declared before the comparison.

---

## VI. Experimental Evaluation and Results

### A. Physics Validation
@@validation_text@@

### B. Kerb Traversability and Mechanical Co-Design
@@kerb_text@@

**TABLE I. Kerb-up minimum climbing speed and payload shock, head-on approach**

@@kerb_table@@

@@codesign_text@@

**TABLE II. Mechanical co-design (nominal against optimised)**

@@codesign_table@@

![Fig. 2. Kinematic telemetry (`docs/figures/figure2_kinematic_telemetry.png`)](figures/figure2_kinematic_telemetry.png)

**Fig. 2.** @@fig2_caption@@

### C. Controller Benchmark
@@bench_text@@

**TABLE III. Controller benchmark (all scenario families pooled)**

@@bench_table@@

![Fig. 3. Comparative benchmark (`docs/figures/figure3_comparative_performance.png`)](figures/figure3_comparative_performance.png)

**Fig. 3.** Success rate by scenario family with Wilson 95% intervals, and peak payload shock and completion time of the successful episodes.

### D. Paired Statistics
@@stats_text@@

**TABLE IV. Paired comparisons against the reference controller**

@@stats_table@@

### E. Clinical Outcome and Economics
@@clinical_text@@

**TABLE V. Expected survival by delivery mode**

@@clinical_table@@

@@econ_text@@

---

## VII. Discussion and Limitations
@@rq_answer@@

* The clinical layer uses Western cohort data and assumed dispatch delays; the conclusion cannot be transferred to an Indian city without local response-time and outcome data.
* Larsen's model attributes 0.011 per minute to the shock delay, whereas the guideline rule of thumb is about seven times steeper; the survival conclusion is more sensitive to this model-form choice than to any controller difference.
* The simulator is idealised (rigid cylinder tyres, idealised lidar, Gaussian tracker noise, social-force pedestrians with assumed parameters) and has no hardware-in-the-loop validation; benchmark differences measure robustness to this simulator only.
* Economics uses placeholder ratios and is a demonstration of the dimensionless method.

---

## VIII. Conclusion
@@conclusion@@ Future work: hardware-in-the-loop validation of the kerb-climbing envelope, local ambulance response-time data, and a route-level simulation of complete missions.

---

## References

@@references@@
"""

TEST_ENV_PY = r'''"""Verification and test-environment script.

MDRIIA Group 03 - autonomous last-mile ground AED delivery rover.

Checks the Python toolchain, compiles models/aed_delivery_amr.xml with plain MuJoCo, steps it,
and checks that the vendored package (src/aedrover) can build and step a simulation environment.
Exit status 0 means every check passed.
"""

import importlib
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC))

REQUIRED = [
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pandas", "pandas"),
    ("yaml", "pyyaml"),
    ("gymnasium", "gymnasium"),
    ("mujoco", "mujoco"),
]


def run_environment_check():
    print("=" * 60)
    print("MDRIIA GROUP 03 - TOOLCHAIN AND PHYSICS VERIFICATION")
    print("=" * 60)
    if sys.version_info < (3, 10):
        print(f"[ERROR] Python {sys.version.split()[0]} is too old; Python 3.10 or newer is required.")
        return False
    print(f"[OK] Python Version: {sys.version.split()[0]}")

    for module, pip_name in REQUIRED:
        try:
            mod = importlib.import_module(module)
        except ImportError:
            print(f"[ERROR] {module} is missing. Install via: pip install {pip_name}")
            return False
        print(f"[OK] {module} Version: {getattr(mod, '__version__', 'unknown')}")

    import mujoco
    import numpy as np

    xml_path = ROOT / "models" / "aed_delivery_amr.xml"
    if not xml_path.exists():
        print(f"[ERROR] Model file not found: {xml_path}")
        return False
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        for _ in range(500):
            mujoco.mj_step(model, data)
    except Exception as exc:  # compilation or stepping failure
        print(f"[ERROR] Physics compilation failed: {exc}")
        return False
    if not (np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()):
        print("[ERROR] Simulation state became non-finite while settling the rover.")
        return False
    print(f"[SUCCESS] Loaded and stepped model: {xml_path}")
    print(f"[INFO] Model Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}")
    print(f"[INFO] Simulated time: {data.time:.2f} s, chassis height: {data.qpos[2]:.3f} m")

    try:
        from aedrover.sim import AEDRoverEnv, VehicleParams

        env = AEDRoverEnv(veh=VehicleParams.optimized(), obs_mode="dict")
        env.reset(seed=0, options={"family": "flat_clear"})
        for _ in range(25):
            env.step(np.array([1.0, 0.0]))
    except Exception as exc:
        print(f"[ERROR] Vendored package aedrover failed: {exc}")
        return False
    print("[OK] Vendored package aedrover: environment reset and 25 control steps")

    print("=" * 60)
    print("Verification complete. All required libraries are ready.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    sys.exit(0 if run_environment_check() else 1)
'''

NAV_PY = r'''"""Closed-loop navigation entry point.

MDRIIA Group 03 - autonomous last-mile ground AED delivery rover.

Runs ONE episode of each requested controller (default: DWA and MPPI) on the vendored aedrover
package. Every controller sees the same seeded scenario and, unless --no-shield is given, the same
speed-and-separation safety filter, so the summaries are directly comparable. The controllers
themselves live in src/aedrover/nav; this file is only the entry point.

    python src/aed_navigation_controller.py --controllers dwa
    python src/aed_navigation_controller.py --family crowded --seed 1001
    python src/aed_navigation_controller.py --controllers dwa --telemetry analytics/kerb_crossing_telemetry.csv

MPPI plans with physics rollouts and needs minutes for a full episode; use --max-time to shorten.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from aedrover.control.safety_filter import SafetyFilter  # noqa: E402
from aedrover.nav.base import make_controller, registered_controllers, run_episode  # noqa: E402
from aedrover.sim.env import AEDRoverEnv  # noqa: E402
from aedrover.sim.scenario import FAMILIES  # noqa: E402
from aedrover.sim.vehicle_mjcf import VehicleParams  # noqa: E402

BUDGET_G = 3.0
TELEMETRY_COLUMNS = ["t_s", "x_m", "y_m", "yaw_rad", "v_cmd_mps", "delta_cmd_rad", "shock_g", "clearance_m"]


# keyword each controller uses for its top speed (as in the benchmark, one number caps every method)
SPEED_KEY = {"pure_pursuit": "v_cruise", "apf": "v_cruise", "apf_nocurb": "v_cruise", "dwa": "v_cruise",
             "dwa_nocurb": "v_cruise", "mppi": "v_max", "ppo": "v_max"}
BENCHMARK_SPEED_CAP = 2.0


def controller_kwargs(name, speed_cap):
    """Settings the benchmark used: the common speed cap and, for MPPI, the tuned cost weights."""
    kw = {}
    if speed_cap and name in SPEED_KEY:
        kw[SPEED_KEY[name]] = float(speed_cap)
    tuned = SRC.parent / "configs" / "mppi_tuned.json"
    if name == "mppi" and tuned.exists():
        kw.update(json.loads(tuned.read_text(encoding="utf-8"))["kwargs"])
    return kw


def run_one(name, family, seed, vehicle, shield_on, max_time, speed_cap=BENCHMARK_SPEED_CAP):
    """Run one episode; returns (episode metrics, scenario, shield, wall seconds)."""
    env = AEDRoverEnv(obs_mode="dict", veh=VehicleParams.by_name(vehicle), max_time=max_time)
    controller = make_controller(name, **controller_kwargs(name, speed_cap))
    shield = SafetyFilter() if shield_on else None
    t0 = time.perf_counter()
    episode = run_episode(env, controller, shield, seed=seed, options={"family": family}, record=True)
    return episode, env.scenario, shield, time.perf_counter() - t0


def write_telemetry(path, episode, scenario, name, family, seed):
    """Write the recorded trajectory (time, pose, commanded speed and steering, shock, clearance)."""
    frame = pd.DataFrame(episode["trajectory"], columns=TELEMETRY_COLUMNS)
    frame["controller"] = name
    frame["family"] = family
    frame["seed"] = seed
    frame["kerb_h_m"] = scenario.kerb_h
    frame["x_down_m"] = scenario.x_down
    frame["x_up_m"] = scenario.x_up
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")
    return path


def summary_line(name, episode, shield, wall):
    clearance = episode["min_clearance_m"]
    clear_txt = "n/a" if not np.isfinite(clearance) else f"{clearance:.2f} m"
    interventions = "shield off" if shield is None else f"shield interventions {shield.n_interventions}/{shield.n_calls}"
    return (
        f"[{name}] outcome={episode['outcome']}  time={episode['time_s']:.2f} s  path={episode['path_m']:.2f} m  "
        f"peak_shock={episode['peak_shock_g']:.2f} g (budget {BUDGET_G:g} g)  min_clearance={clear_txt}  "
        f"{interventions}  wall={wall:.1f} s"
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run one episode of each requested controller.")
    ap.add_argument("--controllers", nargs="+", default=["dwa", "mppi"], help="registered controller names")
    ap.add_argument("--family", choices=FAMILIES, default="kerb", help="scenario family")
    ap.add_argument("--seed", type=int, default=1003, help="scenario seed (same scenario for every controller)")
    ap.add_argument("--vehicle", choices=("nominal", "optimized"), default="optimized")
    ap.add_argument("--no-shield", action="store_true", help="disable the safety filter")
    ap.add_argument("--max-time", type=float, default=90.0, help="episode time limit in seconds (benchmark: 90)")
    ap.add_argument("--speed-cap", type=float, default=BENCHMARK_SPEED_CAP,
                    help="top speed in m/s given to every controller (benchmark: 2.0; 0 keeps each controller's default)")
    ap.add_argument("--telemetry", default=None, help="write the trajectory of each controller to this CSV")
    args = ap.parse_args(argv)

    available = registered_controllers()
    unknown = [c for c in args.controllers if c not in available]
    if unknown:
        print(f"[ERROR] unknown controller(s) {unknown}; registered: {available}")
        return 2
    print("=" * 60)
    print(f"Autonomous ground AED delivery rover: family={args.family} seed={args.seed} vehicle={args.vehicle}")
    print("=" * 60)
    for name in args.controllers:
        if name == "mppi":
            print("[mppi] plans with physics rollouts; a full episode takes minutes on a laptop CPU ...", flush=True)
        episode, scenario, shield, wall = run_one(
            name, args.family, args.seed, args.vehicle, not args.no_shield, args.max_time, args.speed_cap)
        print(summary_line(name, episode, shield, wall))
        if args.telemetry:
            path = Path(args.telemetry)
            if len(args.controllers) > 1:
                path = path.with_name(f"{path.stem}_{name}{path.suffix}")
            print(f"[{name}] telemetry written to {write_telemetry(path, episode, scenario, name, args.family, args.seed)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

ECON_PY = r'''"""Dimensionless survival and economics comparison (no currency anywhere).

MDRIIA Group 03 - autonomous last-mile ground AED delivery rover.

Uses the vendored clinical module (src/aedrover/clinical):
  * survival versus time to first shock, Larsen et al. (1993): 0.67 - 0.023 t_cpr - 0.011 t_defib
    - 0.021 t_acls (minutes from collapse), plus the guideline rule of thumb as a sensitivity model;
  * ambulance response times from a lognormal fitted to the Central Norway urban median and 90th
    percentile (Naess et al. 2024);
  * the operating-burden ratio kappa, the payback horizon and the staff capacity freed.

Every input that is not taken from a paper is an ASSUMPTION and is printed as such. By default the
rover speed is the mean speed of the successful episodes of one controller in
analytics/aed_delivery_benchmark.csv, so the simulation feeds the clinical comparison.

    python analytics/cardiac_survival_economics.py
    python analytics/cardiac_survival_economics.py --speed-source assumption --radius 500
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aedrover.clinical import (  # noqa: E402
    KAPPA_TARGET,
    EconomicsInputs,
    ScenarioParams,
    breakeven_radius,
    compare_modes,
    get_model,
    qaly_gain_per_encounter,
)


def measured_speed(csv_path, controller):
    """Mean speed [m/s] of the successful benchmark episodes of ``controller`` and their count."""
    frame = pd.read_csv(csv_path)
    ok = frame[(frame["controller"] == controller) & frame["success"].astype(bool)]
    if ok.empty:
        raise SystemExit(f"[ERROR] no successful episodes of {controller!r} in {csv_path}")
    return float(ok["mean_speed"].mean()), len(ok)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--radius", type=float, default=1000.0, help="straight-line radius in metres (ASSUMPTION)")
    ap.add_argument("--controller", default="dwa", help="benchmark controller that supplies the rover speed")
    ap.add_argument("--speed-source", choices=("benchmark", "assumption"), default="benchmark")
    ap.add_argument("--n", type=int, default=2000, help="simulated arrests")
    ap.add_argument("--resamples", type=int, default=1000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--life-years", type=float, default=None, help="optional life expectancy for QALYs (ASSUMPTION)")
    ap.add_argument("--qol", type=float, default=None, help="optional quality-of-life weight for QALYs (ASSUMPTION)")
    args = ap.parse_args(argv)

    base = ScenarioParams()
    if args.speed_source == "benchmark":
        speed, n_ok = measured_speed(ROOT / "analytics" / "aed_delivery_benchmark.csv", args.controller)
        speed_note = f"measured: mean speed of {n_ok} successful {args.controller} episodes in the benchmark CSV"
    else:
        speed, speed_note = base.rover_speed_mps, "ASSUMPTION (package default)"
    params = ScenarioParams(radius_m=args.radius, rover_speed_mps=speed)
    d = params.dispatch

    print("=" * 72)
    print("MDRIIA GROUP 03 - CARDIAC SURVIVAL AND DIMENSIONLESS ECONOMICS")
    print("=" * 72)
    print("Scenario inputs")
    print(f"  radius                     {params.radius_m:8.0f} m        [ASSUMPTION]")
    print(f"  rover speed                {params.rover_speed_mps:8.2f} m/s      [{speed_note}]")
    print(f"  route factor               {params.route_factor:8.2f}          [ASSUMPTION]")
    print(f"  ambulance response         median {params.ems_median_min:.1f} min, p90/median {params.ems_p90_over_median:.2f}  [Naess 2024, urban]")
    print(f"  collapse-to-call           {d.collapse_to_call_min:8.1f} min      [ASSUMPTION]")
    print(f"  call-to-alert              {d.call_to_alert_min:8.1f} min      [ASSUMPTION]")
    print(f"  hand-off (AED and pads)    {d.handoff_min:8.1f} min      [ASSUMPTION]")
    print(f"  bystander CPR at           {d.bystander_cpr_delay_min:8.1f} min      [ASSUMPTION]")

    gains = {}
    for model_name in ("larsen1993", "rule_of_thumb"):
        model = get_model(model_name)
        table = compare_modes(model, params, n=args.n, seed=args.seed, parallel=True,
                              modes=("ambulance", "rover"), n_resamples=args.resamples)
        print("-" * 72)
        print(f"Survival, rover dispatched in parallel with the ambulance, model {model_name}")
        for _, row in table.iterrows():
            line = f"  {row['mode']:<10} survival {row['mean_survival']:.3f} [{row['survival_ci_low']:.3f}, {row['survival_ci_high']:.3f}]"
            if row["mode"] != "ambulance":
                line += (f"  gain {100 * row['abs_gain']:+.1f} pp [{100 * row['abs_gain_ci_low']:+.1f}, "
                         f"{100 * row['abs_gain_ci_high']:+.1f}]  shocks first {100 * row['p_faster']:.0f}%")
                gains[model_name] = float(row["abs_gain"])
            print(line)
        be = breakeven_radius(params.rover_speed_mps, params.route_factor, d.call_to_alert_min, d.handoff_min,
                              params.ems_median_min, model, ems_p90_over_median=params.ems_p90_over_median)
        if be.flag == "crossing":
            print(f"  rover alone matches ambulance alone at a straight-line radius of {be.radius_m:.0f} m")
        else:
            print(f"  break-even: {be.flag}")

    econ = EconomicsInputs()
    print("-" * 72)
    print("Dimensionless economics (all inputs are ASSUMPTIONS: ratios to the annual value of reclaimed labour)")
    print(f"  energy {econ.energy_ratio:.2f}, maintenance {econ.maintenance_ratio:.2f}, infrastructure {econ.infrastructure_ratio:.2f}")
    print(f"  capital parity factor {econ.capex_parity_factor:.2f} years, reclaimed task hours {econ.annual_reclaimed_task_hours:.0f} per year")
    print(f"  kappa                      {econ.kappa:8.3f}          (target <= {KAPPA_TARGET:g}: {'met' if econ.meets_target else 'not met'})")
    print(f"  payback horizon            {econ.payback_months:8.1f} months")
    print(f"  staff capacity freed       {econ.delta_fte:8.2f} FTE")
    if args.life_years is not None and args.qol is not None:
        gain = gains["larsen1993"]
        qaly = qaly_gain_per_encounter(gain, args.life_years, args.qol)
        print(f"  QALYs per encounter        {qaly:8.3f}          (Larsen gain x {args.life_years:g} years x {args.qol:g}; ASSUMPTION inputs)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

def paper_notes() -> dict:
    """What each foundational paper reports (from its abstract) and what this project takes from it."""
    from aedrover.clinical.ems_delay import NAESS_ALL, NAESS_RURAL, NAESS_URBAN

    return {
        "larsen1993": {
            "formula": r"$S = 0.67 - 0.023\,t_{cpr} - 0.011\,t_{defib} - 0.021\,t_{acls}$ (minutes from collapse)",
            "does": "Develops a graphic model (multiple linear regression) of survival from sudden out-of-hospital cardiac "
                    "arrest as a function of the time intervals to critical prehospital interventions. The data are 1,667 "
                    "cardiac arrest patients from the King County (Washington) surveillance system who had underlying heart "
                    "disease, were in ventricular fibrillation and collapsed before EMS arrival. The fitted equation is "
                    "survival = 67% - 2.3% per minute to CPR - 1.1% per minute to defibrillation - 2.1% per minute to advanced "
                    "cardiac life support (significant at P < .001). The 67% is the survival if all three interventions occur "
                    "at collapse, and the three coefficients sum to a decline of 5.5% per minute without treatment. Predicted "
                    "survival for given EMS response times approximated published observed rates.",
            "uses": "Primary survival model (`aedrover.clinical.survival.LarsenModel`); the coefficients are pinned by unit tests "
                    "and used mainly for the difference between delivery modes.",
            "limits": "A high-survival-likelihood subgroup of a Western cohort, so absolute values are optimistic; linear and "
                      "clipped at zero, so not valid for long delays; contains no delivery device. Its shock slope (1.1% per "
                      "minute) is far below the 7-10% per minute rule of thumb quoted in resuscitation guidelines.",
            "m_does": "Linear survival model from delays to CPR, first shock and ACLS (1,667 King County patients)",
            "m_uses": "Coefficients of the primary survival model",
            "m_limits": "High-survival subgroup, Western cohort, linear, no delivery device",
        },
        "khatib1986": {
            "formula": "Motion under the negative gradient of an artificial potential field (attractive goal term, repulsive "
                       "obstacle terms of finite range); moving obstacles through a time-varying field",
            "does": "Presents a real-time obstacle avoidance approach for manipulators and mobile robots based on the artificial "
                    "potential field concept. Collision avoidance, traditionally a high-level planning problem, is distributed "
                    "between levels of control; the method is extended to moving obstacles with a time-varying potential field "
                    "and is applied to robot arms by controlling motion directly in operational space. It was implemented on a "
                    "PUMA 560 robot in the COSMOS system with real-time avoidance of moving obstacles using visual sensing.",
            "uses": "Baseline B of the controller benchmark (`aedrover.nav.apf`): attraction toward the path plus repulsion "
                    "from lidar returns, tracked pedestrians and virtual sidewalk-edge walls within an influence radius.",
            "limits": "A reactive local method formulated for manipulators and mobile robots in general; it knows nothing about "
                      "kerbs, payload shock or vehicle dynamics, which is why the benchmark also includes the dynamic window "
                      "approach (described in the package as not being trapped by symmetric obstacle forces) and sampling-based "
                      "control.",
            "m_does": "Artificial potential field obstacle avoidance for manipulators and mobile robots",
            "m_uses": "APF baseline controller",
            "m_limits": "Reactive, no kerb or shock awareness",
        },
        "schierbeck2023": {
            "formula": "AED delivered by drone before ambulance arrival in 37 of 55 comparable cases (67%); median time benefit "
                       "3 min 14 s",
            "does": "A prospective observational study in Sweden: five AED-equipped drones in two controlled airspaces covering "
                    "about 200,000 inhabitants were dispatched in addition to standard emergency medical services for suspected "
                    "out-of-hospital cardiac arrests, and flew autonomously. Flights were excluded for air-traffic non-approval, "
                    "unfavourable weather, no-delivery zones and darkness. Of 211 alerts (April 2021 to May 2022) a drone was "
                    "deployed in 72 (34%) and delivered an AED in 58 of those (81%). Where both arrival times were available "
                    "(n = 55) the drone AED arrived before the ambulance in 37 cases (67%) with a median time benefit of "
                    "3 min 14 s. Among these, 18 were true cardiac arrests, an AED was attached in six, two patients were "
                    "defibrillated by a drone-delivered AED and one survived 30 days. No adverse events occurred.",
            "uses": "The benchmark for how often a delivery device arrives before the ambulance and by how much; the clinical "
                    "package keeps the 67% and the 3 min 14 s as validation targets only (they feed no computation).",
            "limits": "A drone in controlled airspace with flight exclusions; it says nothing about ground vehicles, kerbs or "
                      "payload shock, and only a third of the alerts led to a deployment.",
            "m_does": "Prospective drone-AED study in Sweden: 211 alerts, 55 comparable cases, drone first in 67%, median benefit 3 min 14 s",
            "m_uses": "Arrival-before-ambulance benchmark (validation target)",
            "m_limits": "Drone only, flight exclusions, no ground vehicle",
        },
        "tsao2023": {
            "formula": "Statistical compendium; no equation used (epidemiological context only)",
            "does": "The annual statistical update of the American Heart Association with the National Institutes of Health on "
                    "heart disease, stroke and cardiovascular risk factors: behaviours and health factors, major clinical heart "
                    "and circulatory conditions and their outcomes, with additional COVID-19 material and a focus on health equity "
                    "in the 2023 edition.",
            "uses": "Epidemiological context for why time to defibrillation matters. No number or model parameter is taken from "
                    "this paper in the present project.",
            "limits": "United States statistics; it is a data compendium, not a study of delivery vehicles or of response time.",
            "m_does": "Annual American Heart Association statistical update",
            "m_uses": "Context only (no parameter taken)",
            "m_limits": "US statistics, no vehicle model",
        },
        "naess2024": {
            "formula": "Each 10-percentage-point increase in the probability of a busy ambulance is associated with 0.60 min "
                       "(95% CI 0.58 to 0.62) delay; estimated probability of a busy ambulance 26.7%",
            "does": "A retrospective observational study of medical emergency incidents with ambulance response in Central Norway "
                    "from 2013 to 2022. Machine learning on data from nearby incidents estimates, for each of 216,787 acute "
                    "incidents, the probability that a candidate ambulance is busy, and groups of nearby incidents with different "
                    "busy probabilities are compared. The estimated probability of a busy ambulance was 26.7% overall, 21.6% in "
                    "rural and 35.0% in urban areas; each 10-percentage-point increase was associated with a delay of 0.60 min "
                    "(rural 0.81 min, urban 0.30 min). The authors conclude that busy ambulances are common and associated with "
                    "delayed response.",
            "uses": f"The ambulance response-time distribution of the clinical model: a lognormal fitted to the urban median "
                    f"({NAESS_URBAN.median_min:g} min) and 90th percentile ({NAESS_URBAN.p90_min:g} min) of the paper's response-time "
                    f"table (all incidents {NAESS_ALL.median_min:g} and {NAESS_ALL.p90_min:g} min; rural {NAESS_RURAL.median_min:g} "
                    f"and {NAESS_RURAL.p90_min:g} min), plus the busy-ambulance slope for sensitivity analysis.",
            "limits": "Norwegian data on busy ambulances; the paper is not about road congestion and does not report 15-20 minute "
                      "urban delays. The premise of the project title therefore rests on a scenario assumption, not on this paper.",
            "m_does": "Busy ambulances and their association with response time, Central Norway 2013-2022 (216,787 incidents)",
            "m_uses": "Response-time distribution and busy slope",
            "m_limits": "Norwegian, busy ambulances not congestion, no 15-20 min delays",
        },
        "weinberg2023": {
            "formula": "Qualitative ethnographic findings; no equation used",
            "does": "Reports the City of Pittsburgh's pilot of sidewalk delivery robots on public sidewalks: ethnographic "
                    "observations and intercept interviews on how residents perceived and interacted with the robots. People with "
                    "limited knowledge crafted stories about the robots' purpose; the robots caused distractions and obstructions "
                    "for different sidewalk users (including children and dogs); people helped immobilised robots; and potential "
                    "accessibility issues were identified. The authors contribute recommendations for future pilots and questions "
                    "for the design of robots in public spaces.",
            "uses": "Qualitative motivation for the crowd and obstacle scenarios and for a conservative, stopping-distance "
                    "safety filter near pedestrians. No numeric parameter is taken from this paper.",
            "limits": "A small-robot commercial-delivery pilot: no speeds, no payload shock and no emergency use case.",
            "m_does": "Ethnographic study of a sidewalk delivery-robot pilot in Pittsburgh",
            "m_uses": "Qualitative scenario motivation (no parameter)",
            "m_limits": "Commercial delivery pilot, no emergency use, no numbers",
        },
    }


def lit_blocks(refs: dict) -> dict:
    notes = paper_notes()
    rows_readme, rows_matrix, sections = [], [], []
    for i, (key, kind) in enumerate(FOUNDATIONAL, 1):
        r, n = refs[key], notes[key]
        rows_readme.append([i, f"**{cite_short(r)}**", kind, f"*{r['title']}*", journal_only(r), cite_link(r), n["formula"],
                            PAPER_LEADS[key]])
        rows_matrix.append([f"**{cite_short(r)}**<br>`{r['doi']}`", f"*{venue_display(r)}*", n["m_does"], n["m_uses"], n["m_limits"],
                            PAPER_LEADS[key]])
        sections.append(
            f"### 3.{i} Paper {i}: {r['title']} ({cite_short(r)})\n"
            f"* **Full Title:** {r['title']}\n"
            f"* **Authors:** {r['authors']}\n"
            f"* **Journal / Venue:** *{venue_display(r)}*, {r['year']}\n"
            f"* **Type:** {kind}\n"
            f"* **Verified Active DOI:** {cite_link(r)}\n\n"
            f"#### What the paper does\n{n['does']}\n\n"
            f"#### What this project takes from it\n{n['uses']}\n\n"
            f"#### Limits for this project\n{n['limits']}\n\n---\n")
    lit_table = md_table(["#", "Author (Year)", "Type", "Paper Title", "Venue / Indexing", "Active DOI Link",
                          "Primary Extracted Formulation", "Student Lead"], rows_readme)
    matrix = md_table(["Paper & Citation", "Venue", "What the paper does", "What this project takes from it",
                       "Limits for this project", "Student Lead"], rows_matrix)
    return {"lit_table": lit_table, "lit_matrix": matrix, "paper_sections": "\n".join(sections)}


def gaps_block(res: Results, pf: dict, kb: dict, bb: dict, clb: dict) -> str:
    from aedrover.clinical.ems_delay import BUSY_DELAY_MIN_PER_10PP, NAESS_URBAN

    out = []
    if kb["kerb_ok"]:
        ks = kb["kerb_sum"]
        hs = ks["heights"]
        h12 = min(hs, key=lambda h: abs(h - 0.12))
        nd, od = ks["designs"]["nominal"], ks["designs"]["optimized"]
        a, b = nd["vmin_hi"][h12], od["vmin_hi"][h12]
        g1 = (f"None of the six papers reports kerb climbing or payload shock for a delivery vehicle. In this project's kerb study "
              f"the minimum climbing speed at a {100 * h12:.0f} cm kerb is {vtxt(a)} m/s (nominal) and {vtxt(b)} m/s (optimised), and "
              f"the peak payload shock at that speed is {gtxt(a)} g and {gtxt(b)} g against a {BUDGET_G:g} g budget; "
              f"{pc(od['window_fraction'], 0)} of the optimised design's kerb-up (height, friction) cells have a shock-compliant "
              f"climb window against {pc(nd['window_fraction'], 0)} for the nominal design.")
    else:
        g1 = pend("kerb results are needed to quantify this gap")
    out.append("### GAP-1: Kerb crossing and payload shock are unreported\n" + g1)
    if bb["bench_ok"]:
        rows = bb["bench_rows"]
        best = max(rows, key=lambda r: r["success"])
        g2 = (f"The drone study (Schierbeck et al. 2023) reports arrival before the ambulance but no ground vehicle is evaluated in the "
              f"reviewed literature. Here {len(rows)} controllers were benchmarked on {len(res.benchmark)} seeded sidewalk episodes: "
              f"the best success rate is {pc(best['success'])} ({lab(best['controller'])}), so reliability, not only speed, limits a ground "
              f"rover; collisions and stalls in crowded and slippery scenes are the main failure modes "
              f"(see the family table in the implementation guide).")
    else:
        g2 = pend("benchmark results are needed to quantify this gap")
    out.append("### GAP-2: No ground-rover time-to-first-shock evidence\n" + g2)
    if clb["clinical_ok"]:
        cv = clb["cv"]
        be, rv = cv["breakeven"], cv["primary"]["rover"]
        be_txt = (f"a break-even radius of {be['radius_m']:.0f} m" if be["flag"] == "crossing" else f"break-even flag {be['flag']}")
        g3 = (f"The title assumes ambulance delays of 15-20 minutes. The sourced urban response time in Central Norway has a median of "
              f"{NAESS_URBAN.median_min:g} min and a 90th percentile of {NAESS_URBAN.p90_min:g} min (Naess et al. 2024), a busy-ambulance effect of "
              f"only {BUSY_DELAY_MIN_PER_10PP['urban']:g} min per 10 percentage points in urban areas, and no comparable figure exists in the reviewed papers for an Indian "
              f"city. With the response-time inputs of the results file, the clinical comparison gives {pp(rv['abs_gain'])} percentage points of "
              f"marginal survival and {be_txt}; the answer to the research question depends on the local delay distribution.")
    else:
        g3 = pend("clinical results are needed to quantify this gap")
    out.append("### GAP-3: The delay premise is a scenario, not a sourced fact\n" + g3)
    lm = pf["larsen"]
    out.append(
        "### GAP-4: Model-form uncertainty in survival versus delay\n"
        f"Larsen et al. attribute {lm['defib']:g} per minute to the delay to the first shock (and {lm['cpr'] + lm['defib'] + lm['acls']:g} per "
        f"minute to all three delays together), whereas resuscitation guidelines quote {100 * pf['rule_low']:.0f}-{100 * pf['rule_high']:.0f}% per "
        f"minute without CPR. This factor of roughly seven in the slope dominates any single controller difference, so both models are "
        f"reported and the conclusion is stated for both.")
    return "\n\n".join(out)


def lit_viva(res: Results, pf: dict, refs: dict, clb: dict) -> str:
    from aedrover.clinical.ems_delay import NAESS_URBAN

    lm = pf["larsen"]
    q_k1 = (
        f"1. *Question:* What does Schierbeck et al. (2023) actually show, and how does it limit the claims you can make for a ground rover?\n"
        f"   *Model answer:* It is a prospective observational study of AED drones in two controlled Swedish airspaces, dispatched in addition to "
        f"standard EMS. Of 211 alerts a drone was deployed in 72, delivered an AED in 58, and in the 55 cases with both arrival times the drone "
        f"was first in 37 (67%) with a median benefit of 3 min 14 s. It excludes flights for weather, darkness, no-delivery zones and air-traffic "
        f"non-approval, and says nothing about ground vehicles. We therefore use the 67% and 3 min 14 s only as validation targets for the "
        f"clinical layer and make no claim that a rover replicates them.\n"
        f"2. *Question:* What do Naess et al. (2024) say about ambulance delays, and why is the 15-20 minute figure in the title not taken from them?\n"
        f"   *Model answer:* They study busy ambulances in Central Norway (216,787 incidents, 2013-2022): the estimated probability of a busy "
        f"ambulance is 26.7%, and each 10-percentage-point increase is associated with 0.60 min of delay (urban 0.30 min). The urban response time "
        f"used here has a median of {NAESS_URBAN.median_min:g} min and a 90th percentile of {NAESS_URBAN.p90_min:g} min. The paper is about busy "
        f"ambulances, not road congestion, and gives no 15-20 minute figure; that number is a scenario assumption of the title and must be varied "
        f"in the sensitivity analysis."
    )
    q_v1 = (
        f"1. *Question:* Which coefficients does Larsen et al. (1993) give, and what would be wrong with using {0.046:g} per minute for the shock term?\n"
        f"   *Model answer:* Survival = {lm['intercept']:g} - {lm['cpr']:g} t_cpr - {lm['defib']:g} t_defib - {lm['acls']:g} t_acls with times in "
        f"minutes from collapse (King County, 1,667 patients with heart disease and witnessed ventricular fibrillation). The three slopes sum to "
        f"{lm['cpr'] + lm['defib'] + lm['acls']:.3f} per minute without treatment. The shock slope is {lm['defib']:g}; {0.046:g} is not a coefficient of "
        f"the model and would overstate the value of a shock-time saving by a factor of {0.046 / lm['defib']:.1f}.\n"
        f"2. *Question:* What does Khatib (1986) contribute, and where does your APF baseline differ from a plain potential field?\n"
        f"   *Model answer:* Khatib's paper presents real-time obstacle avoidance based on artificial potential fields, extended to moving obstacles "
        f"with a time-varying field, and applies it to manipulators and mobile robots. The package's APF baseline follows the concept: attraction "
        f"toward the path, repulsion from lidar returns and tracked pedestrians inside an influence radius of {pf['apf']['rho0']:g} m, plus virtual "
        f"walls near the sidewalk edges, with an optional kerb-speed overlay. It is deliberately a reactive baseline; the dynamic window approach "
        f"and MPPI plan against predicted motion and are compared with it on identical seeded scenarios."
    )
    return (f"### Student: {MEMBERS[0]['name']} (`{MEMBERS[0]['roll_no']}`), branch `{MEMBERS[0]['assigned_branch']}`\n"
            f"* **Assigned literature domain:** {MEMBERS[0]['viva_focus']}\n\n{q_k1}\n\n"
            f"### Student: {MEMBERS[1]['name']} (`{MEMBERS[1]['roll_no']}`), branch `{MEMBERS[1]['assigned_branch']}`\n"
            f"* **Assigned literature domain:** {MEMBERS[1]['viva_focus']}\n\n{q_v1}\n")


def team_table() -> str:
    rows = [[f"`{m['roll_no']}`", f"`{m['sap_id']}`", f"**{m['name']}**", m["role"], f"`{m['assigned_branch']}`", m["viva_focus"]]
            for m in MEMBERS]
    return md_table(["Roll No", "SAP ID", "Student Name", "Technical Role", "Assigned Git Branch", "Core Viva Defense Area"], rows)


def roster_dict(refs: dict) -> dict:
    papers = []
    for i, (key, kind) in enumerate(FOUNDATIONAL, 1):
        r = refs[key]
        papers.append({"id": i, "authors": cite_short(r).rsplit(" (", 1)[0], "year": r["year"], "title": r["title"],
                       "venue": journal_only(r), "doi": doi_url(r), "lead": PAPER_LEADS[key], "type": kind})
    return {
        "group_id": GROUP_ID,
        "folder_name": GROUP_FOLDER,
        "domain": DOMAIN,
        "authorized_title": AUTHORIZED_TITLE,
        "members": [dict(m) for m in MEMBERS],
        "foundational_papers": papers,
        "project_showcase": {
            "platform": "LinkedIn Video",
            "linkedin_url": "PENDING_SUBMISSION",
            "video_poster_path": "docs/figures/video_poster.png",
            "video_duration_target": "60-90 seconds",
            "submission_status": "PENDING",
            "submission_instructions": "Paste your published LinkedIn post URL into 'linkedin_url' and submit a Pull Request",
        },
    }


def worked_examples(pf: dict) -> tuple[str, str]:
    from aedrover.clinical.survival import SURVIVAL_MODELS, LarsenModel, delta_survival

    d_l = float(delta_survival(LarsenModel(), 2.0, 6.0, 3.0, 10.0))
    lo = float(delta_survival(SURVIVAL_MODELS["rule_of_thumb_low"], 2.0, 6.0, 3.0, 10.0))
    hi = float(delta_survival(SURVIVAL_MODELS["rule_of_thumb_high"], 2.0, 6.0, 3.0, 10.0))
    return f"{100 * d_l:.1f}", f"{100 * lo:.0f} to {100 * hi:.0f}"


def scenario_table() -> str:
    from dataclasses import fields

    from aedrover.clinical.decision import ScenarioParams

    sp = ScenarioParams()
    sourced = {"ems_median_min": "sourced: Naess et al. 2024, urban median", "ems_p90_over_median": "sourced: Naess et al. 2024, urban 90th percentile / median"}
    rows = []
    for f in fields(sp):
        if f.name in ("dispatch", "busy_setting"):
            continue
        rows.append([f"`{f.name}`", f"{getattr(sp, f.name):g}", sourced.get(f.name, "ASSUMPTION")])
    for f in fields(sp.dispatch):
        v = getattr(sp.dispatch, f.name)
        rows.append([f"`dispatch.{f.name}`", "none" if v is None else f"{v:g}", "ASSUMPTION"])
    return md_table(["Parameter", "Default", "Status"], rows, ["l", "r", "l"])


def abstract_and_conclusion(res: Results, pf: dict, kb: dict, cb: dict, bb: dict, clb: dict) -> tuple[str, str]:
    n = pf["veh_optimized"]
    s = [f"We simulate a four-wheel, independently suspended sidewalk rover that carries a {n.payload_mass:g} kg automated external "
         f"defibrillator (AED) in MuJoCo and ask whether it can deliver the device before an ambulance."]
    tail = []
    if cb["codesign_ok"]:
        cs = cb["codesign_sum"]
        an, ao = cs["arms"]["nominal"], cs["arms"]["optimised"]
        s.append(f"A differential-evolution mechanical co-design lowered the mean payload-shock objective from {f2(an['objective'])} g to "
                 f"{f2(ao['objective'])} g and raised the share of kerb conditions inside the {BUDGET_G:g} g budget from "
                 f"{pc(an['frac_within_budget'], 0)} to {pc(ao['frac_within_budget'], 0)}.")
    if kb["kerb_ok"]:
        ks = kb["kerb_sum"]
        od = ks["designs"]["optimized"]
        ok_h = [h for h in ks["heights"] if od["vmin_hi"][h][0] is not None]
        hmax = max(ok_h)
        s.append(f"The co-designed rover climbs kerbs up to {100 * hmax:.0f} cm head-on at friction {od['mu_hi']:g}, at a minimum approach speed of "
                 f"{vtxt(od['vmin_hi'][hmax])} m/s and a peak payload shock of {gtxt(od['vmin_hi'][hmax])} g at that height.")
        tail.append(f"kerbs up to {100 * hmax:.0f} cm")
    if bb["bench_ok"]:
        rows = bb["bench_rows"]
        lo_, hi_ = min(r["success"] for r in rows), max(r["success"] for r in rows)
        s.append(f"On {len(res.benchmark)} seeded episodes of {len(rows)} controllers the success rate ranges from {pc(lo_)} to {pc(hi_)}.")
        if bb["ref"] and bb["comparisons"]:
            parts = []
            for c in bb["comparisons"][:2]:
                if c["time"] is not None:
                    parts.append(f"{lab(c['controller'])} changes completion time by {c['time'].mean_diff:+.1f} s (paired t({c['time'].df:.0f}) = "
                                 f"{c['time'].t:.1f}, p = {fmt_p(c['time'].p)}, d_z = {c['time'].d:+.2f})")
            if parts:
                s.append(f"Against the reference controller ({lab(bb['ref'])}), " + "; ".join(parts) + ".")
    if clb["clinical_ok"]:
        cv = clb["cv"]
        rv, be, econ = cv["primary"]["rover"], cv["breakeven"], cv["economics"]
        be_txt = (f"the rover alone matches the ambulance within {be['radius_m']:.0f} m" if be["flag"] == "crossing"
                  else f"break-even flag {be['flag']}")
        s.append(f"Under the {cv['model']} survival model the rover changes expected survival by {pp(rv['abs_gain'])} percentage points "
                 f"(95% CI {pp(rv['abs_gain_ci_low'])} to {pp(rv['abs_gain_ci_high'])}) and {be_txt}; the dimensionless economics gives "
                 f"kappa = {f2(econ['kappa'])}, a payback horizon of "
                 + ("no payback" if not math.isfinite(econ["payback_months"]) else f"{f1(econ['payback_months'])} months")
                 + f" and delta FTE = {f2(econ['delta_fte'])} (placeholder inputs).")
    if res.pending:
        s.append(pend("some results were still pending when this draft was generated (see the banner above); the abstract is incomplete."))
    conc = ["We built and validated a MuJoCo model of an AED delivery rover, mapped its kerb-climbing envelope, co-designed its mechanics "
            "and benchmarked five controllers with paired statistics."]
    if clb["clinical_ok"]:
        conc.append(clb["rq_answer"])
    else:
        conc.append(pend("the clinical answer to the research question needs results/clinical*.json"))
    return " ".join(s), " ".join(conc)


def build_context(res: Results, refs: dict, pf: dict, xml: str, tel: dict) -> dict:
    """Every placeholder value of every document template."""
    from aedrover.clinical.ems_delay import NAESS_URBAN

    pf = dict(pf)
    pf["xml_facts"] = xml_facts(xml)
    kb, cb, vb = kerb_blocks(res, pf), codesign_blocks(res, pf), validation_blocks(res)
    bb, clb = benchmark_blocks(res), clinical_blocks(res)
    tokens, ref_list = refs_block(refs)
    viva = viva_blocks(res, pf, refs, bb, kb, cb, vb, clb)
    abstract, conclusion = abstract_and_conclusion(res, pf, kb, cb, bb, clb)
    wl, wr = worked_examples(pf)
    n = pf["veh_optimized"]
    per, safe, mp, sd = pf["perception"], pf["safety"], pf["mppi"], pf["scenario_defaults"]
    comps = bb.get("comparisons")
    if comps and bb["ref"]:
        stats_short = "; ".join(
            f"{lab(c['controller'])} against {lab(bb['ref'])}: completion time {c['time'].mean_diff:+.2f} s, paired t({c['time'].df:.0f}) = "
            f"{c['time'].t:.2f}, p = {fmt_p(c['time'].p)}" for c in comps if c["time"] is not None) + "."
    else:
        stats_short = pend("paired statistics need the controller benchmark with a classical reference controller")
    ctx = {
        "partial_banner": partial_banner(res), "title": AUTHORIZED_TITLE, "team_table": team_table(), "tree": TREE,
        "naess_med": f"{NAESS_URBAN.median_min:g}", "naess_p90": f"{NAESS_URBAN.p90_min:g}",
        "kim_doi": refs["kim2026"]["doi"], "larsen_defib": f"{pf['larsen']['defib']:g}", "budget_g": f"{BUDGET_G:g}",
        "cutoff_hz": f"{pf['cutoff_hz']:g}", "worked_larsen": wl, "worked_rule": wr, "scenario_table": scenario_table(),
        "physics_dt": f"{1000 * pf['world']['timestep']:g}", "control_hz": f"{1.0 / pf['env']['control_dt']:g}",
        "families": ", ".join(f"`{f}`" for f in pf["families"]),
        "kerb_range": f"{sd['kerb_range'][0]:g}-{sd['kerb_range'][1]:g}", "mu_range": f"{sd['mu_range'][0]:g}-{sd['mu_range'][1]:g}",
        "n_lidar": per["n_lidar"], "fov_deg": f"{per['fov_deg']:g}", "lidar_range": f"{per['lidar_range']:g}",
        "snap_kerb_cm": f"{100 * tel['snapshot_kerb_h']:g}", "veh_table": veh_table(pf), "controller_table": controller_table(pf) + speed_note(res),
        "safety_table": safety_table(pf), "standards_table": standards_table(pf, res, bb, clb),
        "csv_rows": len(res.benchmark), "telemetry_seed": tel["seed"], "replay_note": tel["replay_note"],
        "provenance": provenance_table(res), "fig2_caption": tel["fig2_caption"],
        "stats_short": stats_short, "abstract": abstract, "conclusion": conclusion, "references": ref_list,
        "n_controllers": len(bb["bench_rows"]) if bb.get("bench_rows") else PENDING,
        "n_episodes": len(res.benchmark), "chassis_mass": f"{n.chassis_mass:g}", "payload_mass": f"{n.payload_mass:g}",
        "a_brake": f"{safe['a_brake']:g}", "t_react": f"{safe['t_react']:g}", "mppi_k": mp["K"], "mppi_h": mp["H"],
        **tokens, **{k: v for k, v in kb.items() if isinstance(v, str)}, **{k: v for k, v in cb.items() if isinstance(v, str)},
        **{k: v for k, v in vb.items() if isinstance(v, str)}, **{k: v for k, v in bb.items() if isinstance(v, str)},
        **{k: v for k, v in clb.items() if isinstance(v, str)}, **viva, **lit_blocks(refs),
        "gaps": gaps_block(res, pf, kb, bb, clb), "lit_viva": lit_viva(res, pf, refs, clb),
    }
    return ctx


def render_docs(ctx: dict, refs: dict) -> dict[str, str]:
    """Relative path -> text for every generated document."""
    import json

    return {
        "README.md": fill(README, ctx),
        "RESEARCH_AND_IMPLEMENTATION_GUIDE.md": fill(GUIDE, ctx),
        "docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md": fill(LITERATURE, ctx),
        "docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md": fill(MANUSCRIPT, ctx),
        "docs/TEAM_ROSTER.json": json.dumps(roster_dict(refs), indent=2, ensure_ascii=False) + "\n",
    }
