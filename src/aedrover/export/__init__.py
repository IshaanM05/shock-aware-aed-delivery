"""NMIMS course-template export: constants, results loading, statistics and package facts.

This package builds the course folder ``Group_03_Kashish_Vaishnavi`` (see ``scripts/export_nmims.py``
and ``docs/NMIMS_EXPORT.md``). Design rules:

* Every number in the exported text is read from a file under ``results/`` at export time, or from
  the package constants that define the model (``package_facts``). Nothing is typed in by hand.
* A missing results file or key raises :class:`MissingResult` with an actionable message. With
  ``allow_partial`` the gap is recorded in ``Results.pending`` instead and the templates emit a
  clearly labelled ``PENDING`` marker.
* Citations come only from ``docs/references.json`` (Crossref/DataCite verified).

Contents: ``load_results`` (I/O and validation), ``benchmark_summary`` / ``paired_comparisons`` /
``kerb_summary`` / ``codesign_summary`` / ``clinical_view`` (derived numbers), ``build_snapshot_xml``
(standalone MJCF), ``package_facts`` (constants read from the live package).
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PENDING = "PENDING"
GROUP_FOLDER = "Group_03_Kashish_Vaishnavi"
GROUP_ID = "MDRIIA_GROUP_03"
DOMAIN = "Autonomous Last-Mile Ground AED Delivery Robot for Sudden Cardiac Arrest"
AUTHORIZED_TITLE = (
    "Can an autonomous last-mile ground AED delivery vehicle simulated in MuJoCo reduce "
    "time-to-first-shock below urban ambulance congestion delays (15-20 minutes), given that "
    "sudden cardiac arrest survival drops 7-10% for every minute without defibrillation?"
)

# Members exactly as in the course draft (roles, branches and viva focus unchanged).
MEMBERS: tuple[dict, ...] = (
    {
        "roll_no": "E026",
        "sap_id": "70362400060",
        "name": "Kashish Praveen Jain",
        "role": "Lead Autonomous Navigation & Traffic Congestion Modeling Specialist",
        "assigned_branch": "feat/e026-lead-autonomous-navi",
        "viva_focus": "Sidewalk navigation dynamics, pedestrian crowd evasion, dynamic routing through "
                      "urban choke points, and arrival latency budgets.",
        "github_handle": "",
        "institutional_email": "",
    },
    {
        "roll_no": "E046",
        "sap_id": "70362400074",
        "name": "Vaishnavi Parashar",
        "role": "MuJoCo Dynamic Chassis Modeler & Path Optimization Engineer",
        "assigned_branch": "feat/e046-mujoco-dynamic-chass",
        "viva_focus": "Four-wheel independent suspension, curb-climbing dynamics, shock isolation for "
                      "biphasic AED pads, and contact friction stability.",
        "github_handle": "",
        "institutional_email": "",
    },
)

# The six foundational papers: 2 seminal : 4 recent. Keys into docs/references.json.
FOUNDATIONAL: tuple[tuple[str, str], ...] = (
    ("schierbeck2023", "Recent"),
    ("tsao2023", "Recent"),
    ("naess2024", "Recent"),
    ("weinberg2023", "Recent"),
    ("larsen1993", "Seminal"),
    ("khatib1986", "Seminal"),
)
_BOTH = "Kashish Praveen Jain (E026) & Vaishnavi Parashar (E046)"
_K = "Kashish Praveen Jain (E026)"
_V_K = "Vaishnavi Parashar (E046) & Kashish Praveen Jain (E026)"
PAPER_LEADS = {
    "schierbeck2023": _BOTH,
    "tsao2023": _BOTH,
    "naess2024": _K,
    "weinberg2023": _BOTH,
    "larsen1993": _V_K,
    "khatib1986": _K,
}
# Additional verified references that the guide and manuscript cite for methods.
METHOD_REFS: tuple[str, ...] = ("todorov2012", "fox1997", "williams2017", "schulman2017",
                                "helbing1995", "kim2026")
JOURNAL_NAMES = {
    "Lancet Digit Health": "The Lancet Digital Health",
    "Circulation": "Circulation",
    "PLOS ONE": "PLOS ONE",
    "Multimodal Technol Interact": "Multimodal Technologies and Interaction",
    "Ann Emerg Med": "Annals of Emergency Medicine",
    "Int J Robot Res": "The International Journal of Robotics Research",
    "Resuscitation": "Resuscitation",
    "IROS": "IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)",
    "ICRA": "IEEE International Conference on Robotics and Automation (ICRA)",
    "IEEE Robot Autom Mag": "IEEE Robotics & Automation Magazine",
    "Phys Rev E": "Physical Review E",
    "Clin Exp Emerg Med": "Clinical and Experimental Emergency Medicine",
    "arXiv": "arXiv preprint",
}

REQUIRED_CONTROLLERS = ("dwa", "mppi", "ppo")
CLASSICAL_CONTROLLERS = ("pure_pursuit", "apf_nocurb", "apf", "dwa_nocurb", "dwa")
MIN_EPISODES_PER_CONTROLLER = 50      # course sample-size mandate (N >= 50)
MIN_CSV_ROWS = 80
BUDGET_G = 3.0

BENCH_REQUIRED = ("outcome", "success", "time_s", "path_m", "peak_shock_g", "shock_over_budget",
                  "min_clearance_m", "controller", "family", "seed", "sc_kerb_h", "sc_x_down",
                  "sc_x_up", "vehicle")
KERB_REQUIRED = ("direction", "kerb_h", "speed", "angle_deg", "wheel_radius", "mu", "outcome",
                 "success", "peak_g")
WINDOW_REQUIRED = ("direction", "kerb_h", "wheel_radius", "mu", "v_min_success",
                   "v_max_within_budget", "feasible_window")
CLINICAL_MODE_COLUMNS = ("mode", "model", "n", "mean_survival", "survival_ci_low", "survival_ci_high",
                         "abs_gain", "abs_gain_ci_low", "abs_gain_ci_high", "p_faster",
                         "median_saving_when_faster_min")
ECONOMICS_KEYS = ("kappa", "payback_months", "delta_fte", "meets_kappa_target")

CLINICAL_SCHEMA_HELP = (
    "expected results/clinical*.json = {\n"
    '  "model": "larsen1993",                          # primary survival model (optional)\n'
    '  "parallel": true,                               # device dispatched with the ambulance (optional)\n'
    '  "scenario": {...},                              # ScenarioParams as a dict (optional)\n'
    '  "modes": [ one dict per (mode, model) with the columns of '
    "aedrover.clinical.evaluate_modes: " + ", ".join(CLINICAL_MODE_COLUMNS) + " ],\n"
    '  "breakeven": {"radius_m": ..., "flag": ..., "ambulance_survival": ..., "rover_survival_at_zero": ...},\n'
    '  "economics": {' + ", ".join(f'"{k}"' for k in ECONOMICS_KEYS) + ', "inputs": {...}}\n'
    "}"
)


class ExportError(RuntimeError):
    """The export cannot be produced faithfully."""


class MissingResult(ExportError):
    """A required results file or key is missing.

    ``short`` is the one-line summary used in PENDING markers; the full message carries the fix.
    """

    def __init__(self, msg: str, short: str | None = None):
        super().__init__(msg)
        self.short = short or msg.splitlines()[0]


# ---------------------------------------------------------------------------------------------
# references
# ---------------------------------------------------------------------------------------------
def load_references(repo: Path) -> dict[str, dict]:
    path = Path(repo) / "docs" / "references.json"
    if not path.exists():
        raise ExportError(f"missing {path}; citations can only come from docs/references.json")
    refs = json.loads(path.read_text(encoding="utf-8"))["references"]
    out = {r["key"]: r for r in refs}
    need = [k for k, _ in FOUNDATIONAL] + list(METHOD_REFS)
    lacking = [k for k in need if k not in out]
    if lacking:
        raise ExportError(f"docs/references.json lacks keys {lacking}")
    return out


def doi_url(ref: dict) -> str:
    return "https://doi.org/" + ref["doi"]


def surname(ref: dict) -> str:
    return ref["authors"].split()[0].rstrip(",")


def cite_short(ref: dict) -> str:
    """'Larsen et al. (1993)' or 'Khatib (1986)' from the verified author string."""
    name = surname(ref)
    if "et al" in ref["authors"] or "," in ref["authors"]:
        name += " et al."
    return f"{name} ({ref['year']})"


def venue_display(ref: dict) -> str:
    """Expand the journal abbreviation of references.json; keep volume and pages verbatim."""
    raw = ref["venue"]
    for abbr in sorted(JOURNAL_NAMES, key=len, reverse=True):
        if raw.startswith(abbr):
            rest = raw[len(abbr):].strip()
            return f"{JOURNAL_NAMES[abbr]}, {rest}" if rest else JOURNAL_NAMES[abbr]
    return raw


def journal_only(ref: dict) -> str:
    return venue_display(ref).split(",")[0]


def strip_doi(text: str) -> str:
    return re.sub(r"^https?://(dx\.)?doi\.org/", "", text.strip(), flags=re.I).lower()


# ---------------------------------------------------------------------------------------------
# results loading
# ---------------------------------------------------------------------------------------------
@dataclass
class Source:
    label: str
    path: Path
    sha12: str
    detail: str


@dataclass
class Results:
    results_dir: Path
    allow_partial: bool
    pending: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    validation: dict | None = None
    kerb: dict | None = None
    codesign: dict | None = None
    benchmark: pd.DataFrame | None = None
    benchmark_path: Path | None = None
    benchmark_is_standard: bool = False
    clinical: dict | None = None

    @property
    def is_partial(self) -> bool:
        return bool(self.pending)


def sha12(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def _rel(path: Path, base: Path) -> str:
    try:
        return path.resolve().relative_to(base.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _need_file(path: Path, hint: str) -> Path:
    if not path.exists():
        raise MissingResult(f"{path.name} not found in {path.parent} - {hint}")
    return path


def _read_json(path: Path, hint: str):
    _need_file(path, hint)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MissingResult(f"{path.name} is not valid JSON ({exc}); {hint}") from exc


def _need_keys(obj: dict, keys, where: str, hint: str) -> None:
    missing = [k for k in keys if k not in obj]
    if missing:
        raise MissingResult(f"{where}: missing keys {missing} (found {sorted(obj)}); {hint}")


def _need_columns(df: pd.DataFrame, cols, where: str, hint: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise MissingResult(f"{where}: missing columns {missing} (found {list(df.columns)}); {hint}")


def _load_validation(rd: Path, res: Results) -> None:
    hint = "run: python experiments/01_validate_suspension.py"
    path = rd / "validation_suspension.json"
    d = _read_json(path, hint)
    _need_keys(d, ("vehicle", "static", "tyre", "ringdown", "rolling", "soak"), path.name, hint)
    _need_keys(d["static"], ("ride_height_m", "ride_height_expected_m", "ride_height_err_mm",
                             "payload_acc_z", "payload_acc_expected"), path.name + "[static]", hint)
    _need_keys(d["tyre"], ("kt_eff_N_per_m", "wheel_load_N"), path.name + "[tyre]", hint)
    _need_keys(d["soak"], ("finite", "steps"), path.name + "[soak]", hint)
    if not d["ringdown"] or not d["rolling"]:
        raise MissingResult(f"{path.name}: empty ringdown/rolling lists; {hint}")
    res.validation = d
    res.sources.append(Source("physics validation", path, sha12(path),
                              f"{len(d['ringdown'])} ring-down and {len(d['rolling'])} rolling runs"))


def _load_kerb(rd: Path, res: Results) -> None:
    hint = ("run: python experiments/02_curb_traversability.py --preset standard --design nominal "
            "(and --design optimized)")
    out: dict = {"windows": {}}
    for design in ("nominal", "optimized"):
        p = _need_file(rd / f"curb_traversability_{design}_standard.csv", hint)
        df = pd.read_csv(p)
        _need_columns(df, KERB_REQUIRED, p.name, hint)
        df["success"] = df["success"].astype(bool)
        out[design] = df
        res.sources.append(Source(f"kerb traversability ({design})", p, sha12(p), f"{len(df)} trials"))
        w = _read_json(rd / f"curb_window_{design}_standard.json", hint)
        if not isinstance(w, list) or not w:
            raise MissingResult(f"curb_window_{design}_standard.json must be a non-empty list; {hint}")
        for row in w:
            _need_keys(row, WINDOW_REQUIRED, f"curb_window_{design}_standard.json row", hint)
        out["windows"][design] = w
        wp = rd / f"curb_window_{design}_standard.json"
        res.sources.append(Source(f"kerb climb windows ({design})", wp, sha12(wp), f"{len(w)} cells"))
    res.kerb = out


def _load_codesign(rd: Path, res: Results) -> None:
    hint = "run: python experiments/06_mech_codesign.py"
    path = rd / "codesign.json"
    d = _read_json(path, hint)
    if "optimized" in d and "optimised" not in d:
        d["optimised"] = d["optimized"]
    _need_keys(d, ("nominal", "optimised", "history", "nfev", "bounds"), path.name, hint)
    for arm in ("nominal", "optimised"):
        _need_keys(d[arm], ("design", "eval"), f"{path.name}[{arm}]", hint)
        _need_keys(d[arm]["eval"], ("objective", "up", "down", "n_fail", "frac_within_budget"),
                   f"{path.name}[{arm}].eval", hint)
    res.codesign = d
    res.sources.append(Source("mechanical co-design", path, sha12(path),
                              f"{d['nfev']} evaluations, {len(d['history'])} generations"))


def select_benchmark(rd: Path, allow_partial: bool, override: Path | None = None
                     ) -> tuple[Path, bool]:
    """Newest ``benchmark_standard*.csv``; with ``allow_partial`` fall back to the newest
    ``benchmark_*.csv`` that carries the ``vehicle`` column (labelled as not standard)."""
    if override is not None:
        return Path(override), "standard" in Path(override).name
    key = lambda p: (p.stat().st_mtime, p.name)  # noqa: E731
    std = sorted(rd.glob("benchmark_standard*.csv"), key=key)
    if std:
        return std[-1], True
    if not allow_partial:
        raise MissingResult(
            "no results/benchmark_standard*.csv - run: python experiments/03_controller_benchmark.py "
            "--n 100 --tag standard --controllers pure_pursuit apf_nocurb apf dwa mppi ppo")
    for p in sorted(rd.glob("benchmark_*.csv"), key=key, reverse=True):
        head = pd.read_csv(p, nrows=1)
        if "vehicle" in head.columns:
            return p, False
    raise MissingResult("no results/benchmark_*.csv with a 'vehicle' column found")


def _load_benchmark(rd: Path, res: Results, override: Path | None) -> None:
    hint = "run: python experiments/03_controller_benchmark.py --n 100 --tag standard ..."
    path, is_std = select_benchmark(rd, res.allow_partial, override)
    _need_file(path, hint)
    df = pd.read_csv(path)
    _need_columns(df, BENCH_REQUIRED, path.name, hint)
    df["success"] = df["success"].astype(bool)
    if len(df) < MIN_CSV_ROWS:
        raise ExportError(f"{path.name} has only {len(df)} rows; the course CSV needs at least "
                          f"{MIN_CSV_ROWS} real episode rows (run experiment 03 with a larger --n)")
    res.benchmark, res.benchmark_path, res.benchmark_is_standard = df, path, is_std
    res.sources.append(Source("controller benchmark", path, sha12(path),
                              f"{len(df)} episodes, controllers {sorted(df['controller'].unique())}"))
    per = df.groupby("controller").size()
    if not is_std:
        res.pending.append(
            f"benchmark: only {path.name} (a non-standard run, N per controller = "
            f"{int(per.min())}-{int(per.max())}) is available; results/benchmark_standard*.csv is pending")
    missing_c = [c for c in REQUIRED_CONTROLLERS if c not in set(df["controller"])]
    if missing_c:
        msg = (f"{path.name} lacks controllers {missing_c}; re-run experiment 03 with "
               f"--controllers pure_pursuit apf_nocurb apf dwa mppi ppo")
        if not res.allow_partial:
            raise MissingResult(msg)
        res.pending.append("benchmark controllers " + ", ".join(missing_c) + " not yet in the benchmark file")
    low = per[per < MIN_EPISODES_PER_CONTROLLER]
    if len(low):
        msg = (f"{path.name}: controllers with fewer than {MIN_EPISODES_PER_CONTROLLER} episodes: "
               f"{low.to_dict()} (course mandate N >= {MIN_EPISODES_PER_CONTROLLER})")
        if not res.allow_partial:
            raise ExportError(msg)
        res.warnings.append(msg)


def find_clinical(rd: Path, override: Path | None = None) -> Path:
    if override is not None:
        return Path(override)
    cands = sorted(rd.glob("clinical*.json"), key=lambda p: (p.stat().st_mtime, p.name))
    if not cands:
        raise MissingResult(
            "no results/clinical*.json (survival, break-even and economics results). Track B must "
            "write it; " + CLINICAL_SCHEMA_HELP, short="results/clinical*.json has not been produced yet")
    return cands[-1]


def _load_clinical(rd: Path, res: Results, override: Path | None) -> None:
    hint = CLINICAL_SCHEMA_HELP
    path = find_clinical(rd, override)
    d = _read_json(path, hint)
    _need_keys(d, ("modes", "breakeven", "economics"), path.name, hint)
    if not isinstance(d["modes"], list) or not d["modes"]:
        raise MissingResult(f"{path.name}: 'modes' must be a non-empty list; {hint}")
    for row in d["modes"]:
        _need_keys(row, CLINICAL_MODE_COLUMNS, f"{path.name} modes row", hint)
    _need_keys(d["breakeven"], ("radius_m", "flag"), f"{path.name}[breakeven]", hint)
    _need_keys(d["economics"], ECONOMICS_KEYS, f"{path.name}[economics]", hint)
    primary = d.get("model", "larsen1993")
    prim_modes = {r["mode"] for r in d["modes"] if r["model"] == primary}
    if not {"ambulance", "rover"} <= prim_modes:
        raise MissingResult(f"{path.name}: model {primary!r} needs rows for modes 'ambulance' and "
                            f"'rover' (found {sorted(prim_modes)}); {hint}")
    d["model"] = primary
    res.clinical = d
    res.sources.append(Source("clinical and economics", path, sha12(path),
                              f"{len(d['modes'])} mode rows, model {primary}"))


def staleness(res: Results, current_vehicle: dict, optimized_params) -> list[str]:
    """Differences between the results files and the package that is being shipped."""
    out: list[str] = []
    if res.validation is not None:
        for k, v in res.validation["vehicle"].items():
            cur = current_vehicle.get(k)
            cur = list(cur) if isinstance(cur, tuple) else cur
            if cur != v:
                out.append(f"validation_suspension.json was generated with {k}={v} but the current "
                           f"VehicleParams has {k}={cur}; re-run experiments/01_validate_suspension.py")
    if res.codesign is not None:
        design = res.codesign["optimised"]["design"]
        for k, v in design.items():
            cur = getattr(optimized_params, k)
            if not math.isclose(float(cur), float(v), rel_tol=5e-4, abs_tol=1e-4):
                out.append(f"codesign.json optimised {k}={v:.5g} differs from configs/vehicle_optimized.yaml "
                           f"({cur:.5g}); re-run experiments/06_mech_codesign.py")
    if res.benchmark is not None:
        veh = sorted(set(res.benchmark["vehicle"].dropna().astype(str)))
        if veh != ["optimized"]:
            out.append(f"benchmark file used vehicle(s) {veh} but the exported model is the optimised "
                       "design; re-run experiment 03 with --vehicle optimized")
    return out


def load_results(results_dir: Path, *, allow_partial: bool = False, benchmark: Path | None = None,
                 clinical: Path | None = None, accept_stale: bool = False) -> Results:
    """Load and validate every results file. Strict mode raises one ``ExportError`` that lists
    everything that is missing; ``allow_partial`` records gaps in ``Results.pending`` instead."""
    from aedrover.sim.vehicle_mjcf import VehicleParams

    rd = Path(results_dir)
    res = Results(results_dir=rd, allow_partial=allow_partial)
    missing: list[tuple[str, MissingResult]] = []
    steps = (
        ("physics validation (experiments/01)", lambda: _load_validation(rd, res)),
        ("kerb traversability (experiments/02)", lambda: _load_kerb(rd, res)),
        ("mechanical co-design (experiments/06)", lambda: _load_codesign(rd, res)),
        ("controller benchmark (experiments/03)", lambda: _load_benchmark(rd, res, benchmark)),
        ("clinical and economics (Track B)", lambda: _load_clinical(rd, res, clinical)),
    )
    for name, fn in steps:
        try:
            fn()
        except MissingResult as exc:
            missing.append((name, exc))
    if missing:
        if not allow_partial:
            raise ExportError(f"{len(missing)} required result(s) missing or incomplete:\n  - "
                              + "\n  - ".join(f"{n}: {e}" for n, e in missing)
                              + "\nUse --allow-partial to export with PENDING markers.")
        res.pending.extend(f"{n}: {e.short}" for n, e in missing)
    stale = staleness(res, VehicleParams().to_dict(), VehicleParams.optimized())
    if stale:
        if allow_partial or accept_stale:
            res.warnings.extend(stale)
        else:
            raise ExportError("results are inconsistent with the shipped package:\n  - "
                              + "\n  - ".join(stale) + "\nRe-run the named experiments (or pass "
                              "--accept-stale to export anyway).")
    return res


# ---------------------------------------------------------------------------------------------
# derived numbers: benchmark
# ---------------------------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    if n <= 0:
        return math.nan, math.nan, math.nan
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def controller_order(df: pd.DataFrame) -> list[str]:
    present = list(df["controller"].unique())
    known = [c for c in ("pure_pursuit", "apf_nocurb", "apf", "dwa_nocurb", "dwa", "mppi", "ppo")
             if c in present]
    return known + sorted(c for c in present if c not in known)


def benchmark_summary(df: pd.DataFrame, budget_g: float = BUDGET_G) -> list[dict]:
    rows = []
    for c in controller_order(df):
        sub = df[df["controller"] == c]
        ok = sub[sub["success"]]
        p, lo, hi = wilson(int(sub["success"].sum()), len(sub))
        outcome = sub["outcome"].value_counts()
        rows.append({
            "controller": c, "n": len(sub), "success": p, "ci_low": lo, "ci_high": hi,
            "median_time_s": float(ok["time_s"].median()) if len(ok) else math.nan,
            "median_peak_g": float(ok["peak_shock_g"].median()) if len(ok) else math.nan,
            "over_budget": float((sub["peak_shock_g"] > budget_g).mean()),
            "collision": int(outcome.get("collision", 0)),
            "rollover": int(outcome.get("rollover", 0)),
            "off_sidewalk": int(outcome.get("off_sidewalk", 0)),
            "stall_timeout": int(outcome.get("stall", 0) + outcome.get("timeout", 0)),
        })
    return rows


def family_success(df: pd.DataFrame) -> tuple[list[str], list[str], dict]:
    fams = [f for f in ("flat_clear", "kerb", "crowded", "mixed", "slippery") if f in set(df["family"])]
    fams += sorted(set(df["family"]) - set(fams))
    ctrls = controller_order(df)
    cell = {}
    for c in ctrls:
        for f in fams:
            sub = df[(df["controller"] == c) & (df["family"] == f)]
            cell[(c, f)] = (float(sub["success"].mean()) if len(sub) else math.nan, len(sub))
    return ctrls, fams, cell


def pick_reference(df: pd.DataFrame) -> str | None:
    """Pre-declared rule: the classical controller with the highest success rate, ties broken by the
    lower median completion time of its successful episodes."""
    best, key = None, None
    for r in benchmark_summary(df):
        if r["controller"] not in CLASSICAL_CONTROLLERS:
            continue
        k = (-r["success"], r["median_time_s"] if not math.isnan(r["median_time_s"]) else math.inf)
        if key is None or k < key:
            best, key = r["controller"], k
    return best


def paired_comparisons(df: pd.DataFrame, reference: str, min_pairs: int = 3) -> list[dict]:
    """Paired comparison of every other controller against ``reference`` on identical scenarios.

    Pairs are scenarios (family, seed). Time and peak shock use a paired t test over the pairs in
    which both controllers reached the goal; success uses an exact McNemar test over all pairs.
    ``diff`` is other minus reference. Holm adjustment covers all t tests together.
    """
    from scipy.stats import binomtest

    from aedrover.analysis.stats import holm_correction, paired_t

    ref = df[df["controller"] == reference].set_index(["family", "seed"])
    if ref.index.duplicated().any():
        raise ExportError("benchmark file has repeated (family, seed) rows for the reference "
                          "controller; pairing needs one row per scenario")
    out: list[dict] = []
    for c in controller_order(df):
        if c == reference:
            continue
        oth = df[df["controller"] == c].set_index(["family", "seed"])
        if oth.index.duplicated().any():
            raise ExportError(f"benchmark file has repeated (family, seed) rows for controller {c!r}")
        idx = ref.index.intersection(oth.index)
        if len(idx) < min_pairs:
            continue
        r, o = ref.loc[idx], oth.loc[idx]
        both = r["success"].to_numpy() & o["success"].to_numpy()
        b = int((r["success"].to_numpy() & ~o["success"].to_numpy()).sum())   # reference only
        cc = int((~r["success"].to_numpy() & o["success"].to_numpy()).sum())  # other only
        p_mcnemar = float(binomtest(b, b + cc, 0.5).pvalue) if b + cc > 0 else 1.0
        rec: dict = {"controller": c, "n_pairs": int(len(idx)), "n_both": int(both.sum()),
                     "ref_only": b, "other_only": cc, "p_mcnemar": p_mcnemar,
                     "success_diff": float(o["success"].mean() - r["success"].mean())}
        for metric, col in (("time", "time_s"), ("shock", "peak_shock_g")):
            if int(both.sum()) >= min_pairs:
                rec[metric] = paired_t(o.loc[both, col].to_numpy(), r.loc[both, col].to_numpy())
            else:
                rec[metric] = None
        out.append(rec)
    pvals, slots = [], []
    for i, rec in enumerate(out):
        for metric in ("time", "shock"):
            if rec[metric] is not None and not math.isnan(rec[metric].p):
                pvals.append(rec[metric].p)
                slots.append((i, metric))
    if pvals:
        adj = holm_correction(pvals)
        for (i, metric), a in zip(slots, adj, strict=True):
            out[i][f"p_{metric}_holm"] = float(a)
    return out


# ---------------------------------------------------------------------------------------------
# derived numbers: kerb and co-design
# ---------------------------------------------------------------------------------------------
def _vmin_table(df: pd.DataFrame, radius: float, mu: float) -> dict[float, tuple[float | None, float | None]]:
    sub = df[(df["direction"] == "up") & (df["angle_deg"] == 0.0) & np.isclose(df["wheel_radius"], radius)
             & np.isclose(df["mu"], mu)]
    out: dict[float, tuple[float | None, float | None]] = {}
    for h, g in sub.groupby("kerb_h"):
        ok = g[g["success"]].sort_values("speed")
        out[float(h)] = (float(ok["speed"].iloc[0]), float(ok["peak_g"].iloc[0])) if len(ok) else (None, None)
    return out


def _nearest(values, target: float) -> float:
    vals = sorted({float(v) for v in values})
    return min(vals, key=lambda v: abs(v - target))


def kerb_summary(kerb: dict, nominal_radius: float, optimized_radius: float,
                 budget_g: float = BUDGET_G) -> dict:
    out: dict = {"budget_g": budget_g, "designs": {}}
    for design in ("nominal", "optimized"):
        df: pd.DataFrame = kerb[design]
        r = _nearest(df["wheel_radius"].unique(), nominal_radius if design == "nominal" else optimized_radius)
        mus = sorted(float(m) for m in df["mu"].unique())
        mu_hi, mu_lo = max(mus), min(mus)
        up = df[(df["direction"] == "up") & np.isclose(df["wheel_radius"], r)]
        # oblique retention: cells that succeed head-on and also succeed at each other angle
        keys = ["kerb_h", "speed", "mu"]
        base = up[up["angle_deg"] == 0.0].set_index(keys)["success"]
        retention = {}
        for a in sorted(up["angle_deg"].unique()):
            if a == 0.0:
                continue
            oth = up[up["angle_deg"] == a].set_index(keys)["success"]
            idx = base[base].index.intersection(oth.index)
            retention[float(a)] = float(oth.loc[idx].mean()) if len(idx) else math.nan
        win = [w for w in kerb["windows"][design] if w["direction"] == "up"
               and math.isclose(w["wheel_radius"], r, abs_tol=1e-3)]
        down = df[(df["direction"] == "down") & np.isclose(df["wheel_radius"], r) & df["success"]]
        out["designs"][design] = {
            "wheel_radius": r, "n_trials": int(len(df)), "outcomes": df["outcome"].value_counts().to_dict(),
            "mu_hi": mu_hi, "mu_lo": mu_lo,
            "vmin_hi": _vmin_table(df, r, mu_hi), "vmin_lo": _vmin_table(df, r, mu_lo),
            "oblique_retention": retention,
            "window_fraction": float(np.mean([w["feasible_window"] for w in win])) if win else math.nan,
            "n_window_cells": len(win),
            "down_peak_g_median": float(down["peak_g"].median()) if len(down) else math.nan,
            "down_peak_g_max": float(down["peak_g"].max()) if len(down) else math.nan,
        }
    out["heights"] = sorted(out["designs"]["nominal"]["vmin_hi"])
    return out


def damping_ratio(k: float, c: float, mass: float) -> float:
    return c / (2.0 * math.sqrt(k * mass))


def codesign_summary(cd: dict, veh_nominal) -> dict:
    """Design variables, derived suspension/isolator dynamics and objective of both arms."""
    out: dict = {"nfev": int(cd["nfev"]), "generations": len(cd["history"]),
                 "history_first": float(cd["history"][0]) if cd["history"] else math.nan,
                 "history_last": float(cd["history"][-1]) if cd["history"] else math.nan,
                 "bounds": cd["bounds"], "arms": {}}
    m_s = veh_nominal.sprung_per_wheel
    m_p = veh_nominal.payload_mass
    for arm in ("nominal", "optimised"):
        d, ev = cd[arm]["design"], cd[arm]["eval"]
        out["arms"][arm] = {
            "design": d, "objective": ev["objective"], "n_fail": ev["n_fail"],
            "frac_within_budget": ev["frac_within_budget"], "up": ev["up"], "down": ev["down"],
            "zeta_susp": damping_ratio(d["susp_k"], d["susp_c"], m_s),
            "omega_susp": math.sqrt(d["susp_k"] / m_s),
            "zeta_iso": damping_ratio(d["iso_kz"], d["iso_cz"], m_p),
            "omega_iso": math.sqrt(d["iso_kz"] / m_p),
        }
    return out


def clinical_view(cd: dict) -> dict:
    """Primary-model rows by mode, sensitivity rows, breakeven and economics."""
    model = cd["model"]
    prim = {r["mode"]: r for r in cd["modes"] if r["model"] == model}
    other = [r for r in cd["modes"] if r["model"] != model]
    return {"model": model, "primary": prim, "other": other, "breakeven": cd["breakeven"],
            "economics": cd["economics"], "scenario": cd.get("scenario"), "parallel": cd.get("parallel")}


# ---------------------------------------------------------------------------------------------
# package facts (read from the live package, never typed in)
# ---------------------------------------------------------------------------------------------
def _defaults(obj) -> dict:
    sig = inspect.signature(obj)
    return {k: v.default for k, v in sig.parameters.items() if v.default is not inspect.Parameter.empty}


def package_facts() -> dict:
    """Constants of the shipped model and controllers, collected from the importable package."""
    from aedrover.clinical import survival
    from aedrover.control.curb import CurbNegotiator
    from aedrover.control.safety_filter import SafetyParams
    from aedrover.learning import rl_env
    from aedrover.nav.apf import APFController
    from aedrover.nav.dwa import DWAController
    from aedrover.nav.mppi import MPPIController
    from aedrover.nav.pure_pursuit import PurePursuit
    from aedrover.sim import scenario, sensors
    from aedrover.sim.env import AEDRoverEnv
    from aedrover.sim.metrics import DEFAULT_CUTOFF_HZ
    from aedrover.sim.pedestrians import ROBOT_HALF_L, ROBOT_HALF_W
    from aedrover.sim.vehicle_mjcf import VehicleParams
    from aedrover.sim.world import WorldSpec

    env = _defaults(AEDRoverEnv.__init__)
    cn = CurbNegotiator()
    ppo_src = Path(rl_env.__file__).with_name("train_ppo.py").read_text(encoding="utf-8")
    m_arch = re.search(r"net_arch=dict\(pi=\[([\d, ]+)\]", ppo_src)
    return {
        "curb": {k: getattr(cn, k) for k in ("mu_assumed", "margin", "v_down", "a_dec", "hold_after",
                                             "detect_thresh", "max_step", "width_tol")},
        "ppo_net_arch": [int(x) for x in m_arch.group(1).split(",")] if m_arch else None,
        "ppo_tanh": "Tanh" in ppo_src,
        "footprint": (2 * ROBOT_HALF_L, 2 * ROBOT_HALF_W),
        "scenario_defaults": _defaults(scenario.sample_scenario),
        "env": env,
        "world": dict(WorldSpec().__dict__),
        "perception": dict(sensors.PerceptionSpec().__dict__),
        "safety": dict(SafetyParams().__dict__),
        "mppi": _defaults(MPPIController.__init__),
        "dwa": _defaults(DWAController.__init__),
        "apf": _defaults(APFController.__init__),
        "pure_pursuit": _defaults(PurePursuit.__init__),
        "families": tuple(scenario.FAMILIES),
        "corridor_half_width": scenario.CORRIDOR_HALF_WIDTH,
        "sidewalk_half_width": scenario.SIDEWALK_HALF_WIDTH,
        "cutoff_hz": DEFAULT_CUTOFF_HZ,
        "train_ranges": dict(rl_env.TRAIN_RANGES),
        "ood_ranges": dict(rl_env.OOD_RANGES),
        "family_probs": dict(rl_env.DEFAULT_FAMILY_PROBS),
        "rl_repeat": _defaults(rl_env.RLRoverEnv.__init__).get("repeat"),
        "rl_budget_g": _defaults(rl_env.RLRoverEnv.__init__).get("budget_g"),
        "larsen": {"intercept": survival.LARSEN_INTERCEPT, "cpr": survival.LARSEN_SLOPE_CPR,
                   "defib": survival.LARSEN_SLOPE_DEFIB, "acls": survival.LARSEN_SLOPE_ACLS},
        "rule_low": survival.RULE_RATE_LOW, "rule_high": survival.RULE_RATE_HIGH,
        "veh_nominal": VehicleParams.nominal(), "veh_optimized": VehicleParams.optimized(),
    }


def arch_facts(pf: dict | None = None) -> dict:
    """The dictionary that ``figures.ARCH_FACTS`` must equal (the exporter writes it into the copy)."""
    pf = pf or package_facts()
    return {
        "n_families": len(pf["families"]),
        "n_lidar": pf["perception"]["n_lidar"],
        "fov_deg": float(pf["perception"]["fov_deg"]),
        "lidar_range_m": float(pf["perception"]["lidar_range"]),
        "control_hz": int(round(1.0 / pf["env"]["control_dt"])),
        "physics_dt_ms": float(pf["world"]["timestep"]) * 1000.0,
        "cutoff_hz": float(pf["cutoff_hz"]),
        "budget_g": float(pf["env"]["budget_g"]),
        "a_brake": float(pf["safety"]["a_brake"]),
        "t_react_s": float(pf["safety"]["t_react"]),
        "corridor_half_width_m": float(pf["corridor_half_width"]),
        "mppi_samples": int(pf["mppi"]["K"]),
        "mppi_horizon": int(pf["mppi"]["H"]),
    }


# ---------------------------------------------------------------------------------------------
# standalone MJCF snapshot
# ---------------------------------------------------------------------------------------------
def build_snapshot_xml(kerb_h: float = 0.12, x_down: float = 16.0, x_up: float = 22.0) -> str:
    """Standalone MJCF of the optimised rover on a kerbed sidewalk crossing.

    ``aedrover.sim.world.build_xml`` creates terrain as parked mocap slabs that the environment moves
    at run time. For a file that loads with plain ``mujoco.MjModel.from_xml_path`` the two sidewalk
    slabs are moved into place inside the XML (walk A ends at ``x_down``, walk B starts at ``x_up``,
    both with their top at ``kerb_h``) and the rover starts on walk A.
    """
    from aedrover.sim.vehicle_mjcf import VehicleParams
    from aedrover.sim.world import WorldSpec, build_xml

    spec = WorldSpec()
    veh = VehicleParams.optimized()
    xml = build_xml(veh, spec, start_x=0.0, start_y=0.0, start_h=kerb_h, start_yaw=0.0)
    half = 0.5 * spec.slab_len
    placements = {"kerb_a": (x_down - half, 0.0, kerb_h - 0.5), "kerb_b": (x_up + half, 0.0, kerb_h - 0.5)}
    for name, pos in placements.items():
        pat = re.compile(rf'(<body name="{name}" mocap="true" pos=")[^"]*(")')
        if len(pat.findall(xml)) != 1:
            raise ExportError(f"build_xml no longer emits exactly one mocap body {name!r}; update "
                              "aedrover.export.build_snapshot_xml")
        xml = pat.sub(lambda m, p=pos: m.group(1) + " ".join(f"{v:.6g}" for v in p) + m.group(2), xml)
    header = (
        "<!-- Standalone snapshot generated by aedrover.export (scripts/export_nmims.py).\n"
        "     Source: aedrover.sim.world.build_xml(VehicleParams.optimized(), ...), the co-designed rover\n"
        f"     ({veh.wheel_radius:.4f} m wheels, suspension {veh.susp_k:.1f} N/m and {veh.susp_c:.1f} N s/m).\n"
        f"     Scene: sidewalk (top {100 * kerb_h:.0f} cm above the road) until x = {x_down:g} m, road until\n"
        f"     x = {x_up:g} m, sidewalk again beyond. Pedestrian and obstacle slots are parked 50 m below the\n"
        "     road; the vendored Python package (src/aedrover) moves them at run time.\n"
        "     Load with: mujoco.MjModel.from_xml_path('models/aed_delivery_amr.xml') -->\n"
    )
    i = xml.index("<mujoco")
    return xml[:i] + header + xml[i:]


def fmt_p(p: float) -> str:
    """Exact p value: fixed notation above 1e-3, scientific below."""
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    return f"{p:.4f}" if p >= 1e-3 else f"{p:.2e}"
