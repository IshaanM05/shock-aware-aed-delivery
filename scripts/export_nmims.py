"""Build the NMIMS course folder ``Group_03_Kashish_Vaishnavi`` from this repository's real results.

    python scripts/export_nmims.py                      # strict: fails loudly if a result is missing
    python scripts/export_nmims.py --allow-partial      # PENDING markers for missing results
    python scripts/export_nmims.py --apply              # copy into the course repository (never commits)

The folder is written to ``dist/nmims/Group_03_Kashish_Vaishnavi`` (cleaned first, deterministic).
The course repository is never touched unless ``--apply`` is given, and ``--apply`` only ever copies
into its ``Group_03_Kashish_Vaishnavi/`` directory, refuses to run if that directory has uncommitted
changes, and never runs ``git add`` or ``git commit``. See ``docs/NMIMS_EXPORT.md``.

Exit status: 0 = export built and every check passed; 1 = export or a check failed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import pprint
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd  # noqa: E402

from aedrover import export as E  # noqa: E402
from aedrover.export import figures  # noqa: E402
from aedrover.export import templates as T  # noqa: E402

TEXT_EXT = {".py", ".md", ".json", ".yaml", ".yml", ".csv", ".txt", ".xml", ".cfg", ".toml"}
VENDOR_IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "tests", "export")
SNAPSHOT_KERB_H = 0.12
EXPECTED_FILES = (
    "README.md", "RESEARCH_AND_IMPLEMENTATION_GUIDE.md", "docs/TEAM_ROSTER.json",
    "docs/LITERATURE_REVIEW_AND_FOUNDATIONAL_PAPERS.md", "docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md",
    "docs/figures/figure1_system_architecture.png", "docs/figures/figure2_kinematic_telemetry.png",
    "docs/figures/figure3_comparative_performance.png", "docs/figures/video_poster.png",
    "models/aed_delivery_amr.xml", "models/.gitkeep", "src/test_env.py", "src/aed_navigation_controller.py",
    "src/aedrover/__init__.py", "analytics/cardiac_survival_economics.py", "analytics/generate_paper_figures.py",
    "analytics/aed_delivery_benchmark.csv", "analytics/kerb_crossing_telemetry.csv", "analytics/.gitkeep",
    "configs/vehicle.yaml", "configs/vehicle_optimized.yaml", "configs/curb_table.json", "configs/mppi_tuned.json",
)
LINKEDIN_IMG = r"\[!\[Watch Video Demonstration on LinkedIn\]\([^\)]+\)\]\([^\)]+\)"
LINKEDIN_TXT = r"\* \*\*Video Demonstration:\*\* \[Watch 60-Second Walkthrough on LinkedIn\]\([^\)]+\)"


@dataclass
class Report:
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.checks.append((name, ok, detail))
        if not ok:
            self.violations.append(f"{name}: {detail}" if detail else name)
        return ok


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------
def default_nmims_repo() -> Path:
    env = os.environ.get("NMIMS_REPO")
    return Path(env) if env else REPO.parent / "NMIMS-MPSTME-MDRIIA-2026"


def load_audit(path: Path):
    """Import ``audit_course_boundaries_and_rules.py`` by path without letting it hijack stdout."""
    if not path.exists():
        raise E.ExportError(f"course audit script not found: {path} (pass --nmims-repo or --audit-script)")
    spec = importlib.util.spec_from_file_location("nmims_audit", path)
    mod = importlib.util.module_from_spec(spec)
    saved = (sys.stdout, sys.stderr)
    try:
        spec.loader.exec_module(mod)
    finally:
        replaced = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = saved
        for stream in replaced:
            if stream not in saved:
                try:
                    stream.detach()      # keep the underlying buffers open
                except Exception:        # noqa: BLE001
                    pass
    if not hasattr(mod, "audit_file"):
        raise E.ExportError(f"{path} has no audit_file()")
    return mod


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def copy_normalised(src: Path, dst: Path) -> None:
    """Copy a file; text files get LF line endings so the export is identical on every platform."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = src.read_bytes()
    if src.suffix.lower() in TEXT_EXT:
        data = data.replace(b"\r\n", b"\n")
    dst.write_bytes(data)


def vendor_package(dst: Path) -> None:
    src = REPO / "src" / "aedrover"
    for path in sorted(src.rglob("*")):
        rel = path.relative_to(src)
        if any(part in ("__pycache__", "tests", "export") for part in rel.parts) or path.suffix == ".pyc":
            continue
        if path.is_file():
            copy_normalised(path, dst / rel)


def clean_target(target: Path, nmims: Path) -> None:
    resolved = target.resolve()
    if resolved.name != E.GROUP_FOLDER:
        raise E.ExportError(f"refusing to clean {resolved}: not a {E.GROUP_FOLDER} folder")
    if resolved in (REPO.resolve(), Path(resolved.anchor)):
        raise E.ExportError(f"refusing to clean {resolved}")
    try:
        resolved.relative_to(nmims.resolve())
        raise E.ExportError(f"--out must not be inside the course repository ({nmims})")
    except ValueError:
        pass
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True)


def run(cmd: list[str], cwd: Path, timeout: int = 900) -> tuple[bool, str]:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"timed out after {timeout} s"
    return p.returncode == 0, (p.stdout or "") + (p.stderr or "")


def purge_pycache(root: Path) -> None:
    for d in sorted(root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(d, ignore_errors=True)


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def fingerprint(root: Path) -> str:
    h = hashlib.sha256()
    for p in tree_files(root):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# building blocks
# ---------------------------------------------------------------------------------------------
def write_static(target: Path, res: E.Results, facts: dict, xml: str) -> None:
    write_text(target / "src" / "test_env.py", T.TEST_ENV_PY)
    write_text(target / "src" / "aed_navigation_controller.py", T.NAV_PY)
    write_text(target / "analytics" / "cardiac_survival_economics.py", T.ECON_PY)
    fig_src = Path(figures.__file__).read_text(encoding="utf-8").replace("\r\n", "\n")
    block = "ARCH_FACTS = " + pprint.pformat(facts, sort_dicts=False) + "\n"
    fig_src, n = re.subn(r"(# --- BEGIN ARCH_FACTS[^\n]*\n).*?(# --- END ARCH_FACTS ---)",
                         lambda m: m.group(1) + block + m.group(2), fig_src, flags=re.S)
    if n != 1:
        raise E.ExportError("figures.py lost its ARCH_FACTS markers")
    write_text(target / "analytics" / "generate_paper_figures.py", fig_src)
    for keep in ("models/.gitkeep", "analytics/.gitkeep"):
        write_text(target / keep, "")
    write_text(target / "models" / "aed_delivery_amr.xml", xml)
    vendor_package(target / "src" / "aedrover")
    for name in ("vehicle.yaml", "vehicle_optimized.yaml", "curb_table.json", "mppi_tuned.json"):
        copy_normalised(REPO / "configs" / name, target / "configs" / name)
    text = res.benchmark_path.read_text(encoding="utf-8").replace("\r\n", "\n")
    write_text(target / "analytics" / "aed_delivery_benchmark.csv", text)


def pick_episode(df: pd.DataFrame) -> pd.Series:
    """Representative episode for Figure 2: the lowest-seed successful kerb episode of the reference
    classical controller (DWA when present)."""
    ref = "dwa" if "dwa" in set(df["controller"]) else E.pick_reference(df)
    for family_filter in (df["family"] == "kerb", df["sc_kerb_h"] > 1e-6):
        sub = df[(df["controller"] == ref) & df["success"] & family_filter].sort_values("seed")
        if len(sub):
            return sub.iloc[0]
    raise E.ExportError(f"no successful kerb episode of controller {ref!r} in the benchmark file; cannot draw Figure 2")


def make_telemetry(target: Path, res: E.Results, report: Report) -> dict:
    row = pick_episode(res.benchmark)
    tel_path = target / "analytics" / "kerb_crossing_telemetry.csv"
    cmd = [sys.executable, str(target / "src" / "aed_navigation_controller.py"), "--controllers", str(row["controller"]),
           "--family", str(row["family"]), "--seed", str(int(row["seed"])), "--vehicle", str(row["vehicle"]),
           "--telemetry", str(tel_path)]
    if "shield" in row.index and not bool(row["shield"]):
        cmd.append("--no-shield")
    ok, out = run(cmd, target)
    report.check("entry point runs (DWA episode, telemetry written)", ok and tel_path.exists(), out.strip()[-400:] if not ok else "")
    if not ok:
        raise E.ExportError("src/aed_navigation_controller.py failed:\n" + out[-1500:])
    m = re.search(r"outcome=(\w+)\s+time=([\d.]+) s.*?peak_shock=([\d.]+) g", out)
    tel = pd.read_csv(tel_path)
    if m is None or tel.empty:
        raise E.ExportError("could not read the entry point summary:\n" + out[-800:])
    outcome, t_s, peak = m.group(1), float(m.group(2)), float(m.group(3))
    dt = abs(t_s - float(row["time_s"]))
    same = outcome == str(row["outcome"]) and dt <= 0.1
    note = (f"replayed benchmark episode ({row['controller']}, {row['family']}, seed {int(row['seed'])}): outcome {outcome} "
            f"(benchmark {row['outcome']}), time {t_s:.2f} s (benchmark {float(row['time_s']):.2f} s), episode peak shock "
            f"{peak:.2f} g (benchmark {float(row['peak_shock_g']):.2f} g). "
            + ("The replay reproduces the benchmark row." if same else
               "The replay DIFFERS from the benchmark row: the exported entry point or its settings no longer match "
               "the benchmark run (speed cap, vehicle, controller settings) or the simulator changed; re-run the benchmark."))
    if not same:
        res.warnings.append("telemetry replay differs from its benchmark row: " + note)
    kerb_h = float(tel["kerb_h_m"].iloc[0])
    caption = (f"Kinematic telemetry of one simulated episode ({T.lab(str(row['controller']))}, {row['family']} family, seed "
               f"{int(row['seed'])}, {100 * kerb_h:.1f} cm kerb): speed command, per-step payload shock and lateral position; the "
               f"shaded band is the road crossing. The episode peak shock (low-pass metric) is {peak:.2f} g against the "
               f"{E.BUDGET_G:g} g budget.")
    return {"seed": int(row["seed"]), "controller": str(row["controller"]), "replay_note": note, "fig2_caption": caption,
            "snapshot_kerb_h": SNAPSHOT_KERB_H, "replay_same": same}


# ---------------------------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------------------------
def audit_canary(audit) -> bool:
    """The auditor must actually flag a violation (guards against path-based skipping)."""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "canary.md"
        p.write_text("cost USD 500 and a \\$ sign\n", encoding="utf-8")
        return bool(audit.audit_file(str(p)))


def run_audit(audit, target: Path) -> list[str]:
    violations: list[str] = []
    old = Path.cwd()
    os.chdir(target)
    try:
        for p in tree_files(Path(".")):
            violations.extend(audit.audit_file(str(p)))
    finally:
        os.chdir(old)
    return violations


def doi_candidates(text: str) -> list[str]:
    return re.findall(r"10\.\d{4,9}/\S+", text)


def unknown_dois(text: str, known: set[str]) -> list[str]:
    bad = []
    for cand in doi_candidates(text):
        c = cand.strip().lower()
        while c and c not in known and c[-1] in ".,;:)]}*\"'`>":
            c = c[:-1]
        if c not in known:
            # allow a DOI that is a prefix-match followed by markdown punctuation only
            if not any(c.startswith(k) for k in known):
                bad.append(cand)
    return bad


def link_problems(root: Path) -> list[str]:
    problems = []
    for md in sorted(root.rglob("*.md")):
        text = md.read_text(encoding="utf-8")
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"^(https?:|mailto:|#)", target):
                continue
            path = (md.parent / target.split("#")[0]).resolve()
            if not path.exists():
                problems.append(f"{md.relative_to(root)} links to missing {target}")
    return problems


def check_roster(root: Path, refs: dict, report: Report) -> None:
    roster = json.loads((root / "docs" / "TEAM_ROSTER.json").read_text(encoding="utf-8"))
    known = {r["doi"].lower() for r in refs.values()}
    members = roster["members"]
    report.check("roster: two members exactly as in the draft",
                 [{k: m[k] for k in E.MEMBERS[0] if k not in ("github_handle", "institutional_email")} for m in members]
                 == [{k: m[k] for k in E.MEMBERS[0] if k not in ("github_handle", "institutional_email")} for m in E.MEMBERS])
    report.check("roster: empty github_handle and institutional_email",
                 all(m.get("github_handle") == "" and m.get("institutional_email") == "" for m in members))
    papers = roster["foundational_papers"]
    report.check("roster: exactly 6 foundational papers", len(papers) == 6, f"found {len(papers)}")
    kinds = [p["type"] for p in papers]
    report.check("roster: 2 seminal and 4 recent", kinds.count("Seminal") == 2 and kinds.count("Recent") == 4, str(kinds))
    report.check("roster: every DOI is in docs/references.json",
                 all(E.strip_doi(p["doi"]) in known for p in papers), str([p["doi"] for p in papers if E.strip_doi(p["doi"]) not in known]))
    report.check("roster: titles equal docs/references.json",
                 all(p["title"] == refs[k]["title"] for p, (k, _) in zip(papers, E.FOUNDATIONAL, strict=True)))
    report.check("roster: showcase pending, authorized title verbatim",
                 roster["project_showcase"]["linkedin_url"] == "PENDING_SUBMISSION" and roster["authorized_title"] == E.AUTHORIZED_TITLE)


def check_png(root: Path, report: Report) -> None:
    from PIL import Image

    for name in ("figure1_system_architecture", "figure2_kinematic_telemetry", "figure3_comparative_performance"):
        p = root / "docs" / "figures" / f"{name}.png"
        if not p.exists():
            report.check(f"figure {name} exists", False)
            continue
        with Image.open(p) as im:
            dpi = im.info.get("dpi", (0, 0))[0]
            report.check(f"figure {name} is 300 DPI", abs(dpi - 300) < 0.5, f"dpi={dpi}, size={im.size}")
    p = root / "docs" / "figures" / "video_poster.png"
    if p.exists():
        with Image.open(p) as im:
            report.check("video poster is 1280x720", im.size == (1280, 720), str(im.size))


def structural_checks(target: Path, refs: dict, res: E.Results, report: Report, allow_partial: bool) -> None:
    for rel in EXPECTED_FILES:
        report.check(f"file exists: {rel}", (target / rel).exists())
    readme = (target / "README.md").read_text(encoding="utf-8")
    report.check("README carries the authorized title verbatim", E.AUTHORIZED_TITLE in readme)
    report.check("README LinkedIn image line matches the sync regex", len(re.findall(LINKEDIN_IMG, readme)) == 1)
    report.check("README LinkedIn text line matches the sync regex", len(re.findall(LINKEDIN_TXT, readme)) == 1)
    tree_block = readme.split("## 5. Repository Directory Architecture", 1)[-1].split("## 6.", 1)[0]
    missing = [Path(r).name for r in EXPECTED_FILES if Path(r).name not in tree_block and not r.startswith("src/aedrover/")]
    report.check("README directory tree lists every exported file", not missing, str(missing))
    for md in ("README.md", "docs/RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md"):
        text = (target / md).read_text(encoding="utf-8")
        report.check(f"{md}: six foundational references with DOIs",
                     all(refs[k]["doi"] in text for k, _ in E.FOUNDATIONAL))
    manuscript = (target / "docs" / "RESEARCH_PAPER_MANUSCRIPT_BLUEPRINT.md").read_text(encoding="utf-8")
    for n in (1, 2, 3):
        report.check(f"manuscript references Figure {n}", f"figures/figure{n}_" in manuscript)
    report.check("manuscript has at least two tables", manuscript.count("**TABLE ") >= 2)
    known = {r["doi"].lower() for r in refs.values()}
    bad: list[str] = []
    for p in tree_files(target):
        if p.suffix.lower() in (".md", ".json", ".py", ".xml"):
            bad += [f"{p.relative_to(target)}: {c}" for c in unknown_dois(p.read_text(encoding="utf-8", errors="replace"), known)]
    report.check("every DOI in the export is in docs/references.json", not bad, "; ".join(bad[:5]))
    links = link_problems(target)
    report.check("relative markdown links resolve", not links, "; ".join(links[:5]))
    check_roster(target, refs, report)
    check_png(target, report)
    pend: dict[str, int] = {}
    for p in tree_files(target):
        if p.suffix.lower() in (".md", ".json"):
            n = p.read_text(encoding="utf-8").count(f"**{E.PENDING}**")
            if n:
                pend[p.relative_to(target).as_posix()] = n
    if allow_partial:
        report.check("partial export: every PENDING marker is labelled", True,
                     f"{sum(pend.values())} markers in {len(pend)} files" if pend else "none")
    else:
        report.check("strict export: no PENDING markers", not pend, str(pend))
    import mujoco

    try:
        m = mujoco.MjModel.from_xml_path(str(target / "models" / "aed_delivery_amr.xml"))
        d = mujoco.MjData(m)
        for _ in range(250):
            mujoco.mj_step(m, d)
        report.check("MJCF compiles with plain mujoco.MjModel.from_xml_path and steps", bool(pd.notna(d.qpos).all()),
                     f"nbody={m.nbody} njnt={m.njnt} nu={m.nu}")
    except Exception as exc:  # noqa: BLE001
        report.check("MJCF compiles with plain mujoco.MjModel.from_xml_path and steps", False, str(exc))
    report.check("benchmark CSV has at least 80 rows",
                 len(pd.read_csv(target / "analytics" / "aed_delivery_benchmark.csv")) >= E.MIN_CSV_ROWS)


def runtime_checks(target: Path, report: Report) -> None:
    ok, out = run([sys.executable, str(target / "src" / "test_env.py")], target, 300)
    report.check("exported src/test_env.py exits 0 with [OK] lines",
                 ok and "[OK] Python Version" in out and "[SUCCESS] Loaded and stepped model" in out,
                 out.strip()[-500:] if not ok else "")
    ok, out = run([sys.executable, str(target / "analytics" / "generate_paper_figures.py")], target, 600)
    report.check("exported analytics/generate_paper_figures.py runs", ok, out.strip()[-500:] if not ok else "")
    ok, out = run([sys.executable, str(target / "analytics" / "cardiac_survival_economics.py"), "--n", "600",
                   "--resamples", "300"], target, 600)
    report.check("exported analytics/cardiac_survival_economics.py runs", ok and "kappa" in out, out.strip()[-500:] if not ok else "")


# ---------------------------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------------------------
def merge_student_fields(new: dict, old: dict) -> dict:
    """Keep what the students already entered in the course repository's roster."""
    merged = json.loads(json.dumps(new))
    by_roll = {m.get("roll_no"): m for m in old.get("members", [])}
    for m in merged["members"]:
        prev = by_roll.get(m["roll_no"], {})
        for k in ("github_handle", "institutional_email"):
            if prev.get(k):
                m[k] = prev[k]
    show_old = old.get("project_showcase", {})
    url = show_old.get("linkedin_url", "PENDING_SUBMISSION")
    if url and url != "PENDING_SUBMISSION" and "linkedin.com" in url:
        merged["project_showcase"]["linkedin_url"] = url
        merged["project_showcase"]["submission_status"] = show_old.get("submission_status", "SUBMITTED")
    return merged


def apply_linkedin(readme: str, url: str) -> str:
    """The substitution of scripts/sync_showcase_links.py, so a merged README stays in sync."""
    readme = re.sub(LINKEDIN_IMG, f"[![Watch Video Demonstration on LinkedIn](docs/figures/video_poster.png)]({url})", readme)
    return re.sub(LINKEDIN_TXT, f"* **Video Demonstration:** [Watch 60-Second Walkthrough on LinkedIn]({url})", readme)


def git_status(nmims: Path) -> str:
    p = subprocess.run(["git", "-C", str(nmims), "status", "--porcelain", "--", E.GROUP_FOLDER],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise E.ExportError(f"git status failed in {nmims}: {p.stderr.strip()}")
    return p.stdout


def apply_export(built: Path, nmims: Path) -> list[str]:
    """Copy ``built`` into ``<nmims>/Group_03_Kashish_Vaishnavi``. Never stages or commits anything."""
    if not (nmims / ".git").exists():
        raise E.ExportError(f"{nmims} is not a git repository; --apply needs the course repository checkout")
    target = nmims / E.GROUP_FOLDER
    if not target.is_dir():
        raise E.ExportError(f"{target} does not exist; refusing to create new group folders in the course repository")
    dirty = git_status(nmims).strip()
    if dirty:
        raise E.ExportError(f"{E.GROUP_FOLDER} has uncommitted changes in the course repository; commit or discard them first:\n{dirty}")
    files = tree_files(built)
    rel_new = {p.relative_to(built).as_posix() for p in files}
    old_roster_path = target / "docs" / "TEAM_ROSTER.json"
    roster = json.loads((built / "docs" / "TEAM_ROSTER.json").read_text(encoding="utf-8"))
    readme = (built / "README.md").read_text(encoding="utf-8")
    if old_roster_path.exists():
        roster = merge_student_fields(roster, json.loads(old_roster_path.read_text(encoding="utf-8")))
        url = roster["project_showcase"]["linkedin_url"]
        if url != "PENDING_SUBMISSION":
            readme = apply_linkedin(readme, url)
    stale = sorted(p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()
                   and p.relative_to(target).as_posix() not in rel_new and "__pycache__" not in p.parts)
    for p in files:
        rel = p.relative_to(built).as_posix()
        dst = target / rel
        if rel == "docs/TEAM_ROSTER.json":
            write_text(dst, json.dumps(roster, indent=2, ensure_ascii=False) + "\n")
        elif rel == "README.md":
            write_text(dst, readme)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(p, dst)
    return stale


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Export the NMIMS Group 03 course folder from real results.")
    ap.add_argument("--out", default=str(REPO / "dist" / "nmims"), help="output root (default dist/nmims)")
    ap.add_argument("--results-dir", default=str(REPO / "results"))
    ap.add_argument("--allow-partial", action="store_true", help="fill PENDING markers for missing results")
    ap.add_argument("--accept-stale", action="store_true", help="export although results predate the current model parameters")
    ap.add_argument("--benchmark", default=None, help="explicit benchmark CSV instead of the newest benchmark_standard*.csv")
    ap.add_argument("--clinical", default=None, help="explicit clinical JSON instead of results/clinical*.json")
    ap.add_argument("--nmims-repo", default=str(default_nmims_repo()), help="course repository (read-only unless --apply)")
    ap.add_argument("--audit-script", default=None, help="path of audit_course_boundaries_and_rules.py")
    ap.add_argument("--apply", action="store_true", help="copy the finished folder into the course repository")
    ap.add_argument("--verify-citations", action="store_true", help="also run scripts/verify_citations.py (network)")
    ap.add_argument("--skip-runtime-checks", action="store_true", help="development only: skip subprocess checks")
    args = ap.parse_args(argv)

    nmims = Path(args.nmims_repo)
    out_root = Path(args.out).resolve()        # the checks below run scripts with cwd inside the export
    report = Report()
    try:
        if args.apply and args.allow_partial:
            raise E.ExportError("--apply cannot be combined with --allow-partial: a partial export would replace the students' folder with PENDING content")
        audit_path = Path(args.audit_script) if args.audit_script else nmims / "scripts" / "audit_course_boundaries_and_rules.py"
        audit = load_audit(audit_path)
        report.check("course auditor flags a planted violation (canary)", audit_canary(audit))
        refs = E.load_references(REPO)
        res = E.load_results(Path(args.results_dir), allow_partial=args.allow_partial,
                             benchmark=Path(args.benchmark) if args.benchmark else None,
                             clinical=Path(args.clinical) if args.clinical else None, accept_stale=args.accept_stale)
        if res.benchmark is None:
            raise E.ExportError("no benchmark results: analytics/aed_delivery_benchmark.csv needs at least "
                                f"{E.MIN_CSV_ROWS} real episode rows, so even a partial export needs a benchmark file")
        pf = E.package_facts()
        facts = E.arch_facts(pf)
        report.check("figures.ARCH_FACTS default equals the live package", figures.ARCH_FACTS == facts,
                     f"update figures.ARCH_FACTS to {facts}" if figures.ARCH_FACTS != facts else "")
        lar = pf["larsen"]
        report.check("Larsen 1993 coefficients are 0.67, 0.023, 0.011, 0.021",
                     (lar["intercept"], lar["cpr"], lar["defib"], lar["acls"]) == (0.67, 0.023, 0.011, 0.021), str(lar))
        target = out_root / E.GROUP_FOLDER
        clean_target(target, nmims)
        xml = E.build_snapshot_xml(SNAPSHOT_KERB_H)
        write_static(target, res, facts, xml)
        tel = make_telemetry(target, res, report)
        ctx = T.build_context(res, refs, pf, xml, tel)
        for rel, text in T.render_docs(ctx, refs).items():
            write_text(target / rel, text)
        ok, out = run([sys.executable, str(target / "analytics" / "generate_paper_figures.py")], target, 600)
        if not ok:
            raise E.ExportError("analytics/generate_paper_figures.py failed:\n" + out[-1500:])
        structural_checks(target, refs, res, report, args.allow_partial)
        if not args.skip_runtime_checks:
            runtime_checks(target, report)
        if args.verify_citations:
            ok, out = run([sys.executable, str(REPO / "scripts" / "verify_citations.py")], REPO, 600)
            report.check("scripts/verify_citations.py passes", ok, out.strip()[-400:] if not ok else "")
        purge_pycache(target)
        violations = run_audit(audit, target)
        report.check(f"course audit: 0 violations over {len(tree_files(target))} files", not violations, "; ".join(violations[:5]))
    except E.ExportError as exc:
        print(f"\nEXPORT FAILED: {exc}", file=sys.stderr)
        return 1

    # ---- report -------------------------------------------------------------------------
    print("=" * 78)
    print(f"NMIMS export: {target}")
    print("=" * 78)
    print(f"mode: {'PARTIAL (--allow-partial)' if args.allow_partial else 'strict'}; benchmark file: {res.benchmark_path.name}")
    print("result sources:")
    for s in res.sources:
        print(f"  {s.label:<32} {s.path.name}  [{s.sha12}]  {s.detail}")
    if res.pending:
        print("pending items:")
        for p in res.pending:
            print("  - " + p.splitlines()[0])
    if res.warnings:
        print("warnings:")
        for w in res.warnings:
            print("  - " + w)
    print("compliance and validation report:")
    for name, ok, detail in report.checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and (not ok or len(detail) < 90) else ""))
    print(f"files: {len(tree_files(target))}; tree fingerprint {fingerprint(target)[:16]}")
    if report.violations:
        print(f"\nEXPORT FAILED: {len(report.violations)} problem(s)", file=sys.stderr)
        for v in report.violations:
            print("  - " + v, file=sys.stderr)
        return 1
    print("EXPORT OK" + (" (partial: PENDING markers present)" if res.pending else ""))

    if args.apply:
        try:
            stale = apply_export(target, nmims)
        except E.ExportError as exc:
            print(f"\nAPPLY REFUSED: {exc}", file=sys.stderr)
            return 1
        print(f"APPLIED to {nmims / E.GROUP_FOLDER} (nothing staged or committed; review with git diff).")
        if stale:
            print("files in the course folder that are not part of the export (left untouched): " + ", ".join(stale))
    return 0


if __name__ == "__main__":
    sys.exit(main())
