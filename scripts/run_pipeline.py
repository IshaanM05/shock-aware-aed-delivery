"""Run the full evaluation pipeline as one resumable job.

Each step writes its outputs under results/ and its log under runs/pipeline_<step>.log. A step whose
marker output already exists is skipped (``--force`` re-runs it), so the job can be stopped and
restarted at any point.

    python scripts/run_pipeline.py                       # everything
    python scripts/run_pipeline.py --only benchmark ood  # selected steps
    python scripts/run_pipeline.py --n-bench 30 --n-ood 20 --n-abl 20 --n-tune 6   # quick dress rehearsal
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def steps(a: argparse.Namespace) -> list[tuple[str, list[str], str]]:
    """(name, command, marker file whose existence means the step is done)."""
    tag = a.tag
    return [
        ("select_ppo", [PY, "scripts/select_ppo.py", "--n", str(a.n_select)], "checkpoints/ppo_selected/selection.json"),
        ("tune_mppi", [PY, "experiments/tune_mppi.py", "--n", str(a.n_tune)], "configs/mppi_tuned.json"),
        ("benchmark", [PY, "experiments/03_controller_benchmark.py", "--n", str(a.n_bench), "--seed0", "5000", "--n-mppi", str(max(1, int(a.n_bench * a.mppi_fraction))), "--tag", tag,
                       "--controllers", "pure_pursuit", "apf", "dwa", "mppi", "ppo", "--ppo-path", "checkpoints/ppo_selected"],
         f"results/benchmark_{tag}.csv"),
        ("ood", [PY, "experiments/04_ood_generalization.py", "--n", str(a.n_ood), "--n-mppi", str(max(1, int(a.n_ood * a.mppi_fraction))), "--tag", tag, "--controllers", "dwa", "mppi",
                 "ppo", "--ppo-path", "checkpoints/ppo_selected"], f"results/ood_{tag}.csv"),
        ("ablations", [PY, "experiments/07_ablations.py", "--n", str(a.n_abl), "--n-mppi", str(max(1, int(a.n_abl * a.mppi_fraction))), "--tag", tag, "--controllers", "dwa", "mppi",
                       "ppo", "--ppo-path", "checkpoints/ppo_selected"], f"results/ablation_speed_cap_{tag}.csv"),
        ("clinical", [PY, "experiments/05_clinical_analysis.py", "--bench", f"results/benchmark_{tag}.csv", "--controllers",
                      "dwa", "mppi", "ppo"], "results/clinical.json"),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tag", default="standard")
    ap.add_argument("--n-select", type=int, default=20)
    ap.add_argument("--n-tune", type=int, default=20)
    ap.add_argument("--n-bench", type=int, default=100)
    ap.add_argument("--n-ood", type=int, default=50)
    ap.add_argument("--n-abl", type=int, default=40)
    ap.add_argument("--mppi-fraction", type=float, default=0.5, help="MPPI gets this fraction of the seeds (it is ~100x costlier)")
    a = ap.parse_args()
    (ROOT / "runs").mkdir(exist_ok=True)
    t_all = time.perf_counter()
    for name, cmd, marker in steps(a):
        if a.only and name not in a.only:
            continue
        if (ROOT / marker).exists() and not a.force:
            print(f"[skip] {name} (found {marker})", flush=True)
            continue
        print(f"[run ] {name}: {' '.join(cmd[1:])}", flush=True)
        t0 = time.perf_counter()
        with open(ROOT / "runs" / f"pipeline_{name}.log", "w", encoding="utf-8") as log:
            rc = subprocess.call(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        print(f"[{'done' if rc == 0 else 'FAIL'}] {name} in {(time.perf_counter() - t0) / 60:.1f} min (rc={rc})", flush=True)
        if rc != 0:
            raise SystemExit(f"step {name} failed, see runs/pipeline_{name}.log")
    print(f"pipeline finished in {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
