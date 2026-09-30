"""Pick the PPO policy to benchmark by validation performance, never by test performance.

Candidates are every finished run's final policy and every checkpoint that has its observation
normalisation stats. Each is evaluated on validation seeds (50000+, disjoint from all benchmark, OOD
and ablation seeds). The best safe-delivery rate wins, ties broken by collisions then time. The winner
is copied to ``checkpoints/ppo_selected`` so every later experiment uses one fixed path.

    python scripts/select_ppo.py --n 20
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from aedrover.analysis.report import delivery_success
from aedrover.learning.evaluate import evaluate_policy

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"


def candidates() -> list[tuple[str, str]]:
    out = []
    for run in sorted(p for p in CKPT.iterdir() if p.is_dir() and p.name.startswith("ppo_shielded")):
        if (run / "final.zip").exists() and (run / "vecnormalize.pkl").exists():
            out.append((run.name, "final"))
        for z in sorted(run.glob("ppo_*_steps.zip")):
            stats = run / z.name.replace("ppo_", "ppo_vecnormalize_", 1).replace(".zip", ".pkl")
            if stats.exists():
                out.append((run.name, z.stem))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=50000)
    ap.add_argument("--families", nargs="+", default=["kerb", "crowded", "mixed", "slippery"])
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args()

    rows = []
    for run, weights in candidates():
        df = evaluate_policy(str(CKPT / run), n=args.n, seed0=args.seed0, families=args.families, references=(),
                             both_shield_modes=False, workers=args.workers, weights=weights)
        df["safe"] = delivery_success(df)
        rows.append({"run": run, "weights": weights, "n": len(df), "safe": df.safe.mean(), "success": df.success.mean(),
                     "collision": (df.outcome == "collision").mean(), "time_med": df[df.success].time_s.median()})
        print(rows[-1], flush=True)
    tab = pd.DataFrame(rows).sort_values(["safe", "collision", "time_med"], ascending=[False, True, True])
    (ROOT / "results").mkdir(exist_ok=True)
    tab.to_csv(ROOT / "results" / "ppo_selection.csv", index=False)
    best = tab.iloc[0]
    dst = CKPT / "ppo_selected"
    dst.mkdir(exist_ok=True)
    src = CKPT / best.run
    if best.weights == "final":
        shutil.copy2(src / "final.zip", dst / "final.zip")
        shutil.copy2(src / "vecnormalize.pkl", dst / "vecnormalize.pkl")
    else:
        shutil.copy2(src / f"{best.weights}.zip", dst / "final.zip")
        shutil.copy2(src / (best.weights.replace("ppo_", "ppo_vecnormalize_", 1) + ".pkl"), dst / "vecnormalize.pkl")
    (dst / "selection.json").write_text(json.dumps({**{k: (v if not hasattr(v, "item") else v.item()) for k, v in best.items()},
                                                    "validation_seeds": [args.seed0, args.seed0 + args.n - 1]}, indent=2),
                                        encoding="utf-8")
    print(tab.round(3).to_string(index=False))
    print("selected", best.run, best.weights, "-> checkpoints/ppo_selected")


if __name__ == "__main__":
    main()
