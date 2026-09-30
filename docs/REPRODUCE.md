# Reproducing the results

Everything runs on CPU. The reference machine is a 24-core / 32-thread laptop CPU with 32 GB RAM;
run times below are wall-clock on that machine using 28 worker processes. Seeds are fixed, the
environment is deterministic given a seed (tested), and every result file records the seeds used.

## Environment

```bash
python -m venv .venv
.venv/Scripts/activate                       # Linux/macOS: source .venv/bin/activate
pip install -e ".[dev,viz,rl,geo]"           # rl: PyTorch + Stable-Baselines3 (CPU build is enough)
pytest -m "not slow" -n auto                 # about 200 tests, about 1 minute
python scripts/verify_citations.py           # every DOI resolves and its title matches (needs internet)
```

## Steps, in order

| Step | Command | Output | Time |
|---|---|---|---|
| Validate the physics | `python experiments/01_validate_suspension.py` then `python scripts/make_validation_doc.py` | `results/validation_*.json`, `docs/VALIDATION.md` | 1 min |
| Kerb traversability map | `python experiments/02_curb_traversability.py --design nominal` and `--design optimized` (6,048 and 1,512 trials) | `results/curb_traversability_*_standard.csv` | 2 min |
| Kerb lookup tables | `python scripts/build_curb_table.py` | `configs/curb_table.json` | seconds |
| Mechanical co-design | `python experiments/06_mech_codesign.py` | `results/codesign.json`, `configs/vehicle_optimized.yaml` | 10 min |
| Train PPO (two seeds) | `python -m aedrover.learning.train_ppo --steps 12000000 --n-envs 24 --seed 0 --out checkpoints/ppo_shielded` and `--steps 4000000 --seed 1 --out checkpoints/ppo_shielded_s1` | `checkpoints/`, `runs/*.log` | 69 and 23 min |
| Auto-updating PPO findings | `python scripts/watch_ppo.py --run ppo_shielded_s1 --follow` | `docs/PPO_TRAINING*.md`, curves | runs alongside |
| Real route geometry | `python experiments/08_osm_routes.py` (uses the cached OSM download; `--refresh` re-downloads) | `data/osm/` | 1 min cached |
| Everything else | `python scripts/run_pipeline.py` | see below | a few hours |
| Report and figures | `python scripts/make_report.py` | `docs/RESULTS.md`, `docs/figures/` | 1 min |
| Course export | `python scripts/export_nmims.py --out dist/nmims` | `dist/nmims/` | 1 min |

`scripts/run_pipeline.py` is resumable (a step is skipped when its output exists; `--force` re-runs it):

1. `select_ppo` picks the PPO checkpoint on validation seeds (50000+).
2. `tune_mppi` selects MPPI settings on tuning seeds (30000+) and freezes them in `configs/mppi_tuned.json`.
3. `benchmark` runs pure pursuit, potential field, dynamic window, MPPI and PPO on 100 paired seeds (5000+) per family.
4. `ood` evaluates the same controllers inside and outside the training distribution (seeds 20000+).
5. `ablations` covers vehicle design, safety filter and speed cap (seeds 40000+).
6. `clinical` composes the simulated segments into routes and evaluates survival.

Evaluation, tuning, validation and ablation seed ranges are disjoint by construction, so nothing is tuned
on evaluation data.

## Compute notes

* MPPI is the expensive controller (one 10 Hz plan is 128 rollouts of 2 s each); everything else is cheap.
  MPPI rollouts run single-threaded per worker process, with parallelism across episodes.
* GPU acceleration is deliberately out of scope for this version: the simulator is small enough that CPU
  throughput (about 60x real time per core, 250-320x aggregate) is not the bottleneck.
