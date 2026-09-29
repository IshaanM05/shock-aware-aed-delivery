# Convenience targets. On Windows use the equivalent commands from the README (PowerShell).
PY ?= python

.PHONY: install test lint validate kerbs codesign benchmark-smoke benchmark train-ppo demo report export-nmims

install:
	$(PY) -m pip install -e ".[dev,viz,rl]"

test:
	$(PY) -m pytest -m "not slow" -n auto

lint:
	$(PY) -m ruff check src tests scripts experiments

validate:            # analytic checks of the MuJoCo model (suspension, rolling, stability)
	$(PY) experiments/01_validate_suspension.py

kerbs:               # kerb traversability map (RQ1)
	$(PY) experiments/02_curb_traversability.py --preset standard --design optimized

codesign:            # mechanical co-design search (RQ1), about 10 minutes on 28 cores
	$(PY) experiments/06_mech_codesign.py

benchmark-smoke:     # a few minutes
	$(PY) experiments/03_controller_benchmark.py --n 10 --tag smoke

benchmark:           # the full study (see docs/REPRODUCE.md for the exact commands and run times)
	$(PY) experiments/03_controller_benchmark.py --n 100 --tag standard --controllers pure_pursuit apf dwa mppi ppo

train-ppo:
	$(PY) -m aedrover.learning.train_ppo --steps 12000000 --n-envs 24 --out checkpoints/ppo_shielded

demo:
	$(PY) scripts/render_demo.py --controller mppi --family mixed --seed 1002 --name mixed_mppi

report:
	$(PY) scripts/make_report.py

export-nmims:
	$(PY) scripts/export_nmims.py --out dist/nmims
