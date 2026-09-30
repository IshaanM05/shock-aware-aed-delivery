# Handoff: rich-world mode and drone visuals

This file is for whoever picks up the two next tasks. Read it top to bottom once; sections 1 to 3 are context,
sections 4 and 5 are the tasks, section 6 is what "done" means.

## 1. State of the repository (2026-10-01)

Public repo: https://github.com/IshaanM05/shock-aware-aed-delivery (branch `main`, Python package `aedrover`).

| Area | State |
|---|---|
| Simulator (`src/aedrover/sim`) | Done. Compile-once MJCF world, mocap slots, 50 Hz control. Episodes are deterministic and independent of job order since the inertia fix (engineering note 15). |
| Controllers (`nav`, `learning`, `control`) | Done: pure pursuit, potential field, dynamic window, MPPI, PPO, swept-footprint safety filter. PPO policy is committed in `models/ppo_selected`. |
| Experiments and results | Done, but **computed before the inertia fix** (see section 2, item 1). |
| Clinical and drone layers | Done (`clinical`, `drone`), numbers in `results/clinical*.csv` and `docs/DRONE_COMPARATOR.md`. |
| Cinematic renderer (`src/aedrover/viz`) | Done: recorded episodes replayed through MuJoCo 3.14's PBR (Filament) renderer. See `docs/RENDERING.md`. |
| Media | `assets/showcase.mp4` (68 s, 1080p60, 22 MB), `assets/hero.gif`, `assets/showcase_poster.jpg`. Master (134 MB) is in `.cache/`, ignored. |
| Live views | `scripts/live_cinematic.py` (dressed scene, plays back a freshly simulated episode in real time) and `scripts/live_viewer.py` (MuJoCo's plain viewer). |
| Tests | About 235. `pytest -m "not slow and not gpu" -n auto` is the CI set; `gpu` tests run locally only. |
| NMIMS export | `scripts/export_nmims.py` builds `dist/nmims/Group_03_Kashish_Vaishnavi` and passes the course audit. It has **not** been applied to the course repo. It predates the inertia fix and the new assets. |

## 2. Things to know before you start

1. **Open decision: re-run the pipeline.** Results in `results/` were produced while `World.set_payload_mass`
   rescaled inertia in place, so individual episodes can differ from a fresh replay (same seed, different history).
   Each row is still a valid sample and conclusions are expected to hold, but that is unverified. Re-running
   (`python scripts/run_pipeline.py --force`, about 4.5 hours on a 24-core laptop, then `scripts/make_report.py`,
   `scripts/make_osm_doc.py`, README numbers, `scripts/export_nmims.py`) would make every row reproducible. Ask the
   project owner before doing it: it moves every published number slightly.
2. **Constraints that must hold.**
   * No mention of any AI assistant, its vendor, or co-author trailers in commits, PR text or repo files.
     Check: `git log --format=%B | grep -ci "claude\|anthropic\|co-authored"` prints 0.
   * Never commit to or push to the NMIMS course repo, and never `git add -A` there, without the owner's explicit go-ahead.
   * The standard environment, its seeds and `results/` are the benchmark. New work must not change them (prove it
     with a test, see section 6).
   * Every number shown in media, README or docs comes from `results/` or from code, never typed by hand. Captions
     must claim only what the data supports (for example "one scenario, not a statistic").
   * NMIMS export rules: no currency tokens, no emoji, exactly six verified DOIs in the roster.
3. **Environment.** Python 3.11 venv at `.venv` (`pip install -e ".[dev,viz,rl,geo]"`), MuJoCo 3.14, an NVIDIA GPU with
   OpenGL for the PBR renderer. Simulation and training are CPU-only. CI runners have no GPU.
4. **Renderer quirks that cost hours** are all in `docs/RENDERING.md`. The ones most likely to bite again:
   * The depth material is shared between models; calibrate the depth decoder before any scene does a depth pass,
     or Filament aborts the whole process (handled in `FilamentBackend.__init__`).
   * `geom_rgba` and `geom_size` changes do not reach the renderer per frame; mocap poses and material colour or
     emission do. Overlays are therefore pools of bead-spheres on mocap bodies.
   * `ModelDecorations.update` cannot be called from Python.
   * Filament aborts at teardown if objects are destroyed out of order; scripts end with `os._exit(0)`. Run scripts
     with `python -u` so buffered output is not lost if the process aborts.
5. **File sizes.** GitHub rejects files over 100 MB and warns over 50 MB. Keep the committed film near 25 MB
   (`render_showcase.py` re-encodes the master at crf 28) and GIFs under about 5 MB.
6. **Rendering cost** on the reference laptop: a finished 1080p frame costs 0.2 to 0.3 s all-in; the 68 s film takes
   11 to 17 minutes. Recordings are cached under `.cache/recordings`.

## 3. Architecture in one page

* `World` (`sim/world.py`): one MJCF compiled once; kerb slabs, ramps, obstacles and pedestrians are mocap bodies with
  fixed compile-time sizes that are moved, never resized (engineering notes 1 to 4).
* `AEDRoverEnv` (`sim/env.py`): steps the rover, pedestrians (social force), perception (lidar by `mj_multiRay`
  over the geom groups in `sim/sensors.py`) and metrics. Collision and outcome detection use an analytic clearance to
  pedestrians and the scenario's obstacle list (`_clearance`, `_check_done`), not MuJoCo contacts.
* `viz`: `recording.py` (record an episode), `render_model.py` (dressed copy of the physics MJCF, replay, backends),
  `dressing.py` / `people.py` / `rover_visuals.py` / `overlays.py` (visual-only content and animators),
  `post.py` (haze, depth of field, bloom, grade), `film.py` / `shots.py` / `hud.py` (cinematography).
  The render model adds visual-only geoms after every physics element, so all indices stay valid.

## 4. Task A: rich-world mode (collidable street, truly live)

**Goal.** An opt-in mode in which parked vehicles and motorbikes, stalls, lamp posts and tree trunks are real
obstacles, and a live run steps the simulation and the renderer together, with the dressed look. The user wants to be
able to launch a live MuJoCo demo at any time and see the rendered world, with the dressing physically affecting the
run. It must never change the standard benchmark.

**Design decisions to make (recommendations in brackets).**
1. *Where the obstacles live.* Street furniture positions depend only on the scenario seed, so [compile a per-scenario
   model with the furniture as static collidable geoms] instead of mocap slots; the compile-once rule exists for the
   benchmark's speed and is not needed here. Keep it behind a new flag (for example `AEDRoverEnv(rich=True)` building
   its MJCF via a separate builder) and leave `build_xml` output for the default world byte-identical.
2. *Keep the corridor feasible.* The lateral corridor limit is `CORRIDOR_HALF_WIDTH = 1.4 m` (`sim/scenario.py`). Place
   furniture outside it, or leave a guaranteed clear path, or scenarios become unsolvable. Decide whether some items
   (a parked bike, a stall) may intrude and require the planner to route around them.
3. *Outcome and clearance.* `_clearance` and `_check_done` ignore anything not in `crowd` or `scenario.obstacles`. Extend them
   to cover the rich obstacles (analytic boxes or cylinders are enough), or detect collisions from `data.contact`.
   Without this the rover will stall against a bike and be reported as a "stall", not a collision.
4. *Perception.* Lidar already sees every collidable geom in the groups it masks (`_GROUP_MASK` in `sim/sensors.py`), so
   the new obstacles are visible to the safety filter, DWA, APF and PPO observations. Check the PPO policy still behaves
   (it was trained without them; expect degraded results and report them honestly as out of distribution).
5. *MPPI's internal model.* MPPI plans on its own `World` copy (`nav/mppi.py`, `_pw`) that knows the scenario's
   obstacles only. Either build it with the same rich geometry or document that MPPI is blind to furniture in this mode.
6. *Live stepping.* `RenderScene` takes a whole `Recording`. For live use, add a streaming path:
   * copy `data.qpos` and mocap arrays from the live env into the render model each frame (same indices as today);
   * make `PedAnimator` streaming: heading from a causal low-pass of velocity, stride phase accumulated as you go,
     instead of arrays precomputed from the full episode;
   * the "actual upcoming path" ribbon needs the future; for live use draw the controller's plan instead (MPPI rollouts
     via `ctrl.capture`, DWA's chosen arc) or drop it.
7. *Window.* `scripts/live_cinematic.py` shows frames in a Tk window at about 15 to 20 fps without haze or bloom.
   A streaming version can reuse it; consider running the simulation in a thread so the GPU and the CPU overlap.

**Suggested order.** (a) rich MJCF builder and a scenario-to-furniture placement function, with tests that nothing is
inside the corridor; (b) extend clearance and outcome detection; (c) a rich variant of the dressing that shares the same
geometry (so what you see is what collides, one source of truth for furniture placement); (d) streaming render path;
(e) `scripts/live_cinematic.py --rich` and README and `docs/RENDERING.md` updates; (f) a small rich-world evaluation, clearly
labelled as not comparable with `results/`.

**Acceptance.** See section 6, plus: the rover is physically stopped by a parked car; lidar returns hit it; the episode
outcome reports a collision when expected; the live window runs a full episode without a pre-simulation pass.

## 5. Task B: drone visuals and a rover-vs-drone shot

**Goal.** Draw the simulated AED quadrotor with the same quality as the rover and add a shot (or short sequence) of the
rover and the drone dispatched together, with captions driven by `results/`.

**What exists.** `src/aedrover/drone`: `quadrotor_mjcf.py` (`QuadParams`, `build_mjcf`, `QuadSim`), `flight_ctrl.py`,
`wind.py`, `energy.py`, `mission.py` (`simulate_mission`, `mission_time_s`, `time_to_scene_s`, `can_reach`,
`p_available`). `docs/DRONE_COMPARATOR.md` lists every parameter and which are assumptions (60 s launch latency,
10 m/s wind limit, 15 m/s cruise, 50 m altitude). The clinical comparison is in `results/clinical_dispatch_policies.csv`
and `results/clinical_survival_vs_radius.csv`.

**Design.**
1. *Keep the architecture: record, then replay.* The drone has its own MJCF and simulator, so do not try to merge it into
   the rover's physics. Simulate a mission with `QuadSim` (`simulate_mission`), store its pose over time (position,
   quaternion, rotor speeds if available), then draw it in the rover's render model as **mocap-driven visual parts**
   (body, four arms, four rotors, AED payload, status light), the same way `people.py` poses pedestrian limbs. Add a
   `DroneAnimator` with `bind` and `apply` like `PedAnimator`. Spin the rotors by rotating their mocap bodies.
2. *Time alignment.* The drone flies a straight line at altitude to the patient while the rover drives the ground route.
   Build one timeline: dispatch at t = 0, drone launch after the assumed latency, rover sets off immediately. For the
   film, time-compress the long parts and show a clock. Use the recorded rover episode for the ground route and the drone
   mission for the air route; both are deterministic.
3. *Scene geometry.* The rover scene is a 3 m corridor with a road crossing; the drone flies over the dressed buildings.
   Choose the patient location (for example the rover's goal, `scenario.x_goal`) and a launch point beyond the
   skyline, and keep the drone above the building heights (up to about 21 m) or route it along the street.
4. *Captions and numbers.* Show arrival times and survival from `results/clinical_*.csv` (radius 500 m or 1 km, PPO rover,
   Larsen model, drone availability as the parameter). State that launch latency and wind limit are assumptions, and that
   the drone is an upper bound (see `docs/DRONE_COMPARATOR.md`). Do not tune the drone to lose or to win.
5. *Film integration.* Add a `Shot` (`viz/film.py`) with a new rig if needed (`viz/shots.py`: a high tracking shot
   following both vehicles, or a split view), register the recording in `scripts/render_showcase.py`, and keep the film
   under 90 s and the file near 25 MB. Update the README caption and `docs/RENDERING.md`.

**Suggested order.** (a) a `record_drone_mission` helper producing the pose arrays and a small `DroneRecording` dataclass
with save and load (mirror `viz/recording.py`); (b) drone visuals and `DroneAnimator`, checked on stills; (c) a
two-vehicle timeline and camera; (d) HUD additions (two clocks) and captions; (e) film shot, hero GIF decision, docs, tests.

**Acceptance.** See section 6, plus: drone visuals follow the simulated pose exactly (test the animator against the
recorded arrays); the shown arrival times equal `mission_time_s` plus latency and the rover's recorded time; captions
contain only values read from files.

## 6. Definition of done for either task

* `pytest -m "not slow and not gpu" -n auto` passes; GPU tests pass locally (`pytest -m gpu`); `ruff check .` is clean.
* **Benchmark untouched:** a test that runs one standard job (for example dynamic window, mixed family, seed 5010 at a 2.0 m/s cap) and
  asserts outcome, time and peak shock equal fixed reference values, so any accidental change to the default world fails it.
* New code lives in its own modules or behind flags; the default `build_xml` and `AEDRoverEnv` behaviour is unchanged.
* Docs updated (`docs/RENDERING.md`, README quick start, this file's status), and every claim in captions traceable to a file.
* Commits are small, plain-language, with no assistant or co-author attribution; push to `main`; check the GitHub Actions run
  afterwards (`gh run list`), because the hosted runners have no GPU.
* Media stays under the size limits in section 2.

## 7. Suggested sequencing

1. Decide about the pipeline re-run (section 2, item 1) and, if yes, start it first; it is CPU-only and runs unattended.
2. Task B first (smaller, self-contained, extends the film). Task A second (touches simulator-adjacent code, needs the most care).
3. While the pipeline runs, GPU work (rendering) can proceed, but expect slower frames because both compete for the CPU.
