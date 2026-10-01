# Handoff: state of the repository

The two tasks the previous version of this file planned, the drone visuals with a rover-versus-drone shot and the opt-in
rich-world mode with a truly live view, are done (2026-10-01). This file says what exists, what to know before touching it,
and what could come next. Sections 1 to 3 are context, sections 4 and 5 describe the two pieces of work, section 6 is what
"done" means, section 7 lists open items.

## 1. State of the repository (2026-10-01)

Public repo: https://github.com/IshaanM05/shock-aware-aed-delivery (branch `main`, Python package `aedrover`).

| Area | State |
|---|---|
| Simulator (`src/aedrover/sim`) | Done. Compile-once MJCF world, mocap slots, 50 Hz control. Episodes are deterministic and independent of job order since the inertia fix (engineering note 15). An opt-in rich world (`AEDRoverEnv(rich=True)`) adds collidable street furniture (section 4). |
| Controllers (`nav`, `learning`, `control`) | Done: pure pursuit, potential field, dynamic window, MPPI, PPO, swept-footprint safety filter. PPO policy is committed in `models/ppo_selected`. MPPI plans in a world with the rich furniture when there is any. |
| Experiments and results | Done, but **computed before the inertia fix** (section 2, item 1). `experiments/08_rich_world_eval.py` is a separate small study in the rich world (`results/rich_world/`, `docs/RICH_WORLD.md`), not comparable with `results/`. |
| Clinical and drone layers | Done (`clinical`, `drone`), numbers in `results/clinical*.csv` and `docs/DRONE_COMPARATOR.md`. The clinical model times its drone more simply than the drone simulation (section 5). |
| Cinematic renderer (`src/aedrover/viz`) | Done: recorded episodes replayed through MuJoCo 3.14's PBR (Filament) renderer, the AED drone, the rover-versus-drone dispatch shot, and a streaming path for live use. See `docs/RENDERING.md`. |
| Media | `assets/showcase.mp4` (87 s, 1080p60, 27 MB), `assets/hero.gif`, `assets/showcase_poster.jpg`. Master (172 MB) is in `.cache/`, ignored. |
| Live views | `scripts/live_cinematic.py` (the simulation and the dressed PBR renderer run together; `--rich`, `--drone` for the rover and a live drone, `--replay`, `--record` for a windowless video) and `scripts/live_viewer.py` (MuJoCo's plain viewer; `--rich`). |
| Tests | About 320. `pytest -m "not slow and not gpu" -n auto` is the CI set; `gpu` tests run locally only. |
| NMIMS export | `scripts/export_nmims.py` builds `dist/nmims/Group_03_Kashish_Vaishnavi` and passes the course audit. It has **not** been applied to the course repo. It predates the inertia fix and the new assets. |

## 2. Things to know before you start

1. **Open decision: re-run the pipeline.** Results in `results/` were produced while `World.set_payload_mass`
   rescaled inertia in place, so individual episodes can differ from a fresh replay (same seed, different history).
   Each row is still a valid sample and conclusions are expected to hold, but that is unverified. Re-running
   (`python scripts/run_pipeline.py --force`, about 4.5 hours on a 24-core laptop, then `scripts/make_report.py`,
   `scripts/make_osm_doc.py`, README numbers, `scripts/export_nmims.py`) would make every row reproducible. On 2026-10-01 the owner
   chose to skip it for now. Ask before doing it: it moves every published number slightly.
2. **Constraints that must hold.**
   * No mention of any AI assistant, its vendor, or co-author trailers in commits, PR text or repo files. Before pushing,
     read `git log --format=%B` and confirm no commit carries such a line.
   * Never commit to or push to the NMIMS course repo, and never `git add -A` there, without the owner's explicit go-ahead.
   * The standard environment, its seeds and `results/` are the benchmark. New work must not change them;
     `tests/test_benchmark_lock.py` hashes the default world XML and pins one standard episode.
   * Every number shown in media, README or docs comes from `results/` or from code, never typed by hand. Captions
     must claim only what the data supports (for example "one scenario, not a statistic").
   * NMIMS export rules: no currency tokens, no emoji, exactly six verified DOIs in the roster.
3. **Environment.** Python 3.11 venv at `.venv` (`pip install -e ".[dev,viz,rl,geo]"`), MuJoCo 3.14, an NVIDIA GPU with
   OpenGL for the PBR renderer. Simulation and training are CPU-only. CI runners have no GPU.
4. **Platforms differ in contact dynamics** (engineering note 17). The same standard job gives 24.7 s on Windows and 23.3 s on Linux
   and still reaches the goal. The lock test checks the exact reference on Windows only; do not expect benchmark rows to be bitwise
   reproducible across operating systems.
5. **Renderer quirks that cost hours** are all in `docs/RENDERING.md`. The ones most likely to bite again:
   * The depth material is shared between models; calibrate the depth decoder before any scene does a depth pass,
     or Filament aborts the whole process (handled in `FilamentBackend.__init__`).
   * `geom_rgba` and `geom_size` changes do not reach the renderer per frame; mocap poses and material colour or
     emission do. Overlays are therefore pools of bead-spheres on mocap bodies, and the drone's status light is two parked-or-shown
     spheres.
   * `ModelDecorations.update` cannot be called from Python.
   * Filament aborts at teardown if objects are destroyed out of order; scripts end with `os._exit(0)`. Run scripts
     with `python -u` so buffered output is not lost if the process aborts.
   * Small far objects (a 0.9 m drone beyond 50 to 100 m) fall out of the depth-based haze and turn into sky; keep the camera close.
6. **File sizes.** GitHub rejects files over 100 MB and warns over 50 MB. Keep the committed film near 25 to 28 MB
   (`render_showcase.py` re-encodes the master at crf 28) and GIFs under about 5 MB. The hero GIF and poster are cut by shot name,
   so inserting shots does not move them; re-rendering does not change them either, so do not recommit them.
7. **Rendering cost** on the reference laptop: a finished 1080p frame costs 0.2 to 0.3 s all-in; the 87 s film takes about 15 minutes.
   Recordings are cached under `.cache/recordings`.
8. **Shell quirk** when scripting edits: very long heredocs that mix many quote styles can be rejected by the tool wrapper; write such
   files with an editor tool instead.

## 3. Architecture in one page

* `World` (`sim/world.py`): one MJCF compiled once; kerb slabs, ramps, obstacles and pedestrians are mocap bodies with
  fixed compile-time sizes that are moved, never resized (engineering notes 1 to 4). Obstacle slots are compiled non-colliding
  (note 16). `build_xml_rich` / `World(furniture=...)` add static furniture proxies; without furniture the XML is the default, byte for byte.
* `AEDRoverEnv` (`sim/env.py`): steps the rover, pedestrians (social force), perception (lidar by `mj_multiRay`
  over the geom groups in `sim/sensors.py`) and metrics. Collision and outcome detection use an analytic clearance to
  pedestrians, the scenario's obstacle list and, in rich mode, the furniture boxes (`_clearance`, `_check_done`).
* `sim/furniture.py`: `place_furniture(scenario)` (pure, own random stream, feasibility by construction), `FurnitureItem`,
  `footprint_gap` (rover footprint versus boxes), `cover_discs` (for MPPI).
* `viz`: `recording.py` (record an episode), `render_model.py` (dressed copy of the physics MJCF, replay, `set_live`, backends),
  `dressing.py` / `people.py` / `rover_visuals.py` / `overlays.py` (visual-only content and animators),
  `drone_recording.py` / `drone_visuals.py` / `dispatch.py` (the drone and the dispatch shot), `live.py` (streaming path),
  `post.py` (haze, depth of field, bloom, grade), `film.py` / `shots.py` / `hud.py` (cinematography).
  The render model adds visual-only geoms after every physics element, so all indices stay valid.

## 4. Rich-world mode and the live view (done 2026-10-01)

**What exists.** `AEDRoverEnv(rich=True)` compiles each scenario's street furniture into the physics: parked cars, rows of motorbikes, stalls, lamp
posts and tree trunks as static collision boxes from the road up. The lidar sees them, the clearance and outcome count them
(`collision_kind = "furniture"`), MPPI plans in a world with the same boxes, pedestrians walk round them, and the dressing draws every
item exactly on its boxes (`Recording.meta["furniture"]`). `scripts/live_cinematic.py` steps the simulation and draws it together with
no pre-simulation pass (`--rich` for the street furniture, `--replay` for the old behaviour). Details are in `docs/RENDERING.md`; the
paired evaluation is in `docs/RICH_WORLD.md`.

**Decisions and why.**

* Per-scenario compile, not mocap pools: arbitrary box sizes, and the compile cost (about 14 ms) is irrelevant outside the benchmark.
  `env.world`, `env.rover` and `env.perc` are therefore rebuilt in `reset` when the street changes; anything that held them across a
  reset (a viewer) must read them again. `live_viewer.py --rich` opens a window per episode for this reason.
* Furniture may encroach on the footway by at most 1.2 m, so a 1.6 m lane always stays free; opposite-side items keep 4 m apart; nothing in the
  start zone, goal zone or road crossing, or within 3 m of a scenario bollard. `tests/test_furniture.py` checks this over 200 scenarios. The
  dressing's own random stream is entangled with the film, so the rich placement has its own function and stream and the standard dressing
  is untouched (hash-tested).
* Collision proxies reach from the road up, because a lidar at 0.33 m misses a motorbike frame and a box with its underside above the road
  made the rover climb and bounce.
* The analytic footprint matches the physical contact to about 1 cm (`test_driving_into_furniture_ends_in_a_furniture_collision_at_the_moment_of_contact`).
* Live rendering has no future, so the recorded "path ahead" ribbon is not drawn live; MPPI's sampled rollouts are.

**Live drone.** `--drone` adds the quadrotor, simulated live (`drone.mission.MissionStepper`, same mission as `simulate_mission`) and drawn by
`LiveDroneAnimator`, launched from the rover's start one metre to its right toward the same goal. The mission is scaled to the 36 m street (6 m
cruise altitude), not the 1 km clinical case; the HUD footnote says so. A camera behind the rover could not keep the small drone in frame, so
the live camera is at the goal looking back (`camera.watch_both`).

**Limits.** Nothing moves or falls over; PPO is out of distribution in this world
(it never saw furniture) and `docs/RICH_WORLD.md` says so; MPPI cannot hold real time and the live HUD shows its speed factor; there is no worker
thread (the simulation costs about 20 ms per simulated second except for MPPI, so it did not seem to be needed).

## 5. Drone visuals and the rover-versus-drone shot (done 2026-10-01)

**What exists.**

* `drone/mission.py`: `simulate_mission(..., trace=[])` appends `(t, pos, quat, rotor thrusts)` per control tick; with
  `trace=None` nothing changes.
* `viz/drone_recording.py`: `DroneRecording` (save, load, `pose_at` with clamping, so the drone sits before liftoff and hovers
  after release), `record_drone_mission`, `load_or_record`.
* `viz/drone_visuals.py`: `dress_drone`, `DroneAnimator` (settable clock `t`, `active`), `Placement` (rigid transform of the
  flight into the street), `DroneSpec`. `dress_scene(..., drone=spec, clear_flight_corridor=...)` draws it; both options are off
  by default and `tests/test_drone_viz.py` hashes the default dressing to prove nothing else moved.
* `viz/dispatch.py`: `DispatchNumbers` / `load_numbers` (everything the shot states, from code and `results/`),
  `DispatchShot` (one beat on the shared dispatch clock), `fit_rate`, rigs `rig_aerial_follow`, `rig_head_on`, `rig_arrival`,
  `RoverHider`. `viz/hud.py` has `draw_dispatch_hud`; `Film.render` accepts any item with `frames_from(film)`.
* `scripts/render_showcase.py` (`dispatch_items`): two beats between the MPPI rollout shot and the results card, 87 s film, 27 MB.

**Decisions made with the owner, and why.**

* The rover scene is a 36 m segment, so the film cannot show a 1 km flight. The 3D view shows the drone's last approach and the rover's
  arrival; the HUD carries the true distances and one dispatch clock. After beat 1 there is a deliberate time skip.
* 1 km, PPO, 4 crossings per km, Larsen model (the rows the README quotes). Rover arrival is the median travel time of the routes it
  delivers safely (45.5% of routes); the drone arrives at `time_to_scene_s(1000)` = 166 s in still air.
* **The clinical CSVs do not use `aedrover.drone`.** They time the drone as 30 s launch plus a straight 15 m/s flight, 1.16 min faster
  than the simulated flight. The film clocks use the simulation, quote survival from the CSV, and say so on screen. Using the simulated
  timing in the clinical model gives 28.5% instead of 29.7% at 1 km (`docs/DRONE_COMPARATOR.md`, section 9). The README sentence that
  attributed the 60 s latency to the published numbers was corrected.

## 6. Definition of done for any further work

* `pytest -m "not slow and not gpu" -n auto` passes; GPU tests pass locally (`pytest -m gpu`); `ruff check src tests scripts experiments` is clean.
* **Benchmark untouched:** `tests/test_benchmark_lock.py` passes (default world XML hash; the exact reference episode on Windows).
* New code lives in its own modules or behind flags; the default `build_xml` and `AEDRoverEnv` behaviour is unchanged.
* Docs updated (`docs/RENDERING.md`, README quick start, this file's status), and every claim in captions traceable to a file.
* Commits are small, plain-language, with no assistant or co-author attribution; push to `main`; check the GitHub Actions run
  afterwards (`gh run list`), because the hosted runners have no GPU and run Linux.
* Media stays under the size limits in section 2.

## 7. Open items and ideas

1. The pipeline re-run (section 2, item 1), then `make_report.py`, `make_osm_doc.py`, the README numbers and the NMIMS export. The NMIMS
   export also predates the new assets and the drone and rich-world code.
2. Make the clinical model use the simulated drone timing, or state the gap wherever the 29.7% appears (it is in the README and the
   DRONE_COMPARATOR document now).
3. Film: a 500 m variant of the dispatch section (one parameter), wind and gusts in the recorded flight, an ambulance marker on the timeline.
4. Rich world: a worker thread so the GPU and the CPU overlap for MPPI; a harder layout level (a real chicane) for planners; furniture that
   moves.
5. Obstacle slots are analytic-only (note 16); whether bollards and planters should be physical would change every benchmark episode, so it
   is a decision for the owner together with the re-run.
