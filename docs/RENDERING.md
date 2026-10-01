# Cinematic rendering

The simulator draws with a plain grey checker floor. For demos, the README and the showcase film the
project has a second, optional rendering layer that turns any recorded episode into a golden-hour street
scene drawn with MuJoCo's own physically based renderer (Filament, shipped in MuJoCo 3.14). Nothing in the
simulator, the controllers, the benchmark or the training code imports it, and it never touches the physics.

```bash
pip install -e ".[viz]"
python scripts/render_showcase.py --quality draft                    # 720p30 preview film
python scripts/render_showcase.py --quality high --hero --poster     # 1080p60 film, README GIF, poster still
```

## Design: record once, replay as cinematography

1. **Record** (`viz/recording.py`). An episode is simulated exactly like the benchmark (same environment, same
   controller settings, same safety filter) and its full visual state is stored at 50 Hz: generalised positions,
   mocap poses (kerb slabs, ramps, obstacles, pedestrians), shock, lidar returns, and, for MPPI, the sampled
   rollouts of every plan. Recording wraps `env.step` from outside; `AEDRoverEnv` is unchanged.
2. **Dress** (`viz/dressing.py`, `rover_visuals.py`, `people.py`, `overlays.py`). The unmodified physics MJCF is
   restyled (new look blocks, sun, sky) and given visual-only geoms (`contype=0`) appended after every physics
   element, so every joint, body and mocap slot keeps its index. Replay is then just copying recorded arrays into
   `MjData` and running kinematics; the render model is never simulated. `tests/test_viz.py` checks that
   original body poses in the render model equal the physics model's to 1e-9.
3. **Render** (`viz/backend.py`). `FilamentBackend` draws with PBR materials, image-based lighting, soft shadows
   and reflections, headless on the GPU. `ClassicBackend` (the usual `mujoco.Renderer`) is the fallback where no
   GPU or no Filament is available.
4. **Finish** (`viz/post.py`). Depth-aware atmospheric haze, depth of field, bloom, a warm/teal grade, vignette and
   grain, all on uint8 frames with OpenCV.
5. **Cut** (`viz/shots.py`, `camera.py`, `hud.py`, `film.py`). Camera rigs with inertia, slow motion around the shock
   peak, a live heads-up display, cards, and H.264 encoding straight from the frame stream.

The AED quadrotor follows the same pattern: `aedrover.drone` simulates a mission in its own model, `viz/drone_recording.py`
stores its pose and rotor thrusts every control tick (`simulate_mission(..., trace=...)`), and `viz/drone_visuals.py` draws
it as mocap-driven visual parts. The drone is never merged into the rover's physics.

Because shots are rendered from recordings, one episode can be filmed from several cameras and in slow motion
without simulating it again. A dynamic-window, an MPPI and a PPO episode on the same seed share one street, so the
kerb comparison in the film is the same scene three times.

## What is in the scene

* **Sky and light.** A procedural golden-hour sky (gradient, sun disc, clouds; `viz/assets.py`) is the skybox and
  the image light; a low, warm sun gives long shadows. Constants live in `configs/render/golden_hour.yaml`.
* **Street.** Raised paving with grout and normal maps on the physics slabs (widened visually on the same mocap
  bodies, so kerb height and ramps always match the physics), kerbstones, tactile strips, zebra crossing and lane
  paint at the scenario's positions, building rows with lit windows, shopfronts with signboards, awnings,
  balconies, air-conditioner boxes, rooftop tanks, street lamps, trees, parked cars (taxi colours included),
  parked motorbikes, street stalls and a hazy skyline. Nothing is placed inside the rover's corridor. The wet,
  glossy road appears only for the `slippery` family, matching its low friction.
* **Rover.** Shell, AED case with green crosses, spoked wheels with tread lugs, suspension arms, lidar puck and
  beacon, all attached to the real suspension and isolator bodies. The original box and cylinder geoms stay in
  the model but are shrunk to a point in the render copy.
* **People.** Low-poly pedestrians made of mocap limb segments. Heading comes from the smoothed recorded velocity and
  stride phase from the distance walked, so feet do not slide and a standing pedestrian stands.
* **Drone.** The comparator quadrotor drawn from the dimensions of the simulated vehicle (0.45 m arms, 0.25 m rotors,
  AED case with a white cross slung underneath, skids). One mocap body carries the airframe, four turn the rotors, and two
  parked-or-shown glowing spheres are the status light (amber in flight, green once the mission has released), because
  material colour is not updated per frame. The airframe pose is exactly the recorded pose moved into the street by a rigid
  `Placement`; `tests/test_drone_viz.py` checks it against the recorded arrays to 1e-9. Rotors turn at 5% of the simulated speed
  (a real rotor at about 40 rev/s would strobe at 60 frames per second), against their reaction torque, with an orange tip on
  the front pair so the spin and the heading read.
* **Dispatch shot** (`viz/dispatch.py`). The street is a 36 m segment but the clinical comparison is about a 1 km radius, so the
  film does not pretend to show the real distances. Two beats share one *dispatch clock* (simulated seconds since the alert): the
  drone's last approach and descent from its simulated flight, then, after a time skip, the end of the rover's recorded segment
  arriving beneath the hovering drone. The HUD shows the clock, the distance each vehicle still has to cover at true scale and
  when each arrives. Every number comes from code (`MissionParams`, `mission_time_s`, `ScenarioParams`) or from `results/`
  (`clinical_routes.csv`, `clinical_survival_vs_radius.csv`): the drone arrives at `time_to_scene_s`, the rover at the median
  travel time of the route model (PPO, 4 crossings per km), the survival panel quotes the ambulance, rover and drone rows. The
  clinical model times its drone more simply than the flown simulation (`docs/DRONE_COMPARATOR.md`, section 9), and the panel says
  by how much. Launch latency and the wind limit are assumptions and the drone is an upper bound; the footnote says so.
* **Data overlays.** (MPPI's rollout shot draws 24 of the 128 sampled rollouts per plan, spread over the cost ranking.) A cyan trail of the rover's actual upcoming path (works for every controller), a lidar
  bubble that dents inward where something is detected, MPPI's 24 sampled rollouts coloured by cost rank with the
  best one highlighted, and a halo that turns green, amber or red with the live payload shock.

## Engineering notes (MuJoCo 3.14, verified on Windows 11 with an RTX GPU)

These are behaviours that cost time to find; each is handled in the code.

* **Headless start order.** A headless `Window` must exist before the Filament context is created (it registers
  the resource providers). There is one engine per process.
* **`get_image` is broken for NumPy.** The returned object's `__array_interface__` is defined as a method, so the
  raw `.pixels` bytes are read with `np.frombuffer`.
* **Physical light units.** As soon as a light has an `intensity` the scene is in physical units: a sun of about
  5e4 lux with an image light of about 1.4e4 is well exposed; values of 1 to 100 render black. The sky texture can
  serve as both the visible skybox and the image light.
* **Per-frame updates.** Mocap poses and material colour or emission update every frame. `geom_rgba` and
  `geom_size` changes do not. So overlays are pools of fixed-size emissive beads on mocap bodies, coloured by a small
  set of materials (rank and fade levels). Glow colour comes from an emissive texture over a near-black surface;
  otherwise lighting adds white and everything turns pastel.
* **No `ModelDecorations` from Python.** `update` wants raw `mjvOption_` structs that the Python bindings do not
  expose, so extra geometry is built into the model instead.
* **Fog is ignored** by the PBR renderer, so haze is added in post from a distance map.
* **Depth.** Only one read-back per render call is allowed, so depth is a second pass with its views swapped.
  The depth image is 8-bit and non-linear (its `near` plane sets the encoding), so it is calibrated into a
  distance table with flat walls and drawn at quarter resolution. Ground pixels use the exact ray-plane distance;
  the buffer is used only where something is clearly nearer than the ground behind it, because it bands
  across a receding road.
* **Teardown order.** The depth material instance is shared between models. Calibrating (which builds and
  destroys small models) after a scene has done a depth pass destroys it under that scene's renderables and aborts
  the process; the decoder is therefore calibrated when the first depth-capable backend is created. A closed
  backend also drops every reference to its scene before its GPU objects are released.
* **Texture tiling.** With `texuniform` one texture tile spans `2 / texrepeat` metres.
* **Skybox faces.** World directions of the six faces, measured with sign-coded textures, are listed in
  `viz/assets.py`.
* **Small, far objects turn into sky.** The haze distance comes from a quarter-resolution depth map with a median filter, so a
  0.9 m drone beyond about 50 to 100 m from the camera drops out of it and gets full haze and heavy blur. The drone shots keep a
  camera 8 to 15 m from the drone, use a lighter `Grade.haze_strength`, and focus depth of field on the drone.
* **Distant towers and a flight line.** The skyline towers (260 m and 420 m rings, 25 to 110 m tall) are higher than the drone's
  50 m cruise, so `dress_scene(..., clear_flight_corridor=...)` leaves out the ones near the flight line, consuming the random
  stream identically so nothing else in the scene moves.

## Speed

On an RTX 4080 laptop, drawing one 1080p frame with PBR, shadows and image-based lighting takes about 6 ms in a
minimal scene and 20 to 50 ms in the dressed street (about 2,000 geoms, 3,000 with MPPI's rollout overlay). The finishing chain (haze, depth of
field, bloom, grade) adds roughly 80 ms, the depth pass about 30 ms, the HUD about 20 ms. The laptop's
performance state makes these vary by about 2x from run to run. Simulation is unaffected: recording an
episode runs at full CPU speed (a PPO or dynamic-window episode in a few seconds, an MPPI one in about half a
minute with its multi-threaded rollouts).

## Limits

This is a rasteriser with image-based lighting, not a path tracer: no global illumination, no true volumetric
light, hard-edged low-poly building and tree shapes. The visuals are for communication; every quantity the
project reports comes from the physics simulation and the files under `results/`. The PBR path needs MuJoCo 3.14
or newer and a GPU with OpenGL; elsewhere the classic backend draws a simpler image and the depth-based effects are
skipped. Filament's Python API lives under `mujoco.experimental` and may change.

## The live view and the rich-world mode

```bash
python scripts/live_cinematic.py                      # the simulation and the renderer run together, in a window
python scripts/live_cinematic.py --rich               # the same, in a street whose furniture is physical
python scripts/live_cinematic.py --drone              # the rover and the AED drone together, both simulated live
python scripts/live_cinematic.py --drone --record dispatch.mp4   # the same run, no window, fixed time step, written to a video
python scripts/live_cinematic.py --replay             # simulate each episode first, then play it back
python scripts/live_viewer.py --rich                  # MuJoCo's plain viewer on the rich world (a window per episode)
```

**Live, with no recording in between** (`viz/live.py`). Each frame the simulation is stepped until its clock catches up with the
wall clock, the simulation's state (`qpos` and mocap poses) is copied into the render model, and that state is drawn. The recorded
animators look at the whole episode, so the live path has causal versions: `StreamingPedAnimator` takes the heading from a
low-pass of the pedestrian's velocity (time constant 0.3 s) and accumulates the stride phase as the pedestrian walks, and
`LiveOverlayAnimator` draws the halo, the lidar bubble and MPPI's sampled rollouts from the live state. The recorded "path ahead"
ribbon needs the future, so it is not drawn live. A planner slower than real time (MPPI) runs slower than real time and the HUD says
by how much; `--replay` keeps the earlier pre-simulated playback for that case. On the reference laptop the render loop alone runs
at about 50 frames per second without the window; the Tk window is what limits it.

**A live drone** (`--drone`). The AED quadrotor is simulated live as well: `drone.mission.MissionStepper` advances the same mission
as `simulate_mission` one 8 ms control tick at a time (a test checks that its release time and trajectory are identical), in the
drone's own MuJoCo model with its cascaded flight controller, and `LiveDroneAnimator` draws whatever state it has reached, with the
rotor angles accumulated tick by tick. It is dispatched at the same moment as the rover and leaves from the rover's own start, one metre
to its right, bound for the same goal, so the two are in the picture together from the first frame. The 36 m street cannot hold the
1 km clinical comparison, so the mission is scaled to it (a 36 m flight at a 6 m cruise altitude by default, `--drone-distance` and
`--drone-altitude` change it): same physics and controller, different distances, and the footnote on screen says so. The camera
(`camera.watch_both`) stands at the patient's end of the street looking back, so both vehicles come toward it and grow; a camera
behind the rover kept losing a 1.4 m drone that is 20 to 60 m ahead. `--record` runs the same session with a fixed time step and
writes it to a video instead of opening a window.

**Rich world** (`sim/furniture.py`, `AEDRoverEnv(rich=True)`). Opt-in and separate from the benchmark: parked cars, rows of
motorbikes, stalls, lamp posts and tree trunks are static collision boxes from the road up, placed by a pure function of the scenario
(its own random stream, so no scenario draw moves) where a footpath really has them, encroaching on the footway by at most 1.2 m so a
1.6 m lane always stays free. The lidar sees them (so do the safety filter, DWA, APF and PPO's observation), the clearance and outcome
checks count them (`collision_kind = "furniture"`), and MPPI plans in a world with the same boxes. The model is compiled per
scenario (about 14 ms), so `env.world` changes at `reset` and anything that held it across a reset must read it again. The dressing
draws every item exactly on its collision boxes (`dress_scene` reads `rec.meta["furniture"]`) and hides the boxes, so what you see is
what collides; the street's other, purely visual furniture stays where it was. Results in this mode are not comparable with
`results/`; a small paired evaluation is in `docs/RICH_WORLD.md`.

Pedestrians walk round the furniture: the social-force crowd gets the boxes as solids (repulsion with a sidestep along the surface,
which stops a pedestrian walking straight at a stall from stalling against it, and a hard push-out so nobody overlaps a box). In the
standard world that code does not run. Not in the rich world: nothing moves or falls over, and the layout rule keeps every scenario
solvable.
