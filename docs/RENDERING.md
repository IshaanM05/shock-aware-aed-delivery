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

## Next step: a rich-world mode

Today the look is cosmetic by design: kerbs, ramps, bollards, planters and pedestrian capsules are physical, while
buildings, shopfronts, parked cars and motorbikes, stalls, trees and lamps are visual only, and the live cinematic view
(`scripts/live_cinematic.py`) plays back an episode that was simulated first. The planned next piece of work is an
opt-in **rich-world mode**:

* make selected street furniture collidable (parked vehicles and bikes, stalls, lamp posts, tree trunks) as extra
  physics bodies, so the rover and the controllers must really avoid them;
* step the simulation and the renderer together, so a live run is truly live, with no pre-simulation pass;
* keep it separate from the benchmark: its scenarios and results are not comparable with `results/`, and it must
  never change the standard environment, its seeds or its numbers.

Open questions: how the planners and the safety filter perceive the new obstacles (lidar already sees any collidable
geom), and how to animate the pedestrians' limbs incrementally instead of from a whole recorded episode.
