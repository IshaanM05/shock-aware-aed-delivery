# Engineering notes: MuJoCo pitfalls found while building this

Every item below silently produced wrong physics or wrong data, and every one now has a
regression test. They are recorded because none of them raises an error.

| # | Symptom | Root cause | Fix and test |
|---|---------|------------|--------------|
| 1 | The rover drove straight through a kerb whose pose was set at runtime | MuJoCo builds the collision mid-phase BVH of **static** bodies at compile time, so `model.geom_pos` edits on a static geom are ignored by collision | Every re-positionable piece of terrain is a **mocap body** moved through `data.mocap_pos` (`tests/test_physics.py::test_kerb_is_actually_solid`) |
| 2 | A mocap slab was still ignored | Per-body `contype` / `conaffinity` are aggregated at compile time; a slot compiled with `contype=0` is filtered out at body level whatever is written to `geom_contype` later | Compile every slot collidable and deactivate it by parking it 50 m underground |
| 3 | A resized slab did not collide where it should | The per-body BVH (`bvh_aabb`) is also compile-time, so resizing a geom leaves a stale collision volume | Fixed compile-time sizes (200 m slabs, 3 m ramps, two obstacle kinds); slots are only moved or rotated |
| 4 | Terrain vanished after `mj_resetData` | It restores mocap poses to compiled values | The `World` object owns the authoritative mocap state and `apply_mocap()` re-applies it after every reset (`test_mocap_state_survives_reset`) |
| 5 | **MPPI planned blind to kerbs** | `mujoco.rollout` resets mocap bodies to compiled positions, deleting all terrain from the rollouts | Feed mocap poses through `control_spec = CTRL \| MOCAP_POS \| MOCAP_QUAT`; this also puts the predicted pedestrian trajectories in the physics (`tests/test_mppi.py`) |
| 6 | Lidar and terrain scans returned "no hit" beyond ~8 m from the origin | `mj_multiRay(cutoff=...)` pre-filters on the distance from the ray origin to each geom's **centre**; an infinite plane's centre is the world origin | Huge cutoff plus manual clipping (`tests/test_sensors.py::test_height_scan_sees_the_road_far_from_the_origin`) |
| 7 | Renders showed only sky, with the near ground clipped | `<map znear>` is in units of the model **extent**, which the 200-400 m slabs inflated to hundreds of metres | Pin `<statistic extent>` explicitly |
| 8 | Kerb-climb success drifted downward across trials in one worker | The motor torque derating rewrote `model.actuator_forcerange`, and each new `Rover` snapshotted the already-derated values as "nominal" | Nominal torque limit derived from the vehicle parameters and restored on reset (`test_motor_derating_does_not_leak_between_rovers`) |
| 9 | `ValueError: mass and inertia of moving bodies must be larger than mjMINVAL` | A massless intermediate body between the suspension slide and the wheel hinge | Put both joints on one body |
| 10 | Payload isolator sagged twice as much as designed | `springref` sign: for a child body loaded by its own weight the equilibrium is `qref - m g / k`, so the reference must be `+m g / k` to sit at q = 0 | Static-equilibrium test (`test_static_equilibrium_matches_design`) |
| 11 | Accelerometer read 1 g at rest | MuJoCo accelerometers report *proper* acceleration | Shock metric subtracts `g * up_body` using the chassis attitude, so pitch and roll do not leak (`tests/test_metrics.py`) |
| 12 | Only `info["episode"]` metrics of the first process survived in PPO logs | SB3's `VecMonitor` overwrites `info["episode"]` | Metrics are re-published as `ep_metrics` |
| 13 | A safety filter that was too conservative deadlocked planners | A straight-ahead lane test flagged an obstacle 0.5 m to the side as "ahead" | The filter sweeps the actual footprint along the commanded steering arc (`test_safety_filter_ignores_an_obstacle_beside_the_path...`) |
| 14 | Pedestrians appeared next to the rover | The stream model respawned pedestrians at the entry point regardless of the rover | Respawn only when the entry is at least 6 m from the rover |

## Performance

* Raw stepping of the rover model: about 125k physics steps/s on one thread; `mujoco.rollout`
  reaches about 1.1M steps/s on 16 threads.
* The Gymnasium environment runs at roughly 3k control steps/s per core (50 Hz control), i.e. more
  than 60x real time per core, and about 250-320x real time aggregate on 28 workers.
* Where the time goes: `mj_step` is about 40 percent of a step; the rest is Python glue.
  Optimisations that mattered: native velocity/position actuators (no Python wheel controller),
  `mj_step(nstep=k)`, batched ray casts, cached attitude, lazy observation building, pedestrians
  stepped at 25 Hz, coarser shock logging for training (100 Hz instead of 250 Hz).
* One compiled model per worker process; per-episode randomisation is array writes, not a
  recompile.

## Windows notes

* Worker processes are spawned, not forked: entry points are guarded by `if __name__ == "__main__"`
  and jobs are picklable dataclasses.
* Model arena is 8 MB (`<size memory="8M">`); with 28 workers each caching a model, larger arenas
  exhausted memory ("Could not allocate memory for texture").
* H.264 encoding needs even frame dimensions.
* Two large multi-process jobs cannot overlap: 24 PPO environment processes (each with PyTorch loaded) plus 20
  spawned evaluation workers exhausted the Windows commit limit ("The paging file is too small for this operation
  to complete" while importing SciPy) and killed the pool. Run heavy stages one after another; the pipeline runner does.
