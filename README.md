# vla_ur5e_ws

Vision-Language-Action (VLA) pick-and-place on a UR5e, via Isaac Sim demo
collection -> LoRA fine-tuning of Physical Intelligence's π0 (through
[openpi](https://github.com/Physical-Intelligence/openpi)) -> deployment,
first in Isaac Sim then on the real UR5e.

π0.7 (the paper this was originally inspired by) has no public weights yet;
π0/π0.5 do, via openpi, which already ships a `examples/ur5/` template --
this project follows that template's exact contract (state = joints[6] +
gripper[1], actions = joint deltas + gripper, two cameras: base + wrist).

## Layout

```
isaac/                    Isaac Sim scene + scripted demo collection (Phase 1 -- fully built, smoke-test this first)
  isaac_sim_common.py      robot/gripper/RMPflow setup helpers
  pick_place_scene.py      scene construction + reset (randomizes cube spawn) + per-tick control
  scripted_pick_place.py   waypoint "expert" demonstrator (approach/grasp/lift/transport/place/retract)
  check_cameras.py         renders what each camera sees + sweeps wrist orientations (18-candidate, all rendered) -- RUN THIS FIRST
  check_wrist_mount_raycast.py  2026-09-21: cheap 294-candidate FOV+raycast screen, THEN renders only the survivors -- this is what actually found the current wrist mount
  check_rmpflow_stability.py    2026-09-21: N fresh episodes, checks joint pos/vel for NaN/Inf every tick, either physics pipeline (--device cpu|gpu) -- found the cube-spawn-edge instability
  check_grasp_alignment.py      2026-09-21: per-tick approach/settle/descend/close trace, gripper close fraction, tool-cube offset by axis (mesh-vertex-based, not link-origin) -- current open investigation, see README
  check_near_field_visibility.py 2026-09-21: places the cube at controlled 5-100mm standoffs on the wrist camera's own axis, checks near-clip/exposure/stale-frame separately
  collect_demos.py         runs episodes, scores each against ground truth, writes a LeRobot dataset
  pick_place_scene_bridge.py   ROS2-facing sim bridge for policy INFERENCE (Phase 3's Isaac Sim side)
  residual_rl_train_env.py     Gymnasium env for Residual RL on top of the frozen pi0 policy (Phase 5, optional)
  train_residual_policy.py     SB3 PPO training script for the residual policy (Phase 5)
  object_configs.py            small (color, shape) object vocabulary for the hybrid pipeline (Phase 6, optional)
  camera_projection.py         pure-numpy pixel<->3D pinhole math -- no Isaac Sim import, smoke-testable directly
  hybrid_pick_place_demo.py    LLM + open-vocabulary-detection pipeline, an alternative to the end-to-end VLA (Phase 6)
  collect_rlds_episodes.py     OpenVLA-flavored demo collection: single camera, Cartesian EE-delta actions (Phase 7, optional, higher risk)
  openvla_pick_place_demo.py   Runs a fine-tuned OpenVLA checkpoint against Isaac Sim, logs results (Phase 7)

perception/                Pure Python, no Isaac Sim dependency -- reusable from the ROS2 side later too (Phase 6)
  llm_command_parser.py     Claude API: free-form instruction -> {"object", "destination"}
  object_detector.py        Grounding DINO: (image, text description) -> pixel location

openpi_integration/        Files to drop into your openpi checkout (Phase 2 -- grounded against openpi's real source, not yet run)
  ur5e_pick_place_policy.py    UR5eInputs/UR5eOutputs/LeRobotUR5eDataConfig -> src/openpi/policies/
  train_config_snippet.py      the pi0_ur5e_pick_place TrainConfig -> append to src/openpi/training/config.py

openvla_integration/       OpenVLA side (Phase 7, optional -- LESS verified than everything else here, see that section)
  ur5e_pick_place_dataset_builder.py   TFDS/RLDS builder -- `tfds build` reads raw_episodes/, produces the dataset OpenVLA trains on
  openvla_transform_snippet.py         registration snippet to drop into your OpenVLA checkout + the finetune.py command

src/vla_bridge/            ROS2 package: queries the openpi policy server, drives the robot (Phase 3)
  vla_bridge/vla_policy_client.py    the main node -- works against Isaac Sim or the real UR5e via robot_backend
  vla_bridge/robot_interface.py      real UR5e (RTDE) -- ADJUST the gripper control section for your actual hardware
  vla_bridge/isaac_robot_interface.py  Isaac Sim backend, talks to isaac/pick_place_scene_bridge.py
```

## Running the Isaac Sim scripts

Isaac Sim on this machine is the **pip install** (5.1.0, in the
`env_isaaclab` conda environment), not a bundled install, so there is no
`python.sh` -- that wrapper only exists in the bundled/Omniverse-Launcher
layout. Run the scripts with that environment's own interpreter:

```bash
ISAAC_ENV=/home/icrs/bigdisk/conda_envs/env_isaaclab
export LD_PRELOAD=$ISAAC_ENV/lib/libstdc++.so.6
$ISAAC_ENV/bin/python check_cameras.py --sweep_wrist
```

The `LD_PRELOAD` is needed. Kit puts the system libraries ahead of the
environment's on the loader path once it starts, and the system
`libstdc++.so.6` here only goes up to `CXXABI_1.3.13` while the
environment's own libraries want `CXXABI_1.3.15`. Without the preload,
`import sqlite3` (and anything built on it) fails inside Kit with

```
ImportError: /lib/x86_64-linux-gnu/libstdc++.so.6: version `CXXABI_1.3.15' not found
```

Startup logs hundreds of those from its own test modules and carries on, so
it looks like noise -- but the same break hits any real code path that
touches sqlite3, LeRobot's dataset writer included. Preloading the
environment's `libstdc++.so.6` (which has `CXXABI_1.3.17`) fixes it and
silences the noise.

Use the **absolute path**, not a bare `python3`. Login shells pick the
environment up from the conda init in `~/.bash_profile`, but interactive
shells (a VS Code terminal, for instance) source `~/.bashrc` instead and get
`/usr/bin/python3`, which fails with `ModuleNotFoundError: No module named
'isaacsim'`. `conda activate` also works, but the environment lives outside
the default `envs` directory so it needs the full path:
`conda activate /home/icrs/bigdisk/conda_envs/env_isaaclab`.

`/home/icrs/isaacsim` is a *source checkout* (build scripts, no runtime) --
not what these scripts run against. Commands below are written as
`python3 ...`, meaning that interpreter; on a machine with a bundled install
instead, run them through its `python.sh` wrapper.

## Check it before you start Isaac Sim

The encoding and framing halves of this repo are plain numpy and scipy, and
run with no Isaac Sim, no GPU, no ROS 2 and no robot. Doing this first
separates "my install is wrong" from "my simulator is wrong", and takes a
few seconds:

```bash
pip install pytest            # the only thing the suite needs beyond numpy/scipy
python -m pytest test/ -q     # 121 checks, ~3 s
```

**If a plain `ModuleNotFoundError: No module named 'lark'` comes out of
`launch_testing`/`launch`, not out of this repo's own code**, it means ROS 2
was sourced in this shell: pytest auto-loads every installed `pytest11`
plugin, including the ones ROS 2 Humble registers (`launch_testing`,
`launch_ros`, `ament_lint`, ...), and one of those imports `lark`, which
ROS's own apt packages don't pull in. `pytest.ini`'s `addopts` now disables
those plugins by name so the command above runs the same whether or not ROS
is sourced -- confirmed 2026-09-15 against a shell with Humble sourced and
an Isaac conda env's Python ahead on `PATH`, the actual day-to-day shell on
this machine. If you still hit it (an older ROS distro registering a
different plugin name), add `-p no:<that plugin>` or just `pip install lark`.

What it covers, and why each part exists:

| file | holds |
|---|---|
| `test_validate_dataset.py` | every failure this project actually hit is still caught, and named |
| `test_metamorphic.py` | the shared-wrong-convention case the three identities cannot see |
| `test_gripper_state.py` | a measured reading is measured, and a fallback says so |
| `test_framing_guard.py` | the derived pixel guard separates the two runs on record |
| `test_camera_framing.py` | the framing model agrees with the one real measurement |
| `test_pivot_logic.py` | the dwell/pivot diagnostic tells dynamics from a wrong TCP |
| `test_robotiq_gripper.py` | the socket driver survives split, merged and abandoned replies |

These are the laptop-runnable checks. The two tests that need the simulator
live next to it instead (`isaac/test_feasibility_gate.py`,
`isaac/test_offset_grasp.py`) and are deliberately not collected here.

One more worth running by hand, because its output is the argument rather
than a pass/fail:

```bash
cd isaac && python verify_action_encoding.py
```

It reproduces the rotvec-subtraction bug numerically, then checks the euler
convention against an independently written extrinsic X-Y-Z. Note what that
check prints: the round trip closes to machine epsilon under a *wrong* axis
order too, which is why the independent reading is there at all. It still
does not prove agreement with OpenVLA — that is step 6 below.

## The next Isaac Sim session

**2026-09-21 update: four of the long-open blockers above (wrist mount,
scene lighting, gripper mass/divergence, cube-spawn edge instability) are
now genuinely fixed and verified on live Isaac Sim with a working GPU --
and fixing them uncovered a fifth, previously-invisible problem (the
scripted grasp closes on the cube but does not lift it) that is now this
project's actual blocker.** Read this section before touching camera
mount, gripper mass, or `CUBE_X_RANGE`/`CUBE_Y_RANGE` again -- re-deriving
any of it from scratch would repeat several hours of this session.

### What is actually fixed and verified (don't re-litigate these)

- **Gripper mass/divergence -- FIXED.** This codebase never set the
  gripper's mass (PhysX was computing it from collider volume at default
  density). A first attempt, `set_gripper_mass_properties`, split
  Robotiq's published 0.925kg total EVENLY across all 9 links -- an actual
  A/B run (`check_cameras.py --sweep_wrist --with_gripper`, otherwise
  identical) showed this made things catastrophically WORSE, not better:
  0 "Invalid PhysX transform" events in baseline vs. **5513** with the
  even split enabled, and reach-check distances that should read tens of
  mm instead spiking as high as **5x10^11 mm (500 billion km)** -- a real
  numerical explosion. Root cause: PhysX's own auto-computed masses were
  ALREADY realistic (housing ~610g, each finger/knuckle 13-43g, summing to
  846.8g -- only 8% under spec) and an even split inverted that ratio,
  handing tiny links far more mass than their real proportions; PhysX's
  reduced-coordinate articulation solver is sensitive to exactly that kind
  of adjacent-link mass-ratio mismatch. Fixed properly by
  `rescale_gripper_mass_to_spec` (`isaac_sim_common.py`): read PhysX's own
  auto-computed per-link masses off the live articulation view (needs a
  physics view, so this runs after the first `robot.initialize()`, not
  before like the collider setup), then multiply every link by the SAME
  scalar (0.925/0.8468 = x1.092) to hit the real total while leaving the
  proportions untouched. **Verified**: 20/20 fresh episodes clean (max
  residual 236-238mm, the same "settle spike" every clean episode already
  shows), 0 divergence events; a full `--sweep_wrist --with_gripper`
  re-run also came back at 0 divergence (vs. 5513 before), reach 53/54
  attempts in 30-61mm.
- **Cube spawn range -- NARROWED, not fully solved.**
  `check_rmpflow_stability.py` (new: drives the scripted trajectory for N
  fresh episodes, checking every joint position/velocity for NaN/Inf every
  tick, on either physics pipeline) found the tool-tracking residual
  spiking to 200mm-1.4m (finite, never NaN/Inf, so NOT the same bug as the
  mass one above) concentrated near the spawn range's EDGES -- and not one
  specific edge: X>=0.49 was the first one caught, then a separate run
  found an X=0.456/Y=0.19 spawn (the Y edge instead) also spiking to
  1.09m, then a third (after narrowing X alone) found a Y=-0.175 spawn
  (the OTHER Y edge) spiking to 887mm. Confirmed identical on both the GPU
  and CPU physics pipelines (15 episodes each), so this is a real
  RMPflow/geometry effect near this arm's reach limit or a singularity
  region, not a GPU-pipeline artifact. All four edges pulled in by 0.07m
  (`CUBE_X_RANGE`: 0.35/0.55 -> **0.42/0.48**; `CUBE_Y_RANGE`: -0.20/0.20
  -> **-0.13/0.13**) as a cheap mitigation. **Verified**: failure rate went
  from 6/15 (40%) before narrowing to 1/15 (7%) after X alone, to
  **0/20 (0%)** combined with the mass fix above. Cost: the spawn area is
  now only 0.06 x 0.26m for a 0.04m cube -- may be too little spatial
  diversity for the eventual policy's domain randomization; deliberately
  not chased further this session (see "what's still open" below).
- **Scene lighting -- FIXED, and this explains most of the old "wrist
  camera aimed wrong" investigation below.** Direct inspection found
  `add_default_ground_plane()`'s own overhead `SphereLight` (intensity
  100000) was the ONLY light this scene ever had -- confirmed by grep,
  zero other `UsdLux`/lighting code anywhere in `isaac_sim_common.py` or
  `pick_place_scene.py`. Measured at reset: `base_rgb` mean=72 (tolerable,
  high up, unobstructed) but `wrist_rgb` mean=27 (low, close to the
  gripper/table, partly shadowed from a single overhead point source).
  This alone explained why a fine-grained wrist-mount search
  (`check_wrist_mount_raycast.py`, see below) could not get near_black
  under ~57-84% no matter the direction/lateral/tilt -- EVERY candidate it
  geometrically verified (cube in-FOV, unoccluded PhysX raycast to its
  centre) still rendered mostly dark. Added one `UsdLux.DomeLight` fill
  light (`/World/fillDomeLight`, intensity 1500) in
  `PickPlaceScene.__init__`. **Verified**: wrist_rgb near_black dropped
  from 40%+ to **0.2%** in a direct A/B check (mean 27 -> 79), with no
  mount/angle change at all.
- **Wrist camera mount -- a real, validated answer, for the first time.**
  Built `check_wrist_mount_raycast.py`: a two-stage search that screens a
  294-candidate grid (6 directions x 7 laterals x 7 tilts) with cheap
  geometry only (FOV-cone membership + an unoccluded PhysX raycast to the
  cube's centre, no rendering) across several fresh grasp poses, then
  renders only the survivors for real. This is necessary because
  `check_cameras.py`'s own render-every-candidate sweep can only afford 18
  candidates before the GPU cost gets prohibitive. With the lighting and
  mass/spawn-range fixes above in place: **`WRIST_CAMERA_FLANGE_ROT_EULER
  = (180, 0, 0)`, `WRIST_CAMERA_LATERAL_M = 0.16`,
  `WRIST_CAMERA_DOWN_TILT_DEG = 15`** -- worst-case 351 cube px, mean 555,
  **0% near-black** across 3 fresh render samples. `preflight_check()`
  passes cleanly (`ok: True, problems: []`) for the first time this
  project has ever recorded. Note: an earlier attempt today set
  `WRIST_CAMERA_DOWN_TILT_DEG = 30` on the strength of an external
  comparison (frontal vs. ~30-degree-down eye-in-hand mounts) -- the one
  direct measurement taken of THAT value (before the lighting fix) showed
  it made things WORSE (39% near-black at tilt=0 vs. 60% at tilt=30 on the
  OLD mount). Lesson already stated elsewhere in this file but worth
  restating: an externally-plausible number is not a substitute for
  measuring it on this specific asset/scene.
- **Two misleading, stale code comments fixed.** `check_cameras.py`'s
  docstring and `_render_wrist`, plus `pick_place_scene.py`'s
  `_sync_gripper_to_flange`, described a "teleport" gripper-attachment
  failure mode (fingers falling off the arm without a per-tick re-sync,
  producing "Invalid PhysX transform"/"Illegal BroadPhaseUpdateData"
  storms) that was real for an attachment scheme this project replaced
  long ago -- the CURRENT scheme welds the gripper into the arm's own
  articulation via a USD variant selection
  (`isaac_sim_common.select_gripper_variant`), so
  `GripperController.sync_pose_to_flange` is a documented no-op and there
  is no separate gripper prim left to fall off. These comments were
  accurate enough-sounding that they produced a very reasonable but
  INCORRECT hypothesis for the mass-related divergence above (confirmed
  by grep: no bare `world.step()` calls anywhere in `check_cameras.py`,
  both before and after the divergence was found). Comments now say so
  explicitly.

### What's new and still open: the grasp closes but does not lift

With all of the above fixed, `collect_demos.py --num_episodes 5` (env:
`lerobot` had to be pip-installed for this -- see the environment note
below) ran end-to-end for the first time -- no camera rejection, no
PhysX divergence -- and then rejected **15/15** attempts as "never
lifted": `max_cube_z` stayed at 0.02-0.041m against `LIFT_Z_THRESHOLD
= 0.08m`. This is a genuinely new (or rather, newly-visible -- every
earlier session got stuck upstream of this point) problem. Investigated
today with a new script, `check_grasp_alignment.py`, logging
approach/settle/descend/close every tick plus the gripper's actual close
fraction and the tool-cube offset. Findings, in the order they were found
(each one corrected the previous step's naive reading -- read in order if
picking this back up, the wrong turns are as informative as the right
ones):

1. **First reading (WRONG): "the fingers land ~140mm above the cube."**
   Measuring the `left_inner_finger`/`right_inner_finger` LINK's own
   `prim_world_pose()` put them 129-151mm above the cube's centre at the
   end of every close segment -- physically impossible for this finger's
   real length. Root cause: a link's coordinate origin is at its PROXIMAL
   joint (where it mounts to the knuckle), not its contact surface, and
   the `Defeatured_2F_85_PAD_OPEN_fingertipsstep` mesh under it has NO
   transform op of its own -- its real extent is baked into its vertex
   positions, which `prim_world_pose()` (transform-only) cannot see.
   `mesh_world_bbox_center()` (new, in `check_grasp_alignment.py`) reads
   the mesh's actual points and transforms them to world space instead.
   With that fix, the pad centroid sits +2.5 to +16.2mm above the cube's
   centre at the end of close -- unremarkable, not the story.
2. **Axis decomposition, done right: X is NOT the gripper's closing axis
   -- Y is.** Across 4 fresh episodes, `left_dx`/`right_dx` were the SAME
   sign and nearly the same magnitude every time (e.g. +16.6/+16.3,
   -23.9/-25.0, -17.8/-19.0, +27.6/+27.2) -- a closing-axis error would
   push one pad closer and the other farther, i.e. OPPOSITE signs, which
   is instead what `dy` showed every time (-12.9/+14.9, -13.9/+9.8,
   -9.8/+14.2, -7.0/+16.0). So X is a COMMON-MODE offset shared by both
   fingers (the whole gripper reaching to the wrong place), not a
   closing-axis/grip-symmetry problem.
3. **The common-mode X offset (16-28mm) is large relative to the cube's
   20mm half-width -- close to its edge, not its centre.** This matches,
   in both mechanism and rough scale, a comment already sitting in
   `scripted_pick_place.py` since 2026-09-16 (`GRASP_HEIGHT` targets the
   cube's TOP FACE with zero clearance, and "20-40mm off-centre against a
   40mm cube is enough to clip an edge instead of centring on top of it")
   -- except that comment's own proposed mechanism (premature contact
   during descent knocking the arm sideways) does NOT hold up: see next.
4. **Timing correlation test: the contact hypothesis is REJECTED.**
   Logged the left pad's real bbox height against the cube's actual top
   face every 4 ticks through descend+close, specifically to test whether
   the X drift's onset lines up with the pad's lowest point reaching the
   cube's top face (which would confirm early contact). It does not: the
   pad-vs-cube-top gap crosses zero smoothly (e.g. +3.4mm -> -0.5mm
   between two adjacent samples) with NO discontinuity in `dx` at that
   moment, and `dx` was already smoothly drifting tens of mm earlier, while
   the pad was still 50-90mm above the cube -- clearly too far to touch
   anything. The drift is not a contact event.
5. **What it actually looks like: a smooth, monotonic, ~40mm drift over
   the ENTIRE descend+close duration (roughly 60+ ticks), with the
   commanded X/Y target completely FROZEN the whole time.** One traced
   episode: `dx` = -10.7mm (tick 320, still descending) -> -7.2 -> -2.6 ->
   +3.8 (tick 356) -> ... -> +32.8mm (tick 402, end of close) -- one
   direction, never reversing, never plateauing, across both the
   pure-vertical descend (target X/Y unchanged) AND the close segment
   (target position ALSO frozen, only the gripper joint ramps). A target
   that never moves but a tracked point that keeps drifting the entire
   time it's held is exactly the shape `pivot_dwell_check.py` was
   ORIGINALLY written to detect -- its own docstring's "DYNAMICS" row:
   "gravity and the gripper's mass sagging against drive gains that are
   too soft to hold a loaded pose." Ran it fresh today
   (`pivot_dwell_check.py`, with gripper, post-mass-fix) to check, but its
   own test design turned out not to be comparable: it commands its probe
   target from a COLD START (tick 1) instead of arriving via a gradual
   settle like `scripted_pick_place.py` does, so it reports a 480-488mm
   initial error that mostly (but not cleanly monotonically -- there's an
   odd hump at ticks 90-120 in the grasp-target case) converges by tick
   180, which isn't the same "already-settled, then does it creep" question
   this drift raises. Also noticed in passing: its own verdict-printing
   logic has a sign bug -- it printed "the residual GROWS by -311.3mm...
   that is dynamics" for a case that was actually SHRINKING (a negative
   number), because it only checks `abs(growth) > tolerance` without
   looking at the sign. Not fixed yet.

**Next step, not yet done:** extend `check_grasp_alignment.py` itself
(same measurement methodology already validated above, rather than
`pivot_dwell_check.py`'s differently-shaped test) to hold the end-of-close
target for several hundred EXTRA ticks with nothing else changing, and
watch whether the common-mode `dx` keeps growing without bound, or
eventually plateaus at some larger-but-finite value. That distinguishes
"the drive gains are genuinely too soft to hold this loaded pose at all"
(keeps growing -- fix per `pivot_dwell_check.py`'s own original
prescription, tune wrist stiffness/damping, tooling for which already
exists via `get_joint_drive_gains`/`set_joint_drive_gains` and
`pivot_dwell_check.py --stiffness-scale`/`--damping-scale`) from "it settles
somewhere past the cube's edge" (a targeting/waypoint problem --
`GRASP_HEIGHT`'s zero-clearance top-face target, or `GRIPPER_TCP_OFFSET_M`'s
2026-09-10 measurement, would be the next things to re-derive, this time
against the pad mesh's own vertex bbox rather than a link origin or an
assumed constant -- see finding 1 above for why the link origin lies).

### Environment note: installing `lerobot` briefly broke `isaacsim`/`isaaclab`

`collect_demos.py` needs `lerobot`, which was not installed anywhere on
this machine. `pip install lerobot` into `env_isaaclab` pulled in
`numpy==2.4.6` as a transitive dependency upgrade, which conflicts with
`isaacsim-kernel==5.1.0.0` (`numpy==1.26.0` exact pin), `isaaclab`,
`isaaclab_rl`, `isaaclab_tasks` (`numpy<2`), `numba`, and `cmeel-boost` --
all installed in that same shared environment for the broader IsaacLab
work, not just this project. Fixed by `pip install "numpy==1.26.0"`
afterward (confirmed both `isaacsim` and `lerobot` import correctly at
that pin; some other lerobot-transitive version conflicts remain
unresolved but non-blocking -- `click`, `psutil`, `typing_extensions`,
`packaging`, `rerun-sdk` all logged mismatches pip's resolver didn't
fully reconcile). **If `env_isaaclab` starts behaving strangely on
unrelated IsaacLab work, check `pip list | grep numpy` first** -- it
should read 1.26.0, not 2.x.

### Old (2026-09-16 and earlier) "next session" plan below this point

Superseded by the above for the wrist-mount and reach/divergence
questions specifically; kept for its own still-possibly-relevant detail
on how the 2026-09-16 cube-position bug was found and fixed, and because
the pivot_dwell_check.py methodology critique in "what's still open"
above only applies to that specific test's cold-start design, not to the
underlying dwell/pivot concept.

#### 1. A genuine wrist-camera-mount result is in, and it is real this time: no mount works

With reach no longer confounding the score, `check_cameras.py --sweep_wrist
--wrist_samples 5` (all 18 direction x lateral candidates, 5 cube spawns
each) still reports:

```
NO MOUNT WORKS -- even the most reliable candidate's WORST sample saw only
0 cube pixels (588 mean) at the grasp, where an eye-in-hand camera should
see thousands.
```

Every candidate's worst sample over 5 spawns was 0 px; means ranged
0-588px. This is no longer a measurement artifact -- it is the mount
genuinely failing to keep the cube in frame across `CUBE_X_RANGE` (0.35-
0.55m) x `CUBE_Y_RANGE` (-0.20-0.20m), a 0.20 x 0.40m spawn area no single
fixed eye-in-hand direction+offset covers from every point in it. (**2026-
09-21: superseded -- see above. The actual root causes were missing scene
lighting and too coarse a candidate grid, not the spawn range**, though
the spawn range was independently narrowed anyway for the unrelated
reach/divergence reasons above.)

#### 2. Chase the 5.26m contact-explosion event -- but re-measure it first

The 12-episode with-gripper run that found this was itself run before the
second cube-position bug above was fixed, so its cube-drift numbers
(28-236mm) are suspect the same way the camera sweep's were. (**2026-09-21:
not yet specifically re-chased under today's fixes -- the closest thing
today's session has to an answer is "what's new and still open" above,
which is a smooth drift, not a single explosive event, but a genuine
5+ metre single-tick launch has not been specifically searched for since
the mass fix.**)

#### 3. Re-run pivot_dwell_check.py for a genuine settled-residual number

```bash
python3 pivot_dwell_check.py
python3 pivot_dwell_check.py --no-gripper
```

(**2026-09-21: run, see "what's new and still open" above for why its
cold-start test design didn't give a clean answer to the sustained-hold
question it was needed for.**)

#### 4. Smoke-test before committing to a long collection run

```bash
python3 collect_demos.py --num_episodes 5
```

(**2026-09-21: run. It completed end-to-end -- no camera rejection, no
divergence -- for the first time, then rejected 15/15 as "never lifted".
See "what's new and still open" above.**)


## Bring-up order

**1. Check the cameras (30 seconds -- do this before every collection run)**
```bash
cd isaac
python3 check_cameras.py --sweep_wrist
```
Writes PNG contact sheets of what each camera actually sees. Three collection
runs have now been spent on cameras pointing somewhere other than the task
(a 1.0 m near-clip plane that removed the whole scene, a guessed Euler tilt
aimed at the sky, and a wrist camera still imaging empty background after
the other two were fixed). Every one of those was invisible in the logs and
obvious in a single rendered frame.

`--sweep_wrist` settles the wrist camera's orientation by measurement. It
renders each of the six axis-aligned directions at the moment the scripted
expert closes on the cube and scores them by how many cube pixels each
actually sees -- an eye-in-hand camera at the grasp should see the cube
large, so the right answer wins by a wide margin rather than needing a
judgement call. It prints the winning line ready to paste into
`pick_place_scene.WRIST_CAMERA_FLANGE_ROT_EULER`, and writes contact sheets
to check it by eye.

The sweep works in the **flange** frame on purpose. The committed derivation
is analytically correct as far as it goes (a USD camera images along its own
local -Z, so rolling 180° about X aims that axis along the commanded tool's
+Z, its approach direction) and still came out pointing at the sky -- which
puts the error upstream, in the flange→tool0 offset read from
`rmpflow.get_end_effector_pose()`, whose return layout and reference frame
are both unverified. Working in the flange frame needs none of that.

**2. Collect demonstrations (Isaac Sim machine, no GPU/openpi needed yet)**
```bash
python3 collect_demos.py --num_episodes 5 --repo_id you/ur5e_pick_place
```
Start with 5 to confirm the scene builds and the scripted expert actually
completes the task, before committing to a real run (50-200 episodes).

`--num_episodes` is the number of **successful** episodes: each attempt is
scored against ground truth (was the cube lifted clear of the table, did it
end up within `--place_tolerance_m` of the target) and a failed attempt is
discarded and retried rather than written to the dataset. Watch two numbers
in the per-episode log:

- `peak_cube_px` -- how many pixels of the cube the base camera ever saw.
  The run aborts if the first episode never gets the cube properly in frame.
- the closing success rate. Below 90% means the scripted expert itself is
  unreliable, which in this scene means an unstable grasp -- fix that before
  training, since the policy has to reproduce it. See `ISAAC_GRIPPER_ATTACH`
  under Known gaps.

**3. LoRA fine-tune (GPU machine, 2x RTX 3090)**
- `git clone` [openpi](https://github.com/Physical-Intelligence/openpi), follow its own setup instructions.
- Copy `openpi_integration/ur5e_pick_place_policy.py` to `src/openpi/policies/`.
- Append the `pi0_ur5e_pick_place` TrainConfig from `openpi_integration/train_config_snippet.py`
  to `src/openpi/training/config.py`'s config list, with `REPO_ID` pointing at your dataset from step 1.
- ```bash
  uv run scripts/compute_norm_stats.py --config-name pi0_ur5e_pick_place
  uv run scripts/train.py pi0_ur5e_pick_place --exp-name=ur5e_pick_place_v1
  ```
  LoRA fits on one 3090 (>22.5GB needed); only reach for multi-GPU if a
  single card is too slow. Run a short step count first to confirm the
  pipeline works end-to-end before committing to the full `num_train_steps`.

**4. Serve + evaluate in Isaac Sim (the real go/no-go checkpoint)**
```bash
# terminal 1, GPU machine:
uv run scripts/serve_policy.py --config pi0_ur5e_pick_place --checkpoint <path-to-your-checkpoint>

# terminal 2, Isaac Sim machine:
cd isaac && python3 pick_place_scene_bridge.py

# terminal 3, ROS2 workspace:
cd .. && colcon build --symlink-install && source install/setup.bash
ros2 launch vla_bridge vla_bridge.launch.py robot_backend:=isaac_sim
```
Watch whether the policy can pick-and-place a cube at a spawn position it
wasn't specifically shown during training -- this is the signal to trust (or
not) before ever pointing this at the real robot.

**5. Real UR5e**
Same `scripts/serve_policy.py` command, then
`ros2 launch vla_bridge vla_bridge.launch.py robot_backend:=rtde robot_ip:=<your UR5e's IP>`,
with your real camera driver(s) publishing to `base_image_topic`/
`wrist_image_topic` (see `config/params.yaml`). **Before this step**, wire
up real gripper control in `robot_interface.py` -- its current
`set_gripper()` is a binary-relay placeholder, see that file's docstring.

**6. (optional) Residual RL -- freeze pi0, train a small correction policy on top**

Per the [ICLR 2026 VLA research survey](https://mbreuss.github.io/blog_post_iclr_26_vla.html),
post-training a frozen VLA with a lightweight RL correction policy ("Residual
RL") is one of two directions currently gaining traction for closing the
sim/real success-rate gap -- and an explicitly open question ("no method has
yet established dominance"), not a solved problem. This is the actual
experiment worth reporting, not just "it runs":

```bash
# with the pi0_ur5e_pick_place server from step 3 already running:
cd isaac
python3 train_residual_policy.py --policy_host <gpu-host> --policy_port 8000 --total_timesteps 500   # smoke test first
python3 train_residual_policy.py --policy_host <gpu-host> --policy_port 8000 --total_timesteps 50000  # real run
```

Every step here makes a live call to the policy server AND steps real Isaac
Sim physics -- slow by design, same tradeoff as `collect_demos.py` and
`quadruped_parkour_ws`'s CIGR training.

Then set `use_residual_policy:=true` and `residual_model_path` (to
`models/residual_policy_final.zip`) in `config/params.yaml` or on the launch
command line, and **compare against the same cube spawn positions with it
off vs on** -- e.g. 20 fixed positions, log success/fail each way, both in
Isaac Sim and (once you trust it) on the real UR5e. That comparison table is
the deliverable, not the code existing.

The residual policy's observation is deliberately just proprioception + pi0's
own proposed action (no cube/target position, even though that's free ground
truth in sim) -- see `residual_rl_train_env.py`'s docstring for why: using
privileged sim-only state there would make the correction policy itself
sim-only, on top of the sim/real gap already being studied for pi0.

**7. (optional) LLM + open-vocabulary hybrid pipeline -- an alternative to the end-to-end VLA**

Modeled on Pusan National University's RoboCup@Home-winning "타이디보이"
team (robot: Anubis/아누비스): their hardest mission (EGPSR) was solved with
**LLM-based task planning** decomposing a free-form command, plus explicit
object recognition -- a modular architecture, not one end-to-end model. This
is kept alongside (not replacing) the pure π0 pipeline above, specifically
as a comparison arm:

```
"pick up the red cube and put it in the target zone"
        |
        v
  LLM (Claude API) parses -> {"object": "a red cube", "destination": "target_zone"}
        |
        v
  Grounding DINO locates "a red cube" in the base camera image -> pixel (x, y)
        |
        v
  camera_projection.py ray-casts that pixel to a 3D table position
        |
        v
  scripted_pick_place.ScriptedPickPlace (same class Phase 1 uses, unmodified) executes it
```

Requires `ANTHROPIC_API_KEY` set and `pip install anthropic transformers
torch` in the Isaac Sim python.sh environment (Grounding DINO's weights
download from HuggingFace Hub on first use). Run:
```bash
cd isaac
python3 hybrid_pick_place_demo.py --instruction "pick up the blue cube and put it in the target zone" --n_objects 3 --n_trials 10
```
Each trial prints every intermediate result (what the LLM parsed, where the
detector found the object, the localization error vs. ground truth, the
final placement error) so a failure's actual cause -- misparsed command,
missed detection, bad pixel-to-3D localization, or a failed grasp -- is
easy to tell apart, rather than just "it didn't work."

**The comparison this is for**: once Phase 1/2 are redone with multi-object
demonstration data (needed for a fair comparison -- the current
`pi0_ur5e_pick_place` checkpoint only ever saw one cube), fix N scene
layouts and run each through both this hybrid pipeline and the pure-VLA
pipeline, reporting success rate side by side. That comparison -- not
either pipeline in isolation -- is the actual portfolio/paper result.

The object vocabulary (`isaac/object_configs.py`) is deliberately tiny (4
color/shape combinations) for this first version; extending it is mostly
just adding entries there plus more varied demonstration data if you want
the pure-VLA side of the comparison to keep up.

**8. (optional, higher risk) OpenVLA -- a third comparison arm**

**This section is less verified than everything above.** Phases 1-6 each
had a concrete official template to mirror (openpi's `examples/ur5/`,
Grounding DINO's model card); OpenVLA has no equivalent for a generic
single-arm robot, only robot-specific examples (BridgeData V2 WidowX,
LIBERO). What's here is grounded in OpenVLA's actual fine-tuning script and
the community-standard `kpertsch/rlds_dataset_builder` RLDS scaffold, but
expect to debug the dataset-building step more than anything else in this
project.

**Why add it anyway**: OpenVLA (7B params, pretrained on 970k Open-X-
Embodiment episodes) is specifically documented as strong at exactly the
task this project cares about -- multi-object, language-grounded
manipulation -- more so than π0, which we've only ever fine-tuned on a
single cube. Alongside Phase 6's hybrid pipeline, this makes a genuine
**three-way comparison** possible: hybrid (LLM+detection) vs. π0 (fine-tuned
end-to-end VLA) vs. OpenVLA (a VLA whose pretraining already emphasizes
this exact capability).

Important: OpenVLA's data convention differs from π0's in two ways, so its
data collection is a **separate pass**, not a reuse of Phase 1's dataset --
single camera image (not base+wrist), and **Cartesian end-effector pose
deltas** for actions (not joint positions).

```bash
# 1. Collect (Isaac Sim machine) -- multi-object scene, same as Phase 6
cd isaac
python3 collect_rlds_episodes.py --num_episodes 5   # smoke test first
python3 collect_rlds_episodes.py --num_episodes 100  # real run

# 2. Build the RLDS/TFDS dataset from those raw episodes
cd ../openvla_integration
pip install tensorflow tensorflow_datasets apache_beam
tfds build   # this is the step most likely to need iteration -- see the file's own ADJUST notes

# 3. Register with OpenVLA (GPU machine)
git clone https://github.com/openvla/openvla
# follow openvla_transform_snippet.py's instructions to copy its 3 pieces
# into the OpenVLA checkout, then run its printed finetune.py command

# 4. Evaluate the fine-tuned checkpoint in Isaac Sim
cd ../isaac
python3 openvla_pick_place_demo.py --checkpoint <path-to-checkpoint> --n_trials 20
```

Once all three pipelines (hybrid, π0, OpenVLA) independently work: fix N
multi-object scene layouts and run each through all three, reporting
success rate side by side. **That three-way table is the actual
deliverable** -- more valuable for a portfolio/paper than any one pipeline
working in isolation.

All three now log results in the same shape for exactly this purpose:
`hybrid_pick_place_demo.py` and `openvla_pick_place_demo.py` write
`(trial, success, steps, xy_error_mm[, localization_error_mm])` rows to
`--results_csv`, and `vla_policy_client.py`'s `eval_mode:=true` (isaac_sim
backend only) does the same via `results_csv_path` -- see that node's own
new parameters (`n_trials`, `success_xy_tolerance_m`, etc.) in
`config/params.yaml`.

## What is still unverified, and what verifies it

Nothing in this repository has been run end to end. The perception and
encoding halves have been checked offline — against analytic geometry,
stubbed backends and synthetic episodes — but every claim that involves the
simulator, a camera or a robot is untested. Ordered so each failure is cheap
and interpretable.

| # | what | needs | how you know it worked | status (2026-09-21) |
|---|---|---|---|---|
| 1 | Wrist camera mount | Isaac Sim | `check_cameras.py --sweep_wrist` picks a winner; pin it in `WRIST_CAMERA_FLANGE_ROT_EULER`, which currently reads `None` | **DONE.** `check_wrist_mount_raycast.py` validated `(180,0,0)@0.16, tilt=15` -- 0% near-black, `preflight_check()` passes. See "The next Isaac Sim session" above. |
| 2 | Base camera framing | a decision | `preflight framing` prints the ceiling; 30 px across is **not reachable** at the current workspace and resolution | Still not reachable (ceiling now 23.8px, up from 20.5, after narrowing the spawn range for unrelated reasons) -- decision not yet made, still context-only for the wrist camera to carry approach. |
| 3 | Is the residual frame, contact, or dynamics? | Isaac Sim | `pivot_dwell_check.py` — free-space drift means dynamics, grasp-only drift means contact | **Still open, and this is the current blocker.** Fresh evidence (`check_grasp_alignment.py`) rules out contact (no discontinuity when the pad reaches the cube) and points at DYNAMICS (smooth ~40mm drift over a fully-frozen held target) -- but `pivot_dwell_check.py`'s own cold-start test design couldn't confirm it cleanly. See "what's new and still open" above for the exact next step. |
| 4 | Does the scripted expert grasp? | Isaac Sim | `collect_demos.py --num_episodes 5` completes without the attempt cap | Ran for the first time end-to-end (no camera/divergence rejection) -- and hit the attempt cap for a NEW reason, "never lifted" (0/15). This row and row 3 are now the same open question. |
| 5 | Demonstration collection | Isaac Sim | 50–200 episodes with the reject rate low and the gate quiet | Blocked on row 3/4. |
| 6 | Euler axis order vs OpenVLA | an OpenVLA checkout | compare its dataloader against `EULER_SEQ`; the metamorphic check proves self-consistency, **not** agreement with OpenVLA | Unchanged. |
| 7 | Fine-tune | 2× RTX 3090 | training converges; `compute_norm_stats` runs without shape errors | Unchanged. |
| 8 | Serving + bridge | GPU host + ROS 2 | the policy client steps without timing out | Unchanged. |
| 9 | Real gripper driver | UR5e + Robotiq | gripper state logs `measured`, not `NOT MEASURED` | Unchanged. |

Step 1 gates 4 and 5: openpi's UR5 contract feeds the policy both views, and
the wrist view carries the fine manipulation signal, so collecting with it
mis-aimed wastes the run. Step 3 decides whether there is a dynamics problem
at all — the code's own note says the residual came from the fingers
contacting the table, which the frame fix addressed, while a later review
suggested gravity and drive gains. Only a free-space hold separates them, and
the answer changes whether there is work to do. **2026-09-21: step 1 is now
done, and step 3 is the live question -- see "The next Isaac Sim session"
above for exactly where that investigation stands.**

## When something goes wrong

Messages quoted as the scripts actually print them.

### Collection

| you see | it means | do |
|---|---|---|
| `preflight check failed -- refusing to collect` | a camera is flat or mostly black | run `check_cameras.py`; a near-clip plane or a camera inside a link |
| `the cube peaked at N pixels ... The task is not visible` | the framing cannot show the task | this is step 2 — reframe or accept the base view as context only |
| `gave up after N attempts with only M/K successful episodes` | the scripted expert is not completing the task | a grasp problem, not a data one. Do step 3 and 4 before collecting more |
| `attempt N: REJECTED` with encoding complaints | the recording is malformed | the listed identity says which: position, rotation or gripper |
| `the target was never lifted ... the gripper closed on nothing` | a flawless recording of a failed grasp | same as the attempt-cap row: fix the grasp |
| three rejections in a row then an abort | the setup is wrong, not unlucky | the abort is deliberate — fix what the problems say before re-running |

### Dataset validation

| you see | it means | do |
|---|---|---|
| `N/M transitions do not satisfy state[t+1] == state[t] (+) action[t]` | the recorded action does not carry the recorded state | an encoding or logging bug; `verify_action_encoding.py` isolates it |
| `the recorded angles do not mean extrinsic (fixed-axis) XYZ` | `EULER_SEQ` is not what the data was written in | do not change one without the other; they share the constant deliberately |
| `frames are black ... Check the camera near-clip plane` | the default 1.0 m clipped the scene | set it to 0.01 m and confirm it took effect in **headless** mode |
| `the red target peaked at N pixels` | the target is not in the observations | no amount of data will help; fix framing first |
| `state has shape (N, 7), expected (N, 8)` | the POS_EULER pad slot is missing | every field from the rotation on would be read one slot left |

### Serving and execution

| you see | it means | do |
|---|---|---|
| `gripper state is not measured (...)` | the policy is being fed the value it commanded | sim: check `joint_state` has 7 values. Real: pass a `gripper_driver` |
| `joint_state has 6 values, expected 7 with the gripper last` | the sim bridge is not appending the gripper | `pick_place_scene_bridge.publish_observation` |
| `gripper driver read failed: ...` | the socket dropped | the loop continues on the last command — fix it before trusting a grasp |
| `gripper commanded 1.00, measured 0.05` | it closed on nothing | this is the signal that used to be invisible. Believe it |
| policy client waits forever on images | topics do not match the driver | `image_topics` in the launch parameters |

### The one that hides

A collection run that finishes and a dataset that loads are not evidence of
anything. Both expensive failures here looked exactly like success: 21,000
well-formed frames in which the cube never appeared, and 243 of 499 rotation
deltas inflated tenfold with correct shapes and finite values throughout. The
gate now catches both, but it shares the collector's conventions by design —
step 6 is what confirms those conventions are the ones OpenVLA reads.


## Known gaps / ADJUST markers to resolve on real hardware

- **The scripted expert did not actually reach the cube -- found and partly
  fixed 2026-09-14.** A smoke test of `check_cameras.py --sweep_wrist`
  printed its own built-in warning ("the tool did not actually reach the
  cube"), so this was chased down with a standalone reach probe
  (per-tick tool/target/cube logging, not committed -- reproduce by driving
  `PickPlaceScene.step_towards` over `ScriptedPickPlace.generate_frames()`
  and comparing `grip_point_world()` against the waypoint). Two real bugs,
  fixed here:
  - `steps_per_segment` defaulted to 30 (0.5s/segment). RMPflow's default
    UR5e gains cannot track a Cartesian target moving that fast: the tool
    was still 259mm from the cube when the gripper closed. Raised to 90
    (1.5s/segment); with the gripper disabled entirely to isolate this from
    the bug below, that converges to ~20-40mm before the close segment
    starts.
  - `generate_frames()` interpolated Cartesian position per tick but jumped
    the gripper target straight to 1.0 (fully closed) on the first tick of
    the close segment -- an instant snap, not a close. `try_compliant_close.py`
    (an existing, uncommitted-conclusion exploration script already in this
    repo) had already measured that snapping shut is what makes contact
    non-deterministic; its fix (ramp the close gradually) was generalized
    into `generate_frames()` itself so every caller gets it, not just that
    one script.
  **Contact explosion during the close segment -- found, then substantially
  (not fully) reduced, 2026-09-14.** Even with both fixes above, one trial
  had the CUBE -- not the arm -- launched from its resting pose to over 2m
  away during the (now-gradual) close segment, right after tracking had
  converged to 20mm; another trial had the ARM fling itself 90cm straight up
  instead. Two causes chased down and fixed in `isaac_sim_common.py`:
  - `add_gripper_colliders` put a single `convexHull` on every gripper mesh,
    including the inner-finger pads (`left_inner_finger`/`right_inner_finger`
    -- confirmed by listing the un-instanced mesh names: `finger4step` is the
    pad body, `fingertipsstep` the rubber insert). A hull can only puff a
    concave grip face outward, so the close could start already
    interpenetrating the object before the solver saw a normal contact. Those
    two links now get `convexDecomposition` instead; everything else (rigid
    housing that never touches the object) keeps the cheaper `convexHull`.
  - Nothing capped how fast PhysX may separate two interpenetrating bodies.
    Added `PhysxRigidBodyAPI.maxDepenetrationVelocity = 0.5` (m/s) on both the
    cube (`add_shape`) and the two pad links -- a real overlap still resolves
    within a few ticks, but a bad one can no longer produce a multi-meter
    single-tick launch.
  **Measured effect** (reach_probe, gripper enabled, `steps_per_segment=90`,
  4 trials after both fixes, only counting trials where the approach itself
  had converged before the close segment -- an unconverged approach is the
  timing gap above, not this bug): 3/4 ended in an ordinary few-cm settle: 2
  clean, 1 with a temporary ~15cm cube shift that recovered. The 4th still
  drifted the tool ~300mm during the close -- much smaller than the >2m/900mm
  launches before, but not zero. **Net effect: no longer catastrophic, not
  yet reliable.** The remaining ~300mm case is consistent with the residual
  ~20-40mm tracking offset from the timing fix above: against a 40mm cube,
  that's a large fraction of the object's own size, so some closes are
  simply off-center enough to still interpenetrate a corner rather than
  land flat on a face. Narrowing that residual offset further (tighter
  RMPflow convergence, or a compliant/vision-guided re-center just before
  closing) is the next thing to try, ahead of any more solver tuning --
  the collision/solver changes here have likely done most of what they can
  for a grasp that starts this far off-center.

- **The wrist camera is aimed wrong and this is the top blocker.** Confirmed
  by inspecting the frames of `smoketest/clipfix_check`, the run collected to
  verify the 2026-09-08 camera fixes: the base camera came out correct (the
  cube peaks at 139 px) but the wrist camera images background grid and empty
  space for the entire episode -- never the cube, the gripper or the table,
  with half of every frame near-black. openpi's UR5 contract feeds the policy
  base **and** wrist views, and the wrist view is the one that carries the
  fine manipulation signal, so training on this would waste the run. Fix with
  `check_cameras.py --sweep_wrist` before collecting, then pin the winner in
  `WRIST_CAMERA_FLANGE_ROT_EULER` (which currently reads `None`, meaning
  "use the derivation that produced this bug").
  **Update (2026-09-09):** the sweep ran and rejected all six directions --
  every one rendered solid black, not "aimed at the sky". The camera was
  buried inside the UR5e's own `wrist_3_link`: the original mount put it 5 cm
  from the flange origin, well within that link. Direction was never the
  problem. The mount is now `WRIST_CAMERA_STANDOFF_M` along the direction the
  camera looks, and `check_cameras.py` sweeps standoff as well as direction.
  **Update (2026-09-15): the sweep ran for real on live Isaac Sim, and found
  a second bug behind it -- both cameras' vertical FOV was never set.** The
  first live run picked `(90, 0, 0)` at lateral 0.08m, but by a 688-vs-679
  margin the script itself flagged as not a clear winner, and the contact
  sheets showed why: `wrist_rolls_at_grasp.png` -- four renders of that same
  direction, rotated only about its own viewing axis -- showed the cube at
  roll 0 and at literally none of rolls 90/180/270. A pure in-plane
  rotation cannot make real content vanish, only move it; the only
  explanation left was a non-square field of view trading horizontal reach
  for vertical as it turns. `_setup_cameras` confirmed it: both the base and
  wrist cameras call `CreateHorizontalApertureAttr` but never
  `CreateVerticalApertureAttr`, so vertical sat at USD's schema default
  (15.2908mm) regardless of the horizontal value computed from
  `*_HORIZONTAL_FOV_DEG` -- on the wrist camera's 24mm focal length that is
  a real vertical FOV near 35 degrees against the intended 70 horizontal,
  rendered onto a SQUARE 256x256 image the whole project has been treating
  as square-FOV. Every frame either camera has ever rendered was vertically
  compressed relative to horizontal by that ratio, this ADJUST item's own
  suspicion two sections above having gone unchecked until the roll sweep
  forced the question. Fixed by setting each camera's vertical aperture
  equal to its horizontal one (correct for this project's square
  `CAMERA_RESOLUTION`; a non-square resolution would need the vertical
  aperture scaled by its own aspect ratio instead). Re-running the sweep
  after the fix picked a DIFFERENT direction, `(0, 90, 0)` at lateral
  0.08m, by a clearly wider 1532-vs-1011 margin, and that direction's own
  rolls now agree with each other (present at 90/180/270, occluded by the
  gripper itself only at roll 0 -- a real occlusion, not the vanishing act).
  Pinned as the new default. One preflight run afterward showed
  `wrist_rgb cube_px=0` at a *different* random grasp, but `base_rgb` for
  that same frame shows why: tool was 59mm from the cube, at the edge of
  the convergence warning band, cube visibly still on the table beside the
  gripper rather than under it -- the grasp-residual tracking issue this
  file already documents (see "Bring-up order" step 1 and Track B above),
  not a new camera problem. The wrist mount looks at where a converged
  grasp puts the cube; it was simply given a grasp that had not converged.
- **`pivot_dwell_check.py` was run for the first time 2026-09-15, and its own
  stdout-loss bug ate the first run before a second run's numbers pointed at
  contact, not dynamics.** The first run produced no output at all despite
  69 seconds of real simulation and no exception -- the exact
  fastShutdown-swallows-block-buffered-stdout failure `check_cameras.py`
  already documents and works around with its own `say()`/`_REPORT`
  mirroring, which this script did not have; the 2026-09-15 01:17 fix
  upstream had only added a `flush=True` print on the exception path, not
  the (much larger) success path. Fixed the same way: every `print()` in
  this file now goes through a `say()` that flushes immediately.

  With that fixed, both runs (with and without `--no-gripper`) produced
  real numbers, but at a scale the file's own docstring (24mm -> 43mm) did
  not prepare for: 300-600mm, not tens of mm. The reason is methodological,
  not physical -- `main()` commands `hold()` straight from `scene.reset()`'s
  home configuration in one shot (432mm and 500mm initial error at tick 1
  in the two runs), so most of `DWELL_SAMPLES`' tick budget (which the
  original 24/43 observation implicitly assumed started near-converged) is
  spent on the initial approach, not a settled hold -- and the pivot test's
  60-tick samples land mid-approach for the same reason, so its spread/mean-
  offset numbers are not yet a trustworthy TCP-frame measurement either.

  One comparison inside the noise is clean, though, and it answers the
  question this test exists to answer: at tick 180, free space converges to
  9.4mm (with gripper) and 13.7mm (without) -- both settled, gripper or not.
  The grasp target converges to 16.6mm without the gripper, but WITH it
  diverges: 229.9mm (tick 60) -> 146.6mm (tick 90) -> 205.8mm (tick 120) ->
  575.6mm (tick 180), growing after appearing to improve. Present only with
  the gripper AND only at the position where its fingers reach the table is
  exactly the CONTACT row of this file's own decision table, and the scale
  (fully diverging, not settling to an elevated residual) matches this
  project's own documented contact-explosion history (`isaac_sim_common.py`'s
  `convexDecomposition` + `maxDepenetrationVelocity` fixes for the transient
  close segment) more than a stable one -- worth checking whether that same
  fix holds up under a SUSTAINED hold rather than the brief close-and-lift
  those fixes were tuned against, which is what neither this run nor those
  fixes' own validation has tested. Not yet done this session: re-running
  with a longer tick budget (or a target the arm starts already near) to get
  a genuine settled-residual number and a trustworthy pivot spread.
- **The scripted expert fails to reach the cube on most random spawns --
  found 2026-09-15 while trying to pick a wrist camera mount, and it turns
  out to be the real reason no mount could be found, not a camera problem
  at all.** Started as: pick a `check_cameras.py --sweep_wrist` winner and
  it kept picking a different "clear" winner each run (688px, then 1532px,
  then 1011px) that turned out to be one lucky cube spawn each time -- a
  held-out check of 4 fresh spawns against three of those exact settings
  came back 0-534px, mostly 0. That forced fixing the sweep itself first
  (see the entry below), and scoring every candidate over several spawns
  instead of one immediately reported `NO MOUNT WORKS` -- correctly, this
  time: of 91 grasp attempts across the fixed sweep, **80 (88%) never
  reached the cube** (tool-to-cube distance at the nominal grasp point:
  median 130mm, worst 290mm, only 11 under the 60mm bar this file's own
  bring-up table checks against). No camera placement fixes an arm that
  is not there.

  The cause is in `scripted_pick_place.py`, not RMPflow: `steps_per_segment
  =90` is a FIXED tick budget per segment regardless of how far that
  segment actually has to travel. The docstring's own validation ("converges
  to ~40mm before the close segment starts") was measured once, not across
  `CUBE_X_RANGE`/`CUBE_Y_RANGE` -- the first segment's distance (arm's fixed
  start pose to `above_cube`) varies with where the cube randomly spawned,
  and 90 ticks (1.5s) that comfortably converges a short reach can leave a
  long one most of the way there and no further, every single time, at
  exactly frame 225/630 (the fixed point three segments end at) regardless
  of how far off that leaves the tool. This is upstream of, and likely
  explains a good share of, Track B's own dwell numbers above (432mm and
  500mm initial error commanded in one shot from reset) and probably
  `collect_demos.py`'s real attempt/reject rate, neither of which has been
  re-examined with this in mind yet. Not yet done this session: making
  `steps_per_segment` (or just the first segment's) scale with the actual
  Cartesian distance instead of being a flat constant, then re-running the
  reach-rate check above to see how much of the 88% failure rate that
  actually closes.
- **`check_cameras.py --sweep_wrist`'s single-sample scoring was itself the
  bug that hid the finding above -- fixed 2026-09-15.** Scoring at one
  random cube spawn cannot tell a mount that is reliably mediocre from one
  that is occasionally excellent and usually useless, and there was no
  reason to expect otherwise until three different "clear winners" in a row
  each failed a held-out check. `sweep_wrist` now takes `--wrist_samples`
  (default 5) and resamples the cube for every one of them at the grasp
  stage (at_reset does not need to -- the arm's reset pose does not depend
  on where the cube spawned), ranks candidates by their WORST sample rather
  than their single or mean score, and warns explicitly when a winner's
  worst sample sits under half its mean -- the exact shape of the deception
  this fix closes. `WRIST_CAMERA_FLANGE_ROT_EULER`/`_LATERAL_M` are left
  uncommitted to a "winner" for now: with the reach failure above still
  open, no mount can be honestly validated, since most of the samples any
  sweep takes are measuring a grasp that never happened.
- **Base camera framing is small, and it is geometry, not occlusion --
  corrected 2026-09-15, this entry previously said the opposite.** The cube
  peaked at 139 of 65,536 px over a whole episode (~1 px at reset, first
  clearly visible around frame 90), which this entry used to read as the arm
  occluding the task. `isaac/camera_framing.py` (added since, no Isaac Sim
  needed to run it) checks that against pure trigonometry and it is not an
  occlusion number at all: a 4 cm cube at this camera's distance subtends
  about 14 px across (~200 px2) regardless of what else is in frame, and the
  workspace that has to stay in frame (cube spawn spread + place target, 0.50
  m of it) caps it at ~20.5 px across (419 px2) at the current 256 px
  resolution -- no repositioning or lensing choice beats that ceiling, only
  shrinking the required workspace or raising resolution does. Moving the
  camera to "a less occluded viewpoint" would not have changed the number.
  `pick_place_scene.py` already derives its preflight floor
  (`MIN_CUBE_PIXELS_FLOOR`, via `BASE_FRAMING.usable_peak_area_px`) from this
  same ceiling rather than a flat guess, so the collection-time gate is
  already geometry-aware; what is still an open choice, not yet made, is
  whether to also act on it before a full run -- `camera_framing.py`'s own
  output spells out the three levers and their cost: tighten `CUBE_Y_RANGE`/
  `PLACE_TARGET_POSITION` to ~0.34 m of spread, raise `CAMERA_RESOLUTION` to
  ~375 px, or accept the base view as coarse context and lean on the wrist
  camera (already ~3.9x larger at the grasp) for the fine approach signal.
- `isaac/isaac_sim_common.py`: exact Robotiq gripper asset path and its
  drive-joint name/limits are unverified placeholders (least-certain part
  of the whole pipeline -- everything downstream assumes this works).
- `isaac/isaac_sim_common.py`: the gripper now comes from the UR asset's own
  `Gripper` USD variant set, so its joints belong to the **arm's articulation**
  -- which is what Isaac's own manipulator API assumes (`ParallelGripper`
  drives joints by index within the robot, `SingleManipulator` wraps that
  arrangement; see Isaac's `test_single_manipulators.py`). This replaced two
  approaches that were each tried and each confirmed broken, both of which
  kept the gripper as a *separate* articulation: a per-tick pose teleport,
  whose finger links diverged to world positions around 1e15 m, and a
  `UsdPhysics.FixedJoint` weld, which made PhysX absorb the gripper and kill
  the app during `World.reset()`. **Unverified:** whether the ur5e asset ships
  that variant set at all (the ur10e one ships `Robotiq_2f_140`).
  `select_gripper_variant` lists the asset's actual variants if the expected
  names are missing, and the scene prints the robot's joint names at startup
  so a variant that silently did nothing is visible immediately.
- `openpi_integration/ur5e_pick_place_policy.py`: the import paths for
  `DataConfig`/`DataConfigFactory`/`AssetsConfig`/`ModelTransformFactory`
  are inferred from openpi's docs, not fetched verbatim from
  `config.py`'s own imports -- check against your checkout if they don't resolve.
- `vla_bridge/robot_interface.py`: real-gripper POSITION CONTROL is still a
  placeholder relay unless a `gripper_driver` is passed (duck-typed, see that
  file's docstring -- a Robotiq over its socket interface supplies both
  position and the gOBJ object-detection byte; the discrete I/O coupling
  keeps object detection but loses position feedback).
  **Gripper SENSING is no longer the blocker it was.** Both backends now
  return a `GripperState` that says whether the value was actually measured
  (`vla_bridge/gripper_state.py`), the sim reads the position it was already
  publishing and the client was discarding, and a fallback to the command
  echo is announced rather than silent. Three things followed from that echo
  and are now fixed: the policy could not observe a grasp that failed to
  close, the eval metric scored `is_holding` off the COMMAND so a run could
  report success having picked nothing up, and inference fed the command
  while the demonstrations were recorded with the measured position -- a
  train/serve mismatch. What remains is wiring a real driver on real
  hardware.
- Fixed 2026-09-09, worth knowing about when reading older notes: the ROS 2
  control loop used to share one `ReentrantCallbackGroup` with its
  subscriptions, which let the timer re-enter itself (rclpy's
  `ReentrantCallbackGroup.can_execute()` returns True unconditionally, and the
  executor re-arms a timer as soon as it is *taken*). Since `_run_step`
  blocks far longer than its 100 ms period, runs overlapped -- publishing
  conflicting joint targets and calling `policy.infer()` concurrently on a
  single websocket. The timer now has its own `MutuallyExclusiveCallbackGroup`
  while subscriptions stay reentrant, so polled state still refreshes.
- Fixed 2026-09-09: `move_joints` on real hardware used blocking `moveJ`,
  which re-plans a full trapezoidal profile per tick -- stuttering motion, and
  each call outlasting the control period. It now streams with `servoJ`,
  whose `servo_time` is wired to the node's `control_hz`. The old behaviour is
  still available as `move_joints_blocking` for discrete repositioning.
  Untested on hardware; tune `servo_lookahead_time`/`servo_gain` at low speed.
- The scripted demonstrations are, well, scripted -- crisp and
  perfectly-timed in a way real teleoperated human demos aren't. Worth
  swapping in real teleoperation data once this pipeline is proven
  end-to-end, especially before trying tasks scripted logic can't easily
  generate (anything requiring visual feedback mid-motion, not just a fixed
  waypoint sequence).
- `isaac/pick_place_scene.py`'s `BASE_CAMERA_FOCAL_LENGTH_MM`/aperture
  setup (for Phase 6's `camera_projection.py` math to be accurate) **did
  not** behave the standard way -- confirmed and fixed 2026-09-15, see the
  wrist-camera entry above: `verticalAperture` was never set on either
  camera, leaving it at USD's schema default regardless of the horizontal
  value computed from `*_HORIZONTAL_FOV_DEG`, a real vertical/horizontal FOV
  mismatch on a nominally-square render. Now set equal to the horizontal
  aperture on both cameras. If detected objects' localization error (printed
  by `hybrid_pick_place_demo.py`) is still consistently large on your
  install, check the actual rendered FOV against `BASE_CAMERA_HORIZONTAL_FOV_DEG`
  again -- that was the right instinct, just not chased down until a
  wrist-camera roll sweep forced the question.
- Phase 6's comparison against the pure-VLA pipeline isn't apples-to-apples
  yet: `pi0_ur5e_pick_place` was only ever fine-tuned on one cube (Phase 1's
  single-object data). A fair comparison needs Phase 1/2 redone with
  multi-object demonstrations first (noted in Phase 6's own section above).
- Phase 7 (OpenVLA) is the biggest open risk in this project: the RLDS/TFDS
  builder (`openvla_integration/ur5e_pick_place_dataset_builder.py`) has
  never been run through `tfds build`, and the OpenVLA-side registration
  snippet's enum names (`StateEncoding.POS_EULER`, `ActionEncoding.EEF_POS`)
  are inferred from the general OXE convention, not fetched from your
  checkout's actual `configs.py` -- verify both against an existing config
  entry there before trusting them.
- **`get_cube_position()` has been reading a FROZEN, wrong cube position
  since at least whenever this scene started recreating the cube prim on
  every reset -- found and fixed 2026-09-16, and this retroactively puts a
  question mark over every prior measurement that scored against it.**
  Chasing the 88% reach-failure finding above further (why did giving the
  approach MORE time, then a whole extra settle segment, change nothing?)
  led to logging cube position every tick instead of once: the cube
  "jumped" 100-300mm in a single tick at frame 0 of episodes 2+ in a run,
  to a position that turned out to be an EARLIER episode's actual spawn
  point, with the arm still essentially at its reset pose -- not a
  physically possible contact event. `check_pose_readout_multireset.py`
  (a faithful repro of `reset()`'s actual prim lifecycle, not
  `check_pose_readout.py`'s existing single-add test, which does not
  reproduce this) confirmed it directly: `prim_world_pose`'s
  `ComputeLocalToWorldTransform` reads correctly on the very first episode
  of a process and then FREEZES at that first episode's transform forever
  -- 0, 156, 184, 260, 267mm of error against 5 fresh episodes' true spawn
  points, never once correcting itself no matter how many further resets
  or physics steps ran. Root cause, as best determined: `reset()` did
  `stage.RemovePrim(CUBE_PRIM_PATH)` then re-`add_cube`-d a brand new prim
  at that same path every episode; Fabric's own stage/sim-history cache
  (keyed by prim path, and per this same session's own shutdown log --
  `gFabricState->gUsdStageToSimStageWithHistoryMap had 1 outstanding
  SimStageWithHistory(s)` -- not obviously invalidated by a raw
  remove-and-recreate at that path within one still-running Kit session)
  is the leading candidate, though not independently confirmed beyond that
  log line agreeing with the symptom.

  **This means every prior claim in this file that scored against
  `get_cube_position()` -- yesterday's 88% reach-failure rate, today's
  earlier "the cube gets launched 0-301mm" findings, and by the same logic
  `collect_demos.py`'s own `grasp_succeeded`/`place_error_m` scoring and
  `cube_pixels_visible`'s ground truth -- was comparing against the WRONG
  cube position for every episode after the first in whatever process ran
  it.** Neither the `steps_per_segment` distance-scaling fix nor the
  `SETTLE_TICKS` dwell added earlier this session were wrong to add (both
  are still reasonable, and neither was reverted), but neither had
  anything real to fix: more time cannot help a comparison that was never
  measuring the cube's true position to begin with.

  Fixed by not repeating the mistake `TARGET_MARKER_PRIM_PATH` already
  avoids: the cube prim is now created ONCE (guarded by the same
  `IsValid()` check the marker uses) and repositioned in place on every
  reset via the new `set_rigid_body_translation` (isaac_sim_common.py --
  reuses the prim's existing translate op rather than adding a second one,
  and explicitly zeroes velocity since this prim is now reused rather than
  fresh every episode) instead of being torn down and rebuilt.

  **With the measurement fixed, a real (if messier) picture emerged from
  12 fresh with-gripper episodes**: most (8 of 12) now show plausible
  numbers -- tool tracking 11-40mm off its intended target, cube drift
  28-236mm, roughly the scale this file's other sections already expect
  from a light 50g object near a closing gripper. But two clearly did not:
  one episode's cube moved **5.26 metres** in the approach -- squarely
  this project's own already-documented contact-explosion shape (see the
  `convexDecomposition`/`maxDepenetrationVelocity` entries elsewhere in
  this file), just not fully closed off by those fixes -- and two others
  showed genuine large tracking misses (304mm, 412mm) unrelated to the
  cube at all.

  **This "fix" was itself incomplete, found the same day (2026-09-16,
  second pass) chasing why a 34-sample `--sweep_wrist` run still failed to
  reach the cube 94% of the time (32/34, 100-400mm off) with the above fix
  already in place and no gripper attached at all.** `set_rigid_body_translation`
  writes the new position to a raw USD translate op BEFORE calling
  `world.reset()` -- but `world.reset()`'s Stop+Play cycle does not
  re-read that attribute the way it looks like it should. Stop() snaps a
  simulated rigid body back to the pose PhysX cached from its very FIRST
  Play, discarding any USD-attribute write made since. Confirmed directly
  (`diag_reset_stages.py`, scratchpad): across 10 fresh resets with 10
  different random spawns, the cube landed at the exact same
  `[0.396, -0.177, 0.02]` after every single `world.reset()` call, matching
  neither that episode's spawn nor any physically sensible drift -- a fixed
  constant, not noise. Everything measured against `get_cube_position()`
  between the two 2026-09-16 fixes (the 8/12-good 12-episode number above,
  the 5.26m explosion, both `NO MOUNT WORKS` sweep results) inherited this
  bug too and should be treated the same way the frozen-read numbers were:
  suggestive of scale, not to be cited as a rate.

  Fixed by repositioning through a persistent
  `isaacsim.core.prims.SingleRigidPrim` instead: construct it once (only
  possible after the first `world.reset()` has Played), call
  `.initialize()` every episode the same way `self.robot` already is, then
  `.set_world_pose()` AFTER `world.reset()`, not a raw USD op before it --
  `set_world_pose` writes directly into the live PhysX rigid-body view
  (`RigidPrim.set_world_poses`'s own `physics_view.set_transforms(...)`
  path), which Stop() cannot discard because nothing is Stopping in
  between the write and the next read. `set_rigid_body_translation` is
  removed (no remaining callers). **Verified**: 15 fresh episodes,
  `with_gripper=False`, cube drift exactly **0.0mm** in every one; the full
  `--sweep_wrist --wrist_samples 5` sweep (90 grasp attempts) then reached
  the cube in **17-55mm** in literally every sample, zero
  `WARNING: the tool did not actually reach` lines. See "The next Isaac Sim
  session" above for what this now genuinely shows about the wrist camera
  mount (a real `NO MOUNT WORKS`, not a reach-confounded one) and what's
  still unverified (the 5.26m explosion, `pivot_dwell_check.py`'s residual).
