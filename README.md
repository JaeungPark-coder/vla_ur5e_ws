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

### 2026-09-21, continued: merged in a parallel session's independent work

Before any of the friction/stiffness follow-up above happened, `git push`
was attempted and rejected -- `origin/master` had moved 6 commits ahead
without this session's knowledge, from what turned out to be a second,
independent investigation running in parallel against the SAME open
problems (the wrist mount / cube range / missing light / stiffness
questions above). Per explicit direction, resolved this with a real
`git merge origin/master` (not "pick one side and discard the other"),
reviewing every conflict by hand rather than taking either side blind.
What that surfaced, file by file:

- **`isaac_sim_common.py`, `pivot_dwell_check.py`,
  `scripted_pick_place.py` -- auto-merged cleanly**, no manual conflicts,
  but NOT free of overlap: the remote session had independently added its
  own gripper-pad friction material binding
  (`bind_grip_friction_material`/`GRIP_MATERIAL_PRIM_PATH`) and its own
  gripper-drive stiffness fix (`GripperController._fix_drive_gains`,
  `finger_joint` kp raised 171.89 -> 20000), while this session had an
  UNCOMMITTED, still-stashed friction-material implementation of its own
  (`get_or_create_friction_material`/`bind_physics_material`,
  `FRICTION_MATERIAL_PRIM_PATH`) sitting on top of the same file. Because
  git can't see stashed changes during a merge, there was no textual
  conflict -- but popping that stash back will reintroduce a real,
  un-auto-resolvable duplication (two friction-material prims, two
  binding calls) that still needs a human/model decision, not just a
  merge tool. **Not yet reconciled** -- tracked as an open item below.
  `scripted_pick_place.py`'s remote change also added a SETTLE_TICKS
  segment between descend and close (a fix for the same "cold-start isn't
  comparable" issue this session had separately flagged in
  `pivot_dwell_check.py`), which likely means `check_grasp_alignment.py`'s
  hardcoded assumption of 4 waypoint segments (`waypoints[:4]`,
  `segment_names`) is now off by one and needs checking before that script
  is trusted again.
- **`pick_place_scene.py` -- 4 real conflicts, resolved by hand.** The
  DomeLight addition was the same fix independently found by both
  sessions (see "Scene lighting -- FIXED" above and the remote session's
  own writeup of the identical discovery further down this file) --
  kept once, with a comment crediting both. `CUBE_X_RANGE`/`CUBE_Y_RANGE`
  were narrowed to DIFFERENT (overlapping) boxes by each session for
  different reasons (this session: RMPflow edge instability; remote:
  wrist-camera FOV coverage) -- resolved to their INTERSECTION,
  `(0.42, 0.48) x (-0.10, 0.10)`, on the reasoning that either session's
  own justification for narrowing still holds inside the smaller box.
  `WRIST_CAMERA_LATERAL_M`/`WRIST_CAMERA_FLANGE_ROT_EULER` were a genuine
  disagreement at merge time -- **RESOLVED 2026-09-22**, see that item in
  the numbered list below for the decisive A/B test.
- **`check_cameras.py` -- 1 conflict, resolved by union.** Both sessions
  extended the same `WRIST_LATERAL_CANDIDATES` sweep list independently;
  kept the remote's candidate values (with its own history comment) and
  appended this session's separate `WRIST_DOWN_TILT_CANDIDATES` addition
  alongside it -- the two additions don't overlap in what they vary, so
  there was no real choice to make here, just a mechanical union.
- **This file (`README.md`) -- 2 conflicts, resolved by keeping both
  sessions' narratives** rather than deleting either: this session's new
  "2026-09-21 update" section stays at the top as the current status, the
  remote session's own detailed 2026-09-19 writeup (including its
  independent discovery of the same missing-dome-light bug, from its own
  `diag_lighting.py` check) stays below under "Old ... next session plan"
  with a note that it's superseded for the specific numbers but not
  wrong -- it's real, corroborating work, not a discarded draft.

**New open items the merge itself created:**
1. **RESOLVED 2026-09-22.** Reconciled the two friction-material
   implementations (this session's `get_or_create_friction_material`/
   `bind_physics_material` vs. the remote's already-integrated
   `bind_grip_friction_material`) by deleting this session's duplicate and
   standardizing on the remote's, which was already wired into
   `add_gripper_colliders`/`add_shape` and carries a documented empirical
   basis. `check_grasp_alignment.py` re-pointed at `GRIP_MATERIAL_PRIM_PATH`.
   Also fixed a second, related merge fallout while in there: this file's
   hardcoded 4-segment waypoint slice (`approach/settle/descend/close`)
   was stale against `scripted_pick_place.py`'s now-5-segment layout
   (`approach/settle/descend/settle/close`, from the remote's 2026-09-19
   `SETTLE_TICKS` fix) -- updated to match.
2. **Still open.** `wrist_3_joint` stiffness: this session measured
   `(1000.08, 0.0043)` via `get_joint_drive_gains` (PhysX
   `UsdPhysics.DriveAPI` directly); a remote-session code comment claims
   `kp=57300` via `ArticulationController.get_gains()`. These may be
   reading different things (authored USD drive params vs. the live
   solver-side value) -- not yet checked which, or whether they actually
   disagree once units are equalized. Not tested 2026-09-22 (session ran
   out of scope before reaching this item) -- still next-session work.
3. **RESOLVED 2026-09-22, and it turned out not to be a real
   contradiction.** Re-examined this session's own 2026-09-21 tick data
   (`left_dx`/`right_dx` per-pad vs. `dy` per-pad): averaging the two
   pads' `dy` per episode (+1.0, -2.05, +2.2, +4.5mm) reproduces almost
   exactly the remote session's own single-point measurement of "Y within
   +-5mm" -- because the remote session only ever tracked ONE point (the
   formula-based grip-point, effectively the average of both pads), a
   symmetric left/right closing-axis deviation cancels out of that
   average and becomes invisible. The remote's X residual range (-8 to
   +26mm) likewise matches this session's per-pad common-mode `dx`
   magnitude (16-28mm) closely. **Conclusion: both sessions measured the
   same underlying drift correctly, just at different resolutions** --
   X really is the dominant, common-mode positioning error (what the
   remote saw), and Y really is where the smaller true closing-axis
   asymmetry lives (what this session's per-pad decomposition saw). Not
   a disagreement to adjudicate, just two granularities of one dataset.
4. **RESOLVED 2026-09-22 by a direct paired A/B test** (`ab_wrist_mount.py`,
   new script, 8 identical cube spawns rendered under both candidates in
   the same run). `(-90,0,0)@0.12` (remote/2026-09-19) beat
   `(180,0,0)@0.16` (this session/2026-09-21) decisively: worst-case
   198px vs. 0px, over an identical spawn sequence. `pick_place_scene.py`
   updated to the winner; `WRIST_CAMERA_DOWN_TILT_DEG` reset from 15 to 0
   in the same change since 15 was tuned only for the losing mount and
   was never measured against the winner (both the remote's original
   validation and today's A/B test used tilt=0 throughout) -- re-sweeping
   tilt specifically for the new mount is unstarted follow-up work, not
   assumed to be optimal at 0.
5. **Still open, and now has a strong new lead -- see the new section
   directly below.** Whether a friction-stable, geometrically-good static
   hold actually translates into a successful dynamic lift.

### 2026-09-22: fully-merged baseline still fails, and a strong new lead why

With everything above reconciled (both sessions' fixes combined, plus
today's mount A/B), `check_grasp_alignment.py --episodes 5 --hold-ticks 0`
was re-run as the actual outcome test -- the first time 2026-09-19's three
fixes (settle2, friction, gripper kp) and 2026-09-21's four fixes (mass,
cube range, lighting, wrist mount) have ever run together. **0/5 lifted**,
`max_cube_z` 0.022-0.033m against `LIFT_Z_THRESHOLD=0.08m` -- directly
answering this session's opening question (does 2026-09-19's work, applied
on top of the current merged state, solve the live blocker): no, not on
its own, and not combined with everything from 2026-09-21 either. This
matches 2026-09-19's own commit message, which already said as much before
any of 2026-09-21's fixes existed -- still true now that they do.

**But the fresh per-tick trace surfaced something neither session had
seen before, because it could only appear once both sessions' fixes
coexisted:** during `settle2` (the remote's 2026-09-19 addition -- a
90-tick dwell between descend and close, added to kill residual velocity
before closing), the LEFT pad's own mesh bbox (not the formula-based grip
point) sits 30-39mm BELOW the cube's top face in every one of the 5 fresh
episodes (range across episodes: -2mm to -39mm) -- i.e. the still-OPEN
gripper pad is pressing down into the cube for the entire 90-tick dwell,
well before closing ever starts. As the `close` segment then runs and the
fingers actually close, the overlap steadily shrinks (-39mm -> -29mm in
the traced episode) -- consistent with the Robotiq 2F-85's real four-bar
linkage geometry, where an OPEN pad's tip hangs measurably lower than a
CLOSED one's.

**Likely mechanism:** `GRASP_HEIGHT=0.02m` targets the cube's top face
with zero clearance (known since 2026-09-16), and `GRIPPER_TCP_OFFSET_M
=0.12m` was calibrated against the CLOSED-pad geometry (this session's own
earlier finding 1, above, found the pad centroid only +2.5 to +16.2mm off
at END of close -- i.e. well-calibrated for that state). But the descend
and settle2 segments hold the gripper at that same zero-clearance target
while the fingers are still OPEN, and an open pad's true lowest point
sits lower than the closed-pad calibration assumes -- driving the real
pad tens of mm into the cube during a 90-tick STATIC hold, before the
fingers ever start closing. This is a plausible, previously-invisible
root cause: an open, static, deeply-embedded pad dwelling in a cube for
90 ticks would very plausibly push, tip, or otherwise displace it before
closing has any real object left to grip cleanly -- consistent with the
persistent ~10-30mm common-mode `dx` drift documented above, and with
`max_cube_z` barely lifting off its resting height rather than the cube
simply falling out of an otherwise-good grip.

**Why neither session could have found this alone:** 2026-09-19 had no
per-pad mesh measurement (only the formula-based grip point, which can't
see pad/cube interpenetration at all) and its trace predates the settle2
segment it was itself adding, so there was no 90-tick static dwell for it
to observe yet. 2026-09-21's own earlier axis analysis (finding 4,
"contact hypothesis REJECTED", above) checked whether `dx` drift onset
lined up with contact -- but that trace also predates settle2 (this
session's own pre-merge code only had ONE settle segment, before descend),
so there was likewise no sustained open-pad dwell in contact for it to
have seen. The dwell-in-contact phenomenon is a genuine interaction
effect between the two sessions' fixes, only visible after both were
merged and re-measured together.

**Not yet done:** this is a lead, not a confirmed fix. Candidate next
steps, none attempted yet: (a) re-measure `GRIPPER_TCP_OFFSET_M`
specifically against the OPEN-pad mesh bbox (this session's finding 1
methodology, `mesh_world_bbox_center`, re-run with the gripper at
`GRIPPER_OPEN_POS` instead of closed) to get the actual open-state
offset error directly rather than inferring it from the overlap depth;
(b) give the descend/settle2 target extra Z clearance (approximately the
open-state offset error) and only complete the final descent as part of
the close segment itself, so the open pad never has to dwell inside the
cube; (c) shorten or remove `settle2`'s dwell specifically at the
now-corrected height once (a) or (b) land, since the dwell's own original
purpose (kill residual velocity before closing) doesn't require it to
happen already-embedded in the object.

### 2026-09-22, continued: the stiffness sweep, finally measured live

A pasted analysis earlier today claimed specific friction/stick-slip
experiment results against this project (2x stiffness, static friction
0.9 -> 0.5/0.6, "stick-slip" degrading 3/4 episodes to 86-200mm, an
IPC-GraspSim citation) -- checked against this repo and this session's
own record, NONE of it had actually been run here: only the underlying
tooling existed (`get_joint_drive_gains`/`set_joint_drive_gains`, the
`--stiffness-scale` CLI flag, `bind_grip_friction_material`'s 0.9/0.7
defaults already in code). Re-measured from scratch, live, seed=42 paired
comparisons throughout, per the same request that caught the discrepancy:
verify before building on it, exactly as this project's own culture
already demands.

**Hold-phase** (`--hold-ticks 300`, frozen target, static equilibrium
offset only -- NOT comparable to a real lift, see this script's own
docstring):
  - 1x: 96.6, 98.9, 45.5, 178.9mm
  - 2x: 151.4, 137.1, 161.2, 20.4mm -- 3/4 episodes WORSE, one
    dramatically better. Both runs: `any non-finite joint state: no`,
    so this is a real equilibrium shift, not incipient divergence.

**Actual lift outcome** (`--hold-ticks 0`, the same `max_cube_z >=
LIFT_Z_THRESHOLD` criterion `collect_demos.py` gates on):
  - 1x: max_cube_z 0.0227, 0.0278, 0.0223, 0.0303m -- 0/4 lifted
  - 2x (damping left at x1): 0.0317, 0.0322, 0.0386, 0.0340m -- 0/4
    lifted, but improved in ALL 4 episodes -- the OPPOSITE direction
    from what the hold-phase proxy predicted for 3 of those 4.
  - 3x stiffness + 3x damping this time (scaled together, to hold the
    damping ratio roughly constant rather than drift toward
    underdamped): 0.0200, 0.0290, 0.0338, 0.0690m -- 0/4 lifted,
    non-monotonic against 2x (below it in 3/4 episodes), but episode 4
    jumped to 0.069m -- 86% of the 0.08m threshold, the closest any
    tested setting has come today.

**Two real lessons, not the ones the unverified report would have
taught:** (1) the hold-phase proxy and the actual lift outcome pointed in
OPPOSITE directions for the same 1x->2x change -- trust `--hold-ticks 0`,
not the frozen-target proxy, exactly as this file's own docstring already
warns. (2) Stiffness's effect on the real outcome is not linear even
holding the damping ratio constant: a third point (3x) landed BELOW 2x in
three episodes and produced the single best result of the day in the
fourth. Extrapolating "how many x clears 0.08m" from two points would
have been the same mistake this project has made before with a proxy
metric.

**Not yet done:** a 4th stiffness point to see whether the episode-4-style
jump is reproducible or a one-off; a friction sweep now that stiffness
alone is established as directionally real but insufficient (best case
0.069m, still under threshold); and `wrist_3_joint`'s own still-unresolved
stiffness-measurement discrepancy (`1000.08` via `get_joint_drive_gains`
vs. a remote-session comment's `57300` via
`ArticulationController.get_gains()`) matters here specifically, since it
bears on what "x1" is actually a multiple of.

### 2026-09-23: the wrist_3_joint unit mismatch above is resolved, and a paired human+model review of the rest of the codebase found nine more real bugs

**The `1000.08` vs `57300` discrepancy two sections up is a units mismatch,
not two different numbers competing to be true.** `get_joint_drive_gains`
reads `UsdPhysics.DriveAPI` off the USD prim (degree-based, per Isaac
Sim's own Articulation Controller docs: "Angular units are expressed in
radians while angles in USD are expressed in degrees and will be adjusted
accordingly by the articulation controller"); the `57300` figure came from
`ArticulationController.get_gains()` (radian-based, live PhysX view).
`57300 / 1000.08 = 57.2954`, `180/pi = 57.2958` -- agrees to 4 significant
figures. `get_joint_drive_gains`/`set_joint_drive_gains` (both used by the
wrist sweep above) read AND write through the same USD-degree API
consistently, so the sweep's own 1x/2x/3x results are unaffected by this --
nothing to re-measure there. What this resolves is only the cross-
comparison: `GripperController._fix_drive_gains`'s `kp=20000` (below) is in
the OTHER (radian, live-controller) unit space, so comparing it to the
wrist joints' USD-degree numbers needs the 57.3x conversion, not a raw
read.

**Then a session-long paired review (a pasted external analysis
cross-checked line-by-line against this actual repo, not trusted at face
value) found real bugs across the parts of this project nobody had
re-read since they were written -- worth recording in one place since
several of these were invisible without directly executing code, not
just reading it:**

1. **`GripperController._fix_drive_gains` only ever ran ONCE per process,
   guarded by `_resolve_joint_index`'s `if self._drive_joint_index is
   None` -- true only for episode 1**, since this controller instance
   persists for the whole multi-episode run. It writes only to the live
   `controller.get_gains()`/`set_gains()` view, never authoring back into
   USD (unlike the wrist joints' `set_joint_drive_gains`) -- and this
   project's own `rescale_gripper_mass_to_spec` docstring already
   establishes why that matters: "USD physics schema is only re-parsed at
   Play time", which is exactly why mass has to be authored into USD to
   survive a reset. Every episode's `world.reset()` is a hard Stop+Play
   that invalidates the arm's physics handles (`robot.initialize()` is
   redone every episode BECAUSE of this) -- so `finger_joint` most likely
   ran at its intended `kp=20000/kd=500` for episode 1 only, and silently
   reverted to the as-shipped `171.89/0.0115` (47% closure in free space)
   from episode 2 on, in every multi-episode run since the 2026-09-19 fix
   landed. The "98.75%" closure figure that fix's own verification cites
   came from a single-episode diagnostic (`diag_gripper_gains_mimic.py`),
   which could not have exposed this. Fixed: `GripperController.
   reapply_drive_gains()`, called from `PickPlaceScene.reset()` right
   after `robot.initialize()`, every episode.
2. **The scripted expert's place waypoint drove the cube CUBE_Z (2cm) into
   the table.** `at_cube` and `at_target` both added `+GRASP_HEIGHT`, but
   to positions in different reference frames: `cube_position` is the
   cube's CENTRE (resting on the table), `target_position`
   (`PLACE_TARGET_POSITION`) is the TABLE SURFACE. Invisible until now
   because no episode has ever reached the place phase, and
   `place_error_m()` only checks XY. Fixed by deriving the same
   cube-centre-height offset from `cube_position`'s own z (correct
   per-object in the multi-object scene too).
3. **Two `validate_dataset.py` blind spots**, found by constructing
   adversarial episodes and feeding them straight to `validate_episode`:
   all frames byte-identical to the first (a frozen/stale camera with a
   perfectly normal-looking state/action trace) passed cleanly, and so did
   a gripper that closes and never reopens (no place/release phase, so
   "gripper moved at all" was satisfied by the one close). Both fixed. The
   existing `good_episode`/`correct_episode` test fixtures turned out to
   already have exactly these two defects -- fixed the fixtures too,
   rather than loosen the new checks to fit them.
4. **`collect_demos.py`'s cube-visibility gate only fired on `n_success ==
   1`** -- catches a broken-from-the-start camera but not a mid-run
   regression, since the scripted expert drives off ground-truth
   `cube_position`, not vision, so a later episode can lift+place
   correctly (get saved) with the cube barely visible in its own recorded
   frames. Now checked on every SAVED episode.
5. **`params.yaml` was silently reverting the control_hz/gripper_driver/
   lifted_z_threshold fixes below** (item 6) **back to their old values at
   launch.** `ros2 launch` loads `parameters=[params_file, ...]`, which
   overrides `vla_policy_client.py`'s own `declare_parameter` defaults --
   so fixing the Python defaults alone was a no-op for anyone launching
   via `vla_bridge.launch.py`, the path this README documents. Fixed in
   the yaml too.
6. **Three train/serve mismatches**, found by cross-referencing constants
   across files: (a) `LIFT_Z_THRESHOLD` (0.08, what `collect_demos.py`
   gates a saved demo on) was duplicated as a stray `0.03` in
   `residual_rl_train_env.py`, `openvla_pick_place_demo.py`, and
   `vla_policy_client.py`'s `lifted_z_threshold` default -- the last of
   which is the flag `eval_mode` writes to `results_csv` as "success", so
   a 1-3cm lift `collect_demos.py` would reject was being logged SUCCESS
   there. All three now import `LIFT_Z_THRESHOLD` or copy it with a
   sync-manually comment, instead of redeclaring it independently. (b)
   `control_hz` defaulted to 10.0 while demonstrations are recorded at
   `CONTROL_HZ=60.0` and only `action_chunk[0]` is applied per control
   tick (re-inferring every tick) -- serving at 10Hz replayed 60Hz-sized
   deltas roughly 6x too slowly. Raised to 60.0; whether a real UR5e can
   sustain a `policy.infer()` round trip at 60Hz is still unverified. (c)
   `gripper_driver` defaulted to `'none'` (unmeasured gripper, command
   echoed back as observation) for the real-hardware path, contradicting
   the node's own runtime warning that this is a train/serve mismatch --
   changed the default to `'robotiq_socket'`.
7. **`hybrid_pick_place_demo.py` had no lift check at all** -- success was
   final XY placement error alone, so shoving the object to within 30mm of
   the target counted identically to an actual grasp-lift-place. Since all
   three pipelines share one `results_csv` schema specifically for a
   direct comparison, this made hybrid structurally easier to "win" for a
   reason unrelated to task performance. Added `max_object_z` tracking and
   required it past `LIFT_Z_THRESHOLD`, matching the other two pipelines.
8. **`camera_projection.py`'s forward axis was +Z; USD cameras image along
   local -Z; the hybrid pipeline's every single trial returned "camera ray
   never crosses the table plane" as a result.** Reproduced directly
   against this project's real `BASE_CAMERA_POSITION`/`BASE_CAMERA_AIM_
   POINT` and the actual rotation matrix `_look_at_quat` would build for
   it: `ray_world`'s z came out positive (pointing up and away from the
   table) when it needed to be negative, so `pixel_to_table_position`
   returned `None` for every pixel -- indistinguishable in the logs from a
   rare, correctly-handled edge case, but actually 100% of runs,
   independent of every grasp/lift issue elsewhere in this project. Not
   caught by the module's own round-trip smoke test because projection and
   its inverse always agree with EACH OTHER under whatever convention
   either uses, by construction. Fixed (`pixel_to_camera_ray`'s z to -1.0,
   `project_point_to_pixel`'s depth to `-direction_cam[2]`), and added two
   smoke tests that duplicate `_look_at_quat`'s math (can't import
   `pick_place_scene.py` directly -- needs Isaac Sim at import time) to
   catch a repeat.
9. **`llm_command_parser.py`'s JSON extraction only stripped a fence at
   the very START of the response** -- the two most common real
   deviations (a leading sentence before a fenced block, and a leading
   sentence with no fence at all, neither starting with a fence) still
   failed. Replaced with try-whole-response, then a fence found anywhere
   via regex, then a quote/escape-aware balanced-`{...}` scan as a last
   resort. Added this
   module's first test coverage (`test_llm_command_parser.py`) since it
   had none.

**Confirmed fine, no fix needed:** the cube's collider (native
`UsdGeom.Cube`, an exact analytic box, not a mesh needing convexHull/SDF
approximation at all); the two different pixel-visibility thresholds
between `collect_demos.py` and `validate_dataset.py` (deliberately
different jobs -- one geometry-derived and collection-time, one a
portable post-hoc floor); `EULER_SEQ='xyz'` (checked directly against
OpenVLA's own `prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py`,
which uses `tfg.rotation_matrix_3d.from_euler`, `R = R_z R_y R_x` -- scipy
`as_euler('xyz')` agrees to 0.000000000deg, `'zyx'` would have disagreed by
up to 145deg).

**Still open, deliberately not fixed without a decision or a live run:**
OpenVLA's base-camera view puts the cube at ~14px across, which 224x224
resizing shrinks under one ViT patch -- narrowing the workspace, raising
`CAMERA_RESOLUTION`, or adding the wrist camera to the OpenVLA collector
(favoured: already exists, already used by pi0, and would make the
pi0-vs-OpenVLA comparison itself more apples-to-apples) are the three
options, unpicked. `feasibility_gate.py`'s own `workspace_ok` (pure numpy,
run directly, no Isaac Sim needed) rejects the grasp AND place waypoints
outright -- `at_cube`/`at_target` sit at z=40mm across the whole cube
spawn range, all under `Z_MIN_M=0.08`, the same height regime as the
gate's own CONFIRMED failing case (elbow wind-up at z=22mm). Whether that
means the height genuinely needs raising (a pedestal under the cube) or
`Z_MIN_M` was just an untested-but-safe guess needs `check()`'s
condition-number/manipulability numbers, which need the live Lula solver
-- next session's task, not resolved here.

**None of the nine fixes above have been run against a live Isaac Sim
session** -- every one was found and fixed by reading code, cross-
referencing constants, or running the pure-numpy/pure-Python pieces
directly (`workspace_ok`, `camera_projection.py`, `_extract_json_text`,
etc.) outside Isaac Sim. `py_compile` and `pytest test/ -q` (130 passed,
up from 121) are clean throughout, which rules out syntax and regression
against the existing suite, not correctness against the simulator. The
next Isaac Sim session's first job is re-running the finger-gain sweep
(now that fix 1 makes it mean what it's supposed to across episodes,
which the original 2026-09-22 sweep numbers above may not have) and
checking whether each of these actually holds up live.

### 2026-09-23, continued: E-4 and B-1 measured live, F-3 found broken a second time, and eval_mode hits a new blocker before its own fix can even be exercised

Picked up the "next session" list from the section above, in the order it
gave: E-4's condition number, B-1's finger sweep, then F-3/F-4 live.

**E-4 resolved -- `Z_MIN_M=0.08` is not guarding a real singularity here.**
Added `check_grasp_height_condition.py`, which runs `check()`'s own
IK+Jacobian math directly (bypassing `workspace_ok()`'s height cutoff) at
the real `at_cube`/`at_target` corners and a z-sweep from 20mm to 140mm at
the cube-spawn centre. Result: `condition_number` is flat at 6.9-8.5 and
`manipulability` at 0.048-0.078 across the *entire* range, including
z=20mm -- nowhere near `CONDITION_SOFT=17`. This xy/z region is not
near-singular by this measure, so `Z_MIN_M=0.08` looks like an
untested-but-safe guess rather than something this task's own geometry
needs. Caveat: the gate's own CONFIRMED bad case (elbow wound to 162deg)
was *also* never run through this IK/Jacobian check -- `test_feasibility_gate.py`
rejects it on the height check alone, before IK -- so this doesn't
retroactively explain that incident, which came from `check_singularity.py`
driving RMPflow directly (dynamic tracking behavior this static check may
not capture). Didn't lower `Z_MIN_M` on this alone; the scripted lift
already drives through z=40mm every episode, so watch for wind-up there
before trusting this number over the original incident.

**B-1 reconfirmed live: the gain-reapplication fix holds, still 0/5
lifted, and a new alignment clue.** Ran `check_grasp_alignment.py
--episodes 5 --hold-ticks 0 --seed 42` at baseline (stiffness/damping x1).
max_cube_z per episode: 22.7 / 39.2 / 23.2 / 28.4 / 39.9mm -- no
episode-1-only advantage and no monotonic decay, which is what fix 1
(reapply drive gains every episode) was supposed to produce and does. No
non-finite joint state either. Still 0/5 past the 80mm threshold. New in
this run's diagnostics: at the end of the close segment, *both* fingertip
mesh bboxes read the same-direction ~+16 to +17mm X offset from the cube
centre (left: dx=+16.9mm, right: dx=+17.3mm) -- not a left/right asymmetry,
the whole gripper is closing off-centre in X, consistently. Worth checking
against the wrist-camera alignment tooling before the next sweep.

**F-3 was broken again -- same bug, different commit.** Re-grepped
`hybrid_pick_place_demo.py` after pulling the "nine bugs" round and found
`policy = ScriptedPickPlace(...)` missing *again*: the commit that added
the lift check (item 7 above, tracking `max_object_z`) rewrote this same
block and dropped the assignment a second time while keeping the
now-orphaned `policy.generate_frames()` call below it. Re-added it (commit
`eecb4c5`), with a comment explaining why it keeps disappearing -- the
constructor's `target_position` parameter (the place destination) and this
function's own `target_position` local (the perceived pick location) share
a name, and whoever's editing this block reads that collision as "already
wired" and doesn't notice `policy` itself never got assigned.

**F-4's own fix is still unverified live -- eval_mode hits a bigger
blocker first.** Set up ROS2 for real: this machine already has ROS2 Iron
installed system-wide (not Humble/Jazzy, which is all Isaac Sim 5.1 bundles
internally), and the two do not mix -- a system-Iron `ros2` CLI joining the
graph crashed the Isaac-Sim-side bridge process outright (segfault inside
`librmw_dds_common` deserializing `ParticipantEntitiesInfo`, a cross-FastDDS-version
wire mismatch in ROS2's own internal graph-discovery topic, not this
project's code). Worked around it by running `vla_policy_client.py` itself
under Isaac Sim's bundled Humble `rclpy` too (Isaac Sim's kit python is
3.11; system Iron is built for 3.10, so the two were never binary-compatible
anyway) -- `source setup_python_env.sh` plus `ROS_DISTRO=humble`,
`AMENT_PREFIX_PATH=<isaac-sim>/exts/isaacsim.ros2.bridge/humble`,
`LD_LIBRARY_PATH+=.../humble/lib`, `PYTHONPATH+=.../humble/rclpy`, run with
`<isaac-sim>/kit/python/bin/python3` directly (no `SimulationApp` needed on
the client side). `cv_bridge` isn't in Isaac Sim's bundled humble libs at
all; `pip install cv_bridge` (a pure-Python wheel, no compiled bindings)
into Isaac Sim's own `python.sh` environment fixes that on the bridge
side. With both sides on the same distro, real cross-process ROS2 messaging
was confirmed working (`/vla/joint_state`, images via `cv_bridge`, all
real data). Tested with `openpi`/the LLM stack deliberately left out of
scope -- `vla_policy_client.py`'s `WebsocketClientPolicy` import is local to
`__init__`, not module-level, so a throwaway stub package (hold-still
action, cycles the gripper) standing in for `openpi_client` drives the
real, unmodified control loop without needing a checkpoint or a server.

That got an eval run going -- and immediately found something more
fundamental than the caching bug F-4 fixes: **`pick_place_scene_bridge.py`'s
Kit app cleanly shuts itself down** ("Simulation App Shutting Down", no
crash, no traceback) the moment it processes the first `/vla/eval/reset`
while a ROS2 client is attached. Reproduced twice, in the same spot both
times: bridge runs fine solo for 12+ minutes, client connects, trial 1
completes normally (max_steps reached, logged FAILURE), client publishes
the reset, bridge calls `scene.reset()`, app exits. Other scripts
(`check_grasp_alignment.py`, `sweep_gripper_axis.py`) call `scene.reset()`
repeatedly all the time with no issue, so it isn't `reset()` itself --
something specific to resetting while `enable_extension("isaacsim.ros2.bridge")`
is loaded and a client is connected, in headless mode. Root cause not
found. This blocks eval_mode from completing even one reset, which means
F-4's cache-clearing fix (still believed correct by code review -- the
mechanism it targets is real and the fix is a straightforward instance of
the existing "wait for fresh data" guard) has **not** been exercised
against a live trial boundary yet. Next Isaac Sim session: chase this
shutdown before anything else eval_mode-related, since nothing past it is
reachable.

### 2026-09-23, continued again: E-4's asymmetry resolved, F-3 static-analysis-proofed, B-1's frame-mismatch hypothesis ruled out live

**E-4 fully resolved.** Ran `check_grasp_height_condition.py` a third time,
this time against the gate's own CONFIRMED wind-up pose itself
([0.4618, 0.0636, 0.0222]): condition_number=7.7, manipulability=0.0574 --
indistinguishable from every other pose in the 6.9-8.5 range already
measured. The asymmetry from the write-up above is gone: the original
wind-up was never a manipulability problem, so `Z_MIN_M`'s rationale needs
to come from RMPflow's dynamic tracking behavior, not kinematic
conditioning, which this class of check cannot see at all.

**F-3's failure mode is now structurally prevented, and it caught a second,
independent bug immediately.** Two separate commits had each independently
deleted `hybrid_pick_place_demo.py`'s `policy = ScriptedPickPlace(...)`
line while editing the same block for an unrelated reason -- a pattern that
code review alone was 0-for-2 against. Ran `ruff check --select F821,F823`
(undefined-name / used-before-assignment, pure AST, no imports or Isaac Sim
needed) across the whole repo: F821 was clean (confirming F-3 itself is
fixed), but it found a live F823 in `pivot_dwell_check.py` -- a
`from isaac_sim_common import ROBOT_PRIM_PATH` inside `main()`, duplicating
the same name already imported at module level, which makes Python treat
every use of that name *anywhere in `main()`* as the local one, including
an earlier use before the local import line ever runs. `UnboundLocalError`,
but only when `--stiffness-scale`/`--damping-scale` is passed with the
gripper on -- exactly how the next finger sweep would call it. Fixed
(removed the redundant local import) and added `test_static_analysis.py`
so `pytest test/` catches this class of bug going forward without needing
Isaac Sim.

**B-1's new fingertip-offset clue: not dynamics, and NOT a frame mismatch
either -- still open.** Extended `pivot_dwell_check.py`'s `hold()` with a
per-axis variant (`hold_xyz`/`report_dwell_xyz`) to see the signed
grip-point error instead of just its norm. Result: **free space** holds a
constant dx=-4.7 dy=-3.2 dz=-5.7mm from tick 1 through tick 180 (zero
growth); **at the grasp pose**, a constant dx=+24.1 dy=+6.4 dz=+26.8mm,
also zero growth. Two things this rules out cleanly:

- Not gravity sag *growing over the hold* -- both are flat from the first
  sample.
- Not a base-frame mismatch between RMPflow's internal kinematics and the
  world/USD frame everything else (cube position, grip targets) is
  expressed in -- added `check_rmpflow_base_frame.py`, which reads
  `rmpflow.get_end_effector_pose(q0)` and compares it directly against the
  flange prim's actual `prim_world_pose()` at the same joint state (the
  position half of a comparison `reset()`'s wrist-camera setup already does
  for rotation only, at `r_flange_to_tool0`, but never checked for
  position). They agree to **0.0mm**. RMPflow's FK is not lying about where
  the arm actually is.

What's left, and not yet tested: `settle_to()` runs 360 ticks *before*
`hold()`'s tick-1 sample, so a gravity/stiffness steady-state sag that
fully develops during that settle phase would look identical to a static
offset in this test -- "no growth after tick 1" does not distinguish "no
sag" from "sag that already finished." The offset's sign and magnitude
also scale with reach/height (small and negative in free space at 0.35m,
large and positive at the ~0.04m grasp pose), which is what load-dependent
steady-state P-control error looks like, not what a fixed coordinate
transform error looks like (that would be closer to constant regardless of
target). Next check: rerun `pivot_dwell_check.py --stiffness-scale 3` (or
higher) and see if the static offset shrinks -- if it's stiffness-limited
steady-state sag, more stiffness should shrink it; if it doesn't move, that
argument is wrong too and something else is going on.

### 2026-09-23, F-4 finally resolved: the eval_mode "silent shutdown" was never about reset at all

Instrumented `pick_place_scene_bridge.py`'s main loop with an actual
exception handler (it had none -- an unhandled exception just vanished
into Kit's headless shutdown, `--installSignalHandlers=0`/`--no-window`,
with no trace) and reproduced with `--n_trials 1`, which never publishes
`/vla/eval/reset` at all. Same silent-looking death. Reset was never the
cause -- it was coincidental timing in the earlier runs, not causation.

The real exception: `ValueError: shape mismatch: value array of shape
(1,6) could not be broadcast to indexing result of shape (1,12)`, in
`scene.robot.apply_action(ArticulationAction(joint_positions=<6 values>))`
at `pick_place_scene_bridge.py`'s inline joint-target handler. Confirmed
via a live `scene.robot.num_dof` log that the articulation is 12-DOF (6
arm + gripper) from the very first tick, not something that changes
mid-session. `ArticulationAction`'s `joint_indices` defaults to `None`,
which means "joint_positions must cover every dof" -- passing 6 values
against a 12-dof articulation without it is wrong regardless of what
triggers the actual raise (still not fully explained -- roughly 80 ticks
ran without incident before it happened once; some Isaac-Sim-internal
articulation-view state, not this project's code, decides exactly when).
Not a GPU/CUDA issue: checked directly, no CUDA/memory errors anywhere
near the crash, GPU sat at ~400MB/16GB the whole time, and the traceback
itself is pure CPU-side numpy indexing
(`isaacsim.core.utils.numpy.tensor.assign`).

Fixed by passing `joint_indices=np.arange(6)` explicitly, in **both**
places this exact unindexed pattern existed:
`pick_place_scene_bridge.py`'s inline handler and
`pick_place_scene.py`'s `apply_joint_targets()` (used by
`residual_rl_train_env.py`) -- the second one had just been luckier so
far, not correct, and would have hit the identical crash under the same
articulation-view condition.

**Verified live end-to-end after the fix:** a real 3-trial `eval_mode` run
(bridge + `vla_policy_client.py`, both on Isaac Sim's bundled Humble
`rclpy` per the ROS2 setup two sections up, `openpi` stubbed out) now
completes all 3 trials across 2 real resets with zero crashes -- CSV shows
`1,False,80,-1.0 / 2,False,80,-1.0 / 3,False,80,-1.0` (FAILUREs are
expected and correct: the stub policy holds still and toggles the
gripper, it was never going to lift the cube). This is also the first
live confirmation that **F-4's original cache-clearing fix works**: no
false SUCCESS at either trial boundary, and the "waiting for camera
images" guard visibly engages for the ticks right after a reset before
each new trial's data is fresh.

### 2026-09-23, B-1's --stiffness-scale sweep: confirms the alignment offset, but NOT the lift itself

Ran `pivot_dwell_check.py --stiffness-scale 3 --damping-scale 1.7320508`
(sqrt(3), to hold the damping ratio constant while scaling stiffness --
this file's own argparse help already warns that raising stiffness alone
is "a classic source of the exact contact blow-up this project has
already hit"). Result: the static per-axis offset the previous entry
found (free space dx=-4.7/dy=-3.2/dz=-5.7mm, grasp pose
dx=+24.1/dy=+6.4/dz=+26.8mm) **collapses to near-zero** at 3x stiffness --
free space dx=-0.1/dy=-0.3/dz=-0.0mm, grasp pose dx=-0.5/dy=-0.1/dz=-0.4mm.
A 48-67x reduction at the grasp pose, well beyond the ~3x a naive
error-proportional-to-1/stiffness P-control model predicts. Confirms the
offset is stiffness-limited steady-state error, not a fixed geometric bug
-- B-1's alignment question is resolved.

**But this does not fix the actual grasp-lift problem.** Immediately
tested whether the same damping-matched 3x stiffness improves real
grasp-and-lift outcomes: `check_grasp_alignment.py --episodes 5
--hold-ticks 0 --seed 42 --stiffness-scale 3 --damping-scale 1.7320508`.
Result: max_cube_z = 20.0 / 46.0 / 25.4 / 25.3 / 38.8mm -- still **5/5
lifted=no**, barely different from the unscaled baseline measured earlier
today (22.7 / 39.2 / 23.2 / 28.4 / 39.9mm). The earlier 2026-09-22
stiffness sweep's 0/4-at-every-setting result was NOT an artifact of
unmatched damping -- properly damping-matched stiffness still doesn't
close the ~34mm gap to `LIFT_Z_THRESHOLD=0.08m`.

The two questions this project has been treating as related turn out to
be separable: **holding a commanded pose accurately** (B-1, now solved by
stiffness) and **actually lifting the cube** (A-1, still 0% regardless).
Whatever limits the lift is not the same steady-state positioning error
pivot_dwell_check measures -- it's something specific to the
close-and-lift dynamics (finger contact/friction during closing, grip
centering during the CLOSE motion rather than a static hold, or genuinely
needing gravity compensation once the cube's weight is added as an actual
payload once grasped, which `pivot_dwell_check.py` never tests -- it
holds a pose with nothing in the gripper). The gravity-compensation /
effort-control path (get_generalized_gravity_forces() +
hand-computed PD + Jacobian-based payload compensation, researched this
session but not yet implemented) is still the next candidate specifically
for the closed-gripper/lifting case -- not for the alignment problem this
sweep just closed.

### 2026-09-23, continued again: close/lift dynamics, gravity comp researched but not yet needed, and a new 30%-rate divergence at 3x stiffness

**Gravity compensation researched (not implemented -- the data below
argues against reaching for it yet).** PhysX joints are one control mode
at a time -- position OR effort, never "add torque on top of the PD
drive" -- so real gravity compensation means switching the arm's joints
to effort control entirely and computing
`tau = kp*(q_des-q) + kd*(-qdot) + G(q)` by hand every tick.
`SingleArticulation.get_generalized_gravity_forces()` (same family as
`get_coriolis_and_centrifugal_forces()`/`get_mass_matrices()`) already
computes `G(q)` from the loaded USD/mass data, no manual UR5e dynamics
derivation needed. Two pitfalls documented by others hitting this exact
problem: damping must NOT be driven fully to 0 in effort mode (an NVIDIA
forum UR16e report found this "recommended" setting unstable in
practice, ~10 damping needed); and `get_generalized_gravity_forces()`
only knows the robot's own mass, not a grasped payload's -- an IsaacLab
discussion thread reports `set_external_force_and_torque()` as a
workaround that "had no effect while picking." `feasibility_gate.py`'s
existing `_numerical_jacobian` would extend to a payload term
(`tau_payload = J^T . F_gravity_cube`) if this becomes necessary.

**Friction confirmed bound correctly (not the problem).** Added
`check_close_lift_dynamics.py`, which reads `GRIP_MATERIAL_PRIM_PATH`
directly: static=0.9, dynamic=0.7, shared between the pads and the cube
as designed. No friction fix needed.

**Close-and-lift dynamics, logged tick-by-tick through close AND lift
(check_grasp_alignment.py's own log stops at end-of-close and never
looks at lift) -- 3 episodes, 3 different failure shapes:**
gripper closes to 97.6% with no resistance and barely rises (2.7cm);
gripper stalls at 60.1% (real contact) and rises the most (4.6cm) but
then slips back down with 212mm of sideways drift during lift; gripper
closes to 97.1% with no resistance and gets shoved 80mm sideways instead
of lifting. Real contact happened in only 1 of 3, and even that one
didn't survive the lift -- separate problems, not one.

**Cross-referencing against check_grasp_alignment.py's existing per-tick
dz log refines (not replaces) the squeeze-reaction mechanism this project
documented back on 2026-09-21.** At 3x stiffness, the pre-close
`settle2` phase converges to dz=19.6-19.9mm (right at the 20mm
`GRASP_HEIGHT` target) and sits completely flat there -- ruling out
"needs more `SETTLE_TICKS`" as the next lever; that convergence is
already done before close even starts. But partway through the CLOSE
motion itself -- specifically once closure crosses roughly 60-80% toward
the commanded 100% -- dz grows sharply again, +10 to +20mm in the last
20-40% of closing, in episodes where the fingers keep being driven
toward full closure without a firm early stop (dz stays flat all through
close in the one episode that DID stall early at 60%). Same mechanism
2026-09-21 named ("closure% is the real independent variable, not tick"),
now pinned down more precisely: 3x stiffness fixed the STATIC steady-state
part of the problem but not this ACTIVE squeeze-reaction part, which is a
transient contact force, not a steady load -- consistent with why gravity
compensation (aimed at steady loads) isn't obviously the right next tool
for this half of the problem either.

**New finding, arguably higher priority than the squeeze-reaction
question above: 3x stiffness diverges outright in 30% of episodes, and it
correlates with which side of the workspace the cube spawns on.** One of
the 3 episodes above never got near the cube at all -- `dz` stuck at
163mm through the whole close segment (should be ~20mm), with 100mm+
swings in `dx` during descend that never damped out. Not a fluke: ran 10
more seeded episodes at the same 3x stiffness / damping-matched setting
and counted 3/10 with the identical signature (dz stuck at 163-168mm,
cube never leaves spawn height). Laid out each episode's cube-spawn Y
against outcome: **all 3 divergent episodes spawned at negative Y; all 6
convergent episodes with clearly-negative or positive Y split 0
divergent-at-negative / 6 convergent-at-positive**, with the one
near-zero-Y episode (y=-0.0099) the sole exception that still converged.
Six-episode-long streak of positive-Y episodes with zero divergences
against 3-of-4 negative-Y episodes diverging is a real workspace-side
asymmetry, not noise -- something about 3x stiffness behaves differently
on one side of the robot's reach (candidate causes not yet checked:
RMPflow picking a different elbow-up/elbow-down configuration depending
on target sign, or joint-limit-avoidance terms that aren't symmetric).
This means roughly 30% of today's "still 0/N lifted" statistics at 3x
stiffness are contaminated by a failure that has nothing to do with
squeeze-reaction or grip force -- the arm just never got there. Not yet
determined whether this divergence is new at 3x stiffness or was already
present at baseline and simply never sampled (no baseline run has been
checked against this specific signature). Next Isaac Sim session:
this -- not squeeze-shove -- is the natural next thing to chase, since it
undermines the reliability of every 3x-stiffness conclusion above it,
including today's B-1 resolution itself.

### 2026-09-24: verified an external gripper-tuning comparison against the real IsaacLab source, added the tooling it recommends, laptop-only

**Yesterday's "squeeze-shove" hypothesis (the close-phase dz growth
2026-09-23 pinned to closure crossing ~60-80%) now has a specific,
externally-sourced candidate cause, checked against the primary source
rather than trusted from a pasted summary.** A comparison against
IsaacLab's own reference Robotiq 2F-85 config claimed this project's
`finger_joint` stiffness (20000) is ~1176x IsaacLab's (17). Verified by
fetching `franka.py` directly from `github.com/isaac-sim/IsaacLab` (main)
rather than trusting the claim: `FRANKA_ROBOTIQ_GRIPPER_CFG`'s
`gripper_drive` actuator is exactly `stiffness=17, damping=0.02,
effort_limit_sim=1650`; `gripper_finger` (the inner finger joints) is
`stiffness=0.2, damping=0.001, effort_limit_sim=50`; `gripper_passive`
(the knuckle/follower joints) is `stiffness=0.0, damping=0.0,
effort_limit_sim=1.0`, with the source comment `"set PD to zero for
passive joints in close-loop gripper"` -- confirmed to the exact number
and the exact comment text, not approximately. This project's own
`finger_joint` has no effort limit set at all (grep-confirmed) and the 5
follower joints have kept their as-shipped `171.89/0.0115` drive the
whole time -- `_fix_drive_gains` only ever touches `finger_joint` itself.
IsaacLab's own config also sets `disable_gravity=True` on this asset (a
real caveat: the reference isn't fighting the gripper's own weight the
way this project's does), but the squeeze force itself is `k*delta_theta`
regardless of gravity, so the stiffness comparison is still apples to
apples for "how hard does an over-commanded closing target push."

**Added the tooling to test this, not the change itself** -- consistent
with how `--finger-kp`/`--finger-kd` were added 2026-09-22 rather than
silently changing the 20000/500 baseline default:
- `isaac_sim_common.set_joint_max_force` -- effort limit via the same
  `UsdPhysics.DriveAPI` pattern `set_joint_drive_gains` already uses.
- `isaac_sim_common.resolve_gripper_follower_joint_names` /
  `zero_follower_joint_drives` -- discovers the gripper's non-drive
  joints by walking the `Gripper` subtree and checking
  `UsdPhysics.Joint`, rather than hardcoding IsaacLab's
  Franka+Robotiq joint-name patterns (`.*_inner_finger_joint` etc.)
  against this project's different asset (a bare UR5e + `Robotiq_2F_85`
  variant, not Franka+Robotiq) -- a wrong hardcoded guess would silently
  match nothing, so this discovers the real names instead and
  loud-warns if it still comes back empty.
- `check_grasp_alignment.py --finger-effort-limit`/`--zero-follower-pd`,
  alongside the existing `--finger-kp`/`--finger-kd`. IsaacLab's exact
  pairing is one command away: `--finger-kp 17 --finger-kd 0.02
  --finger-effort-limit 1650 --zero-follower-pd`.

**Deliberately not added in the same pass: `armature` and
`solver_position_iteration_count`.** The same external comparison
(referencing IsaacLab's `FRANKA_PANDA_CFG` armature and the lift task's
solver iteration counts) argues these matter for the unrelated 3x-stiffness
30% Y-side divergence above -- a real, separate lead worth chasing -- but
their exact PhysX schema attribute names were not confirmed against a
live run or a second primary source the way the gripper gains above were,
and a wrong guess on an unfamiliar API is more likely to silently no-op
than the well-established `DriveAPI` calls above (which fail loud via
`AttributeError` if wrong). Left as a next-session code addition rather
than risk that.

**None of this has been run against Isaac Sim yet** -- same posture as
2026-09-23's nine-bug round: the IsaacLab numbers were confirmed by
fetching the actual source file, and the new tooling's own Python is
`py_compile`/`pytest`/`ruff` clean (131 passed, 0 F821/F823), but whether
`--finger-kp 17 --finger-kd 0.02 --finger-effort-limit 1650
--zero-follower-pd` actually reduces the close-phase dz growth is
untested. **Next Isaac Sim session priority, in order:** (1) the 3x-stiffness
30% Y-side divergence flagged at the end of 2026-09-23 -- still first,
since it undermines any conclusion drawn at 3x stiffness including
squeeze-reaction measurements; (2) this session's IsaacLab-reference
gripper pairing, cheap to try immediately after, directly targeting the
squeeze-reaction mechanism the divergence chase doesn't touch.

### 2026-09-25: the armature idea above is retracted, and a specific, code-confirmed root cause for the squeeze problem replaces it -- priority order revised

**Retracted: `armature`/`solver_position_iteration_count` as a fix for the
30% Y-side divergence.** The reasoning above (an explicit-integration
stability limit roughly `k*dt^2/I`, armature raising the effective `I`)
assumes the arm's joint drives compute an explicit force from gains and
state at the START of the timestep. Checked directly against PhysX's own
docs (`nvidia-omniverse.github.io/PhysX/physx/5.6.0/docs/Articulations.html`,
fetched, not assumed): **articulation drives are implicit** -- "the
position and velocity constraints imposed by the drive... are with
respect to the end of the time step, and not... an explicit, constant-
during-time-step drive force." An explicit-integration stability argument
does not apply to an implicit solve. (One nuance: the specific supporting
citation used alongside this -- "armature is an IsaacLab remedy
specifically for explicit actuators, per IsaacLab#2497" -- did not hold up
under its own re-check of that issue, which discusses ImplicitActuator
documentation confusion but never mentions armature at all; the core
PhysX-implicit-drives fact above is independently confirmed regardless,
this citation specifically just isn't.) Good news buried in the
retraction: deferring armature/solver-iteration tooling 2026-09-24 for
"API uncertainty" turned out to also dodge a wrong mechanism, not just an
uncertain API -- worth remembering as a habit, not just a one-off save.

**Specific, code-confirmed mechanism for the squeeze/shove problem
(blocker 1 above), upgrading follower-joint PD from "worth trying" to
leading hypothesis:** `GripperController.set_target` (isaac_sim_common.py)
commands `ArticulationAction(joint_positions=[target],
joint_indices=[idx])` -- confirmed by reading it again, `idx` is
`finger_joint`'s own index ONLY. The 5 follower/knuckle joints' own drive
targets are never touched by any code in this project, ever -- they keep
pulling toward whatever target the USD asset shipped with, while the
mimic constraint drags their POSITION along with finger_joint regardless.
A mimic constraint enforces relative position, not that the follower's
own PD drive goes idle -- a live drive still generates torque against
whatever its own unmoved target says, and that torque adds into the
linkage's overall equilibrium. This reframes 2026-09-19's own finding
(free-space closure stuck at 47% of target, "fixed" by raising finger_joint
kp 116x) as likely: the follower drives were never a non-issue as their
own lockstep-tracking confirmation implied (lockstep proves relative
position, not that their drive torque is zero) -- kp=20000 didn't remove
their resistance, it overpowered it, and it overpowers it just as hard
once contact starts (matching IsaacLab's own reference config, which sets
the follower/passive joints' PD to exactly zero: "set PD to zero for
passive joints in close-loop gripper"). A quick self-consistency check: a
single-opposing-spring model (`theta = theta_target * k_drive /
(k_drive + n*k_follower)`) fit to the one 47%-at-kp=171.89 data point
(giving n=1.14) predicts 99.0% closure at kp=20000 -- close to the
measured 98.75%, out of sample. Not proof (two data points, illustrative
only), but consistent with a real opposing torque of about this
magnitude, and no other candidate of this size has been identified.

**Revised priority: try this BEFORE chasing the 30% Y-divergence, not
after.** No new code needed -- `--finger-kp`/`--zero-follower-pd` already
exist (this file's own previous section). Cheapest decisive test: free
space (no cube), 2x2 grid of {follower PD as-shipped, zeroed} x
{finger_kp 171.89 (as-shipped), 17 (IsaacLab reference)}, read steady-
state closure fraction. Predicted if this hypothesis is right: zeroing
follower PD alone gets as-shipped kp=171.89 to ~100% closure (no 116x
finger-kp hike needed at all), and kp=17 with zeroed follower PD also
reaches full closure, just slower. If true, this may also make the whole
30% Y-divergence investigation moot: a soft, correctly-behaving parallel
gripper self-centers on what it grips, so the ~16mm alignment offset that
motivated sweeping arm stiffness up to 3x (and produced the divergence)
might not need correcting via arm stiffness at all once the gripper
itself stops fighting its own follower joints.

### Operational note: `check_cameras.py` is unsafe to import from

**CONFIRMED 2026-09-22, the hard way:** `check_cameras.py` calls
`SimulationApp(...)` at module level (line ~64), unconditionally --
not guarded by `if __name__ == "__main__":`. A new script
(`ab_wrist_mount.py`, written today for the wrist-mount A/B test above)
imported `_render_wrist`/`_metrics`/`_drive_to_grasp` from it after
already calling its own `SimulationApp(...)`, which silently tried to
start a SECOND Kit instance in the same process. This corrupted Kit's own
native app registry and crashed identically 3 times in a row (`libomni.
kit.app.plugin` -> a `std::unordered_map::operator[]` segfault inside
`_app.cpython-*.so`), each crash slowly symbolicated by the crash
reporter (~14s/frame via `addr2line`) before the process actually died.
Fixed by reimplementing those three helpers directly in
`ab_wrist_mount.py` instead of importing them. **Any future script that
wants check_cameras.py's helper functions must copy them, not import
them, until check_cameras.py's own `SimulationApp(...)` call is moved
under an `if __name__ == "__main__":` guard** (not done here -- out of
scope for this session, and every existing caller only ever runs it as
`__main__`, so it wasn't broken until something else tried to import it).
A second, separate gotcha found in the same incident: killing a Kit
process with `kill -9` (rather than letting it exit normally through
`simulation_app.close()`) can leave `/dev/shm/sem.carbonite-sharedmemory`
behind, and the NEXT Kit launch on the machine will crash on startup
trying to use it -- `rm -f /dev/shm/sem.carbonite-sharedmemory
/dev/shm/*RStringInternals*` before retrying fixed it both times this
happened today.

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

**2026-09-19 update.** With reach fixed (above), `check_cameras.py
--sweep_wrist --wrist_samples 5` still reported `NO MOUNT WORKS` (best
worst-case 0px, best mean 588px @ `(180,0,0)@0.08`) across the original
`CUBE_X_RANGE` (0.35-0.55m) x `CUBE_Y_RANGE` (-0.20-0.20m) spawn area.
Chased two hypotheses for why, in order:

1. **Widen the wrist camera's standoff (lateral offset), on the theory
   that the ~17-55mm grasp residual was exceeding the frame's half-width
   at the object plane** (a pinhole estimate: `tan(35deg) * 0.08m ~= 56mm`
   half-width at `WRIST_CAMERA_HORIZONTAL_FOV_DEG=70`, and the cube needs
   to stay within ~36mm of the optical axis to render fully). **Falsified
   by direct measurement**: adding 0.14m to `WRIST_LATERAL_CANDIDATES` and
   re-sweeping still came back `NO MOUNT WORKS`, and didn't even beat the
   existing best mean for the winning direction (624px @0.08 vs 506px
   @0.14 in that run). The camera's real geometry (aimed at
   `WRIST_CAMERA_FOCUS_M` along the tool axis, offset back by
   `WRIST_CAMERA_BACK_M`) doesn't reduce to a simple pinhole half-width, so
   lateral offset alone was never the lever.
2. **Narrow `CUBE_X_RANGE`/`CUBE_Y_RANGE`, on the sweep script's own
   second stated hypothesis** (a fixed mount cannot cover a spawn area
   wider than its own field of view from every point in it). **Confirmed**:
   narrowed both ranges to a quarter of the original area, same centre
   (`(0.40, 0.50)` x `(-0.10, 0.10)`, committed in `pick_place_scene.py`).
   The exact candidate that had scored 0px worst-case across the full
   range scored 588-760px across 8 fresh spawns confined to the narrower
   box -- the spawn range, not the mount, was the actual cause of
   `NO MOUNT WORKS`.

Every candidate's worst sample over 5 spawns was 0 px; means ranged
0-588px. This is no longer a measurement artifact -- it is the mount
genuinely failing to keep the cube in frame across `CUBE_X_RANGE` (0.35-
0.55m) x `CUBE_Y_RANGE` (-0.20-0.20m), a 0.20 x 0.40m spawn area no single
fixed eye-in-hand direction+offset covers from every point in it. (**2026-
09-21: superseded by this same session's own further work below** -- the
actual root causes were missing scene lighting and too coarse a candidate
grid, not the spawn range, though the spawn range was independently
narrowed anyway for the unrelated reach/divergence reasons above.)

**Re-running the full 18-candidate sweep at the narrowed range** (not just
that one spot-checked candidate) surfaced a second lesson: the spot-check
above was itself premature. The properly controlled comparison picked
`(-90, 0, 0)@0.12m` as the actual best of all 18 by worst-case pixels
(134px worst-case, 206px mean) -- the spot-checked `(180,0,0)@0.08`
candidate scored **0px worst-case** in that same controlled run, because a
single candidate checked in isolation (even across several samples) can't
rule out that every other candidate is comparably variable. This is the
same lesson `check_cameras.py`'s own multi-sample fix already enforces
one level down (across samples of one candidate) -- it turned out to apply
one level up too (across candidates). `pick_place_scene.py`'s
`WRIST_CAMERA_FLANGE_ROT_EULER`/`WRIST_CAMERA_LATERAL_M` now hold this
corrected result, with both the mistake and the correction documented
inline so it isn't repeated.

**Then a second, separate problem surfaced and got fixed, closing this out:**
a fresh preflight check at the committed constants scored a plausible 151
cube px but **failed `preflight_check` anyway** -- 70% of the wrist frame
read near-black (`MAX_DARK_FRACTION` is 0.30). Root cause (found by
visually inspecting the committed `wrist_dirs_at_grasp_lateral*.png`
contact sheets: a sharp-edged, gradient-free pure-black region in every
direction except the two aimed at the robot's own wrist hub, which always
fills the frame regardless of aim) and confirmed live
(`diag_lighting.py`, scratchpad): **this scene had no environment light.**
`add_default_ground_plane()` brings in Isaac Sim's
`Grid/default_environment.usd`, which bundles exactly one light --
`/World/defaultGroundPlane/SphereLight`, a localized point-like source
whose falloff doesn't reach past the small ground-plane/robot/cube area.
There's no wall, skybox, or background geometry at all, so any camera ray
that misses that small area -- unavoidable for a 70deg-FOV wrist camera at
a short standoff sweeping past the workspace as the arm moves -- hits
nothing lit and renders exactly RGB=0, not a gradient or a horizon the way
a real camera in a real room would. **Fixed** by adding a `UsdLux.DomeLight`
in `PickPlaceScene.__init__`: measured directly, this dropped the wrist
camera's near-black fraction at the committed mount from 69.9% to 1.5%
with no other change -- and it's a real sim-to-real gap fix, not a
check-passing workaround, since a real camera never sees genuine unlit
void.

**Both remaining checks now pass, closing this item out:**
  - Re-ran the full 18-candidate `--sweep_wrist --wrist_samples 5`
    comparison with the light fix in place (the earlier 134px-worst-case
    ranking was itself measured through the same missing-light bug, so it
    needed redoing, not just re-trusting). Result:
    `BEST: rot=(-90, 0, 0) lateral=0.12m -- worst-case 219 cube px, mean
    1912, next-best worst-case 180` -- **the first time this sweep has
    ever cleared its own 200px bar.** Matches what was already committed
    (no further constant change needed). The sweep's own two caveats still
    apply and are worth remembering, not dismissing: worst-case (219) is
    under half the mean (1912), i.e. high spawn-to-spawn variance remains,
    and the runner-up (180 worst-case) isn't decisively beaten -- this is
    a real, measured improvement, not a clean, low-variance winner.
  - Ran `check_cameras.py --at_reset` -- the exact configuration
    `collect_demos.py` actually gates on (`scene.reset()` then
    `preflight_check()`, zero arm movement, see `collect_demos.py`'s own
    call site). Result: **`preflight: OK`**, wrist near-black 1.1%. This
    is the first time this exact call sequence has passed.

Both checks this section's own earlier "not yet validated" note asked for
are done and passing. Next: `collect_demos.py --num_episodes 5` (step 4
below) is now the right next move, not further mount tuning -- the
remaining variance is a training-data-quality question a learning curve
will answer, not something to keep squeezing pixels to fix in advance.

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

**2026-09-19: this smoke test was run for the first time (with the wrist
camera now trustworthy) and failed completely -- 0/5, 15/15 attempts
rejected, one attempt launching the cube 30.7m (a contact-explosion event,
worse than the 5.26m one in item 2 above).** Three separate, real,
independently-confirmed bugs were found and fixed chasing this, all
verified with before/after measurements (see `scripted_pick_place.py`'s
`SETTLE_TICKS` comment and `isaac_sim_common.py`'s `bind_grip_friction_
material`/`GripperController._fix_drive_gains` docstrings for the full
reasoning each):

1. **`SETTLE_TICKS` only existed before the descend segment, not between
   descend and close.** A tick-by-tick trace showed the tool 21mm from the
   cube with nonzero residual velocity at the exact moment CLOSE started.
   Added a second settle segment after descend -- confirmed this drops
   tool speed at close-start to 0-9mm/s in the common case (some episodes
   still show a genuine large tracking miss, an unrelated pre-existing
   failure mode).
2. **No friction material was set anywhere in this codebase** -- on
   either the gripper pads or any pickable object (confirmed by grep: zero
   PhysicsMaterial/MaterialAPI/friction lines existed before this).
   `find_grasp_frame.py` (an existing but never-finished diagnostic)
   re-run confirmed "NOTHING GRASPS" even at 7.2mm tracking error -- the
   tightest alignment measured all session. Added
   `bind_grip_friction_material` (0.9 static / 0.7 dynamic), bound to both
   the gripper pads and every pickable object. Verified bound correctly
   (`UsdShade.MaterialBindingAPI.ComputeBoundMaterial` on both sides
   confirms it), but made **no measurable difference** to grasp success on
   its own -- ruled out as the (sole) cause.
3. **The gripper's own `finger_joint` shipped with kp=171.89, kd=0.0115**
   -- three to four orders of magnitude weaker than the arm's own joints
   (kp in the 5.7e4-5.9e5 range). Driven from open to the fully-closed
   target over a real 90-tick close segment, it converged to only 0.374 of
   its 0.80 rad target (47%) -- in free space, nothing to contact. Fixed
   via `GripperController._fix_drive_gains` (kp=20000, kd=500, applied once
   the drive joint index is first resolved). Verified: now reaches 0.790
   rad (98.75%) over the same 90 ticks. Mimic joints (the other 5 gripper
   joints following `finger_joint`) were checked in the same pass and
   confirmed working correctly -- not a contributing cause.

**None of these three fixes, individually or combined, actually raised
genuine grasp success above 0/5** in repeated 5-episode checks --
`max_cube_z` stayed in the 0.02-0.04m range (well under `LIFT_Z_THRESHOLD`
=0.08m) in every non-explosive attempt, even with 0mm/s tool speed and a
fully-closing, high-friction gripper. All three are real, verified fixes
worth keeping regardless (a mis-timed settle, zero friction, and a
100x-underpowered gripper drive are each independently wrong), but they
were not the bottleneck actually blocking data collection.

**A fourth thing was tried, chasing the strongest lead of the three**:
axis-decomposing the tool-cube offset through the close segment (using the
pre-fix tick-by-tick trace, since the gripper's fixed downward approach
orientation -- and therefore its closing axis -- didn't change with any of
the three fixes above) showed the offset sitting almost entirely on ONE
horizontal axis (X: -8mm to +26mm across the close segment) while the
other stayed near zero throughout (Y: within +-5mm the whole time) -- and
the cube's own observed push direction as it's contacted tracked the same
axis. That is consistent with the residual falling specifically on the
gripper's closing axis (the worst-case direction for a parallel-jaw
gripper: it reduces "bite" on one pad and increases it on the other,
rather than sitting harmlessly along the flat width of the pads) rather
than being spread omnidirectionally.

**Tried a closed-loop, sign-only wrist-camera X correction right before
the close segment** (deliberately closed-loop rather than a precomputed
pixel-to-metre conversion -- this project's own camera geometry has been
wrong multiple times today from precomputed math alone, never from a
measured loop): calibrate the image-column/world-X sign relationship with
one measured nudge (confirmed: moving +X moves the cube's image column
LEFT at the committed wrist mount), then up to 3 iterations of
render -> measure cube centroid column -> if off-centre, step 5mm along
the calibrated direction -> re-settle, before manually holding that
corrected position through the gripper's close ramp (the original close
segment can't be reused as-is here -- it targets the uncorrected point and
would undo the correction).

**Result: inconsistent, and the pre-agreed stop condition was hit.** Of 5
episodes: 1 (ep1) improved monotonically with each correction (-44.4px ->
-45.2px -> -21.5px) but still didn't reach a successful grasp; 3 (ep0,
ep2, ep4) had the cube leave the wrist frame entirely after a single 5mm
correction; 1 (ep3) got WORSE in the same direction after a correction
that should have helped (+49.4px -> +110.7px). `grasp_succeeded` returned
True for 2/5 (ep0, ep4), but their `max_cube_z` (5.65m and 0.24m) match
this project's own well-established contact-explosion pattern, not a
controlled lift -- read as 0/5 genuine successes, not 2/5. 3/5 episodes
scored `max_cube_z < 0.05m`, meeting the stop threshold agreed before this
round started. Diagnostic script kept as
`diag_vision_recenter.py`-equivalent logic (scratchpad only, not committed
-- this was a feasibility probe, not a production implementation) for
whoever picks this up next.

**Two follow-up checks, isolating detection/mapping/control as three
separate variables instead of judging the vision correction by grasp
outcome alone (the same trap that made the friction check ambiguous
earlier):**

- **Pixel-to-mm mapping, measured directly against ground truth (no
  grasping, no rendering-based reasoning about "centred" at all).** Holding
  the tool fixed at a real converged close-start pose (reached via the
  actual policy's approach path -- a direct jump from `scene.reset()`'s
  home pose to a single low target does NOT reliably converge on its own,
  confirmed live: one attempt landed 174mm off in Y over 120 ticks) and
  placing the cube at known X offsets around it: detection was 5/5 at
  -30mm, dropping to 3/5 and 2/5 at -20/-10mm, and **0/5 at 0mm and every
  positive offset tried (+10/+20/+30mm)**.
- **A pure-geometry PhysX raycast (`get_physx_scene_query_interface().
  raycast_closest`, no rendering or colour thresholding at all) from the
  camera's actual eye point to the cube's actual centre, at the same
  offsets, to separate "occluded" from "the detector just isn't finding
  it."** Result: **partially confirmed, partially a different bug.** At
  0mm and +10mm the ray is genuinely blocked by
  `.../Robotiq_2F_85/right_inner_finger`'s pad mesh before reaching the
  cube -- occlusion by the gripper's own near finger, confirmed
  geometrically, exactly matching those two offsets' 0/5 detection. But at
  +20mm and +30mm the ray hits the cube CLEANLY (no occlusion) while RGB
  detection still found nothing -- that miss is a separate detector/framing
  issue, not occlusion, and is not yet explained.

**What this means**: the wrist camera has a confirmed structural blind
spot at exactly the alignment a grasp needs (0 to +10mm on the closing
axis, from this specific mount's near finger) -- pixel-centre re-centring
cannot be the whole answer here, since the target it would converge
toward is inside the occluded band. Worth revisiting per the four options
already on the table: aim `WRIST_CAMERA_FOCUS_M` closer than the TCP so
the approach is visible before the jaws would occlude it, a top-down
mount (more `WRIST_CAMERA_BACK_M`, less `_LATERAL_M`) instead of a side
bracket, using the base camera for final centring instead, or redefining
the correction target as "last visible offset before occlusion" rather
than pixel-centre. Not yet attempted.

**A second, independent lead was chased in parallel: is the grasp failure
actually about alignment at all, or about contact/dynamics?** Three points
argued for the latter: (1) several close attempts today had near-zero
tool speed and 16-55mm alignment yet still failed to lift the cube --
if alignment were the whole story, the best-aligned attempts should
succeed; (2) the 2026-09-15 `pivot_dwell_check.py` run (documented in
"Known gaps" below) found the grasp-pose dwell error diverging (229.9mm
-> 575.6mm) specifically with the gripper attached, while free space
stayed flat -- exactly this file's own CONTACT/DYNAMICS decision table's
CONTACT signature; (3) `grep` confirms gripper link mass/inertia has never
been set anywhere in this codebase (the one `MassAPI` call in
`isaac_sim_common.py` is for the cube, not the gripper) -- today's
`finger_joint` stiffness fix (171.89 -> 20000 kp) raises exactly the risk
`pivot_dwell_check.py`'s own docstring names: a stiff, undamped drive
pressing an uncalibrated-mass body into contact.

That 2026-09-15 run was itself methodologically flawed the same way the
mapping test above nearly was: `hold()` commanded its target in one jump
from `scene.reset()`'s home pose (432-500mm of initial error at tick 1),
so most of its tick budget was spent finishing that jump, not holding a
converged pose -- already flagged as unverified in "Known gaps" below.
Fixed by adding `settle_to()`: a graduated above-then-descend approach
(mirroring the real scripted policy's own path) run BEFORE the dwell
measurement starts, so tick 1 of `hold()` now measures genuine dwell
drift. **Re-run with the fix, and with today's three grasp-pipeline fixes
already in place: `--no-gripper` is flat at 13.1mm (free space) / 4.7mm
(grasp pose) from tick 1 through 180; WITH the gripper, free space is
flat at 9.4mm (matching 2026-09-15 exactly) and -- previously the
diverging case -- the grasp-pose dwell is now ALSO completely flat at
32.4mm, zero growth.** The CONTACT divergence this test exists to catch
is gone. This is strong evidence today's `finger_joint` gains fix (item 3
above) was load-bearing for more than closing force -- it also resolved a
real, previously-diverging contact/dynamics instability, independently of
whether it fixes the overall grasp success rate.

**Caveat that keeps this open, not closed**: `hold()`'s dwell test keeps
the gripper OPEN (`gripper=0.0`) the entire time -- it measures whether
the arm holds a fixed pose near the table, not what happens during the
transient of the fingers actively CLOSING onto an object, which is the
scenario the smoke test's 30.7m explosion actually happened in. A settled
dwell with the gripper open does not yet rule out an under-calibrated
mass/inertia problem specific to the closing transient. Not yet tested:
repeat this same settled-dwell measurement WHILE ramping the gripper
closed on the cube (mirroring the real close segment) rather than holding
it open, to see whether the transient -- not just the static hold --
stays stable too.

**A new, separate, and possibly bigger lead came out of the SAME re-run**:
`pivot_dwell_check.py`'s PIVOT test (swings the grip point through 5 rolls
x 3 tilts to isolate a static flange-to-fingertip transform error from
dwell drift) measured a **~90-100mm transform error, present both with
and without the gripper** (pivot spread 180.9mm/200.1mm -> ~90.4mm/100.1mm
transform error; mean offset from the commanded point 314-321mm, mostly a
separate tracking/GRIPPER_TCP_OFFSET_M bias the rotation test can't
isolate further). A `GRIPPER_TCP_OFFSET_M` or frame-composition error of
this size, constant and direction-dependent on orientation, is large
enough to be a real candidate for the axis-concentrated (X-heavy, Y-near-
zero) offset the earlier axis decomposition found -- a fixed frame bias
would project differently onto world axes than a random tracking residual
would. Not yet cross-checked against the axis-decomposition data or acted
on -- this is this session's newest open thread, not a closed one.

**2026-09-20 update: those PIVOT numbers were almost certainly measured
through the same convergence bug `settle_to()` had just fixed for the
DWELL half, and should not be acted on until re-measured.** Reading the
script after the fact: the 2026-09-19 commit added `settle_to()` to both
dwell measurements but left the PIVOT loop as `scene.reset()` then
`hold(..., 60)` -- 60 ticks in one jump from the home pose, the exact
pattern the dwell fix's own docstring describes opening at 432-500mm of
error. Every signature fits an approach-convergence artifact rather than
a frame error: a 314-321mm mean offset (the dwell tests, once settled,
sit at 4.7-32mm, so a real static bias has to fit inside that), a
180-200mm spread that is just how far each of 15 orientations got in 60
ticks, and identical numbers with and without the gripper (convergence
doesn't care about payload; a real TCP error would at least be measured
more cleanly without it). Fixed in `pivot_dwell_check.py` (the PIVOT loop
now calls `settle_to()` before its hold, same as the dwell half). **Not
yet re-run on Isaac Sim** -- do that first next session
(`python3 pivot_dwell_check.py` and `--no-gripper`); expect a spread in
the tens of mm at most. Only if a large spread survives the settled
measurement does the frame-bias hypothesis above deserve the
axis-decomposition cross-check.

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
  fixes' own validation has tested.

  **Re-run 2026-09-19, with both the methodology and the scene fixed.**
  Added `settle_to()`: a graduated above-then-descend approach run BEFORE
  the dwell measurement starts (mirroring the real scripted policy, which
  never jumps straight to a low target from home either), so `hold()`'s
  tick 1 now measures genuine dwell drift instead of the tail of an
  unconverged jump. With that fix AND today's three grasp-pipeline fixes
  (settle-before-close, friction, `finger_joint` gains -- see "The next
  Isaac Sim session" above) already in place: `--no-gripper` is flat at
  13.1mm (free space) / 4.7mm (grasp pose) from tick 1 through 180 --
  settled, no growth, and with plenty of margin under `SETTLED_TOLERANCE_M`.
  **WITH the gripper, free space is flat at 9.4mm (matching this entry's
  original 2026-09-15 number exactly) and the grasp-pose dwell -- the
  case that diverged to 575.6mm before -- is now ALSO flat at 32.4mm.**
  The CONTACT divergence is gone. Strong evidence the `finger_joint` gains
  fix was load-bearing for this specifically, not just for closing force,
  matching this file's own DYNAMICS theory (soft drive gains sagging
  against an uncalibrated-mass load) more than the CONTACT one. Caveat:
  `hold()` keeps the gripper OPEN throughout -- this confirms a fixed pose
  holds near the table, not that the transient of the fingers actively
  CLOSING onto an object is equally stable, which is the regime the
  smoke test's 30.7m explosion (see "The next Isaac Sim session" above)
  actually occurred in. Not yet tested: repeat this measurement while
  ramping the gripper closed on the cube instead of holding it open.

  The SAME re-run's PIVOT test also turned up something new: a ~90-100mm
  flange-to-fingertip transform error, present both with and without the
  gripper (pivot spread 180.9mm/200.1mm). Not yet cross-checked against
  the axis-decomposition finding above, but a constant, orientation-
  dependent frame bias of this size is a real candidate for at least part
  of the X-heavy, Y-near-zero offset pattern found there -- newest open
  thread, not yet acted on.
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
