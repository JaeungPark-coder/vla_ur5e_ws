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
  check_cameras.py         renders what each camera sees + sweeps wrist orientations -- RUN THIS FIRST
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

## Known gaps / ADJUST markers to resolve on real hardware

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
- **Base camera framing is poor even though it works.** The arm dominates the
  frame and occludes the cube for roughly the first half of every episode
  (measured: ~1 cube pixel at reset, first clearly visible around frame 90,
  peaking near 139 of 65,536 px). The policy therefore gets almost no visual
  evidence of where the cube is during exactly the approach phase where it
  needs it. Worth moving the camera to a less occluded viewpoint, or
  tightening its framing onto the workspace, before a full collection run.
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
  setup (for Phase 6's `camera_projection.py` math to be accurate) assumes
  USD camera focal-length/aperture attributes behave the standard way on
  your Isaac Sim version -- if detected objects' localization error (printed
  by `hybrid_pick_place_demo.py`) is consistently large, check the actual
  rendered FOV against `BASE_CAMERA_HORIZONTAL_FOV_DEG` first.
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
