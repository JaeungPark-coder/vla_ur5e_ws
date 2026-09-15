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

## The next Isaac Sim session: two independent tracks

They do not block each other -- run both. Track A finishes step 1 of the
bring-up table; track B decides HOW to fix the grasp, which is step 3 and has
never been run.

### Track A -- pin the wrist camera (finishes step 1)

```bash
cd isaac
python3 check_cameras.py --sweep_wrist
```

This is trustworthy now in a way it was not before. The sweep scores each
candidate mount by how many pixels of the cube it sees **at the moment the
scripted expert closes**, so it was only ever as good as that moment: with
`steps_per_segment=30` the tool was still 259 mm from the cube when the
gripper closed, and every score was measured from the wrong place. 90 is now
the default in `scripted_pick_place.py`, so running this script picks it up
with no flags. It also runs **without** the gripper by default, so the 1-in-4
~300 mm drift -- which comes from the gripper pads' collision geometry -- is
not even loaded here. The sweep is therefore independent of the grasp
reliability work in track B.

Three things to check before trusting the winner:

| check | why |
|---|---|
| `at grasp after N/M frames: tool is Xmm from the cube` is under 60 mm | over that the script WARNs, and every score below it was taken from the wrong place |
| no WARNING about a close runner-up | the correct orientation should win by a wide margin, not a judgement call |
| the contact sheet PNGs actually show the cube, large and centred | a camera buried inside `wrist_3_link` renders solid black, which has happened |

Then paste the printed values into `pick_place_scene.py`'s
`WRIST_CAMERA_FLANGE_ROT_EULER` and `WRIST_CAMERA_LATERAL_M`, and commit.
Step 1 is done.

### Track B -- find out what the residual actually is (step 3)

```bash
python3 pivot_dwell_check.py                # with the gripper
python3 pivot_dwell_check.py --no-gripper   # payload ablation
```

Do this **before** changing anything about the grasp. The ~20-40 mm tracking
residual has three possible causes with three different fixes, and the commit
that found it already said so: narrowing the residual is the next thing to
try, *not* more collision or solver tuning.

| both runs | reading | the fix that follows |
|---|---|---|
| grows in free space too | dynamics | RMPflow gain tuning, or gripper link mass/inertia -- **which this repo never sets anywhere**; the only `MassAPI` call in it is the cube's 50 g |
| grows only at the grasp | contact | re-check the geometry: the `GRIPPER_TCP_OFFSET_M` frame fix should already have covered this, so find out why it did not |
| neither grows | static TCP offset | the pivot spread is the answer -- correct `GRIPPER_TCP_OFFSET_M` or the flange->tool0 rotation constant |

Why the ablation cannot answer the third case on its own:
`grip_point_world()` is the flange prim plus the **constant**
`GRIPPER_TCP_OFFSET_M`, not a gripper prim. That is what makes both runs
measure the same reference point, and equally what makes a static offset
error appear identically in both. The pivot half of the same script is what
separates it: rotating about the approach axis drags a wrong TCP around an
arc, so the error shows as spread with no mean offset, while a genuine bias
shows as offset with no spread.

One thing recorded in the script's own docstring is worth reading before
acting on the result: lengthening the hold from 60 to 120 ticks made the
residual grow from 24 mm to 43 mm. If that reproduces, adding a dwell at the
end of the descent would be ineffective at best and harmful at worst -- which
is exactly why this measurement comes first.

### After both

Apply the one fix track B points at -- one, not several, or the next run
cannot say which worked -- then smoke-test before committing to a long run:

```bash
python3 collect_demos.py --num_episodes 5
```

It has to complete without hitting the attempt cap. Only then is the 50-200
episode collection worth starting.


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

| # | what | needs | how you know it worked |
|---|---|---|---|
| 1 | Wrist camera mount | Isaac Sim | `check_cameras.py --sweep_wrist` picks a winner; pin it in `WRIST_CAMERA_FLANGE_ROT_EULER`, which currently reads `None` |
| 2 | Base camera framing | a decision | `preflight framing` prints the ceiling; 30 px across is **not reachable** at the current workspace and resolution |
| 3 | Is the residual frame, contact, or dynamics? | Isaac Sim | `pivot_dwell_check.py` — free-space drift means dynamics, grasp-only drift means contact |
| 4 | Does the scripted expert grasp? | Isaac Sim | `collect_demos.py --num_episodes 5` completes without the attempt cap |
| 5 | Demonstration collection | Isaac Sim | 50–200 episodes with the reject rate low and the gate quiet |
| 6 | Euler axis order vs OpenVLA | an OpenVLA checkout | compare its dataloader against `EULER_SEQ`; the metamorphic check proves self-consistency, **not** agreement with OpenVLA |
| 7 | Fine-tune | 2× RTX 3090 | training converges; `compute_norm_stats` runs without shape errors |
| 8 | Serving + bridge | GPU host + ROS 2 | the policy client steps without timing out |
| 9 | Real gripper driver | UR5e + Robotiq | gripper state logs `measured`, not `NOT MEASURED` |

Step 1 gates 4 and 5: openpi's UR5 contract feeds the policy both views, and
the wrist view carries the fine manipulation signal, so collecting with it
mis-aimed wastes the run. Step 3 decides whether there is a dynamics problem
at all — the code's own note says the residual came from the fingers
contacting the table, which the frame fix addressed, while a later review
suggested gravity and drive gains. Only a free-space hold separates them, and
the answer changes whether there is work to do.

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
