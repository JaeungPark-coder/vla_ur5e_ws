"""Why does the wrist camera lose the cube at 20-30mm standoff, with nothing
between camera and cube to occlude it?

The 0-10mm loss already has an explanation (the gripper fingers physically
block the lens at that range -- see WRIST_CAMERA_DOWN_TILT_DEG). 20-30mm is
different: the fingers aren't in the way there, so this script exists to
narrow down WHICH of the three remaining explanations it actually is, rather
than guess:

  NEAR-CLIP     the camera's own clipping plane (CAMERA_NEAR_CLIP_M) removes
                geometry closer than it, the same failure mode that once
                produced all-black frames project-wide at a 1m default clip
                (see pick_place_scene.py's _setup_cameras comment).
  EXPOSURE      the frame renders but the cube is too dark/low-contrast to
                register as cube pixels (cube_pixels_visible's red-dominance
                threshold), the way the mis-aimed wrist camera and the
                clipped first run both showed up as "mostly near-black"
                rather than a clean zero.
  STALE FRAME   the annotator hands back a previous render instead of the
                current one -- check_cameras.py's own _render_wrist already
                documents needing two render() calls per authoring change
                for exactly this reason; this checks whether that still
                holds when only the CUBE moves, not the camera.

Bypasses the arm/gripper entirely: this project's own history
(scripted_pick_place.py's steps_per_segment, the frozen-cube-position bug
fixed in b5e7fb0) is full of cases where "the camera saw nothing" turned out
to mean "the thing that was supposed to be in view wasn't actually there".
Controlling the cube's position directly, independent of whether the arm
reached anywhere close to it, rules that whole class of confound out: the
cube is placed at an exact, known standoff along the wrist camera's own
current viewing axis, so any failure to see it points at the camera, not the
reach.

    python3 check_near_field_visibility.py
    python3 check_near_field_visibility.py --standoffs_mm 5,10,15,20,25,30,40,60
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, WRIST_CAMERA_PRIM_PATH,
    CAMERA_NEAR_CLIP_M, MAX_DARK_FRACTION)
from scripted_pick_place import ScriptedPickPlace  # noqa: E402
from isaac_sim_common import prim_world_pose  # noqa: E402

DEFAULT_STANDOFFS_MM = (5, 10, 15, 20, 25, 30, 35, 40, 50, 75, 100)

# Kit's fastShutdown kills the process before Python flushes a block-buffered
# stdout -- same failure mode check_cameras.py and pivot_dwell_check.py both
# guard against, so this mirrors their say()/_REPORT idiom.
_REPORT = {"path": None}


def say(line=""):
    print(line, flush=True)
    if _REPORT["path"]:
        with open(_REPORT["path"], "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _rgb(frame):
    arr = np.asarray(frame)
    if arr.size == 0:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    return arr[..., :3].astype(np.uint8)


def _drive_to_grasp(scene, obs):
    """Put the arm/gripper at a realistic grasp-time pose, so the wrist
    camera's position and orientation are what a real approach would put
    them at -- only the cube's position is then overridden directly, per
    this file's own docstring on why that isolates the camera from the
    reach. Duplicated from check_cameras.py rather than imported: that
    module calls SimulationApp() at import time, so importing from it here
    would try to boot Kit twice."""
    say("driving the scripted expert to the grasp...")
    policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
    frames = list(policy.generate_frames())
    n = policy.frames_until_grasp()
    for target_pos, target_rotvec, target_gripper in frames[:n]:
        scene.step_towards(target_pos, target_rotvec, target_gripper)
    obs = scene.get_observation()
    # Same check check_cameras.py's own _drive_to_grasp makes, and for the
    # same reason: if the arm never actually reached the cube, the "grasp
    # pose" below is some arbitrary miss, and every standoff probed against
    # it -- however carefully placed on the camera's own axis -- says
    # nothing about a real approach. CONFIRMED 2026-09-21 this project's own
    # RMPflow reach is frequently missing by meters (occasionally far more)
    # in this environment, well past the documented historical rate --
    # treat any run reporting this warning as uninterpretable until that is
    # fixed, not as evidence about near-field visibility.
    reach = float(np.linalg.norm(np.asarray(obs["tool_pos"]) - scene.get_cube_position()))
    say(f"at grasp after {n}/{len(frames)} frames: tool is {reach * 1000:.0f}mm from the cube")
    if reach > 0.06:
        say("WARNING: the tool did not actually reach the cube. The arm's pose below is not a "
            "real grasp -- every standoff result is measured from an arbitrary miss, not the "
            "approach this script is meant to isolate. Fix RMPflow tracking/reach before "
            "trusting anything below.")
    return obs


def wrist_camera_view_axis(scene):
    """(position, unit forward direction) of the wrist camera's CURRENT
    world pose. A USD camera images along its own local -Z."""
    stage = scene.stage
    pos, quat_xyzw = prim_world_pose(stage.GetPrimAtPath(WRIST_CAMERA_PRIM_PATH))
    forward = Rot.from_quat(quat_xyzw).apply(np.array([0.0, 0.0, -1.0]))
    return pos, forward / np.linalg.norm(forward)


def probe_standoff(scene, cam_pos, forward, standoff_m):
    """Place the (persistent) cube prim exactly `standoff_m` along the
    camera's current viewing axis and render.

    CONFIRMED 2026-09-21: an earlier version of this used world.render()
    here, matching check_cameras.py's _render_wrist -- and got 0 cube_px at
    every single standoff from 5mm to 100mm, which turned out to be this
    function, not the camera: set_world_pose() writes straight into the live
    PhysX rigid-body view (see isaac_sim_common.prim_world_pose's own
    docstring, and pick_place_scene.reset()'s identical fix for the cube's
    spawn position), and a bare render() -- unlike world.step(render=True)
    -- never flushes that write into what gets rendered, so every probe was
    silently re-rendering the cube whever it already was, not where it had
    just been placed. world.step(render=True) is what reset() uses for the
    exact same write, so this matches that instead. Costs a tiny, accepted
    amount of gravity drift on the cube per probe (repositioned fresh with
    zeroed velocity each call, so it does not accumulate across probes) --
    the arm's last-commanded joint targets hold steady across the step the
    same way pivot_dwell_check.py's multi-tick holds do, so only the cube
    moves in any way that matters here."""
    target = cam_pos + forward * standoff_m
    scene.cube_rigid_prim.set_world_pose(position=target)
    scene.cube_rigid_prim.set_linear_velocity(np.zeros(3))
    scene.cube_rigid_prim.set_angular_velocity(np.zeros(3))
    scene.world.step(render=True)
    return np.asarray(scene.get_observation()["wrist_rgb"])


def diagnose(standoff_mm, frame, prev_frame, scene):
    rgb = _rgb(frame).astype(np.float32)
    cube_px = scene.cube_pixels_visible(frame)
    dark_fraction = float((rgb.max(axis=2) < 12).mean())
    mean, std = float(rgb.mean()), float(rgb.std())
    # A real stale-annotator bug looks like THIS frame matching the PREVIOUS
    # standoff's frame exactly, even though the cube moved in between --
    # not two same-state renders agreeing with each other (that's the
    # settled, correct case check_cameras.py's own two-render idiom relies
    # on, and comparing within one probe would flag it as a false positive).
    stale = prev_frame is not None and bool(np.array_equal(_rgb(prev_frame), _rgb(frame)))

    verdicts = []
    if standoff_mm / 1000.0 < CAMERA_NEAR_CLIP_M:
        verdicts.append(f"NEAR-CLIP: {standoff_mm}mm is inside CAMERA_NEAR_CLIP_M "
                         f"({CAMERA_NEAR_CLIP_M * 1000:.0f}mm) -- the clip plane alone "
                         f"explains this one")
    if cube_px == 0 and dark_fraction > MAX_DARK_FRACTION:
        verdicts.append("EXPOSURE/AIM: frame is mostly near-black -- consistent with the "
                         "camera aimed past the cube or exposure too low, not a sharp cutoff")
    elif cube_px == 0:
        verdicts.append("cube_px=0 but the frame is NOT dark -- something opaque (arm, "
                         "gripper, table) fills the view instead; this is occlusion, not "
                         "a clipping or exposure problem")
    if stale:
        verdicts.append("STALE FRAME: byte-identical to the PREVIOUS standoff's frame despite "
                         "the cube having moved in between -- the annotator is handing back an "
                         "old render instead of the current one")
    return cube_px, dark_fraction, mean, std, verdicts


def run(out_dir, standoffs_mm):
    os.makedirs(out_dir, exist_ok=True)
    _REPORT["path"] = os.path.join(out_dir, "near_field_report.txt")
    open(_REPORT["path"], "w").close()
    say(f"writing this report to {_REPORT['path']} as well as stdout")

    scene = PickPlaceScene(with_gripper=True)
    say("scene built; resetting...")
    obs = scene.reset()
    obs = _drive_to_grasp(scene, obs)
    cam_pos, forward = wrist_camera_view_axis(scene)
    say(f"wrist camera at grasp pose: pos={np.round(cam_pos, 4).tolist()} "
        f"forward={np.round(forward, 4).tolist()}")
    say(f"CAMERA_NEAR_CLIP_M = {CAMERA_NEAR_CLIP_M * 1000:.1f}mm\n")

    say(f"{'standoff':>9}  {'cube_px':>8}  {'dark%':>6}  {'mean':>7}  {'std':>6}  verdict")
    say("-" * 90)
    any_gap = False
    prev_frame = None
    for mm in standoffs_mm:
        frame = probe_standoff(scene, cam_pos, forward, mm / 1000.0)
        cube_px, dark, mean, std, verdicts = diagnose(mm, frame, prev_frame, scene)
        prev_frame = frame
        gap = cube_px == 0
        any_gap = any_gap or gap
        say(f"{mm:>7}mm  {cube_px:>8d}  {100 * dark:>5.1f}%  {mean:>7.1f}  {std:>6.2f}"
            + ("  <-- NOT VISIBLE" if gap else ""))
        for v in verdicts:
            say(f"           {v}")

    say("")
    if not any_gap:
        say("Every standoff tested saw the cube -- the 20-30mm gap did not reproduce with "
            "the cube placed directly on the camera's own axis. That points at the ARM's "
            "reach during a real approach missing the axis (tracking error, or the cube not "
            "being where the approach assumed), not at the camera itself -- check tool "
            "tracking during the actual approach segment instead.")
    else:
        say("See the per-row verdicts above for which of NEAR-CLIP / EXPOSURE-AIM / "
            "OCCLUSION / STALE-FRAME showed up at which standoff.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out_dir", type=str, default="camera_check",
                        help="directory the report is written to")
    parser.add_argument("--standoffs_mm", type=str, default=None,
                        help=f"comma-separated standoffs to test in mm "
                             f"(default: {','.join(str(v) for v in DEFAULT_STANDOFFS_MM)})")
    args = parser.parse_args()
    standoffs = (DEFAULT_STANDOFFS_MM if args.standoffs_mm is None
                 else [int(v) for v in args.standoffs_mm.split(",")])

    try:
        run(args.out_dir, standoffs)
    except BaseException:
        # Print BEFORE simulation_app.close(): Kit's fastShutdown can take the
        # process down before an unflushed traceback ever gets written -- same
        # idiom as check_cameras.py / pivot_dwell_check.py.
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
