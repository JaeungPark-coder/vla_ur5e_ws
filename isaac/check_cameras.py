"""Renders what each camera actually sees, and settles the wrist camera's
orientation by measurement instead of derivation.

Run this before every collection run. It takes under a minute; a collection
run takes hours, and three of them have now been spent on cameras that were
pointing somewhere other than the task:

  * the first 100-episode run had a default 1.0 m near-clip plane on both
    cameras, which removed the entire scene -- all-black wrist frames, and
    the cube visible in exactly zero of 21,000 frames;
  * the base camera's orientation was a guessed Euler triple that aimed it
    at the sky;
  * after both were "fixed", inspecting the frames of the verification run
    showed the WRIST camera still imaging background and empty space for a
    whole episode -- never the cube, the gripper or the table.

Each of those was invisible in the collection logs and obvious in a single
rendered frame.

    python3 check_cameras.py
    python3 check_cameras.py --sweep_wrist

Runs WITHOUT a gripper by default, mainly so a bare comparison against past
reports (all taken without one) stays apples-to-apples -- the wrist camera
IS mounted close enough to the fingers that gripper presence changes what
it sees (see WRIST_CAMERA_DOWN_TILT_DEG), so "with" is not just a slower
version of "without".

STALE CLAIM REMOVED 2026-09-21: this used to say the gripper "is currently
unstable in both attachment modes (in teleport mode its links diverge to
world positions around 1e15 m ...; in fixed_joint mode it takes the app
down during reset)". Both of those were the OLD, already-replaced
attachment schemes (see isaac_sim_common.py's own history) -- the CURRENT
one welds the gripper into the arm's own articulation via a USD variant
selection, and --with_gripper sweeps of 50+ episodes on 2026-09-21 (both
with and without an explicit gripper mass override) showed no such
instability from the gripper's mere presence. --with_gripper includes it.

Why the sweep rather than more algebra: the committed derivation is
analytically correct as far as it goes -- a USD camera images along its own
local -Z, so rolling 180 degrees about X points that axis along the
commanded tool's +Z, which is the approach direction -- and it still came
out aiming at the sky. That puts the error upstream, in the flange->tool0
offset read from rmpflow.get_end_effector_pose(), whose return layout and
reference frame are both unverified assumptions. So the sweep works in the
FLANGE frame directly, which needs none of that, and covers all six
axis-aligned directions a rigidly-mounted camera could face.

Scoring is automatic: each candidate is rendered at the moment the scripted
expert closes on the cube, and scored by how many pixels of the cube it
actually sees. An eye-in-hand camera at the grasp should see the cube large
and centred, so the correct orientation wins by a wide margin rather than
needing a judgement call. The winning line is printed ready to paste into
pick_place_scene.WRIST_CAMERA_FLANGE_ROT_EULER.
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, WRIST_CAMERA_FLANGE_ROT_EULER,
    WRIST_CAMERA_LATERAL_M, WRIST_CAMERA_FOCUS_M, WRIST_CAMERA_DOWN_TILT_DEG)
from scripted_pick_place import ScriptedPickPlace  # noqa: E402

# The six axis-aligned directions a flange-mounted camera can face, as XYZ
# Euler degrees from the flange frame to the camera frame. A USD camera
# images along its own local -Z, so e.g. (0,0,0) images along the flange's
# -Z and (180,0,0) images along its +Z.
WRIST_DIRECTION_CANDIDATES = [
    (0.0, 0.0, 0.0),
    (180.0, 0.0, 0.0),
    (90.0, 0.0, 0.0),
    (-90.0, 0.0, 0.0),
    (0.0, 90.0, 0.0),
    (0.0, -90.0, 0.0),
]
# Once the direction is known, these rolls about the viewing axis choose
# which way is "up" in the image. Purely cosmetic for a policy, but a
# consistently upright wrist view is easier to sanity-check by eye.
WRIST_ROLL_CANDIDATES = [0.0, 90.0, 180.0, 270.0]
# How far to the side of the tool axis to bracket the camera. Swept because
# distance, not direction, is what defeated the first two attempts: at 5 cm
# the camera sat inside the UR5e's wrist_3_link (black at every orientation),
# and displacing it along its own viewing axis drove it through the workpiece
# and below the table.
#
# 0.14 added, then its hypothesis falsified, 2026-09-19 (a separate,
# parallel session -- see README's merge note for 2026-09-21): NO MOUNT
# WORKS (every candidate's worst sample over 5 spawns was 0 px) held even
# after fixing the reach measurement. A back-of-envelope pinhole
# calculation suggested this might be a framing problem rather than a
# direction one -- at WRIST_CAMERA_HORIZONTAL_FOV_DEG=70 (half-angle
# 35deg, tan=0.700) and the old 0.08m lateral offset, the frame covers
# roughly +-56mm at the object plane, and the RMPflow tracking residual
# actually measured at the grasp (17-55mm) exceeds that in the worst case
# -- so a wider standoff (0.14m -> ~98mm half-width) should have swallowed
# the residual with room to spare, IF the camera's effective distance to
# the object scaled with lateral offset the way that estimate assumed. It
# doesn't: a full re-sweep with 0.14m added came back NO MOUNT WORKS again
# and did not even beat the existing best mean for the winning direction
# (624px @0.08 vs 506px @0.14) -- the camera's real geometry (aimed at
# WRIST_CAMERA_FOCUS_M along the tool axis, offset back by
# WRIST_CAMERA_BACK_M) doesn't reduce to that simple pinhole picture, so
# lateral offset alone was never the lever. CUBE_X_RANGE/CUBE_Y_RANGE in
# pick_place_scene.py being wider than any fixed mount could cover -- this
# function's own second stated hypothesis, below -- IS what NO MOUNT WORKS
# was measuring: confirmed directly, see CUBE_X_RANGE's own comment.
# Re-running the full 18-candidate grid at the narrowed range picked 0.12m
# (paired with (-90,0,0), not (180,0,0)) as the best of all four laterals
# by worst-case pixels in THAT session -- see README's merge note on why a
# different, raycast-screened session-2 sweep landed on 0.16m/(180,0,0)
# instead, and why the two have not yet been reconciled.
WRIST_LATERAL_CANDIDATES = [0.05, 0.08, 0.12, 0.14]
# Whether tilting the camera down (see pick_place_scene.WRIST_CAMERA_DOWN_
# TILT_DEG) away from the pure approach axis clears the finger-occlusion
# symptom at close standoff -- external literature suggests ~30 degrees, but
# neither the magnitude nor the sign of "down" in this asset's flange frame
# has been checked here, so both signs plus "off" are swept like every other
# unproven mount parameter in this file.
WRIST_DOWN_TILT_CANDIDATES = [0.0, 30.0, -30.0]

# Kit's fastShutdown kills the process before Python flushes a block-buffered
# stdout, so piping this script through `tee` silently loses every line it
# printed -- the same failure mode that cost a collection run its last
# parquet chunk. Mirror everything into a file, flushed as it is written, so
# the result survives regardless.
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


def _metrics(frame, scene):
    rgb = _rgb(frame).astype(np.float32)
    return {
        "mean": float(rgb.mean()),
        "std": float(rgb.std()),
        "dark": float((rgb.max(axis=2) < 12).mean()),
        "cube_px": scene.cube_pixels_visible(frame),
    }


def _fmt(name, m):
    return (f"  {name:18s} cube_px={m['cube_px']:6d}  near_black={100 * m['dark']:5.1f}%  "
            f"mean={m['mean']:6.1f}  std={m['std']:6.2f}")


def _contact_sheet(panels, path, cell=256):
    """panels: list of (label, HxWx3 uint8). Writes a labelled row of images."""
    from PIL import Image, ImageDraw

    band = 18
    sheet = Image.new("RGB", (cell * len(panels), cell + band), (20, 20, 24))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(panels):
        sheet.paste(Image.fromarray(img).resize((cell, cell)), (i * cell, band))
        draw.text((i * cell + 4, 4), label, fill=(235, 235, 240))
    sheet.save(path)
    say(f"wrote {path}")


def _drive_to_grasp(scene, obs):
    """Run the scripted expert up to the moment it closes on the cube. That
    is the pose the wrist camera exists for -- judging it from the arm's
    default pose, where it may legitimately be staring off into space, tells
    you very little.

    This is the only part of this script that steps physics. It runs without
    a gripper by default, so nothing here depends on the grasp working."""
    say("driving the scripted expert to the grasp...")
    policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
    frames = list(policy.generate_frames())
    # Read the boundary off the waypoint list rather than guessing a fraction
    # of it. A previous guess of 45% landed 19 frames into the LIFT, with the
    # tool 14.5 cm from the cube -- which made every wrist-camera candidate
    # score zero for reasons that had nothing to do with the camera.
    n = policy.frames_until_grasp()
    for target_pos, target_rotvec, target_gripper in frames[:n]:
        scene.step_towards(target_pos, target_rotvec, target_gripper)
    obs = scene.get_observation()
    reach = float(np.linalg.norm(np.asarray(obs["tool_pos"]) - scene.get_cube_position()))
    say(f"at grasp after {n}/{len(frames)} frames: tool is {reach * 1000:.0f}mm from the cube")
    if reach > 0.06:
        say("WARNING: the tool did not actually reach the cube. Every wrist-camera score below "
            "is measured from the wrong place -- fix the reach (RMPflow tracking, or the "
            "waypoint heights in scripted_pick_place.py) before trusting the sweep.")
    return obs


def _render_wrist(scene, euler, lateral=None, down_tilt=None):
    scene.set_wrist_camera_flange_rotation(euler, lateral_m=lateral, down_tilt_deg=down_tilt)
    # render(), NOT step(): re-aiming a camera needs a fresh render pass, not
    # more physics -- stepping here would let the arm drift further under
    # whatever pose _drive_to_grasp left it holding, making candidates
    # compared at slightly different scene states instead of the same one.
    #
    # STALE CLAIM REMOVED 2026-09-21: this comment used to say stepping here
    # was actively harmful because a "teleport" gripper mode held the
    # gripper on the flange only via PickPlaceScene.step_towards's per-tick
    # re-sync, so a bare world.step() would let it fall away and
    # interpenetrate the arm. CONFIRMED that mechanism cannot happen with
    # the CURRENT gripper attachment (isaac_sim_common.select_gripper_
    # variant welds the gripper into the arm's OWN articulation via a USD
    # variant selection -- GripperController.sync_pose_to_flange is a
    # documented no-op, kept only for call-site compatibility with the old
    # scheme): there is no separate gripper prim left to fall off. The
    # "teleport" and "fixed_joint" attachment modes this described were
    # both replaced (see isaac_sim_common.py's own history on that), and
    # this comment was never updated to match -- misleading enough that it
    # produced a very reasonable but incorrect hypothesis for a genuine,
    # unrelated divergence found by an actual gripper-mass A/B test on
    # 2026-09-21 (a bare `grep world.step(` across this file confirms it
    # never calls one outside PickPlaceScene.step_towards, both before and
    # after that test). World.render() refreshes the app (and so the
    # annotators) with /app/player/playSimulations off. Two passes because
    # the first one after an authoring change can still hand back the
    # previous frame.
    for _ in range(2):
        scene.world.render()
    return scene.get_observation()["wrist_rgb"]


def sweep_wrist(scene, out_dir, stage_name, at_grasp, n_samples):
    """Sweep the wrist camera's mount -- direction AND lateral offset -- and score
    each by how much of the cube it actually sees, averaged over `n_samples`
    independent random cube spawns when scoring at the grasp.

    Both direction and lateral are swept because the first attempt failed on
    the one that was not being varied: every orientation rendered solid
    black, because at the original 5 cm the camera was inside the arm's own
    wrist_3_link. Direction alone cannot fix a camera that is buried.

    MULTIPLE SAMPLES PER CANDIDATE, not one -- added 2026-09-15 after a
    single-sample sweep's own "winners" (688, then 1532, then 1011 cube px,
    each looking like a clear pick in its own report) turned out to each be
    one lucky cube spawn. A held-out check across 4 fresh spawns of THREE of
    those exact settings came back 0-534 px, most of them 0 more often than
    not -- the wrist camera's view of the cube at the grasp depends on where
    in CUBE_X_RANGE/CUBE_Y_RANGE the cube spawned, not just on the mount, and
    one spawn cannot tell a mount that is reliably mediocre from one that is
    occasionally excellent and often useless. Only `at_grasp` sweeps resample
    the cube each time; `at_reset` does not need to -- the arm's reset pose
    does not depend on where the cube spawned, so the wrist view there is
    already deterministic given the mount, confirmed by inspection."""
    derived = scene.derived_wrist_rotation().as_euler("xyz", degrees=True)
    say(f"\nwrist sweep -- committed: rot={WRIST_CAMERA_FLANGE_ROT_EULER} "
        f"lateral={WRIST_CAMERA_LATERAL_M}m focus={WRIST_CAMERA_FOCUS_M}m "
        f"(rot None resolves to the derived {np.round(derived, 1).tolist()})")
    if at_grasp:
        say(f"scoring each candidate over {n_samples} independent cube spawns "
            f"(pass --wrist_samples to change this)")

    scored, panels = [], []
    for lateral in WRIST_LATERAL_CANDIDATES:
        for euler in WRIST_DIRECTION_CANDIDATES:
            label = f"{tuple(int(v) for v in euler)}@{lateral:g}"
            samples = []
            last_frame = None
            for _ in range(n_samples if at_grasp else 1):
                if at_grasp:
                    _drive_to_grasp(scene, scene.reset())
                # tilt fixed at 0 for this pass -- it is swept separately,
                # against the winning direction/lateral only, below.
                last_frame = _render_wrist(scene, euler, lateral, down_tilt=0.0)
                samples.append(_metrics(last_frame, scene))
            px_values = [m["cube_px"] for m in samples]
            mean_px = sum(px_values) / len(px_values)
            min_px, max_px = min(px_values), max(px_values)
            mean_dark = sum(m["dark"] for m in samples) / len(samples)
            if at_grasp:
                say(f"  {label:18s} cube_px mean={mean_px:6.0f} min={min_px:5d} max={max_px:5d}  "
                    f"near_black={100 * mean_dark:5.1f}%  cam_at={np.round(scene.wrist_camera_world_position(), 3)}")
            else:
                say(_fmt(label, samples[0]) + f"  cam_at={np.round(scene.wrist_camera_world_position(), 3)}")
            scored.append((mean_px, min_px, max_px, euler, lateral, mean_dark))
            panels.append((label, _rgb(last_frame)))
    for i in range(0, len(panels), len(WRIST_DIRECTION_CANDIDATES)):
        row = panels[i:i + len(WRIST_DIRECTION_CANDIDATES)]
        tag = row[0][0].split("@")[1]
        _contact_sheet(row, os.path.join(out_dir, f"wrist_dirs_{stage_name}_lateral{tag}.png"))

    # Rank by the WORST sample, not the average: a mount that is reliably
    # mediocre beats one that is occasionally excellent and often at 0,
    # because collect_demos.py's preflight and every episode after it has to
    # survive whatever the random spawn happens to be, not the best case.
    scored.sort(key=lambda s: (-s[1], -s[0]))
    best_mean, best_min, best_max, best_euler, best_lateral, best_dark = scored[0]
    runner_up_min = scored[1][1]

    if best_min < 200:
        all_black = all(dark > 0.95 for _, _, _, _, _, dark in scored)
        say(f"\nNO MOUNT WORKS -- even the most reliable candidate's WORST sample saw only "
            f"{best_min} cube pixels ({best_mean:.0f} mean) at the grasp, where an eye-in-hand "
            f"camera should see thousands.")
        if all_black:
            say("Every candidate is essentially all-black at every offset, which means the "
                "camera is not seeing the scene at all rather than seeing the wrong part of it. "
                "Widen WRIST_LATERAL_CANDIDATES first -- if even the largest offset is black, "
                "the camera is probably still inside geometry, or its clipping range is cutting "
                "away everything close to it.")
        else:
            say("Some candidates do render the scene, so the mount is roughly right but no "
                "direction reliably sees the cube across different spawns. A wider "
                "CUBE_X_RANGE/CUBE_Y_RANGE than any fixed eye-in-hand mount can cover is one "
                "explanation worth ruling out before trying more mount candidates.")
        return None

    say(f"\nBEST: rot={tuple(int(v) for v in best_euler)} lateral={best_lateral:g}m -- "
        f"worst-case {best_min} cube px, mean {best_mean:.0f}, next-best worst-case {runner_up_min} "
        f"({100 * best_dark:.0f}% near-black on average)")
    if best_min < 0.5 * best_mean:
        say(f"WARNING: this candidate is unreliable even though it won -- its worst sample "
            f"({best_min}px) is under half its mean ({best_mean:.0f}px). A single-sample sweep "
            f"would have reported whichever spawn it got as if it were representative; this is "
            f"exactly the failure mode that made this a warning to check for.")
    if best_min < 2 * max(runner_up_min, 1):
        say("WARNING: the winner is not a clear one on worst-case either. Look at the contact "
            "sheets before trusting it -- the right panel shows the table filling the view from "
            "close up.")

    # Whether tilting down off the pure approach axis helps at the winning
    # direction/lateral -- see WRIST_DOWN_TILT_CANDIDATES for why this is a
    # separate pass rather than folded into the direction/lateral grid above
    # (that grid was already 18-wide before this; entangling an unvalidated
    # third axis into it would have made every existing number here
    # incomparable to past sweep reports for no real gain).
    say(f"\ntilt sweep at the winning rot={tuple(int(v) for v in best_euler)} "
        f"lateral={best_lateral:g}m:")
    tilt_scored = []
    for down_tilt in WRIST_DOWN_TILT_CANDIDATES:
        samples = []
        for _ in range(n_samples if at_grasp else 1):
            if at_grasp:
                _drive_to_grasp(scene, scene.reset())
            frame = _render_wrist(scene, best_euler, best_lateral, down_tilt=down_tilt)
            samples.append(_metrics(frame, scene))
        px_values = [m["cube_px"] for m in samples]
        mean_px = sum(px_values) / len(px_values)
        min_px = min(px_values)
        say(f"  tilt={down_tilt:>5.1f}deg  cube_px mean={mean_px:6.0f} min={min_px:5d}")
        tilt_scored.append((min_px, mean_px, down_tilt))
    tilt_scored.sort(key=lambda s: (-s[0], -s[1]))
    best_tilt = tilt_scored[0][2]
    if tilt_scored[0][0] <= best_min:
        say(f"  tilt={best_tilt:g}deg does not clearly beat tilt=0 -- keep 0 unless a later "
            f"finger-occlusion-specific check (close-standoff frames) shows otherwise.")

    # Now settle the roll about the winning viewing axis (and tilt), on a
    # fresh spawn -- scene is otherwise left wherever the last-tested
    # candidate's last sample happened to land, which is not representative
    # of the winner.
    if at_grasp:
        _drive_to_grasp(scene, scene.reset())
    roll_panels = []
    for roll in WRIST_ROLL_CANDIDATES:
        from scipy.spatial.transform import Rotation as Rot
        euler = (Rot.from_euler("xyz", best_euler, degrees=True)
                 * Rot.from_euler("z", roll, degrees=True)).as_euler("xyz", degrees=True)
        roll_panels.append((f"roll {int(roll)}",
                             _rgb(_render_wrist(scene, euler, best_lateral, down_tilt=best_tilt))))
    _contact_sheet(roll_panels, os.path.join(out_dir, f"wrist_rolls_{stage_name}.png"))
    say("Rolls of the winning direction are in the wrist_rolls sheet -- all see the same thing, "
        "so pick whichever looks upright.")

    say("\nPaste into pick_place_scene.py:")
    say(f"    WRIST_CAMERA_FLANGE_ROT_EULER = "
        f"({best_euler[0]:.1f}, {best_euler[1]:.1f}, {best_euler[2]:.1f})")
    say(f"    WRIST_CAMERA_LATERAL_M = {best_lateral:g}")
    say(f"    WRIST_CAMERA_DOWN_TILT_DEG = {best_tilt:g}")
    return best_euler, best_lateral, best_tilt


def run(out_dir, do_sweep, at_grasp, with_gripper, wrist_samples):
    os.makedirs(out_dir, exist_ok=True)
    _REPORT["path"] = os.path.join(out_dir, "report.txt")
    open(_REPORT["path"], "w").close()
    say(f"writing this report to {_REPORT['path']} as well as stdout "
        "(Kit's shutdown discards a block-buffered stdout when piped)")
    scene = PickPlaceScene(with_gripper=with_gripper)
    if not with_gripper:
        say("running WITHOUT a gripper -- the cameras are mounted on the arm, so leaving the "
              "gripper out changes nothing this script measures and avoids its instability "
              "(pass --with_gripper to include it).")
    say("scene built; resetting...")
    obs = scene.reset()
    say(f"reset OK -- cube at {np.round(scene.cube_position, 3)}, "
        f"wrist camera at {np.round(scene.wrist_camera_world_position(), 3)}")

    if at_grasp:
        obs = _drive_to_grasp(scene, obs)
        stage_name = "at_grasp"
    else:
        stage_name = "at_reset"

    say(f"\ncamera readings ({stage_name}, cube at {np.round(scene.cube_position, 3)}):")
    say(_fmt("base_rgb", _metrics(obs["base_rgb"], scene)))
    say(_fmt("wrist_rgb", _metrics(obs["wrist_rgb"], scene)))

    ok, problems = scene.preflight_check(verbose=False)
    say(f"\npreflight: {'OK' if ok else 'PROBLEMS'}")
    for p in problems:
        say(f"  - {p}")

    _contact_sheet(
        [("base_rgb", _rgb(obs["base_rgb"])), ("wrist_rgb", _rgb(obs["wrist_rgb"]))],
        os.path.join(out_dir, f"cameras_{stage_name}.png"))

    if do_sweep:
        if not at_grasp:
            say("\n(sweep scores by cube pixels at the grasp -- rerun without --at_reset for a "
                  "meaningful ranking)")
        sweep_wrist(scene, out_dir, stage_name, at_grasp, wrist_samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="camera_check",
                        help="directory the PNG contact sheets are written to")
    parser.add_argument("--sweep_wrist", action="store_true",
                        help="sweep wrist camera orientations, score each by how much of the cube "
                             "it sees, and print the winning setting")
    parser.add_argument("--with_gripper", action="store_true",
                        help="build the scene with the gripper. Off by default: the gripper is "
                             "unstable in both attachment modes (its links diverge to ~1e15 m in "
                             "teleport mode, and fixed_joint mode kills the app during reset), and "
                             "none of that affects where the cameras point")
    parser.add_argument("--at_reset", action="store_true",
                        help="look at the arm's default pose instead of at the grasp (default is "
                             "the grasp, which is what the wrist camera is actually for)")
    parser.add_argument("--wrist_samples", type=int, default=5,
                        help="independent random cube spawns to average each --sweep_wrist "
                             "candidate over (at_grasp only -- a single sample was confirmed "
                             "2026-09-15 to pick winners that were really just lucky spawns)")
    args = parser.parse_args()

    # Print any traceback BEFORE closing the app. simulation_app.close() runs
    # Kit's fastShutdown, which terminates the process before Python gets to
    # report an escaping exception -- so a crash in run() otherwise looks
    # exactly like a clean exit, and has done three times now.
    try:
        run(args.out_dir, args.sweep_wrist, at_grasp=not args.at_reset,
            with_gripper=args.with_gripper, wrist_samples=args.wrist_samples)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
