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

Runs WITHOUT a gripper by default. The cameras are mounted on the arm, so
the gripper changes nothing this script measures -- and it is currently
unstable in both attachment modes (in teleport mode its links diverge to
world positions around 1e15 m, drowning the log; in fixed_joint mode it
takes the app down during reset). Leaving it out keeps the camera question
answerable while the gripper is still unresolved. --with_gripper includes it.

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
    WRIST_CAMERA_LATERAL_M, WRIST_CAMERA_FOCUS_M)
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
WRIST_LATERAL_CANDIDATES = [0.05, 0.08, 0.12]

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


def _render_wrist(scene, euler, lateral=None):
    scene.set_wrist_camera_flange_rotation(euler, lateral_m=lateral)
    # render(), NOT step(): re-aiming a camera needs a fresh render pass, not
    # more physics. Stepping here was actively harmful -- in the default
    # "teleport" gripper mode the gripper is held on the flange only by
    # PickPlaceScene.step_towards's per-tick re-sync, so every bare
    # world.step() let it drop away from the arm and interpenetrate it,
    # producing the "Invalid PhysX transform" / "Illegal BroadPhaseUpdateData"
    # storm. World.render() refreshes the app (and so the annotators) with
    # /app/player/playSimulations off, leaving the scene exactly where
    # _drive_to_grasp left it. Two passes because the first one after an
    # authoring change can still hand back the previous frame.
    for _ in range(2):
        scene.world.render()
    return scene.get_observation()["wrist_rgb"]


def sweep_wrist(scene, out_dir, stage_name):
    """Sweep the wrist camera's mount -- direction AND lateral offset -- and score
    each by how much of the cube it actually sees.

    Both axes are swept because the first attempt failed on the one that was
    not being varied: every orientation rendered solid black, because at the
    original 5 cm the camera was inside the arm's own wrist_3_link. Direction
    alone cannot fix a camera that is buried."""
    derived = scene.derived_wrist_rotation().as_euler("xyz", degrees=True)
    say(f"\nwrist sweep -- committed: rot={WRIST_CAMERA_FLANGE_ROT_EULER} "
        f"lateral={WRIST_CAMERA_LATERAL_M}m focus={WRIST_CAMERA_FOCUS_M}m "
        f"(rot None resolves to the derived {np.round(derived, 1).tolist()})")

    scored, panels = [], []
    for lateral in WRIST_LATERAL_CANDIDATES:
        for euler in WRIST_DIRECTION_CANDIDATES:
            frame = _render_wrist(scene, euler, lateral)
            m = _metrics(frame, scene)
            label = f"{tuple(int(v) for v in euler)}@{lateral:g}"
            say(_fmt(label, m) + f"  cam_at={np.round(scene.wrist_camera_world_position(), 3)}")
            scored.append((m["cube_px"], euler, lateral, m))
            panels.append((label, _rgb(frame)))
    for i in range(0, len(panels), len(WRIST_DIRECTION_CANDIDATES)):
        row = panels[i:i + len(WRIST_DIRECTION_CANDIDATES)]
        tag = row[0][0].split("@")[1]
        _contact_sheet(row, os.path.join(out_dir, f"wrist_dirs_{stage_name}_lateral{tag}.png"))

    scored.sort(key=lambda s: -s[0])
    best_px, best_euler, best_lateral, best_m = scored[0]
    runner_up = scored[1][0]

    if best_px < 200:
        all_black = all(m["dark"] > 0.95 for _, _, _, m in scored)
        say(f"\nNO MOUNT WORKS -- the best sees only {best_px} cube pixels at the grasp, where "
            f"an eye-in-hand camera should see thousands.")
        if all_black:
            say("Every candidate is essentially all-black at every offset, which means the "
                "camera is not seeing the scene at all rather than seeing the wrong part of it. "
                "Widen WRIST_LATERAL_CANDIDATES first -- if even the largest offset is black, "
                "the camera is probably still inside geometry, or its clipping range is cutting "
                "away everything close to it.")
        else:
            say("Some candidates do render the scene, so the mount is roughly right but no "
                "direction is looking at the cube. Check that the arm actually reached the cube "
                "(the base_rgb panel of the cameras_* sheet shows where it ended up).")
        return None

    say(f"\nBEST: rot={tuple(int(v) for v in best_euler)} lateral={best_lateral:g}m "
        f"with {best_px} cube pixels ({100 * best_m['dark']:.0f}% near-black), "
        f"next best {runner_up}")
    if best_px < 2 * max(runner_up, 1):
        say("WARNING: the winner is not a clear one. Look at the contact sheets before trusting "
            "it -- the right panel shows the table filling the view from close up.")

    # Now settle the roll about the winning viewing axis.
    roll_panels = []
    for roll in WRIST_ROLL_CANDIDATES:
        from scipy.spatial.transform import Rotation as Rot
        euler = (Rot.from_euler("xyz", best_euler, degrees=True)
                 * Rot.from_euler("z", roll, degrees=True)).as_euler("xyz", degrees=True)
        roll_panels.append((f"roll {int(roll)}", _rgb(_render_wrist(scene, euler, best_lateral))))
    _contact_sheet(roll_panels, os.path.join(out_dir, f"wrist_rolls_{stage_name}.png"))
    say("Rolls of the winning direction are in the wrist_rolls sheet -- all see the same thing, "
        "so pick whichever looks upright.")

    say("\nPaste into pick_place_scene.py:")
    say(f"    WRIST_CAMERA_FLANGE_ROT_EULER = "
        f"({best_euler[0]:.1f}, {best_euler[1]:.1f}, {best_euler[2]:.1f})")
    say(f"    WRIST_CAMERA_LATERAL_M = {best_lateral:g}")
    return best_euler, best_lateral


def run(out_dir, do_sweep, at_grasp, with_gripper):
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
        sweep_wrist(scene, out_dir, stage_name)


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
    args = parser.parse_args()

    # Print any traceback BEFORE closing the app. simulation_app.close() runs
    # Kit's fastShutdown, which terminates the process before Python gets to
    # report an escaping exception -- so a crash in run() otherwise looks
    # exactly like a clean exit, and has done three times now.
    try:
        run(args.out_dir, args.sweep_wrist, at_grasp=not args.at_reset,
            with_gripper=args.with_gripper)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
