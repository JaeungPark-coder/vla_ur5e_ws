"""Two-stage wrist mount search: cheap geometry first, expensive rendering
only on what survives it.

check_cameras.py --sweep_wrist scores every candidate by actually rendering
it -- correct, but each render needs the RTX pipeline (a full frame,
GPU-bound), so the candidate grid stays small (18 direction/lateral combos
today) to keep runtime sane. That grid has come back "NO MOUNT WORKS" twice
now (once confounded by a since-fixed reach bug, once for real -- see
pick_place_scene.py's CUBE_X_RANGE/CUBE_Y_RANGE comment and this project's
own git history), which means the 18-candidate grid itself may simply be
too coarse, not that no mount exists.

This script separates "is the cube even geometrically visible from here"
(FOV cone membership + an unoccluded PhysX raycast to the cube's centre --
both cheap, no RTX, no pixel readback) from "does it actually render well"
(still needs real frames, but only for however many candidates survive
stage 1). That lets stage 1 sweep a MUCH finer grid -- lateral and tilt at
fine steps, not just the handful check_cameras.py can afford to render --
before stage 2 spends any GPU time confirming the survivors.

    python3 check_wrist_mount_raycast.py
    python3 check_wrist_mount_raycast.py --grasp-samples 5 --render-top 8
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
import omni.physx  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, CUBE_PRIM_PATH,
    WRIST_CAMERA_HORIZONTAL_FOV_DEG, WRIST_CAMERA_PRIM_PATH,
    WRIST_CAMERA_FOCUS_M, WRIST_CAMERA_BACK_M, MAX_DARK_FRACTION)
from scripted_pick_place import ScriptedPickPlace
from isaac_sim_common import prim_world_pose

# The 6 axis-aligned viewing directions check_cameras.py already covers --
# kept identical so a stage-1 winner's direction is directly comparable to
# past render-based sweep reports.
DIRECTION_CANDIDATES = [
    (0.0, 0.0, 0.0), (180.0, 0.0, 0.0), (90.0, 0.0, 0.0),
    (-90.0, 0.0, 0.0), (0.0, 90.0, 0.0), (0.0, -90.0, 0.0),
]
# Finer than check_cameras.py's [0.05, 0.08, 0.12] -- cheap here, since
# nothing in stage 1 renders a frame.
LATERAL_CANDIDATES = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16]
# Signed tilt off the pure approach axis (see WRIST_CAMERA_DOWN_TILT_DEG) --
# both signs, since this project has no verified convention for which sign
# means "down" in the flange frame.
TILT_CANDIDATES = [-45.0, -30.0, -15.0, 0.0, 15.0, 30.0, 45.0]

# Half-angle of the FOV cone used for the "would the cube even land in
# frame" check. WRIST_CAMERA_HORIZONTAL_FOV_DEG with a square aperture/
# resolution (see pick_place_scene._setup_cameras's verticalAperture fix)
# means horizontal and vertical FOV are equal, so a single circular cone
# half-angle is a reasonable (slightly conservative on the corners of a
# square frame) stand-in for the real rectangular frustum -- stage 2's
# actual render is what confirms it for real.
HALF_FOV_RAD = np.radians(WRIST_CAMERA_HORIZONTAL_FOV_DEG) / 2.0


def say(line=""):
    print(line, flush=True)


def _drive_to_grasp(scene):
    """Same pattern as check_cameras.py's own _drive_to_grasp -- duplicated
    rather than imported for the same reason check_near_field_visibility.py
    already gives (that module boots its own SimulationApp)."""
    obs = scene.reset()
    policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
    frames = list(policy.generate_frames())
    n = policy.frames_until_grasp()
    for target_pos, target_rotvec, target_gripper in frames[:n]:
        scene.step_towards(target_pos, target_rotvec, target_gripper)
    obs = scene.get_observation()
    reach = float(np.linalg.norm(np.asarray(obs["tool_pos"]) - scene.get_cube_position()))
    return reach


def evaluate_candidate(scene, euler, lateral, tilt):
    """Geometry-only verdict for one mount at the CURRENT grasp pose:
    (in_fov, unoccluded, distance_m). No rendering."""
    scene.set_wrist_camera_flange_rotation(
        euler, lateral_m=lateral, focus_m=WRIST_CAMERA_FOCUS_M,
        back_m=WRIST_CAMERA_BACK_M, down_tilt_deg=tilt)

    cam_pos, cam_quat = prim_world_pose(scene.stage.GetPrimAtPath(WRIST_CAMERA_PRIM_PATH))
    forward = Rot.from_quat(cam_quat).apply(np.array([0.0, 0.0, -1.0]))
    forward /= np.linalg.norm(forward)

    cube_pos = scene.get_cube_position()
    to_cube = cube_pos - cam_pos
    distance = float(np.linalg.norm(to_cube))
    if distance < 1e-6:
        return False, False, distance
    to_cube_dir = to_cube / distance

    cos_angle = float(np.dot(forward, to_cube_dir))
    in_fov = cos_angle >= np.cos(HALF_FOV_RAD)

    hit = omni.physx.get_physx_scene_query_interface().raycast_closest(
        tuple(cam_pos.tolist()), tuple(to_cube_dir.tolist()), distance + 0.02)
    # "unoccluded" means the first thing this ray hits IS the cube, not
    # something (a finger, the flange, the table) sitting in front of it.
    unoccluded = bool(hit.get("hit")) and str(hit.get("rigidBody", "")).startswith(CUBE_PRIM_PATH)
    return in_fov, unoccluded, distance


def run(grasp_samples, render_top, wrist_render_samples, out_dir):
    scene = PickPlaceScene(with_gripper=True)
    say(f"scene built; sampling {grasp_samples} fresh grasp poses for stage 1...")

    grasp_reaches = []
    for i in range(grasp_samples):
        reach = _drive_to_grasp(scene)
        grasp_reaches.append(reach)
        if reach > 0.06:
            say(f"  WARNING: grasp sample {i + 1} missed by {reach * 1000:.0f}mm -- "
                f"stage 1 will screen against wherever the tool actually ended up, "
                f"not a real grasp, for this sample")

        # --- stage 1: raycast screening across the full candidate grid ---
        results = {}
        for euler in DIRECTION_CANDIDATES:
            for lateral in LATERAL_CANDIDATES:
                for tilt in TILT_CANDIDATES:
                    key = (tuple(euler), lateral, tilt)
                    in_fov, unoccluded, distance = evaluate_candidate(scene, euler, lateral, tilt)
                    ok = in_fov and unoccluded
                    prev = results.get(key, (0, 0))
                    results[key] = (prev[0] + (1 if ok else 0), prev[1] + 1)
        if i == 0:
            tally = results
        else:
            for key, (hits, total) in results.items():
                prev_hits, prev_total = tally.get(key, (0, 0))
                tally[key] = (prev_hits + hits, prev_total + total)

    n_combos = len(DIRECTION_CANDIDATES) * len(LATERAL_CANDIDATES) * len(TILT_CANDIDATES)
    say(f"\nstage 1 done: {n_combos} candidates x {grasp_samples} grasp samples, "
        f"{n_combos * grasp_samples} raycasts, no rendering")
    say(f"grasp reach across samples: {[f'{r * 1000:.0f}mm' for r in grasp_reaches]}")

    ranked = sorted(tally.items(), key=lambda kv: -kv[1][0])
    say(f"\ntop {min(render_top, len(ranked))} stage-1 survivors "
        f"(hits/{grasp_samples} grasp samples, in-FOV AND unoccluded both required):")
    survivors = []
    for (euler, lateral, tilt), (hits, total) in ranked[:render_top]:
        say(f"  rot={tuple(int(v) for v in euler)!s:18s} lateral={lateral:.2f} tilt={tilt:>5.0f}deg "
            f"-> {hits}/{total}")
        if hits > 0:
            survivors.append((euler, lateral, tilt))

    if not survivors:
        say("\nNO CANDIDATE ever had the cube in-FOV and unoccluded, even once, across "
            f"{grasp_samples} grasp samples. That is a stronger result than check_cameras.py's "
            "render-based 'NO MOUNT WORKS' -- it rules out framing/exposure as confounds "
            "entirely, so if nothing renders well either, the mount geometry itself (not just "
            "the angle) needs rethinking -- e.g. WRIST_CAMERA_BACK_M/FOCUS_M, not just direction "
            "and lateral.")
        return

    # --- stage 2: render the survivors for real ---
    say(f"\nstage 2: rendering the {len(survivors)} survivor(s) for real "
        f"({wrist_render_samples} fresh grasp samples each)...")
    os.makedirs(out_dir, exist_ok=True)
    scored = []
    for euler, lateral, tilt in survivors:
        samples = []
        for _ in range(wrist_render_samples):
            _drive_to_grasp(scene)
            scene.set_wrist_camera_flange_rotation(
                euler, lateral_m=lateral, focus_m=WRIST_CAMERA_FOCUS_M,
                back_m=WRIST_CAMERA_BACK_M, down_tilt_deg=tilt)
            for _ in range(2):
                scene.world.render()
            frame = np.asarray(scene.get_observation()["wrist_rgb"])
            rgb = frame[..., :3].astype(np.float32)
            dark = float((rgb.max(axis=2) < 12).mean())
            cube_px = scene.cube_pixels_visible(frame)
            samples.append((cube_px, dark))
        mean_px = sum(s[0] for s in samples) / len(samples)
        min_px = min(s[0] for s in samples)
        mean_dark = sum(s[1] for s in samples) / len(samples)
        say(f"  rot={tuple(int(v) for v in euler)!s:18s} lateral={lateral:.2f} tilt={tilt:>5.0f}deg "
            f"-> cube_px mean={mean_px:6.0f} min={min_px:5d}  near_black={100 * mean_dark:5.1f}%"
            + ("  <-- under MAX_DARK_FRACTION" if mean_dark < MAX_DARK_FRACTION else ""))
        scored.append((min_px, mean_px, mean_dark, euler, lateral, tilt))

    scored.sort(key=lambda s: (-s[0], -s[1]))
    best = scored[0]
    say(f"\nBEST (by worst-case cube_px): rot={tuple(int(v) for v in best[3])} "
        f"lateral={best[4]:.2f} tilt={best[5]:.0f}deg -- worst-case {best[0]} cube px, "
        f"mean {best[1]:.0f}, near_black {100 * best[2]:.0f}%")
    if best[2] >= MAX_DARK_FRACTION:
        say(f"WARNING: even the best survivor's near_black ({100 * best[2]:.0f}%) is still >= "
            f"MAX_DARK_FRACTION ({100 * MAX_DARK_FRACTION:.0f}%) -- collect_demos.py's preflight "
            f"would still reject this one.")
    else:
        say("This candidate would pass preflight_check's near_black gate.")
    say("\nPaste into pick_place_scene.py if it looks right:")
    say(f"    WRIST_CAMERA_FLANGE_ROT_EULER = "
        f"({best[3][0]:.1f}, {best[3][1]:.1f}, {best[3][2]:.1f})")
    say(f"    WRIST_CAMERA_LATERAL_M = {best[4]:g}")
    say(f"    WRIST_CAMERA_DOWN_TILT_DEG = {best[5]:g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--grasp-samples", type=int, default=5,
                        help="fresh random grasp poses stage 1 screens the whole grid against")
    parser.add_argument("--render-top", type=int, default=10,
                        help="how many top stage-1 survivors stage 2 actually renders")
    parser.add_argument("--wrist-render-samples", type=int, default=3,
                        help="fresh grasp samples stage 2 renders per surviving candidate")
    parser.add_argument("--out_dir", type=str, default="camera_check")
    args = parser.parse_args()

    try:
        run(args.grasp_samples, args.render_top, args.wrist_render_samples, args.out_dir)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
