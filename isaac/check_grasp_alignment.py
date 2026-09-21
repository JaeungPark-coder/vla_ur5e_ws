"""Why does the gripper close on the cube without lifting it?

collect_demos.py (2026-09-21, this session, with a working camera mount and
no PhysX divergence for the first time) rejected 15/15 attempts as "never
lifted" -- max_cube_z stayed at 0.02-0.041m against a 0.08m LIFT_Z_THRESHOLD.
That is the only verified fact so far; this measures two things that could
explain it, fresh, rather than assuming either:

  GRIP CLOSURE    does the drive joint actually reach GRIPPER_CLOSED_POS (or
                  close to it) by the time the close segment ends, or does
                  it stall short (e.g. hitting the cube's own geometry
                  before the joint target is reached)?
  AXIS ALIGNMENT  at the moment the gripper finishes closing, how far off
                  is the grip point from the cube's centre, decomposed into
                  X/Y/Z separately? A parallel gripper is far more sensitive
                  to error along its OWN closing axis (one pad overbites,
                  the other undershoots) than to error perpendicular to it.

Logs every tick through the approach/settle/descend/close segments (not
just a single snapshot at the end), so a stall or a drift shows up as a
trend, not a single number that could be a fluke of exactly when it was
sampled.

    python3 check_grasp_alignment.py
    python3 check_grasp_alignment.py --episodes 5
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pxr import Usd, UsdGeom  # noqa: E402

from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, GRIPPER_TCP_OFFSET_M, CUBE_SIZE_M)
from scripted_pick_place import ScriptedPickPlace, GRASP_HEIGHT  # noqa: E402
from isaac_sim_common import GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, prim_world_pose  # noqa: E402

# CONFIRMED 2026-09-21 by direct prim inspection (Usd.PrimRange under
# /World/ur5e/Gripper): the two pad links that actually contact a grasped
# object. Used here to measure the REAL fingertip position independently of
# grip_point_world()/GRIPPER_TCP_OFFSET_M -- see this file's own printed
# note on why that formula-based measurement cannot detect a miscalibrated
# GRIPPER_TCP_OFFSET_M (it uses the same constant to set the target AND to
# read it back, so the two cancel algebraically regardless of the constant's
# real-world accuracy).
LEFT_INNER_FINGER_PRIM_PATH = "/World/ur5e/Gripper/Robotiq_2F_85/left_inner_finger"
RIGHT_INNER_FINGER_PRIM_PATH = "/World/ur5e/Gripper/Robotiq_2F_85/right_inner_finger"
# The actual rubber pad mesh, found by direct prim traversal (2026-09-21) --
# a child of the link above, not the link's own origin. The link origin
# turned out to be ~140mm above the cube even mid-grasp (a finger this
# short cannot physically span that), which only makes sense if the link's
# origin is the pad's PROXIMAL joint (where it mounts to the knuckle), not
# its contact surface. CONFIRMED this mesh prim carries no transform op of
# its own (its ComputeLocalToWorldTransform is identical to its parent
# link's), so the pad's real offset from the link origin is baked into the
# mesh's own vertex positions -- mesh_world_bbox_center below reads those
# directly and transforms them to world space, rather than trusting the
# prim's own (uninformative here) transform.
LEFT_FINGERTIP_MESH_PATH = (
    "/World/ur5e/Gripper/Robotiq_2F_85/left_inner_finger/visuals/"
    "Defeatured_2F_85_PAD_OPEN_fingertipsstep_01/Defeatured_2F_85_PAD_OPEN_fingertipsstep")
RIGHT_FINGERTIP_MESH_PATH = (
    "/World/ur5e/Gripper/Robotiq_2F_85/right_inner_finger/visuals/"
    "Defeatured_2F_85_PAD_OPEN_fingertipsstep_01/Defeatured_2F_85_PAD_OPEN_fingertipsstep")


def say(line=""):
    print(line, flush=True)


def mesh_world_bbox_center(prim):
    """World-space centroid of a mesh's own vertex bounding box -- unlike
    prim_world_pose (which reads only the prim's transform op, and returns
    the SAME point as an unshifted parent when the mesh authors no
    transform of its own), this reads the actual vertex positions, so a
    pad mesh whose real extent is baked into its points rather than its
    transform still gives a physically meaningful location."""
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_points = np.array([[*mat.Transform(p)] for p in points], dtype=float)
    return world_points.mean(axis=0), world_points.min(axis=0), world_points.max(axis=0)


def run(episodes):
    scene = PickPlaceScene(with_gripper=True)
    say(f"GRIPPER_OPEN_POS={GRIPPER_OPEN_POS} GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS} rad")

    for ep in range(1, episodes + 1):
        obs = scene.reset()
        policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
        waypoints = policy.waypoints
        # waypoints[:4] = approach, settle, descend, close (see
        # ScriptedPickPlace.frames_until_grasp's own comment on this layout)
        n_through_close = sum(n for _, _, n in waypoints[:4])
        segment_bounds = np.cumsum([n for _, _, n in waypoints[:4]])
        segment_names = ["approach", "settle", "descend", "close"]

        say(f"\n=== episode {ep}/{episodes}: cube at {np.round(scene.cube_position, 4).tolist()} ===")
        say(f"{'tick':>5} {'seg':>8} {'gripper_tgt':>11} {'gripper_act':>11} "
            f"{'dx_mm':>7} {'dy_mm':>7} {'dz_mm':>7} {'|d|_mm':>7} {'pad_z_gap_mm':>13}")

        frames = list(policy.generate_frames())
        seg_idx = 0
        for tick, (target_pos, target_rotvec, target_gripper) in enumerate(frames[:n_through_close], start=1):
            scene.step_towards(target_pos, target_rotvec, target_gripper)
            while seg_idx < len(segment_bounds) - 1 and tick > segment_bounds[seg_idx]:
                seg_idx += 1
            seg_name = segment_names[seg_idx]

            grip_point = scene.grip_point_world()
            cube_pos = scene.get_cube_position()
            # Grip point sits GRIPPER_TCP_OFFSET_M ABOVE where the fingers
            # actually pinch (see _grip_pose_to_tool0/grip_point_world) --
            # compare against the cube's centre directly, same quantity the
            # scripted waypoints (at_cube = cube_position + GRASP_HEIGHT)
            # target, so dz should read close to GRASP_HEIGHT once the
            # close segment starts, not 0.
            delta = grip_point - cube_pos
            actual_gripper = scene.gripper.get_normalized_position()

            # ONLY during descend/close, and only every few ticks (this is
            # the real vertex-bbox read, not the formula -- worth the extra
            # cost here specifically to test whether the dx drift's onset
            # lines up with the pad's own lowest point reaching the cube's
            # actual top face, which would confirm early contact rather
            # than a pure control/kinematic effect).
            pad_gap_str = ""
            if seg_name in ("descend", "close") and tick % 4 == 0:
                result = mesh_world_bbox_center(scene.stage.GetPrimAtPath(LEFT_FINGERTIP_MESH_PATH))
                if result is not None:
                    _, bbox_min, _ = result
                    cube_top_z = cube_pos[2] + CUBE_SIZE_M / 2.0
                    pad_gap_str = f"{(bbox_min[2] - cube_top_z) * 1000:+13.1f}"

            # Print every tick during "close" (short segment, matters most),
            # every 4th during descend (this is where the swing happens),
            # every 20th otherwise (long segments, just show the trend).
            if (seg_name == "close" or (seg_name == "descend" and tick % 4 == 0)
                    or tick % 20 == 0 or tick == n_through_close):
                say(f"{tick:>5} {seg_name:>8} {target_gripper:>11.3f} {actual_gripper:>11.3f} "
                    f"{delta[0] * 1000:>7.1f} {delta[1] * 1000:>7.1f} {delta[2] * 1000:>7.1f} "
                    f"{float(np.linalg.norm(delta)) * 1000:>7.1f} {pad_gap_str:>13}")

        final_gripper_frac = scene.gripper.get_normalized_position()
        final_joint_rad = GRIPPER_OPEN_POS + final_gripper_frac * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
        say(f"\n  end of close segment: gripper at {100 * final_gripper_frac:.1f}% of target range "
            f"({final_joint_rad:.4f} rad vs GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS} rad)")
        grip_point = scene.grip_point_world()
        cube_pos = scene.get_cube_position()
        delta = grip_point - cube_pos
        say(f"  final grip-point - cube offset (FORMULA, self-consistent -- see module "
            f"docstring on why this cannot reveal a bad GRIPPER_TCP_OFFSET_M): "
            f"dx={delta[0] * 1000:+.1f}mm dy={delta[1] * 1000:+.1f}mm dz={delta[2] * 1000:+.1f}mm "
            f"(target dz was GRASP_HEIGHT={GRASP_HEIGHT * 1000:.0f}mm)")

        # INDEPENDENT ground truth: the actual pad links' own live world
        # pose, read straight off the physics-driven prims -- does not go
        # through GRIPPER_TCP_OFFSET_M at all, so a miscalibration there
        # (invisible to the formula-based delta above) would show up here
        # as the pads sitting somewhere other than the cube's actual body.
        for label, path in (("left_inner_finger (LINK origin)", LEFT_INNER_FINGER_PRIM_PATH),
                             ("right_inner_finger (LINK origin)", RIGHT_INNER_FINGER_PRIM_PATH)):
            pos, _ = prim_world_pose(scene.stage.GetPrimAtPath(path))
            d = pos - cube_pos
            say(f"  {label:32s} - cube: dx={d[0] * 1000:+7.1f}mm dy={d[1] * 1000:+7.1f}mm "
                f"dz={d[2] * 1000:+7.1f}mm "
                f"(cube top +{1000 * CUBE_SIZE_M / 2:.0f}mm, bottom -{1000 * CUBE_SIZE_M / 2:.0f}mm)")

        # Same, but the pad MESH's actual vertex bounding box in world space
        # (see mesh_world_bbox_center's own docstring on why the link
        # origin above cannot be trusted as "where the pad actually is").
        for label, path in (("left fingertip MESH bbox", LEFT_FINGERTIP_MESH_PATH),
                             ("right fingertip MESH bbox", RIGHT_FINGERTIP_MESH_PATH)):
            result = mesh_world_bbox_center(scene.stage.GetPrimAtPath(path))
            if result is None:
                say(f"  {label:32s} - mesh has no points; skipped")
                continue
            centroid, bbox_min, bbox_max = result
            d_centroid = centroid - cube_pos
            say(f"  {label:32s} centroid-cube: dx={d_centroid[0] * 1000:+7.1f}mm "
                f"dy={d_centroid[1] * 1000:+7.1f}mm dz={d_centroid[2] * 1000:+7.1f}mm "
                f"| bbox z-range {bbox_min[2] * 1000:.1f}..{bbox_max[2] * 1000:.1f}mm "
                f"(cube z={cube_pos[2] * 1000:.1f}mm, top={1000 * (cube_pos[2] + CUBE_SIZE_M / 2):.1f}mm)")

        # Continue driving through lift to report the actual outcome this
        # episode would have scored, for direct comparison with
        # collect_demos.py's own verdict.
        max_cube_z = float(cube_pos[2])
        for target_pos, target_rotvec, target_gripper in frames[n_through_close:]:
            scene.step_towards(target_pos, target_rotvec, target_gripper)
            max_cube_z = max(max_cube_z, float(scene.get_cube_position()[2]))
        say(f"  max_cube_z over the rest of the episode: {max_cube_z:.4f}m "
            f"(lifted={'YES' if max_cube_z >= 0.08 else 'no'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()
    try:
        run(args.episodes)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
