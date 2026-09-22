"""Measures the frames this project has been guessing at, in one Isaac Sim run.

Every reach failure so far traces back to one question nobody measured: what
is the rigid transform between the flange prim, the frame RMPflow drives
("tool0"), and the gripper's fingertips? Three separate guesses have been
committed and all three were wrong, each costing a collection run.

The URDF the RMPflow config itself loads (motion_policy_configs/
universal_robots/ur5e/ur5e.urdf) already answers most of it analytically:

    wrist_3-flange  rpy = (0, -pi/2, -pi/2)   xyz = 0
    flange-tool0    rpy = (pi/2, 0,  pi/2)    xyz = 0

which composes to R_wrist3->tool0 = IDENTITY. So tool0 IS the wrist_3_link
frame, its +Z is the axis pointing out of the wrist -- the approach direction
-- and it shares the flange's ORIGIN exactly (both fixed joints have zero
translation). In that same algebra the flange's own +Z is NOT the tool axis;
flange +X is (ROS-Industrial's flange convention is a rotated frame). The
committed code has it the other way around, which is why correcting for a
"90 degree error" made the reach worse (65-128mm -> 294mm) rather than
better: it rotated a frame that was already right and offset the TCP along
an axis 90 degrees off the fingers.

This script checks that algebra against the actual USD asset -- the URDF and
the USD are different files and only the USD is what runs -- and measures
the two things the URDF cannot know: where the gripper's fingertips actually
are, and whether RMPflow tracks a commanded pose closely enough to grasp.

    python3 measure_frames.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pxr import Usd, UsdGeom  # noqa: E402
from scipy.spatial.transform import Rotation as Rot  # noqa: E402

from isaac_sim_common import (  # noqa: E402
    ROBOT_PRIM_PATH, TOOL_LINK_PRIM_PATH, WRIST_3_LINK_PRIM_PATH, prim_world_pose)
from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402
from scripted_pick_place import DOWNWARD_ROTVEC  # noqa: E402

# What the URDF says, for the USD to be checked against.
URDF_FLANGE_TO_TOOL0_EULER_XYZ_DEG = (90.0, 0.0, 90.0)

_REPORT = {"path": None}


def say(line=""):
    # Same reason as check_cameras.py: Kit's fastShutdown discards a
    # block-buffered stdout when this is piped, losing the whole report.
    print(line, flush=True)
    if _REPORT["path"]:
        with open(_REPORT["path"], "a", encoding="utf-8") as f:
            f.write(line + "\n")


def as_rotation(rot):
    """RMPflow's get_end_effector_pose returns its rotation as either a 3x3
    matrix or a wxyz quaternion depending on version -- accept both rather
    than assuming, since assuming its layout is one of the things that put
    this project here."""
    rot = np.asarray(rot)
    if rot.shape == (3, 3):
        return Rot.from_matrix(rot)
    if rot.shape == (4,):
        return Rot.from_quat(rot[[1, 2, 3, 0]])  # wxyz -> xyzw
    raise RuntimeError(f"unexpected end-effector rotation layout {rot.shape}: {rot}")


# Where the asset's Gripper variant puts the gripper's own root. Printed in
# the tool0 frame at several configurations: if the gripper is genuinely on
# the wrist, that reading is identical in every row.
GRIPPER_BASE_PRIM_PATH = f"{ROBOT_PRIM_PATH}/Gripper/Robotiq_2F_85/base_link"


def settle(scene, q):
    """Move the arm to q and let physics actually get there.

    set_joint_positions writes the STATE; the articulation's drives still
    hold their previous TARGETS and immediately pull back toward them, and
    anything attached to the wrist by a joint (rather than by USD parenting)
    needs real solver steps to catch up with a teleport. Stepping three times
    and reading -- what the first version of this did -- measures the
    transient, not the pose, and made a correctly-attached gripper look like
    a detached one."""
    from isaacsim.core.utils.types import ArticulationAction

    q = np.asarray(q, dtype=float)
    idx = np.arange(6)
    scene.robot.set_joint_positions(q, joint_indices=idx)
    scene.robot.set_joint_velocities(np.zeros(6), joint_indices=idx)
    for _ in range(90):
        scene.robot.apply_action(ArticulationAction(joint_positions=q, joint_indices=idx))
        scene.world.step(render=False)


def measure_frame_offsets(scene):
    """The flange -> tool0 rigid transform, measured at several joint
    configurations. Measuring at more than one is the point: a fixed frame
    offset must come out identical at every configuration, so if these
    disagree the readout itself is wrong (mismatched frames, a stale stage,
    or a robot base RMPflow doesn't know about) rather than merely surprising."""
    say("\n=== flange -> tool0, measured at several joint configurations ===")
    say("(a rigid offset MUST be identical in every row; if not, the readout is wrong)")

    configs = {
        "home": np.zeros(6),
        "tucked": np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0]),
        "reaching": np.array([0.3, -0.9, 1.1, -1.8, -1.57, 0.6]),
    }
    results = {}
    for label, q in configs.items():
        settle(scene, q)

        q_now = np.asarray(scene.robot.get_joint_positions())[:6]
        flange_pos, flange_quat = prim_world_pose(scene.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        w3_pos, w3_quat = prim_world_pose(scene.stage.GetPrimAtPath(WRIST_3_LINK_PRIM_PATH))
        ee_pos, ee_rot = scene.rmpflow.get_end_effector_pose(q_now)
        ee_pos = np.asarray(ee_pos, dtype=float)
        r_ee = as_rotation(ee_rot)
        r_flange = Rot.from_quat(flange_quat)
        r_w3 = Rot.from_quat(w3_quat)

        # Offset expressed in the flange's own frame -- the quantity
        # _grip_pose_to_tool0 assumes is zero.
        offset_in_flange = r_flange.inv().apply(ee_pos - flange_pos)
        r_flange_to_ee = r_flange.inv() * r_ee
        r_w3_to_ee = r_w3.inv() * r_ee

        results[label] = {
            "offset_in_flange_mm": offset_in_flange * 1000.0,
            "flange_to_ee_euler": r_flange_to_ee.as_euler("xyz", degrees=True),
            "w3_to_ee_deg": float(np.degrees(r_w3_to_ee.magnitude())),
            "flange_vs_w3_pos_mm": float(np.linalg.norm(flange_pos - w3_pos) * 1000.0),
            "ee_approach_world": r_ee.apply([0.0, 0.0, 1.0]),
            "flange_z_world": r_flange.apply([0.0, 0.0, 1.0]),
            "flange_x_world": r_flange.apply([1.0, 0.0, 0.0]),
        }
        r = results[label]
        say(f"\n  [{label}] q={np.round(q_now, 3)}")
        say(f"    tool0 pos - flange pos, in flange frame: {np.round(r['offset_in_flange_mm'], 2)} mm")
        say(f"    R(flange->tool0) as xyz euler deg:       {np.round(r['flange_to_ee_euler'], 2)}")
        say(f"    angle between wrist_3_link and tool0:    {r['w3_to_ee_deg']:.2f} deg")
        say(f"    tool0 +Z (approach) in world:            {np.round(r['ee_approach_world'], 4)}")
        say(f"    flange +Z in world:                      {np.round(r['flange_z_world'], 4)}")
        say(f"    flange +X in world:                      {np.round(r['flange_x_world'], 4)}")

    say("\n  --- verdict ---")
    eulers = np.array([r["flange_to_ee_euler"] for r in results.values()])
    offsets = np.array([r["offset_in_flange_mm"] for r in results.values()])
    spread = float(np.abs(eulers - eulers[0]).max())
    if spread > 1.0:
        say(f"    UNRELIABLE: R(flange->tool0) varies by {spread:.1f} deg across configurations, "
            "so it is not a fixed frame offset. Everything below is untrustworthy; the likely "
            "cause is RMPflow reporting in a robot-base frame that is not the world frame "
            "(rmpflow.set_robot_base_pose) or a stage that has not caught up with the joints.")
        return None

    say(f"    R(flange->tool0) is consistent to {spread:.3f} deg across configurations.")
    say(f"    measured : {np.round(eulers[0], 2)} xyz deg")
    say(f"    URDF says: {URDF_FLANGE_TO_TOOL0_EULER_XYZ_DEG} xyz deg")
    if np.abs(eulers[0] - np.array(URDF_FLANGE_TO_TOOL0_EULER_XYZ_DEG)).max() < 1.0:
        say("    -> the USD matches the URDF. tool0 IS wrist_3_link, its +Z is the tool axis,")
        say("       and the flange prim is the rotated ROS-Industrial frame whose +X is that")
        say("       same axis. Commanding RMPflow needs NO rotation correction -- only the TCP")
        say("       offset along tool0 +Z.")
    else:
        say("    -> the USD does NOT match the URDF. Use the MEASURED value; the code must")
        say("       compose this rotation rather than assume identity.")
    say(f"    position offset spread across configs: {np.abs(offsets - offsets[0]).max():.3f} mm")
    say(f"    tool0 and flange share an origin to within {np.abs(offsets[0]).max():.3f} mm "
        "(the URDF says exactly zero)")
    return results


def measure_gripper_tcp(scene):
    """Where the fingertips actually are, relative to tool0.

    Read off the gripper's own finger geometry rather than taken on trust:
    GRIPPER_TCP_OFFSET_M = 0.12 is currently a number in a comment, and the
    axis it was measured along (flange +Z) is one the URDF says is not the
    tool axis at all."""
    say("\n=== gripper fingertips, relative to tool0 ===")
    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    pads = {}
    for prim in Usd.PrimRange(robot_prim):
        name = prim.GetName().lower()
        if name in ("left_inner_finger", "right_inner_finger"):
            pads[name] = prim
    for name, prim in sorted(pads.items()):
        say(f"  {name} is at {prim.GetPath()}")
    if len(pads) != 2:
        say(f"  expected both inner finger pads, found {sorted(pads)} -- cannot measure.")
        return None

    # Measured at several joint configurations for the same reason the frame
    # offset is: if the gripper is rigidly on the wrist, its pose in the tool0
    # frame is identical in every row. If it drifts, the gripper is not
    # actually attached and no TCP constant can be right.
    configs = {
        "tucked": np.array([0.0, -1.2, 1.4, -1.7, -1.57, 0.0]),
        "reaching": np.array([0.3, -0.9, 1.1, -1.8, -1.57, 0.6]),
        "home2": np.array([0.0, -1.0, 1.0, -1.5, -1.57, 0.0]),
    }
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    tcps = []
    for label, q in configs.items():
        settle(scene, q)
        q_now = np.asarray(scene.robot.get_joint_positions())[:6]
        ee_pos, ee_rot = scene.rmpflow.get_end_effector_pose(q_now)
        r_ee = as_rotation(ee_rot)
        ee_pos = np.asarray(ee_pos, dtype=float)

        base = scene.stage.GetPrimAtPath(GRIPPER_BASE_PRIM_PATH)
        if base.IsValid():
            base_pos, _ = prim_world_pose(base)
            say(f"    gripper base_link, in tool0: "
                f"{np.round(r_ee.inv().apply(base_pos - ee_pos) * 1000, 1)} mm")

        centres, spans = {}, {}
        for name, prim in pads.items():
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = np.array(rng.GetMin()), np.array(rng.GetMax())
            corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            in_tool0 = r_ee.inv().apply(corners - ee_pos)
            centres[name] = in_tool0.mean(axis=0)
            spans[name] = (in_tool0.min(axis=0), in_tool0.max(axis=0))

        # The grasp point of a parallel gripper is midway between its pads.
        tcp = 0.5 * (centres["left_inner_finger"] + centres["right_inner_finger"])
        opening = centres["right_inner_finger"] - centres["left_inner_finger"]
        tcps.append(tcp)
        say(f"\n  [{label}]")
        for name in sorted(pads):
            lo, hi = spans[name]
            say(f"    {name:20s} centre {np.round(centres[name] * 1000, 1)} mm  "
                f"x[{lo[0]*1000:6.1f},{hi[0]*1000:6.1f}] "
                f"y[{lo[1]*1000:6.1f},{hi[1]*1000:6.1f}] "
                f"z[{lo[2]*1000:6.1f},{hi[2]*1000:6.1f}]")
        say(f"    TCP (midway between pads), in tool0: {np.round(tcp * 1000, 1)} mm")
        say(f"    opening axis (right - left):         {np.round(opening * 1000, 1)} mm")

    tcps = np.array(tcps)
    drift = float(np.abs(tcps - tcps[0]).max()) * 1000.0
    say("\n  --- verdict ---")
    say(f"    TCP varies by {drift:.2f} mm across joint configurations")
    if drift > 2.0:
        say("    NOT RIGID: the gripper is not moving with the wrist, so no fixed TCP exists. "
            "Fix the attachment before anything else.")
        return None
    tcp = tcps.mean(axis=0)
    dist = float(np.linalg.norm(tcp))
    axis = tcp / dist if dist > 1e-6 else np.array([0.0, 0.0, 1.0])
    say(f"    TCP is rigid at {np.round(tcp * 1000, 1)} mm in the tool0 frame "
        f"({dist * 1000:.1f} mm out)")
    say(f"    the approach direction in the tool0 frame is {np.round(axis, 4)}")
    nearest = int(np.argmax(np.abs(axis)))
    say(f"    -> dominant axis: tool0 {'XYZ'[nearest]}{'+' if axis[nearest] > 0 else '-'}")
    if abs(axis[2] - 1.0) < 0.02:
        say("       That is tool0 +Z, so the fingers point along the frame RMPflow drives: "
            "only the TCP offset needs correcting.")
        say(f"       set GRIPPER_TCP_OFFSET_M = {dist:.3f}")
    else:
        say("       That is NOT tool0 +Z. The gripper is mounted rotated relative to the frame "
            "RMPflow drives, so the commanded orientation has to be composed with that rotation "
            "as well as offset -- use GRIPPER_TCP_IN_TOOL0 below, which carries both.")
        say(f"       set GRIPPER_TCP_IN_TOOL0 = {np.round(tcp, 4).tolist()}")
    return tcp


def grip_to_tool0(target_pos, target_rotvec, tcp_offset, r_flange_to_tool0=None):
    """The corrected conversion: back off along the APPROACH (tool0 +Z) by the
    TCP offset, and command the requested orientation as-is.

    If the USD had disagreed with the URDF, r_flange_to_tool0 would be
    composed here; measure_frame_offsets says whether it must be."""
    r_grip = Rot.from_rotvec(np.asarray(target_rotvec, dtype=float))
    if r_flange_to_tool0 is not None:
        r_grip = r_grip * r_flange_to_tool0
    approach = r_grip.apply(np.array([0.0, 0.0, 1.0]))
    pos = np.asarray(target_pos, dtype=float) - tcp_offset * approach
    quat_xyzw = r_grip.as_quat()
    return pos, quat_xyzw[[3, 0, 1, 2]], approach


def measure_tracking(scene, tcp_offset):
    """Does RMPflow actually converge to a commanded grasp pose, and in how
    many ticks? Two candidate transforms are driven to the same grasp so the
    numbers are directly comparable:

      corrected : command the requested orientation, back off along tool0 +Z
      committed : the current code -- compose r_flange_to_tool0 and back off
                  along the flange's +Z

    The scripted expert's steps_per_segment (30) is also in question, so
    report the error at 30 / 60 / 120 / 200 ticks rather than only at the end.
    """
    say("\n=== RMPflow tracking of a downward grasp pose ===")
    grasp_point = np.array([0.45, 0.0, CUBE_Z])  # centre of the cube spawn area
    checkpoints = [30, 60, 120, 200]
    say(f"  commanding the fingertips to {np.round(grasp_point, 3)} pointing straight down, "
        f"TCP offset {tcp_offset:.3f} m")

    for label in ("corrected", "committed"):
        scene.reset()
        for _ in range(5):
            scene.world.step(render=False)

        r_flange_to_tool0 = scene.r_flange_to_tool0 if label == "committed" else None
        tool0_pos, tool0_quat_wxyz, _ = grip_to_tool0(
            grasp_point, DOWNWARD_ROTVEC, tcp_offset, r_flange_to_tool0)
        say(f"\n  [{label}] tool0 target pos {np.round(tool0_pos, 4)} "
            f"quat_wxyz {np.round(tool0_quat_wxyz, 4)}")

        errors = {}
        for tick in range(1, max(checkpoints) + 1):
            scene.rmpflow.set_end_effector_target(tool0_pos, tool0_quat_wxyz)
            scene.rmpflow.update_world()
            action = scene.articulation_policy.get_next_articulation_action(scene.physics_dt)
            scene.robot.apply_action(action)
            scene.world.step(render=False)
            if tick in checkpoints:
                q_now = np.asarray(scene.robot.get_joint_positions())[:6]
                ee_pos, ee_rot = scene.rmpflow.get_end_effector_pose(q_now)
                r_ee = as_rotation(ee_rot)
                # Where the fingertips ended up, under this candidate's own
                # idea of which axis they stick out along.
                axis = np.array([0.0, 0.0, 1.0])
                if label == "committed":
                    # the committed code offsets along the FLANGE's +Z
                    _, flange_quat = prim_world_pose(
                        scene.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
                    tip = np.asarray(ee_pos) + tcp_offset * Rot.from_quat(flange_quat).apply(axis)
                else:
                    tip = np.asarray(ee_pos) + tcp_offset * r_ee.apply(axis)
                pos_err = float(np.linalg.norm(tip - grasp_point)) * 1000.0
                rot_err = float(np.degrees(
                    (r_ee.inv() * Rot.from_rotvec(DOWNWARD_ROTVEC)).magnitude()))
                errors[tick] = (pos_err, rot_err)
                say(f"    after {tick:3d} ticks: fingertips {pos_err:7.1f} mm from target, "
                    f"approach {rot_err:6.2f} deg off")
        say(f"    -> {label}: {errors[checkpoints[-1]][0]:.1f} mm at "
            f"{checkpoints[-1]} ticks")

    say("\n  A grasp needs the fingertips within roughly half the cube (20mm). Read the")
    say("  smallest tick count that gets there off the rows above -- that is what")
    say("  scripted_pick_place.steps_per_segment has to be.")


def run(out_dir, do_tracking=True):
    os.makedirs(out_dir, exist_ok=True)
    _REPORT["path"] = os.path.join(out_dir, "frames_report.txt")
    open(_REPORT["path"], "w").close()
    say(f"writing this report to {_REPORT['path']} as well as stdout")

    scene = PickPlaceScene(with_gripper=True)
    say("scene built (with gripper); resetting...")
    scene.reset()

    offsets = measure_frame_offsets(scene)
    tcp = measure_gripper_tcp(scene)
    if do_tracking:
        # measure_tracking drives a scalar offset along tool0 +Z; feed it the
        # measured distance when the fingers really do point that way.
        offset = 0.12 if tcp is None else float(np.linalg.norm(tcp))
        measure_tracking(scene, offset)
    return offsets, tcp


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="camera_check")
    parser.add_argument("--skip_tracking", action="store_true",
                        help="geometry only -- skip the RMPflow tracking test")
    args = parser.parse_args()
    # Print any traceback BEFORE closing the app -- simulation_app.close()
    # runs Kit's fastShutdown, which kills the process before Python reports
    # an escaping exception, making a crash look like a clean exit.
    try:
        run(args.out_dir, do_tracking=not args.skip_tracking)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
