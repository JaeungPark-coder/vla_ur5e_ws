"""What are finger_joint's REAL limits on this asset, and does
GRIPPER_CLOSED_POS=0.68 actually reach them?

isaac_sim_common.py's own comment flags GRIPPER_CLOSED_POS as an ADJUST
guess ("should be a firm-but-not-overdriven close"). If the real upper limit
is well past 0.68, commanding 0.68 may leave the fingers short of a full
close -- consistent with the TCP-distance sweep finding zero contact at
every distance tested: the fingers might simply never close far enough to
reach a 40mm cube regardless of standoff distance."""
import os
from isaacsim import SimulationApp
HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

import numpy as np  # noqa: E402
from pick_place_scene import PickPlaceScene  # noqa: E402
from isaac_sim_common import GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, GRIPPER_DRIVE_JOINT_NAME  # noqa: E402

try:
    scene = PickPlaceScene(with_gripper=True)
    scene.reset()
    robot = scene.robot
    dof_names = list(robot.dof_names)
    idx = dof_names.index(GRIPPER_DRIVE_JOINT_NAME)
    lower = robot.dof_properties["lower"][idx] if hasattr(robot, "dof_properties") else None
    print(f"dof_names: {dof_names}", flush=True)
    print(f"GRIPPER_DRIVE_JOINT_NAME={GRIPPER_DRIVE_JOINT_NAME!r} at index {idx}", flush=True)
    print(f"configured GRIPPER_OPEN_POS={GRIPPER_OPEN_POS}, GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS}", flush=True)

    # Try the articulation's own joint-limits API (varies by Isaac Sim version).
    try:
        limits = robot.get_joint_limits()
        print(f"get_joint_limits()[{idx}] = {np.asarray(limits)[idx]}", flush=True)
    except Exception as e:
        print(f"get_joint_limits() failed: {e}", flush=True)

    try:
        from pxr import UsdPhysics
        for jname in ["finger_joint", "left_inner_finger_joint", "right_inner_finger_joint",
                      "left_inner_finger_knuckle_joint", "right_inner_finger_knuckle_joint",
                      "right_outer_knuckle_joint"]:
            prim = scene.stage.GetPrimAtPath(f"/World/ur5e/Gripper/Robotiq_2F_85/{jname}")
            if not prim.IsValid():
                # joints commonly live under a "Joints" scope
                for cand in scene.stage.Traverse():
                    if cand.GetName() == jname:
                        prim = cand
                        break
            if prim.IsValid() and prim.HasAPI(UsdPhysics.DriveAPI, "angular"):
                drive = UsdPhysics.DriveAPI.Get(prim, "angular")
                print(f"  {jname}: target={drive.GetTargetPositionAttr().Get()}", flush=True)
            joint = UsdPhysics.RevoluteJoint(prim) if prim.IsValid() else None
            if joint:
                print(f"  {jname} @ {prim.GetPath()}: lower={joint.GetLowerLimitAttr().Get()} "
                      f"upper={joint.GetUpperLimitAttr().Get()}", flush=True)
    except Exception as e:
        import traceback
        print(f"USD joint-limit probe failed: {e}", flush=True)
        traceback.print_exc()

    # Command well past the configured "closed" value and see how far it actually goes --
    # if it stalls at the same joint reading as GRIPPER_CLOSED_POS, that's the real hard limit.
    from isaacsim.core.utils.types import ArticulationAction
    for target in (0.68, 0.8, 1.0, 1.2):
        for _ in range(150):
            robot.apply_action(ArticulationAction(joint_positions=np.array([target]),
                                                   joint_indices=np.array([idx])))
            scene.world.step(render=False)
        achieved = float(np.asarray(robot.get_joint_positions())[idx])
        print(f"  commanded {target:.2f} rad -> achieved {achieved:.4f} rad after 150 ticks", flush=True)
except BaseException:
    import traceback
    print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
finally:
    simulation_app.close()
