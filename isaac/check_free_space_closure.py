"""Restores (and extends) 2026-09-19's original free-space closure
measurement -- the one that found finger_joint "converging" to 0.374 rad
(47% of GRIPPER_CLOSED_POS) at its as-shipped kp=171.89/kd=0.0115, which
motivated raising it to kp=20000. That original diagnostic
(diag_gripper_gains_mimic.py) was never committed and is not reproducible
from this repo (see README's 2026-09-25 write-up, pattern P1's last
checklist item).

Two independently-researched documents (2026-09-24/25 sessions) argue the
47% reading was never about finger_joint's own drive being too weak: the
5 mimic-constrained follower/knuckle joints keep their as-shipped
171.89/0.0115 drive forever (GripperController.set_target only ever
commands finger_joint's own index -- confirmed by re-reading the code),
so a live PD drive on those joints keeps pulling toward ITS OWN stale
target and opposes the mimic constraint dragging it toward finger_joint's
commanded position. Raising finger_joint's kp to 20000 didn't remove that
opposition, it overpowered it -- and overpowers it just as hard once
real contact starts, which the 2026-09-23 dz-vs-closure% data blames for
the squeeze-shove pattern. IsaacLab's own reference Robotiq config
(FRANKA_ROBOTIQ_GRIPPER_CFG, fetched and confirmed against the real
github.com/isaac-sim/IsaacLab source, not taken on faith) sets exactly
these follower joints' PD to zero, with the comment "set PD to zero for
passive joints in close-loop gripper", and drives finger_joint itself
with kp=17/kd=0.02/effort_limit=1650 -- three orders of magnitude softer
than this project's kp=20000 with no effort limit at all.

IMPORTANT gain-reading gotcha this file's first version got wrong:
finger_joint's gain is NOT set through UsdPhysics.DriveAPI the way the arm
joints and the follower joints are (see set_joint_drive_gains) -- it goes
through SingleArticulation.get_articulation_controller().set_gains() (see
GripperController._fix_drive_gains), a completely separate "live
controller" gain path. get_joint_drive_gains (the raw-USD reader) reads a
DIFFERENT number for finger_joint than what's actually driving the
physics -- it is only correct for the follower joints, which really are
set via UsdPhysics.DriveAPI. This file reads finger_joint's real live gain
via the same controller.get_gains() call _fix_drive_gains itself uses.

Runs each of the 4 cases as its OWN process (not 4x PickPlaceScene() in
one Kit session -- that double-defines camera xformOps on the second
construction and throws). Each case's finger_kp/finger_kd is passed
EXPLICITLY -- finger_kp=None does not mean "true as-shipped 171.89", it
means "this project's already-fixed 20000 default" (see
GripperController._fix_drive_gains), so testing the real pre-fix baseline
needs the literal 171.89/0.0115 passed in.

    python3 check_free_space_closure.py --case 0   # as-shipped kp=171.89/kd=0.0115, follower as-shipped
    python3 check_free_space_closure.py --case 1   # as-shipped kp=171.89/kd=0.0115, follower ZEROED
    python3 check_free_space_closure.py --case 2   # current kp=20000/kd=500, follower as-shipped
    python3 check_free_space_closure.py --case 3   # IsaacLab kp=17/kd=0.02/effort=1650, follower ZEROED
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, ROBOT_PRIM_PATH  # noqa: E402
from isaac_sim_common import (  # noqa: E402
    GRIPPER_DRIVE_JOINT_NAME, GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS,
    get_joint_drive_gains, set_joint_max_force, resolve_gripper_follower_joint_names,
    zero_follower_joint_drives)

CLOSE_TICKS = 90  # matches scripted_pick_place.py's steps_per_segment // 2 for the close segment

CASES = [
    ("as-shipped kp=171.89/kd=0.0115, follower as-shipped",
     dict(finger_kp=171.89, finger_kd=0.0115, finger_effort_limit=None, zero_follower_pd=False)),
    ("as-shipped kp=171.89/kd=0.0115, follower ZEROED",
     dict(finger_kp=171.89, finger_kd=0.0115, finger_effort_limit=None, zero_follower_pd=True)),
    ("current kp=20000/kd=500, follower as-shipped",
     dict(finger_kp=20000.0, finger_kd=500.0, finger_effort_limit=None, zero_follower_pd=False)),
    ("IsaacLab kp=17/kd=0.02/effort=1650, follower ZEROED",
     dict(finger_kp=17.0, finger_kd=0.02, finger_effort_limit=1650.0, zero_follower_pd=True)),
]


def say(line=""):
    print(line, flush=True)


def read_finger_gains(scene):
    dof_names = list(scene.robot.dof_names)
    finger_idx = dof_names.index(GRIPPER_DRIVE_JOINT_NAME)
    controller = scene.robot.get_articulation_controller()
    kps, kds = controller.get_gains()
    return (float(np.asarray(kps)[finger_idx]), float(np.asarray(kds)[finger_idx]))


def run_one(finger_kp, finger_kd, finger_effort_limit, zero_follower_pd):
    scene = PickPlaceScene(with_gripper=True, finger_kp=finger_kp, finger_kd=finger_kd)
    scene.reset()  # this alone already calls gripper.reapply_drive_gains() internally
    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)

    # Read right after reset(), before touching anything else, so this
    # reflects ONLY what reapply_drive_gains applied -- decouples from any
    # possible interaction with the effort-limit/follower-zeroing calls
    # below (case 3 in v1 of this script read back ~171.89 instead of the
    # requested 17.0 here; still investigating why, but confirmed it is NOT
    # this read timing -- see the second read after the other calls too).
    gains_immediately_after_reset = read_finger_gains(scene)

    # Force finger_joint's controller-gain fix to actually apply now
    # (normally lazy, on first gripper command via
    # GripperController._resolve_joint_index) so the effort-limit override
    # below lands after it, not before.
    scene.gripper.set_target(0.0)
    scene.world.step(render=False)

    if finger_effort_limit is not None:
        set_joint_max_force(robot_prim, finger_effort_limit, joint_names=(GRIPPER_DRIVE_JOINT_NAME,))
        # BUG, confirmed live: UsdPhysics.DriveAPI.Apply() on finger_joint's
        # prim (inside set_joint_max_force) reverts its LIVE controller
        # stiffness/damping back to the raw USD schema default (171.89/
        # 0.0115) -- touching the raw schema on a joint whose gain is
        # actually driven through ArticulationController.set_gains()
        # clobbers that separate live-view state. Re-apply immediately
        # after to restore the requested finger_kp/finger_kd.
        # check_grasp_alignment.py's own call order (effort-limit/follower
        # block runs BEFORE its per-episode scene.reset(), which calls
        # reapply_drive_gains() last) happens to dodge this; this script's
        # original ordering did not.
        scene.gripper.reapply_drive_gains()
    follower_names = []
    if zero_follower_pd:
        follower_names = zero_follower_joint_drives(robot_prim)

    finger_live_gains = read_finger_gains(scene)
    follower_gains_before = get_joint_drive_gains(
        robot_prim, joint_names=tuple(resolve_gripper_follower_joint_names(robot_prim)))

    for _ in range(CLOSE_TICKS):
        scene.gripper.set_target(1.0)
        scene.world.step(render=False)

    final_frac = scene.gripper.get_normalized_position()
    final_rad = GRIPPER_OPEN_POS + final_frac * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
    return {
        "gains_immediately_after_reset": gains_immediately_after_reset,
        "finger_live_gains": finger_live_gains,
        "follower_names_zeroed": follower_names,
        "follower_gains_before_zeroing": follower_gains_before,
        "final_closure_frac": final_frac,
        "final_closure_rad": final_rad,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--case", type=int, required=True, choices=range(len(CASES)))
    args = parser.parse_args()
    try:
        label, kwargs = CASES[args.case]
        say(f"\n=== case {args.case}: {label} ===")
        result = run_one(**kwargs)
        say(f"  finger_joint gains immediately after scene.reset(): {result['gains_immediately_after_reset']}")
        say(f"  finger_joint LIVE controller gains (kp, kd): {result['finger_live_gains']}")
        say(f"  follower joints zeroed: {result['follower_names_zeroed'] or '(none -- not requested or none found)'}")
        say(f"  follower gains before zeroing (raw USD DriveAPI): {result['follower_gains_before_zeroing']}")
        say(f"RESULT case={args.case} closure_frac={result['final_closure_frac']:.4f} "
            f"closure_rad={result['final_closure_rad']:.4f}")
    except BaseException:
        import traceback
        say("\n=== FAILED ===\n" + traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
