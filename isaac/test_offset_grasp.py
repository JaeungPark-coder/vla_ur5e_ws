"""Test a grasp using the offset measured by locate_fingers_2d.py, instead
of the "distance straight down tool0 Z" model that found zero contact at
every distance 0.06-0.28m. The overlap-query search found the closed
fingers roughly 90mm off in BOTH world X and Y (not centered under tool0 at
all) and only ~30-90mm below tool0's height, at the one arm configuration
reached for cube position [0.475, 0.159, 0.02]:

    tool0 commanded/reached ~= [0.476, 0.160, 0.143]
    fingers found near world ~= [0.565, 0.249, 0.110]
    => offset (fingers - tool0) ~= [+0.089, +0.090, -0.033]

This computes the tool0 command needed to place THAT SAME offset's fingers
at the cube instead (command = cube - offset), and checks whether the cube
actually lifts. If it does, this offset is real and needs to be converted
into a proper GRIPPER_TCP_IN_TOOL0-style constant (expressed in the tool0
frame, not world, so it transfers to other cube positions/arm configs)."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
SEED = 7
# Measured offset (fingers - tool0), in WORLD coordinates, at the one arm
# configuration this project has actually located the fingers in.
OFFSET_WORLD = np.array([0.089, 0.090, -0.033])


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene._rng = np.random.default_rng(SEED)
        scene.reset()
        cube0 = scene.cube_position.copy()
        target_finger_pos = cube0.copy()
        tool0_command = target_finger_pos - OFFSET_WORLD
        print(f"cube at {np.round(cube0, 4)}", flush=True)
        print(f"commanding tool0 to {np.round(tool0_command, 4)} so the measured offset "
              f"{OFFSET_WORLD.tolist()} lands the fingers on the cube", flush=True)

        # Approach from safely above first (same XY, generous height), THEN
        # descend to the computed command -- descending straight there from
        # the default pose risks sweeping through the cube on the way, the
        # same issue instrument_grasp.py's redundant-reset bug imitated
        # (this time for a real reason: a big single jump).
        above = tool0_command.copy()
        above[2] += 0.20
        drive(scene, above, 220, 0.0)
        drive(scene, tool0_command, 220, 0.0)
        drive(scene, tool0_command, 90, 1.0)
        lift = tool0_command.copy()
        lift[2] += 0.20
        drive(scene, lift, 220, 1.0)

        cube_now = scene.get_cube_position()
        rise = float(cube_now[2] - cube0[2])
        drift = float(np.linalg.norm(cube_now[:2] - cube0[:2]))
        print(f"cube now at {np.round(cube_now, 4)} -- rise={rise*1000:.1f}mm, "
              f"xy_drift={drift*1000:.1f}mm", flush=True)
        if rise * 1000 > 50:
            print("\nSUCCESS -- the offset model works. Convert OFFSET_WORLD into the tool0 "
                  "frame (it's already aligned since DOWNWARD_ROTVEC is what was commanded) "
                  "and set it as the project's real TCP constant.", flush=True)
        else:
            print("\nSTILL NO LIFT with the measured offset -- the offset likely is NOT a "
                  "world-frame constant (probably varies with which IK branch RMPflow lands in "
                  "for a given cube position/approach path), or the two locate_fingers_2d.py "
                  "grids disagreed enough (dx~100mm at z=110mm vs dx~50mm at z=130mm) that this "
                  "single-point estimate missed. Re-run locate_fingers_2d.py centered on THIS "
                  "cube position and command instead.", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
