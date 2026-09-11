"""Opens Isaac Sim's GUI (not headless) with the arm driven to the exact
config that's been giving inconsistent/confusing offset measurements, then
stays open indefinitely so a human can freely rotate/zoom the viewport and
look at the gripper-vs-cube relationship directly -- every numeric method
tried (USD pose reads, physics pose reads, overlap queries from multiple
angles) has given a different, hard-to-reconcile answer for where the
fingers actually are relative to tool0, so seeing it is the next step.

Run this, then in the Isaac Sim window: middle-mouse-drag to pan, right-drag
(or alt+left-drag) to orbit, scroll to zoom. Look at:
  1. Does the gripper hang straight down from the wrist, or off to a side?
  2. When closed, do the finger pads visibly converge toward the cube, or
     toward empty space beside/above it?
  3. Is the whole "Robotiq_2F_85" mount perhaps rotated 90 degrees relative
     to how the arm's own wrist is oriented?

    ISAAC_PICK_PLACE_HEADLESS=0 python3 gui_inspect_gripper.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

# Force GUI regardless of the usual env var default (this script's whole
# point is to NOT run headless).
simulation_app = SimulationApp({"headless": False})

from pick_place_scene import CUBE_Z, PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
DISTANCE = 0.12
SEED = 7
# The best near-miss offset found so far (real contact, ~12mm drift, no
# clean lift) -- park the gripper CLOSED at this commanded position so it's
# immediately visible where things stand.
OFFSET_WORLD = np.array([0.0132, 0.0953, -0.0022])


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=True)


def main():
    scene = PickPlaceScene(with_gripper=True)
    scene._rng = np.random.default_rng(SEED)
    scene.reset()
    cube0 = scene.cube_position.copy()
    tool0_command = cube0 - OFFSET_WORLD
    print(f"cube at {np.round(cube0, 4)}, parking gripper (closed) near "
          f"{np.round(tool0_command, 4)}", flush=True)

    above = tool0_command.copy()
    above[2] += 0.15
    drive(scene, above, 220, 0.0)
    drive(scene, tool0_command, 220, 0.0)
    drive(scene, tool0_command, 90, 1.0)

    print("\nParked. Isaac Sim window is open -- orbit/pan/zoom to inspect the "
          "gripper-vs-cube relationship. This process will keep running (and the "
          "physics loop will keep the gripper holding this pose) until you close "
          "the window or kill the process.", flush=True)

    # Keep the app alive and responsive indefinitely.
    while simulation_app.is_running():
        scene.rmpflow.set_end_effector_target(tool0_command, DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(1.0)
        scene.world.step(render=True)

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
        simulation_app.close()
