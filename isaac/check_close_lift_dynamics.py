"""Why does the grasp-lift fail even after B-1's alignment offset was fixed
(2026-09-23, --stiffness-scale 3 --damping-scale 1.7320508 collapsed the
static dwell offset 48-67x, but check_grasp_alignment.py at the SAME
settings still landed 5/5 lifted=no)? Static-pose tracking accuracy is not
the bottleneck; something specific to the CLOSE-then-LIFT dynamics is.

Three failure patterns are distinguishable from the cube's own trajectory,
logged every tick through close AND lift (check_grasp_alignment.py's own
per-tick log stops at the end of close and never looks at lift):

  cube_z never rises              -- the grip never took hold at all
                                      (friction/grip force too low, or the
                                      fingers never actually contacted it)
  cube_z rises then falls back    -- it WAS held, then slipped during lift
                                      (friction marginal, or lift accel
                                      exceeds what static+dynamic friction
                                      at the measured normal force sustains)
  cube stays near table height
  but drifts sideways (xy)        -- the previously-documented "off-centre
                                      shove" pattern, still present during
                                      the actual close impact even though
                                      pivot_dwell_check's STATIC hold test
                                      (nothing in the gripper) no longer
                                      shows a steady-state offset

Also reports whether the gripper actually reached its commanded closed
position (1.0) or stalled short of it -- stalling means real contact;
reaching 1.0 cleanly means the fingers closed past/around the cube without
ever loading against it (a sizing/timing problem, not a friction one), and
distinguishes "friction is too low once gripped" from "it was never really
gripped to begin with" before reaching for a friction-material fix.

Also prints the grip pads' currently-bound friction coefficients
(read-only -- GRIP_MATERIAL_PRIM_PATH, shared by both the pads and the
cube per bind_grip_friction_material's own comment) so that's confirmed
rather than assumed.

    python3 check_close_lift_dynamics.py --stiffness-scale 3 --damping-scale 1.7320508
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pxr import UsdPhysics  # noqa: E402

from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION, LIFT_Z_THRESHOLD  # noqa: E402
from scripted_pick_place import ScriptedPickPlace  # noqa: E402
from isaac_sim_common import (  # noqa: E402
    ROBOT_PRIM_PATH, ARM_JOINT_NAMES, get_joint_drive_gains, set_joint_drive_gains,
    GRIP_MATERIAL_PRIM_PATH)


def say(line=""):
    print(line, flush=True)


def run(episodes, seed, stiffness_scale, damping_scale):
    scene = PickPlaceScene(with_gripper=True)

    material_prim = scene.stage.GetPrimAtPath(GRIP_MATERIAL_PRIM_PATH)
    material_api = UsdPhysics.MaterialAPI(material_prim)
    say(f"grip pad / cube friction material ({GRIP_MATERIAL_PRIM_PATH}, shared prim): "
        f"static={material_api.GetStaticFrictionAttr().Get():g} "
        f"dynamic={material_api.GetDynamicFrictionAttr().Get():g}")

    scene._rng = np.random.default_rng(seed)
    say(f"cube-spawn RNG seeded with {seed}")

    if stiffness_scale != 1.0 or damping_scale != 1.0:
        robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        before = get_joint_drive_gains(robot_prim, joint_names=ARM_JOINT_NAMES)
        for name, gains in before.items():
            if gains is None:
                continue
            stiffness, damping = gains
            set_joint_drive_gains(
                robot_prim, joint_names=(name,),
                stiffness=(stiffness * stiffness_scale) if stiffness is not None else None,
                damping=(damping * damping_scale) if damping is not None else None)
        say(f"scaled all arm joints x{stiffness_scale:g} stiffness, x{damping_scale:g} damping")

    for ep in range(1, episodes + 1):
        obs = scene.reset()
        policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
        waypoints = policy.waypoints
        # First 6 waypoints = approach, settle, descend, settle2, close, lift
        # -- see scripted_pick_place.py's segments list. Stop right after
        # lift; transport/place aren't relevant to why the lift itself fails.
        n_through_lift = sum(n for _, _, n in waypoints[:6])
        segment_bounds = np.cumsum([n for _, _, n in waypoints[:6]])
        segment_names = ["approach", "settle", "descend", "settle2", "close", "lift"]

        cube0 = scene.cube_position.copy()
        say(f"\n=== episode {ep}/{episodes}: cube spawned at "
            f"{np.round(cube0, 4).tolist()} ===")
        say(f"{'tick':>5} {'seg':>8} {'grip_tgt':>8} {'grip_act':>8} "
            f"{'cube_x_mm':>9} {'cube_y_mm':>9} {'cube_z_mm':>9} {'cube_vz_mms':>11}")

        frames = list(policy.generate_frames())
        seg_idx = 0
        max_cube_z = -np.inf
        close_end_gripper_act = None
        lift_start_xy = None
        for tick, (target_pos, target_rotvec, target_gripper) in enumerate(frames[:n_through_lift], start=1):
            scene.step_towards(target_pos, target_rotvec, target_gripper)
            while seg_idx < len(segment_bounds) - 1 and tick > segment_bounds[seg_idx]:
                seg_idx += 1
            seg_name = segment_names[seg_idx]

            cube_pos = scene.get_cube_position()
            cube_vel = scene.cube_rigid_prim.get_linear_velocity()
            actual_gripper = scene.gripper.get_normalized_position()
            max_cube_z = max(max_cube_z, float(cube_pos[2]))

            if seg_name == "close" and tick == segment_bounds[seg_idx]:
                close_end_gripper_act = actual_gripper
            if seg_name == "lift" and lift_start_xy is None:
                lift_start_xy = cube_pos[:2].copy()

            if seg_name in ("close", "lift") or tick % 20 == 0 or tick == n_through_lift:
                say(f"{tick:>5} {seg_name:>8} {target_gripper:>8.3f} {actual_gripper:>8.3f} "
                    f"{(cube_pos[0] - cube0[0]) * 1000:>9.1f} "
                    f"{(cube_pos[1] - cube0[1]) * 1000:>9.1f} "
                    f"{cube_pos[2] * 1000:>9.1f} "
                    f"{cube_vel[2] * 1000:>11.1f}")

        final_cube_z = float(scene.get_cube_position()[2])
        final_xy_drift_mm = 1000.0 * float(np.linalg.norm(
            scene.get_cube_position()[:2] - (lift_start_xy if lift_start_xy is not None else cube0[:2])))
        lifted = max_cube_z > LIFT_Z_THRESHOLD

        say(f"\n  --- episode {ep} verdict ---")
        say(f"  end-of-close gripper closure: {close_end_gripper_act:.3f} "
            f"({'reached target cleanly -- no contact resistance felt' if close_end_gripper_act is not None and close_end_gripper_act > 0.97 else 'stalled short of 1.0 -- real contact resistance'})")
        say(f"  max_cube_z during lift: {max_cube_z:.4f}m (threshold {LIFT_Z_THRESHOLD}m, lifted={lifted})")
        say(f"  cube_z at end of lift segment: {final_cube_z:.4f}m")
        say(f"  cube xy drift during lift (from lift-start position): {final_xy_drift_mm:.1f}mm")
        if max_cube_z <= scene.cube_position[2] + 0.005:
            pattern = "NEVER RISES -- grip never took hold (friction/grip force)"
        elif final_cube_z < max_cube_z - 0.01:
            pattern = "RISES THEN FALLS -- held briefly, slipped during lift"
        elif final_xy_drift_mm > 15.0:
            pattern = "SHOVED SIDEWAYS -- off-centre close impact, not a lift/friction failure"
        else:
            pattern = "held and rose without falling (check threshold/timing if still lifted=False)"
        say(f"  pattern: {pattern}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stiffness-scale", type=float, default=1.0)
    parser.add_argument("--damping-scale", type=float, default=1.0)
    args = parser.parse_args()
    try:
        run(args.episodes, args.seed, args.stiffness_scale, args.damping_scale)
    except BaseException:
        import traceback
        say("\n=== FAILED ===\n" + traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
