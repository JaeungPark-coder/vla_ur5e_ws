"""2026-09-28 smoke-test of collect_demos.py (with the IsaacLab gripper-gain
fix wired in -- see collect_demos.py's GRIPPER_FINGER_* constants) found the
grasp itself now succeeds much more often (6/10 lifts, max_cube_z 0.158-
0.202m, all clearing LIFT_Z_THRESHOLD=0.08m comfortably) but EVERY SINGLE
lifted attempt still failed place_tolerance_m (0.03m): errors of 44, 93, 110,
138, 242, 392mm. 0/10 attempts completed the full task. This was previously
invisible because with the grasp itself failing ~100% of the time, "does the
placed position match" never had a chance to matter.

Three competing explanations, all raised in the 2026-09-25 report review, not
yet distinguished:

  H3 (revived): the softened grip (kp 20000->17, effort-limited to 1650 --
      needed to stop crushing the cube on contact, see README's 2026-09-28
      section) holds enough static force to clear LIFT_Z_THRESHOLD but not
      enough to survive the ACCELERATION during transport -- the cube slips
      partway through the "transport" segment. A 392mm miss is not "slightly
      off", it is closer to "dropped and it rolled/fell".
  RMPflow tracking error: the arm itself doesn't reach the commanded place
      waypoint accurately (precedent: 9/21 saw 887-1090mm residuals at
      spawn-range edges) -- the grasp would be fine, but the SETPOINT sent
      during transport/descend-to-target is already far from the intended
      target_position.
  Measurement artifact: place_error_m() reads get_cube_position() -- check
      whether that read happens before the cube has settled after release
      (rolling/bouncing), which would inflate "off" without either of the
      above being true. Cheapest to rule out first.

One log settles all three: track dz (grip point Z minus cube Z -- the same
quantity check_grasp_alignment.py already tracks through the close segment,
here extended all the way to release) AND commanded-vs-actual grip-point
position (the interpolated target_pos step_towards() is given each tick vs
scene.grip_point_world()) across the WHOLE episode, not just through lift
(check_close_lift_dynamics.py stops at end of lift; collect_demos.py's own
per-tick view doesn't exist -- it only logs the final place_error_m()).

  dz grows during "transport"/"descend2"     -> H3 (grip not holding under load)
  dz stays ~flat but tcp_err is large/grows  -> RMPflow tracking error
  both stay small, but final "off" is large  -> place_error_m() timing/read bug

    python3 check_place_dynamics.py --episodes 5 --seed 42
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, LIFT_Z_THRESHOLD, ROBOT_PRIM_PATH)
from scripted_pick_place import ScriptedPickPlace  # noqa: E402
from isaac_sim_common import (  # noqa: E402
    GRIPPER_DRIVE_JOINT_NAME, set_joint_max_force, zero_follower_joint_drives)

SEGMENT_NAMES = ["approach", "settle", "descend", "settle2", "close", "lift", "transport", "descend2", "settle3", "release", "retract"]
GRIPPER_FINGER_KP = 17.0
GRIPPER_FINGER_KD = 0.02
GRIPPER_FINGER_EFFORT_LIMIT = 1650.0


def say(line=""):
    print(line, flush=True)


def run(episodes, seed):
    scene = PickPlaceScene(with_gripper=True, finger_kp=GRIPPER_FINGER_KP, finger_kd=GRIPPER_FINGER_KD)
    robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
    set_joint_max_force(robot_prim, GRIPPER_FINGER_EFFORT_LIMIT, joint_names=(GRIPPER_DRIVE_JOINT_NAME,))
    follower_names = zero_follower_joint_drives(robot_prim)
    say(f"finger_kp={GRIPPER_FINGER_KP} finger_kd={GRIPPER_FINGER_KD} "
        f"effort_limit={GRIPPER_FINGER_EFFORT_LIMIT} follower_zeroed={follower_names}")

    scene._rng = np.random.default_rng(seed)
    say(f"cube-spawn RNG seeded with {seed}")

    for ep in range(1, episodes + 1):
        obs = scene.reset()
        policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
        waypoints = policy.waypoints
        segment_bounds = np.cumsum([n for _, _, n in waypoints])
        assert len(segment_bounds) == len(SEGMENT_NAMES), (len(segment_bounds), len(SEGMENT_NAMES))

        cube0 = scene.cube_position.copy()
        say(f"\n=== episode {ep}/{episodes}: cube spawned at {np.round(cube0, 4).tolist()}, "
            f"target at {np.round(PLACE_TARGET_POSITION, 4).tolist()} ===")
        say(f"{'tick':>5} {'seg':>10} {'grip_tgt':>8} {'grip_act':>8} "
            f"{'dz_mm':>8} {'tcp_err_mm':>10} {'cube_z_mm':>9}")

        max_cube_z = -np.inf
        seg_idx = 0
        dz_by_segment = {}  # seg_name -> list of dz (mm)
        tcp_err_by_segment = {}
        for tick, (target_pos, target_rotvec, target_gripper) in enumerate(policy.generate_frames(), start=1):
            scene.step_towards(target_pos, target_rotvec, target_gripper)
            while seg_idx < len(segment_bounds) - 1 and tick > segment_bounds[seg_idx]:
                seg_idx += 1
            seg_name = SEGMENT_NAMES[seg_idx]

            cube_pos = scene.get_cube_position()
            grip_point = scene.grip_point_world()
            actual_gripper = scene.gripper.get_normalized_position()
            dz_mm = 1000.0 * (grip_point[2] - cube_pos[2])
            tcp_err_mm = 1000.0 * float(np.linalg.norm(np.asarray(target_pos, dtype=float) - grip_point))
            max_cube_z = max(max_cube_z, float(cube_pos[2]))

            dz_by_segment.setdefault(seg_name, []).append(dz_mm)
            tcp_err_by_segment.setdefault(seg_name, []).append(tcp_err_mm)

            if tick % 30 == 0 or tick == segment_bounds[seg_idx] or tick == 1:
                say(f"{tick:>5} {seg_name:>10} {target_gripper:>8.3f} {actual_gripper:>8.3f} "
                    f"{dz_mm:>8.1f} {tcp_err_mm:>10.1f} {cube_pos[2] * 1000:>9.1f}")

        final_cube_pos = scene.get_cube_position()
        place_error = scene.place_error_m()
        lifted = scene.grasp_succeeded(max_cube_z)

        say(f"\n  --- episode {ep} verdict ---")
        say(f"  max_cube_z={max_cube_z:.4f}m (threshold {LIFT_Z_THRESHOLD}m, lifted={lifted})")
        say(f"  place_error_m()={place_error * 1000:.1f}mm "
            f"(final cube xy={np.round(final_cube_pos[:2], 4).tolist()}, "
            f"target xy={np.round(PLACE_TARGET_POSITION[:2], 4).tolist()})")
        for seg in SEGMENT_NAMES:
            if seg not in dz_by_segment:
                continue
            dz_vals = dz_by_segment[seg]
            tcp_vals = tcp_err_by_segment[seg]
            say(f"  seg={seg:>10}: dz first/last={dz_vals[0]:+7.1f}/{dz_vals[-1]:+7.1f}mm "
                f"(drift {dz_vals[-1] - dz_vals[0]:+6.1f}mm) | "
                f"tcp_err mean/max={np.mean(tcp_vals):6.1f}/{np.max(tcp_vals):6.1f}mm")

        # Cheap classification against the three hypotheses.
        transport_segs = [s for s in ("transport", "descend2") if s in dz_by_segment]
        transport_dz_drift = max(
            (dz_by_segment[s][-1] - dz_by_segment[s][0] for s in transport_segs), default=0.0,
            key=abs)
        transport_tcp_err = max((np.max(tcp_err_by_segment[s]) for s in transport_segs), default=0.0)
        if abs(transport_dz_drift) > 15.0:
            verdict = f"H3 (grip slipping under load): dz drifted {transport_dz_drift:+.1f}mm during transport/descend2"
        elif transport_tcp_err > 15.0:
            verdict = f"RMPflow tracking error: commanded-vs-actual grip point off by up to {transport_tcp_err:.1f}mm during transport/descend2"
        elif place_error > 0.03:
            verdict = "neither dz drift nor tcp tracking error explains it -- check place_error_m() timing/read (measurement artifact)"
        else:
            verdict = "placed within tolerance"
        say(f"  verdict: {verdict}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run(args.episodes, args.seed)
    except BaseException:
        import traceback
        say("\n=== FAILED ===\n" + traceback.format_exc())
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
