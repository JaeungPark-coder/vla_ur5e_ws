"""B-1 follow-up: the 2026-09-23 pivot_dwell_check run found a STATIC
(non-growing) per-axis grip-point error that differs by pose -- free space
(height 0.35m) reads dx=-4.7 dy=-3.2 dz=-5.7mm, the grasp pose (height
~0.04m) reads dx=+24.1 dy=+6.4 dz=+26.8mm. No growth over a 180-tick hold
rules out dynamics (gravity sag would keep growing); a height-dependent
static offset is exactly what a base-frame mismatch between RMPflow's
internal notion of the robot's root and where the robot prim actually sits
in the world would produce.

wrist_camera_setup (this file's own reset(), see the block computing
r_flange_to_tool0) already calls rmpflow.get_end_effector_pose(q0) and
compares it against the flange prim's REAL world pose -- but only checks
the ROTATION difference (r_flange_to_tool0). The POSITION half of that
same comparison was never looked at. This checks it directly: if RMPflow's
FK position and the prim's actual world position disagree, RMPflow's
internal frame does not line up with the world/USD frame everything else
in this project reads cube/grip positions in, and no amount of gain/
friction tuning can fix that -- the fix is a explicit
rmpflow.set_robot_base_pose() call, currently missing from setup_rmpflow()
(isaac_sim_common.py) entirely (grepped, zero call sites in production
code).
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import PickPlaceScene, TOOL_LINK_PRIM_PATH, prim_world_pose  # noqa: E402


def main():
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene.reset()

        q0 = np.asarray(scene.robot.get_joint_positions())[:6]
        fk_pos, fk_rot = scene.rmpflow.get_end_effector_pose(q0)
        fk_pos = np.asarray(fk_pos, dtype=float)

        prim_pos, _ = prim_world_pose(scene.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        prim_pos = np.asarray(prim_pos, dtype=float)

        delta = fk_pos - prim_pos
        print(f"RMPflow FK position (tool0/flange, robot's internal frame): {fk_pos}", flush=True)
        print(f"Actual flange prim world position (USD/world frame):        {prim_pos}", flush=True)
        print(f"delta (FK - actual world): dx={delta[0]*1000:+.1f}mm  "
              f"dy={delta[1]*1000:+.1f}mm  dz={delta[2]*1000:+.1f}mm  "
              f"|delta|={np.linalg.norm(delta)*1000:.1f}mm", flush=True)

        if np.linalg.norm(delta) > 0.002:
            print("\nVERDICT: RMPflow's FK does not agree with the prim's actual world "
                  "pose -- this is consistent with a missing/wrong robot base pose in "
                  "RMPflow's config (rmpflow.set_robot_base_pose() is never called anywhere "
                  "in this project). Every Cartesian target this project sends "
                  "(set_end_effector_target) is expressed in world/USD coordinates, so "
                  "RMPflow silently mis-executes all of them by this same delta.", flush=True)
        else:
            print("\nVERDICT: RMPflow's FK agrees with the prim's actual world pose to "
                  "within 2mm -- a base-frame mismatch is NOT the explanation for the "
                  "dwell-check offset; look elsewhere.", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
