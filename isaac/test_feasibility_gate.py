"""Sanity-checks feasibility_gate.FeasibilityGate against two known cases:
  - the pose CONFIRMED to make RMPflow wind up (should be REJECTED)
  - the pose CONFIRMED to converge cleanly all session (should PASS)
"""
import os
import numpy as np
from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pick_place_scene import ROBOT_PRIM_PATH, PickPlaceScene  # noqa: E402
from feasibility_gate import FeasibilityGate  # noqa: E402

DOWN_QUAT_WXYZ = [0.0, 0.0, 1.0, 0.0]

try:
    scene = PickPlaceScene(with_gripper=True)
    scene.reset()

    gate = FeasibilityGate(ROBOT_PRIM_PATH)
    print("gate built OK, joint_names:", gate.joint_names, flush=True)

    cases = {
        "CONFIRMED BAD (wound up, elbow->162deg)": [0.4618, 0.0636, 0.0222],
        "CONFIRMED GOOD (0.9mm tracking, clean)": [0.476, 0.1595, 0.1427],
        "typical cube-grasp height in this task": [0.45, 0.0, 0.14],
    }
    for label, pos in cases.items():
        result = gate.check(pos, DOWN_QUAT_WXYZ)
        print(f"\n[{label}] target={pos}", flush=True)
        print(f"  ok={result['ok']}  reason={result['reason']}", flush=True)
        if result["manipulability"] is not None:
            print(f"  manipulability={result['manipulability']:.4f}  "
                  f"condition_number={result['condition_number']:.1f}", flush=True)
except BaseException:
    import traceback
    print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
finally:
    simulation_app.close()
