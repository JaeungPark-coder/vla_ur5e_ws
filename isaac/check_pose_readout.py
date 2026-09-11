"""Which way of reading a rigid body's pose actually tracks it?

Two independent measurements in this project turned out to be reading numbers
that never change: every gripper link reports the wrist origin at every arm
configuration, and the cube reports its spawn height after being dropped from
0.3 m. Both were read with UsdGeom.ComputeLocalToWorldTransform via
isaac_sim_common.prim_world_pose. Anything scored against those numbers --
grasp_succeeded, place_error_m, the reach check in check_cameras -- has been
blind.

This drops a cube and prints its height as reported by each candidate
readout, so the one that follows physics can be adopted deliberately instead
of guessed at again.

    python3 check_pose_readout.py
"""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
import carb  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.utils.stage import get_current_stage  # noqa: E402

from isaac_sim_common import add_cube, prim_world_pose  # noqa: E402

CUBE_PATH = "/World/test_cube"
DROP_FROM = 0.30


def main():
    try:
        settings = carb.settings.get_settings()
        for key in ("/physics/updateToUsd", "/physics/updateParticlesToUsd",
                    "/app/usdrt/scene_delegate/enabled", "/physics/fabricUpdateTransformations"):
            print(f"setting {key} = {settings.get(key)}", flush=True)

        world = World(stage_units_in_meters=1.0)
        world.scene.add_default_ground_plane()
        stage = get_current_stage()
        add_cube(stage, CUBE_PATH, np.array([0.45, 0.0, DROP_FROM]))

        # Candidate readouts.
        from isaacsim.core.prims import SingleRigidPrim
        rigid = SingleRigidPrim(CUBE_PATH, name="test_cube")
        world.scene.add(rigid)
        world.reset()

        prim = stage.GetPrimAtPath(CUBE_PATH)
        print(f"\ndropping a cube from z={DROP_FROM}", flush=True)
        print(f"{'step':>5}  {'usd_prim':>9}  {'rigid_prim':>10}", flush=True)
        for step in range(0, 121):
            if step % 20 == 0:
                usd_z = float(prim_world_pose(prim)[0][2])
                try:
                    rigid_z = float(np.asarray(rigid.get_world_pose()[0])[2])
                except Exception as exc:
                    rigid_z = float("nan")
                    if step == 0:
                        print(f"  (rigid read failed: {type(exc).__name__}: {exc})", flush=True)
                print(f"{step:5d}  {usd_z:9.4f}  {rigid_z:10.4f}", flush=True)
            world.step(render=False)

        usd_z = float(prim_world_pose(prim)[0][2])
        rigid_z = float(np.asarray(rigid.get_world_pose()[0])[2])
        print("\n--- verdict ---", flush=True)
        print(f"  a cube dropped from {DROP_FROM} m should come to rest near 0.02 m", flush=True)
        print(f"  UsdGeom prim read : {usd_z:.4f} m  "
              f"{'TRACKS' if abs(usd_z - DROP_FROM) > 0.05 else 'STALE'}", flush=True)
        print(f"  SingleRigidPrim   : {rigid_z:.4f} m  "
              f"{'TRACKS' if abs(rigid_z - DROP_FROM) > 0.05 else 'STALE'}", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
