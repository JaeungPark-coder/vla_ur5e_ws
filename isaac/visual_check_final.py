"""Render the actual gripper-vs-cube relationship at the measured offset, up
close, from multiple angles -- numeric overlap queries have given
inconsistent/confusing results (offset varies 90deg+ direction between
configs, near-zero net vertical reach past tool0), so see it directly."""
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

from pxr import UsdGeom, Gf  # noqa: E402
from pick_place_scene import PickPlaceScene  # noqa: E402

DOWN_QUAT_WXYZ = np.array([0.0, 0.0, 1.0, 0.0])
SEED = 7
OFFSET_WORLD = np.array([0.0132, 0.0953, -0.0022])


def drive(scene, tool0_pos, ticks, gripper):
    for _ in range(ticks):
        scene.rmpflow.set_end_effector_target(np.asarray(tool0_pos, dtype=float), DOWN_QUAT_WXYZ)
        scene.rmpflow.update_world()
        scene.robot.apply_action(
            scene.articulation_policy.get_next_articulation_action(scene.physics_dt))
        scene.gripper.set_target(gripper)
        scene.world.step(render=False)


def add_closeup_camera(scene, eye, target, path="/World/closeup_cam"):
    stage = scene.stage
    cam = UsdGeom.Camera.Define(stage, path)
    cam.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in eye]))
    forward = np.asarray(target, dtype=float) - np.asarray(eye, dtype=float)
    forward /= np.linalg.norm(forward)
    z_axis = -forward
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(up, z_axis))) > 0.99:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis); x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(np.column_stack((x_axis, y_axis, z_axis))).as_quat()
    cam.AddOrientOp().Set(Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2])))
    cam.CreateClippingRangeAttr().Set(Gf.Vec2f(0.01, 100.0))
    cam.CreateFocalLengthAttr(18.0)
    cam.CreateHorizontalApertureAttr(20.0)
    import omni.replicator.core as rep
    rp = rep.create.render_product(path, (512, 512))
    annot = rep.AnnotatorRegistry.get_annotator("rgb")
    annot.attach([rp])
    return annot


def main():
    from PIL import Image
    try:
        scene = PickPlaceScene(with_gripper=True)
        scene._rng = np.random.default_rng(SEED)
        scene.reset()
        cube0 = scene.cube_position.copy()
        tool0_command = cube0 - OFFSET_WORLD
        above = tool0_command.copy(); above[2] += 0.15
        drive(scene, above, 220, 0.0)
        drive(scene, tool0_command, 220, 0.0)
        drive(scene, tool0_command, 90, 1.0)

        out_dir = "camera_check"
        os.makedirs(out_dir, exist_ok=True)
        views = {
            "side_x": (np.array([cube0[0] + 0.35, cube0[1], cube0[2] + 0.15]), cube0),
            "side_y": (np.array([cube0[0], cube0[1] + 0.35, cube0[2] + 0.15]), cube0),
            "top": (np.array([cube0[0] + 0.01, cube0[1], cube0[2] + 0.35]), cube0),
        }
        for name, (eye, target) in views.items():
            annot = add_closeup_camera(scene, eye, target, path=f"/World/cam_{name}")
            for _ in range(3):
                scene.world.render()
            frame = np.asarray(annot.get_data())
            if frame.size:
                Image.fromarray(frame[..., :3].astype(np.uint8)).save(
                    os.path.join(out_dir, f"closeup_{name}.png"))
                print(f"wrote closeup_{name}.png", flush=True)
            else:
                print(f"{name}: empty frame", flush=True)
    except BaseException:
        import traceback
        print("\n=== FAILED ===\n" + traceback.format_exc(), flush=True)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
