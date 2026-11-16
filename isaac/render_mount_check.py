"""Visual check of the gripper mount (README "BLOCKER: the simulated gripper is mounted 90 deg off the wrist_3 axis").
At a demonstrated grasp joint vector, renders the base camera and an extra side camera with wrist_3 at its value and at
+90 / +180 deg, nothing else changed, and logs flange position, flange-z direction and the fingertip (grip point).
A gripper coaxial with wrist_3 keeps its fingertips in place while wrist_3 turns; one mounted 90 deg off swings them on a
12 cm radius.  Output: <out>/mount_check.png and mount_check.json.
The verdict uses the MEASURED fingertip -- the midpoint of the two inner-finger link origins read from the stage (on the
gripper's centre line by symmetry) -- not grip_point_world, which is sim_profile's formula and would pass v5 by construction
(2026-11-16 note, written before the v5 render).
    GPU=0 scripts/isaac_py.sh render_mount_check.py --out eval_results/mount_check
"""
import argparse
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--q", type=float, nargs=6, default=[-0.209, -0.956, 1.646, -0.677, -1.764, 3.173])  # strat_k25_v3_3k centre trial 1 close
ap.add_argument("--settle", type=int, default=90)
args = ap.parse_args()
sys.argv = [sys.argv[0]]

import check_action_replay as c  # noqa: E402  (starts the SimulationApp)

try:
    from PIL import Image, ImageDraw
    from scipy.spatial.transform import Rotation as Rot
    from pick_place_scene import TOOL_LINK_PRIM_PATH, ROBOT_PRIM_PATH, prim_world_pose
    import sim_profile
    print(sim_profile.describe(), flush=True)
    os.makedirs(args.out, exist_ok=True)
    scene = c.build_scene()
    scene.reset()
    # extra camera looking at the wrist from the side (+Y), so the mount is visible whatever the base camera sees
    from isaacsim.sensors.camera import Camera
    side = Camera(prim_path="/World/mount_check_cam", position=np.array([0.45, 0.55, 0.25]), resolution=(512, 512),
                  orientation=None)
    side.initialize()
    from isaacsim.core.utils.viewports import set_camera_view  # noqa: F401
    from pxr import Gf, UsdGeom
    cam_prim = scene.stage.GetPrimAtPath("/World/mount_check_cam")
    UsdGeom.XformCommonAPI(cam_prim)
    look = Gf.Matrix4d().SetLookAt(Gf.Vec3d(0.45, 0.55, 0.25), Gf.Vec3d(0.42, 0.0, 0.12), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(cam_prim).ClearXformOpOrder()
    UsdGeom.Xformable(cam_prim).AddTransformOp().Set(look)
    q0 = np.asarray(args.q, float)
    rows, info = [], []
    for dq in (0.0, np.pi / 2, np.pi):
        q = q0.copy()
        q[5] += dq
        scene.robot.set_joint_positions(q, joint_indices=np.arange(6))
        for _ in range(args.settle):
            scene.apply_joint_targets(q, 0.0)
        pos, quat = prim_world_pose(scene.stage.GetPrimAtPath(TOOL_LINK_PRIM_PATH))
        grip = scene.grip_point_world()
        measured_tip = None
        try:   # never lose the render to the measurement
            fingers = []
            for side_name in ("left_inner_finger", "right_inner_finger"):
                hits = [p for p in scene.stage.Traverse()
                        if p.GetName() == side_name and str(p.GetPath()).startswith(f"{ROBOT_PRIM_PATH}/")]
                fingers.append(np.asarray(prim_world_pose(hits[0])[0], float) if hits else None)
            if all(f is not None for f in fingers):
                measured_tip = 0.5 * (fingers[0] + fingers[1])
        except Exception as exc:  # noqa: BLE001
            print("finger measurement failed:", exc, flush=True)
        R = Rot.from_quat(quat).as_matrix() if len(quat) == 4 else None
        o = scene.get_observation()
        b = np.asarray(o["base_rgb"])[..., :3].astype(np.uint8)
        s = np.asarray(side.get_rgba())[..., :3].astype(np.uint8)
        d = dict(wrist3_offset_deg=float(np.degrees(dq)), q=q.round(4).tolist(), flange=np.round(pos, 4).tolist(),
                 grip_point=np.round(grip, 4).tolist(),
                 measured_fingertip=None if measured_tip is None else np.round(measured_tip, 4).tolist(), measured_q=np.round(scene.robot.get_joint_positions()[:6], 4).tolist())
        info.append(d)
        print(json.dumps(d), flush=True)
        t = Image.fromarray(np.concatenate([np.asarray(Image.fromarray(b).resize((512, 512))), s], 1))
        ImageDraw.Draw(t).text((5, 5), f"wrist_3 +{np.degrees(dq):.0f} deg  grip {np.round(grip, 3).tolist()}", fill=(255, 255, 0))
        rows.append(np.asarray(t))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(args.out, "mount_check.png"))
    tips = [d["measured_fingertip"] for d in info]
    if all(t is not None for t in tips):
        t0 = np.asarray(tips[0])
        swing = [float(np.linalg.norm(np.asarray(t) - t0) * 1000) for t in tips[1:]]
        verdict = dict(profile=sim_profile.PROFILE, measured_fingertip_shift_mm_at_90_180=swing,
                       coaxial_pass=bool(max(swing) <= 2.0))
    else:
        verdict = dict(profile=sim_profile.PROFILE, error="inner finger prims not found")
    print("VERDICT", json.dumps(verdict), flush=True)
    json.dump(dict(rows=info, verdict=verdict), open(os.path.join(args.out, "mount_check.json"), "w"), indent=1)
    print("saved", os.path.join(args.out, "mount_check.png"), flush=True)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
