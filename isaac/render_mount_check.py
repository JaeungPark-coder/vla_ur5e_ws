"""Visual check of the gripper mount (README "BLOCKER: the simulated gripper is mounted 90 deg off the wrist_3 axis").
At a demonstrated grasp joint vector, renders the base camera and an extra side camera with wrist_3 at its value and at
+90 / +180 deg, nothing else changed, and logs flange position, flange-z direction and the fingertip (grip point).
A gripper coaxial with wrist_3 keeps its fingertips in place while wrist_3 turns; one mounted 90 deg off swings them on a
12 cm radius.  Output: <out>/mount_check.png and mount_check.json.
The verdict uses the MEASURED fingertip -- the midpoint of the two inner-finger link origins read from the stage (on the
gripper's centre line by symmetry) -- not grip_point_world, which is sim_profile's formula and would pass v5 by construction
(2026-11-16 note, written before the v5 render).
2026-11-17 fix: the first version read the inner-finger prims' USD transforms, which stayed at the flange origin in BOTH v4
and v5 (0.1 mm "shift", so v4's known 90 deg mount "passed" -- the negative control failed, the verdict was void).  The
fingertip is now the midpoint of the left/right inner-finger LINKS read from the articulation's physics view (the same
link transforms the solver integrates).  --out is resolved against the repository root (isaac_py.sh runs from isaac/).
Validity: on v4 the 180 deg shift must be >= 100 mm (formula: ~240 mm), else the measurement is void again.
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
if not os.path.isabs(args.out):
    args.out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.out)
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
    try:   # the first version's raw lookAt transform rendered only the far floor
        from isaacsim.core.utils.viewports import set_camera_view
        set_camera_view(eye=np.array([0.45, 0.55, 0.25]), target=np.array([0.42, 0.0, 0.12]),
                        camera_prim_path="/World/mount_check_cam")
    except Exception as exc:  # noqa: BLE001
        print("side camera aim failed:", exc, flush=True)

    def physics_fingertip():
        """Midpoint of the left/right inner-finger links from the articulation's physics view, plus every finger link."""
        av = scene.robot._articulation_view
        names = list(av.body_names)
        tf = np.asarray(av._physics_view.get_link_transforms(), dtype=float).reshape(-1, len(names), 7)[0]

        def one(side):
            exact = [i for i, n in enumerate(names) if n == f"{side}_inner_finger"]
            loose = [i for i, n in enumerate(names) if n.startswith(f"{side}_inner_finger")]
            return (exact or loose or [None])[0]
        li, ri = one("left"), one("right")
        links = {n: np.round(tf[i, :3], 4).tolist() for i, n in enumerate(names) if "finger" in n or "knuckle" in n}
        if li is None or ri is None:
            return None, dict(body_names=names)
        return 0.5 * (tf[li, :3] + tf[ri, :3]), dict(used=[names[li], names[ri]], finger_links=links)
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
        measured_tip, measured_info = None, None
        try:   # never lose the render to the measurement
            measured_tip, measured_info = physics_fingertip()
        except Exception as exc:  # noqa: BLE001
            print("finger measurement failed:", exc, flush=True)
        R = Rot.from_quat(quat).as_matrix() if len(quat) == 4 else None
        o = scene.get_observation()
        b = np.asarray(o["base_rgb"])[..., :3].astype(np.uint8)
        s = np.asarray(side.get_rgba())[..., :3].astype(np.uint8)
        d = dict(wrist3_offset_deg=float(np.degrees(dq)), q=q.round(4).tolist(), flange=np.round(pos, 4).tolist(),
                 grip_point=np.round(grip, 4).tolist(),
                 measured_fingertip=None if measured_tip is None else np.round(measured_tip, 4).tolist(),
                 measured_info=measured_info, measured_q=np.round(scene.robot.get_joint_positions()[:6], 4).tolist())
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
        verdict = dict(profile=sim_profile.PROFILE, error="inner finger links not found in the articulation")
    print("VERDICT", json.dumps(verdict), flush=True)
    json.dump(dict(rows=info, verdict=verdict), open(os.path.join(args.out, "mount_check.json"), "w"), indent=1)
    print("saved", os.path.join(args.out, "mount_check.png"), flush=True)
except BaseException:
    import traceback
    print("EXC", traceback.format_exc(), flush=True)
c.simulation_app.close()
