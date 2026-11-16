"""sim_profile v4/v5 and the coaxial gripper re-mount (README "v5 decision"), without Isaac Sim.

The grip-frame algebra is checked against the property that matters on the real arm: with the gripper pointing down, the
commanded tool0 z -- the wrist_3 axis -- points down too (v1-v4: horizontal, the BLOCKER).  The USD re-mount is checked on a
small hand-built stage with the same structure as the asset (arm link, flange child Xform, gripper subtree, weld joint);
it needs pxr (usd-core) and is skipped without it.
"""
import importlib
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

DOWN = Rot.from_rotvec([0.0, np.pi, 0.0])          # scripted_pick_place.DOWNWARD_ROTVEC
# the asset: tool0 z lies along the flange prim's +X (pick_place_scene GRIPPER_TCP_OFFSET_M note); any rotation taking +Z to
# +X will do for these properties -- the real one is measured from RMPflow at scene start (r_flange_to_tool0)
R_F2T = Rot.from_euler("y", 90.0, degrees=True) * Rot.from_euler("z", 25.0, degrees=True)


@pytest.fixture(autouse=True)
def _fresh_profile_module():
    """Other tests import sim_profile under the default (v4); never leave a v5 module behind."""
    yield
    sys.modules.pop("sim_profile", None)


def load_profile(monkeypatch, name):
    monkeypatch.setenv("VLA_SIM_PROFILE", name)
    sys.modules.pop("sim_profile", None)
    return importlib.import_module("sim_profile")


def test_flange_to_tool0_assumption():
    np.testing.assert_allclose(R_F2T.apply([0, 0, 1]), [1, 0, 0], atol=1e-12)


def test_v4_is_the_old_computation(monkeypatch):
    sp = load_profile(monkeypatch, "v4")
    assert sp.placeholders() == [] and sp.EXPERT_MAX_JOINT_SPEED_RAD_S is None and sp.TABLE_Z_M == 0.0
    assert sp.blockers() == [] and sp.READY_JOINTS is None and sp.BASE_IMAGE_CROP is None and sp.base_image_wh() == (256, 256)
    target = np.array([0.45, 0.0, 0.04])
    fpos, r_t0 = sp.flange_target(target, DOWN, R_F2T)
    np.testing.assert_allclose(fpos, target - 0.12 * DOWN.apply([0, 0, 1]))              # old: step back 0.12 along approach
    np.testing.assert_allclose((r_t0.inv() * (DOWN * R_F2T)).magnitude(), 0, atol=1e-12)  # old: r_grip * r_flange_to_tool0
    # ... which is the blocker: tool0 z (the wrist_3 axis) horizontal while the fingers point down
    assert abs(r_t0.apply([0, 0, 1])[2]) < 1e-9


@pytest.mark.parametrize("roll", [0.0, 37.0])
def test_v5_points_the_wrist_axis_down_and_round_trips(monkeypatch, roll):
    sp = load_profile(monkeypatch, "v5")
    monkeypatch.setattr(sp, "R_GRIP_IN_FLANGE",
                        Rot.from_euler("y", 90, degrees=True) * Rot.from_euler("z", roll, degrees=True))
    monkeypatch.setattr(sp, "GRIPPER_TCP_M", 0.16)
    target = np.array([0.45, 0.05, 0.04])
    fpos, r_t0 = sp.flange_target(target, DOWN, R_F2T)
    np.testing.assert_allclose(r_t0.apply([0, 0, 1]), [0, 0, -1], atol=1e-12)        # tool axis straight down: coaxial
    np.testing.assert_allclose(fpos, target + [0, 0, 0.16], atol=1e-12)              # fingertips TCP below the flange
    r_flange = r_t0 * R_F2T.inv()                                                     # what the arm reaches
    np.testing.assert_allclose(sp.grip_point(fpos, r_flange), target, atol=1e-12)     # and grip_point reads it back
    # the grip frame is reached exactly, so the expert's grasp yaw is unchanged
    assert (r_flange * sp.R_GRIP_IN_FLANGE * DOWN.inv()).magnitude() < 1e-9


def test_v5_lists_lab_placeholders_and_rejects_a_short_tcp(monkeypatch):
    sp = load_profile(monkeypatch, "v5")
    assert {"GRIPPER_TCP_M", "TABLE_Z_M", "CUBE_SIZE_M", "BASE_CAMERA", "CAMERA_RESOLUTION"} <= set(sp.placeholders())
    assert sp.EXPERT_MAX_JOINT_SPEED_RAD_S == pytest.approx(np.pi / 2)
    np.testing.assert_allclose(sp.ur_base_to_world(sp.BASE_CAMERA["position_ur"]), [0.9, 0.0, 0.5])  # = v4's camera
    # 2026-11-16: the real gripper is an RG2 -> blocked until the sim has one; v4 never is
    assert sp.REAL_GRIPPER == "onrobot_rg2" and any("onrobot_rg2" in b for b in sp.blockers())
    assert {"BASE_IMAGE_CROP", "GRIPPER_CLOSE_TIME_S"} <= set(sp.placeholders())
    # the ready pose placeholder (candidate B, checked on URSim): tool straight down, elbow bent up
    import os as _os
    sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), "..", "src", "vla_bridge"))
    from vla_bridge.ur5e_kinematics import flange
    T = flange(sp.READY_JOINTS)[0]
    assert "READY_JOINTS" in sp.placeholders() and T[2, 2] < -0.9999 and sp.READY_JOINTS[2] > 0.8
    monkeypatch.setenv("VLA_SIM_PROFILE", "v6")
    sys.modules.pop("sim_profile", None)
    with pytest.raises(ValueError):
        importlib.import_module("sim_profile")
    sys.modules.pop("sim_profile", None)


def test_zed_intrinsics_give_the_fov_and_the_crop_gives_the_image_size(monkeypatch):
    sp = load_profile(monkeypatch, "v5")
    # a ZED 2i-like left camera at 720p (2.1 mm lens): fx ~ 530 px -> ~100.6 deg horizontal
    cam = dict(position_ur=(-0.7, 0.0, 0.6), aim_ur=(-0.45, 0.0, 0.0),
               intrinsics=dict(width=1280, height=720, fx=530.0, fy=530.0, cx=640.0, cy=360.0))
    monkeypatch.setattr(sp, "CAMERA_RESOLUTION", (448, 252))
    assert sp.base_camera_hfov_deg(cam) == pytest.approx(np.degrees(2 * np.arctan(640 / 530)))
    monkeypatch.setattr(sp, "CAMERA_RESOLUTION", (256, 256))                  # wrong aspect for that camera
    with pytest.raises(ValueError, match="aspect"):
        sp.base_camera_hfov_deg(cam)
    monkeypatch.setattr(sp, "CAMERA_RESOLUTION", (448, 252))
    monkeypatch.setattr(sp, "BASE_IMAGE_CROP", (0.25, 0.2, 0.75, 1.0))
    assert sp.base_image_wh() == (224, 202)                                    # the crop's own pixels ...
    monkeypatch.setattr(sp, "BASE_IMAGE_SIZE", (224, 224))
    assert sp.base_image_wh() == (224, 224)                                    # ... or the size asked for


# ---- the USD re-mount -------------------------------------------------------------------------------------------------
pxr = pytest.importorskip("pxr")


def _xform(stage, path, T):
    from pxr import Gf, UsdGeom
    x = UsdGeom.Xform.Define(stage, path)
    x.AddTransformOp().Set(Gf.Matrix4d(*np.asarray(T, float).T.ravel().tolist()))
    return x.GetPrim()


def _T(rot, pos):
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[:3, 3] = pos
    return T


def test_realign_gripper_mount_moves_links_and_keeps_the_weld_consistent():
    from pxr import Gf, Usd, UsdPhysics
    from isaac_sim_common import _world_matrix, realign_gripper_mount
    stage = Usd.Stage.CreateInMemory()
    _xform(stage, "/World", np.eye(4))
    _xform(stage, "/World/ur5e", np.eye(4))
    W_arm = _T(Rot.from_euler("xyz", [10, -20, 30], degrees=True), [0.3, -0.2, 0.5])
    arm = _xform(stage, "/World/ur5e/wrist_3_link", W_arm)
    UsdPhysics.RigidBodyAPI.Apply(arm)
    X_AF = _T(Rot.from_euler("xyz", [-90, -90, 0], degrees=True), [0, 0.0996, 0])        # flange child Xform, rotated
    _xform(stage, "/World/ur5e/wrist_3_link/flange", X_AF)
    _xform(stage, "/World/ur5e/Gripper", np.eye(4))
    _xform(stage, "/World/ur5e/Gripper/Robotiq_2F_85", np.eye(4))
    W_F = W_arm @ X_AF
    X_FG = _T(Rot.from_euler("z", 15, degrees=True), [0, 0, 0.01])                      # gripper base, on flange +Z
    base = _xform(stage, "/World/ur5e/Gripper/Robotiq_2F_85/base_link", W_F @ X_FG)
    finger = _xform(stage, "/World/ur5e/Gripper/Robotiq_2F_85/left_finger", W_F @ X_FG @ _T(Rot.identity(), [0.02, 0, 0.1]))
    for p in (base, finger):
        UsdPhysics.RigidBodyAPI.Apply(p)
    j = UsdPhysics.FixedJoint.Define(stage, "/World/ur5e/wrist_3_link/gripper_weld")
    j.CreateBody0Rel().SetTargets(["/World/ur5e/wrist_3_link"])
    j.CreateBody1Rel().SetTargets(["/World/ur5e/Gripper/Robotiq_2F_85/base_link"])
    X_Aj = X_AF @ X_FG                                    # joint frame = gripper base frame, expressed in the arm body
    q = Rot.from_matrix(X_Aj[:3, :3]).as_quat()
    j.CreateLocalPos0Attr(Gf.Vec3f(*X_Aj[:3, 3].tolist()))
    j.CreateLocalRot0Attr(Gf.Quatf(float(q[3]), *[float(v) for v in q[:3]]))
    j.CreateLocalPos1Attr(Gf.Vec3f(0, 0, 0))
    j.CreateLocalRot1Attr(Gf.Quatf(1, 0, 0, 0))
    UsdPhysics.FixedJoint.Define(stage, "/World/ur5e/Gripper/Robotiq_2F_85/finger_joint_like")   # internal: ignored

    R = Rot.from_euler("y", 90, degrees=True)
    shift = 0.04 * R.apply([0, 0, 1])
    old_finger = _world_matrix(finger)
    info = realign_gripper_mount(stage, stage.GetPrimAtPath("/World/ur5e"), "/World/ur5e/wrist_3_link/flange", R, shift)
    assert info["joint"] == "/World/ur5e/wrist_3_link/gripper_weld" and info["arm_side"] == 0

    M = _T(R, shift)
    W = W_F @ M @ np.linalg.inv(W_F)
    np.testing.assert_allclose(_world_matrix(base), W @ W_F @ X_FG, atol=1e-6)          # links moved rigidly
    np.testing.assert_allclose(_world_matrix(finger), W @ old_finger, atol=1e-6)
    # weld: arm * localFrame0 == gripper base * localFrame1 (= identity) at the NEW pose
    jj = UsdPhysics.Joint(stage.GetPrimAtPath("/World/ur5e/wrist_3_link/gripper_weld"))
    p0, r0 = jj.GetLocalPos0Attr().Get(), jj.GetLocalRot0Attr().Get()
    im = r0.GetImaginary()
    X0 = _T(Rot.from_quat([im[0], im[1], im[2], r0.GetReal()]), [p0[0], p0[1], p0[2]])
    np.testing.assert_allclose(W_arm @ X0, _world_matrix(base), atol=1e-6)
    # and the gripper axis (old flange +Z) now lies along the flange's +X, shifted 4 cm out
    G_in_F = np.linalg.inv(W_F) @ _world_matrix(base)
    np.testing.assert_allclose(G_in_F[:3, :3] @ [0, 0, 1], [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(G_in_F[:3, 3], R.apply([0, 0, 0.01]) + shift, atol=1e-6)


def test_realign_refuses_when_the_weld_is_ambiguous():
    from pxr import Usd, UsdPhysics
    from isaac_sim_common import realign_gripper_mount
    stage = Usd.Stage.CreateInMemory()
    for p in ("/World", "/World/ur5e", "/World/ur5e/wrist_3_link", "/World/ur5e/wrist_3_link/flange", "/World/ur5e/Gripper",
              "/World/ur5e/Gripper/a"):
        _xform(stage, p, np.eye(4))
    with pytest.raises(RuntimeError, match="exactly one"):
        realign_gripper_mount(stage, stage.GetPrimAtPath("/World/ur5e"), "/World/ur5e/wrist_3_link/flange",
                              Rot.identity(), np.zeros(3))
    for k in range(2):
        j = UsdPhysics.FixedJoint.Define(stage, f"/World/ur5e/w{k}")
        j.CreateBody0Rel().SetTargets(["/World/ur5e/wrist_3_link"])
        j.CreateBody1Rel().SetTargets(["/World/ur5e/Gripper/a"])
    with pytest.raises(RuntimeError, match="exactly one"):
        realign_gripper_mount(stage, stage.GetPrimAtPath("/World/ur5e"), "/World/ur5e/wrist_3_link/flange",
                              Rot.identity(), np.zeros(3))
