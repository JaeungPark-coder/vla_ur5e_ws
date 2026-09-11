"""Upstream feasibility/manipulability gate for Cartesian targets, so a
near-singular or unreachable pose is rejected with an explicit "no" BEFORE
it is ever sent to RMPflow -- instead of RMPflow silently winding up joints
trying to reach it (CONFIRMED 2026-09-11 via check_singularity.py: a target
~22mm above the table, ~466mm from the base, drove elbow_joint to 162.3deg
-- 18deg from full extension -- with wrist_1_joint spinning through -20deg
to -281deg over 400 ticks and position error oscillating 21mm -> 184mm ->
196mm instead of converging).

No new dependencies: cuRobo/Trac-IK are not installed in this environment
(checked), so this uses what Isaac Sim already ships --
isaacsim.robot_motion.motion_generation's LulaKinematicsSolver, the same
Lula library RMPflow itself is built on, which returns an explicit
(joint_positions, success: bool) instead of RMPflow's silent best-effort
push. Manipulability (Yoshikawa index) and the Jacobian condition number are
computed numerically (central-difference Jacobian via repeated forward-
kinematics calls) since Lula's Python bindings don't expose a Jacobian
accessor directly -- this project already trusts LulaKinematicsSolver's own
FK for other measurements (RMPflow.get_end_effector_pose wraps the same
Lula kinematics), so reusing it here is consistent with what's already
verified reliable elsewhere in this project.

Only import from a script already running inside Isaac Sim's Kit runtime.
"""
import numpy as np

# UR5e's own reach spec (Universal Robots UR5e e-Series datasheet: max
# reach 850mm) minus a safety margin -- targets near full extension are
# exactly the elbow-singularity regime CONFIRMED above (elbow near 180deg).
# R_MIN excludes the near-base column where a straight-down wrist orientation
# becomes geometrically forced toward extreme joint configurations (the
# failing target's radius was ~466mm at z=22mm -- comfortably inside
# [R_MIN, R_MAX] by radius alone, which is WHY the height/orientation
# combination -- not radius alone -- is what pushed the elbow toward
# extension; Z_MIN below is the more direct guard for this specific task).
R_MIN_M = 0.20
R_MAX_M = 0.75  # 850mm spec reach minus ~100mm margin
# Height guard: CONFIRMED failing case was tool0 at z=22mm. Every target
# this project has verified converges CLEANLY (0.9-4mm tracking error, no
# oscillation) sat at z >= ~140mm above the table when pointing straight
# down. Keep a margin above the confirmed-bad height.
Z_MIN_M = 0.08

# MoveIt Servo's documented condition-number thresholds (soft/hard) are the
# cited starting point in the research brief this gate implements --
# adapted here rather than re-derived, then meant to be calibrated against
# this specific UR5e + RMPflow combination if it proves too strict/loose.
CONDITION_SOFT = 17.0
CONDITION_HARD = 30.0
MANIPULABILITY_MIN = 0.01  # Yoshikawa index floor; near 0 at any singularity


class FeasibilityGate:
    """Wraps a LulaKinematicsSolver for the UR5e so callers can ask "is this
    Cartesian target safe to send to RMPflow?" and get a real answer instead
    of finding out via joint wind-up.
    """

    def __init__(self, robot_base_prim_path, robot_name="UR5e", ee_frame_name="tool0"):
        from isaacsim.robot_motion.motion_generation.interface_config_loader import (
            load_supported_lula_kinematics_solver_config)
        from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver

        kinematics_config = load_supported_lula_kinematics_solver_config(robot_name)
        self.solver = LulaKinematicsSolver(**kinematics_config)
        self.ee_frame_name = ee_frame_name
        self.joint_names = self.solver.get_joint_names()

    def workspace_ok(self, target_pos):
        """Cheap, simulation-free check first -- catches the confirmed
        failure mode (low + near-base) before spending an IK solve on it."""
        pos = np.asarray(target_pos, dtype=float)
        r = float(np.linalg.norm(pos[:2]))
        z = float(pos[2])
        problems = []
        if r < R_MIN_M:
            problems.append(f"radius {r*1000:.0f}mm < R_MIN {R_MIN_M*1000:.0f}mm (near-base column)")
        if r > R_MAX_M:
            problems.append(f"radius {r*1000:.0f}mm > R_MAX {R_MAX_M*1000:.0f}mm (near full 850mm reach)")
        if z < Z_MIN_M:
            problems.append(f"height {z*1000:.0f}mm < Z_MIN {Z_MIN_M*1000:.0f}mm "
                             f"(CONFIRMED failure case was z=22mm)")
        return (len(problems) == 0), problems

    def _numerical_jacobian(self, joint_positions, eps=1e-4):
        """(6, 6) Jacobian [dposition(3); dlog_rotation(3)] / dq, by central
        differences through the SAME Lula FK RMPflow itself is built on."""
        from scipy.spatial.transform import Rotation as Rot

        q0 = np.asarray(joint_positions, dtype=float)
        pos0, rot0 = self.solver.compute_forward_kinematics(self.ee_frame_name, q0)
        r0 = Rot.from_matrix(rot0)

        jac = np.zeros((6, len(q0)))
        for i in range(len(q0)):
            dq = np.zeros_like(q0)
            dq[i] = eps
            pos_p, rot_p = self.solver.compute_forward_kinematics(self.ee_frame_name, q0 + dq)
            pos_m, rot_m = self.solver.compute_forward_kinematics(self.ee_frame_name, q0 - dq)
            jac[:3, i] = (np.asarray(pos_p) - np.asarray(pos_m)) / (2 * eps)
            # Relative rotation's rotation-vector approximates d(orientation)
            # for a small step -- standard finite-difference angular Jacobian.
            dr = Rot.from_matrix(rot_p) * Rot.from_matrix(rot_m).inv()
            jac[3:, i] = dr.as_rotvec() / (2 * eps)
        return jac

    def check(self, target_pos, target_quat_wxyz, warm_start=None):
        """Full feasibility check: workspace box, IK solvability, then
        manipulability/condition number at the IK solution.

        Returns a dict: {ok, reason, joint_positions, manipulability,
        condition_number} -- `ok` is the single flag callers should gate on;
        `reason` explains a rejection either way.
        """
        ws_ok, ws_problems = self.workspace_ok(target_pos)
        if not ws_ok:
            return {"ok": False, "reason": "workspace: " + "; ".join(ws_problems),
                    "joint_positions": None, "manipulability": None, "condition_number": None}

        # compute_inverse_kinematics wants the quaternion itself (wxyz) --
        # it calls isaacsim.core.utils.numpy.rotations.quats_to_rot_matrices
        # internally, which indexes [1,2,3,0] to reorder wxyz->xyzw for
        # scipy; passing an already-built rotation matrix here (this file's
        # first version did) makes that indexing run on a (3,3) array
        # instead of the (N,4) it expects and throws IndexError.
        quat_wxyz = np.asarray(target_quat_wxyz, dtype=float)

        joint_positions, ik_success = self.solver.compute_inverse_kinematics(
            self.ee_frame_name, np.asarray(target_pos, dtype=float), quat_wxyz,
            warm_start=warm_start)
        if not ik_success:
            return {"ok": False, "reason": "no IK solution within tolerance (unreachable)",
                    "joint_positions": None, "manipulability": None, "condition_number": None}

        jac = self._numerical_jacobian(joint_positions)
        singular_values = np.linalg.svd(jac, compute_uv=False)
        manipulability = float(np.prod(singular_values))  # Yoshikawa index, sqrt(det(JJ^T))
        condition_number = float(singular_values.max() / max(singular_values.min(), 1e-9))

        if manipulability < MANIPULABILITY_MIN:
            return {"ok": False, "reason": f"manipulability {manipulability:.4f} < "
                                            f"{MANIPULABILITY_MIN} (near-singular)",
                    "joint_positions": joint_positions, "manipulability": manipulability,
                    "condition_number": condition_number}
        if condition_number > CONDITION_HARD:
            return {"ok": False, "reason": f"condition number {condition_number:.1f} > "
                                            f"hard limit {CONDITION_HARD} (near-singular)",
                    "joint_positions": joint_positions, "manipulability": manipulability,
                    "condition_number": condition_number}

        return {"ok": True, "reason": (f"soft warning: condition number {condition_number:.1f} > "
                                        f"{CONDITION_SOFT}" if condition_number > CONDITION_SOFT
                                        else "ok"),
                "joint_positions": joint_positions, "manipulability": manipulability,
                "condition_number": condition_number}
