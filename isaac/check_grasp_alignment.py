"""Why does the gripper close on the cube without lifting it?

collect_demos.py (2026-09-21, this session, with a working camera mount and
no PhysX divergence for the first time) rejected 15/15 attempts as "never
lifted" -- max_cube_z stayed at 0.02-0.041m against a 0.08m LIFT_Z_THRESHOLD.

This file already found (same day, earlier runs): the gripper closes to
93-100% of target; the common-mode X offset (both pads shifted the SAME
way -- Y is the real closing axis, opposite signs there) grows smoothly
from ~-10mm to ~+33mm across ticks 300-402, spanning the descend AND close
segments, with NO discontinuity where the pad's own bbox crosses the
cube's actual top face -- ruling out premature contact as the trigger.
That smooth, unbounded-looking growth against a fully FROZEN commanded
target is the shape pivot_dwell_check.py's own DYNAMICS hypothesis
describes (soft drive gains sagging under a loaded pose), but that script's
cold-start test design (488mm initial error, still converging) isn't
comparable to this already-settled-then-drifting scenario.

This run adds the two checks needed to actually tell DYNAMICS apart from
something worse, or from a pure targeting/waypoint error:

  HOLD PHASE       after the close segment's last frame, keep commanding
                   that EXACT SAME frozen target (position, orientation,
                   gripper=1.0) for `--hold-ticks` more ticks, and watch
                   the common-mode offset. Plateauing (asymptotically
                   flat) is classic finite-stiffness droop settling to a
                   new equilibrium under load -- the kick already happened
                   during close, and the hold itself is stable. Continuing
                   to grow (linearly or accelerating) through the hold is
                   a more serious problem (integrator windup, genuinely
                   divergent gains, or an unidentified mechanism) and
                   needs re-diagnosing from scratch.
  CLOSURE-VS-TIME  the existing per-tick close-segment log already records
                   both tick number and the gripper's actual closure
                   fraction every row -- reading the SAME dx column
                   against closure% instead of tick (both already printed)
                   distinguishes "the closing motion itself is what drives
                   this" (dx tracks closure%) from "it's just however long
                   the hold has run" (dx tracks tick, closure% saturates
                   early and dx keeps moving after). No new instrumentation
                   needed for this half -- just read the close-segment
                   rows already logged below differently.

    python3 check_grasp_alignment.py
    python3 check_grasp_alignment.py --episodes 5 --hold-ticks 300

CONFIRMED 2026-09-21, extending the above: closure% (not tick) is the real
independent variable -- dx IMPROVES from ~-11mm toward ~-2 to -6mm as
closure goes 0% -> 35-70%, then WORSENS sharply for the rest of the close
(all 4 traced episodes show the same shape, just at different closure%
thresholds: 35%, 46.7%, 49%, 70%). That is not "however long the target
has been held" (which the hold-phase data also rules out on its own --
+300 ticks past close plateaus, it does not grow further), and not
premature contact either (see above) -- it is specifically the LATTER
PART of closing, when the pads are actually squeezing the object, that
drives it. The hold-phase plateau values themselves are large and highly
variable across otherwise-identical episodes (+31.8, -39.8, -81.7,
-39.5mm) -- consistent with a squeeze-reaction force whose net lateral
component depends on exactly how off-centre that episode's approach
already was, pushing a too-soft position controller to a new (and often
badly off-cube) equilibrium rather than holding the commanded one.

--stiffness-scale/--damping-scale/--joints below exist to test the fix
pivot_dwell_check.py's own DYNAMICS row always prescribed for this
shape -- tune the wrist drive gains -- as a SWEEP (not one guessed value:
a linear-spring approximation needs >=4x effective stiffness to bring an
82mm plateau under the cube's 20mm half-width, so single-point guesses
are unlikely to land in one try) with an explicit eye on whether raising
stiffness reintroduces the kind of instability the gripper-mass A/B test
found earlier today (hence the NaN/Inf check on every tick, same as
check_rmpflow_stability.py). --seed fixes the cube-spawn sequence so
different stiffness runs are directly paired against the SAME spawns,
not different random ones.

2026-09-23: --finger-kp/--finger-kd sweep the GRIPPER's own finger_joint
drive (GripperController's as-shipped-fixed 20000/500 baseline, in live-
controller/radian units -- see its docstring), independent of the wrist
sweep above, which never touched it. Motivated by isaac-sim/IsaacLab#3385
(a Robotiq-85 report of closing-induced end-effector rotation from
amplified left/right contact-force asymmetry at stiffness>=2000 -- this
project's baseline is 10x that) and by this file's own 2026-09-21 finding
that dx worsens specifically in the LATTER part of closing, i.e. while the
pads are actually squeezing -- exactly when a stiff finger drive would
amplify a small contact-force asymmetry into a common-mode push. Hold the
damping ratio constant while sweeping (finger_kd = 500*sqrt(finger_kp/20000)
-- linear kd scaling changes the ratio instead, see the session that
derived this) so a change in dx is attributable to stiffness, not an
incidental over/under-damping shift.
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from pick_place_scene import (  # noqa: E402
    PickPlaceScene, PLACE_TARGET_POSITION, GRIPPER_TCP_OFFSET_M, CUBE_SIZE_M)
from scripted_pick_place import ScriptedPickPlace, GRASP_HEIGHT  # noqa: E402
from isaac_sim_common import (  # noqa: E402
    GRIPPER_OPEN_POS, GRIPPER_CLOSED_POS, prim_world_pose, ROBOT_PRIM_PATH,
    ARM_JOINT_NAMES, get_joint_drive_gains, set_joint_drive_gains,
    GRIP_MATERIAL_PRIM_PATH)

# The last 2 of ARM_JOINT_NAMES -- closest to the tool, so a gripper-tip
# reaction force most directly loads these (smallest inertia, most easily
# deflected by a local tip torque) even though a tip force's raw moment ARM
# is actually larger at the shoulder. --joints=wrist scales only these,
# to compare against scaling the whole arm uniformly.
WRIST_JOINT_NAMES = ARM_JOINT_NAMES[-2:]


def first_nonfinite_index(arr):
    """Index of the first non-finite entry in a flat array, or None -- same
    check check_rmpflow_stability.py uses, needed here too since raising
    stiffness is exactly the kind of change that caused the gripper-mass
    divergence found earlier today."""
    bad = np.where(~np.isfinite(arr))[0]
    return int(bad[0]) if bad.size else None

# CONFIRMED 2026-09-21 by direct prim inspection (Usd.PrimRange under
# /World/ur5e/Gripper): the two pad links that actually contact a grasped
# object. Used here to measure the REAL fingertip position independently of
# grip_point_world()/GRIPPER_TCP_OFFSET_M -- see this file's own printed
# note on why that formula-based measurement cannot detect a miscalibrated
# GRIPPER_TCP_OFFSET_M (it uses the same constant to set the target AND to
# read it back, so the two cancel algebraically regardless of the constant's
# real-world accuracy).
LEFT_INNER_FINGER_PRIM_PATH = "/World/ur5e/Gripper/Robotiq_2F_85/left_inner_finger"
RIGHT_INNER_FINGER_PRIM_PATH = "/World/ur5e/Gripper/Robotiq_2F_85/right_inner_finger"
# The actual rubber pad mesh, found by direct prim traversal (2026-09-21) --
# a child of the link above, not the link's own origin. The link origin
# turned out to be ~140mm above the cube even mid-grasp (a finger this
# short cannot physically span that), which only makes sense if the link's
# origin is the pad's PROXIMAL joint (where it mounts to the knuckle), not
# its contact surface. CONFIRMED this mesh prim carries no transform op of
# its own (its ComputeLocalToWorldTransform is identical to its parent
# link's), so the pad's real offset from the link origin is baked into the
# mesh's own vertex positions -- mesh_world_bbox_center below reads those
# directly and transforms them to world space, rather than trusting the
# prim's own (uninformative here) transform.
LEFT_FINGERTIP_MESH_PATH = (
    "/World/ur5e/Gripper/Robotiq_2F_85/left_inner_finger/visuals/"
    "Defeatured_2F_85_PAD_OPEN_fingertipsstep_01/Defeatured_2F_85_PAD_OPEN_fingertipsstep")
RIGHT_FINGERTIP_MESH_PATH = (
    "/World/ur5e/Gripper/Robotiq_2F_85/right_inner_finger/visuals/"
    "Defeatured_2F_85_PAD_OPEN_fingertipsstep_01/Defeatured_2F_85_PAD_OPEN_fingertipsstep")


def say(line=""):
    print(line, flush=True)


def mesh_world_bbox_center(prim):
    """World-space centroid of a mesh's own vertex bounding box -- unlike
    prim_world_pose (which reads only the prim's transform op, and returns
    the SAME point as an unshifted parent when the mesh authors no
    transform of its own), this reads the actual vertex positions, so a
    pad mesh whose real extent is baked into its points rather than its
    transform still gives a physically meaningful location."""
    mesh = UsdGeom.Mesh(prim)
    points = mesh.GetPointsAttr().Get()
    if not points:
        return None
    mat = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    world_points = np.array([[*mat.Transform(p)] for p in points], dtype=float)
    return world_points.mean(axis=0), world_points.min(axis=0), world_points.max(axis=0)


def report_pose_detail(scene, label):
    """The full formula/link-origin/mesh-bbox measurement trio this file
    already prints once at end-of-close -- factored out so the same report
    can also run at end-of-hold, for a clean before/after comparison."""
    grip_point = scene.grip_point_world()
    cube_pos = scene.get_cube_position()
    delta = grip_point - cube_pos
    say(f"\n  --- {label} ---")
    say(f"  formula grip-point - cube (self-consistent, see module docstring): "
        f"dx={delta[0] * 1000:+.1f}mm dy={delta[1] * 1000:+.1f}mm dz={delta[2] * 1000:+.1f}mm")

    for flabel, path in (("left_inner_finger (LINK origin)", LEFT_INNER_FINGER_PRIM_PATH),
                          ("right_inner_finger (LINK origin)", RIGHT_INNER_FINGER_PRIM_PATH)):
        pos, _ = prim_world_pose(scene.stage.GetPrimAtPath(path))
        d = pos - cube_pos
        say(f"  {flabel:32s} - cube: dx={d[0] * 1000:+7.1f}mm dy={d[1] * 1000:+7.1f}mm "
            f"dz={d[2] * 1000:+7.1f}mm")

    for flabel, path in (("left fingertip MESH bbox", LEFT_FINGERTIP_MESH_PATH),
                          ("right fingertip MESH bbox", RIGHT_FINGERTIP_MESH_PATH)):
        result = mesh_world_bbox_center(scene.stage.GetPrimAtPath(path))
        if result is None:
            say(f"  {flabel:32s} - mesh has no points; skipped")
            continue
        centroid, bbox_min, bbox_max = result
        d_centroid = centroid - cube_pos
        say(f"  {flabel:32s} centroid-cube: dx={d_centroid[0] * 1000:+7.1f}mm "
            f"dy={d_centroid[1] * 1000:+7.1f}mm dz={d_centroid[2] * 1000:+7.1f}mm "
            f"| bbox z-range {bbox_min[2] * 1000:.1f}..{bbox_max[2] * 1000:.1f}mm "
            f"(cube z={cube_pos[2] * 1000:.1f}mm, top={1000 * (cube_pos[2] + CUBE_SIZE_M / 2):.1f}mm)")
    return delta, cube_pos


def run(episodes, hold_ticks, seed, stiffness_scale, damping_scale, joint_choice,
        static_friction, dynamic_friction, finger_kp, finger_kd):
    scene = PickPlaceScene(with_gripper=True, finger_kp=finger_kp, finger_kd=finger_kd)
    say(f"GRIPPER_OPEN_POS={GRIPPER_OPEN_POS} GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS} rad")
    say(f"finger_joint drive gains: kp={finger_kp if finger_kp is not None else 20000.0:g} "
        f"kd={finger_kd if finger_kd is not None else 500.0:g} (None args = as-shipped-fixed "
        f"baseline; resolved lazily on first gripper command, not yet applied here)")

    # Fixed seed: different --stiffness-scale runs need the SAME cube-spawn
    # sequence to be paired comparisons rather than different random draws
    # (the hold-phase plateau already varies +32 to -82mm across otherwise-
    # identical episodes, so an unpaired comparison could easily mistake a
    # lucky/unlucky spawn for a stiffness effect).
    scene._rng = np.random.default_rng(seed)
    say(f"cube-spawn RNG seeded with {seed} (fixed across runs for paired comparison)")

    if static_friction is not None or dynamic_friction is not None:
        # The material is already authored (isaac_sim_common.add_gripper_
        # colliders/add_shape both bind it, unconditionally, at scene-build
        # time) -- re-set its attributes on the SAME shared prim rather than
        # creating a new one, so both the pads and the cube (already bound
        # to this exact prim) pick up the override without re-binding
        # anything. Sweeping this per-run (not editing the isaac_sim_common
        # constants) keeps each value a single labelled process run, same
        # as --stiffness-scale.
        material_prim = scene.stage.GetPrimAtPath(GRIP_MATERIAL_PRIM_PATH)
        material_api = UsdPhysics.MaterialAPI(material_prim)
        before_sf = material_api.GetStaticFrictionAttr().Get()
        before_df = material_api.GetDynamicFrictionAttr().Get()
        if static_friction is not None:
            material_api.CreateStaticFrictionAttr(static_friction)
        if dynamic_friction is not None:
            material_api.CreateDynamicFrictionAttr(dynamic_friction)
        say(f"friction material {GRIP_MATERIAL_PRIM_PATH}: "
            f"static {before_sf:g}->{material_api.GetStaticFrictionAttr().Get():g}, "
            f"dynamic {before_df:g}->{material_api.GetDynamicFrictionAttr().Get():g}")

    joint_names = WRIST_JOINT_NAMES if joint_choice == "wrist" else ARM_JOINT_NAMES
    if stiffness_scale != 1.0 or damping_scale != 1.0:
        robot_prim = scene.stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        before = get_joint_drive_gains(robot_prim, joint_names=joint_names)
        say(f"scaling {joint_choice} joints x{stiffness_scale:g} stiffness, "
            f"x{damping_scale:g} damping")
        say(f"  before: {before}")
        for name, gains in before.items():
            if gains is None:
                continue
            stiffness, damping = gains
            set_joint_drive_gains(
                robot_prim, joint_names=(name,),
                stiffness=(stiffness * stiffness_scale) if stiffness is not None else None,
                damping=(damping * damping_scale) if damping is not None else None)
        after = get_joint_drive_gains(robot_prim, joint_names=joint_names)
        say(f"  after:  {after}")

    hold_plateaus = []  # (episode, final |d| mm) -- printed as a compact summary at the end
    lift_results = []  # (episode, max_cube_z, lifted) -- only populated when hold_ticks == 0
    any_nonfinite = False
    for ep in range(1, episodes + 1):
        obs = scene.reset()
        policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
        waypoints = policy.waypoints
        # waypoints[:5] = approach, settle, descend, settle, close (see
        # ScriptedPickPlace.frames_until_grasp's own comment on this layout
        # -- a second settle segment, between descend and close, was added
        # 2026-09-19 by a parallel session; this file's slice was still [:4]
        # from before that and needed updating after the 2026-09-21 merge).
        n_through_close = sum(n for _, _, n in waypoints[:5])
        segment_bounds = np.cumsum([n for _, _, n in waypoints[:5]])
        segment_names = ["approach", "settle", "descend", "settle2", "close"]

        say(f"\n=== episode {ep}/{episodes}: cube at {np.round(scene.cube_position, 4).tolist()} ===")
        say(f"{'tick':>5} {'seg':>8} {'gripper_tgt':>11} {'gripper_act':>11} "
            f"{'dx_mm':>7} {'dy_mm':>7} {'dz_mm':>7} {'|d|_mm':>7} {'pad_z_gap_mm':>13}")

        frames = list(policy.generate_frames())
        seg_idx = 0
        episode_diverged = False
        for tick, (target_pos, target_rotvec, target_gripper) in enumerate(frames[:n_through_close], start=1):
            scene.step_towards(target_pos, target_rotvec, target_gripper)
            joint_pos = np.asarray(scene.robot.get_joint_positions(), dtype=float)
            joint_vel = np.asarray(scene.robot.get_joint_velocities(), dtype=float)
            if first_nonfinite_index(joint_pos) is not None or first_nonfinite_index(joint_vel) is not None:
                say(f"\n=== NON-FINITE joint state at episode {ep} tick {tick} -- "
                    f"stiffness x{stiffness_scale:g} is likely unstable, stopping this episode ===")
                any_nonfinite = True
                episode_diverged = True
                break
            while seg_idx < len(segment_bounds) - 1 and tick > segment_bounds[seg_idx]:
                seg_idx += 1
            seg_name = segment_names[seg_idx]

            grip_point = scene.grip_point_world()
            cube_pos = scene.get_cube_position()
            # Grip point sits GRIPPER_TCP_OFFSET_M ABOVE where the fingers
            # actually pinch (see _grip_pose_to_tool0/grip_point_world) --
            # compare against the cube's centre directly, same quantity the
            # scripted waypoints (at_cube = cube_position + GRASP_HEIGHT)
            # target, so dz should read close to GRASP_HEIGHT once the
            # close segment starts, not 0.
            delta = grip_point - cube_pos
            actual_gripper = scene.gripper.get_normalized_position()

            # ONLY during descend/close, and only every few ticks (this is
            # the real vertex-bbox read, not the formula -- worth the extra
            # cost here specifically to test whether the dx drift's onset
            # lines up with the pad's own lowest point reaching the cube's
            # actual top face, which would confirm early contact rather
            # than a pure control/kinematic effect).
            pad_gap_str = ""
            if seg_name in ("descend", "settle2", "close") and tick % 4 == 0:
                result = mesh_world_bbox_center(scene.stage.GetPrimAtPath(LEFT_FINGERTIP_MESH_PATH))
                if result is not None:
                    _, bbox_min, _ = result
                    cube_top_z = cube_pos[2] + CUBE_SIZE_M / 2.0
                    pad_gap_str = f"{(bbox_min[2] - cube_top_z) * 1000:+13.1f}"

            # Print every tick during "close" (short segment, matters most),
            # every 4th during descend (this is where the swing happens),
            # every 20th otherwise (long segments, just show the trend).
            if (seg_name == "close" or (seg_name in ("descend", "settle2") and tick % 4 == 0)
                    or tick % 20 == 0 or tick == n_through_close):
                say(f"{tick:>5} {seg_name:>8} {target_gripper:>11.3f} {actual_gripper:>11.3f} "
                    f"{delta[0] * 1000:>7.1f} {delta[1] * 1000:>7.1f} {delta[2] * 1000:>7.1f} "
                    f"{float(np.linalg.norm(delta)) * 1000:>7.1f} {pad_gap_str:>13}")

        if episode_diverged:
            hold_plateaus.append((ep, None))
            continue

        final_gripper_frac = scene.gripper.get_normalized_position()
        final_joint_rad = GRIPPER_OPEN_POS + final_gripper_frac * (GRIPPER_CLOSED_POS - GRIPPER_OPEN_POS)
        say(f"\n  end of close segment: gripper at {100 * final_gripper_frac:.1f}% of target range "
            f"({final_joint_rad:.4f} rad vs GRIPPER_CLOSED_POS={GRIPPER_CLOSED_POS} rad)")
        report_pose_detail(scene, "END OF CLOSE (tick %d)" % n_through_close)

        if hold_ticks > 0:
            # HOLD PHASE: command the EXACT SAME frozen target (position,
            # orientation, gripper=1.0) close's last frame used, for
            # hold_ticks more ticks -- see module docstring for what
            # plateauing vs. continued growth here would each mean. This
            # replaces continuing on to the scripted lift (a moving
            # target would confound the question this phase asks), so
            # max_cube_z below is not comparable to a normal episode when
            # --hold-ticks > 0.
            final_target_pos, final_target_rotvec, final_target_gripper = frames[n_through_close - 1]
            say(f"\n  --- HOLD: target frozen at close's final commanded pose, "
                f"+{hold_ticks} ticks ---")
            say(f"  {'tick':>5} {'dx_mm':>7} {'dy_mm':>7} {'dz_mm':>7} {'|d|_mm':>7}")
            final_delta_norm = None
            for h in range(1, hold_ticks + 1):
                scene.step_towards(final_target_pos, final_target_rotvec, final_target_gripper)
                joint_pos = np.asarray(scene.robot.get_joint_positions(), dtype=float)
                joint_vel = np.asarray(scene.robot.get_joint_velocities(), dtype=float)
                if (first_nonfinite_index(joint_pos) is not None
                        or first_nonfinite_index(joint_vel) is not None):
                    say(f"\n=== NON-FINITE joint state during hold at episode {ep} "
                        f"hold-tick {h} -- stiffness x{stiffness_scale:g} is likely "
                        f"unstable, stopping this episode's hold ===")
                    any_nonfinite = True
                    episode_diverged = True
                    break
                if h <= 5 or h % 20 == 0 or h == hold_ticks:
                    grip_point = scene.grip_point_world()
                    cube_pos = scene.get_cube_position()
                    delta = grip_point - cube_pos
                    final_delta_norm = float(np.linalg.norm(delta)) * 1000.0
                    say(f"  {h:>5} {delta[0] * 1000:>7.1f} {delta[1] * 1000:>7.1f} "
                        f"{delta[2] * 1000:>7.1f} {final_delta_norm:>7.1f}")
            if episode_diverged:
                hold_plateaus.append((ep, None))
                continue
            report_pose_detail(scene, f"END OF HOLD (tick {n_through_close}+{hold_ticks})")
            max_cube_z = float(scene.get_cube_position()[2])
            say(f"  max_cube_z (frozen-hold, NOT a real lift attempt): {max_cube_z:.4f}m")
            hold_plateaus.append((ep, final_delta_norm))
        else:
            # Continue driving through lift to report the actual outcome
            # this episode would have scored, for direct comparison with
            # collect_demos.py's own verdict.
            max_cube_z = float(scene.get_cube_position()[2])
            for target_pos, target_rotvec, target_gripper in frames[n_through_close:]:
                scene.step_towards(target_pos, target_rotvec, target_gripper)
                max_cube_z = max(max_cube_z, float(scene.get_cube_position()[2]))
            lifted = max_cube_z >= 0.08
            say(f"  max_cube_z over the rest of the episode: {max_cube_z:.4f}m "
                f"(lifted={'YES' if lifted else 'no'})")
            lift_results.append((ep, max_cube_z, lifted))

    say(f"\n=== SUMMARY (stiffness x{stiffness_scale:g}, damping x{damping_scale:g}, "
        f"joints={joint_choice}, static_friction={static_friction}, "
        f"dynamic_friction={dynamic_friction}, finger_kp={finger_kp}, finger_kd={finger_kd}, "
        f"seed={seed}) ===")
    say(f"  any non-finite joint state: {'YES -- UNSTABLE' if any_nonfinite else 'no'}")
    for ep_num, plateau in hold_plateaus:
        say(f"  episode {ep_num}: hold-plateau |d| = "
            f"{'DIVERGED' if plateau is None else f'{plateau:.1f}mm'}")
    for ep_num, max_z, lifted in lift_results:
        say(f"  episode {ep_num}: max_cube_z={max_z:.4f}m "
            f"(lifted={'YES' if lifted else 'no'})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--hold-ticks", type=int, default=300,
                        help="after close finishes, hold that exact frozen target this many "
                             "more ticks instead of continuing to the scripted lift -- 0 "
                             "restores the old behavior (continue to lift, report max_cube_z "
                             "as a real episode outcome)")
    parser.add_argument("--seed", type=int, default=42,
                        help="seeds the cube-spawn RNG so different --stiffness-scale runs "
                             "see the SAME spawn sequence -- required for a paired comparison, "
                             "since the hold-plateau already varies +32 to -82mm across "
                             "otherwise-identical episodes")
    parser.add_argument("--stiffness-scale", type=float, default=1.0,
                        help="scale factor on the target joints' drive stiffness, applied "
                             "once before any episode runs (not per-episode)")
    parser.add_argument("--damping-scale", type=float, default=1.0,
                        help="scale factor on the target joints' drive damping, same timing "
                             "as --stiffness-scale")
    parser.add_argument("--joints", choices=["all", "wrist"], default="all",
                        help="'all' scales all 6 arm joints uniformly; 'wrist' scales only "
                             "the last 2 (closest to the tool, see WRIST_JOINT_NAMES)")
    parser.add_argument("--static-friction", type=float, default=None,
                        help="override the shared gripper-pad/cube physics material's static "
                             "friction (isaac_sim_common.bind_grip_friction_material default is "
                             "0.9) for this run only -- None leaves whatever is already authored")
    parser.add_argument("--dynamic-friction", type=float, default=None,
                        help="same as --static-friction but for dynamic friction "
                             "(isaac_sim_common default 0.7)")
    parser.add_argument("--finger-kp", type=float, default=None,
                        help="override GripperController's as-shipped-fixed finger_joint "
                             "stiffness (baseline 20000.0, live-controller/radian units). "
                             "None leaves the baseline in place. Pair with --finger-kd to hold "
                             "the damping ratio constant -- see this file's own module "
                             "docstring for the sqrt-scaling relationship, and pass both or "
                             "neither, since one without the other lets the ratio drift.")
    parser.add_argument("--finger-kd", type=float, default=None,
                        help="override GripperController's as-shipped-fixed finger_joint "
                             "damping (baseline 500.0). See --finger-kp.")
    args = parser.parse_args()
    try:
        run(args.episodes, args.hold_ticks, args.seed, args.stiffness_scale,
            args.damping_scale, args.joints, args.static_friction, args.dynamic_friction,
            args.finger_kp, args.finger_kd)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
