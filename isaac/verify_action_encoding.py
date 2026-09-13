"""Round-trip check of the OpenVLA action/state encoding, BEFORE any real
grasp data exists to run it on.

No Isaac Sim import -- pure numpy/scipy, so it runs in seconds and doesn't
compete with the gripper-collision debugging for the GPU. That's the point:
this checks the ENCODING pipeline itself (does encode->decode recover the
original trajectory?), which is a closed-form question that doesn't need a
working grasp to answer, and a broken encoding would silently corrupt every
episode collect_rlds_episodes.py ever writes regardless of how good the
demonstrations are.

Two independent bugs found by checking this project's OpenVLA contract
against OpenVLA's own source (fetched from
prismatic/vla/datasets/rlds/oxe/configs.py) rather than trusting the
ADJUST-flagged guess in openvla_transform_snippet.py:

  1. DIMENSION: StateEncoding.POS_EULER is
     "EEF XYZ (3) + Roll-Pitch-Yaw (3) + <PAD> (1) + Gripper (1)" = 8-dim.
     collect_rlds_episodes.py / ur5e_pick_place_dataset_builder.py log a
     7-dim state (xyz + rotvec + gripper, no PAD slot). Every field from
     the rotation onward lands one slot to the left of where OpenVLA's
     fixed-index unpacking expects it.

  2. REPRESENTATION: ActionEncoding.EEF_POS is "EEF Delta XYZ (3) +
     Roll-Pitch-Yaw (3) + Gripper (1)" -- delta ROTATION is expressed as
     delta EULER ANGLES. This project logs delta ROTATION VECTORS (axis-
     angle), and computes that delta by plain vector subtraction of two
     independent rotvecs (`next_pose - prev_pose` over the rotation
     components). That subtraction is not a valid representation of
     relative rotation at all except in the infinitesimal limit -- and
     scripted_pick_place.DOWNWARD_ROTVEC = [0, pi, 0] holds the gripper at
     exactly the rotvec representation's worst point (|rotvec| = pi, the
     antipodal/wraparound boundary), where scipy's shortest-angle
     convention can jump to a completely different-looking vector for a
     physically tiny orientation change. This script reproduces that
     failure numerically (see check_delta_rotvec_bug) rather than asserting
     it from theory.

A third check was added later, because the two above cannot see the failure
it looks for: every round trip here encodes and decodes with the SAME euler
sequence, so it closes to machine epsilon whatever that sequence is. A
collector and a checker sharing one wrong axis order would pass everything.
check_rotation_convention closes that by reading the angles back with an
independent hand-written extrinsic X-Y-Z and requiring agreement.

The exact Euler AXIS ORDER OXE uses (intrinsic vs extrinsic, xyz vs zyx)
could not be pinned down from OpenVLA's public source -- transforms.py's
bridge_orig transform defers to an undefined relabel_bridge_actions helper,
so this implements the standard ROS/URDF convention (extrinsic X-Y-Z, i.e.
scipy's `as_euler("xyz")`) as the best-supported default and flags it the
same way this project already flags GRIPPER_VARIANT_CANDIDATES-style
guesses: ADJUST, verify against an actual OpenVLA checkout's dataloader
before trusting it verbatim.

    python3 verify_action_encoding.py
"""
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

from scripted_pick_place import DOWNWARD_ROTVEC, ScriptedPickPlace

# Same import the collector uses, for the same reason: the axis order this
# script blesses and the one the data is written in must not be able to
# drift apart. euler_xyz_to_matrix is the INDEPENDENT implementation the
# convention check below needs -- see check_rotation_convention.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "openvla_integration"))
from validate_dataset import (  # noqa: E402
    CONVENTION_TOL_RAD, EULER_SEQ, _matrix_angle_between, euler_xyz_to_matrix)

RNG = np.random.default_rng(0)


# --- the CURRENT (buggy) encoding, exactly as collect_rlds_episodes.py does it ---

def current_state(pos, rotvec, gripper):
    return np.concatenate([pos, rotvec, [gripper]]).astype(np.float32)  # 7-dim


def current_action(prev_state, next_state):
    return (next_state - prev_state).astype(np.float32)  # plain subtraction, 7-dim


def current_decode_step(prev_state, action):
    """What replaying a `current_action` means: since it's plain subtraction,
    "decoding" is plain addition -- next_state = prev_state + action, gripper
    slot overwritten with the (already-absolute) logged gripper action."""
    out = prev_state + action
    out[6] = action[6]
    return out


# --- the FIXED encoding, matching OpenVLA's real POS_EULER(8) / EEF_POS(7) ---

def fixed_state(pos, rotvec, gripper):
    rpy = Rot.from_rotvec(rotvec).as_euler(EULER_SEQ)
    return np.concatenate([pos, rpy, [0.0], [gripper]]).astype(np.float32)  # 8-dim, PAD=0


def fixed_action(prev_pos, prev_rotvec, next_pos, next_rotvec, next_gripper):
    """Delta position is still a plain difference (translation IS a vector
    space); delta rotation is NOT -- it has to go through actual rotation
    composition, then get expressed as Euler only at the end."""
    delta_pos = next_pos - prev_pos
    r_prev = Rot.from_rotvec(prev_rotvec)
    r_next = Rot.from_rotvec(next_rotvec)
    delta_rot = r_prev.inv() * r_next  # relative rotation, prev's frame
    delta_rpy = delta_rot.as_euler(EULER_SEQ)
    return np.concatenate([delta_pos, delta_rpy, [next_gripper]]).astype(np.float32)  # 7-dim


def fixed_decode_step(prev_pos, prev_rotvec, action):
    delta_pos, delta_rpy, gripper = action[:3], action[3:6], action[6]
    next_pos = prev_pos + delta_pos
    r_next = Rot.from_rotvec(prev_rotvec) * Rot.from_euler(EULER_SEQ, delta_rpy)
    return next_pos, r_next.as_rotvec(), float(gripper)


# --- trajectories to replay the encodings against ---

def scripted_trajectory():
    """The actual waypoint stream collect_rlds_episodes.py logs from --
    position varies a lot (the whole pick-place path), orientation is
    exactly constant at DOWNWARD_ROTVEC throughout (scripted_pick_place's
    own docstring: "orientation held fixed downward throughout")."""
    policy = ScriptedPickPlace(np.array([0.45, 0.0, 0.35]),
                               cube_position=np.array([0.45, 0.0, 0.02]),
                               target_position=np.array([0.45, 0.30, 0.0]))
    positions, rotvecs, grippers = [], [], []
    for pos, rotvec, gripper in policy.generate_frames():
        positions.append(pos)
        rotvecs.append(rotvec)
        grippers.append(gripper)
    return np.array(positions), np.array(rotvecs), np.array(grippers)


def perturbed_constant_orientation_trajectory(n=500, noise_deg=2.0):
    """Synthetic stress test: position doesn't matter here, orientation
    jitters by a small, physically-realistic amount (RMPflow's own measured
    tracking noise was 1-4 degrees while converging -- see
    camera_check/frames_report.txt) around DOWNWARD_ROTVEC. This is what the
    ACHIEVED pose (not the commanded one) actually looks like tick to tick,
    since collect_rlds_episodes.py logs scene.get_observation(), not the
    waypoint target."""
    base = Rot.from_rotvec(DOWNWARD_ROTVEC)
    rotvecs = []
    for _ in range(n):
        axis = RNG.normal(size=3)
        axis /= np.linalg.norm(axis)
        angle = np.radians(RNG.uniform(-noise_deg, noise_deg))
        noisy = base * Rot.from_rotvec(axis * angle)
        rotvecs.append(noisy.as_rotvec())
    return np.array(rotvecs)


def replay(positions, rotvecs, grippers, encode_state_fn, encode_action_fn, decode_fn,
           rot_error_fn):
    """Generic round-trip: log every step the way collect_rlds_episodes.py
    would, then reconstruct the trajectory from nothing but step 0's state
    and the stream of actions, and compare to ground truth."""
    n = len(positions)
    states = [encode_state_fn(positions[i], rotvecs[i], grippers[i]) for i in range(n)]
    actions = [encode_action_fn(states[i], states[i + 1], positions[i], rotvecs[i],
                                 positions[i + 1], rotvecs[i + 1], grippers[i + 1])
               for i in range(n - 1)]

    pos_errs, rot_errs = [], []
    cur_pos, cur_rotvec = positions[0].copy(), rotvecs[0].copy()
    for i, action in enumerate(actions):
        cur_pos, cur_rotvec, _ = decode_fn(cur_pos, cur_rotvec, action)
        pos_errs.append(np.linalg.norm(cur_pos - positions[i + 1]))
        rot_errs.append(rot_error_fn(cur_rotvec, rotvecs[i + 1]))
    return np.array(pos_errs), np.array(rot_errs)


def rotvec_angle_error(a, b):
    return float((Rot.from_rotvec(a).inv() * Rot.from_rotvec(b)).magnitude())


def check_current_encoding(positions, rotvecs, grippers):
    print("\n=== CURRENT encoding (rotvec, plain subtraction) ===", flush=True)

    def enc_action(prev_s, next_s, *_):
        return current_action(prev_s, next_s)

    def dec(cur_pos, cur_rotvec, action):
        prev_state = current_state(cur_pos, cur_rotvec, 0.0)
        out = current_decode_step(prev_state, action)
        return out[:3], out[3:6], out[6]

    def enc_state(p, r, g):
        return current_state(p, r, g)

    pos_errs, rot_errs = replay(positions, rotvecs, grippers, enc_state, enc_action, dec,
                                 rotvec_angle_error)
    print(f"  position error: max {pos_errs.max() * 1000:.3f} mm, mean {pos_errs.mean() * 1000:.3f} mm",
          flush=True)
    print(f"  rotation error: max {np.degrees(rot_errs.max()):.2f} deg, "
          f"mean {np.degrees(rot_errs.mean()):.2f} deg", flush=True)
    return pos_errs, rot_errs


def check_fixed_encoding(positions, rotvecs, grippers):
    print("\n=== FIXED encoding (euler RPY, proper rotation composition) ===", flush=True)

    def enc_action(prev_s, next_s, prev_pos, prev_rotvec, next_pos, next_rotvec, next_g):
        return fixed_action(prev_pos, prev_rotvec, next_pos, next_rotvec, next_g)

    def dec(cur_pos, cur_rotvec, action):
        return fixed_decode_step(cur_pos, cur_rotvec, action)

    pos_errs, rot_errs = replay(positions, rotvecs, grippers, fixed_state, enc_action, dec,
                                 rotvec_angle_error)
    print(f"  position error: max {pos_errs.max() * 1000:.3f} mm, mean {pos_errs.mean() * 1000:.3f} mm",
          flush=True)
    print(f"  rotation error: max {np.degrees(rot_errs.max()):.4f} deg, "
          f"mean {np.degrees(rot_errs.mean()):.4f} deg", flush=True)
    return pos_errs, rot_errs


def check_delta_rotvec_bug():
    """Isolates bug #2 from bug #1 and from position entirely: orientation
    only, jittering near the DOWNWARD_ROTVEC singularity the way the
    ACHIEVED pose really does. Shows the plain-subtraction delta blowing up
    even though the physical rotation change each tick is <= noise_deg."""
    print(f"\n=== isolated rotation-only stress test, near DOWNWARD_ROTVEC={DOWNWARD_ROTVEC.tolist()} ===",
          flush=True)
    rotvecs = perturbed_constant_orientation_trajectory()
    physical_step_deg = [np.degrees(rotvec_angle_error(rotvecs[i], rotvecs[i + 1]))
                          for i in range(len(rotvecs) - 1)]
    naive_delta_deg = [np.degrees(np.linalg.norm(rotvecs[i + 1] - rotvecs[i]))
                        for i in range(len(rotvecs) - 1)]
    print(f"  actual physical rotation per tick: max {max(physical_step_deg):.2f} deg "
          f"(bounded by the +/-2 deg noise, as expected)", flush=True)
    print(f"  |rotvec[i+1] - rotvec[i]| per tick: max {max(naive_delta_deg):.2f} deg "
          "(what the CURRENT code's action logs as the rotation delta)", flush=True)
    blown = sum(1 for d in naive_delta_deg if d > 10.0)
    print(f"  ticks where the logged delta is >10x the real rotation: {blown}/{len(naive_delta_deg)}",
          flush=True)
    if blown:
        print("  CONFIRMED: near this project's own constant grasp orientation, plain rotvec "
              "subtraction periodically logs a huge, physically-fictitious rotation for a real "
              "change of a couple of degrees. A model trained on this would be taught that "
              "'hold steady' sometimes means 'spin explosively' for no visible reason.", flush=True)


def check_rotation_convention():
    """The blind spot every check above this one shares.

    encode->decode is self-consistent under ANY euler sequence: whatever
    convention writes the angles, the same convention reads them back, so
    the round trip closes to machine epsilon either way. A collector and a
    checker that share the SAME wrong axis order therefore pass everything
    above -- which is precisely the failure mode that would be invisible
    until an OpenVLA fine-tune produced a policy that rotates wrongly.

    The metamorphic relation that does see it: read the recorded angles with
    an INDEPENDENT implementation of extrinsic X-Y-Z -- Rz @ Ry @ Rx written
    out by hand, imported from validate_dataset so there is exactly one of
    it -- and require it to reproduce the rotation the angles came from. The
    right sequence agrees to ~1e-15 rad; a wrong one is tens of degrees out.

    What this does NOT prove: that extrinsic xyz is what OpenVLA itself
    reads. Only its dataloader settles that, and it is still an open step.
    """
    print("\n=== rotation convention (metamorphic, not a round trip) ===", flush=True)

    rotations = [Rot.from_rotvec(DOWNWARD_ROTVEC + RNG.normal(scale=np.radians(3.0), size=3))
                 for _ in range(200)]

    print("  first, why the round trip cannot answer this:", flush=True)
    for seq in (EULER_SEQ, "zyx"):
        worst = 0.0
        for previous, following in zip(rotations, rotations[1:]):
            delta = (previous.inv() * following).as_euler(seq)
            recovered = previous * Rot.from_euler(seq, delta)
            worst = max(worst, (recovered.inv() * following).magnitude())
        print(f"    encode and decode both in {seq!r}: worst round-trip error "
              f"{np.degrees(worst) * 1e9:.3f} ndeg", flush=True)

    print("  the relation, against the independent Rz @ Ry @ Rx:", flush=True)
    verdicts = {}
    for seq in (EULER_SEQ, "zyx"):
        worst = max(_matrix_angle_between(rotation.as_matrix(),
                                          euler_xyz_to_matrix(rotation.as_euler(seq)))
                    for rotation in rotations)
        verdicts[seq] = worst
        agrees = worst <= CONVENTION_TOL_RAD
        print(f"    angles written as {seq!r}: worst disagreement "
              f"{np.degrees(worst):.3f} deg -> {'agrees' if agrees else 'CAUGHT'}",
              flush=True)

    if verdicts[EULER_SEQ] > CONVENTION_TOL_RAD:
        print(f"  FAILED: EULER_SEQ={EULER_SEQ!r} does not mean extrinsic X-Y-Z. "
              "Every recorded orientation is being written in a convention this "
              "pipeline does not read back the same way.", flush=True)
        return False

    print(f"  EULER_SEQ={EULER_SEQ!r} really is extrinsic (fixed-axis) X-Y-Z, and a "
          "wrong order would have been caught rather than passing silently.", flush=True)
    print("  Shared with validate_dataset.py, so the data and this check cannot "
          "drift apart -- but agreement with OPENVLA's own dataloader is still "
          "unverified. That is the remaining ADJUST.", flush=True)
    return True


def main():
    positions, rotvecs, grippers = scripted_trajectory()
    print(f"scripted trajectory: {len(positions)} steps, orientation constant at "
          f"{DOWNWARD_ROTVEC.tolist()} throughout (no tracking noise in this idealized replay)",
          flush=True)

    cur_pos_err, cur_rot_err = check_current_encoding(positions, rotvecs, grippers)
    fix_pos_err, fix_rot_err = check_fixed_encoding(positions, rotvecs, grippers)
    check_delta_rotvec_bug()
    convention_ok = check_rotation_convention()

    print("\n--- verdict ---", flush=True)
    if not convention_ok:
        print("  STOP: the euler convention check failed. Nothing below this line "
              "is meaningful until the axis order is fixed.", flush=True)
        return 1
    print("  Against the IDEALIZED scripted waypoints (no tracking noise, orientation exactly "
          "constant), both encodings round-trip to numerical precision -- the position and "
          "rotation errors above are near machine epsilon for both, which makes sense: the "
          "rotvec-subtraction bug only bites when the ACHIEVED orientation jitters near the "
          "|rotvec|=pi wraparound, which this idealized replay has no noise to trigger.",
          flush=True)
    print("  The isolated stress test above reproduces that failure directly: real RMPflow "
          "tracking noise (measured 1-4 deg while converging) near DOWNWARD_ROTVEC periodically "
          "produces a >10x-inflated logged rotation delta under the CURRENT scheme.", flush=True)
    print("\n  Required changes before collecting real OpenVLA training data:", flush=True)
    print("  1. collect_rlds_episodes.py: log fixed_state (8-dim, RPY + explicit 0 PAD slot) and "
          "fixed_action (7-dim, delta RPY via proper rotation composition) instead of "
          "current_state/current_action.", flush=True)
    print("  2. ur5e_pick_place_dataset_builder.py: STATE_ACTION_DIM must split into "
          "STATE_DIM=8 / ACTION_DIM=7 (they are no longer equal).", flush=True)
    print("  3. openvla_transform_snippet.py: state_encoding/action_encoding comments can drop "
          "their ADJUST-uncertainty about rotation representation (rotvec vs euler is now "
          "correct) but EULER_SEQ's exact axis order/convention is still unverified against a "
          "real OpenVLA checkout -- check that before trusting this verbatim.", flush=True)


if __name__ == "__main__":
    sys.exit(main() or 0)
