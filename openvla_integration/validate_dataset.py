"""Pre-training data quality gate for the OpenVLA path.

Run this over `raw_episodes/` BEFORE building the TFDS dataset or starting a
fine-tune. It imports nothing from Isaac Sim, so it runs anywhere -- on the
collection machine, on a laptop, in CI.

WHY THIS EXISTS

Every expensive failure in this project so far has been a dataset that was
*perfectly well-formed and completely useless*, and none of it was visible
without looking at the numbers:

  * A 100-episode run produced 21,000 frames in which the cube appeared
    exactly zero times -- the camera's near-clip plane defaulted to 1.0m
    while the base camera sat 0.66m away and the wrist camera a few cm away,
    so the wrist frames were solid black and the base frames held no red
    pixels at all. Shapes, dtypes and step counts were all correct.
  * The rotational part of the action was computed by subtracting rotation
    vectors, which is only valid in the infinitesimal limit. The scripted
    trajectory holds the tool at |rotvec| = pi, the representation's
    antipodal wraparound, so 243/499 sampled deltas came out more than 10x
    their true physical size. Again: correct shape, correct dtype, finite
    numbers, entirely wrong content.
  * The state vector was 7-dim when OpenVLA's StateEncoding.POS_EULER is 8
    (it has a PAD slot), so every field from the rotation onward would have
    been read one slot to the left.

collect_demos.py already refuses to collect a bad dataset on the pi0/LeRobot
side (scene preflight, per-episode lift/place verification, an abort if the
cube is never visible). collect_rlds_episodes.py on this side has none of
that and saves whatever it recorded, so this gate is where those checks
live for the OpenVLA path.

WHAT IT CHECKS

The strongest checks are the three exact identities the encoding must
satisfy. collect_rlds_episodes._state_vec / ._delta_action define
state as [xyz(3), rpy(3), PAD(1)=0, gripper(1)] and action as
[delta_xyz(3), delta_rpy(3), next_gripper(1)], with the action carrying
step t to step t+1. That means, for correctly logged data:

    state[t+1][:3]      == state[t][:3] + action[t][:3]
    R(state[t+1][3:6])  == R(state[t][3:6]) @ R(action[t][3:6])
    state[t+1][7]       == action[t][6]

These hold exactly, not approximately, and they are what a fine-tune
silently relies on. A subtraction where a composition belongs, an
off-by-one in the logging loop, or a stale observation all break at least
one of them.

THE BLIND SPOT THOSE IDENTITIES HAVE, AND WHAT CLOSES IT

A wrong Euler AXIS ORDER does not break them. The identities are evaluated
in the same convention the collector wrote, so if collector and gate share
a misreading of what "xyz" means, the data is self-consistently wrong and
every check passes. That is the classic test oracle problem: a checker that
shares the assumptions of the thing it checks cannot see errors common to
both.

The standard answer is a metamorphic relation -- a property that must hold
regardless of how the system under test is implemented internally, derived
independently of it. _check_rotation_convention below is that relation
here. It re-implements extrinsic (fixed-axis) XYZ composition from first
principles, R = Rz(yaw) @ Ry(pitch) @ Rx(roll), out of explicit elementary
matrices, and then:

  1. requires scipy's from_euler(EULER_SEQ, ...) to agree with it, which
     fails the moment EULER_SEQ stops meaning extrinsic XYZ -- uppercase
     "XYZ" is scipy's INTRINSIC convention and a different rotation; and
  2. re-checks the composition identity using only those hand-built
     matrices, so the identity is confirmed once by an implementation that
     never routes through scipy's interpretation of an axis string.

Extrinsic lowercase "xyz" is the convention the OpenVLA / Bridge / DROID
stack uses, so that is what this pins. The relation cannot prove the
dataset matches OpenVLA -- only a real checkout can -- but it does prove
the file means what it says it means, which is the part that was
previously unverifiable from inside this project.

USAGE

    python validate_dataset.py                       # check ./raw_episodes
    python validate_dataset.py --episodes-dir PATH
    python validate_dataset.py --quarantine bad/     # move failures aside
    python validate_dataset.py --max-fail-rate 0.0   # any failure is fatal

Exit code is non-zero when the failure rate exceeds --max-fail-rate, so it
can gate a training script or a CI job.
"""
import argparse
import os
import shutil
import sys

import numpy as np
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "isaac"))
from object_configs import COLOR_RGB  # noqa: E402  (isaac/object_configs.py imports only numpy)

# THIS is where the axis order is defined, for the whole repository.
# isaac/collect_rlds_episodes.py (which writes the data) and
# isaac/verify_action_encoding.py (which checks the encoding) both import it
# from here, deliberately: a second copy is a second thing that can disagree,
# and a disagreement about the axis order would not show up in any round trip.
# Resolve it against a real OpenVLA checkout and change it here only.
EULER_SEQ = "xyz"

# scipy reads a LOWERCASE axis string as extrinsic (fixed-axis) and an
# UPPERCASE one as intrinsic (body-fixed). Those are different rotations for
# the same three numbers, and the OpenVLA / Bridge / DROID stack this dataset
# targets is extrinsic. Nothing downstream would notice the difference --
# hence the check, at import, where it cannot be missed.
if EULER_SEQ != EULER_SEQ.lower():
    raise ValueError(
        f"EULER_SEQ is {EULER_SEQ!r}: scipy reads an uppercase axis string as the "
        f"INTRINSIC convention, but this dataset targets extrinsic (fixed-axis) "
        f"angles. Use {EULER_SEQ.lower()!r}.")

STATE_DIM = 8   # OpenVLA StateEncoding.POS_EULER
ACTION_DIM = 7  # OpenVLA ActionEncoding.EEF_POS
PAD_INDEX = 6   # the POS_EULER pad slot, which must stay zero

# A frame is "black" when even its brightest pixel is this dark. The lost run
# produced wrist frames that were solid zero, so this is deliberately a floor
# on the MAXIMUM, not on the mean: a dim-but-real frame passes, a clipped one
# does not.
BLACK_FRAME_MAX_VALUE = 8
MAX_BLACK_FRAME_FRACTION = 0.02

# Channel margin (0-255) for counting pixels of the target's colour. Taken
# from pick_place_scene.cube_pixels_visible, which uses exactly this test for
# red and was tuned against real renders under the scene's blue-ish ambient
# light.
COLOR_DOMINANCE_MARGIN = 40
MIN_TARGET_PIXELS_PEAK = 50

# Tolerances for the three encoding identities. float32 storage of values
# around 1 rad gives ~1e-7 of representation error, and the rotation identity
# additionally round-trips through Euler angles, so 1e-4 is loose enough to
# never fire on correct data and far tighter than any real encoding bug.
CONSISTENCY_POS_TOL_M = 1e-4
CONSISTENCY_ROT_TOL_RAD = 1e-4
CONSISTENCY_GRIPPER_TOL = 1e-4
MAX_INCONSISTENT_FRACTION = 0.0

# Physical plausibility of a single control step. The scripted expert moves
# in small increments; a delta far above this means the action encoding is
# inflating deltas (the rotvec-subtraction bug produced exactly this).
MAX_PLAUSIBLE_STEP_TRANSLATION_M = 0.10
MAX_PLAUSIBLE_STEP_ROTATION_RAD = 0.80

# How far the library's reading of EULER_SEQ may sit from the hand-built
# extrinsic-XYZ composition before the convention is called wrong. A genuine
# convention mismatch is degrees to tens of degrees apart, so this only has
# to clear float noise.
CONVENTION_TOL_RAD = 1e-6

MIN_STEPS_PER_EPISODE = 10
MIN_EPISODE_PATH_LENGTH_M = 0.05


def color_pixel_count(image, color_name, margin=COLOR_DOMINANCE_MARGIN):
    """Pixels whose colour is dominated by `color_name`'s channels.

    Generalises pick_place_scene.cube_pixels_visible from red to the whole
    object vocabulary: split the target colour's channels into "high" and
    "low", then count pixels where the weakest high channel beats the
    strongest low channel by `margin`. For red that reduces to exactly
    cube_pixels_visible's `r - max(g, b) > 40`; for yellow it becomes
    `min(r, g) - b > 40`, which a per-channel threshold could not express.
    """
    img = np.asarray(image)
    if img.size == 0 or img.ndim != 3 or img.shape[-1] < 3:
        return 0

    rgb = img[..., :3].astype(np.int16)
    target = np.asarray(COLOR_RGB[color_name], dtype=float)
    high = target >= 0.5 * target.max()
    if high.all():          # a grey/white target has no dominant channel
        return 0

    high_channels = rgb[..., high].min(axis=-1)
    low_channels = rgb[..., ~high].max(axis=-1)
    return int(((high_channels - low_channels) > margin).sum())


def target_color_from_instruction(instruction):
    """The colour word in "pick up the a red cube and place it ...", or None.

    The instruction is the only per-episode record of which object was the
    target, so it is also the only way to know which colour has to be
    visible.
    """
    if not isinstance(instruction, str):
        return None
    words = set(instruction.lower().replace(".", " ").replace(",", " ").split())
    found = [name for name in COLOR_RGB if name in words]
    return found[0] if len(found) == 1 else None


def _check_schema(steps, problems):
    """Shapes, dtypes and the POS_EULER pad slot."""
    for i, step in enumerate(steps):
        for key in ("image", "state", "action", "language_instruction"):
            if key not in step:
                problems.append(f"step {i}: missing key {key!r}")
                return False

    states = np.asarray([s["state"] for s in steps], dtype=float)
    actions = np.asarray([s["action"] for s in steps], dtype=float)

    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        problems.append(
            f"state has shape {states.shape}, expected (N, {STATE_DIM}) -- OpenVLA's "
            f"StateEncoding.POS_EULER unpacks by fixed index, so a short vector shifts "
            f"every field from the rotation onward one slot left")
        return False
    if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
        problems.append(f"action has shape {actions.shape}, expected (N, {ACTION_DIM})")
        return False

    if not np.all(np.isfinite(states)):
        problems.append(f"state holds {int((~np.isfinite(states)).sum())} non-finite values")
    if not np.all(np.isfinite(actions)):
        problems.append(f"action holds {int((~np.isfinite(actions)).sum())} non-finite values")

    pad = states[:, PAD_INDEX]
    if not np.allclose(pad, 0.0, atol=1e-6):
        problems.append(
            f"state[{PAD_INDEX}] (the POS_EULER PAD slot) is not zero "
            f"(max |value| = {np.abs(pad).max():.4g})")
    return True


def _check_frames(steps, problems):
    """Black or frozen camera frames -- the failure that cost 100 episodes."""
    images = [np.asarray(s["image"]) for s in steps]
    black = sum(1 for img in images if img.size == 0 or int(img.max()) <= BLACK_FRAME_MAX_VALUE)
    black_fraction = black / len(images)
    if black_fraction > MAX_BLACK_FRAME_FRACTION:
        problems.append(
            f"{black}/{len(images)} frames ({black_fraction:.0%}) are black "
            f"(max pixel <= {BLACK_FRAME_MAX_VALUE}). Check the camera near-clip plane: "
            f"the default 1.0m clips everything closer than that, which is every wrist "
            f"frame and most base frames in this scene")

    constant = sum(1 for img in images if img.size and int(img.max()) == int(img.min()))
    if constant > len(images) * MAX_BLACK_FRAME_FRACTION and constant != black:
        problems.append(f"{constant}/{len(images)} frames are a single flat colour")


def _check_target_visible(steps, problems):
    """The target object has to actually appear in the observations."""
    instruction = steps[0].get("language_instruction")
    color = target_color_from_instruction(instruction)
    if color is None:
        problems.append(
            f"could not tell the target colour from the instruction {instruction!r}, "
            f"so target visibility could not be checked")
        return

    peak = max(color_pixel_count(s["image"], color) for s in steps)
    if peak < MIN_TARGET_PIXELS_PEAK:
        problems.append(
            f"the {color} target peaked at {peak} pixels across the whole episode "
            f"(need >= {MIN_TARGET_PIXELS_PEAK}). The task is not visible in the "
            f"observations, so no amount of data will help")


def _check_encoding_consistency(steps, problems):
    """The three exact identities linking state[t], action[t] and state[t+1].

    This is the check that would have caught the rotation-delta bug at
    collection time instead of after training, and it is equally sensitive to
    a wrong Euler axis order or an off-by-one in the logging loop.
    """
    states = np.asarray([s["state"] for s in steps], dtype=float)
    actions = np.asarray([s["action"] for s in steps], dtype=float)
    n_links = len(steps) - 1
    if n_links < 1:
        return

    pos_err = np.linalg.norm(
        (states[:-1, :3] + actions[:-1, :3]) - states[1:, :3], axis=1)

    r_from = Rot.from_euler(EULER_SEQ, states[:-1, 3:6])
    r_delta = Rot.from_euler(EULER_SEQ, actions[:-1, 3:6])
    r_to = Rot.from_euler(EULER_SEQ, states[1:, 3:6])
    rot_err = (r_from * r_delta * r_to.inv()).magnitude()

    grip_err = np.abs(actions[:-1, 6] - states[1:, 7])

    bad = ((pos_err > CONSISTENCY_POS_TOL_M)
           | (rot_err > CONSISTENCY_ROT_TOL_RAD)
           | (grip_err > CONSISTENCY_GRIPPER_TOL))
    bad_fraction = float(bad.mean())
    if bad_fraction > MAX_INCONSISTENT_FRACTION:
        problems.append(
            f"{int(bad.sum())}/{n_links} transitions do not satisfy "
            f"state[t+1] == state[t] (+) action[t]: worst position error "
            f"{pos_err.max() * 1000:.2f}mm, worst rotation error "
            f"{np.degrees(rot_err.max()):.3f}deg, worst gripper error "
            f"{grip_err.max():.4g}. The recorded action does not carry the recorded "
            f"state to the next one, so the policy is being trained on a relationship "
            f"that is not in the data")


def _rot_x(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rot_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _rot_z(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def euler_xyz_to_matrix(rpy):
    """Extrinsic (fixed-axis) XYZ Euler angles -> rotation matrix.

    Deliberately written out of elementary matrices rather than delegating
    to scipy: this is the independent implementation the convention check
    needs, and its whole value is that it does NOT consult the same axis
    string the collector passed. Applying roll about fixed X, then pitch
    about fixed Y, then yaw about fixed Z composes on the left, so the
    product is Rz @ Ry @ Rx.
    """
    roll, pitch, yaw = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    return _rot_z(yaw) @ _rot_y(pitch) @ _rot_x(roll)


def _matrix_angle_between(a, b):
    """Geodesic angle between two rotation matrices, in radians.

    Computed from the Frobenius distance rather than the more familiar
    arccos((trace - 1) / 2), because this check has to resolve angles near
    zero and arccos is ill-conditioned exactly there: its argument
    approaches 1, where a float64 rounding error of eps becomes an angle
    error of sqrt(eps), about 1e-8 rad. For rotations
    ||A - B||_F = 2*sqrt(2)*|sin(theta/2)|, and arcsin near zero is well
    conditioned, so this resolves agreement down to float64 precision
    instead of bottoming out eight orders above it.
    """
    frobenius = float(np.linalg.norm(np.asarray(a) - np.asarray(b)))
    return 2.0 * float(np.arcsin(np.clip(frobenius / (2.0 * np.sqrt(2.0)), 0.0, 1.0)))


def _check_rotation_convention(steps, problems, sample=24):
    """Metamorphic relation: the recorded angles must mean extrinsic XYZ.

    See the module docstring. The three consistency identities cannot see a
    wrong axis order, because they are evaluated in the same convention the
    data was written in. This re-derives the rotations from first
    principles instead, which is independent of how EULER_SEQ is read.
    """
    states = np.asarray([s["state"] for s in steps], dtype=float)
    actions = np.asarray([s["action"] for s in steps], dtype=float)

    # Sample rather than sweep: the convention is a property of the code
    # that wrote the file, so it is either right for every row or wrong for
    # every row, and a couple of dozen rows settle it.
    idx = np.unique(np.linspace(0, len(steps) - 1, min(sample, len(steps))).astype(int))

    worst = 0.0
    for i in idx:
        for angles in (states[i, 3:6], actions[i, 3:6]):
            library = Rot.from_euler(EULER_SEQ, angles).as_matrix()
            independent = euler_xyz_to_matrix(angles)
            worst = max(worst, _matrix_angle_between(library, independent))

    if worst > CONVENTION_TOL_RAD:
        problems.append(
            f"the recorded angles do not mean extrinsic (fixed-axis) XYZ: reading them "
            f"with EULER_SEQ={EULER_SEQ!r} disagrees with an independent extrinsic-XYZ "
            f"composition by up to {np.degrees(worst):.3f}deg. OpenVLA/Bridge/DROID read "
            f"these as extrinsic xyz, so the dataset would be interpreted as a different "
            f"rotation than the one that was recorded")
        return

    # Re-check the composition identity using ONLY the independent
    # implementation, so it is confirmed once without scipy's axis-string
    # interpretation anywhere in the path.
    worst_identity = 0.0
    for i in idx[idx < len(steps) - 1]:
        m_from = euler_xyz_to_matrix(states[i, 3:6])
        m_delta = euler_xyz_to_matrix(actions[i, 3:6])
        m_to = euler_xyz_to_matrix(states[i + 1, 3:6])
        worst_identity = max(worst_identity, _matrix_angle_between(m_from @ m_delta, m_to))

    if worst_identity > CONSISTENCY_ROT_TOL_RAD:
        problems.append(
            f"the composition identity fails under an independent extrinsic-XYZ "
            f"implementation (worst {np.degrees(worst_identity):.4f}deg), even where the "
            f"library-based check passed -- the two readings of the data disagree")


def _check_plausibility(steps, problems):
    """Per-step deltas that are too large to be real motion."""
    actions = np.asarray([s["action"] for s in steps], dtype=float)
    translation = np.linalg.norm(actions[:, :3], axis=1)
    rotation = Rot.from_euler(EULER_SEQ, actions[:, 3:6]).magnitude()

    n_big_t = int((translation > MAX_PLAUSIBLE_STEP_TRANSLATION_M).sum())
    if n_big_t:
        problems.append(
            f"{n_big_t}/{len(steps)} steps translate more than "
            f"{MAX_PLAUSIBLE_STEP_TRANSLATION_M * 100:.0f}cm in one control step "
            f"(max {translation.max() * 100:.1f}cm)")

    n_big_r = int((rotation > MAX_PLAUSIBLE_STEP_ROTATION_RAD).sum())
    if n_big_r:
        problems.append(
            f"{n_big_r}/{len(steps)} steps rotate more than "
            f"{np.degrees(MAX_PLAUSIBLE_STEP_ROTATION_RAD):.0f}deg in one control step "
            f"(max {np.degrees(rotation.max()):.0f}deg). Inflated rotation deltas are the "
            f"signature of subtracting rotation representations instead of composing them")


def _check_task_content(steps, problems):
    """An episode that never moves or never closes the gripper demonstrates
    nothing, however well-formed it is."""
    states = np.asarray([s["state"] for s in steps], dtype=float)

    path_length = float(np.sum(np.linalg.norm(np.diff(states[:, :3], axis=0), axis=1)))
    if path_length < MIN_EPISODE_PATH_LENGTH_M:
        problems.append(
            f"the end effector travelled {path_length * 100:.1f}cm over the whole episode "
            f"(need >= {MIN_EPISODE_PATH_LENGTH_M * 100:.0f}cm)")

    gripper = states[:, 7]
    if float(gripper.max() - gripper.min()) < 1e-3:
        problems.append(
            f"the gripper never moves (constant at {gripper[0]:.4g}) -- no grasp was "
            f"attempted, so this episode cannot demonstrate picking anything up")


def validate_episode(steps):
    """Returns (ok, problems) for one loaded episode."""
    problems = []
    if not isinstance(steps, (list, tuple, np.ndarray)) or len(steps) == 0:
        return False, ["episode is empty"]
    steps = list(steps)

    if len(steps) < MIN_STEPS_PER_EPISODE:
        problems.append(f"only {len(steps)} steps (need >= {MIN_STEPS_PER_EPISODE})")

    if not _check_schema(steps, problems):
        return False, problems

    _check_frames(steps, problems)
    _check_target_visible(steps, problems)
    _check_encoding_consistency(steps, problems)
    _check_rotation_convention(steps, problems)
    _check_plausibility(steps, problems)
    _check_task_content(steps, problems)
    return len(problems) == 0, problems


def load_episode(path):
    return list(np.load(path, allow_pickle=True))


def main():
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_episodes")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--episodes-dir", default=default_dir)
    parser.add_argument("--quarantine", default=None,
                        help="move failing episodes into this directory instead of "
                             "leaving them where a build would pick them up")
    parser.add_argument("--max-fail-rate", type=float, default=0.0,
                        help="exit non-zero once this fraction of episodes fails "
                             "(default 0.0: any failure is fatal)")
    args = parser.parse_args()

    paths = sorted(
        os.path.join(args.episodes_dir, name)
        for name in os.listdir(args.episodes_dir)
        if name.endswith(".npy")
    ) if os.path.isdir(args.episodes_dir) else []

    if not paths:
        print(f"no episodes found in {args.episodes_dir}")
        return 1

    failed = []
    for path in paths:
        name = os.path.basename(path)
        try:
            steps = load_episode(path)
        except Exception as exc:  # noqa: BLE001 -- a corrupt file is a failure like any other
            print(f"FAIL {name}: could not load ({exc})")
            failed.append(path)
            continue

        ok, problems = validate_episode(steps)
        if ok:
            print(f"ok   {name}: {len(steps)} steps")
        else:
            failed.append(path)
            print(f"FAIL {name}: {len(steps)} steps")
            for problem in problems:
                print(f"       - {problem}")

    fail_rate = len(failed) / len(paths)
    print(f"\n{len(paths) - len(failed)}/{len(paths)} episodes passed "
          f"({fail_rate:.0%} failed)")

    if failed and args.quarantine:
        os.makedirs(args.quarantine, exist_ok=True)
        for path in failed:
            shutil.move(path, os.path.join(args.quarantine, os.path.basename(path)))
        print(f"moved {len(failed)} failing episodes to {args.quarantine}")

    if fail_rate > args.max_fail_rate:
        print(f"FAILED: {fail_rate:.0%} exceeds --max-fail-rate {args.max_fail_rate:.0%}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
