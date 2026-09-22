"""Builds synthetic episodes -- one correct, then one reproducing each
failure this project actually hit -- and checks the gate catches exactly the
right thing and nothing else.

Two of these failures cost a full collection run each, and both looked like
success at the time: 21,000 well-formed frames in which the cube never
appeared, and 243 of 499 rotation deltas inflated tenfold with correct
shapes and finite values throughout.
"""
import os
import subprocess
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

from validate_dataset import (
    EULER_SEQ, color_pixel_count, target_color_from_instruction,
    validate_episode)

INSTRUCTION = 'pick up the a red cube and place it in the target zone'
N_STEPS = 40
VALIDATOR = os.path.join(os.path.dirname(__file__), '..',
                         'openvla_integration', 'validate_dataset.py')


@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(11)


def make_image(step_idx=0, with_target=True, black=False, blob=14):
    """step_idx nudges the blob a little each call so a real (non-frozen)
    episode's frames actually differ tick to tick, the way a moving camera's
    would -- validate_dataset.py's own frozen-frame check (2026-09-23) needs
    that to be true of a genuinely good episode's fixture."""
    if black:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    image = np.full((256, 256, 3), (30, 30, 38), dtype=np.uint8)
    if with_target:
        row = 100 + (step_idx % 40)
        image[row:row + blob, 120:120 + blob] = (204, 25, 25)
    return image


def make_poses(rng, n):
    """A plausible tool trajectory: a smooth reach held at the scripted
    expert's downward orientation, which sits at |rotvec| = pi -- the
    representation singularity that broke the original encoding.
    """
    positions, quats, grippers = [], [], []
    base_rotvec = np.array([0.0, np.pi, 0.0])
    for i in range(n):
        t = i / (n - 1)
        positions.append(np.array([0.35 + 0.20 * t, -0.10 + 0.15 * t, 0.40 - 0.18 * t]))
        wobble = rng.normal(scale=np.radians(2.0), size=3)      # RMPflow tracking noise
        quats.append(Rot.from_rotvec(base_rotvec + wobble).as_quat())
        # closes to grasp at t=0.6, releases again at t=0.9 -- a full
        # pick-AND-place, not just a pick (validate_dataset.py's own
        # end-of-episode release check, 2026-09-23, needs a genuinely good
        # episode to actually open the gripper again before it ends).
        grippers.append(0.0 if t < 0.6 else (0.80 if t < 0.9 else 0.0))
    return positions, quats, grippers


def encode(positions, quats, grippers, rotation_delta='compose', inflate=1.0,
           state_dim=8, image_fn=make_image, freeze_gripper=False):
    """Mirrors collect_rlds_episodes._state_vec / ._delta_action, with a hook
    for each historical bug."""
    steps = []
    for i in range(len(positions) - 1):
        rpy = Rot.from_quat(quats[i]).as_euler(EULER_SEQ)
        gripper = grippers[0] if freeze_gripper else grippers[i]
        if state_dim == 8:
            state = np.concatenate([positions[i], rpy, [0.0], [gripper]])
        else:                                       # the original 7-dim mistake
            state = np.concatenate([positions[i], rpy, [gripper]])

        previous, following = Rot.from_quat(quats[i]), Rot.from_quat(quats[i + 1])
        if rotation_delta == 'compose':
            delta_rpy = (previous.inv() * following).as_euler(EULER_SEQ)
        else:                                       # the subtraction bug
            delta_rpy = following.as_euler(EULER_SEQ) - previous.as_euler(EULER_SEQ)

        next_gripper = grippers[0] if freeze_gripper else grippers[i + 1]
        action = np.concatenate([(positions[i + 1] - positions[i]) * inflate,
                                 delta_rpy, [next_gripper]])

        steps.append({'image': image_fn(i),
                      'state': state.astype(np.float32),
                      'action': action.astype(np.float32),
                      'language_instruction': INSTRUCTION})
    return steps


@pytest.fixture(scope='module')
def poses(rng):
    return make_poses(rng, N_STEPS)


@pytest.fixture(scope='module')
def good_episode(poses):
    return encode(*poses)


# --- helpers -------------------------------------------------------------

@pytest.mark.parametrize('instruction,expected', [
    (INSTRUCTION, 'red'),
    ('pick up the a yellow cylinder and place it', 'yellow'),
    ('no colour here', None),
])
def test_the_target_colour_comes_from_the_instruction(instruction, expected):
    assert target_color_from_instruction(instruction) == expected


def test_the_target_blob_is_counted_and_other_colours_are_not():
    image = make_image()
    assert color_pixel_count(image, 'red') == 14 * 14
    assert color_pixel_count(image, 'blue') == 0


def test_a_black_frame_contains_nothing():
    assert color_pixel_count(make_image(black=True), 'red') == 0


def test_a_two_channel_colour_is_not_confused_with_red():
    """The dominance rule has to generalise past single-channel colours."""
    yellow = np.full((16, 16, 3), (217, 191, 25), dtype=np.uint8)
    assert color_pixel_count(yellow, 'yellow') == 256
    assert color_pixel_count(yellow, 'red') == 0


# --- 1. a correct episode passes cleanly ---------------------------------

def test_a_correctly_encoded_episode_passes(good_episode):
    ok, problems = validate_episode(good_episode)
    assert ok, problems


# --- 2. each historical failure is caught, and named -------------------

def mutate(poses, **kwargs):
    return encode(*poses, **kwargs)


@pytest.mark.parametrize('label,build,must_mention', [
    ('rotation delta by subtraction (the 243/499 inflation bug)',
     lambda p: mutate(p, rotation_delta='subtract'), 'state[t+1]'),
    ('7-dim state, missing the POS_EULER pad slot',
     lambda p: mutate(p, state_dim=7), 'StateEncoding.POS_EULER'),
    ('all frames black (near-clip plane left at 1.0 m)',
     lambda p: mutate(p, image_fn=lambda step_idx: make_image(black=True)), 'near-clip'),
    ('well-formed frames the target never appears in',
     lambda p: mutate(p, image_fn=lambda step_idx: make_image(with_target=False)), 'not visible'),
    ('gripper never actuates, so no grasp was attempted',
     lambda p: mutate(p, freeze_gripper=True), 'never moves'),
    ('deltas inflated tenfold',
     lambda p: mutate(p, inflate=10.0), 'state[t+1]'),
])
def test_a_broken_episode_is_rejected_and_says_why(poses, label, build, must_mention):
    ok, problems = validate_episode(build(poses))
    assert not ok, label
    assert must_mention.lower() in ' '.join(problems).lower(), problems


def test_an_arm_that_never_moves_is_rejected(poses):
    positions, quats, grippers = poses
    ok, problems = validate_episode(encode([positions[0]] * N_STEPS, quats, grippers))
    assert not ok
    assert 'travelled' in ' '.join(problems).lower()


def test_too_few_steps_is_rejected(poses):
    positions, quats, grippers = poses
    ok, problems = validate_episode(encode(positions[:6], quats[:6], grippers[:6]))
    assert not ok
    assert 'steps' in ' '.join(problems).lower()


def test_a_single_nan_is_caught(good_episode):
    episode = [dict(step) for step in good_episode]
    episode[3]['action'] = episode[3]['action'].copy()
    episode[3]['action'][2] = np.nan
    ok, problems = validate_episode(episode)
    assert not ok
    assert 'non-finite' in ' '.join(problems).lower()


def test_writing_to_the_pad_slot_is_caught(good_episode):
    """Every field from the rotation on would be read one slot left."""
    episode = [dict(step) for step in good_episode]
    episode[5]['state'] = episode[5]['state'].copy()
    episode[5]['state'][6] = 0.3
    ok, problems = validate_episode(episode)
    assert not ok
    assert 'pad' in ' '.join(problems).lower()


# --- 3. the tolerance has headroom ---------------------------------------

def test_float32_noise_does_not_trip_the_identities(good_episode):
    """Tolerances are 100 um and 0.0057 deg; correct data sits ~1e4 below."""
    states = np.asarray([step['state'] for step in good_episode], dtype=float)
    actions = np.asarray([step['action'] for step in good_episode], dtype=float)

    position_residual = np.linalg.norm(
        (states[:-1, :3] + actions[:-1, :3]) - states[1:, :3], axis=1)
    rotation_residual = (Rot.from_euler(EULER_SEQ, states[:-1, 3:6])
                         * Rot.from_euler(EULER_SEQ, actions[:-1, 3:6])
                         * Rot.from_euler(EULER_SEQ, states[1:, 3:6]).inv()).magnitude()

    assert position_residual.max() < 1e-6
    assert rotation_residual.max() < np.radians(1e-4)


# --- 4. end to end through the CLI, on files -----------------------------

def test_the_cli_quarantines_the_bad_episode_and_exits_non_zero(
        tmp_path, poses, good_episode):
    episodes = tmp_path / 'episodes'
    quarantine = tmp_path / 'quarantine'
    episodes.mkdir()

    np.save(episodes / 'episode_00000.npy', good_episode, allow_pickle=True)
    np.save(episodes / 'episode_00001.npy',
            encode(*poses, rotation_delta='subtract'), allow_pickle=True)
    np.save(episodes / 'episode_00002.npy', good_episode, allow_pickle=True)

    result = subprocess.run(
        [sys.executable, VALIDATOR, '--episodes-dir', str(episodes),
         '--quarantine', str(quarantine)],
        capture_output=True, text=True)

    assert result.returncode == 1, result.stdout
    assert len(list(episodes.iterdir())) == 2, 'the failing episode must be moved out'
    assert (quarantine / 'episode_00001.npy').exists()
