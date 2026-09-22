"""Does the metamorphic relation actually earn its place?

The scenario it exists for is the one the three identities cannot see: the
collector and the gate share the SAME wrong axis order, so the dataset is
self-consistently wrong and every identity passes. That is reproduced here
by encoding an episode end to end in 'zyx' and reading it back with the gate
also set to 'zyx' -- lowercase, so the import guard allows it, and
self-consistent, so the identities are satisfied.

What it does NOT prove: that the convention agrees with OpenVLA's. Only a
comparison against its dataloader settles that, and it is step 6 of the
bring-up table in the README.
"""
import importlib
import os
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as Rot

import validate_dataset as vd

INSTRUCTION = 'pick up the a red cube and place it in the target zone'
VALIDATOR_SOURCE = os.path.join(os.path.dirname(vd.__file__), 'validate_dataset.py')


@pytest.fixture(scope='module')
def rng():
    return np.random.default_rng(3)


def make_image(step_idx=0):
    """step_idx nudges the blob a little each call -- see
    test_validate_dataset.py's own make_image docstring for why a good
    episode's frames need to actually differ tick to tick."""
    image = np.full((256, 256, 3), (30, 30, 38), dtype=np.uint8)
    row = 100 + (step_idx % 40)
    image[row:row + 14, 120:134] = (204, 25, 25)
    return image


def build_episode(rng, seq, n=30):
    """An episode encoded end to end in `seq`.

    Self-consistent by construction whatever `seq` is -- which is what makes
    it the right probe: nothing internal to the data can reveal the error.
    """
    steps = []
    base = np.array([0.0, np.pi, 0.0])
    rotations = [Rot.from_rotvec(base + rng.normal(scale=np.radians(3.0), size=3))
                 for _ in range(n + 1)]

    for i in range(n):
        t, t_next = i / n, (i + 1) / n
        position = np.array([0.35 + 0.2 * t, -0.1 + 0.15 * t, 0.4 - 0.18 * t])
        next_position = np.array([0.35 + 0.2 * t_next, -0.1 + 0.15 * t_next,
                                  0.4 - 0.18 * t_next])
        # closes at t=0.6, releases again at t=0.9 -- see
        # test_validate_dataset.py's make_poses for why a good episode's
        # gripper needs to actually reopen before the end.
        gripper = 0.0 if t < 0.6 else (0.8 if t < 0.9 else 0.0)
        next_gripper = 0.0 if t_next < 0.6 else (0.8 if t_next < 0.9 else 0.0)

        state = np.concatenate([position, rotations[i].as_euler(seq), [0.0], [gripper]])
        action = np.concatenate([
            next_position - position,
            (rotations[i].inv() * rotations[i + 1]).as_euler(seq),
            [next_gripper]])

        steps.append({'image': make_image(i),
                      'state': state.astype(np.float32),
                      'action': action.astype(np.float32),
                      'language_instruction': INSTRUCTION})
    return steps


@pytest.fixture(scope='module')
def correct_episode(rng):
    return build_episode(rng, 'xyz')


@pytest.fixture(scope='module')
def self_consistently_wrong_episode(rng):
    return build_episode(rng, 'zyx')


# --- 1. the independent implementation must itself be right --------------

def test_the_hand_built_matrix_matches_the_library(rng):
    """Rz @ Ry @ Rx, written out, against scipy's extrinsic 'xyz'."""
    worst = 0.0
    for _ in range(2000):
        angles = rng.uniform(-np.pi, np.pi, size=3)
        angles[1] = rng.uniform(-np.pi / 2, np.pi / 2)
        worst = max(worst, vd._matrix_angle_between(
            Rot.from_euler('xyz', angles).as_matrix(),
            vd.euler_xyz_to_matrix(angles)))
    assert worst < 1e-13


def test_the_hand_built_matrix_disagrees_with_the_intrinsic_reading(rng):
    """Otherwise the relation would be satisfied by both conventions and
    would prove nothing at all."""
    differences = []
    for _ in range(200):
        angles = rng.uniform(-np.pi, np.pi, size=3)
        angles[1] = rng.uniform(-np.pi / 2, np.pi / 2)
        differences.append(vd._matrix_angle_between(
            Rot.from_euler('XYZ', angles).as_matrix(),
            vd.euler_xyz_to_matrix(angles)))
    assert np.median(differences) > np.radians(10)


# --- 2. the import guard -------------------------------------------------

def test_uppercase_euler_seq_is_refused_at_import(tmp_path):
    """Before any data is looked at, because an intrinsic sequence here would
    make the relation check itself meaningless."""
    source = open(VALIDATOR_SOURCE, encoding='utf-8').read()
    patched = source.replace('EULER_SEQ = "xyz"', 'EULER_SEQ = "XYZ"', 1)
    assert patched != source, 'the EULER_SEQ literal moved; update this test'

    (tmp_path / 'vd_upper.py').write_text(patched, encoding='utf-8')
    sys.path.insert(0, str(tmp_path))
    try:
        with pytest.raises(ValueError):
            importlib.import_module('vd_upper')
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop('vd_upper', None)


# --- 3. the blind spot the relation exists for ---------------------------

def test_a_correctly_encoded_episode_passes(correct_episode):
    ok, problems = vd.validate_episode(correct_episode)
    assert ok, problems


def test_a_wrong_convention_is_caught_while_the_gate_still_reads_xyz(
        self_consistently_wrong_episode):
    ok, _ = vd.validate_episode(self_consistently_wrong_episode)
    assert not ok


@pytest.fixture
def gate_reading_zyx():
    """Data and checker agreeing on the wrong order -- the shared
    misunderstanding case."""
    vd.EULER_SEQ = 'zyx'
    try:
        yield
    finally:
        vd.EULER_SEQ = 'xyz'


def test_the_three_identities_see_nothing_wrong(
        self_consistently_wrong_episode, gate_reading_zyx):
    """This is the point. The data satisfies every internal consistency
    check, because it was generated and read with the same wrong rule."""
    problems = []
    vd._check_encoding_consistency(self_consistently_wrong_episode, problems)
    assert not problems


def test_the_metamorphic_relation_catches_it(
        self_consistently_wrong_episode, gate_reading_zyx):
    problems = []
    vd._check_rotation_convention(self_consistently_wrong_episode, problems)
    assert problems
    assert 'extrinsic' in problems[0]


def test_and_the_episode_is_rejected_overall(
        self_consistently_wrong_episode, gate_reading_zyx):
    ok, _ = vd.validate_episode(self_consistently_wrong_episode)
    assert not ok


# --- 4. no false positive, with margin -----------------------------------

def test_correct_data_clears_the_tolerance_by_orders_of_magnitude(correct_episode):
    states = np.asarray([step['state'] for step in correct_episode], dtype=float)
    worst = max(vd._matrix_angle_between(Rot.from_euler('xyz', angles).as_matrix(),
                                         vd.euler_xyz_to_matrix(angles))
                for angles in states[:, 3:6])
    assert worst < vd.CONVENTION_TOL_RAD / 100.0


# --- 5. the same blind spot in verify_action_encoding --------------------

def test_the_encoding_verifier_shares_one_euler_constant_with_the_gate():
    """It used to declare its own. Two constants meaning the same thing are
    two constants that can disagree, and this is the one whose disagreement
    would be invisible."""
    import verify_action_encoding as vae
    assert vae.EULER_SEQ is vd.EULER_SEQ


def test_the_encoding_verifiers_convention_check_passes():
    import verify_action_encoding as vae
    assert vae.check_rotation_convention() is True
