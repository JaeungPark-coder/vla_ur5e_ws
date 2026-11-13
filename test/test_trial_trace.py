"""trial_trace.py: the blow-up criteria and per-tick trace the evaluation relies on."""
import json

import numpy as np

import trial_trace as tt


def _run(cubes, out_dir=None):
    t = tt.TrialTrace(out_dir)
    t.start(0, cube=cubes[0])
    for i, c in enumerate(cubes[1:], 1):
        t.add(i, c, [0, 0, 0.3], np.zeros(6), 0.0)
    return t


def test_normal_pick_and_place_is_not_a_blowup():
    # rest, lift 4 mm per tick up to 0.16 m, carry, put down
    z = np.concatenate([np.full(50, 0.02), np.linspace(0.02, 0.16, 35), np.linspace(0.16, 0.02, 35)])
    cubes = [[0.45, 0.0, v] for v in z]
    s = _run(cubes).summary()
    assert not s["blowup"] and s["first_trip"] is None
    assert s["max_cube_z"] == 0.16 and s["max_one_tick_jump_m"] < 0.005


def test_one_tick_jump_inside_a_block_is_caught_even_if_it_lands_again():
    cubes = [[0.45, 0.0, 0.02]] * 10 + [[0.45, 0.06, 0.02]] + [[0.45, 0.06, 0.02]] * 5   # shoved 6 cm in one tick
    s = _run(cubes).summary()
    assert s["blowup"] and s["reasons"] == {"jump": 1}
    assert s["first_trip"]["tick"] == 9 and s["first_trip"]["jump_m"] > 0.059


def test_flung_up_and_fallen_back_still_counts():
    cubes = [[0.45, 0.0, 0.02], [0.45, 0.0, 0.05], [0.45, 0.0, 0.30], [0.45, 0.0, 0.05], [0.45, 0.0, 0.02]]
    s = _run(cubes).summary()
    assert s["blowup"] and "z_high" in s["reasons"] and s["max_cube_z"] == 0.3


def test_fell_far_and_nan():
    assert "fell" in tt.judge_tick([0.45, 0.0, -0.05], None)
    assert "far" in tt.judge_tick([1.2, 0.0, 0.02], None)
    assert tt.judge_tick([np.nan, 0, 0], None) == ["nan"]
    assert tt.judge_tick([0.45, 0.1, 0.02], [0.45, 0.1, 0.02]) == []


def test_first_trip_has_context_and_save_roundtrip(tmp_path):
    t = tt.TrialTrace(str(tmp_path))
    t.start(2, cube=[0.45, 0, 0.02])
    t.add(1, [0.45, 0, 0.02], [0.4, 0, 0.1], np.zeros(6), 0.1)
    t.add(2, [0.45, 0.2, 0.02], [0.4, 0.01, 0.05], np.ones(6), 0.7)
    stem = t.save()
    assert stem.endswith("trial_003")                       # trial = epoch + 1
    s = json.load(open(stem + ".json"))
    assert s["blowup"] and s["first_trip"]["gripper"] == 0.7 and s["first_trip"]["grip_point"] == [0.4, 0.01, 0.05]
    z = np.load(stem + ".npz")
    assert z["cube"].shape == (2, 3) and z["joints"].shape == (2, 6) and z["request_id"].tolist() == [1, 2]


def test_start_clears_the_previous_trial():
    t = _run([[0.45, 0, 0.02], [0.45, 0, 0.5]])
    assert t.summary()["blowup"]
    t.start(1, cube=[0.45, 0, 0.02])
    assert t.ticks == 0 and not t.summary()["blowup"]


def test_policy_input_images_are_saved_next_to_the_trace(tmp_path):
    t = tt.TrialTrace(str(tmp_path))
    t.start(0, cube=[0.45, 0.0, 0.02])
    t.add(1, [0.45, 0.0, 0.02], [0, 0, 0.3], np.zeros(6), 0.0)
    base, wrist = np.full((8, 8, 3), 7, np.uint8), np.full((8, 8, 3), 9, np.uint8)
    t.add_images(t.ticks, base, wrist)
    t.add_images(t.ticks + 50, base + 1, wrist + 1)
    t.save()
    z = np.load(tmp_path / "trial_001_images.npz")
    assert z["tick"].tolist() == [1, 51] and z["base"].shape == (2, 8, 8, 3) and z["wrist"][1, 0, 0, 0] == 10
    t.start(1, cube=[0.45, 0.0, 0.02])      # a new trial starts with no images
    t.add(1, [0.45, 0.0, 0.02], [0, 0, 0.3], np.zeros(6), 0.0)
    t.save()
    assert not (tmp_path / "trial_002_images.npz").exists()


def test_commanded_targets_are_saved_when_given_every_tick(tmp_path):
    t = tt.TrialTrace(str(tmp_path))
    t.start(0, cube=[0.45, 0.0, 0.02])
    for i in range(3):
        t.add(i + 1, [0.45, 0.0, 0.02], [0, 0, 0.3], np.zeros(6), 0.0, target=np.arange(7) + i)
    t.save()
    z = np.load(tmp_path / "trial_001.npz")
    assert z["target"].shape == (3, 7) and z["target"][2, 0] == 2
