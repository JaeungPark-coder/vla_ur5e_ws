"""Registration snippet for OpenVLA's own checkout -- NOT run as part of
this project. Copy the pieces below into
`prismatic/vla/datasets/rlds/oxe/transforms.py` and
`prismatic/vla/datasets/rlds/oxe/configs.py` inside your OpenVLA clone
(https://github.com/openvla/openvla), then run its `vla-scripts/finetune.py`
against the `ur5e_pick_place` dataset name.

ADJUST warning (more so than any other file in this project): the exact
field names OpenVLA's `configs.py`/`transforms.py` expect (StateEncoding/
ActionEncoding enum members, the transform function signature) are written
here from the general Open-X-Embodiment OXE dataset-registration convention
this codebase family shares, NOT fetched verbatim from OpenVLA's current
`configs.py` source -- check the exact enum names and an existing entry
(e.g. `bridge_orig` or a LIBERO config) in your checkout before trusting
this verbatim, since these have drifted across OpenVLA/OXE versions.

Requires the RLDS dataset to already be built first:
    cd ../openvla_integration && tfds build
"""

# ---------------------------------------------------------------------------
# 1. Add to prismatic/vla/datasets/rlds/oxe/transforms.py
# ---------------------------------------------------------------------------
TRANSFORMS_PY_SNIPPET = '''
def ur5e_pick_place_dataset_transform(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """collect_rlds_episodes.py's action is already (delta_xyz(3),
    delta_roll_pitch_yaw(3), absolute_gripper(1)) and state is
    (xyz(3), roll_pitch_yaw(3), pad(1)=0, gripper(1)) -- verified
    (2026-09-11) against OpenVLA's own prismatic/vla/datasets/rlds/oxe/
    configs.py source: StateEncoding.POS_EULER is exactly
    "EEF XYZ(3) + Roll-Pitch-Yaw(3) + <PAD>(1) + Gripper(1)" and
    ActionEncoding.EEF_POS is exactly "EEF Delta XYZ(3) + Roll-Pitch-Yaw(3) +
    Gripper(1)" -- so this transform mostly just documents that rather than
    reshaping anything. An earlier version of this file logged delta
    ROTATION VECTORS instead (both a dimension short on the state side, and
    wrong about the rotation representation on both sides); see
    isaac/verify_action_encoding.py for the round-trip test that would have
    caught it.

    ADJUST still open: the exact Euler AXIS ORDER/convention OXE uses
    (openvla_integration/validate_dataset.py's EULER_SEQ = "xyz", i.e.
    scipy's standard ROS/URDF extrinsic-XYZ RPY -- defined there once and
    imported by the collector and the verifier) could not be confirmed from
    OpenVLA's public source -- transforms.py's bridge_orig transform defers
    dataset-specific rotation handling to an undefined relabel_bridge_actions
    helper. Check an actual OpenVLA checkout's dataloader before trusting
    this axis order verbatim.
    """
    trajectory["action"] = tf.cast(trajectory["action"], tf.float32)
    return trajectory
'''

# ---------------------------------------------------------------------------
# 2. Add to prismatic/vla/datasets/rlds/oxe/configs.py's OXE_DATASET_CONFIGS dict
# ---------------------------------------------------------------------------
CONFIGS_PY_SNIPPET = '''
"ur5e_pick_place": {
    "image_obs_keys": {"primary": "image", "secondary": None, "wrist": None},
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    # Enum names/values confirmed (2026-09-11) against OpenVLA's own
    # configs.py source -- POS_EULER=8-dim, EEF_POS=7-dim, both RPY-euler
    # rotation, both matched exactly by collect_rlds_episodes.py's
    # _state_vec/_delta_action. Still worth a final check against YOUR
    # checkout in case these enums have drifted since (see this file's
    # module docstring).
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
},
'''

# ---------------------------------------------------------------------------
# 3. Fine-tune, from inside your OpenVLA checkout
# ---------------------------------------------------------------------------
FINETUNE_COMMAND = """
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py \\
    --vla_path openvla/openvla-7b \\
    --data_root_dir <path to the directory containing the built ur5e_pick_place TFDS dataset> \\
    --dataset_name ur5e_pick_place \\
    --run_root_dir ./runs/ur5e_pick_place \\
    --lora_rank 32 \\
    --batch_size 16 \\
    --learning_rate 5e-4
"""
