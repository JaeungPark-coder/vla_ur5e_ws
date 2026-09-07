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
    delta_rotvec(3), absolute_gripper(1)) -- matching most OXE conventions
    directly, so this transform mostly just documents that rather than
    reshaping anything. ADJUST if your OpenVLA version's EEF_POS action
    encoding expects a different rotation representation (e.g. delta
    axis-angle vs. delta euler) than the rotation-vector this project logs.
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
    "state_encoding": StateEncoding.POS_EULER,   # ADJUST: verify this enum name/value against your checkout
    "action_encoding": ActionEncoding.EEF_POS,   # ADJUST: same
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
