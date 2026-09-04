"""The TrainConfig entry to add to your local openpi checkout's
`src/openpi/training/config.py` (append to that file's `_CONFIGS` list) --
LoRA fine-tune of pi0_base on the UR5e pick-and-place dataset from
../isaac/collect_demos.py, via ../ur5e_pick_place_policy.py's transforms.

LoRA (not full fine-tuning) is the only realistic option on 2x RTX 3090:
LoRA needs >22.5GB VRAM (fits one 3090); full fine-tuning needs >70GB.
Pattern confirmed from openpi's own config.py: the LoRA model variant
(`paligemma_variant="gemma_2b_lora"`, `action_expert_variant="gemma_300m_lora"`)
plus `freeze_filter=<same config>.get_freeze_filter()` and `ema_decay=None`.

Run (on ONE of the two 3090s first, for a short step count, before
committing to a full run -- see ../README.md):
    uv run scripts/compute_norm_stats.py --config-name pi0_ur5e_pick_place
    uv run scripts/train.py pi0_ur5e_pick_place --exp-name=ur5e_pick_place_v1
"""
from openpi.models import pi0_config
from openpi.training import weight_loaders
from openpi.training.config import AssetsConfig, TrainConfig

from ur5e_pick_place_policy import LeRobotUR5eDataConfig  # see that file's own docstring on where to place it

# CHANGE ME: your Hugging Face Hub repo id / local LeRobot dataset name --
# matches the --repo_id passed to collect_demos.py.
REPO_ID = "your_hf_username/ur5e_pick_place"

_LORA_MODEL_CONFIG = pi0_config.Pi0Config(
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
)

pi0_ur5e_pick_place_config = TrainConfig(
    name="pi0_ur5e_pick_place",
    model=_LORA_MODEL_CONFIG,
    data=LeRobotUR5eDataConfig(
        repo_id=REPO_ID,
        # Reload the pi0_base checkpoint's UR5e normalization stats rather
        # than computing fresh ones from our (small, scripted-demo) dataset
        # -- UR5e is already a recognized embodiment in pi0's own
        # pretraining data, so this should transfer better than stats from
        # a few dozen scripted episodes alone. See openpi's docs/norm_stats.md.
        assets=AssetsConfig(
            assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            asset_id="ur5e",
        ),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    # LoRA-specific: freeze everything except the LoRA adapter weights, and
    # skip EMA (not meaningful for LoRA fine-tuning).
    freeze_filter=_LORA_MODEL_CONFIG.get_freeze_filter(),
    ema_decay=None,
    num_train_steps=30_000,  # starting point -- with a small scripted-demo dataset, watch for overfitting well before this and stop early
)
