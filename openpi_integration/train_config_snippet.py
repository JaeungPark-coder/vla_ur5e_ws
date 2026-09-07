"""The pieces to add to your local openpi checkout's
`src/openpi/training/config.py` -- LoRA fine-tune of pi0_base on the UR5e
pick-and-place dataset from ../isaac/collect_demos.py, via
ur5e_pick_place_policy.py's transforms.

VERIFIED (2026-09-07) end-to-end against an actual openpi checkout (this
project's earlier draft of this integration was written without one -- see
git history). Three real issues were found and fixed here vs. that draft,
each confirmed against openpi's real source:

  1. LeRobotUR5eDataConfig must live IN config.py (imported classes below),
     not in ur5e_pick_place_policy.py -- config.py has to import that policy
     module to reference UR5eInputs/UR5eOutputs, so the policy module
     importing back from openpi.training.config for DataConfigFactory/etc.
     is a circular import. Same placement openpi's own LeRobotLiberoDataConfig
     uses (defined in config.py, only its transform classes live in
     libero_policy.py).
  2. DataConfigFactory.create_base_config() takes TWO args
     (assets_dirs, model_config), not one -- confirmed against config.py's
     own DataConfigFactory.create_base_config signature and every real
     example's create() override.
  3. prompt_from_task=True is required in the DataConfig -- collect_demos.py
     logs the language instruction via LeRobot's own per-frame `task` field
     (add_frame(..., task=PROMPT)), not a literal "prompt" column.
     prompt_from_task=True is what populates "prompt" from that (see
     data_loader.create_torch_dataset) before repack_transforms runs; without
     it the repack_transform's "prompt": "prompt" mapping has nothing to
     read. Confirmed against openpi's own pi0_libero TrainConfig, which sets
     this for the identical reason.
  4. Don't override `assets` to point at pi0_base's own remote checkpoint
     bucket hoping to reuse a pretrained "ur5e" embodiment's norm stats --
     compute_norm_stats.py always WRITES locally to
     `config.assets_dirs / repo_id` regardless of any assets_dir/asset_id
     override, so overriding assets_dir/asset_id makes create_base_config()
     look for stats in a different (remote, likely nonexistent) place than
     where they actually get written; the default AssetsConfig() (asset_id
     falls back to repo_id, assets_dir falls back to config.assets_dirs)
     is the one that actually matches where compute_norm_stats.py writes.

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

# ---------------------------------------------------------------------------
# 1. Add near config.py's other `import openpi.policies.*_policy as *_policy`
#    lines (top of the file).
# ---------------------------------------------------------------------------
IMPORT_SNIPPET = """
import openpi.policies.ur5e_pick_place_policy as ur5e_pick_place_policy
"""

# ---------------------------------------------------------------------------
# 2. Add anywhere among config.py's other `class LeRobot*DataConfig
#    (DataConfigFactory)` definitions (e.g. right after LeRobotLiberoDataConfig).
# ---------------------------------------------------------------------------
DATA_CONFIG_CLASS_SNIPPET = '''
@dataclasses.dataclass(frozen=True)
class LeRobotUR5eDataConfig(DataConfigFactory):
    """vla_ur5e_ws's UR5e pick-and-place dataset (collect_demos.py) --
    single base + wrist camera, joints(6)+gripper(1) state, joint-delta
    actions with an absolute gripper. See
    openpi.policies.ur5e_pick_place_policy for UR5eInputs/UR5eOutputs."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "base_rgb": "image",
                        "wrist_rgb": "wrist_image",
                        "joints": "joints",
                        "gripper": "gripper",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[ur5e_pick_place_policy.UR5eInputs(model_type=model_config.model_type)],
            outputs=[ur5e_pick_place_policy.UR5eOutputs()],
        )

        # Convert absolute actions to delta actions, except the gripper
        # (7th, index -1) dimension -- by convention gripper stays absolute.
        delta_action_mask = _transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_action_mask)],
            outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
'''

# ---------------------------------------------------------------------------
# 3. Add to config.py's `_CONFIGS` list (CHANGE the repo_id to your own
#    collect_demos.py --repo_id).
# ---------------------------------------------------------------------------
TRAIN_CONFIG_ENTRY_SNIPPET = '''
TrainConfig(
    name="pi0_ur5e_pick_place",
    model=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotUR5eDataConfig(
        repo_id="jaeung/ur5e_pick_place_v1",  # CHANGE ME if you used a different --repo_id
        base_config=DataConfig(prompt_from_task=True),  # see module docstring point 3
        # Default AssetsConfig() deliberately NOT overridden -- see module docstring point 4.
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    freeze_filter=pi0_config.Pi0Config(
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
    num_train_steps=30_000,  # starting point -- watch for overfitting well before this on a small scripted-demo dataset
),
'''
