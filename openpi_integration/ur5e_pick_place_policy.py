"""UR5e pick-and-place policy transforms for openpi, adapted from openpi's
own `examples/ur5/README.md` (fetched verbatim from
github.com/Physical-Intelligence/openpi) -- `UR5Inputs`/`UR5Outputs`/
`LeRobotUR5DataConfig` there, renamed UR5e* here to match this project and
wired to the exact dataset field names collect_demos.py writes
(image/wrist_image/joints/gripper/actions/task).

Drop this file into your local openpi checkout at
`src/openpi/policies/ur5e_pick_place_policy.py`, and add the `pi0_ur5e_pick_place`
TrainConfig from train_config_snippet.py to `src/openpi/training/config.py`'s
`_CONFIGS` list (see that file's own comment for exactly where).

NOTE on import paths below: `UR5Inputs`/`UR5Outputs`/`LeRobotUR5DataConfig`'s
*bodies* are reused near-verbatim from the confirmed openpi README -- that
part is solid. The exact module paths for DataConfig/DataConfigFactory/
AssetsConfig/ModelTransformFactory/weight_loaders (openpi.training.config vs.
some other submodule) are inferred from openpi's docs, not fetched verbatim
from config.py's own import block -- if these don't resolve, check
`src/openpi/training/config.py`'s own imports in your checkout and adjust
the lines below to match.
"""
import dataclasses
import pathlib

import numpy as np
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.training.config import AssetsConfig, DataConfig, DataConfigFactory, ModelTransformFactory


def _parse_image(image) -> np.ndarray:
    """LeRobot stores images as float32 CHW internally -- the model wants
    uint8 HWC. Self-contained reimplementation of the parsing the README
    points at in src/openpi/policies/libero_policy.py, so this file doesn't
    depend on that internal helper's exact location."""
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255.0 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] in (1, 3) and image.shape[0] != image.shape[-1]:
        image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
    return image


@dataclasses.dataclass(frozen=True)
class UR5eInputs(_transforms.DataTransformFn):
    model_type: _model.ModelType = _model.ModelType.PI0

    def __call__(self, data: dict) -> dict:
        # Concatenate joints + gripper into the state vector (matches
        # collect_demos.py's `joints`(6) + `gripper`(1) = 7-dim state).
        state = np.concatenate([data["joints"], data["gripper"]])

        base_image = _parse_image(data["base_rgb"])
        wrist_image = _parse_image(data["wrist_rgb"])

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # No right wrist camera on this setup -- zeroed, masked off below.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_ if self.model_type == _model.ModelType.PI0_FAST else np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class UR5eOutputs(_transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        # 7 action dims: 6 joint deltas + gripper (absolute) -- see the
        # DeltaActions mask in LeRobotUR5eDataConfig below.
        return {"actions": np.asarray(data["actions"][:, :7])}


@dataclasses.dataclass(frozen=True)
class LeRobotUR5eDataConfig(DataConfigFactory):
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Remap collect_demos.py's LeRobot dataset field names to the raw
        # keys UR5eInputs expects.
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
            inputs=[UR5eInputs(model_type=model_config.model_type)],
            outputs=[UR5eOutputs()],
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
            self.create_base_config(assets_dirs),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
