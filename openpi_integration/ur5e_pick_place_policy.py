"""UR5e pick-and-place policy transforms for openpi, adapted from openpi's
own `examples/ur5/README.md` (fetched verbatim from
github.com/Physical-Intelligence/openpi) -- `UR5Inputs`/`UR5Outputs` there,
renamed UR5e* here to match this project and wired to the exact dataset
field names collect_demos.py writes (image/wrist_image/joints/gripper/
actions/task).

VERIFIED (2026-09-07) against an actual openpi checkout (src/openpi/training/
config.py, src/openpi/policies/libero_policy.py) -- this file holds ONLY the
transform classes, matching openpi's own convention (compare libero_policy.py,
which likewise holds only LiberoInputs/LiberoOutputs). The DataConfigFactory
subclass (LeRobotUR5eDataConfig) belongs in training/config.py itself, NOT
here -- see train_config_snippet.py's docstring for why (a real circular
import: config.py must import this module to reference UR5eInputs/
UR5eOutputs, so this module importing back from openpi.training.config would
fail with a partially-initialized-module ImportError. An earlier draft of
this file put LeRobotUR5eDataConfig here and hit exactly that).

Drop this file into your local openpi checkout at
`src/openpi/policies/ur5e_pick_place_policy.py` -- then see
train_config_snippet.py for the config.py-side pieces.
"""
import dataclasses

import numpy as np

from openpi import transforms as _transforms
from openpi.models import model as _model


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
        # DeltaActions mask in train_config_snippet.py's LeRobotUR5eDataConfig.
        return {"actions": np.asarray(data["actions"][:, :7])}
