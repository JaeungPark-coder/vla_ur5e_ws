"""TFDS/RLDS dataset builder for the UR5e pick-and-place demonstrations
collected by ../isaac/collect_rlds_episodes.py, structured for OpenVLA
fine-tuning. Modeled directly on the community-standard
`kpertsch/rlds_dataset_builder` template's `example_dataset_dataset_builder.py`
structure (nested `steps` dataset with observation/action/language fields
plus discount/reward/is_first/is_last/is_terminal flags) -- fetched and
confirmed from that repo, but this specific file's schema is new, not
copied verbatim, so **this is the least-verified file in the whole
project**: it's never been run through `tfds build`. Expect to iterate on
it more than anything else here.

Deliberately skips the Universal Sentence Encoder language-instruction
embedding some RLDS examples include (an extra TF-Hub dependency) -- only
the raw `language_instruction` text is stored. Add the embedding back
(`tensorflow_hub`'s `universal-sentence-encoder-large`, per the template) if
OpenVLA's own training config for your version turns out to require it.

Usage: `cd openvla_integration && tfds build` (needs `tensorflow`,
`tensorflow_datasets`, and `apache_beam` installed -- see
`kpertsch/rlds_dataset_builder`'s `environment_ubuntu.yml` for the exact
pinned versions that repo uses, since RLDS/TFDS version compatibility is a
known source of friction).
"""
import glob
import os

import numpy as np
import tensorflow_datasets as tfds

RAW_EPISODES_DIR = os.path.join(os.path.dirname(__file__), "raw_episodes")
IMAGE_SHAPE = (256, 256, 3)  # keep in sync with isaac/pick_place_scene.CAMERA_RESOLUTION
STATE_ACTION_DIM = 7  # xyz(3) + rotvec(3) + gripper(1) -- see collect_rlds_episodes.py


class Ur5ePickPlace(tfds.core.GeneratorBasedBuilder):
    """UR5e multi-object pick-and-place demonstrations for OpenVLA fine-tuning."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial release -- scripted demonstrations from collect_rlds_episodes.py."}

    def _info(self) -> tfds.core.DatasetInfo:
        return tfds.core.DatasetInfo(
            builder=self,
            description=(
                "UR5e picks one of several objects on a table (named by "
                "language instruction) and places it in a target zone. "
                "Scripted demonstrations from vla_ur5e_ws's Isaac Sim "
                "pipeline, not real teleoperation -- see that project's "
                "README for the caveats that implies."
            ),
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        "image": tfds.features.Image(
                            shape=IMAGE_SHAPE, dtype=np.uint8, encoding_format="png",
                            doc="Base (third-person) camera RGB frame."),
                        "state": tfds.features.Tensor(
                            shape=(STATE_ACTION_DIM,), dtype=np.float32,
                            doc="End-effector pose (xyz, rotvec) + gripper (0=open..1=closed)."),
                    }),
                    "action": tfds.features.Tensor(
                        shape=(STATE_ACTION_DIM,), dtype=np.float32,
                        doc="Delta end-effector pose (xyz, rotvec) + absolute gripper target."),
                    "discount": tfds.features.Scalar(dtype=np.float32, doc="Always 1.0 -- no discounting used."),
                    "reward": tfds.features.Scalar(dtype=np.float32, doc="1.0 on the final step of the episode, else 0.0."),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(doc="Path to the raw .npy episode this came from."),
                }),
            }),
        )

    def _split_generators(self, dl_manager):
        del dl_manager  # unused -- reading straight from RAW_EPISODES_DIR, nothing to download
        return {"train": self._generate_examples()}

    def _generate_examples(self):
        episode_paths = sorted(glob.glob(os.path.join(RAW_EPISODES_DIR, "episode_*.npy")))
        if not episode_paths:
            raise RuntimeError(
                f"no episode_*.npy files found in {RAW_EPISODES_DIR} -- run "
                "../isaac/collect_rlds_episodes.py first."
            )

        for episode_path in episode_paths:
            raw_steps = np.load(episode_path, allow_pickle=True)
            n = len(raw_steps)

            episode = []
            for i, step in enumerate(raw_steps):
                episode.append({
                    "observation": {
                        "image": step["image"],
                        "state": step["state"],
                    },
                    "action": step["action"],
                    "discount": np.float32(1.0),
                    "reward": np.float32(1.0 if i == n - 1 else 0.0),
                    "is_first": i == 0,
                    "is_last": i == n - 1,
                    "is_terminal": i == n - 1,
                    "language_instruction": step["language_instruction"],
                })

            sample = {
                "steps": episode,
                "episode_metadata": {"file_path": episode_path},
            }
            yield episode_path, sample
