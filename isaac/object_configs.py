"""Small, fixed object vocabulary shared by the multi-object scene
(pick_place_scene.spawn_random_objects), the LLM command parser
(perception/llm_command_parser.py, which is told this exact vocabulary so
it only ever names objects that can actually exist), and the open-vocabulary
detector (perception/object_detector.py, whose text queries come straight
from this list's descriptions).

Kept intentionally tiny for the first version of the hybrid pipeline --
extending real teleoperated/scripted demo generalization to more shapes and
colors is a natural follow-up once this baseline round-trips correctly.
"""
import numpy as np

# (color, shape) pairs. Colors are RGB display colors used when spawning in
# Isaac Sim (isaac_sim_common.add_shape) -- picked to be visually distinct
# for the detector, not meant to be photorealistic.
OBJECT_VOCABULARY = [
    ("red", "cube"),
    ("blue", "cube"),
    ("green", "sphere"),
    ("yellow", "cylinder"),
]

COLOR_RGB = {
    "red": (0.8, 0.1, 0.1),
    "blue": (0.1, 0.2, 0.8),
    "green": (0.1, 0.7, 0.2),
    "yellow": (0.85, 0.75, 0.1),
}


def description(color, shape):
    """e.g. "a red cube" -- matches Grounding DINO's expected text-query
    phrasing (lowercase; the caller joins multiple with ". " and a
    trailing period, per object_detector.py)."""
    return f"a {color} {shape}"


def all_descriptions():
    return [description(c, s) for c, s in OBJECT_VOCABULARY]


def sample_objects(n, rng: np.random.Generator):
    """Returns `n` distinct (color, shape) pairs sampled from the
    vocabulary without replacement -- capped at the vocabulary size."""
    n = min(n, len(OBJECT_VOCABULARY))
    idx = rng.choice(len(OBJECT_VOCABULARY), size=n, replace=False)
    return [OBJECT_VOCABULARY[i] for i in idx]
