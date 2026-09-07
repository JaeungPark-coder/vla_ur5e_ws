"""Open-vocabulary object detector wrapping Grounding DINO
(`IDEA-Research/grounding-dino-tiny`, via HuggingFace `transformers`) --
given an RGB image and a text description (e.g. "a red cube"), returns
where in the image that specific object is. The perception half of the LLM
+ open-vocabulary hybrid pipeline (see isaac/hybrid_pick_place_demo.py);
`llm_command_parser.py` decides WHICH object to look for, this module finds
WHERE it is.

Usage exactly matches the model's own card (fetched and confirmed against
huggingface.co/IDEA-Research/grounding-dino-tiny): text queries must be
lowercase and end with a period, multiple queries joined with ". ".

Deliberately has no Isaac-Sim or object_configs dependency -- takes the
description string it's given, same reusability rationale as
llm_command_parser.py.

Requires `transformers`, `torch`, and `pillow` (`pip install transformers
torch pillow`); downloads the grounding-dino-tiny weights from HuggingFace
Hub on first use.
"""
import numpy as np

MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DEFAULT_BOX_THRESHOLD = 0.4
DEFAULT_TEXT_THRESHOLD = 0.3


class ObjectDetector:
    def __init__(self, box_threshold=DEFAULT_BOX_THRESHOLD, text_threshold=DEFAULT_TEXT_THRESHOLD):
        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoProcessor.from_pretrained(MODEL_ID)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(self.device)

    def locate(self, image, description):
        """image: HxWx3 uint8 RGB array (e.g. pick_place_scene's base_rgb
        observation). description: a single text query, e.g. "a red cube"
        (object_configs.description()'s format -- lowercase, no trailing
        period needed here, added automatically).

        Returns (pixel_x, pixel_y, confidence) for the highest-scoring
        detection above box_threshold, or None if nothing matched --
        callers MUST handle the None case (the named object may not
        actually be visible/present)."""
        import torch
        from PIL import Image

        pil_image = Image.fromarray(np.asarray(image)[..., :3])
        text_query = description.lower().rstrip(".") + "."

        inputs = self.processor(images=pil_image, text=text_query, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[pil_image.size[::-1]],  # (height, width), per the model card
        )[0]

        if len(results["boxes"]) == 0:
            return None

        best_idx = int(torch.argmax(results["scores"]))
        x0, y0, x1, y1 = [float(v) for v in results["boxes"][best_idx]]
        confidence = float(results["scores"][best_idx])
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0, confidence
