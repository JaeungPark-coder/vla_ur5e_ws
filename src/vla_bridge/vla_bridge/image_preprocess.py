"""The one crop-and-resize that every base-camera image goes through before the policy sees it -- in the sim dataset
(isaac/collect_demos.py) and at run time (vla_policy_client.py, for the sim bridge and the real camera alike).

Why (README 2026-11-16): the lab's base camera is a Stereolabs ZED.  Through the wide lens a 4 cm cube 70 cm away is about
7 px wide after openpi's 224 x 224 resize; cropping to the work area first keeps it visible.  The crop is given as fractions
of the full image (u0, v0, u1, v1), so the sim may render at a smaller resolution than the real camera as long as the
intrinsics -- FOV and aspect -- match (sim_profile.BASE_CAMERA).  Both sides must apply the SAME crop with the SAME
resampler, which is why it lives here and nowhere else.
"""
import numpy as np
from PIL import Image

FULL = (0.0, 0.0, 1.0, 1.0)


def crop_box_px(width, height, crop):
    """Fractional (u0, v0, u1, v1) -> integer pixel box (left, top, right, bottom), rounded to the nearest pixel."""
    u0, v0, u1, v1 = (float(c) for c in crop)
    if not (0.0 <= u0 < u1 <= 1.0 and 0.0 <= v0 < v1 <= 1.0):
        raise ValueError(f'crop {crop}: expected 0 <= u0 < u1 <= 1 and 0 <= v0 < v1 <= 1')
    return (int(round(u0 * width)), int(round(v0 * height)), int(round(u1 * width)), int(round(v1 * height)))


def crop_resize(image, crop=None, out_wh=None):
    """image: HxWx3 uint8.  crop: fractional box or None (whole image).  out_wh: (width, height) or None (crop size).
    Bilinear on a PIL image, so the sim (numpy from Replicator) and the real camera (numpy from cv_bridge) are resampled
    by the same code."""
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f'expected HxWx3, got {arr.shape}')
    if (crop is None or tuple(crop) == FULL) and (out_wh is None or tuple(out_wh) == (arr.shape[1], arr.shape[0])):
        return arr
    img = Image.fromarray(arr.astype(np.uint8))
    if crop is not None and tuple(crop) != FULL:
        img = img.crop(crop_box_px(arr.shape[1], arr.shape[0], crop))
    if out_wh is not None and tuple(out_wh) != img.size:
        img = img.resize((int(out_wh[0]), int(out_wh[1])), Image.BILINEAR)
    return np.asarray(img)


def crop_aspect(width, height, crop):
    """Aspect (w / h) of the cropped region -- out_wh should keep it, or the cube is stretched."""
    left, top, right, bottom = crop_box_px(width, height, crop or FULL)
    return (right - left) / (bottom - top)
