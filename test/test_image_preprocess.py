"""image_preprocess: the base-image crop applied identically to the sim dataset and at run time (README 2026-11-16, ZED)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'vla_bridge'))
from vla_bridge.image_preprocess import crop_aspect, crop_box_px, crop_resize  # noqa: E402


def test_identity_returns_the_same_array():
    img = np.random.default_rng(0).integers(0, 255, (72, 128, 3), dtype=np.uint8)
    assert crop_resize(img) is img
    assert crop_resize(img, (0, 0, 1, 1), (128, 72)) is img


def test_same_view_from_a_smaller_render():
    """A fractional crop of a half-resolution render matches the same crop of the full frame, resized: the sim may render
    smaller than the ZED as long as FOV and aspect match."""
    yy, xx = np.mgrid[0:720, 0:1280]
    full = np.stack([xx * 255 // 1279, yy * 255 // 719, (xx + yy) % 256], -1).astype(np.uint8)
    half = full[::2, ::2]
    crop = (0.25, 0.30, 0.75, 0.85)
    a = crop_resize(full, crop, (224, 224)).astype(int)
    b = crop_resize(half, crop, (224, 224)).astype(int)
    assert a.shape == b.shape == (224, 224, 3)
    assert np.median(np.abs(a[..., :2] - b[..., :2])) <= 1        # the smooth channels agree to a grey level


def test_crop_box_and_aspect():
    assert crop_box_px(1280, 720, (0.25, 0.25, 0.75, 1.0)) == (320, 180, 960, 720)
    assert crop_aspect(1280, 720, (0.25, 0.25, 0.75, 1.0)) == pytest.approx(640 / 540)
    with pytest.raises(ValueError):
        crop_box_px(100, 100, (0.5, 0, 0.4, 1))
    with pytest.raises(ValueError):
        crop_resize(np.zeros((10, 10), np.uint8), (0, 0, 0.5, 0.5))
