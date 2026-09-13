"""Checks the framing geometry is self-consistent, and that the constants
really are read from the scene file rather than restated here.

The expensive mistake this measures: 21,000 well-formed frames were
collected in which the cube was never visible. The base view's 139 px was
its apparent size, not an occlusion -- and once that was modelled, the
ceiling turned out to be 20.5 px, so the old flat 30 px floor was asking for
something the workspace and resolution cannot deliver.
"""
import numpy as np
import pytest

import camera_framing as cf

CUBE_SIZE_M = 0.04
RESOLUTION_PX = 256


# --- 1. the three conversions invert each other --------------------------

@pytest.mark.parametrize('distance_m', [0.3, 0.666, 1.2])
@pytest.mark.parametrize('target_px', [10.0, 20.5, 55.0])
def test_required_hfov_and_distance_invert_span_px(distance_m, target_px):
    hfov_deg = cf.required_hfov_deg(CUBE_SIZE_M, distance_m, RESOLUTION_PX, target_px)
    assert cf.span_px(CUBE_SIZE_M, distance_m, RESOLUTION_PX,
                      hfov_deg) == pytest.approx(target_px, abs=1e-9)
    assert cf.required_distance_m(CUBE_SIZE_M, hfov_deg, RESOLUTION_PX,
                                  target_px) == pytest.approx(distance_m, abs=1e-9)


# --- 2. the ceiling really is a ceiling ----------------------------------

def test_no_distance_and_lens_pair_beats_the_computed_ceiling():
    """The lens must still cover the whole workspace, and that constraint is
    what caps the cube's apparent size -- moving closer forces a wider lens
    and gains nothing. This is why 30 px was unreachable.
    """
    must_cover_m = 0.50
    ceiling_px = cf.max_span_px(CUBE_SIZE_M, RESOLUTION_PX, must_cover_m)

    best = 0.0
    for distance_m in np.linspace(0.15, 3.0, 60):
        # the tightest lens that still covers must_cover_m at this distance
        hfov_deg = np.degrees(2 * np.arctan(must_cover_m / (2 * distance_m)))
        best = max(best, cf.span_px(CUBE_SIZE_M, distance_m, RESOLUTION_PX, hfov_deg))

    assert best == pytest.approx(ceiling_px, abs=1e-9)


# --- 3. the constants come from the scene, not from here -----------------

@pytest.fixture(scope='module')
def scene():
    return cf.read_constants(cf.SCENE_PATH)


@pytest.mark.parametrize('key', [
    'CAMERA_RESOLUTION', 'BASE_CAMERA_POSITION', 'BASE_CAMERA_AIM_POINT',
    'BASE_CAMERA_HORIZONTAL_FOV_DEG', 'CUBE_X_RANGE', 'CUBE_Y_RANGE', 'CUBE_Z',
    'PLACE_TARGET_POSITION', 'WRIST_CAMERA_LATERAL_M', 'WRIST_CAMERA_BACK_M',
    'WRIST_CAMERA_HORIZONTAL_FOV_DEG', 'CUBE_SIZE_M', 'TARGET_CUBE_SPAN_PX',
    'MIN_CUBE_PIXELS_FLOOR'])
def test_every_constant_the_analysis_needs_is_readable(scene, key):
    assert key in scene


def test_numpy_array_literals_are_parsed_too(scene):
    assert scene['PLACE_TARGET_POSITION'] == [0.45, 0.30, 0.0]


def test_the_cube_size_is_read_from_the_scene(scene):
    assert scene['CUBE_SIZE_M'] == CUBE_SIZE_M
    assert cf.read_default_arg(cf.COMMON_PATH, 'add_shape', 'size') == CUBE_SIZE_M


def test_the_derived_guard_is_an_expression_not_a_literal(scene):
    """MIN_CUBE_PIXELS_IN_BASE_VIEW is now derived from the framing, so a
    change to the camera moves the guard with it instead of leaving a stale
    number behind."""
    assert 'MIN_CUBE_PIXELS_IN_BASE_VIEW' not in scene


# --- 4. the model against the one real measurement -----------------------

def test_the_model_agrees_with_the_measured_peak(scene):
    """139 px^2 was measured in the base view. The model has to land in the
    same band, or the ceiling above is arithmetic about nothing.
    """
    camera = np.asarray(scene['BASE_CAMERA_POSITION'], dtype=float)
    aim = np.asarray(scene['BASE_CAMERA_AIM_POINT'], dtype=float)
    distance_m = float(np.linalg.norm(aim - camera))

    span = cf.span_px(CUBE_SIZE_M, distance_m, scene['CAMERA_RESOLUTION'][0],
                      scene['BASE_CAMERA_HORIZONTAL_FOV_DEG'])
    predicted_area = span * span

    assert 0.4 * predicted_area <= cf.MEASURED_BASE_PEAK_PIXELS <= 1.6 * predicted_area
