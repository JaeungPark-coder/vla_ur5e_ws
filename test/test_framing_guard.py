"""Checks the framing guard as it is actually wired into the scene.

pick_place_scene imports Isaac Sim, so it cannot be imported here. Its
top-level assignments are executed in order instead -- the real BASE_FRAMING
and MIN_CUBE_PIXELS_IN_BASE_VIEW expressions, compiled from the shipped
source, not restated.
"""
import ast
from pathlib import Path

import numpy as np
import pytest

import camera_framing as cf

SCENE = Path(__file__).resolve().parents[1] / 'isaac' / 'pick_place_scene.py'

# The two runs on record, in peak cube pixels seen anywhere in an episode.
KNOWN_GOOD_PEAK_PX2 = 139       # smoketest / clipfix_check, cameras working
KNOWN_BROKEN_PEAK_PX2 = 0       # the first 100-episode run, cube never visible


@pytest.fixture(scope='module')
def scene_constants():
    """Every top-level assignment in the scene that runs without Isaac Sim."""
    tree = ast.parse(SCENE.read_text(encoding='utf-8'))
    namespace = {'np': np, 'camera_framing': cf}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        try:
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         str(SCENE), 'exec'), namespace)
        except Exception:
            continue            # depends on Isaac Sim or on a runtime value
    return namespace


@pytest.fixture(scope='module')
def framing(scene_constants):
    return scene_constants['BASE_FRAMING']


# --- 1. the derivation in the scene actually runs ------------------------

@pytest.mark.parametrize('symbol', [
    'CUBE_SIZE_M', 'TARGET_CUBE_SPAN_PX', 'MIN_CUBE_PIXELS_FLOOR',
    'BASE_FRAMING', 'MIN_CUBE_PIXELS_IN_BASE_VIEW'])
def test_every_guard_symbol_evaluates(scene_constants, symbol):
    assert symbol in scene_constants


def test_base_framing_is_a_real_analysis_not_a_stub(framing):
    assert isinstance(framing, cf.FramingAnalysis)


# --- 2. the numbers ------------------------------------------------------

def test_the_guard_is_derived_from_the_achievable_peak(scene_constants, framing):
    guard = scene_constants['MIN_CUBE_PIXELS_IN_BASE_VIEW']
    floor = scene_constants['MIN_CUBE_PIXELS_FLOOR']
    assert guard == max(floor, round(0.5 * framing.expected_peak_area_px))
    assert guard >= floor, 'the floor must still hold'


def test_this_scene_cannot_reach_the_target_span(framing):
    """The finding, stated as a test: 30 px across is not available at this
    workspace and resolution, so the old flat threshold was asking for
    something the geometry cannot deliver."""
    assert not framing.target_reachable


# --- 3. it separates the runs on record ----------------------------------

def test_it_does_not_reject_the_run_that_actually_worked(scene_constants):
    assert KNOWN_GOOD_PEAK_PX2 >= scene_constants['MIN_CUBE_PIXELS_IN_BASE_VIEW']


def test_it_rejects_the_run_where_the_cube_was_never_visible(scene_constants):
    assert KNOWN_BROKEN_PEAK_PX2 < scene_constants['MIN_CUBE_PIXELS_IN_BASE_VIEW']


def test_the_old_flat_threshold_could_not_see_a_half_occluded_run(scene_constants, framing):
    """A run at a quarter of the achievable peak passes the old floor and
    fails the derived guard -- which is the whole reason for deriving it."""
    half_occluded = int(0.25 * framing.expected_peak_area_px)
    assert half_occluded >= scene_constants['MIN_CUBE_PIXELS_FLOOR']
    assert half_occluded < scene_constants['MIN_CUBE_PIXELS_IN_BASE_VIEW']


# --- 4. the runtime path and the CLI path agree --------------------------

def analysis_from_parsed_constants(must_cover_m=None):
    """What `preflight framing` computes, from constants parsed out of the
    scene file rather than evaluated inside it."""
    scene = cf.read_constants(cf.SCENE_PATH)
    if must_cover_m is None:
        must_cover_m = cf.workspace_span_m(scene['CUBE_Y_RANGE'],
                                           scene['PLACE_TARGET_POSITION'][1])
    return cf.framing_analysis(
        object_size_m=scene['CUBE_SIZE_M'],
        camera_position=scene['BASE_CAMERA_POSITION'],
        sample_points=[(x, y, scene['CUBE_Z'])
                       for x in scene['CUBE_X_RANGE']
                       for y in scene['CUBE_Y_RANGE']],
        hfov_deg=scene['BASE_CAMERA_HORIZONTAL_FOV_DEG'],
        resolution_px=scene['CAMERA_RESOLUTION'][0],
        must_cover_m=must_cover_m,
        target_span_px=scene['TARGET_CUBE_SPAN_PX'],
        reference_point=scene['BASE_CAMERA_AIM_POINT'])


@pytest.mark.parametrize('field', [
    'best_span_px', 'ceiling_span_px', 'required_hfov_deg',
    'required_resolution_px', 'required_workspace_m'])
def test_the_scene_and_the_cli_produce_the_same_analysis(framing, field):
    standalone = analysis_from_parsed_constants()
    assert getattr(framing, field) == pytest.approx(getattr(standalone, field),
                                                    abs=1e-12)


# --- 5. the guard responds to the thing it is about ----------------------

def test_tightening_the_workspace_is_what_unlocks_the_target(framing):
    """Not a bigger sensor and not a closer camera: the lens has to cover the
    workspace, so the workspace is the binding constraint."""
    tighter = analysis_from_parsed_constants(must_cover_m=0.30)
    assert tighter.ceiling_span_px > framing.ceiling_span_px
    assert tighter.target_reachable


def test_describe_explains_the_result_rather_than_only_numbering_it(framing):
    lines = cf.describe(framing)
    assert lines and all(isinstance(line, str) for line in lines)
