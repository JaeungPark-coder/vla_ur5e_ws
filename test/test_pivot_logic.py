"""Checks pivot_dwell_check's analysis logic without Isaac Sim.

pivot_dwell_check calls SimulationApp() at module level, so importing it
boots Kit. The functions under test are lifted out of the file by ast and
compiled from that same source instead -- not a copy of the logic, but the
shipped text of it, executed in a bare namespace.

What the diagnostic is for: the residual tracking error was read once as the
fingers contacting the table, and later suggested to be gravity and drive
gains. Only a free-space hold separates them, and the answer changes whether
there is work to do at all.
"""
import ast
import io
from pathlib import Path

import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'isaac' / 'pivot_dwell_check.py'

WANTED_FUNCTIONS = {'hold', 'report_dwell'}
WANTED_CONSTANTS = {'DWELL_SAMPLES', 'SETTLED_TOLERANCE_M',
                    'PIVOT_ROLLS_DEG', 'PIVOT_TILTS_DEG'}


def _load():
    tree = ast.parse(io.open(SOURCE, encoding='utf-8').read())
    nodes = [node for node in tree.body
             if (isinstance(node, ast.FunctionDef) and node.name in WANTED_FUNCTIONS)
             or (isinstance(node, ast.Assign)
                 and isinstance(node.targets[0], ast.Name)
                 and node.targets[0].id in WANTED_CONSTANTS)]
    namespace = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'),
         namespace)
    return namespace


PIVOT = _load()
hold, report_dwell = PIVOT['hold'], PIVOT['report_dwell']
DWELL_SAMPLES = PIVOT['DWELL_SAMPLES']
SETTLED_TOLERANCE_M = PIVOT['SETTLED_TOLERANCE_M']

TARGET = np.array([0.45, 0.0, 0.35])
DOWNWARD_ROTVEC = np.array([0.0, np.pi, 0.0])


class FakeScene:
    """An arm whose grip point sits `error_mm_at_tick(tick)` off the command."""

    def __init__(self, error_mm_at_tick):
        self.error_at = error_mm_at_tick
        self.tick = 0
        self.target = None

    def step_towards(self, position, rotvec, gripper):
        self.tick += 1
        self.target = np.asarray(position, dtype=float)

    def grip_point_world(self):
        return self.target + np.array([self.error_at(self.tick) / 1000.0, 0.0, 0.0])


def test_every_requested_sample_tick_is_recorded():
    samples = hold(FakeScene(lambda t: 3.0), TARGET, DOWNWARD_ROTVEC,
                   max(DWELL_SAMPLES))
    assert set(samples) == set(DWELL_SAMPLES)


def test_a_settled_arm_reads_as_settled():
    """Constant error is a calibration offset, not a dynamics problem."""
    samples = hold(FakeScene(lambda t: 3.0), TARGET, DOWNWARD_ROTVEC,
                   max(DWELL_SAMPLES))
    growth = report_dwell('settled (constant 3 mm)', samples)
    assert abs(growth) <= SETTLED_TOLERANCE_M


def test_it_reproduces_the_original_drifting_observation():
    """24 mm at tick 60 growing to 43 mm at tick 120 -- the case that
    prompted the check."""
    drifting = FakeScene(lambda t: 24.0 + (43.0 - 24.0) * (t - 60) / 60.0)
    samples = hold(drifting, TARGET, DOWNWARD_ROTVEC, max(DWELL_SAMPLES))

    assert samples[60] == pytest.approx(0.024)
    assert samples[120] == pytest.approx(0.043)
    assert report_dwell('drifting', samples) > SETTLED_TOLERANCE_M


def test_a_short_hold_refuses_to_claim_a_verdict():
    """Rather than inventing a growth figure from samples it never took."""
    samples = hold(FakeScene(lambda t: 5.0), TARGET, DOWNWARD_ROTVEC, 30)
    assert max(samples) == 30
    assert report_dwell('30 ticks only', samples) is None


# --- the pivot half: a wrong TCP vs a constant bias -----------------------

def spread_and_offset(landed, commanded):
    landed = np.asarray(landed)
    spread = float(np.max(np.linalg.norm(
        landed[:, None, :] - landed[None, :, :], axis=-1)))
    return spread, float(np.linalg.norm(landed.mean(axis=0) - commanded))


def test_a_constant_bias_shows_as_offset_with_no_spread():
    """Every orientation lands in the same wrong place."""
    landed = [TARGET + np.array([0.004, 0.0, 0.0])] * 6
    spread, offset = spread_and_offset(landed, TARGET)
    assert spread < 1e-9
    assert offset == pytest.approx(0.004)


def test_a_wrong_tcp_shows_as_spread_with_no_offset():
    """Rotating drags the tip around an arc whose radius is the TCP error, so
    the error averages out of the mean and appears only in the spread."""
    angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
    landed = [TARGET + 0.004 * np.array([np.cos(a), np.sin(a), 0.0]) for a in angles]
    spread, offset = spread_and_offset(landed, TARGET)

    assert spread == pytest.approx(0.008), 'the arc diameter is twice the error'
    assert offset < 1e-9
