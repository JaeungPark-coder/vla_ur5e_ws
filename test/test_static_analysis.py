"""Catches the class of bug py_compile and pytest cannot see: an undefined
name or a name used before its assignment.

hybrid_pick_place_demo.py's `policy = ScriptedPickPlace(...)` line went
missing TWICE, in two independent commits that both touched the same
block -- each time the file still compiled and the existing test suite
still passed, because neither checks whether every name used is actually
defined on that path. pivot_dwell_check.py had the mirror-image bug: a
`from isaac_sim_common import ROBOT_PRIM_PATH` inside main() made Python
treat ROBOT_PRIM_PATH as local to the whole function, so an earlier use of
the (module-level-imported) name in that same function raised
UnboundLocalError -- also invisible to py_compile.

ruff's F821 (undefined name) and F823 (local variable referenced before
assignment) are pure-AST checks -- no imports run, no Isaac Sim needed --
so this is cheap enough to run on every commit.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKED_DIRS = ('isaac', 'src', 'perception', 'openvla_integration', 'openpi_integration')


def test_no_undefined_or_used_before_assignment_names():
    if subprocess.run([sys.executable, '-m', 'ruff', '--version'],
                       capture_output=True).returncode != 0:
        pytest.skip("ruff not installed (pip install ruff) -- skipping F821/F823 check")

    result = subprocess.run(
        [sys.executable, '-m', 'ruff', 'check', '--select', 'F821,F823', *CHECKED_DIRS],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, (
        "ruff found undefined-name or used-before-assignment bugs -- these are "
        "invisible to py_compile and pytest alike (see this file's docstring):\n"
        + result.stdout + result.stderr)
