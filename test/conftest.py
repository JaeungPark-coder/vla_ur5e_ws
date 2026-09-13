"""Puts the three source trees on sys.path so these tests run anywhere.

Everything in this directory is deliberately importable without ROS 2,
without Isaac Sim, without a GPU and without a robot -- it is the encoding
and perception half of the repo, checked against analytic geometry and
synthetic episodes. The Isaac-requiring checks live next to the simulator
code in isaac/ instead (test_feasibility_gate.py, test_offset_grasp.py);
those boot a SimulationApp and are not collected here.

    python -m pytest test/ -v        # from the repository root
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

for _subtree in ('src/vla_bridge', 'isaac', 'openvla_integration'):
    _path = str(REPO_ROOT / _subtree)
    if _path not in sys.path:
        sys.path.insert(0, _path)
