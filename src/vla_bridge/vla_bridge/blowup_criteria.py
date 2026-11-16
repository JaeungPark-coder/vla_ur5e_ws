"""The ONE definition of a PhysX contact blow-up in an evaluation trial -- used by the bridge's per-tick trace
(isaac/trial_trace.py) and by the client's results.csv `blowup` column (vla_policy_client.py).  Pure Python/numpy.

Why (2026-11-17): the client's column used only "cube above 0.25 m" while the trace used five criteria, so a cube flung
1.6 m SIDEWAYS (v5 pilot, 5 of 15 trials) was a blow-up in the trace and 0 in the CSV.  Criteria, fixed 2026-11-11:

  z_high   cube centre above BLOWUP_Z_M       (a normal episode peaks near 0.16 m; collect_demos drops > 0.25 m)
  jump     cube moved more than JUMP_M in ONE tick (1.2 m/s at 60 Hz; the arm carries it at 2-4 mm/tick and the
           cube's maxDepenetrationVelocity is capped at 0.5 m/s = 8 mm/tick, so anything above is not a push)
  fell     cube centre below FELL_Z_M         (rests at 0.02 m; below the table top = off the edge / through it)
  far      cube centre farther than FAR_M in the table plane from the task's centre (everything the task uses lies
           within ~0.3 m of it)
  nan      a non-finite cube coordinate

Heights are relative to the table top (the bridge subtracts TABLE_Z_M before it publishes; v4's table is at 0).
"""
import numpy as np

BLOWUP_Z_M = 0.25
JUMP_M = 0.02
FELL_Z_M = 0.0
FAR_M = 0.5
TASK_CENTRE_XY = (0.45, 0.10)
REASONS = ("z_high", "jump", "fell", "far", "nan")   # bit i of the mask = REASONS[i]


def judge_tick(cube, prev_cube=None):
    """Reasons ([] = fine) that this tick's cube position trips a criterion.  Without prev_cube, `jump` is not judged
    (a position sampled once per block says nothing about one-tick motion)."""
    cube = np.asarray(cube, dtype=float)
    if not np.all(np.isfinite(cube)):
        return ["nan"]
    reasons = []
    if cube[2] > BLOWUP_Z_M:
        reasons.append("z_high")
    if cube[2] < FELL_Z_M:
        reasons.append("fell")
    if float(np.hypot(cube[0] - TASK_CENTRE_XY[0], cube[1] - TASK_CENTRE_XY[1])) > FAR_M:
        reasons.append("far")
    if prev_cube is not None and float(np.linalg.norm(cube - np.asarray(prev_cube, dtype=float))) > JUMP_M:
        reasons.append("jump")
    return reasons


def reasons_to_mask(reasons):
    """Iterable of reason names (or a {name: count} dict) -> int bitmask."""
    return sum(1 << REASONS.index(r) for r in set(reasons))


def mask_to_reasons(mask):
    mask = int(mask)
    return [r for i, r in enumerate(REASONS) if mask & (1 << i)]
