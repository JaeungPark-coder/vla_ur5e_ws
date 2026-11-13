#!/bin/bash
# A2 (2026-11-13): the same two strips and seeds as strip_eval_driver.sh (v2 short_3k/2999, X 0.42-0.48, seeds 9000-9019), now WITH the
# position-target step limit fixed in advance at 1.5x the largest per-tick joint change of the v2 demonstrations. Paired with the
# no-limit runs eval_runs/strip_posY|strip_negY. Policy input images saved.
cd "$(dirname "$0")/.." || exit 1
CK=checkpoints/pi0_ur5e_pick_place/short_3k/2999
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48
export CLIENT_EXTRA_PARAMS="-p target_step_limit_rad:=[0.0362,0.0423,0.0381,0.0555,0.0617,0.0728]"
VLA_BRIDGE_CUBE_Y_RANGE=0.10,0.16 scripts/eval_checkpoint.sh "$CK" eval_runs/limit_posY 20 9000 50
VLA_BRIDGE_CUBE_Y_RANGE=-0.16,-0.10 scripts/eval_checkpoint.sh "$CK" eval_runs/limit_negY 20 9000 50
echo "$(date +%H:%M) limit evaluations finished"
