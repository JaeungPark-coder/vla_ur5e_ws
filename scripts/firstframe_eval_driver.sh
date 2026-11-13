#!/bin/bash
# First-frame test (2026-11-13): v2 short_3k/2999 with a CONVERGED first policy input (VLA_BRIDGE_RESET_RENDERS=30), paired with the
# earlier runs on the same seeds: +Y / -Y strips (X 0.42-0.48, seeds 9000-9019; vs eval_runs/strip_posY|negY) and the training box
# (seeds 8000-8019; vs the K=50 trials of eval_runs/k3_2999). No position limit. Images and commanded targets saved.
cd "$(dirname "$0")/.." || exit 1
CK=checkpoints/pi0_ur5e_pick_place/short_3k/2999
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30
VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=0.10,0.16 scripts/eval_checkpoint.sh "$CK" eval_runs/ff_posY 20 9000 50
VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.16,-0.10 scripts/eval_checkpoint.sh "$CK" eval_runs/ff_negY 20 9000 50
scripts/eval_checkpoint.sh "$CK" eval_runs/ff_train_box 20 8000 50
echo "$(date +%H:%M) first-frame evaluations finished"
