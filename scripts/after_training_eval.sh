#!/bin/bash
# Waits for a training process (PID given) to end, then evaluates its final checkpoint on the three boxes (K=50, 20 trials, seeds 8000-8019).
#   scripts/after_training_eval.sh <training PID> <checkpoint dir relative to $OPENPI> <name prefix> [POLICY_CONFIG]
# Output: eval_runs/<prefix>_train_box|_step1|_step2/ ; progress in the log you redirect to.
cd "$(dirname "$0")/.." || exit 1
PID="$1"; CKPT="$2"; PREFIX="$3"; export POLICY_CONFIG="${4:-pi0_ur5e_pick_place}"
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
while kill -0 "$PID" 2>/dev/null; do sleep 30; done
sleep 20
[ -d "$OPENPI_DIR/$CKPT" ] || { echo "$(date +%H:%M) no checkpoint at $OPENPI_DIR/$CKPT -- not evaluating"; exit 1; }
echo "$(date +%H:%M) training ended, checkpoint present; starting the evaluations"
scripts/eval_checkpoint.sh "$CKPT" "eval_runs/${PREFIX}_train_box" 20 8000 50
VLA_BRIDGE_CUBE_Y_RANGE=-0.13,0.13 scripts/eval_checkpoint.sh "$CKPT" "eval_runs/${PREFIX}_step1" 20 8000 50
VLA_BRIDGE_CUBE_X_RANGE=0.38,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.16,0.16 scripts/eval_checkpoint.sh "$CKPT" "eval_runs/${PREFIX}_step2" 20 8000 50
echo "$(date +%H:%M) all three evaluations finished"
