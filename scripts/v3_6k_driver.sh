#!/bin/bash
# 2026-11-14: is the centre bias under-training? Same v3 data, 6000 steps with the LR schedule ending at 6000, compared with v3 3k at the end of its own schedule.
# Order (Isaac Sim and training never overlap on the GPUs):
#   1) v3 3k on the two strips (X 0.42-0.48, seeds 9000-9019; stale first frame, as the v2 strip runs) -> pairs for the 6k comparison
#   2) v3 6k training (both GPUs; seed 42 as v3 3k; save every 1000 steps so a killed run can resume)
#   3) v3 6k on training box, step 2 and the two strips (K=50, 20 trials each)
# Prediction and decision rule: README "2026-11-14: v3 6k" (written before this was started).
cd "$(dirname "$0")/.." || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
export POLICY_CONFIG=pi0_ur5e_pick_place_v3 VLA_BRIDGE_SAVE_IMAGES=1
strips() {  # $1 checkpoint (relative to openpi), $2 name prefix
    VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=0.10,0.16 scripts/eval_checkpoint.sh "$1" "eval_runs/$2_strip_posY" 20 9000 50
    VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.16,-0.10 scripts/eval_checkpoint.sh "$1" "eval_runs/$2_strip_negY" 20 9000 50
}
echo "$(date +%H:%M) v3 3k strip evaluations start"
strips checkpoints/pi0_ur5e_pick_place_v3/v3_3k/2999 v3
echo "$(date +%H:%M) v3 3k strip evaluations finished"

sleep 20
if pgrep -f "kit/python|isaac-sim|serve_policy" >/dev/null; then echo "$(date +%H:%M) Isaac/server still running -- not starting training"; exit 1; fi
echo "$(date +%H:%M) v3 6k training start"
(cd "$OPENPI_DIR" && CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train.py pi0_ur5e_pick_place_v3 --exp-name=v3_6k \
    --num-train-steps=6000 --lr-schedule.decay-steps=6000 --lr-schedule.warmup-steps=300 --save-interval=1000 --keep-period=3000 --log-interval=50 \
    --num-workers=16 --no-wandb-enabled > "$OPENPI_DIR/v3_6k.log" 2>&1)
echo "$(date +%H:%M) v3 6k training ended (exit $?)"
CK=checkpoints/pi0_ur5e_pick_place_v3/v3_6k/5999
[ -d "$OPENPI_DIR/$CK" ] || { echo "$(date +%H:%M) no checkpoint at $OPENPI_DIR/$CK -- not evaluating"; exit 1; }
sleep 20
scripts/eval_checkpoint.sh "$CK" eval_runs/v3_6k_train_box 20 8000 50
VLA_BRIDGE_CUBE_X_RANGE=0.38,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.16,0.16 scripts/eval_checkpoint.sh "$CK" eval_runs/v3_6k_step2 20 8000 50
strips "$CK" v3_6k
echo "$(date +%H:%M) v3 6k evaluations finished"
