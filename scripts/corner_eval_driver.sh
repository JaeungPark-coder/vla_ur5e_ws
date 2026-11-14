#!/bin/bash
# Corner confirmation test (2026-11-14): v3 6k on the near +Y corner (X 0.38-0.42, Y +0.11..+0.16) and its mirror (Y -0.16..-0.11),
# seeds 9100-9119 each, K=50, CONVERGED first frame (VLA_BRIDGE_RESET_RENDERS=30, the real-robot condition), policy inputs saved;
# then the scripted expert on the same seeds (one corner per GPU). Prediction: README "corner confirmation test" (written before this ran).
#   setsid nohup scripts/corner_eval_driver.sh > eval_runs/corner_driver.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
CK=checkpoints/pi0_ur5e_pick_place_v3/v3_6k/5999
[ -d "$OPENPI_DIR/$CK" ] || { echo "no checkpoint $CK"; exit 1; }
export POLICY_CONFIG=pi0_ur5e_pick_place_v3 VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30 VLA_BRIDGE_CUBE_X_RANGE=0.38,0.42
unset CLIENT_EXTRA_PARAMS
echo "$(date +%H:%M) +Y corner start (RESET_RENDERS=$VLA_BRIDGE_RESET_RENDERS)"
VLA_BRIDGE_CUBE_Y_RANGE=0.11,0.16 scripts/eval_checkpoint.sh "$CK" eval_runs/corner_v3_6k_posY 20 9100 50
echo "$(date +%H:%M) -Y corner start"
VLA_BRIDGE_CUBE_Y_RANGE=-0.16,-0.11 scripts/eval_checkpoint.sh "$CK" eval_runs/corner_v3_6k_negY 20 9100 50
echo "$(date +%H:%M) expert start"
GPU=0 scripts/isaac_py.sh expert_baseline.py --label corner_posY --seeds 9100 9119 --x_range 0.38 0.42 --y_range 0.11 0.16 --out "$PWD/eval_results/expert_corner_posY.jsonl" > eval_runs/corner_expert_posY.log 2>&1 &
GPU=1 scripts/isaac_py.sh expert_baseline.py --label corner_negY --seeds 9100 9119 --x_range 0.38 0.42 --y_range -0.16 -0.11 --out "$PWD/eval_results/expert_corner_negY.jsonl" > eval_runs/corner_expert_negY.log 2>&1 &
wait
echo "$(date +%H:%M) corner evaluations finished"
