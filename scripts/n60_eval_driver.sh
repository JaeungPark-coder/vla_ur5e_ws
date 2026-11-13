#!/bin/bash
# n=60 paired efficacy evaluation on the wide box (X 0.38-0.48, Y +-0.16), seeds 8000-8059, current settings for every model
# (K=50, no position limit, stale first frame = VLA_BRIDGE_RESET_RENDERS unset, as in training; policy inputs saved). Then the scripted
# expert on the same 60 seeds (two halves on the two GPUs). Prediction: README "2026-11-14: n=60 wide box" (written before this ran).
#   setsid nohup scripts/n60_eval_driver.sh > eval_runs/n60_driver.log 2>&1 &
# Waits for the v3 6k driver to finish first (Isaac and training/other evaluations never overlap on the GPUs).
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
while pgrep -f "v3_6k_driver.sh" >/dev/null; do sleep 60; done
sleep 30
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job still running -- not starting"; exit 1; fi
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_CUBE_X_RANGE=0.38,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.16,0.16
unset VLA_BRIDGE_RESET_RENDERS CLIENT_EXTRA_PARAMS
run() {  # $1 config, $2 checkpoint (relative to openpi), $3 output name
    if [ ! -d "$OPENPI_DIR/$2" ]; then echo "$(date +%H:%M) $3: no checkpoint $2 -- skipped"; return; fi
    echo "$(date +%H:%M) $3 start"
    POLICY_CONFIG="$1" scripts/eval_checkpoint.sh "$2" "eval_runs/$3" 60 8000 50
    echo "$(date +%H:%M) $3 finished"
}
run pi0_ur5e_pick_place    checkpoints/pi0_ur5e_pick_place/short_3k/2999   n60_v2_3k
run pi0_ur5e_pick_place_v3 checkpoints/pi0_ur5e_pick_place_v3/v3_3k/2999   n60_v3_3k
run pi0_ur5e_pick_place_v3 checkpoints/pi0_ur5e_pick_place_v3/v3_6k/5999   n60_v3_6k
echo "$(date +%H:%M) expert baseline start"
GPU=0 scripts/isaac_py.sh expert_baseline.py --label n60a --seeds 8000 8029 --x_range 0.38 0.48 --y_range -0.16 0.16 --out "$PWD/eval_results/expert_baseline_n60_a.jsonl" > eval_runs/n60_expert_a.log 2>&1 &
GPU=1 scripts/isaac_py.sh expert_baseline.py --label n60b --seeds 8030 8059 --x_range 0.38 0.48 --y_range -0.16 0.16 --out "$PWD/eval_results/expert_baseline_n60_b.jsonl" > eval_runs/n60_expert_b.log 2>&1 &
wait
cat eval_results/expert_baseline_n60_a.jsonl eval_results/expert_baseline_n60_b.jsonl > eval_results/expert_baseline_n60.jsonl
echo "$(date +%H:%M) n=60 evaluations finished"
