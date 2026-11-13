#!/bin/bash
# +Y strip versus -Y strip, v2 `short_3k/2999`, NEW seeds 9000-9019 (the 8000s found the pattern, these test it), policy input images saved.
# Then the scripted expert on the same seeds/strips (cube visibility, reach). Prediction is in README (2026-11-13) and was written before this ran.
cd "$(dirname "$0")/.." || exit 1
CK=checkpoints/pi0_ur5e_pick_place/short_3k/2999
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48
VLA_BRIDGE_CUBE_Y_RANGE=0.10,0.16 scripts/eval_checkpoint.sh "$CK" eval_runs/strip_posY 20 9000 50
VLA_BRIDGE_CUBE_Y_RANGE=-0.16,-0.10 scripts/eval_checkpoint.sh "$CK" eval_runs/strip_negY 20 9000 50
echo "$(date +%H:%M) strip evaluations finished"
GPU=0 scripts/isaac_py.sh expert_baseline.py --label posY --seeds 9000 9019 --x_range 0.42 0.48 --y_range 0.10 0.16 --out "$PWD/eval_results/expert_baseline_stripposY.jsonl" > eval_runs/strip_expert_posY.log 2>&1 &
GPU=1 scripts/isaac_py.sh expert_baseline.py --label negY --seeds 9000 9019 --x_range 0.42 0.48 --y_range -0.16 -0.10 --out "$PWD/eval_results/expert_baseline_stripnegY.jsonl" > eval_runs/strip_expert_negY.log 2>&1 &
wait
echo "$(date +%H:%M) strip expert baselines finished"
