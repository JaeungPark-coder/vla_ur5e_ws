#!/bin/bash
# v4 diagnostic (README "v4 diagnostic", written after the core results, before these runs): D1 original ground K=25,
# D2 ground_grey K=50; v4 3k, centre, seeds 9200-9214, converged frame.   setsid nohup scripts/v4_diag.sh > eval_runs/v4_diag.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30 POLICY_CONFIG=pi0_ur5e_pick_place_v4
export VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.10,0.10
unset CLIENT_EXTRA_PARAMS
CK=checkpoints/pi0_ur5e_pick_place_v4/v4_3k/2999
# name|visual spec|K
for c in "none|none|25" "ground_grey_k50|ground=0.15:0.15:0.15|50"; do
    IFS='|' read -r name spec k <<< "$c"
    out=eval_runs/vis_v4_3k_$name
    if [ -f "$out/results.csv" ] && [ "$(($(wc -l < "$out/results.csv") - 1))" -ge 15 ]; then echo "$name done -- skipped"; continue; fi
    echo "$(date +%H:%M) $name ($spec, K=$k) start"
    export VLA_BRIDGE_VISUAL="$spec"   # "none" exactly as visual_eval_driver.sh passes it for vis_none
    scripts/eval_checkpoint.sh "$CK" "$out" 15 9200 "$k"
    n=$(($(wc -l < "$out/results.csv" 2>/dev/null || echo 1) - 1)); [ "$n" -ge 15 ] || echo "WARNING: $name has $n rows"
done
echo "$(date +%H:%M) v4 diagnostic finished"
