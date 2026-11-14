#!/bin/bash
# Inference latency emulated in ticks (lockstep kept), v3 6k, converged first frame, K=25. Prediction: README "inference latency in
# simulation" (written before this ran). 3 regions x 15 seeds (9200-9214, the stratified seeds -> also paired with the K=50 results).
#   conditions, in this order: d0 (baseline), async d10, sync d10, async d20
#   setsid nohup scripts/latency_eval_driver.sh > eval_runs/latency_driver.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30 POLICY_CONFIG=pi0_ur5e_pick_place_v3
V36=checkpoints/pi0_ur5e_pick_place_v3/v3_6k/5999
[ -d "$OPENPI_DIR/$V36" ] || { echo "no checkpoint $V36"; exit 1; }
K=25
REGIONS="centre:0.42,0.48:-0.10,0.10 posY_strip:0.42,0.48:0.10,0.16 negY_strip:0.42,0.48:-0.16,-0.10"
# name:delay:mode
CONDITIONS="d0:0:sync async10:10:async sync10:10:sync async20:20:async"
for c in $CONDITIONS; do
    IFS=: read -r cname d mode <<< "$c"
    for r in $REGIONS; do
        IFS=: read -r name xr yr <<< "$r"
        echo "$(date +%H:%M) $cname $name start"
        CLIENT_EXTRA_PARAMS="-p chunk_delay_ticks:=$d -p chunk_delay_mode:=$mode" \
            VLA_BRIDGE_CUBE_X_RANGE="$xr" VLA_BRIDGE_CUBE_Y_RANGE="$yr" \
            scripts/eval_checkpoint.sh "$V36" "eval_runs/lat_${cname}_$name" 15 9200 "$K"
    done
done
echo "$(date +%H:%M) latency evaluations finished"
