#!/bin/bash
# K=25 additional evaluation (README "K=25 additional evaluation", written before these runs): stratified 6 regions x seeds
# 9200-9214, converged frame, K=25.  E1 v4 3k original ground, E2 v3 3k original ground, E3 v4 3k on held-out H0.
#   setsid nohup scripts/k25_strat_driver.sh > eval_runs/k25_strat_driver.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30
unset CLIENT_EXTRA_PARAMS
V4=checkpoints/pi0_ur5e_pick_place_v4/v4_3k/2999; V3=checkpoints/pi0_ur5e_pick_place_v3/v3_3k/2999
H0="heldout=0,h0=0.85:0.85:0.83"
REGIONS="centre:0.42,0.48:-0.10,0.10 posY_strip:0.42,0.48:0.10,0.16 negY_strip:0.42,0.48:-0.16,-0.10 posY_corner:0.38,0.42:0.11,0.16 negY_corner:0.38,0.42:-0.16,-0.11 mid_lowX:0.38,0.42:-0.11,0.11"
# set|checkpoint|config|visual spec
SETS="strat_k25_v4_3k|$V4|pi0_ur5e_pick_place_v4|none strat_k25_v3_3k|$V3|pi0_ur5e_pick_place_v3|none strat_h0_k25_v4_3k|$V4|pi0_ur5e_pick_place_v4|$H0"
short=""
for s in $SETS; do
    IFS='|' read -r set ck cfg vis <<< "$s"
    for r in $REGIONS; do
        IFS=: read -r name xr yr <<< "$r"
        out=eval_runs/${set}_$name
        if [ -f "$out/results.csv" ] && [ "$(($(wc -l < "$out/results.csv") - 1))" -ge 15 ]; then echo "$(date +%H:%M) $set $name done -- skipped"; continue; fi
        echo "$(date +%H:%M) $set $name ($vis) start"
        POLICY_CONFIG=$cfg VLA_BRIDGE_VISUAL="$vis" VLA_BRIDGE_CUBE_X_RANGE="$xr" VLA_BRIDGE_CUBE_Y_RANGE="$yr" \
            scripts/eval_checkpoint.sh "$ck" "$out" 15 9200 25
        [ "$vis" = none ] || grep -m1 "VLA_BRIDGE_VISUAL" "$out/bridge.log" || echo "WARNING: perturbation not confirmed in $out/bridge.log"
        n=$(($(wc -l < "$out/results.csv" 2>/dev/null || echo 1) - 1)); [ "$n" -ge 15 ] || short="$short ${set}_$name($n)"
    done
done
[ -z "$short" ] && echo "$(date +%H:%M) K=25 evaluation finished (18 runs, 15 rows each)" || echo "$(date +%H:%M) K=25 evaluation INCOMPLETE:$short"
