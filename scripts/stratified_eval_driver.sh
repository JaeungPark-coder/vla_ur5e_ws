#!/bin/bash
# Stratified evaluation under the converged first frame (2026-11-14 evening). Prediction: README "stratified evaluation under the converged
# first frame" (written before this ran).
#   A) v3 6k on the 7 n=60 +Y-corner seeds (wide box, 1-trial runs) -> same-seed pairs with the stale-frame n=60 result
#   B) 6 regions x 15 seeds (9200-9214): v2 3k, v3 3k, v3 6k; then the expert on the same 90 seeds (regions split over the two GPUs)
#   setsid nohup scripts/stratified_eval_driver.sh > eval_runs/strat_driver.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30
unset CLIENT_EXTRA_PARAMS
V2=checkpoints/pi0_ur5e_pick_place/short_3k/2999
V33=checkpoints/pi0_ur5e_pick_place_v3/v3_3k/2999
V36=checkpoints/pi0_ur5e_pick_place_v3/v3_6k/5999
for c in $V2 $V33 $V36; do [ -d "$OPENPI_DIR/$c" ] || { echo "no checkpoint $c"; exit 1; }; done
# name xrange yrange
REGIONS="centre:0.42,0.48:-0.10,0.10 posY_strip:0.42,0.48:0.10,0.16 negY_strip:0.42,0.48:-0.16,-0.10 posY_corner:0.38,0.42:0.11,0.16 negY_corner:0.38,0.42:-0.16,-0.11 mid_lowX:0.38,0.42:-0.11,0.11"

echo "$(date +%H:%M) A: v3 6k on the n=60 +Y-corner seeds, converged frame"
for s in 8007 8012 8015 8019 8022 8033 8036; do
    POLICY_CONFIG=pi0_ur5e_pick_place_v3 VLA_BRIDGE_CUBE_X_RANGE=0.38,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.16,0.16 \
        scripts/eval_checkpoint.sh "$V36" "eval_runs/ffA_v3_6k_seed$s" 1 "$s" 50
done
echo "$(date +%H:%M) A finished"

model() {  # $1 config $2 checkpoint $3 name
    for r in $REGIONS; do
        IFS=: read -r name xr yr <<< "$r"
        echo "$(date +%H:%M) $3 $name start"
        POLICY_CONFIG="$1" VLA_BRIDGE_CUBE_X_RANGE="$xr" VLA_BRIDGE_CUBE_Y_RANGE="$yr" \
            scripts/eval_checkpoint.sh "$2" "eval_runs/strat_$3_$name" 15 9200 50
    done
}
model pi0_ur5e_pick_place    $V2  v2_3k
model pi0_ur5e_pick_place_v3 $V33 v3_3k
model pi0_ur5e_pick_place_v3 $V36 v3_6k

echo "$(date +%H:%M) expert start"
expert() {  # $1 GPU, rest: region specs
    local g=$1; shift
    for r in "$@"; do
        IFS=: read -r name xr yr <<< "$r"
        GPU=$g scripts/isaac_py.sh expert_baseline.py --label "strat_$name" --seeds 9200 9214 --x_range ${xr/,/ } --y_range ${yr/,/ } \
            --out "$PWD/eval_results/expert_strat_$name.jsonl" > "eval_runs/strat_expert_$name.log" 2>&1
    done
}
set -- $REGIONS
expert 0 "$1" "$2" "$3" &
expert 1 "$4" "$5" "$6" &
wait
echo "$(date +%H:%M) stratified evaluations finished"
