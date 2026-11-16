#!/bin/bash
# v4 6k (README "v4 6k -- design and prediction", written before training): the v3 6k recipe on the v4 data, then stratified
# 6 x 15 at K=25 on the original ground and on H0.  Unattended; each stage skipped when complete, so a rerun resumes.
#   setsid nohup scripts/v4_6k_driver.sh > eval_runs/v4_6k_driver.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
say() { echo "$(date '+%m-%d %H:%M') $*"; }
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then say "GPU job running -- not starting"; exit 1; fi
CK=checkpoints/pi0_ur5e_pick_place_v4/v4_6k/5999
if [ -d "$OPENPI_DIR/$CK" ]; then
    say "training already done -- skipped"
else
    say "v4 6k training start (free on big disk: $(df -h --output=avail "$OPENPI_DIR" | tail -1))"
    (cd "$OPENPI_DIR" && env -u HF_LEROBOT_HOME CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train.py \
        pi0_ur5e_pick_place_v4 --exp-name=v4_6k --num-train-steps=6000 --lr-schedule.decay-steps=6000 --lr-schedule.warmup-steps=300 \
        --save-interval=1000 --keep-period=3000 --log-interval=50 --num-workers=16 --no-wandb-enabled > "$OPENPI_DIR/v4_6k.log" 2>&1)
    say "v4 6k training ended: $(grep -E '^Step 59[05]0' "$OPENPI_DIR/v4_6k.log" | tail -1)"
    [ -d "$OPENPI_DIR/$CK" ] || { say "no checkpoint $CK -- stopping"; exit 1; }
fi
sleep 20
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30 POLICY_CONFIG=pi0_ur5e_pick_place_v4
unset CLIENT_EXTRA_PARAMS
REGIONS="centre:0.42,0.48:-0.10,0.10 posY_strip:0.42,0.48:0.10,0.16 negY_strip:0.42,0.48:-0.16,-0.10 posY_corner:0.38,0.42:0.11,0.16 negY_corner:0.38,0.42:-0.16,-0.11 mid_lowX:0.38,0.42:-0.11,0.11"
short=""
for s in "strat_k25_v4_6k|none" "strat_h0_k25_v4_6k|heldout=0,h0=0.85:0.85:0.83"; do
    IFS='|' read -r set vis <<< "$s"
    for r in $REGIONS; do
        IFS=: read -r name xr yr <<< "$r"
        out=eval_runs/${set}_$name
        if [ -f "$out/results.csv" ] && [ "$(($(wc -l < "$out/results.csv") - 1))" -ge 15 ]; then continue; fi
        say "$set $name start"
        VLA_BRIDGE_VISUAL="$vis" VLA_BRIDGE_CUBE_X_RANGE="$xr" VLA_BRIDGE_CUBE_Y_RANGE="$yr" scripts/eval_checkpoint.sh "$CK" "$out" 15 9200 25
        [ "$vis" = none ] || grep -m1 "VLA_BRIDGE_VISUAL" "$out/bridge.log" || echo "WARNING: perturbation not confirmed in $out/bridge.log"
        n=$(($(wc -l < "$out/results.csv" 2>/dev/null || echo 1) - 1)); [ "$n" -ge 15 ] || short="$short ${set}_$name($n)"
    done
done
[ -z "$short" ] && say "v4 6k finished (12 evaluations, 15 rows each)" || say "v4 6k evaluation INCOMPLETE:$short"
