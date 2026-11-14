#!/bin/bash
# Visual-variation evaluation: one visual factor changed per run (isaac/visual_perturbation.py), v3 6k, K=25, d=0, converged first
# frame, centre region seeds 9200-9214 (the same spawns as eval_runs/lat_d0_centre). Prediction: README "visual variation" (written
# before this ran).   setsid nohup scripts/visual_eval_driver.sh > eval_runs/visual_driver.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30 POLICY_CONFIG=pi0_ur5e_pick_place_v3
export VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.10,0.10
unset CLIENT_EXTRA_PARAMS
V36=checkpoints/pi0_ur5e_pick_place_v3/v3_6k/5999
[ -d "$OPENPI_DIR/$V36" ] || { echo "no checkpoint $V36"; exit 1; }
# name|spec
CONDITIONS="none|none light05|light=0.5 light2|light=2 tint|tint=1.0:0.8:0.6 ground_grey|ground=0.15:0.15:0.15 ground_wood|ground=0.7:0.55:0.35
cube_dark|cube=0.5:0.05:0.05 cube_orange|cube=0.9:0.35:0.1 basecam3|basecam=0:0.03:0 basecam6|basecam=0:0.06:0.03 wrist3|wristtilt=3 wrist8|wristtilt=8"
for c in $CONDITIONS; do
    IFS='|' read -r name spec <<< "$c"
    echo "$(date +%H:%M) $name ($spec) start"
    VLA_BRIDGE_VISUAL="$spec" scripts/eval_checkpoint.sh "$V36" "eval_runs/vis_$name" 15 9200 25
    grep -m1 "VLA_BRIDGE_VISUAL" "eval_runs/vis_$name/bridge.log" || [ "$spec" = none ] || echo "WARNING: perturbation not confirmed in bridge.log"
done
echo "$(date +%H:%M) visual evaluations finished"
