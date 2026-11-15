#!/bin/bash
# Visual-variation evaluation: one visual factor changed per run (isaac/visual_perturbation.py), v3 6k, K=25, d=0, converged first
# frame, centre region seeds 9200-9214 (the same spawns as eval_runs/lat_d0_centre). Prediction: README "visual variation" (written
# before this ran).   setsid nohup scripts/visual_eval_driver.sh > eval_runs/visual_driver.log 2>&1 &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30 POLICY_CONFIG="${VIS_CONFIG:-pi0_ur5e_pick_place_v3}"
export VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.10,0.10
unset CLIENT_EXTRA_PARAMS
# another model (v4 evaluation): VIS_CKPT=<checkpoint rel. to openpi> VIS_CONFIG=<train config> VIS_TAG=_v4_3k -> eval_runs/vis_v4_3k_<cond>
V36="${VIS_CKPT:-checkpoints/pi0_ur5e_pick_place_v3/v3_6k/5999}"
TAG="${VIS_TAG:-}"
[ -d "$OPENPI_DIR/$V36" ] || { echo "no checkpoint $V36"; exit 1; }
# name|spec
CONDITIONS="none|none light05|light=0.5 light2|light=2 tint|tint=1.0:0.8:0.6 ground_grey|ground=0.15:0.15:0.15 ground_wood|ground=0.7:0.55:0.35
cube_dark|cube=0.5:0.05:0.05 cube_orange|cube=0.9:0.35:0.1 basecam3|basecam=0:0.03:0 basecam6|basecam=0:0.06:0.03 wrist3|wristtilt=3 wrist8|wristtilt=8"
# grid lines vs ground colour follow-up (README "grid lines or ground colour?"):  VIS_SET=grid scripts/visual_eval_driver.sh
[ "${VIS_SET:-}" = grid ] && CONDITIONS="gridrebuilt|gridcolor=0.278:0.451:0.584 gridgrey|gridcolor=0.424:0.424:0.424
nogrid|gridcolor=0.278:0.451:0.584,nogrid=1"
# v4 core (README "v4 ... design", amended): held-out grounds H0 (plain, the lab table colour V4_H0="r:g:b") and H1-H3; VIS_SET=heldout
H0="${V4_H0:-}"
if [ "${VIS_SET:-}" = heldout ] || [ "${VIS_SET:-}" = v4core ]; then
    [ -n "$H0" ] || { echo "V4_H0 (the held-out plain colour r:g:b) is not set"; exit 1; }
    CONDITIONS="heldout0|heldout=0,h0=$H0 heldout1|heldout=1 heldout2|heldout=2 heldout3|heldout=3"
    [ "${VIS_SET}" = v4core ] && CONDITIONS="$CONDITIONS ground_grey|ground=0.15:0.15:0.15 ground_wood|ground=0.7:0.55:0.35 nogrid|gridcolor=0.278:0.451:0.584,nogrid=1"
fi
for c in $CONDITIONS; do
    IFS='|' read -r name spec <<< "$c"
    out="eval_runs/vis${TAG}_$name"
    if [ -f "$out/results.csv" ] && [ "$(($(wc -l < "$out/results.csv") - 1))" -ge 15 ]; then echo "$(date +%H:%M) $name already complete -- skipped"; continue; fi
    echo "$(date +%H:%M) $name ($spec) start"
    VLA_BRIDGE_VISUAL="$spec" scripts/eval_checkpoint.sh "$V36" "$out" 15 9200 25
    grep -m1 "VLA_BRIDGE_VISUAL" "$out/bridge.log" || [ "$spec" = none ] || echo "WARNING: perturbation not confirmed in bridge.log"
done
echo "$(date +%H:%M) visual evaluations finished"
