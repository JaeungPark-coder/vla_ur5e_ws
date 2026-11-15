#!/bin/bash
# v4 (visual randomization) end to end, unattended: collect -> merge -> norm stats -> 3k training -> core evaluation.
# Design, prediction and decision rule: README "v4 (visual randomization) -- design" (written before any v4 data).
#   V4_H0="r:g:b"   the held-out plain ground colour (lab table colour), excluded from training colours  [required]
#   setsid nohup scripts/v4_pipeline.sh > eval_runs/v4_pipeline.log 2>&1 < /dev/null &
# Every stage is skipped if its output is already complete, so a rerun resumes.
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
REPO=$PWD
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
export HF_LEROBOT_HOME=/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/hf_cache
H0="${V4_H0:?set V4_H0=r:g:b (held-out plain ground colour)}"
H0_SPACE="${H0//:/ }"
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then echo "$(date +%H:%M) GPU job running -- not starting"; exit 1; fi
mkdir -p eval_runs/collect
say() { echo "$(date '+%m-%d %H:%M') $*"; }

# ---- 1. collection: 152 old box + 150 wide box (v3's spawn mix), randomized scenes, converged first frame -------------------
OLD="--x_range 0.42 0.48 --y_range -0.10 0.10"
WIDE="--x_range 0.38 0.48 --y_range -0.16 0.16"
collect_set() {  # $1 GPU $2 set name $3 episodes $4 visual seed, rest: box args
    local g=$1 name=$2 n=$3 seed=$4; shift 4
    local log=eval_runs/collect/v4_$name.log
    if grep -q "^done -- dataset written" "$log" 2>/dev/null; then say "collect $name already done -- skipped"; return 0; fi
    say "collect $name ($n episodes, GPU $g) start"
    setsid env GPU=$g HF_LEROBOT_HOME=$HF_LEROBOT_HOME scripts/isaac_py.sh collect_demos.py --num_episodes "$n" \
        --repo_id "vla_ur5e_ws/v4_$name" "$@" --reset_renders 30 --visual_random --visual_exclude_rgb $H0_SPACE \
        --visual_seed "$seed" --overwrite > "$log" 2>&1 &
    local pid=$!
    wait $pid
    kill -- -$pid 2>/dev/null   # finished collectors leave image-writer children holding GPU memory (README 2026-11-12)
    grep -q "^done -- dataset written" "$log" || { say "collect $name FAILED (see $log)"; return 1; }
    say "collect $name done: $(grep -m1 '^collected' "$log")"
}
( collect_set 0 old_a 76 401 $OLD && collect_set 0 wide_a 75 403 $WIDE ) & A=$!
( collect_set 1 old_b 76 402 $OLD && collect_set 1 wide_b 75 404 $WIDE ) & B=$!
wait $A; ra=$?; wait $B; rb=$?
[ $ra -eq 0 ] && [ $rb -eq 0 ] || { say "collection failed -- stopping"; exit 1; }

# ---- 2. merge -> vla_ur5e_ws/ur5e_pick_place_v4, link as Jaeung12/ur5e_pick_place_v4 -------------------------------------
DS=$HF_LEROBOT_HOME/vla_ur5e_ws/ur5e_pick_place_v4
if [ -f "$DS/meta/info.json" ]; then
    say "merge already done -- skipped"
else
    say "merge start"
    HF_HUB_OFFLINE=1 /home/icrs/bigdisk/conda_envs/env_isaaclab/bin/lerobot-edit-dataset --repo_id vla_ur5e_ws/ur5e_pick_place_v4 \
        --operation.type merge \
        --operation.repo_ids "['vla_ur5e_ws/v4_old_a','vla_ur5e_ws/v4_old_b','vla_ur5e_ws/v4_wide_a','vla_ur5e_ws/v4_wide_b']" \
        > eval_runs/collect/v4_merge.log 2>&1 || { say "merge FAILED"; exit 1; }
fi
python3 -c "import json;d=json.load(open('$DS/meta/info.json'));print('merged episodes',d['total_episodes'],'frames',d['total_frames'])"
cat $HF_LEROBOT_HOME/vla_ur5e_ws/v4_{old_a,old_b,wide_a,wide_b}_episode_visuals.jsonl > eval_runs/collect/v4_episode_visuals_all.jsonl
LINK=$HOME/.cache/huggingface/lerobot/Jaeung12/ur5e_pick_place_v4
[ -e "$LINK" ] || ln -s "$DS" "$LINK"

# ---- 3. norm stats --------------------------------------------------------------------------------------------------------
NS="$OPENPI_DIR/assets/pi0_ur5e_pick_place_v4/Jaeung12/ur5e_pick_place_v4/norm_stats.json"
if [ -f "$NS" ]; then
    say "norm stats already done -- skipped"
else
    say "norm stats start"
    (cd "$OPENPI_DIR" && uv run --no-sync python "$REPO/scripts/compute_norm_stats_workers.py" pi0_ur5e_pick_place_v4 --workers 32 \
        > "$REPO/eval_runs/collect/norm_stats_v4.log" 2>&1)
    [ -f "$NS" ] || { say "norm stats FAILED"; exit 1; }
fi

# ---- 4. training: the v3 3k recipe -----------------------------------------------------------------------------------------
CK=checkpoints/pi0_ur5e_pick_place_v4/v4_3k/2999
if [ -d "$OPENPI_DIR/$CK" ]; then
    say "training already done -- skipped"
else
    say "v4 3k training start"
    (cd "$OPENPI_DIR" && CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train.py pi0_ur5e_pick_place_v4 \
        --exp-name=v4_3k --num-train-steps=3000 --lr-schedule.decay-steps=3000 --lr-schedule.warmup-steps=300 --save-interval=1000 \
        --keep-period=3000 --log-interval=50 --num-workers=16 --no-wandb-enabled > "$OPENPI_DIR/v4_3k.log" 2>&1)
    say "v4 3k training ended"
    [ -d "$OPENPI_DIR/$CK" ] || { say "no checkpoint $CK -- stopping"; exit 1; }
fi
sleep 20

# ---- 5. core evaluation (~4.3 h): v4 base scene stratified (K=50), v4 held-out + ground conditions, v3 3k held-out ----------
export VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30
REGIONS="centre:0.42,0.48:-0.10,0.10 posY_strip:0.42,0.48:0.10,0.16 negY_strip:0.42,0.48:-0.16,-0.10 posY_corner:0.38,0.42:0.11,0.16 negY_corner:0.38,0.42:-0.16,-0.11 mid_lowX:0.38,0.42:-0.11,0.11"
for r in $REGIONS; do
    IFS=: read -r name xr yr <<< "$r"
    out=eval_runs/strat_v4_3k_$name
    if [ -f "$out/results.csv" ] && [ "$(($(wc -l < "$out/results.csv") - 1))" -ge 15 ]; then continue; fi
    say "strat v4 3k $name start"
    POLICY_CONFIG=pi0_ur5e_pick_place_v4 VLA_BRIDGE_CUBE_X_RANGE="$xr" VLA_BRIDGE_CUBE_Y_RANGE="$yr" \
        scripts/eval_checkpoint.sh "$CK" "$out" 15 9200 50
done
VIS_SET=v4core V4_H0="$H0" VIS_CKPT="$CK" VIS_CONFIG=pi0_ur5e_pick_place_v4 VIS_TAG=_v4_3k scripts/visual_eval_driver.sh
VIS_SET=heldout V4_H0="$H0" VIS_CKPT=checkpoints/pi0_ur5e_pick_place_v3/v3_3k/2999 VIS_CONFIG=pi0_ur5e_pick_place_v3 VIS_TAG=_v3_3k \
    scripts/visual_eval_driver.sh
say "v4 pipeline finished"
