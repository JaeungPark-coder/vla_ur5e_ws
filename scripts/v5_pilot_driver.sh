#!/bin/bash
# v5 PILOT -- a pipeline test run, NOT for conclusions (README 2026-11-16 night).  Runs only if tonight's v5 expert smoke run
# passed its pre-set verdicts (eval_results/v5_smoke/summary.json smoke_pass); otherwise stops and the cause is written up.
#   collect ~50 stratified episodes (placeholders allowed) -> merge vla_ur5e_ws/ur5e_pick_place_v5pilot -> norm stats ->
#   1k-step training (pi0_ur5e_pick_place_v5pilot) -> centre 15-trial evaluation (K=25, v5 scene) -> v5_pilot_check.py
#   setsid nohup scripts/v5_pilot_driver.sh <smoke driver PID> > eval_runs/v5_pilot/driver.log 2>&1 < /dev/null &
# Every stage is skipped if its output is complete, so a rerun resumes.  Nothing is deleted except the merge's arrow cache.
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
REPO=$PWD
OPENPI_DIR="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
BIG=/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267
export HF_LEROBOT_HOME=$BIG/hf_cache
export VLA_SIM_PROFILE=v5
WAIT_PID="${1:-}"
OUT=eval_runs/v5_pilot
RES=eval_results/v5_pilot
mkdir -p "$OUT" "$RES"
say() { echo "$(date '+%m-%d %H:%M') $*"; }
say "v5 PILOT driver (pipeline test, not for conclusions)"

if [ -n "$WAIT_PID" ]; then
    say "waiting for PID $WAIT_PID (v5 smoke driver)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
SUM=eval_results/v5_smoke/summary.json
if ! python3 -c "import json,sys; sys.exit(0 if json.load(open('$SUM'))['smoke_pass'] is True else 1)" 2>/dev/null; then
    say "v5 expert smoke run did NOT pass (or no summary) -- pilot not started: $(python3 -c "import json;d=json.load(open('$SUM'));print(d['verdict'])" 2>&1)"
    exit 0
fi
say "smoke passed: $(python3 -c "import json;d=json.load(open('$SUM'));print(d['verdict'], d['total'])")"
sleep 20
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then say "GPU job running -- not starting"; exit 1; fi
free_g=$(df --output=avail -BG "$BIG" | tail -1 | tr -dc 0-9)
[ "$free_g" -ge 60 ] || { say "only ${free_g}G free on the big disk -- not starting"; exit 1; }

# ---- 1. collection: 50 saved episodes, stratified like the evaluation regions (centre 15, the other five 7 each) ----------
H0="0.85 0.85 0.83"
collect() {  # $1 GPU $2 region $3 episodes $4 visual seed $5 x0 $6 x1 $7 y0 $8 y1
    local log="$OUT/v5pilot_$2.log"
    if grep -q "^done -- dataset written" "$log" 2>/dev/null; then say "collect $2 already done -- skipped"; return 0; fi
    say "collect $2 ($3 episodes, GPU $1) start"
    setsid timeout 7200 env VLA_SIM_PROFILE=v5 GPU="$1" HF_LEROBOT_HOME="$HF_LEROBOT_HOME" scripts/isaac_py.sh collect_demos.py \
        --num_episodes "$3" --repo_id "vla_ur5e_ws/v5pilot_$2" --x_range "$5" "$6" --y_range "$7" "$8" \
        --reset_renders 30 --visual_random --visual_exclude_rgb $H0 --visual_seed "$4" --allow_placeholders --overwrite \
        > "$log" 2>&1 &
    local p=$!; wait $p; kill -- -$p 2>/dev/null
    say "collect $2 end: $(grep -m1 '^collected' "$log" || echo 'NOT finished -- see log')"
}
( collect 0 centre 15 601 0.42 0.48 -0.10 0.10
  collect 0 posY_strip 7 602 0.42 0.48 0.10 0.16
  collect 0 negY_strip 7 603 0.42 0.48 -0.16 -0.10 ) &
A=$!
( collect 1 posY_corner 7 604 0.38 0.42 0.11 0.16
  collect 1 negY_corner 7 605 0.38 0.42 -0.16 -0.11
  collect 1 mid_lowX 7 606 0.38 0.42 -0.11 0.11 ) &
B=$!
wait $A $B
REGIONS="centre posY_strip negY_strip posY_corner negY_corner mid_lowX"
SRC=(); expected=0
for r in $REGIONS; do
    log="$OUT/v5pilot_$r.log"
    grep -q "^done -- dataset written" "$log" || { say "collect $r did not finish -- stopping"; exit 1; }
    SRC+=("'vla_ur5e_ws/v5pilot_$r'")
    expected=$((expected + $(python3 -c "import json;print(json.load(open('$HF_LEROBOT_HOME/vla_ur5e_ws/v5pilot_$r/meta/info.json'))['total_episodes'])")))
done
say "collected $expected episodes in total"

# ---- 2. merge -> vla_ur5e_ws/ur5e_pick_place_v5pilot, link as Jaeung12/ur5e_pick_place_v5pilot -------------------------------
DS=$HF_LEROBOT_HOME/vla_ur5e_ws/ur5e_pick_place_v5pilot
MERGE_CACHE=$BIG/hf_cache/merge_tmp_datasets_cache_v5pilot
if [ -f "$DS/.merge_ok" ]; then
    say "merge already done -- skipped"
else
    [ -e "$DS" ] && { say "partial merge output $DS exists -- remove it by hand first (ask)"; exit 1; }
    say "merge start"
    mkdir -p "$MERGE_CACHE"
    list=$(IFS=,; echo "${SRC[*]}")
    HF_HUB_OFFLINE=1 HF_DATASETS_CACHE="$MERGE_CACHE" /home/icrs/bigdisk/conda_envs/env_isaaclab/bin/lerobot-edit-dataset \
        --repo_id vla_ur5e_ws/ur5e_pick_place_v5pilot --operation.type merge --operation.repo_ids "[$list]" \
        > "$OUT/merge.log" 2>&1; rc=$?
    rm -rf /media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/hf_cache/merge_tmp_datasets_cache_v5pilot
    [ $rc -eq 0 ] || { say "merge FAILED (see $OUT/merge.log)"; exit 1; }
    python3 -c "import json,sys;d=json.load(open('$DS/meta/info.json'));sys.exit(d['total_episodes']!=$expected)" \
        || { say "merged episode count is not $expected"; exit 1; }
    touch "$DS/.merge_ok"
fi
say "merged: $(python3 -c "import json;d=json.load(open('$DS/meta/info.json'));print('episodes',d['total_episodes'],'frames',d['total_frames'])")"
LINK=$HOME/.cache/huggingface/lerobot/Jaeung12/ur5e_pick_place_v5pilot
[ -e "$LINK" ] || ln -s "$DS" "$LINK"
unset HF_LEROBOT_HOME

# ---- 3. norm stats --------------------------------------------------------------------------------------------------------
NS="$OPENPI_DIR/assets/pi0_ur5e_pick_place_v5pilot/Jaeung12/ur5e_pick_place_v5pilot/norm_stats.json"
if [ -f "$NS" ]; then
    say "norm stats already done -- skipped"
else
    say "norm stats start"
    (cd "$OPENPI_DIR" && env -u HF_LEROBOT_HOME uv run --no-sync python "$REPO/scripts/compute_norm_stats_workers.py" \
        pi0_ur5e_pick_place_v5pilot --workers 32 > "$REPO/$OUT/norm_stats.log" 2>&1)
    [ -f "$NS" ] || { say "norm stats FAILED (see $OUT/norm_stats.log)"; exit 1; }
fi

# ---- 4. training: 1k steps, the v4 recipe scaled down ----------------------------------------------------------------------
CK=checkpoints/pi0_ur5e_pick_place_v5pilot/v5pilot_1k/999
if [ -d "$OPENPI_DIR/$CK" ]; then
    say "training already done -- skipped"
else
    say "1k training start"
    (cd "$OPENPI_DIR" && env -u HF_LEROBOT_HOME CUDA_VISIBLE_DEVICES=0,1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run --no-sync scripts/train.py \
        pi0_ur5e_pick_place_v5pilot --exp-name=v5pilot_1k --num-train-steps=1000 --lr-schedule.decay-steps=1000 \
        --lr-schedule.warmup-steps=100 --save-interval=1000 --keep-period=1000 --log-interval=50 --num-workers=16 \
        --no-wandb-enabled > "$REPO/$OUT/train.log" 2>&1)
    say "training ended"
    [ -d "$OPENPI_DIR/$CK" ] || { say "no checkpoint $CK -- stopping (see $OUT/train.log)"; exit 1; }
fi
sleep 20

# ---- 5. centre evaluation, 15 trials, K=25, the v5 scene ---------------------------------------------------------------------
EV=eval_runs/v5pilot_1k_centre
if [ -f "$EV/results.csv" ] && [ "$(($(wc -l < "$EV/results.csv") - 1))" -ge 15 ]; then
    say "evaluation already done -- skipped"
else
    say "centre evaluation start"
    VLA_SIM_PROFILE=v5 VLA_BRIDGE_SAVE_IMAGES=1 VLA_BRIDGE_RESET_RENDERS=30 POLICY_CONFIG=pi0_ur5e_pick_place_v5pilot \
        VLA_BRIDGE_CUBE_X_RANGE=0.42,0.48 VLA_BRIDGE_CUBE_Y_RANGE=-0.10,0.10 \
        scripts/eval_checkpoint.sh "$CK" "$EV" 15 9200 25 > "$OUT/eval.log" 2>&1
    say "evaluation end: $(tail -1 "$OUT/eval.log")"
fi
grep -q "sim profile v5" "$EV/bridge.log" && say "bridge ran the v5 scene" || say "WARNING: bridge log does not show the v5 profile"

# ---- 6. pilot check ---------------------------------------------------------------------------------------------------------
DIRS=()
for r in $REGIONS; do DIRS+=("$BIG/hf_cache/vla_ur5e_ws/v5pilot_$r"); done
python3 scripts/v5_pilot_check.py --datasets "${DIRS[@]}" --eval_dir "$EV" --out "$RES/pilot_check.json" > "$OUT/pilot_check.log" 2>&1
say "pilot check: $(python3 -c "
import json; d=json.load(open('$RES/pilot_check.json'))['summary']
print('start_pose_as_intended', d['start_pose_as_intended'], '| gripper_as_intended_in_data', d['gripper_as_intended_in_data'],
      '| data', d['dataset']['ok'], '/', d['dataset']['episodes'], '| eval', d['evaluation'])")"
n=$(($(wc -l < "$EV/results.csv" 2>/dev/null || echo 1) - 1))
[ "$n" -ge 15 ] && say "v5 PILOT finished end to end (15/15 trials logged; results are a pipeline test, not for conclusions)" \
                || say "v5 PILOT INCOMPLETE: evaluation logged $n/15 trials"
