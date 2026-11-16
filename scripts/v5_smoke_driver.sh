#!/bin/bash
# v5 GPU checks, unattended, after tonight's v4 jobs (README "Tomorrow on GPU, verdicts set now" and 2026-11-16 night notes):
#   1. v5 mount render (measured fingertip shift while wrist_3 turns 0/90/180 deg; pass <= 2 mm)
#   2. v5 expert smoke run, 2F-85, placeholders allowed, stratified by the evaluation regions: 20 saved episodes =
#      centre 5 + 3 in each of posY_strip, negY_strip, posY_corner, negY_corner, mid_lowX; two GPUs in parallel
#   3. v5_expert_check.py over all six datasets, then scripts/v5_smoke_summary.py against the pre-set verdicts
#   setsid nohup scripts/v5_smoke_driver.sh <PID to wait for> > eval_runs/v5_smoke/driver.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
WAIT_PID="${1:-}"
OUT=eval_runs/v5_smoke
RES=eval_results/v5_smoke
mkdir -p "$OUT" "$RES"
export HF_LEROBOT_HOME=/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/hf_cache
say() { echo "$(date '+%m-%d %H:%M') $*"; }

if [ -n "$WAIT_PID" ]; then
    say "waiting for PID $WAIT_PID (v4 6k evaluation -> v4 mount render)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi
sleep 30
if pgrep -f "scripts/train.py|serve_policy|kit/python" >/dev/null; then say "GPU job still running -- not starting"; exit 1; fi
say "v4 jobs done: $(grep -E 'v4 6k finished|INCOMPLETE|stopping' eval_runs/v4_6k_driver.log | tail -1)"

# ---- 1. v5 mount render ---------------------------------------------------------------------------------------------
say "v5 mount render start"
setsid timeout 1800 env VLA_SIM_PROFILE=v5 GPU=0 scripts/isaac_py.sh render_mount_check.py --out eval_results/mount_check_v5 \
    > "$OUT/mount_check_v5.log" 2>&1 &
p=$!; wait $p; kill -- -$p 2>/dev/null
say "v5 mount render end: $(grep -m1 '^VERDICT' "$OUT/mount_check_v5.log" || echo 'no verdict -- see log')"

# ---- 2. v5 expert smoke run, stratified ------------------------------------------------------------------------------
H0="0.85 0.85 0.83"   # the held-out plain ground colour, as in v4
collect() {  # $1 GPU $2 region $3 episodes $4 visual seed $5 x0 $6 x1 $7 y0 $8 y1
    local log="$OUT/v5smoke_$2.log"
    say "collect $2 ($3 episodes, GPU $1) start"
    setsid timeout 3000 env VLA_SIM_PROFILE=v5 GPU="$1" HF_LEROBOT_HOME="$HF_LEROBOT_HOME" scripts/isaac_py.sh collect_demos.py \
        --num_episodes "$3" --repo_id "vla_ur5e_ws/v5smoke_$2" --x_range "$5" "$6" --y_range "$7" "$8" \
        --reset_renders 30 --visual_random --visual_exclude_rgb $H0 --visual_seed "$4" --allow_placeholders --overwrite \
        > "$log" 2>&1 &
    local p=$!; wait $p; kill -- -$p 2>/dev/null   # image-writer children keep GPU memory otherwise (README 2026-11-12)
    say "collect $2 end: $(grep -m1 '^collected' "$log" || echo 'not finished -- see log')"
}
( collect 0 centre 5 501 0.42 0.48 -0.10 0.10
  collect 0 posY_strip 3 502 0.42 0.48 0.10 0.16
  collect 0 negY_strip 3 503 0.42 0.48 -0.16 -0.10 ) &
A=$!
( collect 1 posY_corner 3 504 0.38 0.42 0.11 0.16
  collect 1 negY_corner 3 505 0.38 0.42 -0.16 -0.11
  collect 1 mid_lowX 3 506 0.38 0.42 -0.11 0.11 ) &
B=$!
wait $A $B

# ---- 3. checks against the pre-set verdicts -------------------------------------------------------------------------
DIRS=()
for r in centre posY_strip negY_strip posY_corner negY_corner mid_lowX; do
    d="$HF_LEROBOT_HOME/vla_ur5e_ws/v5smoke_$r"
    [ -d "$d/data" ] && DIRS+=("$d")
done
if [ ${#DIRS[@]} -gt 0 ]; then
    python3 scripts/v5_expert_check.py "${DIRS[@]}" --tcp_m 0.12 --json "$RES/expert_check.json" > "$OUT/expert_check.log" 2>&1
fi
python3 scripts/v5_smoke_summary.py "$OUT" "$RES/expert_check.json" eval_results/mount_check_v5/mount_check.json \
    --out "$RES/summary.json" > "$OUT/summary.log" 2>&1
say "summary: $(python3 -c "import json;d=json.load(open('$RES/summary.json'));print(d['verdict'], 'smoke_pass', d['smoke_pass'], d['total'])")"
say "v5 smoke finished"
