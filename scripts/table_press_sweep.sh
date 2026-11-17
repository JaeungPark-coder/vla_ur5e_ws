#!/bin/bash
# Contact-setting sweep against the table-press test (README 2026-11-17), v5 profile, two GPUs in parallel.
#   setsid nohup scripts/table_press_sweep.sh > eval_runs/table_press/sweep_driver.log 2>&1 < /dev/null &
cd "$(dirname "$0")/.." || exit 1
source scripts/lib_cache_env.sh || exit 1
mkdir -p eval_runs/table_press eval_results/table_press
say() { echo "$(date '+%m-%d %H:%M') $*"; }
run() {  # $1 GPU $2 label $3 tune JSON
    say "sweep $2 start (GPU $1): $3"
    setsid timeout 2400 env VLA_SIM_PROFILE=v5 GPU="$1" VLA_PHYSICS_TUNE="$3" scripts/isaac_py.sh table_press_test.py \
        --out "$PWD/eval_results/table_press/v5_$2" > "eval_runs/table_press/v5_$2.log" 2>&1 &
    local p=$!; wait $p; kill -- -$p 2>/dev/null
    say "sweep $2 end: $(grep -m1 '^SUMMARY\|^EXC' "eval_runs/table_press/v5_$2.log" | cut -c1-200)"
}
( run 0 T1_depen_all '{"depen_all": 0.5}'
  run 0 T3_physics240 '{"physics_hz": 240}'
  run 0 T5_arm_force '{"arm_max_force": [150, 150, 150, 28, 28, 28]}' ) &
A=$!
( run 1 T2_solver32 '{"solver_pos": 32, "solver_vel": 4}'
  run 1 T4_offsets '{"contact_offset": 0.01, "rest_offset": 0.0}'
  run 1 T6_combo '{"depen_all": 0.5, "solver_pos": 32, "solver_vel": 4, "physics_hz": 240}' ) &
B=$!
wait $A $B
say "sweep finished"
