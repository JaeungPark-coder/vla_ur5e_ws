#!/bin/bash
# One closed-loop evaluation of one checkpoint, start to finish: policy server (SERVER_GPU, default 1) -> warm-up ->
# lockstep bridge (GPU, default 0) -> client -> wait -> shut everything down.
#   scripts/eval_checkpoint.sh <checkpoint dir, absolute or relative to $OPENPI> <out dir> [n_trials=20] [seed=8000] [K=50]
# Trial k spawns the cube at default_rng(seed+k) (VLA_BRIDGE_SEED_REPEAT=1), so two checkpoints run with the same seed and
# n_trials face identical scenes. Output: <out>/results.csv, <out>/trace/ (per-tick traces), server/bridge/client logs.
source "$(dirname "$0")/lib_isaac_env.sh"
CKPT="$1"; OUT="$2"; N="${3:-20}"; SEED="${4:-8000}"; K="${5:-50}"
[ -n "$CKPT" ] && [ -n "$OUT" ] || { sed -n 2,7p "$0"; exit 1; }
SERVER_GPU="${SERVER_GPU:-1}"; PORT="${POLICY_PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi0_ur5e_pick_place}"   # v3 checkpoints: POLICY_CONFIG=pi0_ur5e_pick_place_v3 (matching norm stats)
mkdir -p "$OUT"; OUT="$(cd "$OUT" && pwd)"; rm -rf "$OUT/trace" "$OUT/results.csv"
stop_all() { kill -9 $(pgrep -f "vla_policy_client|pick_place_scene_bridge|serve_policy.py" | grep -vx "$$") 2>/dev/null; }
trap stop_all EXIT
stop_all; sleep 3

( cd "$OPENPI" && CUDA_VISIBLE_DEVICES="$SERVER_GPU" nohup uv run --no-sync scripts/serve_policy.py --port "$PORT" \
    policy:checkpoint --policy.config="$POLICY_CONFIG" --policy.dir="$CKPT" > "$OUT/server.log" 2>&1 & )
for _ in $(seq 1 120); do grep -q "server listening" "$OUT/server.log" && break; sleep 2; done
grep -q "server listening" "$OUT/server.log" || { echo "server did not start, see $OUT/server.log"; exit 1; }
# the norm stats must belong to the config (v3 weights with v2 statistics would be a silent mistake): print the lines and stop on a mismatch
grep "Loaded norm stats from" "$OUT/server.log"
case "$POLICY_CONFIG" in
  *_v3) grep "Loaded norm stats from" "$OUT/server.log" | grep -q "ur5e_pick_place_v3" || { echo "WRONG norm stats for $POLICY_CONFIG, see $OUT/server.log"; exit 1; } ;;
  *)    grep "Loaded norm stats from" "$OUT/server.log" | grep -q "ur5e_pick_place_v2" || { echo "unexpected norm stats for $POLICY_CONFIG, see $OUT/server.log"; exit 1; } ;;
esac
( cd "$OPENPI" && uv run --no-sync python "$REPO/scripts/warmup_policy_server.py" 2>&1 | tail -3 )

( GPU="$GPU" VLA_BRIDGE_SEED="$SEED" VLA_BRIDGE_SEED_REPEAT=1 VLA_BRIDGE_TRACE_DIR="$OUT/trace" VLA_BRIDGE_SAVE_IMAGES="${VLA_BRIDGE_SAVE_IMAGES:-}" VLA_BRIDGE_RESET_RENDERS="${VLA_BRIDGE_RESET_RENDERS:-}" \
    nohup "$REPO/scripts/run_lockstep_bridge.sh" > "$OUT/bridge.log" 2>&1 & )
for _ in $(seq 1 90); do grep -q "world step index after reset" "$OUT/bridge.log" && break; sleep 2; done
grep -q "world step index after reset" "$OUT/bridge.log" || { echo "bridge did not start, see $OUT/bridge.log"; exit 1; }

( nohup "$REPO/scripts/run_client_node.sh" -p eval_mode:=true -p n_trials:="$N" -p execute_horizon:="$K" \
    -p results_csv_path:="$OUT/results.csv" ${CLIENT_EXTRA_PARAMS:-} > "$OUT/client.log" 2>&1 & )
# ~1.2 min per trial; give up after 3x that
# "client gone" only counts as the end once the client has been seen running (checked right after the launch it is not
# visible yet: that race ended a run with 0 trials on 2026-11-12) or after 3 minutes without ever seeing it
seen=0
for i in $(seq 1 $((N * 6))); do
  [ "$(($(grep -c . "$OUT/results.csv" 2>/dev/null || echo 1) - 1))" -ge "$N" ] && break
  if pgrep -f vla_policy_client >/dev/null; then seen=1; elif [ "$seen" = 1 ] || [ "$i" -gt 6 ]; then break; fi
  sleep 30
done
sleep 5
echo "finished: $(($(grep -c . "$OUT/results.csv") - 1))/$N trials -> $OUT/results.csv"
