#!/bin/bash
# The Isaac Sim side of policy evaluation: pick_place_scene_bridge.py in lockstep mode, headless.
# Needs a stand-in for cv_bridge (no Python 3.11 build exists here): test/fakes/shim.
#   GPU=0 scripts/run_lockstep_bridge.sh
#   GPU=0 VLA_BRIDGE_SEED=7000 scripts/run_lockstep_bridge.sh   # trial k spawns the cube at default_rng(7000+k): paired runs
# NOTE: this script runs under `env -i`, so only the variables listed below reach the bridge.
source "$(dirname "$0")/lib_isaac_env.sh"
cd "$REPO/isaac" || exit 1
exec "${CLEAN_ENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" LD_PRELOAD="$ISAAC_ENV/lib/libstdc++.so.6" \
  LD_LIBRARY_PATH="$HUMBLE_LIB" PYTHONPATH="$REPO/test/fakes/shim" \
  ISAAC_PICK_PLACE_HEADLESS=1 VLA_BRIDGE_LOCKSTEP=1 VLA_BRIDGE_SEED="${VLA_BRIDGE_SEED:-}" VLA_BRIDGE_SEED_REPEAT="${VLA_BRIDGE_SEED_REPEAT:-1}" \
  VLA_BRIDGE_TRACE_DIR="${VLA_BRIDGE_TRACE_DIR:-}" VLA_BRIDGE_CUBE_X_RANGE="${VLA_BRIDGE_CUBE_X_RANGE:-}" \
  VLA_BRIDGE_CUBE_Y_RANGE="${VLA_BRIDGE_CUBE_Y_RANGE:-}" \
  "$ISAAC_ENV/bin/python" -u pick_place_scene_bridge.py
