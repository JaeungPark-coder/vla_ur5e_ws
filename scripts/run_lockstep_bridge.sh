#!/bin/bash
# The Isaac Sim side of policy evaluation: pick_place_scene_bridge.py in lockstep mode, headless.
# Needs a stand-in for cv_bridge (no Python 3.11 build exists here): test/fakes/shim.
#   GPU=0 scripts/run_lockstep_bridge.sh
source "$(dirname "$0")/lib_isaac_env.sh"
cd "$REPO/isaac" || exit 1
exec "${CLEAN_ENV[@]}" CUDA_VISIBLE_DEVICES="$GPU" LD_PRELOAD="$ISAAC_ENV/lib/libstdc++.so.6" \
  LD_LIBRARY_PATH="$HUMBLE_LIB" PYTHONPATH="$REPO/test/fakes/shim" \
  ISAAC_PICK_PLACE_HEADLESS=1 VLA_BRIDGE_LOCKSTEP=1 \
  "$ISAAC_ENV/bin/python" -u pick_place_scene_bridge.py
