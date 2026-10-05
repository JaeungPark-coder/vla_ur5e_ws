#!/bin/bash
# Run one Isaac Sim python script (replay checks, noise study, collect_demos ...) on one GPU.
#   GPU=1 scripts/isaac_py.sh check_action_replay.py --episodes 3 --out /tmp/replay.json
# (LD_PRELOAD is needed: README "Running the Isaac Sim scripts".)
source "$(dirname "$0")/lib_isaac_env.sh"
cd "$REPO/isaac" || exit 1
CUDA_VISIBLE_DEVICES="$GPU" LD_PRELOAD="$ISAAC_ENV/lib/libstdc++.so.6" exec "$ISAAC_ENV/bin/python" -u "$@"
