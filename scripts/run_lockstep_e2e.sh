#!/bin/bash
# check_lockstep_e2e.py: scripted stand-in client against a running lockstep bridge (no policy, no GPU).
# Use a FRESH bridge for each run (the sim is not reset between runs).
#   scripts/run_lockstep_e2e.sh --ticks 300 --horizon 25 --close_ticks 60
source "$(dirname "$0")/lib_isaac_env.sh"
cd "$REPO/isaac" || exit 1
exec "${CLEAN_ENV[@]}" LD_LIBRARY_PATH="$HUMBLE_LIB" PYTHONPATH="$HUMBLE_PY:$REPO/test/fakes/shim" \
  "$ISAAC_ENV/bin/python" -u check_lockstep_e2e.py "$@"
