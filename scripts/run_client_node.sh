#!/bin/bash
# The real vla_policy_client ROS 2 node, with the validated serving recipe. Extra args are ROS parameters
# and come last, so they override the recipe:
#   scripts/run_client_node.sh                                  # real policy server at localhost:8000
#   POLICY_HOST=gpu-box scripts/run_client_node.sh -p eval_mode:=true -p n_trials:=20
#   FAKE_POLICY=1 scripts/run_client_node.sh -p max_steps:=200  # no model needed (test/fakes)
# Start the bridge first (scripts/run_lockstep_bridge.sh), same ROS_DOMAIN_ID.
source "$(dirname "$0")/lib_isaac_env.sh"
cd "$REPO" || exit 1
if [ "${FAKE_POLICY:-0}" = 1 ]; then
  CLIENT_PKG="$REPO/test/fakes"
  EXTRA=(FAKE_POLICY_EPISODE="${FAKE_POLICY_EPISODE:-0}" FAKE_POLICY_LOG="${FAKE_POLICY_LOG:-/tmp/fake_policy.jsonl}")
else
  CLIENT_PKG="$OPENPI/packages/openpi-client/src"
  EXTRA=()
fi
exec "${CLEAN_ENV[@]}" "${EXTRA[@]}" LD_LIBRARY_PATH="$HUMBLE_LIB" \
  PYTHONPATH="$CLIENT_PKG:$HUMBLE_PY:$REPO/test/fakes/shim:$REPO/src/vla_bridge" \
  "$ISAAC_ENV/bin/python" -u -c "from vla_bridge.vla_policy_client import main; main()" --ros-args \
  -p robot_backend:=isaac_sim -p lockstep:=true -p velocity_feedforward:=true -p execute_horizon:=25 \
  -p policy_host:="${POLICY_HOST:-localhost}" -p policy_port:="${POLICY_PORT:-8000}" "$@"
