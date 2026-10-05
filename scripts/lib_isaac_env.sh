# Sourced by the other scripts. Override any of these from the environment.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-/home/icrs/bigdisk/conda_envs/env_isaaclab}"
ISAAC_PKG="$ISAAC_ENV/lib/python3.11/site-packages/isaacsim"
OPENPI="${OPENPI:-/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267/apps/openpi}"
HUMBLE_LIB="$ISAAC_PKG/exts/isaacsim.ros2.bridge/humble/lib"
HUMBLE_PY="$ISAAC_PKG/exts/isaacsim.ros2.bridge/humble/rclpy"
ROS_DOMAIN="${ROS_DOMAIN_ID:-77}"
GPU="${GPU:-0}"
# `env -i`: the interactive shell's PYTHONPATH/LD_LIBRARY_PATH point at the SYSTEM ROS 2 (Python 3.10);
# mixing it with Isaac's bundled Humble crashes the bridge in FastDDS discovery (README 2026-09-23).
CLEAN_ENV=(env -i HOME="$HOME" PATH=/usr/bin:/bin USER="$USER" DISPLAY=
           ROS_DISTRO=humble RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID="$ROS_DOMAIN")
