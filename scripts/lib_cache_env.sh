#!/bin/bash
# Source this at the top of every driver that trains, serves or reads the dataset:
#     source "$(dirname "$0")/lib_cache_env.sh" || exit 1
# It exports the three cache variables (bigdisk) and then REFUSES to continue unless they are set, point to existing directories
# off the root disk, and the pi0_base weights are where openpi will look for them.
# Why (2026-11-14): twice a run started without them (scripts run under `env -i` / setsid lose the interactive environment); the second
# time openpi began re-downloading pi0_base (11 GB) into ~/.cache/openpi and datasets built a 26 GB Arrow cache on / (root 87% full).
BIG=/media/icrs/0549ec1c-684e-41a5-beca-53599cda1267
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$BIG/hf_cache/datasets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$BIG/openpi_cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$BIG/uv_cache}"
export PYTHONUNBUFFERED=1

require_cache_env() {
    local bad=0 v d
    for v in HF_DATASETS_CACHE OPENPI_DATA_HOME UV_CACHE_DIR; do
        d="${!v}"
        if [ -z "$d" ]; then echo "cache check: $v is not set" >&2; bad=1; continue; fi
        if [ ! -d "$d" ]; then echo "cache check: $v=$d does not exist" >&2; bad=1; continue; fi
        case "$(cd "$d" && pwd -P)" in
            "$HOME"/*|/root/*|/tmp/*) echo "cache check: $v=$d is on the root disk" >&2; bad=1 ;;
        esac
    done
    if [ ! -d "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi0_base/params" ]; then
        echo "cache check: pi0_base weights not found under $OPENPI_DATA_HOME (openpi would download 11 GB)" >&2; bad=1
    fi
    if [ "$bad" -ne 0 ]; then echo "cache check FAILED -- not starting" >&2; return 1; fi
    echo "cache check ok: HF_DATASETS_CACHE=$HF_DATASETS_CACHE OPENPI_DATA_HOME=$OPENPI_DATA_HOME UV_CACHE_DIR=$UV_CACHE_DIR"
}
require_cache_env
