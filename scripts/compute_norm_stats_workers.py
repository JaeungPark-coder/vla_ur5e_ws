"""openpi's compute_norm_stats.py with more DataLoader workers (it takes --max-frames but no worker option; the default of 2 left
34 of 36 cores idle decoding PNGs and would take ~4 h on this dataset: README 2026-11-10 'Runtime lesson').

    cd $OPENPI && HF_DATASETS_CACHE=<bigdisk>/hf_cache/datasets HF_HUB_OFFLINE=1 \
        uv run --no-sync python <repo>/scripts/compute_norm_stats_workers.py pi0_ur5e_pick_place_v3 --workers 32

Needs the `if __name__ == "__main__"` guard: the DataLoader workers are spawned.
"""
import argparse
import dataclasses
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config_name")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    sys.path.insert(0, os.path.join(os.getcwd(), "scripts"))   # run from the openpi checkout
    import compute_norm_stats as cns

    real_get_config = cns._config.get_config

    def get_config(name):
        return dataclasses.replace(real_get_config(name), num_workers=args.workers)

    cns._config.get_config = get_config
    cns.main(args.config_name, args.max_frames)


if __name__ == "__main__":
    main()
