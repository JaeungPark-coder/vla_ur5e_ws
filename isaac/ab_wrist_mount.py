"""Focused A/B test between the two wrist-mount candidates left in
disagreement by the 2026-09-21 merge (see README.md's "wrist mount" open
item):

    A: (180, 0, 0) @ 0.16m   -- this session's (2026-09-21) own measurement,
                                currently committed in pick_place_scene.py
    B: (-90, 0, 0) @ 0.12m   -- the parallel session's (2026-09-19) own
                                measurement, at the time also scored best

Both were measured under a DOME LIGHT already in place (both sessions
independently added the same fix), but never against each other under the
SAME merged code state (current mass fix, cube range, friction, gripper
stiffness fix all together) until now. Reimplements (rather than imports)
check_cameras.py's own _render_wrist/_metrics/_drive_to_grasp logic --
CONFIRMED 2026-09-22: check_cameras.py calls SimulationApp(...) itself at
MODULE level, unconditionally (not guarded by `if __name__ == "__main__"`),
so importing anything from it after this script's own SimulationApp() call
already started Kit creates a SECOND app instance in the same process --
this crashed Kit's own app-registry (`libomni.kit.app.plugin`, a native
std::unordered_map corrupted by the double-init) 3 times in a row before
this was found, each with an identical crash signature. Do not import from
check_cameras.py from any other script for this reason.

    python3 ab_wrist_mount.py --samples 8
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION  # noqa: E402
from scripted_pick_place import ScriptedPickPlace  # noqa: E402

CANDIDATES = [
    ("A: (180,0,0)@0.16 [2026-09-21]", (180.0, 0.0, 0.0), 0.16),
    ("B: (-90,0,0)@0.12 [2026-09-19]", (-90.0, 0.0, 0.0), 0.12),
]


def say(line=""):
    print(line, flush=True)


def _drive_to_grasp(scene, obs):
    """Same as check_cameras.py's own _drive_to_grasp (reimplemented here,
    not imported -- see module docstring for why importing that module is
    unsafe)."""
    policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
    frames = list(policy.generate_frames())
    n = policy.frames_until_grasp()
    for target_pos, target_rotvec, target_gripper in frames[:n]:
        scene.step_towards(target_pos, target_rotvec, target_gripper)
    return scene.get_observation()


def _render_wrist(scene, euler, lateral=None, down_tilt=None):
    """Same as check_cameras.py's own _render_wrist."""
    scene.set_wrist_camera_flange_rotation(euler, lateral_m=lateral, down_tilt_deg=down_tilt)
    for _ in range(2):
        scene.world.render()
    return scene.get_observation()["wrist_rgb"]


def _metrics(frame, scene):
    """Same as check_cameras.py's own _metrics."""
    rgb = np.asarray(frame)[..., :3].astype(np.float32)
    return {
        "dark": float((rgb.max(axis=2) < 12).mean()),
        "cube_px": scene.cube_pixels_visible(frame),
    }


def run(samples, seed):
    scene = PickPlaceScene(with_gripper=True)
    scene._rng = np.random.default_rng(seed)
    say(f"cube-spawn RNG seeded with {seed} -- SAME spawn sequence for both "
        f"candidates (paired comparison, not two independent draws)")

    results = {label: [] for label, _, _ in CANDIDATES}
    for i in range(samples):
        obs = scene.reset()
        obs = _drive_to_grasp(scene, obs)
        for label, euler, lateral in CANDIDATES:
            frame = _render_wrist(scene, euler, lateral, down_tilt=0.0)
            m = _metrics(frame, scene)
            results[label].append(m)
            say(f"  sample {i + 1}/{samples}  {label:32s} cube_px={m['cube_px']:6d} "
                f"dark={100 * m['dark']:5.1f}%")

    say("\n=== SUMMARY (paired over the same {} cube spawns) ===".format(samples))
    for label, _, _ in CANDIDATES:
        px = [m["cube_px"] for m in results[label]]
        dark = [m["dark"] for m in results[label]]
        say(f"{label:32s} worst={min(px):6d} mean={sum(px) / len(px):7.1f} "
            f"max={max(px):6d}  near_black_mean={100 * sum(dark) / len(dark):5.1f}%")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        run(args.samples, args.seed)
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
