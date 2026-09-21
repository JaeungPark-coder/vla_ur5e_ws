"""Where, exactly, does the arm's state first go non-numeric?

check_cameras.py --sweep_wrist --with_gripper reproducibly hit PhysX
divergence ("Invalid PhysX transform detected" floods, "Illegal
BroadPhaseUpdateData" errors) and reach-check distances up to 5x10^11 mm --
a real numerical explosion, not measurement noise. That was found by reading
PhysX's own log warnings after the fact; this script instead watches the
articulation's own joint positions/velocities every single tick and stops
at the FIRST tick either goes non-finite, so the state immediately before
and after that tick is available instead of a wall of post-hoc log spam.

Also makes the GPU/CPU physics pipeline a CLI choice rather than whatever
SimulationManager auto-selects (GPU pipeline whenever a CUDA device is
present -- CONFIRMED in this environment by Warp's own startup banner).
That distinction matters here specifically: PhysicsContext disables CCD
(continuous collision detection) whenever the GPU pipeline is active
("Disable CCD for GPU dynamics as its not supported" -- see
isaacsim.core.api.physics_context.PhysicsContext.__init__), and this
project's gripper pads are exactly the kind of thin, convex-decomposed
geometry CCD exists to protect against tunneling through. If the same
sequence explodes on CPU too, the GPU pipeline is not the cause; if only
GPU explodes, CCD (or something else the GPU path disables) is a real lead.

    python3 check_rmpflow_stability.py                       # GPU (default), no gripper
    python3 check_rmpflow_stability.py --device cpu           # CPU pipeline, no gripper
    python3 check_rmpflow_stability.py --with-gripper          # GPU, WITH gripper (contact possible)
    python3 check_rmpflow_stability.py --device cpu --with-gripper
"""
import argparse
import os

import numpy as np

from isaacsim import SimulationApp

HEADLESS = os.environ.get("ISAAC_PICK_PLACE_HEADLESS", "1") != "0"
simulation_app = SimulationApp({"headless": HEADLESS})

# --- everything below must be imported AFTER SimulationApp() starts Kit ---
from isaacsim.core.simulation_manager import SimulationManager  # noqa: E402


def say(line=""):
    print(line, flush=True)


def first_nonfinite_index(arr):
    """Index of the first non-finite entry in a flat array, or None."""
    bad = np.where(~np.isfinite(arr))[0]
    return int(bad[0]) if bad.size else None


def run(device, with_gripper, max_ticks, episodes):
    if device is not None:
        # Read BEFORE forcing it, so the report says what was actually
        # requested vs. what was picked automatically (GPU whenever a CUDA
        # device is visible) -- see this file's own docstring.
        say(f"forcing physics sim device to {device!r} "
            f"(auto-selected would be {SimulationManager.get_physics_sim_device()!r})")
        SimulationManager.set_physics_sim_device(device)

    # Imported only now: pick_place_scene.py builds a physics scene at
    # import-adjacent construction time (PickPlaceScene.__init__ -> World()),
    # which is what actually reads SimulationManager's device setting.
    from pick_place_scene import PickPlaceScene, PLACE_TARGET_POSITION
    from scripted_pick_place import ScriptedPickPlace

    scene = PickPlaceScene(with_gripper=with_gripper)
    say(f"physics sim device in effect: {SimulationManager.get_physics_sim_device()!r}")
    say(f"scene built ({'WITH' if with_gripper else 'WITHOUT'} gripper)")

    # Looped rather than one episode: the divergence check_cameras.py hit was
    # NOT on every attempt (that sweep's log mixed normal ~20-30mm reaches
    # with occasional multi-km explosions), so it depends on something about
    # the specific random cube spawn / resulting contact geometry -- one
    # clean episode does not rule it out, only many do.
    max_residual_overall = 0.0
    for episode in range(1, episodes + 1):
        obs = scene.reset()
        policy = ScriptedPickPlace(obs["tool_pos"], scene.cube_position, PLACE_TARGET_POSITION)
        frames = list(policy.generate_frames())
        say(f"\n--- episode {episode}/{episodes}: cube at {np.round(scene.cube_position, 3).tolist()}, "
            f"driving {min(max_ticks, len(frames))}/{len(frames)} ticks ---")

        residuals = []
        for tick, (target_pos, target_rotvec, target_gripper) in enumerate(frames[:max_ticks], start=1):
            scene.step_towards(target_pos, target_rotvec, target_gripper)
            joint_pos = np.asarray(scene.robot.get_joint_positions(), dtype=float)
            joint_vel = np.asarray(scene.robot.get_joint_velocities(), dtype=float)

            pos_bad = first_nonfinite_index(joint_pos)
            vel_bad = first_nonfinite_index(joint_vel)
            if pos_bad is not None or vel_bad is not None:
                say(f"\n=== NON-FINITE STATE at episode {episode} tick {tick}/{len(frames)} ===")
                say(f"  cube position this episode: {np.round(scene.cube_position, 4).tolist()}")
                say(f"  target this tick: pos={np.round(target_pos, 4).tolist()} "
                    f"gripper={target_gripper:.2f}")
                if residuals:
                    last = residuals[-1]
                    say(f"  tool residual on the tick before this one: {last * 1000:.1f}mm")
                say(f"  joint_positions: {joint_pos.tolist()}")
                say(f"  joint_velocities: {joint_vel.tolist()}")
                say(f"  first non-finite index -- positions: {pos_bad}, velocities: {vel_bad}")
                say("\nVERDICT: a genuine numerical explosion (NaN/Inf), not just a large but "
                    "finite tracking residual -- whatever happened at this tick broke the "
                    "physics state itself, past which every further tick is meaningless.")
                return

            tool_pos = scene.grip_point_world() if with_gripper else scene.get_observation()["tool_pos"]
            residual = float(np.linalg.norm(np.asarray(tool_pos) - target_pos))
            residuals.append(residual)
            if residual > 1.0:
                say(f"  tick {tick:4d}: tool residual {residual * 1000:8.1f}mm  <-- large")

        ep_max = max(residuals)
        max_residual_overall = max(max_residual_overall, ep_max)
        say(f"  episode {episode} max residual: {ep_max * 1000:.1f}mm, "
            f"final: {residuals[-1] * 1000:.1f}mm")

    say(f"\nNo non-finite joint state across {episodes} episode(s).")
    say(f"largest residual seen across all episodes: {max_residual_overall * 1000:.1f}mm")
    if max_residual_overall > 1.0:
        say("VERDICT: stayed numerically finite, but tracking got very large at some point -- "
            "a gain/tuning problem, not a solver explosion.")
    else:
        say("VERDICT: clean. No explosion, no large residual, over any episode this run.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--device", choices=["cpu", "gpu"], default=None,
                        help="force the PhysX pipeline device; default leaves whatever "
                             "SimulationManager auto-selects (GPU if a CUDA device is visible)")
    parser.add_argument("--with-gripper", action="store_true",
                        help="include the gripper (contact with the cube is possible). Default "
                             "is bare arm only -- no gripper, no contact -- to isolate pure "
                             "RMPflow tracking from contact dynamics per this file's own purpose.")
    parser.add_argument("--max-ticks", type=int, default=2000,
                        help="stop after this many ticks even if nothing goes non-finite "
                             "(a full scripted episode is roughly 800-1000 ticks)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="number of fresh scene.reset() episodes to run -- the divergence "
                             "this script hunts for was NOT on every attempt in the sweep that "
                             "found it, so more episodes raise the odds of reproducing it")
    args = parser.parse_args()

    device = None if args.device is None else ("cuda:0" if args.device == "gpu" else "cpu")
    try:
        run(device, args.with_gripper, args.max_ticks, args.episodes)
    except BaseException:
        import traceback
        say("\n=== FAILED ===")
        say(traceback.format_exc())
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
