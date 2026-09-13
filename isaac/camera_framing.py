"""How big does the task actually look? Camera framing, computed.

Pure numpy and `ast` -- no Isaac Sim import -- so this answers framing
questions on any machine, before booting the simulator. It reads the live
constants straight out of pick_place_scene.py and isaac_sim_common.py rather
than restating them, so it cannot drift from the scene it is describing.

WHY THIS EXISTS

`check_cameras.py` renders what the cameras see, which is the right tool for
"is this aimed at the task". It cannot answer "how big CAN the cube get, and
what would it take to make it bigger" -- that is geometry, and answering it
by rendering means booting Isaac Sim to discover a number that a few lines
of trigonometry already determine.

It matters because the base camera's measured peak of 139 red pixels across
a whole episode was read as an occlusion problem (the arm dominating the
frame). Run the geometry and 139 is simply what a 4 cm cube subtends at that
distance through that field of view: about 13 px across, so roughly 140-200
px of area depending on how many faces show. Nothing was occluding it. A fix
aimed at occlusion would not have moved the number.

THE ONE RELATION WORTH KNOWING

Distance and field of view only ever appear together, as the width of the
swath the camera covers at the object's distance:

    visible_width = 2 * distance * tan(hfov / 2)
    span_px       = resolution_px * object_size / visible_width

So the cube's pixel size depends on nothing but how wide a strip of world
the image spans. Moving closer and narrowing the lens are the same lever,
and -- the part that constrains this scene -- everything that must stay in
frame sets a floor on visible_width, which sets a ceiling on span_px that no
camera placement can beat.
"""
import argparse
import ast
import dataclasses
import os

import numpy as np

SCENE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pick_place_scene.py")
COMMON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "isaac_sim_common.py")

# collect_demos.py's measured peak for the base camera over a whole episode,
# from the smoketest/clipfix_check run. Used to check this model against
# reality rather than to configure anything.
MEASURED_BASE_PEAK_PIXELS = 139


def _literal(node):
    """Value of a simple assignment node, including `np.array([...])`."""
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        pass
    if (isinstance(node, ast.Call) and node.args
            and isinstance(node.func, ast.Attribute) and node.func.attr == "array"):
        return ast.literal_eval(node.args[0])
    raise ValueError("not a literal")


def read_constants(path):
    """Top-level literal constants of a module, without importing it.

    pick_place_scene imports Isaac Sim at module scope, so it cannot be
    imported here; parsing gives the same numbers with no runtime.
    """
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue
        try:
            out[target.id] = _literal(node.value)
        except ValueError:
            continue
    return out


def read_default_arg(path, function_name, arg_name):
    """A function's default argument value -- the cube's size lives on
    isaac_sim_common.add_shape(size=...), not in a module constant."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            args = node.args.args
            defaults = node.args.defaults
            for name, default in zip(args[len(args) - len(defaults):], defaults):
                if name.arg == arg_name:
                    return _literal(default)
    raise KeyError(f"{function_name}({arg_name}=...) not found in {path}")


# --- the geometry ---------------------------------------------------------

def visible_width_m(distance_m, hfov_deg):
    """Width of world the image spans at `distance_m`."""
    return 2.0 * distance_m * np.tan(np.radians(hfov_deg) / 2.0)


def span_px(object_size_m, distance_m, resolution_px, hfov_deg):
    """How many pixels across the object appears."""
    return resolution_px * object_size_m / visible_width_m(distance_m, hfov_deg)


def required_hfov_deg(object_size_m, distance_m, resolution_px, target_span_px):
    """Field of view that would put the object at `target_span_px`, holding
    the camera where it is."""
    width = resolution_px * object_size_m / target_span_px
    return float(np.degrees(2.0 * np.arctan(width / (2.0 * distance_m))))


def required_distance_m(object_size_m, hfov_deg, resolution_px, target_span_px):
    """Distance that would do it, holding the lens as it is."""
    width = resolution_px * object_size_m / target_span_px
    return float(width / (2.0 * np.tan(np.radians(hfov_deg) / 2.0)))


def max_span_px(object_size_m, resolution_px, must_cover_width_m):
    """The ceiling: the largest the object can appear while a swath of
    `must_cover_width_m` still fits in frame. Independent of where the camera
    is or what lens it has -- which is exactly why it is a ceiling."""
    return resolution_px * object_size_m / must_cover_width_m


# --- the analysis, as a function both the CLI and the scene can call -------

# What fraction of the geometrically achievable peak the cube must actually
# reach during an episode before the framing counts as working. Set from the
# ceiling rather than as a bare pixel count so it means "the cube was about
# as visible as this camera can make it" instead of "the cube was visible at
# all" -- MIN_CUBE_PIXELS_IN_BASE_VIEW's old flat 30 px sat at 15% of the
# best case, low enough to pass a camera that was framing the task badly.
USABLE_FRACTION_OF_PEAK = 0.5


def workspace_span_m(y_range, place_y):
    """How wide a strip has to stay in frame: the cube's spawn spread plus
    wherever it gets placed."""
    lo, hi = float(min(y_range)), float(max(y_range))
    return max(hi, float(place_y)) - min(lo, float(place_y))


@dataclasses.dataclass
class FramingAnalysis:
    """What this camera can and cannot show, given where it is and what has
    to stay in frame."""
    best_span_px: float
    expected_peak_area_px: float
    must_cover_m: float
    ceiling_span_px: float
    target_span_px: float
    required_hfov_deg: float
    required_distance_m: float
    required_resolution_px: int
    required_workspace_m: float

    @property
    def target_reachable(self):
        """False when no camera placement or lens reaches target_span_px,
        because the swath that must stay in frame already caps it."""
        return self.ceiling_span_px >= self.target_span_px

    @property
    def usable_peak_area_px(self):
        """The episode-level floor worth enforcing (see
        USABLE_FRACTION_OF_PEAK)."""
        return USABLE_FRACTION_OF_PEAK * self.expected_peak_area_px


def framing_analysis(object_size_m, camera_position, sample_points, hfov_deg,
                     resolution_px, must_cover_m, target_span_px=30.0,
                     reference_point=None):
    """Geometry only -- no rendering, no simulator.

    sample_points: world points the object can occupy (e.g. the corners of
    its spawn range); the closest one to the camera sets the best case.
    must_cover_m: width of workspace that has to stay in frame, which is
    what turns a preference into a ceiling.
    """
    camera_position = np.asarray(camera_position, dtype=float)
    distances = [float(np.linalg.norm(np.asarray(p, dtype=float) - camera_position))
                 for p in sample_points]
    spans = [span_px(object_size_m, d, resolution_px, hfov_deg) for d in distances]
    best = max(spans)

    reference_distance = (min(distances) if reference_point is None
                          else float(np.linalg.norm(
                              np.asarray(reference_point, dtype=float) - camera_position)))

    return FramingAnalysis(
        best_span_px=best,
        expected_peak_area_px=best * best,
        must_cover_m=must_cover_m,
        ceiling_span_px=max_span_px(object_size_m, resolution_px, must_cover_m),
        target_span_px=target_span_px,
        required_hfov_deg=required_hfov_deg(
            object_size_m, reference_distance, resolution_px, target_span_px),
        required_distance_m=required_distance_m(
            object_size_m, hfov_deg, resolution_px, target_span_px),
        required_resolution_px=int(np.ceil(target_span_px * must_cover_m / object_size_m)),
        required_workspace_m=resolution_px * object_size_m / target_span_px,
    )


def describe(analysis):
    """The analysis as lines of text, for a preflight log or the CLI."""
    a = analysis
    lines = [
        f"the cube can reach at most {a.best_span_px:.1f} px across "
        f"({a.expected_peak_area_px:.0f} px2) from this camera",
        f"{a.must_cover_m:.2f} m of workspace has to stay in frame, which caps it at "
        f"{a.ceiling_span_px:.1f} px across no matter where the camera goes",
    ]
    if a.target_reachable:
        lines.append(
            f"{a.target_span_px:.0f} px is reachable: hfov {a.required_hfov_deg:.1f} deg "
            f"or distance {a.required_distance_m:.3f} m")
    else:
        lines.append(
            f"{a.target_span_px:.0f} px is NOT reachable by reframing -- it needs the "
            f"workspace tightened to ~{a.required_workspace_m:.2f} m, or the render "
            f"widened to ~{a.required_resolution_px} px, or the wrist camera to carry "
            f"the approach")
    return lines


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--target-span-px", type=float, default=30.0,
                        help="how many pixels across the cube should be (default 30)")
    parser.add_argument("--wrist-link-radius-m", type=float, default=0.045,
                        help="UR5e wrist_3_link radius the wrist camera must clear -- "
                             "UNVERIFIED default, measure it on the asset")
    args = parser.parse_args()

    scene = read_constants(SCENE_PATH)
    cube_size = scene.get("CUBE_SIZE_M")
    if cube_size is None:  # older scenes left the size on add_shape's default
        cube_size = read_default_arg(COMMON_PATH, "add_shape", "size")
    res_w, res_h = scene["CAMERA_RESOLUTION"]

    cam = np.asarray(scene["BASE_CAMERA_POSITION"], dtype=float)
    aim = np.asarray(scene["BASE_CAMERA_AIM_POINT"], dtype=float)
    hfov = scene["BASE_CAMERA_HORIZONTAL_FOV_DEG"]
    x_lo, x_hi = scene["CUBE_X_RANGE"]
    y_lo, y_hi = scene["CUBE_Y_RANGE"]
    cube_z = scene["CUBE_Z"]
    place = np.asarray(scene["PLACE_TARGET_POSITION"], dtype=float)

    print("=" * 70)
    print("BASE CAMERA")
    print("=" * 70)
    print(f"  position {np.round(cam, 3).tolist()}  aim {np.round(aim, 3).tolist()}  "
          f"hfov {hfov} deg  resolution {res_w}x{res_h}")
    print(f"  cube {cube_size * 100:.0f} cm (isaac_sim_common.add_shape size default)")
    print()
    print(f"  {'spawn corner':<22} {'distance':>9} {'swath':>8} {'span':>8} {'area':>9}")
    print("  " + "-" * 60)

    corners = {
        "nearest to camera": np.array([x_hi, y_hi, cube_z]),
        "aim point": aim,
        "farthest from camera": np.array([x_lo, y_lo, cube_z]),
    }
    spans = {}
    for label, point in corners.items():
        distance = float(np.linalg.norm(point - cam))
        width = visible_width_m(distance, hfov)
        s = span_px(cube_size, distance, res_w, hfov)
        spans[label] = s
        print(f"  {label:<22} {distance:>8.3f}m {width:>7.3f}m {s:>7.1f}px "
              f"{s * s:>8.0f}px2")

    best = max(spans.values())
    print()
    print(f"  model says the cube peaks near {best * best:.0f} px2 of area "
          f"({best:.0f} px across)")
    print(f"  collect_demos measured a peak of {MEASURED_BASE_PEAK_PIXELS} px "
          f"over a whole episode")
    if 0.4 * best * best <= MEASURED_BASE_PEAK_PIXELS <= 1.6 * best * best:
        print("  -> the measurement IS the geometry. The cube was never being")
        print("     occluded down to 139 px; that is simply its apparent size.")
    else:
        print("  -> model and measurement disagree; something else is going on")

    print()
    print("  What would make it bigger, holding everything else:")
    d_aim = float(np.linalg.norm(aim - cam))
    for target in (20.0, args.target_span_px, 40.0):
        print(f"    {target:>4.0f} px across  ->  hfov "
              f"{required_hfov_deg(cube_size, d_aim, res_w, target):>5.1f} deg  "
              f"or distance {required_distance_m(cube_size, hfov, res_w, target):>5.3f} m")

    # --- the ceiling nobody can beat --------------------------------------
    must_cover = workspace_span_m((y_lo, y_hi), place[1])
    analysis = framing_analysis(
        cube_size, cam, list(corners.values()), hfov, res_w, must_cover,
        target_span_px=args.target_span_px, reference_point=aim)
    ceiling = analysis.ceiling_span_px
    print()
    print("  THE CONSTRAINT:")
    print(f"    the cube spawns over y in [{y_lo}, {y_hi}] and is placed at "
          f"y={place[1]}, so {must_cover:.2f} m has to stay in frame.")
    print(f"    at {res_w} px wide that caps the cube at {ceiling:.1f} px across "
          f"({ceiling * ceiling:.0f} px2) --")
    print(f"    no camera position or lens beats it, because both only move")
    print(f"    the swath width, and the swath cannot go below {must_cover:.2f} m.")
    print()
    if ceiling < args.target_span_px:
        print(f"    {args.target_span_px:.0f} px is therefore NOT reachable by reframing. "
              f"To get there:")
        needed_width = res_w * cube_size / args.target_span_px
        needed_res = int(np.ceil(args.target_span_px * must_cover / cube_size))
        print(f"      - shrink the workspace to about {needed_width:.2f} m of spread "
              f"(tighten CUBE_Y_RANGE / move PLACE_TARGET_POSITION closer), or")
        print(f"      - raise CAMERA_RESOLUTION to about {needed_res} px wide, or")
        print(f"      - lean on the wrist camera for the approach and accept the")
        print(f"        base view as context only.")
    else:
        print(f"    {args.target_span_px:.0f} px is reachable within that constraint.")

    floor = scene.get("MIN_CUBE_PIXELS_FLOOR", scene.get("MIN_CUBE_PIXELS_IN_BASE_VIEW"))
    if floor is not None:
        derived = max(floor, int(round(analysis.usable_peak_area_px)))
        print()
        print(f"  collect_demos aborts when an episode never peaks above "
              f"{derived} px2,")
        print(f"  which is {100.0 * derived / analysis.expected_peak_area_px:.0f}% of the "
              f"{analysis.expected_peak_area_px:.0f} px2 this geometry can achieve "
              f"(floor {floor} px2).")
        print(f"  Deriving it from the ceiling is the point: a flat {floor} px2 sits at "
              f"{100.0 * floor / analysis.expected_peak_area_px:.0f}% of the best case,")
        print(f"  which catches a camera pointed at nothing but not one framing the "
              f"task badly.")

    # --- wrist camera ------------------------------------------------------
    lateral = scene["WRIST_CAMERA_LATERAL_M"]
    back = scene["WRIST_CAMERA_BACK_M"]
    wrist_hfov = scene["WRIST_CAMERA_HORIZONTAL_FOV_DEG"]
    print()
    print("=" * 70)
    print("WRIST CAMERA")
    print("=" * 70)
    print(f"  mounted {lateral * 100:.0f} cm lateral, {back * 100:.0f} cm back, "
          f"hfov {wrist_hfov} deg")

    clears = lateral > args.wrist_link_radius_m
    print(f"  lateral offset vs wrist_3_link radius ({args.wrist_link_radius_m * 100:.1f} cm, "
          f"UNVERIFIED): {'clears' if clears else 'STILL INSIDE THE LINK'}")

    grasp_distance = float(np.hypot(lateral, back))
    wrist_span = span_px(cube_size, grasp_distance, res_w, wrist_hfov)
    print(f"  at the grasp the cube sits about {grasp_distance:.3f} m away "
          f"-> {wrist_span:.0f} px across ({wrist_span * wrist_span:.0f} px2)")
    print(f"  that is {wrist_span / best:.1f}x the base camera's best case, which is")
    print(f"  why the wrist view is the one that carries the fine detail.")

    near_clip = scene.get("CAMERA_NEAR_CLIP_M")
    if near_clip is not None:
        print(f"  near clip {near_clip} m vs {grasp_distance:.3f} m standoff: "
              f"{'ok' if grasp_distance > near_clip else 'CLIPPED'}")


if __name__ == "__main__":
    main()
