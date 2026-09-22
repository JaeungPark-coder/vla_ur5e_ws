"""Pure pinhole-camera geometry: pixel <-> 3D world position, used by the
LLM + open-vocabulary hybrid pipeline (hybrid_pick_place_demo.py) to turn a
detected 2D bounding-box center into a 3D grasp target.

Deliberately has NO Isaac Sim import -- unlike almost everything else under
isaac/, this is plain numpy, so (a) it's usable unchanged once real
hand-eye/camera calibration (README Phase 4) supplies real intrinsics/
extrinsics instead of the sim's known ones, and (b) it's the one file in
this addition that can actually be smoke-tested in an environment without
Isaac Sim -- see the bottom of this file.

Camera convention used throughout, in camera-local space: -Z is forward
(out of the lens), +X is right, +Y is up. `camera_rotation_matrix` maps a
camera-local direction to world-frame; pass whatever rotation matches how
you actually orient the camera prim.

2026-09-23 CONFIRMED (independently reproduced, not just reasoned about):
this MUST be -Z forward, not +Z -- it was +Z until this fix, and that is
NOT an arbitrary choice this module was free to make independently of USD,
despite what an earlier version of this docstring claimed ("our own
choice... not tied to any particular USD/Isaac-Sim internal camera
convention"). get_base_camera_pose() (pick_place_scene.py) feeds this
module the camera prim's REAL rotation matrix, read straight off USD --
and a USD camera images along its own local -Z (pick_place_scene.py's
_look_at_quat has the same fact in its own docstring: "USD cameras look
along their own local -Z, so that axis -- not +Z -- is what gets aimed").
A +Z-forward ray direction fed through that real rotation matrix points
the wrong way entirely: computed against this project's actual
BASE_CAMERA_POSITION/BASE_CAMERA_AIM_POINT, ray_world had a POSITIVE z
component pointing up and away from the table, so the ray-vs-table-plane
intersection returned None for every pixel (t < 0, "table plane is behind
the camera" -- which was true only because the ray was backwards) -- the
hybrid pipeline's every single trial, not a rare edge case. The two-
function round trip below could not have caught this: projection and its
inverse always agree with EACH OTHER regardless of which axis either one
calls "forward", since they share the same (right or wrong) convention by
construction -- only checking against an independently-built, USD-style
rotation matrix (added below) can.

Assumes square pixels (fx == fy, derived from horizontal FOV) -- fine for
the synthetic Isaac Sim camera and close enough for most real depth/RGB
cameras; revisit if your real camera's intrinsics (from its own calibration)
say otherwise.
"""
import numpy as np


def camera_intrinsics(resolution, horizontal_fov_deg):
    """resolution: (width, height) in pixels. Returns (fx, fy, cx, cy)."""
    width, height = resolution
    fx = (width / 2.0) / np.tan(np.radians(horizontal_fov_deg) / 2.0)
    fy = fx  # square-pixel assumption -- see module docstring
    cx = width / 2.0
    cy = height / 2.0
    return fx, fy, cx, cy


def pixel_to_camera_ray(pixel_xy, resolution, horizontal_fov_deg):
    """Unit direction of the ray through `pixel_xy` = (u, v), in
    camera-local space (-Z forward, +Y up -- see module docstring)."""
    fx, fy, cx, cy = camera_intrinsics(resolution, horizontal_fov_deg)
    u, v = pixel_xy
    direction = np.array([(u - cx) / fx, (cy - v) / fy, -1.0])
    return direction / np.linalg.norm(direction)


def pixel_to_table_position(pixel_xy, camera_position, camera_rotation_matrix,
                             resolution, horizontal_fov_deg, table_z):
    """Casts a ray from the camera through `pixel_xy` and intersects it
    with the horizontal plane z = table_z -- the actual function
    hybrid_pick_place_demo.py calls to turn a detected object's pixel
    center into a 3D grasp target, assuming (true for a tabletop
    pick-and-place setup) the object rests at a known height.

    Returns None if the ray is parallel to the table plane (shouldn't
    happen for a camera looking down at a table, but guarded rather than
    dividing by ~0) or points away from it."""
    camera_position = np.asarray(camera_position, dtype=float)
    ray_cam = pixel_to_camera_ray(pixel_xy, resolution, horizontal_fov_deg)
    ray_world = camera_rotation_matrix @ ray_cam

    if abs(ray_world[2]) < 1e-8:
        return None
    t = (table_z - camera_position[2]) / ray_world[2]
    if t < 0:
        return None  # table plane is behind the camera along this ray
    return camera_position + t * ray_world


def project_point_to_pixel(world_point, camera_position, camera_rotation_matrix,
                            resolution, horizontal_fov_deg):
    """Inverse of pixel_to_table_position's ray step -- projects a known 3D
    world point to its pixel coordinates. Only used for testing
    pixel_to_table_position (a real detector never needs this direction),
    but kept here since it shares the same camera convention and intrinsics."""
    fx, fy, cx, cy = camera_intrinsics(resolution, horizontal_fov_deg)
    world_point = np.asarray(world_point, dtype=float)
    camera_position = np.asarray(camera_position, dtype=float)

    relative = world_point - camera_position
    direction_cam = camera_rotation_matrix.T @ relative
    # -Z forward (see module docstring): a point in front of the camera has
    # NEGATIVE local z, so "depth along the viewing direction" is -z, not z.
    depth = -direction_cam[2]
    if depth <= 0:
        return None  # point is behind the camera

    u = fx * (direction_cam[0] / depth) + cx
    v = cy - fy * (direction_cam[1] / depth)
    return np.array([u, v])


def _usd_style_look_at_matrix(eye, target, up=(0.0, 0.0, 1.0)):
    """Duplicates pick_place_scene.py's _look_at_quat, as a rotation matrix
    instead of a Gf.Quatf, so the smoke test below can build a REAL
    USD-convention camera matrix without importing pick_place_scene.py --
    which needs Isaac Sim's own modules at import time and would defeat
    this module's whole reason for staying plain numpy. Deliberately
    duplicated rather than shared; if you change the real one, change this
    one too, or this test stops meaning what it claims to."""
    eye = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye
    forward /= np.linalg.norm(forward)
    z_axis = -forward  # camera local +Z points away from what it looks at
    up = np.asarray(up, dtype=float)
    if abs(float(np.dot(up, z_axis))) > 0.999:
        up = np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(up, z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return np.column_stack((x_axis, y_axis, z_axis))


if __name__ == "__main__":
    # Smoke test 1: project a known table point to a pixel, then recover it
    # via pixel_to_table_position, and confirm we get the same point back.
    # This only proves internal self-consistency (see the module docstring
    # for why a -Z-vs-+Z convention bug survived this test for a full
    # session) -- test 2 below is what actually catches a convention
    # mismatch against USD.
    resolution = (256, 256)
    hfov_deg = 60.0
    table_z = 0.0
    camera_position = np.array([0.9, 0.0, 0.5])
    aim_point = np.array([0.45, 0.10, table_z])
    camera_rotation_matrix = _usd_style_look_at_matrix(camera_position, aim_point)

    true_point = np.array([0.45, 0.10, table_z])
    pixel = project_point_to_pixel(true_point, camera_position, camera_rotation_matrix,
                                    resolution, hfov_deg)
    assert pixel is not None, "test point projected behind the camera -- adjust the test setup"

    recovered = pixel_to_table_position(pixel, camera_position, camera_rotation_matrix,
                                         resolution, hfov_deg, table_z)
    assert recovered is not None
    error = np.linalg.norm(recovered - true_point)
    assert error < 1e-6, f"round-trip error too large: {error}"
    print(f"smoke test 1 (round trip, self-consistency only) OK (error={error:.2e})")

    # Smoke test 2: the one the round trip above cannot do. Build a REAL
    # USD-style look-at matrix (same helper _look_at_quat uses) aimed at a
    # known point, and check that pixel_to_table_position, given the pixel
    # this camera is DEFINED to be centred on, actually recovers a point
    # near that aim point -- not None, and not somewhere the geometry says
    # it shouldn't be. This is the test that would have caught the -Z/+Z
    # bug: it fails immediately (returns None for the centre pixel) against
    # a +Z-forward convention, no matter how internally self-consistent
    # that convention's own round trip is.
    centre_pixel = (resolution[0] / 2.0, resolution[1] / 2.0)
    recovered_centre = pixel_to_table_position(
        centre_pixel, camera_position, camera_rotation_matrix, resolution, hfov_deg, table_z)
    assert recovered_centre is not None, (
        "the camera's own aim point, through its own centre pixel, returned None -- "
        "this is the -Z/+Z forward-axis bug this test exists to catch")
    centre_error = np.linalg.norm(recovered_centre - aim_point)
    assert centre_error < 1e-6, (
        f"camera aimed at {aim_point} but its centre pixel recovers "
        f"{recovered_centre} instead (error {centre_error:.4f}m)")
    print(f"smoke test 2 (USD-convention look-at matrix, matches _look_at_quat) "
          f"OK (centre-pixel error={centre_error:.2e})")

    # Also exercise this project's actual base-camera constants specifically
    # (not just an arbitrary aim point), so a future change to those numbers
    # re-runs the same check against the real geometry, not a stand-in.
    real_camera_position = np.array([0.9, 0.0, 0.5])         # BASE_CAMERA_POSITION
    real_aim_point = np.array([0.45, 0.10, 0.02])             # BASE_CAMERA_AIM_POINT
    real_rotation = _usd_style_look_at_matrix(real_camera_position, real_aim_point)
    real_centre = pixel_to_table_position(
        centre_pixel, real_camera_position, real_rotation, resolution, hfov_deg,
        table_z=real_aim_point[2])
    assert real_centre is not None
    real_error = np.linalg.norm(real_centre - real_aim_point)
    assert real_error < 1e-6, f"real base-camera geometry check failed: error {real_error:.4f}m"
    print(f"smoke test 3 (this project's actual BASE_CAMERA_POSITION/AIM_POINT) "
          f"OK (centre-pixel error={real_error:.2e})")
