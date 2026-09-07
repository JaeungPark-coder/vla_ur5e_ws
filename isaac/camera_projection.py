"""Pure pinhole-camera geometry: pixel <-> 3D world position, used by the
LLM + open-vocabulary hybrid pipeline (hybrid_pick_place_demo.py) to turn a
detected 2D bounding-box center into a 3D grasp target.

Deliberately has NO Isaac Sim import -- unlike almost everything else under
isaac/, this is plain numpy, so (a) it's usable unchanged once real
hand-eye/camera calibration (README Phase 4) supplies real intrinsics/
extrinsics instead of the sim's known ones, and (b) it's the one file in
this addition that can actually be smoke-tested in an environment without
Isaac Sim -- see the bottom of this file.

Camera convention used throughout (our own choice, consistent between
projection and its inverse -- not tied to any particular USD/Isaac-Sim
internal camera convention): in camera-local space, +Z is forward (out of
the lens), +X is right, +Y is up. `camera_rotation_matrix` maps a
camera-local direction to world-frame; pass whatever rotation matches how
you actually orient the camera prim (pick_place_scene.py's base camera is
set up with this convention in mind).

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
    camera-local space (+Z forward, +Y up -- see module docstring)."""
    fx, fy, cx, cy = camera_intrinsics(resolution, horizontal_fov_deg)
    u, v = pixel_xy
    direction = np.array([(u - cx) / fx, (cy - v) / fy, 1.0])
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
    if direction_cam[2] <= 0:
        return None  # point is behind the camera

    u = fx * (direction_cam[0] / direction_cam[2]) + cx
    v = cy - fy * (direction_cam[1] / direction_cam[2])
    return np.array([u, v])


if __name__ == "__main__":
    # Smoke test: project a known table point to a pixel, then recover it
    # via pixel_to_table_position, and confirm we get the same point back.
    # Run directly with plain `python camera_projection.py` -- no Isaac Sim needed.
    from scipy.spatial.transform import Rotation as Rot

    resolution = (256, 256)
    hfov_deg = 60.0
    table_z = 0.0
    camera_position = np.array([0.9, 0.0, 0.5])
    # Looking down and back toward the origin-ish workspace, matching the
    # spirit of pick_place_scene.py's base camera placement.
    camera_rotation_matrix = Rot.from_euler("xyz", [0, 55, 180], degrees=True).as_matrix()

    true_point = np.array([0.45, 0.10, table_z])
    pixel = project_point_to_pixel(true_point, camera_position, camera_rotation_matrix,
                                    resolution, hfov_deg)
    assert pixel is not None, "test point projected behind the camera -- adjust the test setup"

    recovered = pixel_to_table_position(pixel, camera_position, camera_rotation_matrix,
                                         resolution, hfov_deg, table_z)
    assert recovered is not None
    error = np.linalg.norm(recovered - true_point)
    assert error < 1e-6, f"round-trip error too large: {error}"
    print(f"camera_projection.py smoke test OK (round-trip error={error:.2e})")
