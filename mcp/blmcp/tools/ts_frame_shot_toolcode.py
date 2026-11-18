"""Tool-code: create/update TS_CAM and frame the car (or a detail of it)."""

__all__ = ("Params", "main")

from typing import NamedTuple

_SENSOR_MM = 36.0
_ITERATIONS = 10
_MAX_ELEVATION_DEG = 89.5  # straight down (90) degenerates the camera's up axis
_DETAIL_MIN_DISTANCE_M = 0.3
# Stand-ins inside the frame and nearer than this fraction of the camera-car distance are hidden:
# they would stand between the camera and the product, and a blurred near object whose painted
# edge does not match its depth edge leaves a hard seam in the depth of field.
_NEAR_CLEAR_FRAC = 0.6


class Params(NamedTuple):
    azimuth_deg: float
    elevation_deg: float
    focal_mm: float
    distance_m: float
    car_width_frac: float
    target_height_m: float
    frame_x: float
    frame_y: float
    roll_deg: float
    resolution_x: int
    resolution_y: int
    aim_x: float
    aim_y: float
    aim_z: float
    subject_width_m: float


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end

# @include_begin: _template_ts_layout.py
# @include_end


def _camera():
    import bpy  # pylint: disable=import-error
    cam = bpy.data.objects.get("TS_CAM")
    if cam is None or cam.type != "CAMERA":
        data = bpy.data.cameras.get("TS_CAM") or bpy.data.cameras.new("TS_CAM")
        cam = bpy.data.objects.new("TS_CAM", data)
        cam["ts_role"] = "camera"
    rig = _ts_collection(TS_RIG)
    if rig not in cam.users_collection:
        rig.objects.link(cam)
    return cam


def _car_corners(root):
    from mathutils import Vector  # pylint: disable=import-error
    b = [float(v) for v in root["ts_bbox_local"]]
    M = root.matrix_world
    return [tuple(M @ Vector((x, y, z))) for x in (b[0], b[3]) for y in (b[1], b[4]) for z in (b[2], b[5])]


def _aim_local(root, ax: float, ay: float, az: float):
    """Car-normalised aim -> point in the car root's frame: x -1 = car's left side .. +1 right
    (local -X / +X), y -1 = tail .. +1 = nose (local -Y / +Y), z 0 = bottom .. 1 = roof."""
    from mathutils import Vector  # pylint: disable=import-error
    b = [float(v) for v in root["ts_bbox_local"]]
    return Vector(((b[0] + b[3]) / 2.0 + ax * (b[3] - b[0]) / 2.0,
                   (b[1] + b[4]) / 2.0 + ay * (b[4] - b[1]) / 2.0,
                   b[2] + az * (b[5] - b[2])))


def _cube_corners(root, centre_local, side: float):
    """World corners of a car-aligned cube of edge `side` centred on `centre_local`."""
    from mathutils import Vector  # pylint: disable=import-error
    h = side / 2.0
    M = root.matrix_world
    return [tuple(M @ (centre_local + Vector((x, y, z)))) for x in (-h, h) for y in (-h, h) for z in (-h, h)]


def main(params: Params):
    import math
    import bpy  # pylint: disable=import-error
    from mathutils import Matrix, Quaternion, Vector  # pylint: disable=import-error

    if _ts_busy():
        return _ts_busy_result()
    root = _ts_car_root()
    if root is None:
        return _ts_result("error", message="No car loaded (call load_car first)")
    scene = bpy.context.scene
    W, H = int(params.resolution_x), int(params.resolution_y)
    rd = scene.render
    rd.resolution_x, rd.resolution_y, rd.resolution_percentage = W, H, 100

    cam = _camera()
    cd = cam.data
    cd.type = "PERSP"
    cd.lens = float(params.focal_mm)
    cd.sensor_fit = "HORIZONTAL"
    cd.sensor_width = _SENSOR_MM
    cd.clip_start = 0.1
    cd.clip_end = 30000.0
    scene.camera = cam

    detail = float(params.subject_width_m) > 0.0
    dims = [float(v) for v in root["ts_dims_m"]]
    car_corners = _car_corners(root)
    if detail:
        side = float(params.subject_width_m)
        aim_local = _aim_local(root, float(params.aim_x), float(params.aim_y), float(params.aim_z))
        target = root.matrix_world @ aim_local
        corners = _cube_corners(root, aim_local, side)
        size, min_d = side, _DETAIL_MIN_DISTANCE_M
    else:
        target = root.matrix_world.translation + Vector((0.0, 0.0, float(params.target_height_m)))
        corners = car_corners
        size, min_d = max(dims[0], dims[1]) * 0.9, 0.6 * max(dims)
    elevation = max(-_MAX_ELEVATION_DEG, min(_MAX_ELEVATION_DEG, float(params.elevation_deg)))
    az, el = math.radians(params.azimuth_deg), math.radians(elevation)
    local_dir = Vector((-math.sin(az), math.cos(az), 0.0))  # 0 = front, 90 = car's left
    dir_h = (root.matrix_world.to_quaternion() @ local_dir)
    dir_h.z = 0.0
    dir_h.normalize()
    view_dir = dir_h * math.cos(el) + Vector((0.0, 0.0, math.sin(el)))
    roll = Quaternion((0.0, 0.0, 1.0), math.radians(params.roll_deg))
    rot = view_dir.to_track_quat("Z", "Y") @ roll  # the camera looks along its -Z

    want = max(0.05, min(0.98, float(params.car_width_frac)))
    solve = params.distance_m <= 0.0
    if solve:
        d = size / (want * _SENSOR_MM / cd.lens)
    else:
        d = float(params.distance_m)
    cd.shift_x = 0.5 - params.frame_x
    cd.shift_y = (params.frame_y - 0.5) * H / W

    def place(dist):
        cam.matrix_world = Matrix.LocRotScale(target + view_dir * dist, rot, None)
        bpy.context.view_layer.update()
        return _box_of(corners)

    def _box_of(points):
        pts = _ts_project(points)
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return [min(xs), min(ys), max(xs), max(ys)], all(p[2] for p in pts)

    for _ in range(_ITERATIONS):
        box, front = place(d)
        wf = (box[2] - box[0]) / W
        cx, cy = (box[0] + box[2]) / 2.0 / W, (box[1] + box[3]) / 2.0 / H
        cd.shift_x += cx - params.frame_x
        cd.shift_y += (params.frame_y - cy) * H / W
        if solve and front and wf > 0:
            d = max(d * wf / want, min_d)
    sbox, sfront = place(d)
    box, front = _box_of(car_corners) if detail else (sbox, sfront)
    cleared = _clear_near_standins(cam, W, H, d)
    cleared += _clear_occluders(cam, [target] + [target.lerp(Vector(c), 0.8) for c in corners])

    # Horizon: a far point straight ahead, at camera height.
    fwd = cam.matrix_world.to_quaternion() @ Vector((0.0, 0.0, -1.0))
    fwd.z = 0.0
    horizon_y = None
    if fwd.length > 1e-6:
        hp = _ts_project([tuple(cam.matrix_world.translation + fwd.normalized() * 1.0e5)])[0]
        horizon_y = round(hp[1] / H, 4) if hp[2] else None

    all_in = front and box[0] >= 0 and box[1] >= 0 and box[2] <= W and box[3] <= H
    cam["ts_azimuth_deg"] = float(params.azimuth_deg)
    cam["ts_elevation_deg"] = elevation
    cam["ts_distance_m"] = float(d)
    _ts_redraw()
    loc = cam.matrix_world.translation
    return _ts_result(
        "ok",
        camera={
            "location": [round(v, 3) for v in loc],
            "rotation_deg": [round(math.degrees(a), 3) for a in cam.matrix_world.to_euler("XYZ")],
            "lens_mm": round(cd.lens, 3),
            "sensor_width_mm": cd.sensor_width,
            "shift_x": round(cd.shift_x, 5),
            "shift_y": round(cd.shift_y, 5),
            "distance_m": round(d, 3),
        },
        car_bbox_px=[round(v, 1) for v in box],
        car_bbox_frac=[round(box[0] / W, 4), round(box[1] / H, 4), round(box[2] / W, 4), round(box[3] / H, 4)],
        all_in_frame=bool(all_in),
        horizon_y_frac=horizon_y,
        resolution=[W, H],
        cleared_standins=cleared,
        subject="car" if not detail else {
            "aim": [float(params.aim_x), float(params.aim_y), float(params.aim_z)],
            "point": [round(v, 3) for v in target],
            "width_m": float(params.subject_width_m),
            "bbox_px": [round(v, 1) for v in sbox],
            "in_frame": bool(sfront and sbox[0] >= 0 and sbox[1] >= 0 and sbox[2] <= W and sbox[3] <= H),
        },
    )


def _clear_near_standins(cam, width: int, height: int, car_distance: float) -> list:
    """Hide stand-ins with a vertex in the frame closer than _NEAR_CLEAR_FRAC of the subject's
    distance. Every framing first shows all stand-ins again, so re-framing never leaves stale
    holes. Stand-in meshes merge many pieces (all buildings, a whole hillside), so the test uses
    their vertices: a bounding box would hide them whenever the camera stands inside it."""
    import bpy  # pylint: disable=import-error
    import numpy as np
    coll = bpy.data.collections.get(TS_SET)
    if coll is None:
        return []
    scene = bpy.context.scene
    # A list: changing visibility while iterating `all_objects` invalidates the iterator.
    obs = [ob for ob in list(coll.all_objects)
           if ob.type == "MESH" and str(ob.get("ts_role", "")).startswith("standin")]
    to_cam = np.array(cam.matrix_world.inverted(), dtype=np.float64)
    limit = _NEAR_CLEAR_FRAC * car_distance
    hidden = []
    for ob in obs:
        ob.hide_render = ob.hide_viewport = False
        n = len(ob.data.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float32)
        ob.data.vertices.foreach_get("co", co)
        world = np.c_[co.reshape(-1, 3), np.ones(n)] @ np.array(ob.matrix_world, dtype=np.float64).T
        depth = -(world @ to_cam.T)[:, 2]
        near = (depth > 0.0) & (depth < limit)
        if not near.any():
            continue
        px, front = _ts_project_np(scene, world[near, :3])
        inside = front & (px[:, 0] >= 0) & (px[:, 0] <= width) & (px[:, 1] >= 0) & (px[:, 1] <= height)
        if inside.any():
            ob.hide_render = ob.hide_viewport = True
            hidden.append(ob.name)
    return hidden


def _clear_occluders(cam, points) -> list:
    """Hide stand-ins hit by a ray from the camera to any of `points` (the subject) before it
    arrives: the vertex test misses large faces whose vertices all lie outside the frame (a wall
    beside a long-lens camera)."""
    import bpy  # pylint: disable=import-error
    scene = bpy.context.scene
    origin = cam.matrix_world.translation.copy()
    hidden = []
    for p in points:
        ray = p - origin
        dist = ray.length
        if dist < 1e-6:
            continue
        for _ in range(6):
            hit, _loc, _n, _i, ob, _m = scene.ray_cast(bpy.context.evaluated_depsgraph_get(), origin,
                                                        ray.normalized(), distance=dist * 0.98)
            if not hit or ob is None or not str(ob.get("ts_role", "")).startswith("standin"):
                break
            src = bpy.data.objects.get(ob.name) or ob
            src.hide_render = src.hide_viewport = True
            hidden.append(src.name)
            bpy.context.view_layer.update()
    return hidden
