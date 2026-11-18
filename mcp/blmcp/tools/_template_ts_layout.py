# Analytic image layout of the Trueshot shot (runs inside Blender).
#
# Include after `_template_ts_common.py` and `_template_ts_scene.py`:
#   # @include_begin: _template_ts_layout.py
#   # @include_end
#
# Everything is computed from the scene (camera, road props, car root, wheel pivots) at the
# scene's render resolution; pixel y is measured from the TOP of the image.

__all__ = ()


def _ts_render_size(scene) -> tuple:
    rd = scene.render
    pct = rd.resolution_percentage / 100.0
    return int(rd.resolution_x * pct), int(rd.resolution_y * pct)


def _ts_project_np(scene, points):
    """Project world points (N, 3) through the scene camera.

    Returns (px (N, 2) float64, in_front (N,) bool) with y from the top of the image.
    """
    import bpy  # pylint: disable=import-error
    import numpy as np
    cam = scene.camera
    w, h = _ts_render_size(scene)
    rd = scene.render
    proj = cam.calc_matrix_camera(bpy.context.evaluated_depsgraph_get(), x=w, y=h,
                                  scale_x=rd.pixel_aspect_x, scale_y=rd.pixel_aspect_y)
    m = np.array(proj @ cam.matrix_world.inverted(), dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    hom = np.c_[pts, np.ones(len(pts))] @ m.T
    wcl = hom[:, 3]
    front = wcl > 1e-6
    safe = np.where(front, wcl, 1.0)
    x = (hom[:, 0] / safe + 1.0) * 0.5 * w
    y = (1.0 - hom[:, 1] / safe) * 0.5 * h
    return np.c_[x, y], front


def _ts_wheel_pivots(root) -> list:
    """Wheels of the car: [{name, center (world xyz), radius_m, object}].

    `ts_wheels` lists wheel pivot empties (`ts_wheel_pivot`, centred on the axle, radius in
    `ts_wheel_radius_m`); for other entries the centre/radius come from the mesh bounds of
    the object and its children in the root's frame.
    """
    from mathutils import Vector  # pylint: disable=import-error
    import bpy  # pylint: disable=import-error
    out = []
    if root is None:
        return out
    inv = root.matrix_world.inverted()
    default_r = float(root.get("ts_wheel_radius_m", 0.0))
    for name in root.get("ts_wheels", []):
        ob = bpy.data.objects.get(str(name))
        if ob is None:
            continue
        if ob.get("ts_wheel_pivot"):
            centre = ob.matrix_world.translation.copy()
            radius = float(ob.get("ts_wheel_radius_m", default_r))
        else:
            pts = [inv @ (d.matrix_world @ Vector(c)) for d in [ob, *ob.children_recursive]
                   if d.type == "MESH" for c in d.bound_box]
            if not pts:
                continue
            lo = Vector([min(p[i] for p in pts) for i in range(3)])
            hi = Vector([max(p[i] for p in pts) for i in range(3)])
            centre = root.matrix_world @ ((lo + hi) / 2)
            radius = max(hi.y - lo.y, hi.z - lo.z) / 2 or default_r
        out.append({"name": ob.name, "center": tuple(centre), "radius_m": radius, "object": ob})
    return out


def _ts_mesh_points(objs, depsgraph):
    """World-space vertices of the evaluated meshes of `objs` as an (N, 3) array."""
    import numpy as np
    chunks = []
    for ob in objs:
        if ob.type != "MESH":
            continue
        me = ob.evaluated_get(depsgraph).data
        n = len(me.vertices)
        if n == 0:
            continue
        co = np.empty(n * 3, dtype=np.float32)
        me.vertices.foreach_get("co", co)
        m = np.array(ob.matrix_world, dtype=np.float64)
        chunks.append(np.c_[co.reshape(-1, 3), np.ones(n)] @ m.T[:, :3])
    return np.concatenate(chunks) if chunks else np.zeros((0, 3))


def _ts_car_bbox_px(scene, objs=None):
    """[x0, y0, x1, y1] of the projected car vertices (visible TS_CAR meshes), or None."""
    import bpy  # pylint: disable=import-error
    import numpy as np
    if objs is None:
        coll = bpy.data.collections.get(TS_CAR)
        objs = [o for o in (coll.all_objects if coll else ()) if not o.hide_render]
    pts = _ts_mesh_points(objs, bpy.context.evaluated_depsgraph_get())
    if len(pts) == 0:
        return None
    px, front = _ts_project_np(scene, pts)
    px = px[front]
    if len(px) == 0:
        return None
    return [round(float(v), 1) for v in (*px.min(axis=0), *px.max(axis=0))]


def _ts_polyline_px(scene, pts3, margin: float = 0.1, min_step_px: float = 3.0) -> list:
    """Project a world polyline; keep points in front of the camera and inside the frame
    +/- `margin`; drop points closer than `min_step_px` to the previous kept one."""
    import numpy as np
    w, h = _ts_render_size(scene)
    px, front = _ts_project_np(scene, pts3)
    inside = front & (px[:, 0] >= -margin * w) & (px[:, 0] <= (1 + margin) * w) \
        & (px[:, 1] >= -margin * h) & (px[:, 1] <= (1 + margin) * h)
    out, last = [], None
    for p in px[inside]:
        if last is None or np.hypot(*(p - last)) >= min_step_px:
            out.append([round(float(p[0]), 1), round(float(p[1]), 1)])
            last = p
    return out


def _ts_layout(scene) -> dict:
    """Analytic layout of the current shot (contract `render_guides` layout)."""
    import math
    import numpy as np
    from mathutils import Vector  # pylint: disable=import-error
    import bpy  # pylint: disable=import-error

    cam = scene.camera
    w, h = _ts_render_size(scene)
    bpy.context.view_layer.update()
    out = {"width": w, "height": h}

    # Horizon: a far point straight ahead of the camera, at the camera's height.
    fwd = cam.matrix_world.to_3x3() @ Vector((0.0, 0.0, -1.0))
    horiz = Vector((fwd.x, fwd.y, 0.0))
    out["horizon_y_px"] = None
    if horiz.length > 1e-6:
        far = cam.matrix_world.translation + horiz.normalized() * 1.0e5
        px, front = _ts_project_np(scene, [tuple(far)])
        if front[0]:
            out["horizon_y_px"] = round(float(px[0, 1]), 1)

    # Road edges: centreline +/- width/2 (left = left of the direction of travel).
    out["road_left_px"], out["road_right_px"] = [], []
    flat = scene.get("ts_road_centerline")
    if flat is not None and len(flat) >= 4:
        cl = np.asarray(list(flat), dtype=np.float64).reshape(-1, 2)
        tan = np.gradient(cl, axis=0)
        tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
        left = np.c_[-tan[:, 1], tan[:, 0]]
        half = float(scene.get("ts_road_width_m", 7.0)) / 2.0
        z = np.full((len(cl), 1), float(scene.get("ts_road_z", 0.0)))
        out["road_left_px"] = _ts_polyline_px(scene, np.c_[cl + left * half, z])
        out["road_right_px"] = _ts_polyline_px(scene, np.c_[cl - left * half, z])

    # Car: wheels, tyre contacts, bbox, direction of travel.
    root = _ts_car_root()
    wheels = _ts_wheel_pivots(root)
    out["wheels"], out["tyre_contacts_px"] = [], []
    out["car_bbox_px"] = _ts_car_bbox_px(scene)
    out["car_forward_dir_px"] = None
    if root is not None:
        rot = root.matrix_world.to_3x3().normalized()
        up, ahead, axle = rot @ Vector((0, 0, 1)), rot @ Vector((0, 1, 0)), rot @ Vector((1, 0, 0))
        for wh in wheels:
            c, r = Vector(wh["center"]), wh["radius_m"]
            ring = [c + (up * math.cos(a) + ahead * math.sin(a)) * r
                    for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)]
            px, front = _ts_project_np(scene, [tuple(c), tuple(c - up * r)] + [tuple(p) for p in ring])
            cpx = px[0]
            rad = float(np.max(np.linalg.norm(px[2:] - cpx, axis=1)))
            out["wheels"].append({"name": wh["name"], "center_px": [round(float(v), 1) for v in cpx],
                                  "radius_px": round(rad, 1),
                                  "side": "left" if (c - root.matrix_world.translation).dot(axle) < 0 else "right"})
            out["tyre_contacts_px"].append([round(float(v), 1) for v in px[1]])
        base = root.matrix_world.translation + up * float(root.get("ts_wheel_radius_m", 0.33))
        px, front = _ts_project_np(scene, [tuple(base), tuple(base + ahead * 2.0)])
        d = px[1] - px[0]
        n = float(np.hypot(*d))
        if n > 1e-6 and front.all():
            out["car_forward_dir_px"] = [round(float(d[0] / n), 4), round(float(d[1] / n), 4)]

    cd = cam.data
    out["camera"] = {
        "location": [round(float(v), 4) for v in cam.matrix_world.translation],
        "rotation_deg": [round(math.degrees(a), 3) for a in cam.matrix_world.to_euler("XYZ")],
        "lens_mm": round(cd.lens, 3), "sensor_width_mm": round(cd.sensor_width, 3),
        "sensor_fit": cd.sensor_fit, "shift_x": round(cd.shift_x, 5), "shift_y": round(cd.shift_y, 5),
    }
    sun = bpy.data.objects.get("TS_SUN")
    out["sun"] = None
    if sun is not None and sun.type == "LIGHT":
        to_sun = sun.matrix_world.to_3x3() @ Vector((0.0, 0.0, 1.0))
        az, el = _ts_dir_to_azel(to_sun)
        out["sun"] = {"azimuth_deg": round(az, 2), "elevation_deg": round(el, 2)}
    return out
