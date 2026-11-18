# Scene helpers shared by the Trueshot scene tools (runs inside Blender).
#
# Include after `_template_ts_common.py`:
#   # @include_begin: _template_ts_scene.py
#   # @include_end
#
# Documented scene state used here (see local://contracts.md):
# - scene props `ts_road_centerline` (flat x0,y0,x1,y1,... every ROAD_STEP_M), `ts_road_width_m`,
#   `ts_road_s0_m` (arc length of the first centreline sample; 0 = the world origin),
#   `ts_road_z` (height of the road surface);
# - compass angles: 0 deg = +Y, 90 deg = +X (clockwise seen from above).

__all__ = ()

ROAD_STEP_M = 0.5

# Flat guide tones by role (contract "Grey guide tones"); also used for viewport object colours.
TS_TONES = {
    "road": 0.22,
    "road_marking": 0.85,
    "ground": 0.55,
    "standin_ridge": 0.42,
    "standin_building": 0.68,
    "standin_tree": 0.35,
    "standin_wall": 0.6,
    "sea": 0.30,
}


def _ts_azel_to_dir(azimuth_deg: float, elevation_deg: float):
    """Unit vector for a compass azimuth (0 = +Y, 90 = +X) and elevation above horizontal."""
    import math
    from mathutils import Vector  # pylint: disable=import-error
    az, el = math.radians(azimuth_deg), math.radians(elevation_deg)
    return Vector((math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), math.sin(el)))


def _ts_dir_to_azel(d) -> tuple:
    """Inverse of `_ts_azel_to_dir`: (azimuth_deg in [0, 360), elevation_deg)."""
    import math
    n = math.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2) or 1.0
    az = math.degrees(math.atan2(d[0], d[1])) % 360.0
    el = math.degrees(math.asin(max(-1.0, min(1.0, d[2] / n))))
    return az, el


def _ts_tone_material(role: str):
    """A plain grey material for a set role (diffuse, the role's guide tone)."""
    import bpy  # pylint: disable=import-error
    name = "TS_mat_" + role
    mat = bpy.data.materials.get(name)
    if mat is None:
        mat = bpy.data.materials.new(name)
    tone = TS_TONES.get(role, 0.5)
    mat.diffuse_color = (tone, tone, tone, 1.0)
    mat.roughness = 1.0
    mat.use_nodes = True
    bsdf = next((n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = (tone, tone, tone, 1.0)
        bsdf.inputs["Roughness"].default_value = 0.6 if role == "sea" else 1.0
    return mat


def _ts_mesh_object(name: str, role: str, verts, faces, collection_name: str = "TS_SET"):
    """Create a mesh object in a TS collection with `ts_role`, a tone material and colour.

    `verts` is an (N, 3) float sequence/array, `faces` a list of index tuples.
    """
    import bpy  # pylint: disable=import-error
    old = bpy.data.objects.get(name)
    if old is not None:
        bpy.data.objects.remove(old, do_unlink=True)
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(map(float, v)) for v in verts], [], [tuple(map(int, f)) for f in faces])
    mesh.validate(clean_customdata=False)
    mesh.update()
    mesh.materials.append(_ts_tone_material(role))
    ob = bpy.data.objects.new(name, mesh)
    ob["ts_role"] = role
    tone = TS_TONES.get(role, 0.5)
    ob.color = (tone, tone, tone, 1.0)
    _ts_collection(collection_name).objects.link(ob)
    return ob


def _ts_road_centerline():
    """Scene road centreline as an (N, 2) numpy array, or None when no set is built."""
    import bpy  # pylint: disable=import-error
    import numpy as np
    flat = bpy.context.scene.get("ts_road_centerline")
    if flat is None or len(flat) < 4:
        return None
    return np.asarray(list(flat), dtype=np.float64).reshape(-1, 2)


def _ts_road_frame(distance_m: float):
    """(point_xy, unit_tangent_xy, clamped_distance) on the road at arc length `distance_m`."""
    import bpy  # pylint: disable=import-error
    import numpy as np
    cl = _ts_road_centerline()
    s0 = float(bpy.context.scene.get("ts_road_s0_m", 0.0))
    s_max = s0 + (len(cl) - 1) * ROAD_STEP_M
    s = min(max(distance_m, s0), s_max)
    f = (s - s0) / ROAD_STEP_M
    i = min(int(f), len(cl) - 2)
    t = f - i
    p = cl[i] * (1.0 - t) + cl[i + 1] * t
    tan = cl[i + 1] - cl[i]
    tan = tan / (np.linalg.norm(tan) or 1.0)
    return p, tan, s
