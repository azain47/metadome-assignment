"""Tool-code: summarise the Trueshot scene state (read-only)."""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    pass


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end

def main(params: Params):  # pylint: disable=unused-argument
    import math
    import bpy  # pylint: disable=import-error

    scene = bpy.context.scene
    r3 = lambda v: [round(float(x), 3) for x in v]  # noqa: E731

    car = None
    root = _ts_car_root()
    if root is not None:
        car = {
            "key": str(root.get("ts_car_key", "")),
            "root": root.name,
            "dims_m": r3(root.get("ts_dims_m", [])),
            "wheels": [str(w) for w in root.get("ts_wheels", [])],
            "wheel_radius_m": round(float(root.get("ts_wheel_radius_m", 0.0)), 3),
            "location": r3(root.location),
            "heading_deg": round((-math.degrees(root.rotation_euler.z)) % 360.0, 2),
            "paint_materials": [str(m) for m in root.get("ts_paint_materials", [])],
            "objects": len(bpy.data.collections[TS_CAR].all_objects),
        }

    set_info = None
    if "ts_set_terrain" in scene:
        cl = scene.get("ts_road_centerline", [])
        set_info = {
            "terrain": str(scene["ts_set_terrain"]),
            "road_shape": str(scene.get("ts_road_shape", "")),
            "road_width_m": float(scene.get("ts_road_width_m", 0.0)),
            "road_length_m": round(max(0, len(cl) // 2 - 1) * ROAD_STEP_M, 1),
            "road_start_m": float(scene.get("ts_road_s0_m", 0.0)),
            "seed": int(scene.get("ts_set_seed", 0)),
            "objects": len(bpy.data.collections[TS_SET].all_objects) if TS_SET in bpy.data.collections else 0,
        }

    camera = None
    cam = scene.camera
    if cam is not None and cam.type == "CAMERA":
        camera = {
            "name": cam.name,
            "location": r3(cam.matrix_world.translation),
            "rotation_deg": [round(math.degrees(a), 2) for a in cam.matrix_world.to_euler("XYZ")],
            "lens_mm": round(cam.data.lens, 3),
            "sensor_width_mm": round(cam.data.sensor_width, 3),
            "shift_x": round(cam.data.shift_x, 4),
            "shift_y": round(cam.data.shift_y, 4),
            "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        }
        for k in ("ts_azimuth_deg", "ts_elevation_deg", "ts_distance_m"):
            if k in cam:
                camera[k[3:]] = round(float(cam[k]), 3)

    sun = None
    sun_ob = bpy.data.objects.get("TS_SUN")
    if sun_ob is not None and sun_ob.name in scene.objects:
        from mathutils import Vector  # pylint: disable=import-error
        az, el = _ts_dir_to_azel(sun_ob.matrix_world.to_quaternion() @ Vector((0.0, 0.0, 1.0)))
        sun = {
            "azimuth_deg": round(az, 2),
            "elevation_deg": round(el, 2),
            "strength": round(sun_ob.data.energy, 3),
            "angle_deg": round(math.degrees(sun_ob.data.angle), 3),
        }

    world = {
        "name": scene.world.name if scene.world else None,
        "sky": str(scene.get("ts_world_sky", "")) or None,
        "hdri_key": str(scene.get("ts_world_hdri_key", "")) or None,
        "strength": float(scene.get("ts_world_strength", 0.0)) if "ts_world_strength" in scene else None,
        "exposure": round(scene.view_settings.exposure, 3),
        "view_transform": scene.view_settings.view_transform,
    }

    return _ts_result(
        "ok",
        car=car,
        set=set_info,
        camera=camera,
        sun=sun,
        world=world,
        render_engine=scene.render.engine,
        is_rendering=_ts_busy(),
    )
