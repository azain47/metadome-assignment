# Shared tool-code helpers for Trueshot tools (runs inside Blender).
#
# Include in a `ts_*_toolcode.py` with:
#   # @include_begin: _template_ts_common.py
#   # @include_end
# The block may stay empty in the source file: the server substitutes this file's
# contents before sending, and these helpers are only called inside `main()`.
#
# Conventions (see local://contracts.md):
# - Collections: TS_CAR (car objects under the empty TS_CAR_ROOT), TS_SET (road, ground,
#   stand-ins, sea), TS_RIG (camera TS_CAM, sun TS_SUN).
# - Every object we create or import gets a custom property `ts_role`.
# - Cross-call state lives in bpy.app.driver_namespace["trueshot"] and in scene custom
#   properties prefixed `ts_`.

__all__ = ()

import os
import tempfile

TS_CAR = "TS_CAR"
TS_SET = "TS_SET"
TS_RIG = "TS_RIG"


def _ts_state() -> dict:
    import bpy  # pylint: disable=import-error
    return bpy.app.driver_namespace.setdefault("trueshot", {})


def _ts_collection(name: str):
    import bpy  # pylint: disable=import-error
    scene = bpy.context.scene
    coll = bpy.data.collections.get(name)
    if coll is None:
        coll = bpy.data.collections.new(name)
    if coll.name not in scene.collection.children:
        scene.collection.children.link(coll)
    return coll


def _ts_busy() -> bool:
    import bpy  # pylint: disable=import-error
    return bool(bpy.app.is_job_running("RENDER"))


def _ts_busy_result() -> dict:
    return {"status": "busy", "message": "A render is running; wait for it to finish (poll its status)."}


def _ts_redraw(camera_view: bool = True) -> None:
    """Show the shot camera in every 3D viewport and redraw all areas."""
    import bpy  # pylint: disable=import-error
    scene = bpy.context.scene
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if camera_view and area.type == "VIEW_3D" and scene.camera is not None:
                space = area.spaces.active
                if space.region_3d.view_perspective != "CAMERA":
                    space.region_3d.view_perspective = "CAMERA"
                region = next((r for r in area.regions if r.type == "WINDOW"), None)
                if region is not None:
                    try:
                        with bpy.context.temp_override(window=window, area=area, region=region):
                            bpy.ops.view3d.view_center_camera()
                    except RuntimeError:
                        pass
            area.tag_redraw()


def _ts_cache_dir() -> str:
    path = os.path.join(tempfile.gettempdir(), "trueshot_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _ts_tmp_dir(sub: str) -> str:
    path = os.path.join(tempfile.gettempdir(), "trueshot_out", sub)
    os.makedirs(path, exist_ok=True)
    return path


def _ts_http_get_cached(url: str, cache_name: str) -> str:
    """Download `url` to the cache once; `cache_name` must identify the content (key + etag)."""
    import requests
    path = os.path.join(_ts_cache_dir(), cache_name)
    if not os.path.exists(path):
        tmp = path + ".part"
        with requests.get(url, stream=True, timeout=300) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        os.replace(tmp, path)
    return path


def _ts_http_put(url: str, path: str, content_type: str) -> None:
    import requests
    with open(path, "rb") as fh:
        r = requests.put(url, data=fh, headers={"Content-Type": content_type}, timeout=300)
    r.raise_for_status()


def _ts_project(points_world) -> list:
    """Project world points through the scene camera.

    Returns [x_px, y_px, in_front] per point, with y measured from the TOP of the image,
    at the scene's current render resolution.
    """
    import bpy  # pylint: disable=import-error
    from bpy_extras.object_utils import world_to_camera_view  # pylint: disable=import-error
    from mathutils import Vector  # pylint: disable=import-error
    scene = bpy.context.scene
    cam = scene.camera
    rd = scene.render
    w = rd.resolution_x * rd.resolution_percentage / 100.0
    h = rd.resolution_y * rd.resolution_percentage / 100.0
    out = []
    for p in points_world:
        co = world_to_camera_view(scene, cam, Vector(p))
        out.append([round(co.x * w, 2), round((1.0 - co.y) * h, 2), bool(co.z > 0)])
    return out


def _ts_objects_by_role(role_prefix: str) -> list:
    import bpy  # pylint: disable=import-error
    return [o for o in bpy.context.scene.objects if str(o.get("ts_role", "")).startswith(role_prefix)]


class _TsResult(dict):
    """A dict result: the calling-convention footer calls `main(...)._asdict()`."""

    def _asdict(self) -> dict:
        return dict(self)


def _ts_result(status: str = "ok", **fields) -> "_TsResult":
    """Return value for `main()`: `return _ts_result("ok", key=value, ...)`."""
    out = _TsResult(fields)
    out["status"] = status
    return out


def _ts_clear_collection(name: str) -> int:
    """Delete every object in a TS collection (and its children collections' objects)."""
    import bpy  # pylint: disable=import-error
    coll = bpy.data.collections.get(name)
    if coll is None:
        return 0
    obs = list(coll.all_objects)
    if obs:
        bpy.data.batch_remove(obs)
    return len(obs)


def _ts_car_root():
    """The car root empty (found by its `ts_car_key` prop), or None."""
    import bpy  # pylint: disable=import-error
    coll = bpy.data.collections.get(TS_CAR)
    if coll is None:
        return None
    for ob in coll.all_objects:
        if ob.type == "EMPTY" and "ts_car_key" in ob:
            return ob
    return None
