"""Tool-code: clear the Trueshot set and rig (and optionally the car)."""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    keep_car: bool


# @include_begin: _template_ts_common.py
# @include_end


def main(params: Params):
    import bpy  # pylint: disable=import-error

    if _ts_busy():
        return _ts_busy_result()
    scene = bpy.context.scene
    removed = {TS_SET: _ts_clear_collection(TS_SET), TS_RIG: _ts_clear_collection(TS_RIG)}
    if not params.keep_car:
        removed[TS_CAR] = _ts_clear_collection(TS_CAR)
    for key in ("ts_road_centerline", "ts_road_width_m", "ts_set_terrain", "ts_road_shape",
                "ts_road_s0_m", "ts_road_z", "ts_set_seed", "ts_sea_side"):
        if key in scene:
            del scene[key]
    if scene.camera is not None and scene.camera.name not in scene.objects:
        scene.camera = None
    bpy.data.orphans_purge(do_local_ids=True, do_linked_ids=True, do_recursive=True)
    for name in (TS_CAR, TS_SET, TS_RIG):
        _ts_collection(name)
    _ts_redraw()
    return _ts_result("ok", removed=removed, kept_car=_ts_car_root() is not None)
