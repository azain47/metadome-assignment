"""Tool-code: put TS_CAR_ROOT on the road centreline."""

__all__ = ("Params", "main", "LANES")

from typing import NamedTuple

# Lateral offset in road widths, positive = right of the direction of travel.
_LANE_OFFSET = {"left": -0.25, "center": 0.0, "right": 0.25}
LANES = tuple(_LANE_OFFSET)


class Params(NamedTuple):
    distance_m: float
    lane: str
    heading_offset_deg: float


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end


def main(params: Params):
    import math
    import bpy  # pylint: disable=import-error

    if _ts_busy():
        return _ts_busy_result()
    root = _ts_car_root()
    if root is None:
        return _ts_result("error", message="No car loaded (call load_car first)")
    if _ts_road_centerline() is None:
        return _ts_result("error", message="No road (call build_set first)")
    if params.lane not in _LANE_OFFSET:
        return _ts_result("error", message="lane must be one of {}".format(LANES))
    scene = bpy.context.scene
    p, tan, s = _ts_road_frame(float(params.distance_m))
    right = (tan[1], -tan[0])
    off = _LANE_OFFSET[params.lane] * float(scene.get("ts_road_width_m", 7.0))
    x, y = float(p[0] + right[0] * off), float(p[1] + right[1] * off)
    heading = math.degrees(math.atan2(tan[0], tan[1])) + float(params.heading_offset_deg)  # compass
    root.location = (x, y, float(scene.get("ts_road_z", 0.0)))
    root.rotation_mode = "XYZ"
    root.rotation_euler = (0.0, 0.0, math.radians(-heading))
    root.scale = (1.0, 1.0, 1.0)
    bpy.context.view_layer.update()
    _ts_redraw()
    return _ts_result(
        "ok",
        location=[round(x, 3), round(y, 3), round(root.location.z, 3)],
        heading_deg=round(heading % 360.0, 2),
        distance_m=round(s, 2),
        lane=params.lane,
    )
