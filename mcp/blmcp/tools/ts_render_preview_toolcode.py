"""Tool-code: fast preview through TS_CAM with the car visible (Workbench, studio light)."""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    action: str  # "start" | "status"
    uploads: tuple = ()  # ((name, presigned_put_url, content_type), ...)
    resolution_x: int = 768


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end

# @include_begin: _template_ts_layout.py
# @include_end

# @include_begin: _template_ts_render.py
# @include_end

PREVIEW_SKY = (0.45, 0.6, 0.85)

_PREVIEW = {
    "render.engine": "BLENDER_WORKBENCH",
    "render.film_transparent": False,
    "render.use_motion_blur": False,
    "display.render_aa": "8",
    "display.shading.light": "STUDIO",
    "display.shading.color_type": "MATERIAL",
    "display.shading.show_shadows": False,
    "display.shading.show_cavity": False,
    "display.shading.show_object_outline": False,
    "display.shading.show_specular_highlight": True,
    "display.shading.show_xray": False,
    "display.shading.use_dof": False,
    "view_settings.view_transform": "Standard",
    "view_settings.look": "None",
}


def _setup_preview(ov, scene, job) -> None:
    groups = _tsr_groups(scene)
    _tsr_show_only(ov, scene, groups["car"] + groups["set"])
    ov.set(scene.world, "color", PREVIEW_SKY)


PASSES = [
    {"name": "preview", "format": "PNG_RGB8", "upload": "preview", "settings": _PREVIEW, "setup": _setup_preview},
]


def _prepare_for(params: Params):
    def prepare(ov, scene, job) -> dict:
        w, h = _ts_render_size(scene)
        width = max(16, int(params.resolution_x))
        height = max(16, round(width * h / max(w, 1)))
        ov.scene(scene, {"render.resolution_x": width, "render.resolution_y": height})
        _tsr_ensure_world(ov, scene)
        job["info"].update(width=width, height=height, car_bbox_px=_ts_car_bbox_px(scene))
        return {}
    return prepare


def _finish(scene, job) -> dict:
    return {k: job["info"][k] for k in ("width", "height", "car_bbox_px")}


def main(params: Params):
    return _tsr_chain(params, "render_preview", PASSES, _prepare_for(params), _finish)
