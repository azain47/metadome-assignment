"""Tool-code: render the car-free guides (grey, road mask, car mask, depth) and the layout."""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    action: str  # "start" | "status"
    uploads: tuple = ()  # ((name, presigned_put_url, content_type), ...)


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end

# @include_begin: _template_ts_layout.py
# @include_end

# @include_begin: _template_ts_render.py
# @include_end

SKY_TONE = 0.92  # guide background (contract "Grey guide tones")


def _role(ob) -> str:
    return str(ob.get("ts_role", ""))


def _show_set(ov, scene, colour_of, background: float) -> None:
    """Only the set (car hidden), flat object colours, flat background."""
    groups = _tsr_groups(scene)
    _tsr_show_only(ov, scene, groups["set"])
    _tsr_object_colours(ov, _tsr_visible(groups["set"]), colour_of)
    ov.set(scene.world, "color", (background,) * 3)


def _setup_grey(ov, scene, job) -> None:
    _show_set(ov, scene, lambda ob: TS_TONES.get(_role(ob), TS_TONES["ground"]), SKY_TONE)


def _setup_road_mask(ov, scene, job) -> None:
    _show_set(ov, scene, lambda ob: 1.0 if _role(ob) in ("road", "road_marking") else 0.0, 0.0)


def _setup_car_mask(ov, scene, job) -> None:
    car = _tsr_groups(scene)["car"]
    _tsr_show_only(ov, scene, car)
    _tsr_object_colours(ov, _tsr_visible(car), lambda ob: 1.0)
    ov.set(scene.world, "color", (0.0, 0.0, 0.0))


def _setup_depth(ov, scene, job) -> None:
    # Cycles depth (exact, planar metres); a black material override and transparent film make
    # the unavoidable Combined layer all zeros, so the EXR stays small.
    _tsr_show_only(ov, scene, _tsr_groups(scene)["set"])
    _tsr_only_passes(ov, {"use_pass_combined", "use_pass_z"})
    ov.scene(scene, {"view_layer.material_override": _tsr_black_material(ov)})


def _after_depth(ov, scene, job, path) -> dict:
    job["info"]["depth_channels"] = _tsr_exr_channels(path)
    return {}


# Depth: one Cycles sample at the pixel centre, no bounces, no motion blur.
_DEPTH = {
    "render.engine": "CYCLES",
    "render.film_transparent": True,
    "render.use_motion_blur": False,
    "cycles.samples": 1,
    "cycles.use_layer_samples": "IGNORE",
    "cycles.use_adaptive_sampling": False,
    "cycles.use_denoising": False,
    "cycles.use_pixel_jitter": False,
    "cycles.max_bounces": 0,
    "cycles.time_limit": 0.0,
}


def _passes() -> list:
    flat = {**TSR_WORKBENCH_FLAT, **TSR_RAW_VIEW}  # template constants: resolve inside Blender
    return [
        {"name": "guide_grey", "format": "PNG_RGB8", "upload": "grey", "settings": flat, "setup": _setup_grey},
        {"name": "mask_road", "format": "PNG_BW8", "upload": "road_mask", "settings": flat, "setup": _setup_road_mask},
        {"name": "mask_car", "format": "PNG_BW8", "upload": "car_mask", "settings": flat, "setup": _setup_car_mask},
        {"name": "depth", "format": "EXR_MULTI32", "upload": "depth", "settings": _DEPTH, "setup": _setup_depth,
         "after": _after_depth},
    ]


def _prepare(ov, scene, job) -> dict:
    import json
    import os
    _tsr_ensure_world(ov, scene)
    layout = _ts_layout(scene)
    job["info"]["layout"] = layout
    path = os.path.join(_ts_tmp_dir("render_guides"), "layout.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(layout, fh, indent=1)
    return {"layout": path}


def _finish(scene, job) -> dict:
    return {"layout": job["info"]["layout"], "depth_channels": job["info"].get("depth_channels", []),
            "depth_units": "metres along the camera axis; sky = 1e10"}


def main(params: Params):
    return _tsr_chain(params, "render_guides", _passes(), _prepare, _finish)
