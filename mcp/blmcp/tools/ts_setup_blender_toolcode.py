"""Tool-code: prepare Blender for Trueshot (GPU device, units, TS collections)."""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    use_gpu: bool


# @include_begin: _template_ts_common.py
# @include_end


def main(params: Params):
    import bpy  # pylint: disable=import-error

    if _ts_busy():
        return _ts_busy_result()
    scene = bpy.context.scene

    device_type = "NONE"
    devices: list = []
    prefs = bpy.context.preferences.addons["cycles"].preferences
    if params.use_gpu:
        for kind in ("OPTIX", "CUDA"):
            try:
                prefs.compute_device_type = kind
            except TypeError:
                continue
            prefs.refresh_devices()
            found = [d for d in prefs.get_devices_for_type(kind) if d.type == kind]
            if found:
                device_type = kind
                for d in prefs.devices:
                    d.use = d.type == kind
                devices = [d.name for d in found]
                break
    if device_type == "NONE":
        prefs.compute_device_type = "NONE"
        device_type = "CPU"
        devices = ["CPU"]
    scene.cycles.device = "GPU" if device_type != "CPU" else "CPU"

    bpy.context.preferences.view.render_display_type = "NONE"
    us = scene.unit_settings
    us.system = "METRIC"
    us.scale_length = 1.0
    us.length_unit = "METERS"

    for name in (TS_CAR, TS_SET, TS_RIG):
        _ts_collection(name)

    # Hide the start-up scene's own objects (default cube, camera, light) so they never
    # appear in our renders; nothing is deleted.
    ours = set()
    for name in (TS_CAR, TS_SET, TS_RIG):
        ours.update(bpy.data.collections[name].all_objects)
    hidden = []
    for ob in scene.objects:
        if ob in ours:
            continue
        if not ob.hide_render or not ob.hide_get():
            ob.hide_render = True
            ob.hide_set(True)
            hidden.append(ob.name)
    if scene.camera is not None and scene.camera not in ours:
        scene.camera = None

    _ts_state()["setup"] = True
    _ts_redraw()
    return _ts_result(
        "ok",
        device_type=device_type,
        devices=devices,
        blender_version=bpy.app.version_string,
        hidden_objects=hidden,
    )
