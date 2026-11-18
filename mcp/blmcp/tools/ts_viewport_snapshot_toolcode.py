"""Tool-code: screenshot of the largest 3D viewport, uploaded as PNG."""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    uploads: tuple = ()  # (("snapshot", presigned_put_url, content_type),)


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_render.py
# @include_end


def main(params: Params):
    import os
    import bpy  # pylint: disable=import-error

    if bpy.app.background:
        return _ts_result("error", message="Screenshots need Blender's UI (not background mode).")
    best = None
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D" and (best is None or area.width * area.height > best[1].width * best[1].height):
                best = (window, area)
    if best is None:
        return _ts_result("error", message="No 3D viewport is open.")
    window, area = best
    path = os.path.join(_ts_tmp_dir("viewport_snapshot"), "snapshot.png")
    if os.path.exists(path):
        os.remove(path)
    region = next((r for r in area.regions if r.type == "WINDOW"), None)
    with bpy.context.temp_override(window=window, screen=window.screen, area=area, region=region):
        bpy.ops.screen.screenshot_area(filepath=path, check_existing=False)
    if not os.path.exists(path):
        return _ts_result("error", message="screenshot_area wrote no file.")
    _name, url, content_type = params.uploads[0]
    _tsr_put(url, path, content_type)
    return _ts_result("ok", width=area.width, height=area.height, bytes=os.path.getsize(path))
