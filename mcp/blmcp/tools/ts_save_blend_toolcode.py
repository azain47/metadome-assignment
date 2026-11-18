"""Tool-code: save a compressed copy of the open scene as a .blend and upload it.

`copy=True` leaves the session's own file path and dirty state untouched. Images (HDRIs,
textures) stay external: they live in the Blender machine's trueshot cache, next to the local copy.
"""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    filename: str
    uploads: tuple = ()  # (("blend", presigned_put_url, content_type),)


# @include_begin: _template_ts_common.py
# @include_end


def main(params: Params):
    import os
    import bpy  # pylint: disable=import-error

    if _ts_busy():
        return _ts_busy_result()
    path = os.path.join(_ts_tmp_dir("blends"), params.filename)
    win = bpy.context.window_manager.windows[0] if bpy.context.window_manager.windows else None
    if win is not None:
        with bpy.context.temp_override(window=win):
            bpy.ops.wm.save_as_mainfile(filepath=path, copy=True, compress=True)
    else:
        bpy.ops.wm.save_as_mainfile(filepath=path, copy=True, compress=True)
    for _name, url, content_type in params.uploads:
        _ts_http_put(url, path, content_type)
    return _ts_result("ok", local_path=path, bytes=os.path.getsize(path))
