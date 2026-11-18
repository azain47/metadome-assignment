"""Tool-code: contact sheet of the car alone from many azimuths/elevations (Workbench, grey).

One temporary camera is keyframed to one pose per frame and the frames are rendered as a single
`INVOKE_DEFAULT` animation (Blender stays responsive; one job instead of one per cell). The
frames are then tiled into one labelled PNG plus a JSON index of the cells. Every scene change
goes through `_TsrOverrides` (restored on success, error and abandoned jobs); the temporary
camera, its data and its actions are removed on restore.
"""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    action: str  # "start" | "status"
    uploads: tuple = ()  # (("sheet", url, "image/png"), ("index", url, "application/json"))
    azimuths: tuple = (0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0, 300.0, 330.0)
    elevations: tuple = (3.0, 15.0)
    focal_mm: float = 85.0
    resolution: int = 320


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_render.py
# @include_end

TOOL_NAME = "render_pose_sheet"
CAM_NAME = "TS_TMP_POSESHEET_CAM"
SENSOR_MM = 36.0
FILL_W, FILL_H = 0.86, 0.74  # car bbox fill of a cell (fractions of width / height)
LABEL_ROOM = 0.05  # car centre this fraction of the cell height below the middle
GREY = (0.62, 0.62, 0.62)
BACKGROUND = (1.0, 1.0, 1.0)
SHEET_COLS = 6
LABEL_SCALE = 2

# 5x7 bitmap glyphs for the cell labels ("az 30 / el 3").
_GLYPHS = {
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "11110", "00001", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "a": ("00000", "00000", "01110", "00001", "01111", "10001", "01111"),
    "z": ("00000", "00000", "11111", "00010", "00100", "01000", "11111"),
    "e": ("00000", "00000", "01110", "10001", "11111", "10000", "01110"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "/": ("00001", "00010", "00010", "00100", "01000", "01000", "10000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    " ": ("00000",) * 7,
}

_SHEET_SETTINGS = {
    "render.engine": "BLENDER_WORKBENCH",
    "render.film_transparent": False,
    "render.use_motion_blur": False,
    "render.use_file_extension": True,
    "render.use_overwrite": True,
    "render.use_placeholder": False,
    "display.render_aa": "8",
    "display.shading.light": "STUDIO",
    "display.shading.color_type": "SINGLE",
    "display.shading.single_color": GREY,
    "display.shading.show_shadows": False,
    "display.shading.show_cavity": True,
    "display.shading.cavity_type": "SCREEN",
    "display.shading.show_object_outline": True,
    "display.shading.object_outline_color": (0.0, 0.0, 0.0),
    "display.shading.show_specular_highlight": True,
    "display.shading.show_xray": False,
    "display.shading.use_dof": False,
    "display.shading.show_backface_culling": False,
    "view_settings.view_transform": "Standard",
    "view_settings.look": "None",
    "view_settings.exposure": 0.0,
    "view_settings.gamma": 1.0,
    "view_settings.use_curve_mapping": False,
}


def _fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-6 else "{:.1f}".format(v)


def _poses(root, azimuths, elevations, lens, width, height) -> list:
    """Camera (location, quaternion, shift_x, shift_y) per (elevation, azimuth), framing the
    car's bounding box. Pure maths (no depsgraph updates): fast for any number of cells."""
    import math
    from mathutils import Vector  # pylint: disable=import-error
    b = [float(v) for v in root["ts_bbox_local"]]
    M = root.matrix_world
    corners = [M @ Vector((x, y, z)) for x in (b[0], b[3]) for y in (b[1], b[4]) for z in (b[2], b[5])]
    centre = sum(corners, Vector()) / 8.0
    quat = M.to_quaternion()
    k = lens / SENSOR_MM
    aspect = width / height
    size = max((c - centre).length for c in corners)
    out = []
    for el_deg in elevations:
        for az_deg in azimuths:
            az, el = math.radians(az_deg), math.radians(el_deg)
            dir_h = quat @ Vector((-math.sin(az), math.cos(az), 0.0))  # frame_shot convention
            dir_h.z = 0.0
            dir_h.normalize()
            view_dir = dir_h * math.cos(el) + Vector((0.0, 0.0, math.sin(el)))
            rot = view_dir.to_track_quat("Z", "Y")
            rinv = rot.inverted()
            d = size * 2.0 * k

            def project(dist):
                loc = centre + view_dir * dist
                pts = [rinv @ (c - loc) for c in corners]
                xs = [k * p.x / -p.z for p in pts]
                ys = [k * p.y / -p.z for p in pts]  # in units of the frame width
                return loc, xs, ys

            for _ in range(8):
                _loc, xs, ys = project(d)
                wf = max(xs) - min(xs)
                hf = (max(ys) - min(ys)) * aspect
                d *= max(wf / FILL_W, hf / FILL_H) ** 0.9
            loc, xs, ys = project(d)
            # Blender's lens shift moves the frame, so the content moves the opposite way; the car
            # sits a little below the cell centre, clear of the label band.
            sx = (max(xs) + min(xs)) / 2.0
            sy = (max(ys) + min(ys)) / 2.0 + LABEL_ROOM / aspect
            out.append({"az": float(az_deg), "el": float(el_deg), "location": tuple(loc),
                        "rotation": tuple(rot), "shift_x": sx, "shift_y": sy, "distance_m": round(d, 3)})
    return out


def _make_camera(ov, scene, poses, lens, frame0):
    import bpy  # pylint: disable=import-error
    for kind in ("objects", "cameras"):
        _tsr_remove_ids(kind, [CAM_NAME])  # leftovers of a crashed run
    data = bpy.data.cameras.new(CAM_NAME)
    cam = bpy.data.objects.new(CAM_NAME, data)
    cam["ts_role"] = "tmp_pose_sheet_camera"
    scene.collection.objects.link(cam)
    ov.call_on_restore("_tsr_remove_ids", "cameras", [data.name])
    ov.call_on_restore("_tsr_remove_ids", "objects", [cam.name])
    data.type = "PERSP"
    data.lens = lens
    data.sensor_fit = "HORIZONTAL"
    data.sensor_width = SENSOR_MM
    data.clip_start = 0.05
    data.clip_end = 10000.0
    cam.rotation_mode = "QUATERNION"
    for i, p in enumerate(poses):
        f = frame0 + i
        cam.location = p["location"]
        cam.rotation_quaternion = p["rotation"]
        data.shift_x, data.shift_y = p["shift_x"], p["shift_y"]
        cam.keyframe_insert("location", frame=f)
        cam.keyframe_insert("rotation_quaternion", frame=f)
        data.keyframe_insert("shift_x", frame=f)
        data.keyframe_insert("shift_y", frame=f)
    actions = [ad.action.name for ad in (cam.animation_data, data.animation_data) if ad and ad.action]
    ov.call_on_restore("_tsr_remove_ids", "actions", actions)
    ov.set(scene, "camera", cam)
    return cam


def _draw_label(px, text: str, x0: int, y0: int, scale: int) -> None:
    """Dark text on a light band; `px` is a top-down HxWx4 float array, (x0, y0) top-left."""
    gw = 6 * scale
    band_w = gw * len(text) + 2 * scale
    band_h = 9 * scale
    px[y0:y0 + band_h, x0:x0 + band_w, :3] = 1.0
    for i, ch in enumerate(text):
        rows = _GLYPHS.get(ch, _GLYPHS[" "])
        for r, row in enumerate(rows):
            for c, bit in enumerate(row):
                if bit == "1":
                    ys = y0 + scale + r * scale
                    xs = x0 + scale + i * gw + c * scale
                    px[ys:ys + scale, xs:xs + scale, :3] = 0.08


def _compose(job) -> tuple:
    """Tile the rendered frames into the sheet PNG and write the JSON index. Returns paths."""
    import json
    import numpy as np
    import bpy  # pylint: disable=import-error
    info = job["info"]
    w, h, n = info["width"], info["height"], len(info["cells"])
    cols = min(SHEET_COLS, n)
    rows = (n + cols - 1) // cols
    gap = 2
    sw, sh = cols * w + (cols - 1) * gap, rows * h + (rows - 1) * gap
    sheet = np.full((sh, sw, 4), 0.55, dtype=np.float32)
    sheet[..., 3] = 1.0
    cells = []
    for i, cell in enumerate(info["cells"]):
        path = info["frame_paths"][i]
        if not os.path.exists(path):
            raise RuntimeError("frame {} was not rendered ({})".format(cell["frame"], path))
        img = bpy.data.images.load(path, check_existing=False)
        try:
            buf = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
            img.pixels.foreach_get(buf)
            tile = buf.reshape(img.size[1], img.size[0], 4)[::-1]  # bottom-up -> top-down
        finally:
            bpy.data.images.remove(img)
        r, c = divmod(i, cols)
        x, y = c * (w + gap), r * (h + gap)
        sheet[y:y + h, x:x + w] = tile[:h, :w]
        label = "az {} / el {}".format(_fmt(cell["azimuth_deg"]), _fmt(cell["elevation_deg"]))
        _draw_label(sheet, label, x + 4, y + 4, LABEL_SCALE)
        cells.append(dict(cell, index=i, label=label, x=x, y=y, w=w, h=h))
    out_dir = _ts_tmp_dir(TOOL_NAME)
    sheet_path = os.path.join(out_dir, "pose_sheet.png")
    img = bpy.data.images.new("TS_TMP_POSESHEET", sw, sh, alpha=False)
    try:
        img.pixels.foreach_set(sheet[::-1].ravel())
        img.filepath_raw = sheet_path
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)
    index = {"width": sw, "height": sh, "cell_width": w, "cell_height": h, "cols": cols, "rows": rows,
             "focal_mm": info["focal_mm"], "convention": (
                 "azimuth 0 = camera in front of the car looking at its front, 90 = car's left side, "
                 "180 = behind, 270 = right side; elevation above horizontal"),
             "cells": cells}
    index_path = os.path.join(out_dir, "pose_sheet.json")
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    info["sheet_size"] = [sw, sh]
    info["cells_out"] = cells
    return sheet_path, index_path


def _start(params: Params, scene, job) -> None:
    import bpy  # pylint: disable=import-error
    root = _ts_car_root()
    if root is None:
        raise RuntimeError("No car loaded (call load_car first)")
    azs = [float(a) for a in params.azimuths]
    els = [float(e) for e in params.elevations]
    if not azs or not els:
        raise RuntimeError("azimuths and elevations must not be empty")
    width = max(64, int(params.resolution))
    height = max(43, int(round(width * 2 / 3)))
    lens = max(10.0, float(params.focal_mm))
    poses = _poses(root, azs, els, lens, width, height)

    ov = _TsrOverrides(job["saved"])
    _tsr_ensure_world(ov, scene)
    frame0 = int(scene.frame_current)
    out_dir = _ts_tmp_dir(TOOL_NAME)
    for name in os.listdir(out_dir):
        if name.startswith("cell_"):
            os.remove(os.path.join(out_dir, name))
    ov.scene(scene, {
        "render.resolution_x": width,
        "render.resolution_y": height,
        "render.resolution_percentage": 100,
        "render.pixel_aspect_x": 1.0,
        "render.pixel_aspect_y": 1.0,
        "render.use_border": False,
        "render.use_single_layer": True,
        "render.use_compositing": False,
        "render.use_sequencer": False,
        "render.use_stamp": False,
        "frame_start": frame0,
        "frame_end": frame0 + len(poses) - 1,
        "frame_step": 1,
        "frame_current": frame0,
        "render.filepath": os.path.join(out_dir, "cell_####"),
        **_SHEET_SETTINGS,
    })
    ov.set(scene.world, "color", BACKGROUND)
    _tsr_set_format(ov, scene, "PNG_RGB8")
    _tsr_show_only(ov, scene, _tsr_groups(scene)["car"])
    _make_camera(ov, scene, poses, lens, frame0)

    job["info"].update(
        width=width, height=height, focal_mm=lens,
        frame_paths=[os.path.join(out_dir, "cell_{:04d}.png".format(frame0 + i)) for i in range(len(poses))],
        cells=[{"azimuth_deg": p["az"], "elevation_deg": p["el"], "frame": frame0 + i,
                "distance_m": p["distance_m"]} for i, p in enumerate(poses)])
    job["pass_t0"] = time.time()
    _ts_state().pop("tsr_render_end_t", None)
    _tsr_handlers(True)
    bpy.ops.render.render("INVOKE_DEFAULT", animation=True)


def main(params: Params):
    import traceback
    import bpy  # pylint: disable=import-error

    t_call = time.perf_counter()
    state = _ts_state()
    scene = bpy.context.scene
    job = state.get("render_job")

    def lap(label: str) -> None:
        job["calls"].append([label, round(time.perf_counter() - t_call, 3)])

    if params.action == "start":
        if bpy.app.is_job_running("RENDER"):
            return _ts_busy_result()
        if job is not None:
            if time.time() - job.get("last_poll", 0) < TSR_STALE_S:
                return _ts_result("busy", message="Render job {!r} is running; poll it first.".format(job["tool"]))
            _tsr_restore_job(job)
        # Same job layout as `_tsr_chain`, so any render tool can restore an abandoned sheet job.
        job = {"tool": TOOL_NAME, "scene": scene.name, "step": 0, "saved": [], "step_saved": [],
               "info": {}, "extra_files": {}, "uploads": [], "pass_s": {}, "errors": [],
               "phase": "render", "out": {}, "t0": time.time(), "last_poll": time.time(), "calls": []}
        state["render_job"] = job
        try:
            _start(params, scene, job)
        except Exception:  # noqa: BLE001
            return _tsr_abort(job, traceback.format_exc()[-3000:])
        lap("start")
        return _ts_result("started", tool=TOOL_NAME, cells=len(job["info"]["cells"]))

    if job is None or job["tool"] != TOOL_NAME:
        return _ts_result("error", message="No {} job is running.".format(TOOL_NAME))
    job["last_poll"] = time.time()
    if job["phase"] == "uploading":
        return _tsr_result(job) or _ts_result("running", tool=TOOL_NAME, pass_name="upload",
                                               elapsed_s=round(time.time() - job["t0"], 1))
    if bpy.app.is_job_running("RENDER"):
        done = sum(os.path.exists(p) for p in job["info"]["frame_paths"])
        return _ts_result("running", tool=TOOL_NAME, pass_name="cells", done=done,
                          total=len(job["info"]["frame_paths"]), elapsed_s=round(time.time() - job["t0"], 1))
    try:
        end_t = _ts_state().get("tsr_render_end_t") or time.time()
        job["pass_s"]["cells"] = round(end_t - job["pass_t0"], 2)
        job["errors"] += _tsr_restore_job(job)  # scene back first; composing needs no settings
        t0 = time.time()
        sheet_path, index_path = _compose(job)
        job["pass_s"]["compose"] = round(time.time() - t0, 2)
        _tsr_upload_async(job, params.uploads, "sheet", sheet_path)
        _tsr_upload_async(job, params.uploads, "index", index_path)
        info = job["info"]
        job["out"] = {"sheet_size": info["sheet_size"], "cell_size": [info["width"], info["height"]],
                      "focal_mm": info["focal_mm"], "cells": info["cells_out"]}
        job["phase"] = "uploading"
    except Exception:  # noqa: BLE001
        return _tsr_abort(job, traceback.format_exc()[-3000:])
    lap("status:finish")
    return _tsr_result(job) or _ts_result("running", tool=TOOL_NAME, pass_name="upload",
                                           elapsed_s=round(time.time() - job["t0"], 1))
