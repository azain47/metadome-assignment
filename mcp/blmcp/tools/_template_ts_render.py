# Render-chain helpers for the Trueshot render tools (runs inside Blender).
#
# Include after `_template_ts_common.py` (and `_template_ts_scene.py` for TS_TONES):
#   # @include_begin: _template_ts_render.py
#   # @include_end
#
# A render tool is a small table of passes run by `_tsr_chain()` through the start/status
# protocol (`run_job` on the server): one `bpy.ops.render.render('INVOKE_DEFAULT',
# write_still=True)` per pass, so Blender stays responsive. Every setting a tool or pass
# changes goes through an `_TsrOverrides` recorder and is restored afterwards: pass-level
# overrides after each pass, tool-level overrides when the chain ends (or fails).
#
# Pass spec (dict):
#   name      str, also the temp file stem
#   format    key of _TSR_FORMATS (how the render is written)
#   upload    upload name for the rendered file ("" = keep local only)
#   settings  {path: value} scene-relative overrides, e.g. {"render.engine": "CYCLES"};
#             paths starting with "view_layer." / "world." resolve on the active view layer /
#             scene world
#   setup     optional fn(ov, scene, job) for computed overrides (visibility, colours, ...)
#   after     optional fn(ov, scene, job, path) -> {upload_name: local_path} for files derived
#             from the finished render (runs before the pass overrides are restored)
# Uploads: Params.uploads = ((name, presigned_put_url, content_type), ...).

__all__ = ()

import os
import time

TSR_STALE_S = 30.0  # a job whose status was not polled for this long is abandoned

_TSR_FORMATS = {
    "PNG_RGB8": (("media_type", "IMAGE"), ("file_format", "PNG"), ("color_mode", "RGB"),
                 ("color_depth", "8"), ("compression", 15), ("color_management", "FOLLOW_SCENE")),
    "PNG_BW8": (("media_type", "IMAGE"), ("file_format", "PNG"), ("color_mode", "BW"),
                ("color_depth", "8"), ("compression", 15), ("color_management", "FOLLOW_SCENE")),
    "PNG_RGBA8": (("media_type", "IMAGE"), ("file_format", "PNG"), ("color_mode", "RGBA"),
                  ("color_depth", "8"), ("compression", 15), ("color_management", "FOLLOW_SCENE")),
    "EXR_MULTI32": (("media_type", "MULTI_LAYER_IMAGE"), ("file_format", "OPEN_EXR_MULTILAYER"),
                    ("color_depth", "32"), ("exr_codec", "ZIP"), ("use_exr_interleave", False)),
}
_TSR_EXT = {"PNG_RGB8": ".png", "PNG_BW8": ".png", "PNG_RGBA8": ".png", "EXR_MULTI32": ".exr"}

# Display settings that make 8-bit outputs a plain encoding of the render values.
TSR_RAW_VIEW = {
    "view_settings.view_transform": "Raw",
    "view_settings.look": "None",
    "view_settings.exposure": 0.0,
    "view_settings.gamma": 1.0,
    "view_settings.use_curve_mapping": False,
    "render.dither_intensity": 0.0,
}

# Workbench: flat, unlit object colours (guides and masks).
TSR_WORKBENCH_FLAT = {
    "render.engine": "BLENDER_WORKBENCH",
    "render.film_transparent": False,
    "display.render_aa": "8",
    "display.shading.light": "FLAT",
    "display.shading.color_type": "OBJECT",
    "display.shading.show_shadows": False,
    "display.shading.show_cavity": False,
    "display.shading.show_object_outline": False,
    "display.shading.show_specular_highlight": False,
    "display.shading.show_xray": False,
    "display.shading.use_dof": False,
    "display.shading.show_backface_culling": False,
    "render.use_motion_blur": False,
}


# ---------------------------------------------------------------------------
# Settings overrides with restore

def _tsr_plain(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    try:
        return tuple(value)
    except TypeError:
        return value


def _tsr_id(kind: str, name: str):
    import bpy  # pylint: disable=import-error
    if kind == "preferences":
        return bpy.context.preferences
    if kind == "view_layers":
        scene_name, layer_name = name.split("\x1f", 1)
        return bpy.data.scenes[scene_name].view_layers[layer_name]
    return getattr(bpy.data, kind)[name]


def _tsr_resolve(kind: str, name: str, path: str):
    owner = _tsr_id(kind, name)
    parts = path.split(".")
    for part in parts[:-1]:
        owner = getattr(owner, part)
    return owner, parts[-1]


_TSR_KINDS = (("Object", "objects"), ("World", "worlds"), ("Scene", "scenes"),
              ("Collection", "collections"), ("Camera", "cameras"), ("Light", "lights"),
              ("Action", "actions"))


class _TsrOverrides:
    """Records [kind, name, path, old_value] before every change; `restore()` undoes them.

    Entries live in a plain list stored in `_ts_state()`, so a later tool call (a fresh exec
    namespace) can restore them. Datablocks are referenced by name, never by Python object.
    Special entries ["__call__", fn_name, args] call a template function on restore.
    """

    def __init__(self, entries: list) -> None:
        self.entries = entries

    def set(self, id_block, path: str, value) -> None:
        import bpy  # pylint: disable=import-error
        if isinstance(id_block, bpy.types.ViewLayer):
            kind, name = "view_layers", id_block.id_data.name + "\x1f" + id_block.name
        elif isinstance(id_block, bpy.types.Preferences):
            kind, name = "preferences", ""
        else:
            kind = next(k for t, k in _TSR_KINDS if isinstance(id_block, getattr(bpy.types, t)))
            name = id_block.name
        owner, attr = _tsr_resolve(kind, name, path)
        old = _tsr_plain(getattr(owner, attr))
        self.entries.append([kind, name, path, old])
        if _tsr_plain(value) != old:
            setattr(owner, attr, value)

    def scene(self, scene, settings: dict) -> None:
        """Scene-relative settings; "view_layer.*" / "world.*" go to the active layer / world."""
        import bpy  # pylint: disable=import-error
        for path, value in settings.items():
            if path.startswith("view_layer."):
                self.set(bpy.context.view_layer, path[len("view_layer."):], value)
            elif path.startswith("world."):
                self.set(scene.world, path[len("world."):], value)
            else:
                self.set(scene, path, value)

    def call_on_restore(self, fn_name: str, *args) -> None:
        self.entries.append(["__call__", fn_name, list(args), None])

    def restore(self) -> list:
        """Undo everything (newest first); retry failures once (e.g. media_type before
        file_format). Returns error strings, never raises."""
        failed = []
        for entry in reversed(self.entries):
            if not _tsr_restore_entry(entry):
                failed.append(entry)
        errors = []
        for entry in failed:
            if not _tsr_restore_entry(entry):
                errors.append("restore failed: {}".format(entry[:3]))
        self.entries.clear()
        return errors


def _tsr_restore_entry(entry) -> bool:
    kind, name, path, value = entry
    try:
        if kind == "__call__":
            globals()[name](*path)
        else:
            owner, attr = _tsr_resolve(kind, name, path)
            setattr(owner, attr, value)
        return True
    except Exception:  # noqa: BLE001  (datablock deleted meanwhile, order dependency, ...)
        return False


def _tsr_remove_ids(kind: str, names: list) -> None:
    import bpy  # pylint: disable=import-error
    coll = getattr(bpy.data, kind)
    for n in names:
        block = coll.get(n)
        if block is not None:
            coll.remove(block)


def _tsr_set_format(ov: "_TsrOverrides", scene, fmt: str) -> None:
    for attr, value in _TSR_FORMATS[fmt]:
        ov.set(scene, "render.image_settings." + attr, value)


# ---------------------------------------------------------------------------
# Scene pieces used by render passes

def _tsr_groups(scene) -> dict:
    """Scene objects split into {"car", "set", "rig", "other"} lists (by TS collection)."""
    import bpy  # pylint: disable=import-error
    groups = {"car": [], "set": [], "rig": [], "other": []}
    member = {}
    for key, cname in (("car", TS_CAR), ("set", TS_SET), ("rig", TS_RIG)):
        coll = bpy.data.collections.get(cname)
        for ob in (coll.all_objects if coll else ()):
            member.setdefault(ob.name, key)
    for ob in scene.objects:
        groups[member.get(ob.name, "other")].append(ob)
    return groups


def _tsr_show_only(ov: "_TsrOverrides", scene, visible) -> None:
    """Render exactly the objects in `visible` (plus lights in TS_RIG); hide everything else.

    Objects hidden before the tool ran stay hidden (`hide_render` is the user's choice).
    """
    import bpy  # pylint: disable=import-error
    keep = {ob.name for ob in visible}
    for cname in (TS_CAR, TS_SET):
        coll = bpy.data.collections.get(cname)
        if coll is not None and coll.hide_render and any(o.name in keep for o in coll.all_objects):
            ov.set(coll, "hide_render", False)
    rig = {o.name for o in _tsr_groups(scene)["rig"]}
    for ob in scene.objects:
        if ob.type == "CAMERA" or ob.hide_render:
            continue
        if ob.type == "LIGHT" and ob.name in rig:
            continue
        if ob.name not in keep:
            ov.set(ob, "hide_render", True)


def _tsr_visible(objs) -> list:
    return [o for o in objs if not o.hide_render]


def _tsr_ensure_world(ov: "_TsrOverrides", scene) -> None:
    import bpy  # pylint: disable=import-error
    if scene.world is None:
        world = bpy.data.worlds.new("TS_TMP_WORLD")
        scene.world = world
        ov.call_on_restore("_tsr_drop_tmp_world", scene.name, world.name)


def _tsr_drop_tmp_world(scene_name: str, world_name: str) -> None:
    import bpy  # pylint: disable=import-error
    scene = bpy.data.scenes.get(scene_name)
    if scene is not None and scene.world is not None and scene.world.name == world_name:
        scene.world = None
    _tsr_remove_ids("worlds", [world_name])


def _tsr_only_passes(ov: "_TsrOverrides", keep) -> None:
    """Enable exactly the view-layer passes in `keep` (attribute names such as "use_pass_z",
    "cycles.use_pass_shadow_catcher") on the active view layer; everything else off."""
    import bpy  # pylint: disable=import-error
    vl = bpy.context.view_layer
    for prefix, owner in (("", vl), ("cycles.", vl.cycles)):
        for attr in dir(owner):
            if attr.startswith("use_pass_") and isinstance(getattr(owner, attr), bool):
                ov.set(vl, prefix + attr, (prefix + attr) in keep)
    ov.set(vl, "cycles.denoising_store_passes", False)


def _tsr_black_material(ov: "_TsrOverrides"):
    """A temporary pure-black diffuse material (removed on restore), e.g. for a view-layer
    override that makes a pass's Combined layer all zeros."""
    import bpy  # pylint: disable=import-error
    mat = bpy.data.materials.new("TS_TMP_BLACK")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    out = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfDiffuse")
    bsdf.inputs["Color"].default_value = (0.0, 0.0, 0.0, 1.0)
    mat.node_tree.links.new(bsdf.outputs[0], out.inputs["Surface"])
    ov.call_on_restore("_tsr_remove_ids", "materials", [mat.name])
    return mat


def _tsr_object_colours(ov: "_TsrOverrides", objs, colour_of) -> None:
    """Workbench OBJECT colours: `colour_of(ob)` -> grey value (linear)."""
    for ob in objs:
        v = float(colour_of(ob))
        ov.set(ob, "color", (v, v, v, 1.0))


def _tsr_device_label(scene) -> str:
    import bpy  # pylint: disable=import-error
    if scene.render.engine != "CYCLES":
        return scene.render.engine
    if scene.cycles.device != "GPU":
        return "CPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    names = [d.name for d in prefs.devices if d.use and d.type == prefs.compute_device_type]
    return "{}:{}".format(prefs.compute_device_type, ",".join(names)) if names else "CPU (no GPU enabled)"


def _tsr_spin_wheels(ov: "_TsrOverrides", scene, pivots, omega_rad_s: float, sign: float) -> None:
    """Spin wheel pivots about their local X axis with linear keyframes around the current
    frame: angle(t) = rest + sign * omega * (t - frame) / fps. The pose at the current frame
    is the rest pose, so the shutter centre matches the guides."""
    from bpy_extras import anim_utils  # pylint: disable=import-error
    fps = scene.render.fps / (scene.render.fps_base or 1.0)
    f = scene.frame_current
    step = sign * omega_rad_s / fps
    created = []
    for piv in pivots:
        ov.set(piv, "rotation_mode", "XYZ")
        rest = tuple(piv.rotation_euler)
        ov.set(piv, "rotation_euler", rest)
        had_action = piv.animation_data is not None and piv.animation_data.action is not None
        if had_action:
            ov.set(piv, "animation_data.action", None)
        for df in (-2, 0, 2):
            piv.rotation_euler[0] = rest[0] + step * df
            piv.keyframe_insert("rotation_euler", index=0, frame=f + df)
        piv.rotation_euler = rest
        anim = piv.animation_data
        bag = anim_utils.action_get_channelbag_for_slot(anim.action, anim.action_slot)
        for fc in bag.fcurves:
            fc.extrapolation = "LINEAR"
            for kp in fc.keyframe_points:
                kp.interpolation = "LINEAR"
        created.append([piv.name, anim.action.name, had_action])
    ov.call_on_restore("_tsr_unspin", created)


def _tsr_unspin(created) -> None:
    import bpy  # pylint: disable=import-error
    for ob_name, action_name, had_action in created:
        ob = bpy.data.objects.get(ob_name)
        if ob is not None and ob.animation_data is not None:
            ob.animation_data.action = None
            if not had_action:
                ob.animation_data_clear()
        _tsr_remove_ids("actions", [action_name])


# ---------------------------------------------------------------------------
# Files

def _tsr_exr_channels(path: str) -> list:
    """Channel names of an OpenEXR file (single- or multi-part), from its headers."""
    import struct
    with open(path, "rb") as fh:
        data = fh.read(1 << 20)
    if data[:4] != b"\x76\x2f\x31\x01":
        return []
    multipart = bool(struct.unpack("<I", data[4:8])[0] & 0x1000)
    pos, names = 8, []

    def cstr(p):
        end = data.index(b"\0", p)
        return data[p:end].decode("latin-1"), end + 1

    while True:
        attr, pos = cstr(pos)
        if attr == "":
            if not multipart or data[pos:pos + 1] == b"\0":
                break
            continue
        atype, pos = cstr(pos)
        size = struct.unpack("<i", data[pos:pos + 4])[0]
        pos += 4
        if atype == "chlist":
            p = pos
            while data[p:p + 1] != b"\0":
                ch, p = cstr(p)
                names.append(ch)
                p += 16
        pos += size
    return names


def _tsr_put(url: str, path: str, content_type: str) -> None:
    """PUT a file to a presigned URL. `localhost` is dialled as 127.0.0.1 (same Host header,
    so the signature holds): on Windows `localhost` tries ::1 first and costs ~1 s."""
    from urllib.parse import urlsplit, urlunsplit
    import requests
    parts = urlsplit(url)
    headers = {"Content-Type": content_type}
    if parts.hostname == "localhost":
        headers["Host"] = parts.netloc
        url = urlunsplit(parts._replace(netloc=parts.netloc.replace("localhost", "127.0.0.1", 1)))
    with open(path, "rb") as fh:
        r = requests.put(url, data=fh, headers=headers, timeout=300)
    r.raise_for_status()


def _tsr_upload_async(job: dict, uploads, name: str, path: str) -> None:
    """Upload in a background thread (network I/O releases the GIL, so Blender's main thread
    stays free); the chain waits for `job["uploads"]` before returning its result."""
    import threading
    match = [u for u in uploads if u[0] == name]
    if not match:
        raise KeyError("no presigned upload for output {!r}".format(name))
    _n, url, content_type = match[0]
    rec = {"name": name, "bytes": os.path.getsize(path), "done": False, "error": None}

    def run() -> None:
        try:
            _tsr_put(url, path, content_type)
        except Exception as ex:  # noqa: BLE001
            rec["error"] = "{}: {}".format(name, ex)
        rec["done"] = True

    threading.Thread(target=run, name="tsr_upload_" + name, daemon=True).start()
    job["uploads"].append(rec)


# ---------------------------------------------------------------------------
# The chain

def _tsr_on_render_end(*_args) -> None:
    import bpy  # pylint: disable=import-error
    bpy.app.driver_namespace.setdefault("trueshot", {})["tsr_render_end_t"] = time.time()


def _tsr_handlers(install: bool) -> None:
    import bpy  # pylint: disable=import-error
    for hlist in (bpy.app.handlers.render_complete, bpy.app.handlers.render_cancel):
        for h in [h for h in hlist if getattr(h, "__name__", "") == "_tsr_on_render_end"]:
            hlist.remove(h)
        if install:
            hlist.append(_tsr_on_render_end)


def _tsr_restore_job(job: dict) -> list:
    errors = _TsrOverrides(job["step_saved"]).restore()
    errors += _TsrOverrides(job["saved"]).restore()
    _tsr_handlers(False)
    return errors


def _tsr_abort(job: dict, message: str):
    errors = _tsr_restore_job(job)
    _ts_state().pop("render_job", None)
    return _ts_result("error", message=message, restore_errors=errors)


def _tsr_start_pass(job: dict, scene, passes) -> None:
    import bpy  # pylint: disable=import-error
    spec = passes[job["step"]]
    ov = _TsrOverrides(job["step_saved"])
    ov.scene(scene, spec.get("settings", {}))
    if spec.get("setup"):
        spec["setup"](ov, scene, job)
    _tsr_set_format(ov, scene, spec["format"])
    path = os.path.join(_ts_tmp_dir(job["tool"]), spec["name"] + _TSR_EXT[spec["format"]])
    if os.path.exists(path):
        os.remove(path)
    ov.set(scene, "render.filepath", path)
    job["pass_path"] = path
    job["pass_t0"] = time.time()
    _ts_state().pop("tsr_render_end_t", None)
    bpy.ops.render.render("INVOKE_DEFAULT", write_still=True)


def _tsr_finish_pass(job: dict, scene, passes, uploads) -> None:
    spec = passes[job["step"]]
    path = job["pass_path"]
    end_t = _ts_state().get("tsr_render_end_t") or time.time()
    job["pass_s"][spec["name"]] = round(end_t - job["pass_t0"], 2)
    files = {}
    if spec.get("upload"):
        files[spec["upload"]] = path
    if spec.get("after"):
        files.update(spec["after"](_TsrOverrides(job["step_saved"]), scene, job, path) or {})
    job["errors"] += _TsrOverrides(job["step_saved"]).restore()
    for name, fpath in files.items():
        _tsr_upload_async(job, uploads, name, fpath)


def _tsr_result(job: dict):
    """The final result once every upload has finished (None while uploads are running)."""
    if not all(u["done"] for u in job["uploads"]):
        return None
    _ts_state().pop("render_job", None)
    failed = [u["error"] for u in job["uploads"] if u["error"]]
    if failed:
        return _ts_result("error", message="Upload failed: " + "; ".join(failed))
    result = _ts_result("ok", **job["out"])
    result.update(uploaded={u["name"]: u["bytes"] for u in job["uploads"]}, pass_s=job["pass_s"],
                  call_s=job["calls"], max_call_s=max(s for _l, s in job["calls"]),
                  total_s=round(time.time() - job["t0"], 2))
    if job["errors"]:
        result["restore_errors"] = job["errors"]
    return result


def _tsr_chain(params, tool: str, passes, prepare=None, finish=None):
    """Run `passes` one render at a time (see the header). Returns an `_ts_result`.

    prepare(ov, scene, job): tool-wide overrides and analytic data (store plain values in
        `job["info"]`); may return {upload_name: local_path} to upload at the end.
    finish(scene, job) -> dict of result fields (after everything is restored).
    Results: "started" / "running" while rendering or uploading; then "ok" with the `finish`
    fields plus `uploaded` {name: bytes}, `pass_s` (render time per pass), `call_s` (main-thread
    time of each non-trivial call), `max_call_s`, `total_s`; or "busy" / "error".
    """
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
            _tsr_restore_job(job)  # abandoned by a dead client: restore its settings
        if scene.camera is None:
            return _ts_result("error", message="The scene has no camera (run frame_shot first).")
        job = {"tool": tool, "scene": scene.name, "step": 0, "saved": [], "step_saved": [],
               "info": {}, "extra_files": {}, "uploads": [], "pass_s": {}, "errors": [],
               "phase": "render", "out": {}, "t0": time.time(), "last_poll": time.time(), "calls": []}
        state["render_job"] = job
        try:
            ov = _TsrOverrides(job["saved"])
            ov.scene(scene, {
                "render.resolution_percentage": 100,
                "render.use_border": False,
                "render.use_single_layer": True,
                "render.use_compositing": False,
                "render.use_sequencer": False,
                "render.use_stamp": False,
            })
            if prepare is not None:
                job["extra_files"] = prepare(ov, scene, job) or {}
            _tsr_handlers(True)
            _tsr_start_pass(job, scene, passes)
        except Exception:  # noqa: BLE001
            return _tsr_abort(job, traceback.format_exc()[-3000:])
        lap("start")
        return _ts_result("started", tool=tool, passes=[p["name"] for p in passes])

    # action == "status"
    if job is None or job["tool"] != tool:
        return _ts_result("error", message="No {} job is running.".format(tool))
    job["last_poll"] = time.time()
    if job["phase"] == "uploading":
        return _tsr_result(job) or _ts_result("running", tool=tool, pass_name="upload",
                                               elapsed_s=round(time.time() - job["t0"], 1))
    if bpy.app.is_job_running("RENDER"):
        return _ts_result("running", tool=tool, pass_name=passes[job["step"]]["name"],
                          elapsed_s=round(time.time() - job["t0"], 1))
    try:
        if not os.path.exists(job["pass_path"]):
            return _tsr_abort(job, "Render pass {!r} produced no file (cancelled?).".format(
                passes[job["step"]]["name"]))
        _tsr_finish_pass(job, scene, passes, params.uploads)
        job["step"] += 1
        if job["step"] < len(passes):
            _tsr_start_pass(job, scene, passes)
            lap("status:" + passes[job["step"] - 1]["name"] + "->next")
            return _ts_result("running", tool=tool, pass_name=passes[job["step"]]["name"],
                              elapsed_s=round(time.time() - job["t0"], 1))
        for name, fpath in job["extra_files"].items():
            _tsr_upload_async(job, params.uploads, name, fpath)
        job["errors"] += _tsr_restore_job(job)
        job["out"] = finish(scene, job) if finish is not None else {}
        job["phase"] = "uploading"
    except Exception:  # noqa: BLE001
        return _tsr_abort(job, traceback.format_exc()[-3000:])
    lap("status:finish")
    return _tsr_result(job) or _ts_result("running", tool=tool, pass_name="upload",
                                           elapsed_s=round(time.time() - job["t0"], 1))
