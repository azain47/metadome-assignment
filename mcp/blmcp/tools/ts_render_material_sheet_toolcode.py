"""Tool-code: per-material highlight contact sheets of the car (Workbench), for a material audit.

Four Workbench animation renders (one frame per view, one temporary keyframed camera):
  look      the car's current material colours and image textures (studio light) on white;
  id_solid  every car material drawn in a unique flat colour (Raw view, no AA): which material
            owns each pixel, see-through materials (alpha/transmission/glass) opaque;
  id_open   the same with see-through materials hidden, so parts behind lenses and glass count
            (skipped when the car has no see-through material);
  shade     the car in neutral grey (studio light, cavity, outlines) on white.
Each material's pixels (from id_open, or id_solid for see-through materials) are painted orange
over the grey render, so N materials cost four renders, not N x views. Materials are ranked by
visible area (mean share of the car's pixels over the views), those under `min_area_pct` are
skipped, and the rows (one per material under a label "#index name area%": its views side by
side, then a close-up of its largest area in the `look` render with its outline in orange) are
tiled into one PNG per `rows_per_sheet` materials, plus `materials.json` (each material's
current Principled values and its row and cell rectangles). Every scene and material change
goes through `_TsrOverrides` and is restored on success, error and abandoned jobs.
"""

__all__ = ("Params", "main", "VIEWS")

from typing import NamedTuple

# Camera direction per view name, frame_shot convention: (azimuth, elevation) in degrees,
# azimuth 0 = looking at the car's front, 90 = its left side, 180 = behind, 270 = right side.
VIEWS = {
    "front": (0.0, 8.0),
    "front34": (40.0, 12.0),
    "front34_r": (320.0, 12.0),
    "side": (90.0, 4.0),
    "side_r": (270.0, 4.0),
    "rear34": (220.0, 12.0),
    "rear34_l": (140.0, 12.0),
    "rear": (180.0, 8.0),
    "top": (0.0, 65.0),
}


class Params(NamedTuple):
    action: str  # "start" | "status"
    uploads: tuple = ()  # (("sheet_<k>", url, "image/png"), ..., ("index", url, "application/json"))
    views: tuple = ("front34", "rear34", "side")
    max_materials: int = 40
    cell_px: int = 320
    min_area_pct: float = 0.05
    rows_per_sheet: int = 12


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_render.py
# @include_end

TOOL_NAME = "render_material_sheet"
CAM_NAME = "TS_TMP_MATSHEET_CAM"
SENSOR_MM = 36.0
LENS_MM = 50.0
FILL_W, FILL_H = 0.9, 0.84  # car bbox fill of a cell (fractions of width / height)
GREY = (0.62, 0.62, 0.62)
HIGHLIGHT = (1.0, 0.36, 0.0)  # sRGB orange, scaled by the grey render's shading
MARKER = (0.85, 0.0, 0.85)  # box drawn around highlights too small to spot
MARKER_BELOW = 0.02  # box a highlight smaller than this share of the car's pixels in the view
MARKER_BLOCK = 16  # px; highlight pieces closer than this share one box
LABEL_SCALE = 2
GAP = 3
ROW_GAP = 6
SEE_THROUGH_ALPHA = 0.6
SEE_THROUGH_TRANSMISSION = 0.5

# 8x14 bitmap font for ASCII 33..126 (DejaVu Sans Mono Bold 13 px), one byte per row.
_FONT_W, _FONT_H = 8, 14
_FONT_HEX = (
    "0000181818181818001818000000000064646464000000000000000000000012167f342cfe68580000000018183c7c78"
    "3c1e1e7e3c181800000070d0d076184e0b0b0e0000000000386470307bdfce6e7f000000000018181818000000000000"
    "00000c181818303030301818180c000030101818181818181818103000000000185e3c3c5e1800000000000000000018"
    "1818fefe18181800000000000000000000000018181830000000000000003c3c00000000000000000000000000000018"
    "18000000000006060c0c081810302060400000003c6e667e7e6666663c000000000078181818181818187e0000000000"
    "3c4e060e0c1830607e00000000003c46063c0e0606467c00000000000c1c3c6c4c7e0c0c0c00000000007c60607c4e06"
    "064e3c00000000003c72607c666666663c00000000007e060c0c1c1818303000000000003c66663c666666663c000000"
    "00003c6e66666e3e064c380000000000000018180000001818000000000000001818000000181818300000000000021e"
    "78e0781e020000000000000000fefe00fefe0000000000000000c0701e061e78c000000000003c46060c181800181800"
    "000000003c625ed2f2b2f2d25e623e000000183c3c3c647e6646c300000000007c6666667c6666667e00000000001c32"
    "6060606060321c00000000007c6e66666666666e7c00000000007e6060607e6060607e00000000007e6060607e606060"
    "6000000000003c7260606e6666763e0000000000666666667e6666666600000000007e181818181818187e0000000000"
    "3e0e0e0e0e0e0c4c7c0000000000666e7c78786c6c6667000000000060606060606060607e0000000000e6e6fefedada"
    "c2c2c20000000000666676767e6e6e6e6600000000003c666666e66666663c00000000007c666666667c606060000000"
    "00003c666666e66666663c0e040000007c6e66666e7c6c666700000000003c6660703c0e06463c0000000000fe181818"
    "1818181818000000000066666666666666663c0000000000c66666666c3c3c3c380000000000c3c3dbdbfe7e7e666600"
    "00000000e6663c3c183c3c66e60000000000c7666e3c381818181800000000007e060e1c183870607e0000001c101010"
    "101010101010101c00000000406020301018080c0c06060038181818181818181818183800000000183c6e4600000000"
    "0000000000000000000000000000000000ff0030180000000000000000000000000000003c46067e66667e0000006060"
    "60607c66666666667c000000000000001c32606060323c000000060606063e6e66e6666e3e000000000000003c66667e"
    "60623c0000000e1818187e181818181818000000000000003e6e6666666e3e064e3c606060607c666666666666000000"
    "181800007818181818187e000000181800003818181818181818187860606060666c78786c6666000000f03030303030"
    "303030181e00000000000000fedadadadadada000000000000007c666666666666000000000000003c66666666663c00"
    "0000000000007c66666666667c606060000000003e6e66e6666e3e060606000000003e38303030303000000000000000"
    "3c64703c0e463c000000000030307e30303030181e0000000000000066666666666e3e000000000000006666662c3c3c"
    "3800000000000000c3c3da5a7e7e6600000000000000663c3c183c7e6600000000000000e666663c3c3c181838700000"
    "00007e0e1c1830707e0000000e181818181870181818180e000018181818181818181818181818007018181818180e18"
    "181818700000000000000000700e000000000000"
)

_PASS_COMMON = {
    "render.engine": "BLENDER_WORKBENCH",
    "render.film_transparent": False,
    "render.use_motion_blur": False,
    "display.shading.show_xray": False,
    "display.shading.use_dof": False,
    "display.shading.show_backface_culling": False,
    "display.shading.show_shadows": False,
}
# ID passes: exact flat material colours (Raw view, no AA, no dither; TSR_RAW_VIEW spelled out
# because this module is also imported outside Blender, where the template is not included).
_ID_SETTINGS = dict(_PASS_COMMON, **{
    "display.render_aa": "OFF",
    "display.shading.light": "FLAT",
    "display.shading.color_type": "MATERIAL",
    "display.shading.show_cavity": False,
    "display.shading.show_object_outline": False,
    "display.shading.show_specular_highlight": False,
    "view_settings.view_transform": "Raw",
    "view_settings.look": "None",
    "view_settings.exposure": 0.0,
    "view_settings.gamma": 1.0,
    "view_settings.use_curve_mapping": False,
    "render.dither_intensity": 0.0,
})
_SHADE_SETTINGS = dict(_PASS_COMMON, **{
    "display.render_aa": "8",
    "display.shading.light": "STUDIO",
    "display.shading.color_type": "SINGLE",
    "display.shading.single_color": GREY,
    "display.shading.show_cavity": True,
    "display.shading.cavity_type": "SCREEN",
    "display.shading.show_object_outline": True,
    "display.shading.object_outline_color": (0.0, 0.0, 0.0),
    "display.shading.show_specular_highlight": True,
    "view_settings.view_transform": "Standard",
    "view_settings.look": "None",
    "view_settings.exposure": 0.0,
    "view_settings.gamma": 1.0,
    "view_settings.use_curve_mapping": False,
})
# The car's current look (material colours and image textures), for the close-up cell.
_LOOK_SETTINGS = dict(_SHADE_SETTINGS, **{
    "display.shading.color_type": "TEXTURE",
    "display.shading.show_cavity": False,
})
_PASS_SETTINGS = {"look": (_LOOK_SETTINGS, (1.0, 1.0, 1.0)), "id_solid": (_ID_SETTINGS, (0.0, 0.0, 0.0)),
                  "id_open": (_ID_SETTINGS, (0.0, 0.0, 0.0)), "shade": (_SHADE_SETTINGS, (1.0, 1.0, 1.0))}


# ---------------------------------------------------------------------------
# Materials

def _id_rgb(k: int) -> tuple:
    """Material k (1-based) -> exact 8-bit RGB triple of its flat ID colour."""
    return (16 * (k & 15) + 8, 16 * ((k >> 4) & 15) + 8, 72 + 16 * (k >> 8))


def _decode_ids(rgb):
    """HxWx3 uint8 ID render -> HxW int32 material number (0 = background / unknown)."""
    import numpy as np
    r, g, b = (rgb[..., i].astype(np.int32) for i in range(3))
    ok = (r % 16 == 8) & (g % 16 == 8) & (b % 16 == 8) & (b >= 72)
    k = (r - 8) // 16 + 16 * ((g - 8) // 16) + 256 * ((b - 72) // 16)
    return np.where(ok, k, 0)


def _to_srgb_hex(rgb) -> str:
    out = []
    for c in list(rgb)[:3]:
        c = min(1.0, max(0.0, float(c)))
        s = c * 12.92 if c <= 0.0031308 else 1.055 * c ** (1.0 / 2.4) - 0.055
        out.append(int(round(s * 255)))
    return "#{:02X}{:02X}{:02X}".format(*out)


def _socket(bsdf, name):
    sock = bsdf.inputs.get(name) if bsdf is not None else None
    if sock is None:
        return None, False
    return sock.default_value, sock.is_linked


def _describe(mat, glass_names) -> dict:
    """Current look of a material (Principled values; diffuse colour when there is none)."""
    nt = getattr(mat, "node_tree", None)
    bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None) if nt else None
    textures = sorted({n.image.name for n in nt.nodes if n.type == "TEX_IMAGE" and n.image}) if nt else []
    base, base_linked = _socket(bsdf, "Base Color")
    d = {
        "base_hex": _to_srgb_hex(base if base is not None and not base_linked else mat.diffuse_color),
        "base_textured": bool(base_linked),
        "texture": bool(textures),
        "textures": textures[:4],
        "principled": bsdf is not None,
    }
    for key, sock in (("metallic", "Metallic"), ("roughness", "Roughness"), ("alpha", "Alpha"),
                      ("transmission", "Transmission Weight"), ("coat", "Coat Weight"),
                      ("emission_strength", "Emission Strength")):
        v, linked = _socket(bsdf, sock)
        d[key] = None if v is None else ("texture" if linked else round(float(v), 3))
    em, em_linked = _socket(bsdf, "Emission Color")
    d["emission_hex"] = None if em is None else ("texture" if em_linked else _to_srgb_hex(em))
    if bsdf is None:
        d["alpha"] = round(float(mat.diffuse_color[3]), 3)
    alpha = d["alpha"] if isinstance(d["alpha"], float) else 1.0
    trans = d["transmission"] if isinstance(d["transmission"], float) else 0.0
    d["see_through"] = bool(alpha < SEE_THROUGH_ALPHA or trans > SEE_THROUGH_TRANSMISSION
                            or mat.name in glass_names)
    return d


def _mat_set(ov, mat, attr: str, value) -> None:
    """Like `_TsrOverrides.set` for a material (the recorder's kind table has no materials)."""
    ov.entries.append(["materials", mat.name, attr, _tsr_plain(getattr(mat, attr))])
    setattr(mat, attr, value)


def _car_materials(root) -> list:
    mats = {}
    for ob in root.children_recursive:
        if ob.type != "MESH" or ob.hide_render:
            continue
        for slot in ob.material_slots:
            if slot.material is not None:
                mats.setdefault(slot.material.name, slot.material)
    return [mats[n] for n in sorted(mats)]


# ---------------------------------------------------------------------------
# Camera

def _poses(root, views, width, height) -> list:
    """Camera (location, quaternion, shift_x, shift_y) per view framing the car's bounding box."""
    import math
    from mathutils import Vector  # pylint: disable=import-error
    b = [float(v) for v in root["ts_bbox_local"]]
    M = root.matrix_world
    corners = [M @ Vector((x, y, z)) for x in (b[0], b[3]) for y in (b[1], b[4]) for z in (b[2], b[5])]
    centre = sum(corners, Vector()) / 8.0
    quat = M.to_quaternion()
    k = LENS_MM / SENSOR_MM
    aspect = width / height
    size = max((c - centre).length for c in corners)
    out = []
    for name in views:
        az, el = (math.radians(a) for a in VIEWS[name])
        dir_h = quat @ Vector((-math.sin(az), math.cos(az), 0.0))
        dir_h.z = 0.0
        dir_h.normalize()
        view_dir = dir_h * math.cos(el) + Vector((0.0, 0.0, math.sin(el)))
        rot = view_dir.to_track_quat("Z", "Y")
        rinv = rot.inverted()

        def project(dist):
            loc = centre + view_dir * dist
            pts = [rinv @ (c - loc) for c in corners]
            return loc, [k * p.x / -p.z for p in pts], [k * p.y / -p.z for p in pts]

        d = size * 2.0 * k
        for _ in range(8):
            _loc, xs, ys = project(d)
            d *= max((max(xs) - min(xs)) / FILL_W, (max(ys) - min(ys)) * aspect / FILL_H) ** 0.9
        loc, xs, ys = project(d)
        out.append({"view": name, "location": tuple(loc), "rotation": tuple(rot),
                    "shift_x": (max(xs) + min(xs)) / 2.0, "shift_y": (max(ys) + min(ys)) / 2.0})
    return out


def _make_camera(ov, scene, poses, frame0):
    import bpy  # pylint: disable=import-error
    for kind in ("objects", "cameras"):
        _tsr_remove_ids(kind, [CAM_NAME])  # leftovers of a crashed run
    data = bpy.data.cameras.new(CAM_NAME)
    cam = bpy.data.objects.new(CAM_NAME, data)
    cam["ts_role"] = "tmp_material_sheet_camera"
    scene.collection.objects.link(cam)
    ov.call_on_restore("_tsr_remove_ids", "cameras", [data.name])
    ov.call_on_restore("_tsr_remove_ids", "objects", [cam.name])
    data.type = "PERSP"
    data.lens = LENS_MM
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


# ---------------------------------------------------------------------------
# Passes

def _frame_path(job, pass_name: str, i: int) -> str:
    return os.path.join(_ts_tmp_dir(TOOL_NAME), "{}_{:04d}.png".format(pass_name, job["info"]["frame0"] + i))


def _start_pass(job, scene) -> None:
    import bpy  # pylint: disable=import-error
    info = job["info"]
    name = info["passes"][job["step"]]
    settings, background = _PASS_SETTINGS[name]
    ov = _TsrOverrides(job["step_saved"])
    ov.scene(scene, settings)
    ov.set(scene.world, "color", background)
    if name.startswith("id_"):
        for k, mname in enumerate(info["materials"], start=1):
            _mat_set(ov, bpy.data.materials[mname], "diffuse_color", tuple(c / 255.0 for c in _id_rgb(k)) + (1.0,))
    if name == "id_open":
        for mname in info["see_through"]:
            mat = bpy.data.materials[mname]
            _mat_set(ov, mat, "diffuse_color", tuple(mat.diffuse_color[:3]) + (0.0,))
            _mat_set(ov, mat, "surface_render_method", "BLENDED")
    ov.set(scene, "render.filepath", os.path.join(_ts_tmp_dir(TOOL_NAME), name + "_####"))
    job["pass_t0"] = time.time()
    _ts_state().pop("tsr_render_end_t", None)
    bpy.ops.render.render("INVOKE_DEFAULT", animation=True)


def _start(params: Params, scene, job) -> None:
    import bpy  # pylint: disable=import-error
    root = _ts_car_root()
    if root is None:
        raise RuntimeError("No car loaded (call load_car first)")
    views = [str(v) for v in params.views]
    unknown = [v for v in views if v not in VIEWS]
    if unknown or not views:
        raise RuntimeError("views must be some of {}, got {}".format(sorted(VIEWS), views))
    width = max(96, int(params.cell_px))
    height = max(64, int(round(width * 2 / 3)))
    mats = _car_materials(root)
    if not mats:
        raise RuntimeError("The car has no materials")
    glass = {str(n) for n in root.get("ts_glass_materials", [])}
    described = {m.name: _describe(m, glass) for m in mats}

    ov = _TsrOverrides(job["saved"])
    _tsr_ensure_world(ov, scene)
    frame0 = int(scene.frame_current)
    out_dir = _ts_tmp_dir(TOOL_NAME)
    for fname in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, fname))
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
        "render.use_file_extension": True,
        "render.use_overwrite": True,
        "render.use_placeholder": False,
        "frame_start": frame0,
        "frame_end": frame0 + len(views) - 1,
        "frame_step": 1,
        "frame_current": frame0,
    })
    _tsr_set_format(ov, scene, "PNG_RGB8")
    _tsr_show_only(ov, scene, _tsr_groups(scene)["car"])
    _make_camera(ov, scene, _poses(root, views, width, height), frame0)
    # Every material opaque (ID passes then paint them in flat ID colours; grey ignores colours).
    for mat in mats:
        _mat_set(ov, mat, "diffuse_color", tuple(mat.diffuse_color[:3]) + (1.0,))
        if hasattr(mat, "surface_render_method"):
            _mat_set(ov, mat, "surface_render_method", "DITHERED")
    see_through = [m.name for m in mats if described[m.name]["see_through"]]
    job["info"].update(
        width=width, height=height, views=views, frame0=frame0, see_through=see_through,
        materials=[m.name for m in mats], described=described,
        passes=["look", "id_solid"] + (["id_open"] if see_through else []) + ["shade"],
        max_materials=int(params.max_materials), min_area_pct=float(params.min_area_pct),
        rows_per_sheet=max(1, int(params.rows_per_sheet)))
    _tsr_handlers(True)
    _start_pass(job, scene)


# ---------------------------------------------------------------------------
# Sheets

def _load_rgb(path: str):
    """PNG -> top-down HxWx3 float32 (stored values, 0..1)."""
    import numpy as np
    import bpy  # pylint: disable=import-error
    img = bpy.data.images.load(path, check_existing=False)
    try:
        buf = np.empty(img.size[0] * img.size[1] * 4, dtype=np.float32)
        img.pixels.foreach_get(buf)
        return buf.reshape(img.size[1], img.size[0], 4)[::-1, :, :3].copy()
    finally:
        bpy.data.images.remove(img)


def _font():
    import numpy as np
    data = bytes.fromhex("".join(_FONT_HEX))
    glyphs = {}
    for i in range(94):
        rows = data[i * _FONT_H:(i + 1) * _FONT_H]
        bits = np.array([[(r >> (_FONT_W - 1 - x)) & 1 for x in range(_FONT_W)] for r in rows], dtype=bool)
        glyphs[chr(33 + i)] = bits
    return glyphs


def _draw_text(px, glyphs, text: str, x0: int, y0: int, scale: int, colour) -> None:
    """Draw `text` into top-down HxWx3 `px` with its top-left at (x0, y0); clipped at the edges."""
    import numpy as np
    h, w = px.shape[:2]
    step = (_FONT_W + 1) * scale
    for i, ch in enumerate(text):
        if ch == " ":
            continue
        bits = glyphs.get(ch if 32 < ord(ch) < 127 else "?")
        if bits is None:
            continue
        big = np.kron(bits, np.ones((scale, scale), dtype=bool))
        x, y = x0 + i * step, y0
        if x >= w:
            break
        sub = big[:max(0, h - y), :max(0, w - x)]
        px[y:y + sub.shape[0], x:x + sub.shape[1]][sub] = colour


def _blobs(mask) -> list:
    """Bounding boxes (y0, y1, x0, x1) of the mask's pieces, grouped on a coarse grid."""
    import numpy as np
    b = MARKER_BLOCK
    h, w = mask.shape
    gh, gw = (h + b - 1) // b, (w + b - 1) // b
    grid = np.zeros((gh * b, gw * b), dtype=bool)
    grid[:h, :w] = mask
    occ = grid.reshape(gh, b, gw, b).any(axis=(1, 3))
    seen = np.zeros_like(occ)
    boxes = []
    for gy, gx in zip(*np.nonzero(occ)):
        if seen[gy, gx]:
            continue
        stack, cells = [(gy, gx)], []
        seen[gy, gx] = True
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for ny in range(max(0, cy - 1), min(gh, cy + 2)):
                for nx in range(max(0, cx - 1), min(gw, cx + 2)):
                    if occ[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        ys0 = min(c[0] for c in cells) * b
        ys1 = min(h, (max(c[0] for c in cells) + 1) * b)
        xs0 = min(c[1] for c in cells) * b
        xs1 = min(w, (max(c[1] for c in cells) + 1) * b)
        ys, xs = np.nonzero(mask[ys0:ys1, xs0:xs1])
        boxes.append((ys0 + ys.min(), ys0 + ys.max(), xs0 + xs.min(), xs0 + xs.max()))
    return boxes


def _cell(shade, mask, car_px: int):
    """Grey render with the mask painted orange (keeping the grey shading), plus magenta boxes
    around the pieces of a highlight too small to find at a glance. Returns (HxWx3, pixel count)."""
    import numpy as np
    out = shade.copy()
    n = int(mask.sum())
    if n:
        lum = shade[..., 0][mask][:, None]
        out[mask] = np.array(HIGHLIGHT, dtype=np.float32)[None, :] * (0.35 + 0.65 * lum)
        if n < MARKER_BELOW * car_px:
            m = np.array(MARKER, dtype=np.float32)
            hh, ww = mask.shape
            pad = 5
            for y0, y1, x0, x1 in _blobs(mask):
                y0, y1 = max(0, y0 - pad), min(hh - 1, y1 + pad)
                x0, x1 = max(0, x0 - pad), min(ww - 1, x1 + pad)
                for t in range(2):
                    out[max(0, y0 - t), x0:x1 + 1] = m
                    out[min(hh - 1, y1 + t), x0:x1 + 1] = m
                    out[y0:y1 + 1, max(0, x0 - t)] = m
                    out[y0:y1 + 1, min(ww - 1, x1 + t)] = m
    return out, n


def _resize(img, ow: int, oh: int, nearest: bool = False):
    """Bilinear (or nearest) resize of a top-down HxW[xC] array to (oh, ow)."""
    import numpy as np
    h, w = img.shape[:2]
    ys = np.clip((np.arange(oh) + 0.5) * h / oh - 0.5, 0, h - 1)
    xs = np.clip((np.arange(ow) + 0.5) * w / ow - 0.5, 0, w - 1)
    if nearest:
        return img[np.rint(ys).astype(int)][:, np.rint(xs).astype(int)]
    y0, x0 = np.floor(ys).astype(int), np.floor(xs).astype(int)
    y1, x1 = np.minimum(y0 + 1, h - 1), np.minimum(x0 + 1, w - 1)
    fy, fx = (ys - y0)[:, None, None], (xs - x0)[None, :, None]
    a, b = img[y0][:, x0], img[y0][:, x1]
    c, d = img[y1][:, x0], img[y1][:, x1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def _closeup(look, mask):
    """The material's current look (colours / textures) around its pixels, magnified to the
    cell size (3:2 crop, padded), with the material's outline drawn in orange."""
    import numpy as np
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cy, cx = (ys.min() + ys.max()) / 2.0, (xs.min() + xs.max()) / 2.0
    bw = max(xs.max() - xs.min() + 1, (ys.max() - ys.min() + 1) * w / h) * 1.4
    bw = min(float(w), max(w / 6.0, bw))
    bh = bw * h / w
    x0 = int(round(min(max(cx - bw / 2.0, 0.0), w - bw)))
    y0 = int(round(min(max(cy - bh / 2.0, 0.0), h - bh)))
    x1, y1 = min(w, x0 + int(round(bw))), min(h, y0 + int(round(bh)))
    out = _resize(look[y0:y1, x0:x1], w, h)
    m = _resize(mask[y0:y1, x0:x1], w, h, nearest=True)
    inner = m.copy()
    inner[1:] &= m[:-1]
    inner[:-1] &= m[1:]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    edge = m & ~inner
    thick = edge.copy()
    thick[1:] |= edge[:-1]
    thick[:, 1:] |= edge[:, :-1]
    out[thick] = HIGHLIGHT
    return out


def _save_png(path: str, px) -> None:
    import numpy as np
    import bpy  # pylint: disable=import-error
    h, w = px.shape[:2]
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[..., :3] = px
    img = bpy.data.images.new("TS_TMP_MATSHEET", w, h, alpha=False)
    try:
        img.pixels.foreach_set(rgba[::-1].ravel())
        img.filepath_raw = path
        img.file_format = "PNG"
        img.save()
    finally:
        bpy.data.images.remove(img)


def _compose(job) -> tuple:
    """Rank materials by visible area, build the sheets and the JSON index.
    Returns ([sheet paths], index path)."""
    import json
    import numpy as np
    info = job["info"]
    views, names = info["views"], info["materials"]
    nv = len(views)
    w, h = info["width"], info["height"]
    see_through = set(info["see_through"])
    shade = [_load_rgb(_frame_path(job, "shade", i)) for i in range(nv)]
    look = [_load_rgb(_frame_path(job, "look", i)) for i in range(nv)]

    def ids(pass_name):
        return [_decode_ids(np.rint(_load_rgb(_frame_path(job, pass_name, i)) * 255.0).astype(np.int32))
                for i in range(nv)]

    solid = ids("id_solid")
    opened = ids("id_open") if "id_open" in info["passes"] else solid
    car_px = [max(1, int((s > 0).sum())) for s in solid]
    n_id = len(names) + 1
    counts_solid = [np.bincount(s.ravel(), minlength=n_id)[:n_id] for s in solid]
    counts_open = [np.bincount(o.ravel(), minlength=n_id)[:n_id] for o in opened]

    entries = []
    for k, name in enumerate(names, start=1):
        counts = counts_solid if name in see_through else counts_open
        view_pct = {v: round(100.0 * int(counts[i][k]) / car_px[i], 3) for i, v in enumerate(views)}
        entries.append({"id": k, "name": name, "area_pct": round(sum(view_pct.values()) / nv, 3),
                        "view_pct": view_pct})
    entries.sort(key=lambda e: -e["area_pct"])
    shown = [e for e in entries if e["area_pct"] >= info["min_area_pct"]][:max(0, info["max_materials"])]
    shown_names = {e["name"] for e in shown}
    skipped = [{"name": e["name"], "area_pct": e["area_pct"],
                "reason": "below min_area_pct" if e["area_pct"] < info["min_area_pct"] else "over max_materials"}
               for e in entries if e["name"] not in shown_names]

    glyphs = _font()
    band = (_FONT_H + 4) * LABEL_SCALE
    row_h = band + h
    sheet_w = (nv + 1) * w + nv * GAP  # the views, then the close-up
    per = info["rows_per_sheet"]
    out_dir = _ts_tmp_dir(TOOL_NAME)
    sheets, materials = [], []
    for s0 in range(0, len(shown), per):
        chunk = shown[s0:s0 + per]
        sheet_h = len(chunk) * row_h + (len(chunk) - 1) * ROW_GAP
        px = np.full((sheet_h, sheet_w, 3), 0.25, dtype=np.float32)
        si = len(sheets)
        for r, e in enumerate(chunk):
            index = s0 + r
            y = r * (row_h + ROW_GAP)
            px[y:y + band] = 1.0
            label = "#{:02d} {}  {:.2f}%".format(index, e["name"], e["area_pct"])
            _draw_text(px, glyphs, label, 6, y + 2 * LABEL_SCALE, LABEL_SCALE, (0.05, 0.05, 0.05))
            cells = []
            src = solid if e["name"] in see_through else opened
            for i, v in enumerate(views):
                tile, n = _cell(shade[i][:h, :w], src[i][:h, :w] == e["id"], car_px[i])
                x = i * (w + GAP)
                px[y + band:y + row_h, x:x + w] = tile
                if n == 0:
                    _draw_text(px, glyphs, "not visible", x + 6, y + band + 4, 1, (0.3, 0.3, 0.3))
                cells.append({"view": v, "x": x, "y": y + band, "w": w, "h": h, "pixels": n})
            best = max(range(nv), key=lambda i: cells[i]["pixels"])
            x = nv * (w + GAP)
            if cells[best]["pixels"]:
                px[y + band:y + row_h, x:x + w] = _closeup(look[best][:h, :w], src[best][:h, :w] == e["id"])
            px[y + band:y + band + _FONT_H + 2, x:x + 8 + 12 * (_FONT_W + 1)] = 1.0
            _draw_text(px, glyphs, "current look", x + 4, y + band + 1, 1, (0.05, 0.05, 0.05))
            cells.append({"view": "closeup", "source_view": views[best], "x": x, "y": y + band, "w": w, "h": h,
                          "pixels": cells[best]["pixels"]})
            materials.append(dict(info["described"][e["name"]], index=index, name=e["name"],
                                  area_pct=e["area_pct"], view_pct=e["view_pct"], sheet=si,
                                  row={"x": 0, "y": y, "w": sheet_w, "h": row_h}, label_h=band, cells=cells))
        path = os.path.join(out_dir, "material_sheet_{}.png".format(si))
        _save_png(path, px)
        sheets.append({"index": si, "file": os.path.basename(path), "size": [sheet_w, sheet_h],
                       "rows": [s0, s0 + len(chunk) - 1], "path": path})
    index = {
        "views": [{"name": v, "azimuth_deg": VIEWS[v][0], "elevation_deg": VIEWS[v][1]} for v in views],
        "cell_size": [w, h], "label_h": band,
        "highlight": ("views: orange on grey, magenta box = small highlight; last cell (view 'closeup'): the "
                      "material's current colours/textures around its largest area, outlined in orange"),
        "car_px": dict(zip(views, car_px)),
        "sheets": [{k: v for k, v in s.items() if k != "path"} for s in sheets],
        "materials": materials, "skipped": skipped,
        "see_through": sorted(see_through),
    }
    index_path = os.path.join(out_dir, "materials.json")
    with open(index_path, "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    info["out"] = {"sheets": len(sheets), "materials": len(materials), "skipped": len(skipped),
                   "material_names": [m["name"] for m in materials]}
    return [s["path"] for s in sheets], index_path


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
        # Same job layout as `_tsr_chain`, so any render tool can restore an abandoned job.
        job = {"tool": TOOL_NAME, "scene": scene.name, "step": 0, "saved": [], "step_saved": [],
               "info": {}, "extra_files": {}, "uploads": [], "pass_s": {}, "errors": [],
               "phase": "render", "out": {}, "t0": time.time(), "last_poll": time.time(), "calls": []}
        state["render_job"] = job
        try:
            _start(params, scene, job)
        except Exception:  # noqa: BLE001
            return _tsr_abort(job, traceback.format_exc()[-3000:])
        lap("start")
        return _ts_result("started", tool=TOOL_NAME, materials=len(job["info"]["materials"]),
                          passes=job["info"]["passes"])

    if job is None or job["tool"] != TOOL_NAME:
        return _ts_result("error", message="No {} job is running.".format(TOOL_NAME))
    job["last_poll"] = time.time()
    if job["phase"] == "uploading":
        return _tsr_result(job) or _ts_result("running", tool=TOOL_NAME, pass_name="upload",
                                               elapsed_s=round(time.time() - job["t0"], 1))
    info = job["info"]
    if bpy.app.is_job_running("RENDER"):
        return _ts_result("running", tool=TOOL_NAME, pass_name=info["passes"][job["step"]],
                          elapsed_s=round(time.time() - job["t0"], 1))
    try:
        name = info["passes"][job["step"]]
        missing = [p for p in (_frame_path(job, name, i) for i in range(len(info["views"]))) if not os.path.exists(p)]
        if missing:
            return _tsr_abort(job, "Render pass {!r} produced no file {} (cancelled?)".format(name, missing[0]))
        end_t = _ts_state().get("tsr_render_end_t") or time.time()
        job["pass_s"][name] = round(end_t - job["pass_t0"], 2)
        job["errors"] += _TsrOverrides(job["step_saved"]).restore()
        job["step"] += 1
        if job["step"] < len(info["passes"]):
            _start_pass(job, scene)
            lap("status:" + name + "->next")
            return _ts_result("running", tool=TOOL_NAME, pass_name=info["passes"][job["step"]],
                              elapsed_s=round(time.time() - job["t0"], 1))
        job["errors"] += _tsr_restore_job(job)  # scene and materials back first
        t0 = time.time()
        sheet_paths, index_path = _compose(job)
        job["pass_s"]["compose"] = round(time.time() - t0, 2)
        for i, path in enumerate(sheet_paths):
            _tsr_upload_async(job, params.uploads, "sheet_{}".format(i), path)
        _tsr_upload_async(job, params.uploads, "index", index_path)
        job["out"] = dict(info["out"], cell_size=[info["width"], info["height"]], views=info["views"])
        job["phase"] = "uploading"
    except Exception:  # noqa: BLE001
        return _tsr_abort(job, traceback.format_exc()[-3000:])
    lap("status:finish")
    return _tsr_result(job) or _ts_result("running", tool=TOOL_NAME, pass_name="upload",
                                           elapsed_s=round(time.time() - job["t0"], 1))
