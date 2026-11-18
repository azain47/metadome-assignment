"""Tool-code: render the car alone in Cycles (transparent film, shadow catcher, spinning wheels).

Default (no plate): reflections and ambient light come from the world, an HDRI environment
texture chosen to match the location. The road, its markings and the ground stay shadow catchers
for camera and diffuse rays (the car's shadow, contact darkening) but are invisible to glossy
rays, so the paint mirrors the HDRI's full surroundings (ground, trees, buildings, sky) rather
than a flat grey plane. Stand-ins and the sea are hidden from every ray type.

Optional (`plate_url`): the plate is camera-projected from TS_CAM onto the whole set for every
non-camera ray, so reflections show the generated scene with parallax. Directions the plate
cannot see (behind the camera) smear its edge pixels; prefer the HDRI default for hero shots.
Stand-ins and sea become visible to glossy rays only. Camera rays on the catchers keep a diffuse
surface so the shadow catcher still works.
"""

__all__ = ("Params", "main")

from typing import NamedTuple


class Params(NamedTuple):
    action: str  # "start" | "status"
    uploads: tuple = ()  # ((name, presigned_put_url, content_type), ...)
    samples: int = 128
    speed_kmh: float = 40.0
    spoke_count: int = 0
    motion_blur: bool = True
    resolution_x: int = 1536
    resolution_y: int = 1024
    plate_url: str = ""  # presigned GET of the plate to project; "" = no projection
    plate_cache_name: str = ""
    plate_strength: float = 1.0
    shutter_s: float = 0.0  # > 0: one exposure for wheels and body (match the plate's motion blur)
    body_slip_m: float = 0.0  # car motion relative to a tracking camera over the shutter
    fstop: float = 0.0  # > 0: depth of field focused on the car
    probes: bool = False  # add a chrome and a grey reference ball beside the car (lighting check)


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end

# @include_begin: _template_ts_layout.py
# @include_end

# @include_begin: _template_ts_render.py
# @include_end

SHADOW_CATCHER_ROLES = ("road", "road_marking", "ground")
MAX_SHUTTER_FRAMES = 4.0  # wheel keyframes cover frame -2..+2
PNG_VIEW_TRANSFORM = "Khronos PBR Neutral"  # same tone mapper as the compositor


def _projection_material(ov, scene, image, strength: float):
    """Material that shows `image` projected through the scene camera (removed on restore).
    Camera rays see a diffuse surface in the plate's colour (shadow catchers need a BSDF).
    Reflection and bounce rays see the plate as emission where the surface lies inside the
    camera's view, fading to transparent at the frame edge: outside the frame the plate has no
    pixels, so those rays pass through to the world (HDRI) instead of a smeared edge."""
    import bpy  # pylint: disable=import-error
    cam = scene.camera
    cd = cam.data
    rx, ry = scene.render.resolution_x, scene.render.resolution_y
    horizontal = cd.sensor_fit == "HORIZONTAL" or (cd.sensor_fit == "AUTO" and rx >= ry)
    sensor = cd.sensor_width if (cd.sensor_fit != "VERTICAL") else cd.sensor_height
    k = cd.lens / sensor  # normalised by the fitted dimension
    aspect = ry / rx if horizontal else rx / ry
    inv = cam.matrix_world.inverted()

    mat = bpy.data.materials.new("TS_TMP_PROJECTION")
    ov.call_on_restore("_tsr_remove_ids", "materials", [mat.name])
    mat.use_nodes = True
    # The ground and stand-ins are huge: as mesh lights they would flood the light tree with
    # millions of square metres of dim emitters. Reflection and bounce rays find them anyway.
    mat.cycles.emission_sampling = "NONE"
    nt = mat.node_tree
    nt.nodes.clear()
    n, link = nt.nodes.new, nt.links.new
    geo = n("ShaderNodeNewGeometry")

    def math(op, a, b, c=None):
        m = n("ShaderNodeMath")
        m.operation = op
        for i, v in enumerate((a, b, c)):
            if v is None:
                continue
            if isinstance(v, (int, float)):
                m.inputs[i].default_value = float(v)
            else:
                link(v, m.inputs[i])
        return m.outputs[0]

    comps = []
    for row in range(3):
        dot = n("ShaderNodeVectorMath")
        dot.operation = "DOT_PRODUCT"
        link(geo.outputs["Position"], dot.inputs[0])
        dot.inputs[1].default_value = tuple(inv[row][:3])
        comps.append(math("ADD", dot.outputs["Value"], inv[row][3]))
    x, y, z = comps
    depth = math("MAXIMUM", math("MULTIPLY", z, -1.0), 1e-3)
    a, b = (k, k / aspect) if horizontal else (k * aspect, k)
    u = math("MULTIPLY_ADD", math("DIVIDE", x, depth), a, 0.5 - (cd.shift_x if horizontal else cd.shift_x / aspect))
    v = math("MULTIPLY_ADD", math("DIVIDE", y, depth), b, 0.5 - (cd.shift_y / aspect if horizontal else cd.shift_y))
    uv = n("ShaderNodeCombineXYZ")
    link(u, uv.inputs[0])
    link(v, uv.inputs[1])
    tex = n("ShaderNodeTexImage")
    tex.image, tex.extension, tex.interpolation = image, "EXTEND", "Linear"
    link(uv.outputs[0], tex.inputs["Vector"])

    def edge(t):  # 0 at the frame edge -> 1 a twentieth of the frame inside
        return math("MINIMUM", math("MULTIPLY", math("MINIMUM", t, math("SUBTRACT", 1.0, t)), 20.0), 1.0)

    in_front = math("GREATER_THAN", math("MULTIPLY", z, -1.0), 0.05)
    inside = math("MAXIMUM", math("MULTIPLY", math("MULTIPLY", edge(u), edge(v)), in_front), 0.0)
    emit = n("ShaderNodeEmission")
    emit.inputs["Strength"].default_value = float(strength)
    link(tex.outputs["Color"], emit.inputs["Color"])
    seen = n("ShaderNodeMixShader")  # outside the frame: let the world through
    link(inside, seen.inputs["Fac"])
    link(n("ShaderNodeBsdfTransparent").outputs[0], seen.inputs[1])
    link(emit.outputs[0], seen.inputs[2])
    diffuse = n("ShaderNodeBsdfDiffuse")
    link(tex.outputs["Color"], diffuse.inputs["Color"])
    path = n("ShaderNodeLightPath")
    mix = n("ShaderNodeMixShader")
    link(path.outputs["Is Camera Ray"], mix.inputs["Fac"])
    link(seen.outputs[0], mix.inputs[1])
    link(diffuse.outputs[0], mix.inputs[2])
    out = n("ShaderNodeOutputMaterial")
    link(mix.outputs[0], out.inputs["Surface"])
    return mat


def _add_probes(ov, root) -> None:
    """VFX reference balls just ahead of the car's nose, in TS_CAR so they render with it: a mirror
    ball shows exactly what the car reflects (is the HDRI there, the right way round, undisturbed
    by stand-ins), an 18% grey ball the light's direction, colour and softness. Removed on restore."""
    import bpy  # pylint: disable=import-error
    from mathutils import Vector  # pylint: disable=import-error
    nose_y = float(root["ts_dims_m"][0]) / 2.0  # origin = tyre contact centre, front = +Y
    radius = 0.3
    specs = (("TS_TMP_CHROME", (1.0, 1.0, 1.0, 1.0), 1.0, 0.0, -0.45),
             ("TS_TMP_GREY", (0.18, 0.18, 0.18, 1.0), 0.0, 0.5, 0.45))
    names = {"objects": [], "meshes": [], "materials": []}
    car_coll = bpy.data.collections.get(TS_CAR)
    for name, colour, metallic, rough, x in specs:
        mesh = bpy.data.meshes.new(name)
        ob = bpy.data.objects.new(name, mesh)
        import bmesh  # pylint: disable=import-error
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=radius)
        bm.to_mesh(mesh)
        bm.free()
        for poly in mesh.polygons:
            poly.use_smooth = True
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        bsdf = next(n for n in mat.node_tree.nodes if n.type == "BSDF_PRINCIPLED")
        bsdf.inputs["Base Color"].default_value = colour
        bsdf.inputs["Metallic"].default_value = metallic
        bsdf.inputs["Roughness"].default_value = rough
        mesh.materials.append(mat)
        car_coll.objects.link(ob)
        ob.parent = root
        ob.location = Vector((x, nose_y + 0.8, radius))
        names["objects"].append(ob.name)
        names["meshes"].append(mesh.name)
        names["materials"].append(mat.name)
    for kind in ("materials", "meshes", "objects"):
        ov.call_on_restore("_tsr_remove_ids", kind, names[kind])


def _setup_car(ov, scene, job) -> None:
    import bpy  # pylint: disable=import-error
    groups = _tsr_groups(scene)
    catchers = [o for o in groups["set"] if str(o.get("ts_role", "")) in SHADOW_CATCHER_ROLES]
    mat = bpy.data.materials.get(job.get("projection_mat", "")) if job.get("projection_mat") else None
    others = [o for o in groups["set"] if o not in catchers and o.type == "MESH"] if mat else []
    _tsr_show_only(ov, scene, groups["car"] + catchers + others)
    for ob in catchers:
        ov.set(ob, "is_shadow_catcher", True)
        if mat is None:
            # The world (HDRI) is the car's only environment: the catchers are flat grey/white stand-in
            # surfaces out to the horizon, so if they were seen by reflection or bounce rays they would
            # replace the HDRI's ground in the paint and wash the car with grey-white fill light. They
            # stay visible to the camera only, where they receive the car's shadow.
            for ray in ("diffuse", "glossy", "transmission", "volume_scatter"):
                ov.set(ob, "visible_" + ray, False)
    # Stand-ins (ridges, buildings, trees, sea) carry the projected plate into the car's
    # reflections only (glossy rays). As diffuse-visible they add almost no bounce light to the
    # car (measured: car mean colour unchanged) but hide sky from the ground, and the catcher
    # records that as false shadow away from the car (up to 19% darker). Ambient light comes
    # from the projected ground, which every ray type sees.
    for ob in others:
        for ray in ("camera", "shadow", "diffuse", "transmission", "volume_scatter"):
            ov.set(ob, "visible_" + ray, False)
    if mat is not None:
        for ob in catchers + others:
            if ob.type == "MESH" and ob.material_slots:
                ov.set(ob, "active_material", mat)
    _tsr_only_passes(ov, {"use_pass_combined", "cycles.use_pass_shadow_catcher"})


def _after_car(ov, scene, job, path) -> dict:
    import os
    import bpy  # pylint: disable=import-error
    job["info"]["exr_channels"] = _tsr_exr_channels(path)
    png = os.path.join(os.path.dirname(path), "car_rgba.png")
    _tsr_set_format(ov, scene, "PNG_RGBA8")
    ov.scene(scene, {"view_settings.view_transform": PNG_VIEW_TRANSFORM, "view_settings.look": "None",
                     "view_settings.use_curve_mapping": False, "render.dither_intensity": 0.0})
    bpy.data.images["Render Result"].save_render(png, scene=scene)
    return {"rgba_png": png}


def _car_settings(params: Params) -> dict:
    return {
        "render.engine": "CYCLES",
        "render.film_transparent": True,
        "cycles.film_transparent_glass": True,
        "cycles.film_transparent_roughness": 0.1,
        "cycles.samples": max(1, int(params.samples)),
        "cycles.use_layer_samples": "IGNORE",
        "cycles.use_adaptive_sampling": True,
        "cycles.adaptive_threshold": 0.01,
        "cycles.time_limit": 0.0,
        "cycles.use_denoising": True,
        "cycles.denoiser": "OPENIMAGEDENOISE",
        "cycles.denoising_use_gpu": True,
        "cycles.denoising_input_passes": "RGB_ALBEDO_NORMAL",
        "cycles.denoising_prefilter": "ACCURATE",
        "cycles.rolling_shutter_type": "NONE",
    }


def _slip_body(ov, scene, root, slip_m: float, shutter_frames: float) -> None:
    """Move the car `slip_m` along its forward axis over the shutter (a tracking camera never
    matches the car's speed exactly), so the body gets the slight blur of a real rolling shot.
    Linear keyframes around the current frame, so the shutter centre keeps the guide pose."""
    from bpy_extras import anim_utils  # pylint: disable=import-error
    from mathutils import Vector  # pylint: disable=import-error
    f = scene.frame_current
    per_frame = (root.matrix_world.to_3x3() @ Vector((0.0, 1.0, 0.0))).normalized() * (slip_m / shutter_frames)
    base = root.location.copy()
    ov.set(root, "location", tuple(base))
    had_action = root.animation_data is not None and root.animation_data.action is not None
    if had_action:
        ov.set(root, "animation_data.action", None)
    for df in (-2, 0, 2):
        root.location = base + per_frame * df
        root.keyframe_insert("location", frame=f + df)
    root.location = base
    anim = root.animation_data
    bag = anim_utils.action_get_channelbag_for_slot(anim.action, anim.action_slot)
    for fc in bag.fcurves:
        fc.extrapolation = "LINEAR"
        for kp in fc.keyframe_points:
            kp.interpolation = "LINEAR"
    ov.call_on_restore("_tsr_unspin", [[root.name, anim.action.name, had_action]])


def _prepare_for(params: Params):
    def prepare(ov, scene, job) -> dict:
        import math
        ov.scene(scene, {"render.resolution_x": int(params.resolution_x),
                         "render.resolution_y": int(params.resolution_y)})
        root = _ts_car_root()
        wheels = _ts_wheel_pivots(root)
        radius = (sum(w["radius_m"] for w in wheels) / len(wheels)) if wheels else \
            float(root.get("ts_wheel_radius_m", 0.33)) if root is not None else 0.33
        omega = (params.speed_kmh / 3.6) / max(radius, 0.05)
        fps = scene.render.fps / (scene.render.fps_base or 1.0)
        if params.shutter_s > 0:
            shutter = params.shutter_s * fps  # one exposure shared with the plate's motion blur
        elif params.spoke_count > 0 and omega > 0:
            shutter = (math.pi / params.spoke_count) / omega * fps
        else:
            shutter = 0.5
        shutter = min(max(shutter, 0.01), MAX_SHUTTER_FRAMES)
        pivots = [w["object"] for w in wheels if w["object"].get("ts_wheel_pivot")]
        spin = bool(params.motion_blur) and omega > 0 and bool(pivots)
        slip = bool(params.motion_blur) and params.body_slip_m > 0 and root is not None
        blur = spin or slip
        ov.scene(scene, {"render.use_motion_blur": blur, "render.motion_blur_shutter": shutter,
                         "render.motion_blur_position": "CENTER"})
        if spin:
            sign = float(root.get("ts_wheel_forward_sign", -1.0))
            _tsr_spin_wheels(ov, scene, pivots, omega, sign)
            # Enough transform steps that each segment of the arc stays small (< ~0.1 rad).
            arc = omega * shutter / fps
            steps = min(7, max(1, math.ceil(math.log2(max(arc / 0.1, 1.0))) + 1))
            for piv in pivots:
                for ob in [piv, *piv.children_recursive]:
                    ov.set(ob, "cycles.use_motion_blur", True)
                    ov.set(ob, "cycles.motion_steps", steps)
            job["info"]["motion_steps"] = steps
        if slip:
            _slip_body(ov, scene, root, params.body_slip_m, shutter)
        if params.fstop > 0 and scene.camera is not None and root is not None:
            cam = scene.camera
            focus = (cam.matrix_world.translation - root.matrix_world.translation).length
            ov.set(cam.data, "dof.use_dof", True)
            ov.set(cam.data, "dof.focus_distance", focus)
            ov.set(cam.data, "dof.aperture_fstop", float(params.fstop))
        job["info"].update(shutter_frames=round(shutter, 4), omega_rad_s=round(omega, 3),
                           motion_blur=blur, body_slip_m=params.body_slip_m if slip else 0.0,
                           fstop=params.fstop, wheels=_ts_layout(scene)["wheels"])
        ov.scene(scene, _car_settings(params))
        if params.probes and root is not None:
            _add_probes(ov, root)
        if params.plate_url:
            import bpy  # pylint: disable=import-error
            path = _ts_http_get_cached(params.plate_url, params.plate_cache_name or "plate.png")
            image = bpy.data.images.load(path, check_existing=False)
            ov.call_on_restore("_tsr_remove_ids", "images", [image.name])
            job["projection_mat"] = _projection_material(ov, scene, image, params.plate_strength).name
        job["info"].update(device=_tsr_device_label(scene), samples=scene.cycles.samples,
                           projected_plate=bool(params.plate_url))
        return {}
    return prepare


PASSES = [
    {"name": "car", "format": "EXR_MULTI32", "upload": "exr", "setup": _setup_car, "after": _after_car},
]


def _finish(scene, job) -> dict:
    info = job["info"]
    return {"exr_channels": info.get("exr_channels", []), "render_s": job["pass_s"].get("car"),
            "samples": info["samples"], "shutter_frames": info["shutter_frames"],
            "omega_rad_s": info["omega_rad_s"], "motion_blur": info["motion_blur"],
            "motion_steps": info.get("motion_steps"), "wheels": info["wheels"], "device": info["device"],
            "png_view_transform": PNG_VIEW_TRANSFORM}


def main(params: Params):
    return _tsr_chain(params, "render_car", PASSES, _prepare_for(params), _finish)
