"""Tool-code: world lighting (procedural sky or HDRI) plus the TS_SUN light."""

__all__ = ("Params", "main", "SKIES")

from typing import NamedTuple

SKIES = ("procedural", "hdri")
_WORLD = "TS_WORLD"


class Params(NamedTuple):
    sky: str
    hdri_key: str
    hdri_url: str
    hdri_cache_name: str
    hdri_rotation_deg: float
    strength: float
    sun_azimuth_deg: float
    sun_elevation_deg: float
    sun_strength: float
    sun_angle_deg: float
    exposure: float
    align_sun: bool = False  # rotate the HDRI so its measured sun sits at align_sun_azimuth_deg
    align_sun_azimuth_deg: float = 0.0


# @include_begin: _template_ts_common.py
# @include_end

# @include_begin: _template_ts_scene.py
# @include_end


def _world_procedural(nt, params):
    import math
    sky = nt.nodes.new("ShaderNodeTexSky")
    sky.sky_type = "MULTIPLE_SCATTERING"
    sky.sun_disc = False  # TS_SUN provides the sun; avoid a second, unshadowed one
    sky.sun_elevation = math.radians(params.sun_elevation_deg)
    # Measured in 5.3 (equirect render of the sky with the disc on): sun_rotation is the
    # compass azimuth (0 = +Y, 90 = +X).
    sky.sun_rotation = math.radians(params.sun_azimuth_deg) % (2.0 * math.pi)
    return sky.outputs["Color"]


def _hdri_image(params):
    import bpy  # pylint: disable=import-error
    path = _ts_http_get_cached(params.hdri_url, params.hdri_cache_name)
    return bpy.data.images.load(path, check_existing=True)


def _hdri_sun(image) -> dict:
    """Brightest region of the equirect map, in Blender's frame before rotation (compass
    azimuth: 0 = +Y at u = 0.25, 90 = +X at u = 0.5), plus peak/median contrast (< ~50 = no
    usable sun, e.g. overcast). Cached per image."""
    import numpy as np
    cache = _ts_state().setdefault("hdri_sun", {})
    if image.name in cache:
        return cache[image.name]
    w, h = image.size
    px = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(px)
    px = px.reshape(h, w, 4)  # rows bottom-up
    lum = 0.2126 * px[..., 0] + 0.7152 * px[..., 1] + 0.0722 * px[..., 2]
    f = max(1, w // 512)
    hh, ww = h // f, w // f
    small = lum[:hh * f, :ww * f].reshape(hh, f, ww, f).mean(axis=(1, 3))
    y, x = np.unravel_index(int(np.argmax(small)), small.shape)
    sun = {"azimuth_deg": round(float(((x + 0.5) / ww - 0.25) * 360.0 % 360.0), 1),
           "elevation_deg": round(float((y + 0.5) / hh * 180.0 - 90.0), 1),
           "contrast": round(float(small[y, x] / max(float(np.median(small)), 1e-6)), 1)}
    cache[image.name] = sun
    return sun


def _world_hdri(nt, params, image, rotation_deg: float):
    import math
    env = nt.nodes.new("ShaderNodeTexEnvironment")
    env.image = image
    coord = nt.nodes.new("ShaderNodeTexCoord")
    mapping = nt.nodes.new("ShaderNodeMapping")
    mapping.inputs["Rotation"].default_value[2] = math.radians(rotation_deg)
    nt.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
    nt.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
    return env.outputs["Color"]


_SKY_BUILDERS = {"procedural": _world_procedural, "hdri": _world_hdri}


def main(params: Params):
    import math
    import bpy  # pylint: disable=import-error

    if _ts_busy():
        return _ts_busy_result()
    if params.sky not in _SKY_BUILDERS:
        return _ts_result("error", message="sky must be one of {}".format(SKIES))
    scene = bpy.context.scene

    world = bpy.data.worlds.get(_WORLD) or bpy.data.worlds.new(_WORLD)
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputWorld")
    bg = nt.nodes.new("ShaderNodeBackground")
    bg.inputs["Strength"].default_value = float(params.strength)
    hdri_sun, rotation = None, params.hdri_rotation_deg
    if params.sky == "hdri":
        image = _hdri_image(params)
        hdri_sun = _hdri_sun(image)
        if params.align_sun:
            # World sun azimuth = image sun azimuth + rotation (Generated coords through Mapping).
            rotation = ((params.align_sun_azimuth_deg - hdri_sun["azimuth_deg"]) + 180.0) % 360.0 - 180.0
        color = _world_hdri(nt, params, image, rotation)
    else:
        color = _world_procedural(nt, params)
    nt.links.new(color, bg.inputs["Color"])
    nt.links.new(bg.outputs["Background"], out.inputs["Surface"])
    scene.world = world

    sun = bpy.data.objects.get("TS_SUN")
    if sun is None or sun.type != "LIGHT":
        data = bpy.data.lights.get("TS_SUN") or bpy.data.lights.new("TS_SUN", "SUN")
        sun = bpy.data.objects.new("TS_SUN", data)
        sun["ts_role"] = "sun"
    rig = _ts_collection(TS_RIG)
    if rig not in sun.users_collection:
        rig.objects.link(sun)
    sun.data.type = "SUN"
    sun.data.energy = float(params.sun_strength)
    sun.data.angle = math.radians(max(0.0, params.sun_angle_deg))
    sun_az, sun_el = params.sun_azimuth_deg, params.sun_elevation_deg
    if hdri_sun is not None and params.align_sun:  # a sun light, if enabled, follows the HDRI's sun
        sun_az, sun_el = params.align_sun_azimuth_deg, hdri_sun["elevation_deg"]
    d = _ts_azel_to_dir(sun_az, sun_el)
    sun.rotation_mode = "QUATERNION"
    sun.rotation_quaternion = d.to_track_quat("Z", "Y")
    sun.location = tuple(d * 20.0)
    sun.hide_render = params.sun_strength <= 0.0

    scene.view_settings.exposure = float(params.exposure)
    scene["ts_world_sky"] = params.sky
    scene["ts_world_hdri_key"] = params.hdri_key if params.sky == "hdri" else ""
    scene["ts_world_strength"] = float(params.strength)
    _ts_redraw()
    return _ts_result(
        "ok",
        sky=params.sky,
        sun={"azimuth_deg": round(sun_az % 360.0, 2), "elevation_deg": sun_el,
             "strength": params.sun_strength, "angle_deg": params.sun_angle_deg},
        hdri_key=params.hdri_key if params.sky == "hdri" else None,
        hdri_rotation_deg=round(rotation, 2) if params.sky == "hdri" else None,
        hdri_sun=hdri_sun,
        strength=params.strength,
        exposure=params.exposure,
    )
