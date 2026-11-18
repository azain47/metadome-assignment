"""Tool-code: assign physically based material presets to the car's materials.

Model files often arrive with broken materials (exporter conversions: black mirrors with
roughness 0, matte paint, metallic glass, placeholder textures). Which source material is which
part is car-specific data (product facts); the presets here are generic.
"""

__all__ = ("Params", "main", "PRESETS")

from typing import NamedTuple

# Principled BSDF values per preset. Colours are sRGB hex. "base2" + "weave_mm" add a woven
# checker (carbon). Every preset except "keep" defines the whole material: inputs it does not
# list get the defaults in _DEFAULTS, and any texture on an input it sets is unlinked.
# Coated finishes (paint, painted wheels, carbon) keep the base layer's specular low: the clear
# coat carries the reflections. A default-level base specular spreads a low sun across whole
# panels and turns a black car gold.
PRESETS = {
    "car_paint": {"base": "#0E0F11", "roughness": 0.45, "specular": 0.1, "coat": 1.0, "coat_roughness": 0.03},
    # Pearl / tri-coat whites: a thin film on the base layer shifts the reflection colour slightly
    # at grazing angles under the clear coat.
    "pearl_paint": {"base": "#F2F2F0", "roughness": 0.4, "specular": 0.1, "coat": 1.0, "coat_roughness": 0.02,
                    "thin_film_nm": 380.0, "thin_film_ior": 1.45},
    # Metallic paints: fine flakes (Voronoi bump on the base layer only) sparkle under a smooth coat.
    "metallic_paint": {"base": "#1A1C20", "metallic": 0.6, "roughness": 0.35, "coat": 1.0, "coat_roughness": 0.03,
                       "flake_mm": 0.08, "flake_strength": 0.15},
    "matte_paint": {"base": "#202020", "roughness": 0.65},
    "carbon_fibre": {"base": "#121316", "base2": "#202227", "weave_mm": 4.0, "roughness": 0.35, "specular": 0.2,
                     "coat": 1.0, "coat_roughness": 0.02},
    "satin_trim": {"base": "#121212", "roughness": 0.55, "specular": 0.3},
    "rubber_tyre": {"base": "#161616", "roughness": 0.78, "specular": 0.3},
    "wheel_rim": {"base": "#0B0B0C", "roughness": 0.4, "specular": 0.1, "coat": 1.0, "coat_roughness": 0.06},
    # Conductors: Metallic BSDF (F82: base colour at normal incidence, `tint` at grazing angles).
    "metal": {"shader": "metal", "base": "#B8B8B8", "tint": "#E6E6E6", "roughness": 0.15},
    "metal_polished": {"shader": "metal", "base": "#D9D9D9", "tint": "#F2F2F2", "roughness": 0.05},
    "metal_brushed": {"shader": "metal", "base": "#8E8A84", "tint": "#C8C4BE", "roughness": 0.3},
    "brake_disc": {"shader": "metal", "base": "#4A4B4D", "tint": "#8A8B8D", "roughness": 0.25},
    "glass_clear": {"base": "#FFFFFF", "transmission": 1.0, "roughness": 0.0, "ior": 1.52},
    "glass_tinted": {"base": "#5A5E62", "transmission": 1.0, "roughness": 0.0, "ior": 1.52},
    # Car windows: thin-sheet glass (Fresnel mix of a tinted Transparent BSDF and a Glossy BSDF).
    # No refraction offset or solid-glass darkening, light reaches the cabin, and where nothing
    # is behind the window the film stays transparent so the plate shows through.
    "glass_car": {"shader": "thin_glass", "base": "#6A7076", "roughness": 0.0, "ior": 1.52},
    # Opaque black dielectric: reads as tinted glass over a dark cabin, for models without an interior.
    "glass_dark": {"base": "#050607", "roughness": 0.0, "ior": 1.52},
    # Solid optics (headlamp covers, lenses): Glass BSDF.
    "lamp_lens": {"shader": "glass", "base": "#FFFFFF", "roughness": 0.02, "ior": 1.49},
    # Lamp emitters seen through their lens: Emission added to a Glass BSDF, so a lit LED keeps
    # glassy reflections instead of reading as a flat white blob.
    "lamp_led": {"shader": "led", "base": "#FFFFFF", "roughness": 0.05, "ior": 1.49, "emission": "#FFFFFF",
                 "emission_strength": 3.0},
    "lamp_emissive": {"base": "#F2F2F2", "roughness": 0.2, "emission": "#FFFFFF", "emission_strength": 8.0},
    "keep": {},
}
_DEFAULTS = {"metallic": 0.0, "specular": 0.5, "coat": 0.0, "coat_roughness": 0.03, "transmission": 0.0,
             "ior": 1.5, "emission": "#000000", "emission_strength": 0.0, "alpha": 1.0, "thin_film_nm": 0.0}
_INPUTS = {"base": "Base Color", "metallic": "Metallic", "roughness": "Roughness", "specular": "Specular IOR Level",
           "coat": "Coat Weight", "coat_roughness": "Coat Roughness", "transmission": "Transmission Weight",
           "ior": "IOR", "emission": "Emission Color", "emission_strength": "Emission Strength", "alpha": "Alpha",
           "thin_film_nm": "Thin Film Thickness", "thin_film_ior": "Thin Film IOR"}
_COLOR_KEYS = ("base", "base2", "emission", "tint")
_KEYS = set(_INPUTS) | {"base2", "weave_mm", "shader", "tint", "flake_mm", "flake_strength"}
_WEAVE_NODE = "TS_WEAVE"
_FLAKE_PREFIX = "TS_FLAKES_"
_SHADER_PREFIX = "TS_SH_"  # nodes of a custom shader graph (removed when the preset changes)


class Params(NamedTuple):
    assignments_json: str  # {"<material name>" | "auto": {"preset": str, <override>: value}}


# @include_begin: _template_ts_common.py
# @include_end


def _hex_to_linear(hex_str: str):
    h = str(hex_str).strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6:
        raise ValueError("colours must look like #RRGGBB, got {!r}".format(hex_str))
    srgb = [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb) + (1.0,)


def _resolve(spec) -> dict:
    """Assignment -> full value dict (raises ValueError on unknown preset/keys/colours)."""
    if not isinstance(spec, dict) or "preset" not in spec:
        raise ValueError("each assignment needs a 'preset' ({})".format(sorted(PRESETS)))
    preset = spec["preset"]
    if preset not in PRESETS:
        raise ValueError("unknown preset {!r}; presets: {}".format(preset, sorted(PRESETS)))
    extra = set(spec) - _KEYS - {"preset"}
    if extra:
        raise ValueError("unknown keys {}; allowed: {}".format(sorted(extra), sorted(_KEYS)))
    values = {} if preset == "keep" else dict(_DEFAULTS)
    values.update(PRESETS[preset])
    values.update({k: v for k, v in spec.items() if k != "preset"})
    for k in _COLOR_KEYS:
        if k in values:
            values[k] = _hex_to_linear(values[k])
    return values


def _set_input(node, name, value) -> None:
    sock = node.inputs.get(name)
    if sock is None:
        return
    for link in list(sock.links):
        node.id_data.links.remove(link)
    sock.default_value = value


def _weave(mat, bsdf, values) -> None:
    """World-space checker between base and base2 on Base Color (carbon twill read at photo scale)."""
    nt = mat.node_tree
    old = nt.nodes.get(_WEAVE_NODE)
    if old is not None:
        nt.nodes.remove(old)
    if "base2" not in values:
        return
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    chk = nt.nodes.new("ShaderNodeTexChecker")
    chk.name = _WEAVE_NODE
    chk.inputs["Scale"].default_value = 1000.0 / max(0.5, float(values.get("weave_mm", 4.0)))
    chk.inputs["Color1"].default_value = values["base"]
    chk.inputs["Color2"].default_value = values["base2"]
    nt.links.new(geo.outputs["Position"], chk.inputs["Vector"])
    nt.links.new(chk.outputs["Color"], bsdf.inputs["Base Color"])


def _output(nt):
    return next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL" and n.is_active_output),
                None) or next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None) or \
        nt.nodes.new("ShaderNodeOutputMaterial")


def _thin_glass(nt, values):
    fresnel = nt.nodes.new("ShaderNodeFresnel")
    fresnel.inputs["IOR"].default_value = float(values.get("ior", 1.52))
    clear = nt.nodes.new("ShaderNodeBsdfTransparent")
    clear.inputs["Color"].default_value = values["base"]
    gloss = nt.nodes.new("ShaderNodeBsdfGlossy")
    gloss.inputs["Roughness"].default_value = float(values.get("roughness", 0.0))
    mix = nt.nodes.new("ShaderNodeMixShader")
    nt.links.new(fresnel.outputs[0], mix.inputs["Fac"])
    nt.links.new(clear.outputs[0], mix.inputs[1])
    nt.links.new(gloss.outputs[0], mix.inputs[2])
    return [fresnel, clear, gloss, mix], mix.outputs[0]


def _glass(nt, values):
    glass = nt.nodes.new("ShaderNodeBsdfGlass")
    glass.inputs["Color"].default_value = values["base"]
    glass.inputs["Roughness"].default_value = float(values.get("roughness", 0.0))
    glass.inputs["IOR"].default_value = float(values.get("ior", 1.5))
    return [glass], glass.outputs[0]


def _metal(nt, values):
    metal = nt.nodes.new("ShaderNodeBsdfMetallic")
    metal.fresnel_type = "F82"
    metal.inputs["Base Color"].default_value = values["base"]
    metal.inputs["Edge Tint"].default_value = values.get("tint", values["base"])
    metal.inputs["Roughness"].default_value = float(values.get("roughness", 0.2))
    return [metal], metal.outputs[0]


def _led(nt, values):
    nodes, glass = _glass(nt, values)
    emit = nt.nodes.new("ShaderNodeEmission")
    emit.inputs["Color"].default_value = values.get("emission", (1.0, 1.0, 1.0, 1.0))
    emit.inputs["Strength"].default_value = float(values.get("emission_strength", 3.0))
    add = nt.nodes.new("ShaderNodeAddShader")
    nt.links.new(emit.outputs[0], add.inputs[0])
    nt.links.new(glass, add.inputs[1])
    return nodes + [emit, add], add.outputs[0]


_SHADERS = {"thin_glass": _thin_glass, "glass": _glass, "metal": _metal, "led": _led}


def _flakes(mat, bsdf, values) -> None:
    """Metallic flakes: a fine Voronoi bump on the base layer's normal (the coat keeps the smooth
    geometry normal, so the flakes sparkle under a mirror-like clear coat)."""
    nt = mat.node_tree
    for node in [n for n in nt.nodes if n.name.startswith(_FLAKE_PREFIX)]:
        nt.nodes.remove(node)
    if "flake_mm" not in values:
        return
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    vor = nt.nodes.new("ShaderNodeTexVoronoi")
    vor.inputs["Scale"].default_value = 1000.0 / max(0.01, float(values["flake_mm"]))
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = float(values.get("flake_strength", 0.15))
    bump.inputs["Distance"].default_value = 0.0005
    for node in (geo, vor, bump):
        node.name = _FLAKE_PREFIX + node.name
    nt.links.new(geo.outputs["Position"], vor.inputs["Vector"])
    nt.links.new(vor.outputs["Distance"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])


def _set_shader(mat, bsdf, values) -> None:
    """Wire the material output to a custom shader graph (values["shader"]) or back to the
    Principled BSDF; any previous custom graph is removed."""
    nt = mat.node_tree
    for node in [n for n in nt.nodes if n.name.startswith(_SHADER_PREFIX)]:
        nt.nodes.remove(node)
    out = _output(nt)
    shader = values.get("shader")
    if shader:
        nodes, socket = _SHADERS[shader](nt, values)
        for node in nodes:
            node.name = _SHADER_PREFIX + node.name
    else:
        socket = bsdf.outputs[0]
    nt.links.new(socket, out.inputs["Surface"])


def main(params: Params):
    import json

    if _ts_busy():
        return _ts_busy_result()
    root = _ts_car_root()
    if root is None:
        return _ts_result("error", message="No car loaded (call load_car first)")
    try:
        assignments = json.loads(params.assignments_json)
        if not isinstance(assignments, dict) or not assignments:
            raise ValueError("assignments must be a non-empty object")
        resolved = {name: _resolve(spec) for name, spec in assignments.items()}
        bad = sorted({v["shader"] for v in resolved.values() if v.get("shader") and v["shader"] not in _SHADERS})
        if bad:
            raise ValueError("unknown shader {}; shaders: {}".format(bad, sorted(_SHADERS)))
    except ValueError as ex:
        return _ts_result("error", message=str(ex), presets=sorted(PRESETS))

    car_mats = {s.material.name: s.material for o in root.children_recursive
                for s in getattr(o, "material_slots", ()) if s.material is not None}
    if "auto" in resolved:
        auto = resolved.pop("auto")
        for name in root.get("ts_paint_materials", []):
            resolved.setdefault(str(name), auto)
    missing = sorted(n for n in resolved if n not in car_mats)
    if missing:
        return _ts_result("error", message="Not car materials: {}".format(missing), available=sorted(car_mats))

    changed, skipped = [], []
    for name, values in resolved.items():
        mat = car_mats[name]
        nt = getattr(mat, "node_tree", None)
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None) if nt else None
        if bsdf is None:
            skipped.append(name)
            continue
        for key, value in values.items():
            if key in _INPUTS:
                _set_input(bsdf, _INPUTS[key], value)
        if "base" in values:
            _weave(mat, bsdf, values)
            _flakes(mat, bsdf, values)
            mat.diffuse_color = values["base"]
        if values.get("shader") or "base" in values:
            _set_shader(mat, bsdf, values)
        if values.get("alpha", 1.0) >= 1.0 and hasattr(mat, "surface_render_method"):
            mat.surface_render_method = "DITHERED"
        changed.append(name)
    _ts_redraw()
    return _ts_result("ok", changed=changed, skipped_no_principled=skipped)
