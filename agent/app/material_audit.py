"""Material audit: decide a physically based preset (and colour) for every material of a car model.

Downloaded models often arrive with broken materials (exporter conversions: black mirrors, lost
colours replaced by a placeholder, metallic glass). Blender's `render_material_sheet` shows each
material orange on the grey car from a few views (one sheet row per material, plus
`materials.json` with the material's current values); this module cuts those rows into one
compact card per material, sends the cards and reference photos of the real car to a vision
model, and turns the answers into a `set_materials(assignments=...)` map. Generic for any car.

    sheets  = [<material_sheet_k.png bytes>, ...]       # from render_material_sheet
    index   = json.loads(<materials.json>)
    refs    = [await fetch_reference(url)]              # JPEG, long side <= 768
    result  = await audit_materials(sheets, index["materials"], refs, "McLaren ... in white")
    await blender.call("set_materials", assignments=result.assignments)

Cost rule: every image is downscaled before the call (cards 2x2 cells of `CARD_CELL_W` px,
reference photos long side <= 768 px, JPEG) and sent with detail "low"; all materials go into as
few calls as possible (`max_per_call`); token usage and dollars accumulate on the `Vlm`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import cv2
import numpy as np
from pydantic import BaseModel

from .imaging.io import decode_rgb, encode_jpeg
from .references import JPEG_QUALITY, FetchBytes, VisionModel, Vlm, _http_bytes, downscale_jpeg

log = logging.getLogger(__name__)

MAX_SIDE = 768  # cost rule: long side of every image sent to the model
CARD_CELL_W = 320  # width of one view in a material card (cards are 2x2 views)
CARD_GAP = 4
MAX_PER_CALL = 48

# The presets of Blender's `set_materials` (mcp/blmcp/tools/ts_set_materials_toolcode.PRESETS; a
# test keeps the names in sync) with what each is for, as the model reads it.
PRESETS: dict[str, str] = {
    "car_paint": "glossy clear-coated solid paint, any colour (body panels, painted stripes, gloss black painted parts)",
    "pearl_paint": "pearl / tri-coat paint, usually white (subtle colour shift at grazing angles under the clear coat)",
    "metallic_paint": "metallic / flake paint with visible sparkle and colour flop",
    "matte_paint": "matte or satin wrapped/painted body panels (no clear-coat reflections)",
    "carbon_fibre": "exposed, clear-coated carbon-fibre weave (splitters, diffusers, mirrors, wings, sills)",
    "satin_trim": "black/dark satin plastic or rubber trim, grilles, mesh, seals, interior, underbody",
    "rubber_tyre": "tyre rubber (sidewall and tread)",
    "wheel_rim": "wheel rims / spokes (painted or machined; give the rim colour)",
    "metal": "bare metal hardware (give its colour)",
    "metal_polished": "chrome or polished metal (exhaust tips, badges, trim strips)",
    "metal_brushed": "brushed / anodised metal (calipers are NOT this; exhaust heat shields, mechanical parts)",
    "brake_disc": "brake discs / rotors",
    "glass_car": "car windows and windscreen over a modelled cabin (thin tinted glass)",
    "glass_clear": "clear solid glass with a modelled interior behind it",
    "glass_tinted": "tinted solid see-through glass with a modelled interior behind it",
    "glass_dark": "windows / windscreen of a model without a usable interior (reads as dark tinted glass)",
    "lamp_lens": "clear headlight / tail-light cover lens",
    "lamp_led": "LED or lamp emitter seen through its lens (headlamp DRL, light bars)",
    "lamp_emissive": "plain light-emitting element without a lens (tail-light emitters, bulbs)",
    "keep": "leave as is (the material's own texture already looks right, e.g. a decal, badge or real carbon texture)",
}
# Presets whose look depends on a base colour the model should pick from the references.
COLOUR_PRESETS = frozenset({"car_paint", "pearl_paint", "metallic_paint", "matte_paint", "satin_trim", "wheel_rim",
                            "metal", "metal_brushed"})

Preset = Literal[
    "car_paint", "pearl_paint", "metallic_paint", "matte_paint", "carbon_fibre", "satin_trim", "rubber_tyre",
    "wheel_rim", "metal", "metal_polished", "metal_brushed", "brake_disc", "glass_car", "glass_clear",
    "glass_tinted", "glass_dark", "lamp_lens", "lamp_led", "lamp_emissive", "keep",
]

_HEX_RE = re.compile(r"^#?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def normalise_hex(value: str) -> str:
    """'#abc' / 'aabbcc' / ' #AaBbCc ' -> '#AABBCC'; raises ValueError for anything else."""
    m = _HEX_RE.match(str(value).strip())
    if not m:
        raise ValueError(f"not a #RRGGBB colour: {value!r}")
    h = m.group(1)
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return "#" + h.upper()


# ---------------------------------------------------------------------------
# Model answer (structured output)


class MaterialAnswer(BaseModel):
    row: int  # the material's row number (#NN on its card caption)
    part: str  # what the orange area is on the car, e.g. "lower door sills and side skirts"
    seen_in_reference: str  # how that part looks on the reference photos, a few words
    preset: Preset
    base_hex: str | None  # sRGB "#RRGGBB" for colour presets, else null
    emission_hex: str | None  # light colour for lamp_emissive (e.g. red tail lights), else null
    confidence: float  # 0..1


class AuditAnswer(BaseModel):
    livery: str  # the real car's colour scheme from the references, one sentence
    materials: list[MaterialAnswer]


# ---------------------------------------------------------------------------
# Result


@dataclass
class MaterialVerdict:
    name: str
    row: int
    preset: str
    base: str | None
    emission: str | None
    part: str
    seen: str
    confidence: float
    area_pct: float

    def assignment(self) -> dict[str, Any] | None:
        """This material's `set_materials` entry (None for "keep": nothing to change)."""
        if self.preset == "keep":
            return None
        spec: dict[str, Any] = {"preset": self.preset}
        if self.base and self.preset in COLOUR_PRESETS:
            spec["base"] = self.base
        if self.emission and self.preset == "lamp_emissive":
            spec["emission"] = self.emission
        return spec


@dataclass
class AuditResult:
    materials: list[MaterialVerdict]
    livery: str = ""
    problems: list[str] = field(default_factory=list)  # unanswered rows, invalid colours, ...
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def assignments(self) -> dict[str, dict[str, Any]]:
        """{material name: {"preset", "base"?, "emission"?}}, ready for `set_materials`."""
        out = {}
        for v in self.materials:
            spec = v.assignment()
            if spec is not None:
                out[v.name] = spec
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"livery": self.livery, "assignments": self.assignments,
                "materials": [asdict(m) for m in self.materials], "problems": self.problems, "usage": self.usage}


# ---------------------------------------------------------------------------
# Images


async def fetch_reference(url: str, max_side: int = MAX_SIDE, *, fetch_bytes: FetchBytes | None = None) -> bytes:
    """A reference photo as a JPEG whose long side is <= max_side (never upscaled)."""
    return downscale_jpeg(await (fetch_bytes or _http_bytes)(url), max_side)


def material_card(sheet: np.ndarray, material: dict[str, Any], cell_w: int = CARD_CELL_W) -> np.ndarray:
    """One material's sheet cells (its views, then the close-up of its current look) tiled two per
    row at `cell_w` px each. The sheet's label band is left out (the caption carries the row)."""
    tiles = [np.ascontiguousarray(sheet[c["y"]:c["y"] + c["h"], c["x"]:c["x"] + c["w"]])
             for c in material["cells"]]
    ch = round(cell_w * tiles[0].shape[0] / tiles[0].shape[1])
    tiles = [cv2.resize(t, (cell_w, ch), interpolation=cv2.INTER_AREA) for t in tiles]
    cols = 2
    rows = (len(tiles) + cols - 1) // cols
    card = np.full((rows * ch + (rows - 1) * CARD_GAP, cols * cell_w + (cols - 1) * CARD_GAP, 3), 90, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        card[r * (ch + CARD_GAP):r * (ch + CARD_GAP) + ch, c * (cell_w + CARD_GAP):c * (cell_w + CARD_GAP) + cell_w] = t
    return card


def material_cards(sheets: list[bytes], materials: list[dict[str, Any]],
                   cell_w: int = CARD_CELL_W) -> list[bytes]:
    """JPEG card per material (order of `materials`), each long side <= MAX_SIDE."""
    decoded = [decode_rgb(s) for s in sheets]
    out = []
    for m in materials:
        card = material_card(decoded[int(m["sheet"])], m, cell_w)
        out.append(downscale_jpeg(encode_jpeg(card, quality=95), MAX_SIDE, JPEG_QUALITY))
    return out


# ---------------------------------------------------------------------------
# The model call

# How the cells of render_material_sheet (VIEWS in its tool code, plus "closeup") read on a card.
VIEW_WORDS = {
    "front": "straight front view", "front34": "front-left 3/4 view", "front34_r": "front-right 3/4 view",
    "side": "left side view", "side_r": "right side view", "rear34": "rear-right 3/4 view",
    "rear34_l": "rear-left 3/4 view", "rear": "straight rear view", "top": "high view from the front",
    "closeup": "CURRENT LOOK close-up",
}

AUDIT_PROMPT = """You fix the materials of a 3D car model so renders match the real car{car}.
First come {n_ref} reference photo(s) of the real car. Then one card per material of the 3D model. Its
panels, left to right and top to bottom: {panels}. The view panels show the SAME grey render of the model with
the parts using that material painted ORANGE (magenta boxes only mark orange areas too small to see easily;
"not visible" = hidden in that view). The CURRENT LOOK close-up shows the model's present colours and
textures around that material's largest area, outlined in orange. The caption before each card gives its
row number, the share of the car's visible surface, and the material's current values. Current colours are
often broken by file conversion: a saturated purple/magenta base is usually a lost-colour placeholder,
black with roughness 0 or metallic 1 are exporter defaults; do not trust them, judge from the references.

For every card: identify the orange part, find that exact part on the reference photos (judge its own
colour, not that of neighbouring panels), and pick the preset that reproduces its real finish:
{presets}
Rules:
- base_hex: for {colour_presets}, the part's real colour as sRGB "#RRGGBB", as it would look in neutral
  daylight (judge the mid-tone between highlights and shadows; white paint is near #F0F0F0, black near
  #0C0C0C); null for other presets.
- Body panels whose colour you can see on the references get car_paint (or metallic_paint / matte_paint when
  the finish clearly is one); parts that carry stripes or two-tone colours get the colour of that part.
- Black parts: gloss painted -> car_paint with a near-black base; visible weave -> carbon_fibre; dull
  plastic, grilles, mesh, underbody, interior -> satin_trim.
- emission_hex: only for lamp_emissive tail/brake lights or coloured lights (e.g. "#FF1A0A"); null otherwise.
- keep: only when the material is textured and the CURRENT LOOK close-up already matches the references
  (e.g. printed livery, decals, lettering, a real carbon weave texture); replacing it would lose that detail.
- confidence 0..1: how sure you are about preset AND colour (low when the part is not visible on any reference).
Return exactly one entry per card with `row` set to the card's row number."""


def _caption(m: dict[str, Any]) -> str:
    vals = [f"base {m.get('base_hex')}" + (" (textured)" if m.get("base_textured") else "")]
    for key in ("metallic", "roughness", "alpha", "transmission"):
        v = m.get(key)
        if v is not None:
            vals.append(f"{key} {v}")
    if m.get("emission_strength"):
        vals.append(f"emission {m.get('emission_hex')} x{m['emission_strength']}")
    if m.get("texture") and not m.get("base_textured"):
        vals.append("has textures")
    if m.get("see_through"):
        vals.append("see-through")
    return (f"Card row #{int(m['index']):02d} (material {m['name']!r}, {float(m.get('area_pct', 0)):.2f}% of the "
            f"visible car); current: {', '.join(vals)}:")


def _chunks(items: list[Any], max_per_call: int) -> list[list[Any]]:
    """Split into the fewest calls of at most `max_per_call`, balanced."""
    if not items:
        return []
    n = (len(items) + max_per_call - 1) // max_per_call
    size = (len(items) + n - 1) // n
    return [items[i:i + size] for i in range(0, len(items), size)]


def resolve_answers(materials: list[dict[str, Any]], answers: list[MaterialAnswer]) -> tuple[list[MaterialVerdict], list[str]]:
    """Model answers -> verdicts in the order of `materials` (first answer per row wins);
    colours normalised, invalid ones dropped with a note; unanswered rows reported."""
    by_row = {int(m["index"]): m for m in materials}
    got: dict[int, MaterialVerdict] = {}
    problems: list[str] = []
    for a in answers:
        m = by_row.get(a.row)
        if m is None:
            problems.append(f"answer for unknown row {a.row}")
            continue
        if a.row in got:
            continue
        colours: dict[str, str | None] = {}
        for key, raw in (("base", a.base_hex), ("emission", a.emission_hex)):
            colours[key] = None
            if raw:
                try:
                    colours[key] = normalise_hex(raw)
                except ValueError as ex:
                    problems.append(f"{m['name']}: {ex}")
        got[a.row] = MaterialVerdict(
            name=str(m["name"]), row=a.row, preset=a.preset, base=colours["base"], emission=colours["emission"],
            part=a.part, seen=a.seen_in_reference, confidence=round(float(np.clip(a.confidence, 0.0, 1.0)), 3),
            area_pct=float(m.get("area_pct", 0.0)))
    out = []
    for m in materials:
        v = got.get(int(m["index"]))
        if v is None:
            problems.append(f"{m['name']}: no answer from the model")
        else:
            out.append(v)
    return out, problems


async def audit_materials(sheets: list[bytes], materials: list[dict[str, Any]], references: list[bytes],
                          car_description: str = "", *, vlm: VisionModel | None = None,
                          max_per_call: int = MAX_PER_CALL) -> AuditResult:
    """Preset (+ colour) for every material shown on the sheets, judged against `references`.

    `sheets`: render_material_sheet PNGs (index = materials[i]["sheet"]); `materials`: the
    `materials` list of its materials.json; `references`: photos of the real car (any size; they are
    downscaled). One VLM call per `max_per_call` materials (references repeated in each)."""
    if not materials:
        return AuditResult(materials=[])
    model = vlm or Vlm(max_side=MAX_SIDE)
    usage_rec = getattr(getattr(model, "usage", None), "per_call", None)
    first_rec = len(usage_rec) if usage_rec is not None else 0
    refs = [downscale_jpeg(r, MAX_SIDE) for r in references]
    cards = material_cards(sheets, materials)
    preset_lines = "\n".join(f"- {k}: {v}" for k, v in PRESETS.items())
    views = [str(c.get("view", "")) for c in materials[0].get("cells", [])]
    panels = ", ".join(VIEW_WORDS.get(v, v) for v in views)
    prompt = AUDIT_PROMPT.format(car=f" ({car_description})" if car_description else "", n_ref=len(refs),
                                 panels=panels, presets=preset_lines,
                                 colour_presets=", ".join(sorted(COLOUR_PRESETS)))
    ref_images = [(f"Reference photo {i + 1} of the real car:", r) for i, r in enumerate(refs)]
    pairs = list(zip(materials, cards))
    answers: list[MaterialAnswer] = []
    liveries: list[str] = []
    for k, chunk in enumerate(_chunks(pairs, max(1, max_per_call))):
        images = ref_images + [(_caption(m), card) for m, card in chunk]
        res = await model.parse(prompt, images, AuditAnswer, label=f"audit_materials[{k}]")
        answers += res.materials
        if res.livery:
            liveries.append(res.livery)
    verdicts, problems = resolve_answers(materials, answers)
    usage: dict[str, Any] = {}
    if usage_rec is not None:  # this audit's calls only (the model object may be shared)
        calls = usage_rec[first_rec:]
        usage = {"calls": len(calls), "per_call": calls,
                 **{k: sum(c[k] for c in calls) for k in ("input_tokens", "cached_tokens", "output_tokens")},
                 "cost_usd": round(sum(c["cost_usd"] for c in calls), 5)}
    return AuditResult(materials=verdicts, livery=liveries[0] if liveries else "", problems=problems, usage=usage)
