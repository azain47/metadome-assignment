"""Grounded livery options: research a car's real paint colours on the web, turn them into liveries.

One OpenAI Responses call with the built-in web search tool and a structured answer: every colour
must be one a real car of this model was delivered or shown in, with its source. Each colour
becomes a livery over the product facts' material groups (body, accents, calipers), so the user
picks between real options rendered on the exact car instead of invented ones.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.references import Vlm, default_vlm

log = logging.getLogger(__name__)

Finish = Literal["solid", "metallic", "pearl", "satin", "matte"]


class ResearchedColour(BaseModel):
    name: str = Field(description="official paint or livery name, e.g. the manufacturer's colour name")
    finish: Finish
    hex: str = Field(description="approximate sRGB #RRGGBB of the paint as it looks in daylight photos")
    accent_hex: str = Field(description="colour of accents/stripes on that car (#RRGGBB); exposed carbon = #121316")
    caliper_hex: str = Field(description="brake caliper colour (#RRGGBB) if documented, else #0A0A0A")
    source_url: str = Field(description="page that documents a real car in this colour")
    evidence: str = Field(description="one sentence from or about the source")


class LiveryResearch(BaseModel):
    colours: list[ResearchedColour]


_FINISH = {
    "solid": {"preset": "car_paint", "roughness": 0.4},
    "metallic": {"preset": "metallic_paint", "metallic": 0.5, "roughness": 0.32, "coat_roughness": 0.02},
    "pearl": {"preset": "pearl_paint"},
    "satin": {"preset": "metallic_paint", "metallic": 0.6, "roughness": 0.38, "coat_roughness": 0.18},
    "matte": {"preset": "matte_paint"},
}


def _hex(value: str, fallback: str) -> str:
    v = value.strip().upper()
    v = v if v.startswith("#") else "#" + v
    return v if re.fullmatch(r"#[0-9A-F]{6}", v) else fallback


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "livery"


def colour_to_livery(c: ResearchedColour) -> tuple[str, dict[str, Any]]:
    """Researched colour -> (livery name, livery spec over the body/accents/calipers groups)."""
    body = {**_FINISH[c.finish], "base": _hex(c.hex, "#808080")}
    accent_hex = _hex(c.accent_hex, "#0B0C0E")
    accents = {"preset": "carbon_fibre"} if accent_hex == "#121316" else {"preset": "car_paint", "base": accent_hex}
    return _slug(c.name), {
        "description": f"{c.name} ({c.finish}) body, accents {accent_hex}, calipers {_hex(c.caliper_hex, '#0A0A0A')}. "
                       f"Source: {c.source_url}",
        "source_url": c.source_url,
        "evidence": c.evidence,
        "groups": {"body": body, "accents": accents,
                   "calipers": {"preset": "car_paint", "base": _hex(c.caliper_hex, "#0A0A0A"), "roughness": 0.3}},
    }


async def research_liveries(car_name: str, n: int = 4, *, vlm: Vlm | None = None) -> dict[str, dict[str, Any]]:
    """Up to `n` real, sourced colour schemes for `car_name` as liveries {name: spec}."""
    v = vlm or default_vlm()
    prompt = (
        f"Find {n} distinct real paint colours or liveries of the {car_name}, preferring its most iconic ones "
        "(official factory or bespoke-division colours, launch cars, well-documented customer cars). For each: "
        "the official colour name, finish (solid / metallic / pearl / satin / matte), an approximate sRGB hex of how "
        "the paint looks in daylight photos, the colour of its accents or stripes (use #121316 when they are bare "
        "carbon fibre), the brake caliper colour if documented, and the URL of a manufacturer page, press release, "
        "auction listing or reputable magazine that shows a real car in that colour. Never invent a colour."
    )
    resp = await v.client.responses.parse(model=v.model, input=prompt, tools=[{"type": "web_search"}],
                                          text_format=LiveryResearch, reasoning={"effort": "low"})
    v.usage.add("research_liveries", getattr(resp, "usage", None))
    found = resp.output_parsed.colours if resp.output_parsed else []
    log.info("researched %d colours for %s", len(found), car_name)
    return dict(colour_to_livery(c) for c in found[:n])
