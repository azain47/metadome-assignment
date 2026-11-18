"""Pick the HDRI whose surroundings look like the shot's location, by looking at thumbnails.

find_hdris ranks Poly Haven's catalogue by words; words cannot tell an alpine meadow from a golf
fairway. One low-detail vision call over the top candidates' thumbnails decides on what matters
for reflections and ambient light: ground type, terrain, vegetation, sky and the colour of light.
"""

from __future__ import annotations

import copy
from typing import Any

from pydantic import BaseModel, Field

from app.references import VisionModel, _http_bytes, default_vlm, downscale_jpeg

THUMB_SIDE = 384


class HdriChoice(BaseModel):
    number: int = Field(description="1-based number of the best matching environment map")
    why: str = Field(description="one sentence: ground, terrain, sky and light colour that match")


async def pick_hdri(candidates: list[dict[str, Any]], description: str, time_of_day: str, *,
                    vlm: VisionModel | None = None, fetch_bytes=None,
                    reasoning_effort: str | None = None) -> dict[str, Any]:
    """`candidates` = find_hdris results. Returns the chosen candidate with `why`; falls back to
    the top-ranked one when thumbnails or the model are unavailable. `reasoning_effort` overrides
    the model's own effort (Vlm default "low") for this call only."""
    fetch = fetch_bytes or _http_bytes
    images: list[tuple[str, bytes]] = []
    kept: list[dict[str, Any]] = []
    for c in candidates:
        try:
            images.append((c["id"], downscale_jpeg(await fetch(c["thumbnail"]), THUMB_SIDE)))
            kept.append(c)
        except Exception:  # noqa: BLE001 - a missing thumbnail just drops that candidate
            continue
    if len(kept) < 2:
        return {**(kept or candidates)[0], "why": "only candidate with a thumbnail"}
    prompt = (
        "The images are numbered equirectangular HDRI environment maps (1 to {n}, in order). One will "
        "light a car and fill its paint and glass reflections in a photo of this place: {desc} Time of "
        "day: {tod}. Pick the map whose surroundings would look right reflected in the car: ground type "
        "and colour, terrain, vegetation, buildings or sea, sky, and colour, height and hardness of the "
        "sun (if the time of day asks for soft or gentle light, prefer a hazier or partly cloudy sky). "
        "Exact landmarks do not matter. Avoid indoor scenes and maps whose light contradicts the time of day."
    ).format(n=len(kept), desc=description, tod=time_of_day)
    model = vlm or default_vlm()
    if reasoning_effort:
        model = copy.copy(model)  # shares the client and usage meter; the shared default stays as is
        model.reasoning_effort = reasoning_effort
    before = _spent(model)
    try:
        choice = await model.parse(prompt, images, HdriChoice, label="pick_hdri")
    except Exception as ex:  # noqa: BLE001 - fall back to the word ranking
        return {**kept[0], "why": f"vision pick failed ({type(ex).__name__}); top word match",
                "cost_usd": round(_spent(model) - before, 5)}
    index = min(max(choice.number, 1), len(kept)) - 1
    return {**kept[index], "why": choice.why, "cost_usd": round(_spent(model) - before, 5)}


def _spent(model: Any) -> float:
    """Dollars the model has billed so far (0 for models without usage tracking)."""
    return float(getattr(getattr(model, "usage", None), "cost_usd", 0.0) or 0.0)
