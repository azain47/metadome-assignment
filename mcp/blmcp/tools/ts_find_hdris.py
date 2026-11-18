"""MCP tool `find_hdris`: search Poly Haven's whole HDRI catalogue (runs in the server, not Blender)."""

__all__ = ("register",)

import re
import time

import httpx
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

_API = "https://api.polyhaven.com"
_CATALOG_TTL_S = 6 * 3600
_cache: dict = {"t": 0.0, "assets": {}}

# Descriptive words -> Poly Haven tag / category vocabulary (time of day, sky, surroundings).
_SYNONYMS = {
    "golden": ("sunrise-sunset",), "sunset": ("sunrise-sunset",), "sunrise": ("sunrise-sunset",),
    "dusk": ("sunrise-sunset",), "dawn": ("sunrise-sunset",), "noon": ("midday",), "midday": ("midday",),
    "morning": ("morning-afternoon",), "afternoon": ("morning-afternoon",), "night": ("night",),
    "clear": ("clear",), "sunny": ("clear", "high contrast"), "cloudy": ("partly cloudy", "overcast"),
    "overcast": ("overcast",), "alpine": ("mountain", "mountains", "alps"), "mountains": ("mountain",),
    "coastal": ("coast", "sea", "beach"), "seaside": ("coast", "sea", "beach"), "sea": ("coast", "beach"),
    "harbour": ("harbor", "coast", "town"), "town": ("urban", "city", "village"), "city": ("urban",),
    "desert": ("arid", "dry"), "arid": ("desert", "dry"), "barren": ("arid", "rocky"), "road": ("asphalt", "street"),
    "meadow": ("grass", "field", "green"), "forest": ("trees", "woods"), "snow": ("snowy", "winter"),
}


def _words(text: str) -> list[str]:
    return [w for w in re.split(r"[^a-z0-9-]+", text.lower()) if len(w) > 1]


async def _catalog() -> dict:
    if not _cache["assets"] or time.time() - _cache["t"] > _CATALOG_TTL_S:
        async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "trueshot"}) as http:
            r = await http.get(f"{_API}/assets", params={"type": "hdris"})
            r.raise_for_status()
            _cache.update(t=time.time(), assets=r.json())
    return _cache["assets"]


# Poly Haven's time-of-day categories; an asset only in another one contradicts the query.
_TIMES = ("sunrise-sunset", "midday", "morning-afternoon", "night")


def _score(query: list[str], asset_id: str, a: dict) -> float:
    tags = {t.lower() for t in a.get("tags", [])}
    cats = {c.lower() for c in a.get("categories", [])}
    vocab = tags | cats | set(_words(a.get("name", "") + " " + asset_id.replace("_", " ")))
    score = 0.0
    for w in query:
        if w in vocab:
            score += 2.0
        elif any(s in vocab for s in _SYNONYMS.get(w, ())):
            score += 1.5
        elif len(w) > 3 and any(v.startswith(w) for v in vocab):
            score += 0.25
    wanted = {t for w in query for t in (w, *_SYNONYMS.get(w, ())) if t in _TIMES}
    has = cats & set(_TIMES)
    if wanted and has and not (wanted & has):
        score -= 3.0
    if "indoor" in cats and "indoor" not in query:
        score -= 5.0
    return score


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Find HDRIs", readOnlyHint=True, openWorldHint=True))
    async def find_hdris(query: str, limit: int = 8) -> dict[str, object]:
        """Search Poly Haven's full HDRI catalogue (CC0) for environments that look like a shot's
        surroundings: describe the place and time of day in plain words, e.g. "alpine mountain
        meadow road golden hour" or "mediterranean coastal town sea midday". Matches names, tags
        and categories (time-of-day words map to Poly Haven's categories), ties broken by
        popularity. Pass the chosen id to set_lighting(polyhaven_id=...). Returns {status,
        results: [{id, name, score, tags, categories, evs_cap, thumbnail}]}."""
        try:
            assets = await _catalog()
        except httpx.HTTPError as ex:
            return {"status": "error", "message": f"Poly Haven unreachable: {ex}"}
        words = _words(query)
        ranked = sorted(assets.items(), key=lambda kv: (-_score(words, kv[0], kv[1]),
                                                        -int(kv[1].get("download_count", 0))))
        results = [{"id": aid, "name": a.get("name", aid), "score": round(_score(words, aid, a), 2),
                    "tags": a.get("tags", []), "categories": a.get("categories", []), "evs_cap": a.get("evs_cap"),
                    "thumbnail": f"https://cdn.polyhaven.com/asset_img/thumbs/{aid}.png?width=512"}
                   for aid, a in ranked[:max(1, min(int(limit), 30))]]
        return {"status": "ok", "query": query, "results": results}
