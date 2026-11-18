"""MCP tool `set_lighting`."""

__all__ = ("register",)

import asyncio
import re
from typing import Literal

import httpx
from blmcp.tools.ts_set_lighting_toolcode import SKIES, Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace, presign_get, s3_head
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)

RESOLUTIONS = ("1k", "2k", "4k", "8k")


async def _polyhaven_url(asset_id: str, resolution: str) -> str:
    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": "trueshot"}) as http:
        r = await http.get(f"https://api.polyhaven.com/files/{asset_id}")
        r.raise_for_status()
        files = r.json()["hdri"]
    res = resolution if resolution in files else max(files, key=lambda k: int(k.rstrip("k")))
    return files[res]["hdr"]["url"]


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Set lighting", idempotentHint=True, openWorldHint=True))
    async def set_lighting(
        sky: Literal[SKIES] = "procedural",
        polyhaven_id: str = "",
        resolution: Literal[RESOLUTIONS] = "4k",
        hdri_key: str = "",
        align_sun_azimuth_deg: float | None = None,
        hdri_rotation_deg: float = 0.0,
        strength: float = 1.0,
        sun_azimuth_deg: float = 135.0,
        sun_elevation_deg: float = 35.0,
        sun_strength: float = 3.0,
        sun_angle_deg: float = 1.0,
        exposure: float = 0.0,
    ) -> dict[str, object]:
        """World lighting: Blender's procedural sky (multiple scattering, sun disc off), or an HDRI
        environment texture that lights the car and fills its reflections. HDRI source: a Poly
        Haven asset (`polyhaven_id` from find_hdris, downloaded at `resolution` and cached on the
        Blender machine) or a bundled file in object storage (`hdri_key` = "hdri/<file>").
        `align_sun_azimuth_deg` rotates the HDRI so its measured sun (brightest region) sits at that
        compass azimuth (0 = +Y, 90 = +X); otherwise `hdri_rotation_deg` (counter-clockwise from
        above) is used. TS_SUN: sun_strength 0 hides it (an HDRI with a visible sun lights alone);
        when aligning, it follows the HDRI's sun. Returns {status, sky, sun {azimuth_deg,
        elevation_deg}, hdri_key, hdri_rotation_deg, hdri_sun {azimuth_deg (unrotated),
        elevation_deg, contrast (< ~50 = no usable sun)}}."""
        url = cache_name = key = ""
        if sky == "hdri":
            if polyhaven_id:
                try:
                    url = await _polyhaven_url(polyhaven_id, resolution)
                except (httpx.HTTPError, KeyError, ValueError) as ex:
                    return {"status": "error", "message": f"Poly Haven HDRI {polyhaven_id!r} unavailable: {ex}"}
                key = f"polyhaven:{polyhaven_id}"
                cache_name = re.sub(r"[^A-Za-z0-9._-]+", "_", f"polyhaven_{polyhaven_id}_{resolution}") + ".hdr"
            else:
                key = hdri_key if "/" in hdri_key else f"hdri/{hdri_key}"
                head = await asyncio.to_thread(s3_head, key) if hdri_key else None
                if head is None:
                    return {"status": "error", "message": f"HDRI not found in object storage: {key!r}"}
                url = presign_get(key)
                cache_name = re.sub(r"[^A-Za-z0-9._-]+", "_", key) + "_" + head["etag"] + "_" + key.rsplit("/", 1)[-1]
        result = await call(TOOL.format(Params(
            sky=sky, hdri_key=key, hdri_url=url, hdri_cache_name=cache_name,
            hdri_rotation_deg=float(hdri_rotation_deg), strength=float(strength),
            sun_azimuth_deg=float(sun_azimuth_deg), sun_elevation_deg=float(sun_elevation_deg),
            sun_strength=float(sun_strength), sun_angle_deg=float(sun_angle_deg), exposure=float(exposure),
            align_sun=align_sun_azimuth_deg is not None,
            align_sun_azimuth_deg=float(align_sun_azimuth_deg or 0.0))))
        await pace()
        return result
