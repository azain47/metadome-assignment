"""MCP tool `render_car`."""

__all__ = ("register",)

import asyncio
import re

from blmcp.tools.ts_render_car_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, presign_get, run_render_job, s3_head
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)

OUTPUTS = {
    "exr": ("car.exr", "image/x-exr"),
    "rgba_png": ("car_rgba.png", "image/png"),
}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Render car", readOnlyHint=False))
    async def render_car(
        job_prefix: str,
        samples: int = 128,
        speed_kmh: float = 40.0,
        spoke_count: int = 0,
        motion_blur: bool = True,
        resolution_x: int = 1536,
        resolution_y: int = 1024,
        plate_key: str = "",
        plate_strength: float = 1.0,
        shutter_s: float = 0.0,
        body_slip_m: float = 0.0,
        fstop: float = 0.0,
        probes: bool = False,
    ) -> dict[str, object]:
        """Render the car alone through TS_CAM in Cycles on the GPU: transparent film and glass,
        road/markings/ground as shadow catchers, wheels spinning at `speed_kmh` with motion blur.
        Shutter: `shutter_s` > 0 sets one exposure for wheels and body (use the same value as the
        plate's rolling-shot blur); otherwise half the spoke gap when `spoke_count` > 0, else 0.5
        frames. `body_slip_m` moves the body that far along its heading during the exposure (a
        tracking camera never matches the car exactly). `fstop` > 0 adds depth of field focused on
        the car. With `plate_key` (the S3 key of the generated plate for this camera), the plate is
        camera-projected onto the set so the car's reflections and bounce light show the generated
        scene (`plate_strength` scales it); otherwise stand-ins are hidden and only the world
        reflects. Uploads car.exr (multilayer: premultiplied scene-linear Combined RGBA + Shadow
        Catcher) and car_rgba.png (8-bit straight alpha, sRGB, PBR Neutral view). `probes` adds a
        chrome and an 18% grey reference ball ahead of the nose for this render only: the chrome
        ball shows exactly what the paint reflects, the grey ball the light's direction (use it to
        check the HDRI is caught correctly, never for a final).
        Returns {status, keys: {exr, rgba_png}, exr_channels, render_s, samples, shutter_frames,
        wheels: [{name, center_px, radius_px}], device}."""
        plate = {}
        if plate_key:
            head = await asyncio.to_thread(s3_head, plate_key)
            if head is None:
                return {"status": "error", "message": f"S3 object not found: {plate_key}"}
            plate = {"plate_url": presign_get(plate_key), "plate_strength": float(plate_strength),
                     "plate_cache_name": re.sub(r"[^A-Za-z0-9._-]+", "_", plate_key) + "_" + head["etag"] + ".png"}
        return await run_render_job(
            TOOL, Params, job_prefix, OUTPUTS, samples=samples, speed_kmh=speed_kmh,
            spoke_count=spoke_count, motion_blur=motion_blur,
            resolution_x=resolution_x, resolution_y=resolution_y, shutter_s=float(shutter_s),
            body_slip_m=float(body_slip_m), fstop=float(fstop), probes=bool(probes), **plate)
