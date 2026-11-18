"""MCP tool `load_car`."""

__all__ = ("register",)

import asyncio
from typing import Literal

from blmcp.tools.ts_load_car_toolcode import FORWARD_AXES, Params
from blmcp.tools_helpers.trueshot import ToolCode, pace, presign_get, run_job, s3_head
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Load car", destructiveHint=True))
    async def load_car(car_key: str, forward_axis: Literal[FORWARD_AXES] = "auto", length_m: float = 0.0) -> dict[str, object]:
        """Import the car glTF/GLB at S3 key `car_key` (e.g. "cars/<id>.glb") into TS_CAR under
        TS_CAR_ROOT, replacing any previous car (its placement is kept). Drops helper meshes that
        never render (all materials fully transparent, e.g. backdrop boxes), then normalises scale
        (to `length_m`, the real overall length in metres, when > 0; else a unit guess) and
        orientation (front = root +Y; "auto" guesses the front from lamp/grille names), puts the
        origin at the tyre contact centre, classifies roles, makes glass transmissive and caches a
        library .blend so the next load of the same key is fast.
        Returns {status, car_key, dims_m [length, width, height], wheels, wheel_radius_m,
        paint_materials, materials [{name, kind}], helpers_removed, forward_axis_used,
        scale_applied, tri_count, import_s, from_cache}."""
        head = await asyncio.to_thread(s3_head, car_key)
        if head is None:
            return {"status": "error", "message": f"S3 object not found: {car_key}"}
        url = presign_get(car_key)

        def make_code(action: str) -> str:
            return TOOL.format(Params(action=action, car_key=car_key, url=url, etag=head["etag"],
                                      forward_axis=forward_axis, length_m=float(length_m)))

        result = await run_job(make_code, poll_s=0.5, timeout_s=250.0)
        await pace()
        return result
