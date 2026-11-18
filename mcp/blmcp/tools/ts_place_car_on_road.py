"""MCP tool `place_car_on_road`."""

__all__ = ("register",)

from typing import Literal

from blmcp.tools.ts_place_car_on_road_toolcode import LANES, Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Place car on road", idempotentHint=True))
    async def place_car_on_road(
        distance_m: float = 0.0,
        lane: Literal[LANES] = "right",
        heading_offset_deg: float = 0.0,
    ) -> dict[str, object]:
        """Put the car on the road centreline at arc length `distance_m` (0 = world origin,
        negative = behind, up to ~400 ahead), offset by lane (+-width/4), facing the direction
        of travel plus `heading_offset_deg` (positive = clockwise seen from above).
        Returns {status, location, heading_deg (compass: 0 = +Y, 90 = +X)}."""
        result = await call(TOOL.format(Params(distance_m=float(distance_m), lane=lane,
                                               heading_offset_deg=float(heading_offset_deg))))
        await pace()
        return result
