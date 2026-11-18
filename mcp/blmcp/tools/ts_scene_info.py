"""MCP tool `scene_info`."""

__all__ = ("register",)

from blmcp.tools.ts_scene_info_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, call
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Scene info", readOnlyHint=True))
    async def scene_info() -> dict[str, object]:
        """Summarise the Trueshot scene: {status, car: {key, dims_m, wheels, wheel_radius_m, ...} | null,
        set: {terrain, road_width_m, road_length_m, ...} | null, camera | null, sun | null, world,
        is_rendering}."""
        return await call(TOOL.format(Params()))
