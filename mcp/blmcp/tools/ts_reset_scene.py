"""MCP tool `reset_scene`."""

__all__ = ("register",)

from blmcp.tools.ts_reset_scene_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Reset scene", destructiveHint=True))
    async def reset_scene(keep_car: bool = True) -> dict[str, object]:
        """Delete the set (TS_SET) and rig (TS_RIG: camera, sun), and the car too unless
        keep_car; purges orphan data. Never loads a file. Returns {status, removed}."""
        result = await call(TOOL.format(Params(keep_car=bool(keep_car))))
        await pace()
        return result
