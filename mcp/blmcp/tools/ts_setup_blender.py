"""MCP tool `setup_blender`."""

__all__ = ("register",)

from blmcp.tools.ts_setup_blender_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Set up Blender", destructiveHint=False, idempotentHint=True))
    async def setup_blender(use_gpu: bool = True) -> dict[str, object]:
        """Prepare Blender: Cycles GPU device (OptiX, else CUDA, else CPU), metric units,
        the TS_CAR / TS_SET / TS_RIG collections; hides the start-up scene's objects.
        Returns {status, device_type, devices, blender_version}."""
        result = await call(TOOL.format(Params(use_gpu=bool(use_gpu))))
        await pace()
        return result
