"""MCP tool `viewport_snapshot`."""

__all__ = ("register",)

from blmcp.tools.ts_viewport_snapshot_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, call, presign_outputs
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Viewport snapshot", readOnlyHint=True))
    async def viewport_snapshot(job_prefix: str, index: int = 0) -> dict[str, object]:
        """Screenshot of the largest 3D viewport, uploaded as `snapshot_<index>.png` under
        `job_prefix`. Returns {status, key, width, height}."""
        keys, uploads = presign_outputs(job_prefix, {"snapshot": (f"snapshot_{index}.png", "image/png")})
        result = await call(TOOL.format(Params(uploads=uploads)))
        if result.get("status") == "ok":
            result["key"] = keys["snapshot"]
        return result
