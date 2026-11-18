"""MCP tool `render_preview`."""

__all__ = ("register",)

from blmcp.tools.ts_render_preview_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, run_render_job
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Render preview", readOnlyHint=False))
    async def render_preview(job_prefix: str, index: int = 0, resolution_x: int = 768) -> dict[str, object]:
        """Quick look through TS_CAM with the car visible (a few seconds), uploaded as
        `preview_<index>.png` under `job_prefix`. Returns {status, key, width, height, car_bbox_px}."""
        outputs = {"preview": (f"preview_{index}.png", "image/png")}
        result = await run_render_job(TOOL, Params, job_prefix, outputs, resolution_x=resolution_x)
        if result.get("status") == "ok":
            result["key"] = result.pop("keys")["preview"]
        return result
