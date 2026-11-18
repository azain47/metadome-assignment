"""MCP tool `save_blend`."""

__all__ = ("register",)

import re

from blmcp.tools.ts_save_blend_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace, presign_outputs
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Save .blend", readOnlyHint=True))
    async def save_blend(job_prefix: str, name: str = "scene") -> dict[str, object]:
        """Save a compressed copy of the current Blender scene as `<name>.blend` under
        `job_prefix` in object storage, keeping a local copy on the Blender machine. The open file
        is left as it is. Call it after building a scene and after the final render, so a shot can
        be reopened or re-run if Blender crashes. Returns {status, key, local_path, bytes}."""
        filename = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "scene"
        filename += ".blend"
        keys, uploads = presign_outputs(job_prefix, {"blend": (filename, "application/x-blender")})
        result = await call(TOOL.format(Params(filename=job_prefix.strip("/").replace("/", "_") + "_" + filename,
                                               uploads=uploads)))
        if result.get("status") == "ok":
            result["key"] = keys["blend"]
        await pace()
        return result
