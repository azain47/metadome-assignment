"""MCP tool `render_guides`."""

__all__ = ("register",)

from blmcp.tools.ts_render_guides_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, run_render_job
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)

OUTPUTS = {
    "grey": ("guide_grey.png", "image/png"),
    "depth": ("depth.exr", "image/x-exr"),
    "road_mask": ("mask_road.png", "image/png"),
    "car_mask": ("mask_car.png", "image/png"),
    "layout": ("layout.json", "application/json"),
}


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Render guides", readOnlyHint=False))
    async def render_guides(job_prefix: str) -> dict[str, object]:
        """Render the car-free guides through TS_CAM and upload them under `job_prefix`
        (`jobs/<job_id>/a<n>/`): guide_grey.png (flat grey tones by role), mask_road.png,
        mask_car.png (car silhouette), depth.exr (metres, float32), layout.json.
        Returns {status, keys: {grey, depth, road_mask, car_mask, layout}, layout: {width, height,
        horizon_y_px, road_left_px, road_right_px, tyre_contacts_px, wheels, car_bbox_px,
        car_forward_dir_px, camera, sun}, depth_channels}."""
        return await run_render_job(TOOL, Params, job_prefix, OUTPUTS)
