"""MCP tool `render_pose_sheet`."""

__all__ = ("register",)

from blmcp.tools.ts_render_pose_sheet_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, run_render_job
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)
MAX_CELLS = 72


def _angles(text: str) -> tuple[float, ...]:
    return tuple(float(s) for s in str(text).replace(";", ",").split(",") if s.strip())


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Render pose sheet", readOnlyHint=False))
    async def render_pose_sheet(
        job_prefix: str,
        azimuths_deg: str = "0,30,60,90,120,150,180,210,240,270,300,330",
        elevations_deg: str = "3,15",
        focal_mm: float = 85.0,
        resolution: int = 320,
    ) -> dict[str, object]:
        """Contact sheet of the car alone (grey Workbench, studio light, white background, set
        hidden) seen from every elevation x azimuth (comma-separated degrees, frame_shot
        convention: azimuth 0 = looking at the car's front, 90 = its left side, 180 = behind,
        270 = right side), each cell `resolution` px wide (3:2) and labelled 'az 30 / el 3'.
        Uploads `pose_sheet.png` and `pose_sheet.json` (cell rectangles and angles) under
        `job_prefix`; the scene camera and render settings are left unchanged.
        Returns {status, keys {sheet, index}, sheet_size, cell_size, focal_mm, cells
        [{index, azimuth_deg, elevation_deg, label, x, y, w, h}], total_s}."""
        try:
            azimuths, elevations = _angles(azimuths_deg), _angles(elevations_deg)
        except ValueError as ex:
            return {"status": "error", "message": f"bad angle list: {ex}"}
        if not azimuths or not elevations:
            return {"status": "error", "message": "azimuths_deg and elevations_deg must list at least one angle"}
        if len(azimuths) * len(elevations) > MAX_CELLS:
            return {"status": "error", "message": f"at most {MAX_CELLS} cells per sheet"}
        outputs = {"sheet": ("pose_sheet.png", "image/png"), "index": ("pose_sheet.json", "application/json")}
        return await run_render_job(
            TOOL, Params, job_prefix, outputs, azimuths=azimuths, elevations=elevations,
            focal_mm=float(focal_mm), resolution=int(resolution))
