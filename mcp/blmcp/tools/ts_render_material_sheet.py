"""MCP tool `render_material_sheet`."""

__all__ = ("register",)

import math

from blmcp.tools.ts_render_material_sheet_toolcode import VIEWS, Params
from blmcp.tools_helpers.trueshot import ToolCode, run_render_job
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)
ROWS_PER_SHEET = 12
MAX_MATERIALS = 120

DESCRIPTION = (
    "Which part of the car is each material? Contact sheets for a material audit: one row per car "
    "material (largest visible area first; materials covering < 0.05% of the car's pixels are skipped) "
    "showing it orange on the grey car (Workbench, set hidden) from each view in `views` (comma-separated: "
    + ", ".join(VIEWS) + "), each cell `cell_px` wide (3:2), then a close-up of its largest area in the car's "
    "CURRENT colours/textures outlined in orange, under a label '#<row> <material> <area %>'; highlights too "
    "small to spot get a magenta box. Uploads `material_sheet_<k>.png` (up to 12 rows each) "
    "and `materials.json` (per material: name, area_pct, view_pct, current base_hex / metallic / roughness / "
    "alpha / transmission / emission, texture, see_through, sheet index and row/cell rectangles) under "
    "`job_prefix`. Scene and materials are left unchanged. Returns {status, keys {sheet_<k>, index}, sheets, "
    "materials, skipped, material_names, cell_size, views, pass_s, total_s}."
)


def register(mcp: FastMCP) -> None:
    @mcp.tool(description=DESCRIPTION, annotations=ToolAnnotations(title="Render material sheet", readOnlyHint=False))
    async def render_material_sheet(
        job_prefix: str,
        views: str = "front34,rear34,side",
        max_materials: int = 40,
        cell_px: int = 320,
    ) -> dict[str, object]:
        names = tuple(v.strip() for v in str(views).replace(";", ",").split(",") if v.strip())
        unknown = [v for v in names if v not in VIEWS]
        if not names or unknown:
            return {"status": "error", "message": f"views must be some of {sorted(VIEWS)}, got {list(names)}"}
        max_materials = max(1, min(MAX_MATERIALS, int(max_materials)))
        outputs = {f"sheet_{k}": (f"material_sheet_{k}.png", "image/png")
                   for k in range(math.ceil(max_materials / ROWS_PER_SHEET))}
        outputs["index"] = ("materials.json", "application/json")
        result = await run_render_job(
            TOOL, Params, job_prefix, outputs, views=names, max_materials=max_materials,
            cell_px=int(cell_px), rows_per_sheet=ROWS_PER_SHEET)
        if result.get("status") == "ok":
            uploaded = result.get("uploaded", {})
            result["keys"] = {k: v for k, v in result["keys"].items() if k in uploaded}
        return result
