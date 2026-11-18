"""MCP tool `build_set`."""

__all__ = ("register",)

import json
from typing import Any, Literal

from blmcp.tools.ts_build_set_toolcode import ROAD_SHAPES, TERRAINS, Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)

DESCRIPTION = (
    "Replace the stand-in set (TS_SET): a road from 60 m behind the origin (travel along +Y at the "
    "origin), a ground plane and grey stand-ins that read clearly in a 1536x1024 frame. "
    "`road`: segments after that first 60 m, in order: {\"straight_m\": 1-1000} or {\"turn_deg\": "
    "-270..270 (+ = left), \"radius_m\": 6-2000}; the road is filled with a straight up to 460 m total. "
    "Non-empty `road` overrides `road_shape` (\"\" = straight, or " + ", ".join(ROAD_SHAPES) + "). "
    "`elements`: stand-ins applied after the `terrain` preset's own (\"\" = none, or "
    + ", ".join(TERRAINS) + "); ranges are [lo, hi] or one number: "
    "hillside {side, grade 0.05-1.5, height_deg 0.1-45}; "
    "mountains {side, count 1-60, distance_m 100-20000 (from the origin), height_deg 0.1-30, width_deg 2-120, "
    "jagged 0-1, flat_top 0-0.6, steepness 0.5-6, road_clearance_m 0-1000}; "
    "buildings {side, count 1-3000, distance_m 0-3000 (from the road edge), size_m 2-200, height_m 1-500, "
    "height_deg 0.1-45}; trees {side, count 1-20000, distance_m 0-3000, height_m 2-80, height_deg 0.1-45}; "
    "sea {side (not both), distance_m 0-2000 shore margin}; wall {side, distance_m 0-20 from the road edge, "
    "height_m 0.3-6, thickness_m 0.1-3}. height_deg = angle above the horizon seen from the origin at 1.5 m. "
    "The sea is applied first; buildings, trees and walls stand on hillsides listed before them. "
    "side: near (the `camera_side` of the road), far, left, right, both; left/right of the direction of travel "
    "at the nearest road point. `seed` varies the layout. Writes scene props ts_road_centerline, "
    "ts_road_width_m, ts_set_terrain. Returns {status, terrain, road_shape, camera_side, road_length_m, "
    "elements [{kind, side, resolved, placed, source}], objects}."
)


def register(mcp: FastMCP) -> None:
    @mcp.tool(description=DESCRIPTION, annotations=ToolAnnotations(title="Build set", destructiveHint=True))
    async def build_set(
        road: list[dict[str, Any]] = [],  # noqa: B006  pylint: disable=dangerous-default-value
        elements: list[dict[str, Any]] = [],  # noqa: B006  pylint: disable=dangerous-default-value
        road_shape: str = "",
        terrain: str = "",
        camera_side: Literal["left", "right"] = "left",
        road_width_m: float = 7.0,
        seed: int = 0,
    ) -> dict[str, object]:
        result = await call(TOOL.format(Params(
            road_json=json.dumps(road), elements_json=json.dumps(elements), road_shape=road_shape,
            terrain=terrain, camera_side=camera_side, road_width_m=float(road_width_m), seed=int(seed))))
        await pace()
        return result
