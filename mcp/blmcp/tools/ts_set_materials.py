"""MCP tool `set_materials`."""

__all__ = ("register",)

import json

from blmcp.tools.ts_set_materials_toolcode import PRESETS, Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)

DESCRIPTION = (
    "Give car materials physically based looks. `assignments` maps a material name (see "
    "load_car's `materials`; key \"auto\" = the detected paint materials) to "
    "{\"preset\": <name>, <override>: value}. Presets: " + ", ".join(sorted(PRESETS)) + ". "
    "Overrides: base / base2 / emission (sRGB \"#RRGGBB\"), metallic, roughness, specular (base layer), coat, coat_roughness, "
    "transmission, ior, emission_strength, alpha, weave_mm. Every preset but \"keep\" defines the whole "
    "material and replaces textures on the inputs it sets; \"keep\" changes only the overrides given. "
    "Example: {\"auto\": {\"preset\": \"car_paint\", \"base\": \"#0E0F11\"}, \".015\": {\"preset\": "
    "\"rubber_tyre\"}}. Returns {status, changed, skipped_no_principled}."
)


def register(mcp: FastMCP) -> None:
    @mcp.tool(description=DESCRIPTION, annotations=ToolAnnotations(title="Set materials", idempotentHint=True))
    async def set_materials(assignments: dict[str, dict[str, object]]) -> dict[str, object]:
        result = await call(TOOL.format(Params(assignments_json=json.dumps(assignments))))
        await pace()
        return result
