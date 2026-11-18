"""MCP tool `frame_shot`."""

__all__ = ("register",)

from blmcp.tools.ts_frame_shot_toolcode import Params
from blmcp.tools_helpers.trueshot import ToolCode, call, pace
from mcp.server.fastmcp import FastMCP  # pylint: disable=import-error,no-name-in-module
from mcp.types import ToolAnnotations  # pylint: disable=import-error,no-name-in-module

TOOL = ToolCode(__file__)


def register(mcp: FastMCP) -> None:
    @mcp.tool(annotations=ToolAnnotations(title="Frame shot", idempotentHint=True))
    async def frame_shot(
        azimuth_deg: float = 35.0,
        elevation_deg: float = 6.0,
        focal_mm: float = 50.0,
        distance_m: float = 0.0,
        car_width_frac: float = 0.55,
        target_height_m: float = 0.7,
        frame_x: float = 0.5,
        frame_y: float = 0.62,
        roll_deg: float = 0.0,
        resolution_x: int = 1536,
        resolution_y: int = 1024,
        aim_x: float = 0.0,
        aim_y: float = 0.0,
        aim_z: float = 0.0,
        subject_width_m: float = 0.0,
    ) -> dict[str, object]:
        """Create/update the shot camera TS_CAM and make it the scene camera.
        azimuth: 0 = in front looking at the car's front, 90 = car's left side, 180 = behind,
        270 = right side; elevation in degrees above horizontal at the target (clamped to
        +/-89.5; 60-89 = aerial looking down). subject_width_m <= 0 frames the whole car: the
        target is the car root + target_height_m and distance_m = 0 solves the distance so the
        car's projected bbox width is car_width_frac of the frame. subject_width_m > 0 frames a
        detail: the target is the aim point in car-normalised coordinates of the car's bbox
        (aim_x -1 = car's left side .. +1 right side, aim_y -1 = tail .. +1 = nose, aim_z 0 =
        bottom .. 1 = roof) and car_width_frac is the frame width filled by a cube of side
        subject_width_m centred on it (minimum camera distance 0.3 m). Lens shift puts the
        subject's centre at (frame_x, frame_y) (fractions from left/top). Returns {status, camera
        {location, rotation_deg, lens_mm, shift_x, shift_y, distance_m}, car_bbox_px (whole car),
        car_bbox_frac, all_in_frame (whole car), horizon_y_frac, subject ("car" or {aim, point,
        width_m, bbox_px, in_frame})}."""
        result = await call(TOOL.format(Params(
            azimuth_deg=float(azimuth_deg), elevation_deg=float(elevation_deg), focal_mm=float(focal_mm),
            distance_m=float(distance_m), car_width_frac=float(car_width_frac),
            target_height_m=float(target_height_m), frame_x=float(frame_x), frame_y=float(frame_y),
            roll_deg=float(roll_deg), resolution_x=int(resolution_x), resolution_y=int(resolution_y),
            aim_x=float(aim_x), aim_y=float(aim_y), aim_z=float(aim_z),
            subject_width_m=float(subject_width_m))))
        await pace()
        return result
