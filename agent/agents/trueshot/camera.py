"""Camera and exposure helpers shared by the scene setup, the car render and the compositor.

The planner places the camera freely (`schemas.CameraPlan`); nothing here picks a shot.
Perspective distortion depends on camera distance, not focal length: whole-car shots should keep
the camera at least MIN_CAMERA_DISTANCE_M from the car (setup warns in the feed otherwise);
detail shots go close on purpose.
"""

from __future__ import annotations

from typing import Any

from .schemas import CameraPlan, MotionPlan

MIN_CAMERA_DISTANCE_M = 7.0
# The car is the product: its own travel through the frame is capped at MAX_BODY_SLIP_M whatever
# the shutter, so badges and spokes stay readable; the camera's tracked share streaks the plate.
MAX_BODY_SLIP_M = 0.02


def camera_side(azimuth_deg: float) -> str:
    """The side of the road the camera stands on (seen driving away from the start): azimuth
    0-180 = the car's left side."""
    return "left" if 0.0 < azimuth_deg % 360.0 < 180.0 else "right"


def lane(placement_lane: str, azimuth_deg: float) -> str:
    """place_car_on_road lane for the plan's lane: near = the lane on the camera's side (road
    between the car and the far edge), far = the other one, center = the centre line."""
    if placement_lane == "center":
        return "center"
    side = camera_side(azimuth_deg)
    if placement_lane == "far":
        return "right" if side == "left" else "left"
    return side


def frame_args(camera: CameraPlan) -> dict[str, Any]:
    """frame_shot arguments: the plan's camera minus the lens aperture (used by the renders)."""
    return camera.model_dump(exclude={"fstop"})


def exposure(camera: CameraPlan, motion: MotionPlan) -> dict[str, float]:
    """One physical exposure for the whole still: {speed_kmh, shutter_s, tracking, fstop}. The car
    travels speed x shutter during the exposure; `tracking` of that is the camera's travel (the
    plate streaks by depth), the rest is the car moving through the frame (body blur); the
    wheels spin at speed. Speed 0 or blur off freezes the exposure and keeps the depth of field."""
    exp = {"speed_kmh": motion.speed_kmh, "shutter_s": motion.shutter_s, "tracking": motion.tracking,
           "fstop": camera.fstop}
    if not motion.blur or exp["speed_kmh"] <= 0:
        exp.update(speed_kmh=0.0, shutter_s=0.0)
    return exp


def car_motion_args(exp: dict[str, float]) -> dict[str, Any]:
    """render_car arguments for `exposure()`: wheel spin, shutter, body travel, depth of field."""
    travel = exp["speed_kmh"] / 3.6 * exp["shutter_s"]
    slip = min((1.0 - exp["tracking"]) * travel, MAX_BODY_SLIP_M)
    return {"fstop": exp["fstop"], "speed_kmh": exp["speed_kmh"], "shutter_s": exp["shutter_s"],
            "body_slip_m": round(slip, 4), "motion_blur": travel > 0}
