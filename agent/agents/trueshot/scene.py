"""The shot's scene as code: the Blender calls that set it up from a plan, and the camera
geometry later stages derive from it. Mirrors scripts/dev/lab.py (cmd_scene, cmd_light_apply,
cmd_comp), the reference pipeline; pure functions, so they are unit-tested without Blender.
"""

from typing import Any

from app import lighting
from app.imaging.lens import Intrinsics, Lens, camera_displacement
from app.livery import material_assignments

from .camera import camera_side, frame_args, lane
from .camera import exposure as camera_exposure
from .schemas import LightPlan, SetElement, ShotPlan

HDRI_RESOLUTION = "4k"
HDRI_CANDIDATES = 6  # find_hdris limit; a vision pick over their thumbnails chooses one


def tool_element(element: SetElement) -> dict[str, Any]:
    """A set element as build_set takes it: only the fields the planner set, and every range as a
    number or [lo, hi] (the planner may write one value as [x], or list several)."""
    out: dict[str, Any] = {}
    for key, value in element.model_dump(exclude_none=True).items():
        if isinstance(value, list):
            value = value[0] if len(value) == 1 else [min(value), max(value)] if value else None
        if value is not None:
            out[key] = value
    return out


def setup_ops(plan: ShotPlan, facts: dict[str, Any]) -> list[dict[str, Any]]:
    """The scene-building Blender calls, in order (lighting is derived from their results).
    Replaying them rebuilds the same scene."""
    car_key = str(facts.get("car_key", "")).strip()
    if not car_key:
        raise ValueError("product facts have no car_key")
    cam = plan.camera
    ops: list[dict[str, Any]] = [
        {"tool": "load_car", "args": {"car_key": car_key, "length_m": facts.get("length_m", 0.0)}}]
    assignments = material_assignments(facts, plan.livery, lamps_on=plan.light.lamps_on)
    if assignments:
        ops.append({"tool": "set_materials", "args": {"assignments": assignments}})
    ops += [
        {"tool": "reset_scene", "args": {"keep_car": True}},
        {"tool": "build_set", "args": {
            "road": [s.model_dump(exclude_none=True) for s in plan.set.road],
            "elements": [tool_element(e) for e in plan.set.elements],
            "camera_side": camera_side(cam.azimuth_deg), "road_width_m": plan.set.road_width_m,
            "seed": plan.set.seed}},
        {"tool": "place_car_on_road", "args": {"distance_m": plan.placement.distance_m,
                                               "lane": lane(plan.placement.lane, cam.azimuth_deg),
                                               "heading_offset_deg": plan.placement.heading_offset_deg}},
        {"tool": "frame_shot", "args": frame_args(cam)},
    ]
    return ops


def sun_rel(light: LightPlan, camera: dict[str, Any], place: dict[str, Any]) -> float:
    """The plan's sun direction, or front-light the flank the camera sees when it gives none."""
    if light.sun_rel_deg is not None:
        return light.sun_rel_deg
    return lighting.front_light_sun_rel(camera["location"], place["location"], place["heading_deg"])


def light_args(light: LightPlan, hdri_id: str, camera: dict[str, Any], place: dict[str, Any]) -> dict[str, Any]:
    """set_lighting arguments that light the scene with HDRI `hdri_id` (a find_hdris result),
    rotated so its measured sun sits at `sun_rel` from the camera's view."""
    heading = lighting.camera_heading(camera["location"], place["location"])
    return {"sky": "hdri", "polyhaven_id": hdri_id, "resolution": HDRI_RESOLUTION,
            "align_sun_azimuth_deg": lighting.sun_azimuth(heading, sun_rel(light, camera, place)), "sun_strength": 0.0}


def exposure(plan: ShotPlan) -> dict[str, float]:
    """The shot's exposure: the camera's aperture and the plan's motion."""
    return camera_exposure(plan.camera, plan.motion)


def lens(exp: dict[str, float], camera: dict[str, Any]) -> Lens | None:
    """Depth of field on the plate, focused on the car (None without an f-stop)."""
    if exp["fstop"] <= 0:
        return None
    return Lens(focal_mm=camera["lens_mm"], focus_m=camera["distance_m"], f_number=exp["fstop"])


def tracking(exp: dict[str, float], camera: dict[str, Any], heading_deg: float,
             width: int, height: int) -> tuple[Intrinsics, Any] | None:
    """Plate motion blur: the camera's own travel over the shutter (the tracked share of the
    car's), streaking the plate by depth through the shot's real lens. None when it stands still."""
    speed = exp["speed_kmh"] * exp["tracking"]
    if speed <= 0:
        return None
    return (Intrinsics.from_camera(camera, width, height),
            camera_displacement(camera["rotation_deg"], heading_deg, speed, exp["shutter_s"]))
