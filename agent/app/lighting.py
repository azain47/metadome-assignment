"""Light geometry shared by Blender lighting and the plate prompt, so both agree on the sun.

Directions are compass azimuths in Blender's frame: 0 = +Y, 90 = +X (clockwise from above).
A shot's sun is planned relative to the camera (`sun_rel_deg`: 0 = straight ahead of the camera,
+90 = to its right, 180 = behind it); `sun_azimuth` turns that into a world direction, the HDRI
is rotated so its measured sun points there (`hdri_rotation`), and `sun_words` describes the same
geometry to the image model.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def _wrap180(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


def camera_heading(camera_xy: Sequence[float], target_xy: Sequence[float]) -> float:
    """Compass heading the camera looks along (camera -> target), in [0, 360)."""
    dx, dy = target_xy[0] - camera_xy[0], target_xy[1] - camera_xy[1]
    return math.degrees(math.atan2(dx, dy)) % 360.0


def sun_azimuth(heading_deg: float, sun_rel_deg: float) -> float:
    return (heading_deg + sun_rel_deg) % 360.0


def hdri_rotation(hdri_sun_azimuth_deg: float, wanted_azimuth_deg: float) -> float:
    """Z rotation for set_lighting's HDRI mapping (world sun = image sun + rotation)."""
    return round(_wrap180(wanted_azimuth_deg - hdri_sun_azimuth_deg), 2)


def front_light_sun_rel(camera_xy: Sequence[float], car_xy: Sequence[float], car_heading_deg: float,
                        behind_deg: float = 135.0) -> float:
    """Product-lighting default: the sun behind the camera's shoulder on the side that front-lights
    the flank of the car the camera sees. Returns sun_rel_deg (+-behind_deg)."""
    heading = camera_heading(camera_xy, car_xy)
    camera_side = _wrap180((heading + 180.0) - car_heading_deg)  # bearing car -> camera vs car heading
    flank_normal = car_heading_deg + (90.0 if camera_side > 0 else -90.0)
    return math.copysign(behind_deg, _wrap180(flank_normal - heading))


def _height(elevation_deg: float) -> str:
    if elevation_deg < 6:
        return "very low, just above the horizon"
    if elevation_deg < 20:
        return "low"
    if elevation_deg < 45:
        return "at mid height"
    return "high"


def _direction(rel: float, near: str, far: str) -> str:
    """Camera-relative direction in words: rel 0 = `far` (ahead), 180 = `near` (behind)."""
    a, side = abs(rel), ("right" if rel > 0 else "left")
    if a < 30:
        return far
    if a > 150:
        return near
    if a < 60:
        return f"{far} and to the {side}"
    if a > 120:
        return f"{near} and to the {side}"
    return f"to the {side}"


def sun_words(sun_rel_deg: float, elevation_deg: float) -> str:
    """Sentence for the image prompt: where the sun is and which way shadows fall, seen from the
    camera. Shadows point away from the sun (sun_rel + 180)."""
    rel = _wrap180(sun_rel_deg)
    where = _direction(rel, "behind the camera", "ahead of the camera, back-lighting the scene")
    shadows = _direction(_wrap180(rel + 180.0), "toward the camera", "away from the camera")
    return f"The sun is {_height(elevation_deg)}, {where}; shadows fall {shadows}."
