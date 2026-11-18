"""A complete ShotPlan for tests: a rolling shot at an alpine hairpin."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from trueshot.schemas import (  # noqa: E402
    CameraPlan,
    LightPlan,
    MotionPlan,
    Placement,
    RoadSegment,
    SetElement,
    SetPlan,
    ShotPlan,
)


def make_plan(**update) -> ShotPlan:
    plan = ShotPlan(
        location_key="alps", location_label="Alps", concept="low rolling shot out of a hairpin",
        set=SetPlan(road=[RoadSegment(straight_m=40.0), RoadSegment(turn_deg=180.0, radius_m=14.0)],
                    elements=[SetElement(kind="hillside", side="near", grade=0.3, height_deg=[3.0, 5.0])], seed=3),
        placement=Placement(distance_m=12.0),
        camera=CameraPlan(azimuth_deg=32.0, elevation_deg=2.5, focal_mm=40.0, car_width_frac=0.62, fstop=5.6),
        motion=MotionPlan(speed_kmh=60.0, shutter_s=0.01),
        light=LightPlan(time_of_day="late morning", hdri_query="alpine road midday", sun_rel_deg=-35.0),
        scene_prompt="an empty, deserted road")
    return plan.model_copy(update=update)


def with_camera(plan: ShotPlan, **update) -> ShotPlan:
    return plan.model_copy(update={"camera": plan.camera.model_copy(update=update)})
