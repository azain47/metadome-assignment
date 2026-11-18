"""The graph's scene calls: the plan maps onto Blender tool calls without translation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from _plans import make_plan, with_camera  # noqa: E402
from app import lighting  # noqa: E402
from app.livery import material_assignments  # noqa: E402
from trueshot import scene  # noqa: E402
from trueshot.camera import car_motion_args  # noqa: E402
from trueshot.schemas import MotionPlan, Placement, SetElement  # noqa: E402

FACTS = {
    "car_key": "cars/car.glb", "length_m": 4.5,
    "materials": {"glass": {"preset": "glass_car"}},
    "material_groups": {"body": ["paint_a", "paint_b"]},
    "livery_default": "grey", "liveries": {"grey": {"description": "grey", "groups": {"body": {"base": "#666666"}}}},
}


def test_setup_ops_map_the_plan_onto_tool_calls() -> None:
    plan = make_plan()
    ops = scene.setup_ops(plan, FACTS)
    assert [o["tool"] for o in ops] == ["load_car", "set_materials", "reset_scene", "build_set",
                                        "place_car_on_road", "frame_shot"]
    args = {o["tool"]: o["args"] for o in ops}
    assert args["load_car"] == {"car_key": "cars/car.glb", "length_m": 4.5}
    # daylight default: lamps are fitted but not lit
    assert args["set_materials"] == {"assignments": material_assignments(FACTS, lamps_on=False)}
    assert args["reset_scene"] == {"keep_car": True}
    # only the fields the planner set travel (the tool keeps its own defaults for the rest)
    assert args["build_set"] == {
        "road": [{"straight_m": 40.0}, {"turn_deg": 180.0, "radius_m": 14.0}],
        "elements": [{"kind": "hillside", "side": "near", "grade": 0.3, "height_deg": [3.0, 5.0]}],
        "camera_side": "left", "road_width_m": 7.0, "seed": 3}
    # azimuth 32 = camera on the car's left -> the left lane, so road shows beyond the car
    assert args["place_car_on_road"] == {"distance_m": 12.0, "lane": "left", "heading_offset_deg": 0.0}
    assert args["frame_shot"] == plan.camera.model_dump(exclude={"fstop"})


def test_the_camera_side_and_lanes_follow_the_azimuth() -> None:
    right = scene.setup_ops(with_camera(make_plan(), azimuth_deg=300.0), FACTS)
    args = {o["tool"]: o["args"] for o in right}
    assert args["build_set"]["camera_side"] == "right" and args["place_car_on_road"]["lane"] == "right"
    far = scene.setup_ops(make_plan(placement=Placement(lane="far")), FACTS)
    assert far[4]["args"]["lane"] == "right"


def test_element_ranges_reach_the_tool_as_a_number_or_lo_hi() -> None:
    wall = SetElement(kind="wall", side="far", distance_m=[0.8], height_m=[0.9])
    hills = SetElement(kind="mountains", count=6, height_deg=[3.0, 1.0, 2.0])
    assert scene.tool_element(wall) == {"kind": "wall", "side": "far", "distance_m": 0.8, "height_m": 0.9}
    assert scene.tool_element(hills) == {"kind": "mountains", "side": "both", "count": 6, "height_deg": [1.0, 3.0]}


def test_setup_ops_need_a_car() -> None:
    with pytest.raises(ValueError):
        scene.setup_ops(make_plan(), {"length_m": 4.5})


def test_light_args_put_the_sun_relative_to_the_camera() -> None:
    camera, place = {"location": [-10.0, 10.0, 1.0]}, {"location": [0.0, 0.0, 0.0]}
    plan = make_plan()
    args = scene.light_args(plan.light.model_copy(update={"sun_rel_deg": 90.0}), "kiara_8_sunset", camera, place)
    heading = lighting.camera_heading(camera["location"], place["location"])
    assert heading == pytest.approx(135.0)
    assert args == {"sky": "hdri", "polyhaven_id": "kiara_8_sunset", "resolution": "4k",
                    "align_sun_azimuth_deg": pytest.approx(225.0), "sun_strength": 0.0}


def test_lens_and_tracking_follow_the_plan() -> None:
    cam = {"lens_mm": 40.0, "distance_m": 9.0, "rotation_deg": [88.0, 0.0, 145.0], "shift_x": 0.0, "shift_y": 0.0}
    exp = scene.exposure(make_plan())
    lens = scene.lens(exp, cam)
    assert (lens.focal_mm, lens.focus_m, lens.f_number) == (40.0, 9.0, 5.6)
    intr, disp = scene.tracking(exp, cam, 0.0, 1536, 1024)
    assert intr.fx == pytest.approx(40.0 / 36.0 * 1536)
    travel = 60.0 / 3.6 / 100
    # camera and car share the car's travel: the plate streaks by the tracked part, the body by the rest
    assert float((disp ** 2).sum()) ** 0.5 == pytest.approx(0.92 * travel)
    assert car_motion_args(exp)["body_slip_m"] == pytest.approx(0.08 * travel, abs=1e-4)
    parked = scene.exposure(make_plan(motion=MotionPlan()))
    assert scene.tracking(parked, cam, 0.0, 1536, 1024) is None
    assert car_motion_args(parked) == {"fstop": 5.6, "speed_kmh": 0.0, "shutter_s": 0.0, "body_slip_m": 0.0,
                                       "motion_blur": False}


def test_long_exposures_streak_the_plate_but_never_blur_the_car_body_much() -> None:
    plan = make_plan(motion=MotionPlan(speed_kmh=90.0, shutter_s=1 / 25, tracking=0.85))
    exp = scene.exposure(plan)
    cam = {"lens_mm": 85.0, "distance_m": 9.0, "rotation_deg": [88.0, 0.0, 145.0], "shift_x": 0.0, "shift_y": 0.0}
    _, disp = scene.tracking(exp, cam, 0.0, 1536, 1024)
    assert float((disp ** 2).sum()) ** 0.5 == pytest.approx(0.85 * 1.0)  # 1 m of travel: heavy plate streaks
    args = car_motion_args(exp)
    # the car is the product: however long the shutter, its own blur is capped at 2 cm
    assert args["body_slip_m"] == 0.02 and args["motion_blur"] is True
    # blur off: frozen wheels, body and plate; depth of field stays
    still = scene.exposure(plan.model_copy(update={"motion": MotionPlan(speed_kmh=90.0, blur=False)}))
    assert scene.tracking(still, cam, 0.0, 1536, 1024) is None
    assert car_motion_args(still)["motion_blur"] is False and still["fstop"] == 5.6
