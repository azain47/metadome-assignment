"""Edits restart at the cheapest stage that is still correct, and plan choices are checked."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from _plans import make_plan, with_camera  # noqa: E402
from trueshot.nodes.intake import stored_job_id  # noqa: E402
from trueshot.nodes.plan import checked_plan  # noqa: E402
from trueshot.schemas import GradePlan, RoadSegment  # noqa: E402
from trueshot.shot import stage_for_change  # noqa: E402

FACTS = {"liveries": {"mclaren_orange": {}, "burton_blue": {}}, "livery_default": "mclaren_orange"}
CATALOG = {"ladakh": ["refs/ladakh/ref_0.jpg"]}
_plan = make_plan


def test_car_only_changes_keep_the_plate_and_grade_changes_skip_blender() -> None:
    old = _plan()
    assert stage_for_change(old, _plan(livery="burton_blue")) == "RENDER"
    assert stage_for_change(old, _plan(light=old.light.model_copy(update={"lamps_on": True}))) == "RENDER"
    assert stage_for_change(old, with_camera(old, fstop=2.0)) == "RENDER"  # aperture: renders only
    assert stage_for_change(old, _plan(grade=GradePlan(car_exposure_ev=-0.7))) == "COMPOSITE"
    assert stage_for_change(old, _plan(light=old.light.model_copy(update={"hdri_query": "alpine road soft haze"}))) \
        == "SETUP"
    assert stage_for_change(old, with_camera(old, elevation_deg=60.0)) == "SETUP"  # the camera moves
    assert stage_for_change(old, _plan()) is None


def test_road_pieces_must_be_a_straight_or_a_full_turn() -> None:
    bad = _plan()
    bad = bad.model_copy(update={"set": bad.set.model_copy(update={
        "road": [*bad.set.road, RoadSegment(turn_deg=90.0), RoadSegment(straight_m=10.0, turn_deg=45.0, radius_m=30.0)]})})
    plan, notes = checked_plan(bad, FACTS, CATALOG, None)
    assert plan.set.road == _plan().set.road and "2 road piece(s)" in notes[0]


def test_unknown_livery_keeps_the_previous_one_and_unknown_refs_are_dropped() -> None:
    previous = _plan(livery="burton_blue")
    plan, notes = checked_plan(_plan(livery="papaya_spark", reference_keys=["refs/ladakh/ref_0.jpg", "refs/alps/x.jpg"]),
                               FACTS, CATALOG, previous)
    assert plan.livery == "burton_blue" and "burton_blue" in notes[0]
    assert plan.reference_keys == ["refs/ladakh/ref_0.jpg"]
    ok, notes = checked_plan(_plan(livery="burton_blue"), FACTS, CATALOG, None)
    assert ok.livery == "burton_blue" and notes == []


def test_edit_finds_an_approved_shot_from_another_session() -> None:
    state = {"user:approved_shots": {"alps": {"job_id": "alps-f91c18", "plan": {"location_label": "Swiss Alps pass"}}}}
    assert stored_job_id(state, "Alps") == "alps-f91c18"
    assert stored_job_id(state, "swiss alps") == "alps-f91c18"
    assert stored_job_id({**state, "shots": {"alps": {"job_id": "alps-new"}}}, "alps") == "alps-new"
    assert stored_job_id(state, "coast-123") == "coast-123"
