"""Budget clamping in dispatch/decide, and plan changes: which stage a retry starts from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from _plans import make_plan, with_camera  # noqa: E402
from trueshot.shot import Budget, ShotState, next_stage, stage_for_change  # noqa: E402


def _shot(**kw) -> ShotState:
    return ShotState(job_id="j", location_key="alps", plan=make_plan(),
                     budget=Budget(images_cap=4, loops_cap=6), **kw)


def test_spend_cap_stops_everything() -> None:
    assert next_stage(_shot(), "SETUP", spend_total=1e9)[0] == "OUT"


def test_loop_cap_stops_everything() -> None:
    shot = _shot()
    shot.budget.loops = 6
    assert next_stage(shot, "COMPOSITE", 0.0)[0] == "OUT"


def test_no_images_left_reuses_best_plate_or_stops() -> None:
    shot = _shot()
    shot.budget.images_used = 4
    assert next_stage(shot, "SETUP", 0.0)[0] == "OUT"
    shot.plate_key = "jobs/j/a1/plate_0.png"
    assert next_stage(shot, "PLATES", 0.0)[0] == "RENDER"


def test_later_stages_fall_back_to_missing_inputs() -> None:
    shot = _shot()
    assert next_stage(shot, "RENDER", 0.0)[0] == "PLATES"
    shot.plate_key = "p.png"
    assert next_stage(shot, "COMPOSITE", 0.0)[0] == "RENDER"
    shot.car = {"keys": {"rgba_png": "c.png"}}
    assert next_stage(shot, "COMPOSITE", 0.0) == ("COMPOSITE", "")


def test_plan_changes_force_the_stage_that_applies_them() -> None:
    plan = _shot().plan
    assert stage_for_change(plan, plan.model_copy(deep=True)) is None
    assert stage_for_change(plan, with_camera(plan, azimuth_deg=150.0)) == "SETUP"
    assert stage_for_change(plan, plan.model_copy(update={"set": plan.set.model_copy(update={"seed": 4})})) == "SETUP"
    moved = plan.model_copy(update={"placement": plan.placement.model_copy(update={"distance_m": 60.0})})
    assert stage_for_change(plan, moved) == "SETUP"
    relit = plan.model_copy(update={"light": plan.light.model_copy(update={"sun_rel_deg": 90.0})})
    assert stage_for_change(plan, relit) == "SETUP"
    worded = plan.model_copy(update={"light": plan.light.model_copy(update={"time_of_day": "dusk"})})
    assert stage_for_change(plan, worded) == "PLATES"
    assert stage_for_change(plan, plan.model_copy(update={"scene_prompt": "a desert road"})) == "PLATES"
