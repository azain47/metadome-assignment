"""The shot being worked on: one typed session-state key (`shot`) that every stage node reads
and writes. Stage nodes never depend on their node input, so any edge (first pass, retry,
repair, re-run) can enter them.

Also the budget rules: pure functions, enforced by `dispatch`/`decide`, never by the LLM.
"""

from typing import Any, MutableMapping

from pydantic import BaseModel, Field

from app.config import settings

from .schemas import STAGE_ORDER, ShotPlan

SHOT = "shot"
SHOTS = "shots"      # {location_key: summary of the latest shot there}, for edits and answers
QUEUE = "queue"      # remaining location requests of a multi-location message
# Camera fields that only the renders use; every other camera field moves the Blender camera.
RENDER_ONLY_CAMERA_FIELDS = {"fstop"}
SUMMARY_CAMERA_FIELDS = {"azimuth_deg", "elevation_deg", "focal_mm", "car_width_frac", "subject_width_m"}


class Budget(BaseModel):
    images_used: int = 0
    images_cap: int = Field(default_factory=lambda: settings.image_budget_per_shot)
    loops: int = 0
    loops_cap: int = Field(default_factory=lambda: settings.loops_per_shot)

    @property
    def images_left(self) -> int:
        return max(0, self.images_cap - self.images_used)


class Candidate(BaseModel):
    """A finished composite and its check report."""
    attempt: int
    key: str
    plate_key: str
    score: float
    passed: bool
    codes: list[str] = Field(default_factory=list)
    report: list[dict[str, Any]] = Field(default_factory=list)


class ShotState(BaseModel):
    job_id: str
    location_key: str
    plan: ShotPlan
    stage: str = "SETUP"               # next stage dispatch will route to (or "OUT"/"REPLAY")
    resume_stage: str = ""             # after REPLAY: the stage to continue with ("" = guides)
    attempt: int = 0
    prefix: str = ""                   # jobs/<job_id>/a<attempt>/
    budget: Budget = Field(default_factory=Budget)
    scene_ops: list[dict[str, Any]] = Field(default_factory=list)  # replayable scene-building Blender calls
    scene: dict[str, Any] = Field(default_factory=dict)  # {camera: frame_shot camera, place: {location,
    #   heading_deg}, light: set_lighting args + {camera_heading_deg, sun_azimuth_deg, sun_words}, blend_key}
    guides: dict[str, Any] = Field(default_factory=dict)  # {"keys": {...}, "layout": {...}}
    plates: list[dict[str, Any]] = Field(default_factory=list)
    plate_key: str = ""
    car: dict[str, Any] = Field(default_factory=dict)   # render_car result (keys, wheels, ...)
    composite_key: str = ""
    report: dict[str, Any] = Field(default_factory=dict)  # last check: {passed, score, codes, results}
    best: Candidate | None = None
    note: str = ""                     # repair reason or the user's revise note, for repair
    history: list[dict[str, Any]] = Field(default_factory=list)


def load(state: MutableMapping[str, Any]) -> ShotState:
    raw = state.get(SHOT)
    if not raw:
        raise RuntimeError("no shot in progress")
    return ShotState.model_validate(raw)


def save(state: MutableMapping[str, Any], shot: ShotState) -> None:
    state[SHOT] = shot.model_dump(mode="json")


def remember_summary(state: MutableMapping[str, Any], shot: ShotState, status: str, final_key: str = "") -> None:
    shots = dict(state.get(SHOTS) or {})
    shots[shot.location_key] = {
        "job_id": shot.job_id, "status": status, "final_key": final_key,
        "location_label": shot.plan.location_label, "concept": shot.plan.concept,
        "camera": shot.plan.camera.model_dump(include=SUMMARY_CAMERA_FIELDS),
        "time_of_day": shot.plan.light.time_of_day,
    }
    state[SHOTS] = shots


def next_stage(shot: ShotState, wanted: str, spend_total: float) -> tuple[str, str]:
    """Clamp the wanted stage to what the budget allows. Returns (stage, reason)."""
    if spend_total >= settings.spend_cap_usd:
        return "OUT", f"session spend ${spend_total:.2f} reached the ${settings.spend_cap_usd:.0f} cap"
    if shot.budget.loops >= shot.budget.loops_cap:
        return "OUT", f"{shot.budget.loops} loops used (cap {shot.budget.loops_cap})"
    needs_plates = wanted in ("SETUP", "PLATES")
    if needs_plates and shot.budget.images_left < 1:
        if shot.plate_key and wanted == "PLATES":
            return "RENDER", "image budget used up; reusing the best plate"
        return "OUT", f"image budget used up ({shot.budget.images_used}/{shot.budget.images_cap})"
    if wanted in ("RENDER", "COMPOSITE") and not shot.plate_key:
        return "PLATES", "no usable plate yet"
    if wanted == "COMPOSITE" and not shot.car:
        return "RENDER", "no car render yet"
    return wanted, ""


def earliest(stages: list[str]) -> str:
    return min(stages, key=STAGE_ORDER.index)


def _framing(plan: ShotPlan) -> dict[str, Any]:
    return plan.camera.model_dump(exclude=RENDER_ONLY_CAMERA_FIELDS)


def stage_for_change(old: ShotPlan, new: ShotPlan) -> str | None:
    """Earliest stage a plan change forces: the set, the car's placement, the camera or the HDRI
    light -> SETUP; what the plate shows -> PLATES; the car itself (livery, lamps) or the
    exposure (motion, aperture) -> RENDER (the plate is car-free, so it stays); the grade ->
    COMPOSITE; None = no change."""
    if (old.set, old.placement) != (new.set, new.placement) or _framing(old) != _framing(new):
        return "SETUP"
    if (old.light.hdri_query, old.light.sun_rel_deg) != (new.light.hdri_query, new.light.sun_rel_deg):
        return "SETUP"
    if (old.scene_prompt, old.light.time_of_day, old.reference_keys) != \
            (new.scene_prompt, new.light.time_of_day, new.reference_keys):
        return "PLATES"
    if (old.motion, old.camera.fstop, old.livery, old.light.lamps_on) != \
            (new.motion, new.camera.fstop, new.livery, new.light.lamps_on):
        return "RENDER"
    if old.grade != new.grade:
        return "COMPOSITE"
    return None
