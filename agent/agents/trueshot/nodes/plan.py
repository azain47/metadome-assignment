"""plan (LLM, PlanOutput) -> accept_plan (fn): turn a request into a typed ShotPlan and open a job.

For an edit, the new job inherits the stored shot's scene, guides, plate and car render, so the
pipeline can restart at a later stage (e.g. PLATES for "golden hour words like last time", RENDER
for a new livery); the scene is rebuilt from the stored Blender calls first (REPLAY) because
Blender may hold another shot by now. A grade-only change goes straight to COMPOSITE (no Blender).
A plan change that alters the scene or its light always restarts at SETUP.
"""

from typing import Any, AsyncIterator

from google.adk import Agent, Context, Event

from app.services import services

from .. import memory, prompts, state
from ..llm import make_model
from ..schemas import PlanOutput, ShotPlan
from ..shot import SHOT, ShotState, earliest, save, stage_for_change

plan = Agent(
    name="plan",
    model=make_model(),
    static_instruction=prompts.STATIC_PLAN,
    instruction=prompts.plan_context,
    output_schema=PlanOutput,
)


def checked_plan(plan: ShotPlan, facts: dict[str, Any], catalog: dict[str, list[str]],
                 previous: ShotPlan | None) -> tuple[ShotPlan, list[str]]:
    """Drop choices the planner cannot make: a livery the car does not have (keeps the previous
    one), reference keys that are not in the catalog, and road pieces that are neither a
    straight nor a full turn."""
    notes: list[str] = []
    liveries = facts.get("liveries") or {}
    if plan.livery and plan.livery not in liveries:
        notes.append(f"No livery named {plan.livery!r}; available: {', '.join(liveries) or 'none'}.")
        plan = plan.model_copy(update={"livery": previous.livery if previous else None})
    known = {k for keys in catalog.values() for k in keys}
    refs = [k for k in plan.reference_keys if k in known]
    if refs != plan.reference_keys:
        plan = plan.model_copy(update={"reference_keys": refs})
    road = [s for s in plan.set.road
            if (s.straight_m is not None) != (s.turn_deg is not None and s.radius_m is not None)]
    if len(road) != len(plan.set.road):
        notes.append(f"Dropped {len(plan.set.road) - len(road)} road piece(s) that were neither a straight "
                     "nor a turn with a radius.")
        plan = plan.model_copy(update={"set": plan.set.model_copy(update={"road": road})})
    return plan, notes


async def accept_plan(ctx: Context, node_input: dict) -> AsyncIterator[Event]:
    out = PlanOutput.model_validate(node_input)
    edited = ctx.state.get(SHOT) or None
    previous = ShotState.model_validate(edited) if edited else None
    checked, notes = checked_plan(out.plan, memory.product_facts(ctx.state), memory.reference_catalog(),
                                  previous.plan if previous else None)
    job_id = await services().db.create_job(ctx.session.id, checked.location_key, checked.model_dump(mode="json"))
    shot = ShotState(job_id=job_id, location_key=checked.location_key, plan=checked, stage=out.start_stage)
    if previous:
        start = earliest([out.start_stage, stage_for_change(previous.plan, checked) or out.start_stage])
        shot.stage, shot.scene = start, previous.scene  # setup keeps its HDRI when the query is unchanged
        if start != "SETUP":
            shot.scene_ops = previous.scene_ops
            shot.guides, shot.plate_key, shot.car = previous.guides, previous.plate_key, previous.car
            # The compositor needs no Blender; every other stage first rebuilds the stored scene.
            shot.stage, shot.resume_stage = ("COMPOSITE", "") if start == "COMPOSITE" else ("REPLAY", start)
            shot.note = f"edit of {previous.job_id}; reuses its scene"
        shot.history.append({"event": "edit", "from_job": previous.job_id, "memory_used": out.memory_used})
    save(ctx.state, shot)
    state.set_run(ctx.state, status="running", node="plan", job_id=job_id, location=shot.location_key, attempt=0)
    used = f" Reusing from memory: {'; '.join(out.memory_used)}." if out.memory_used else ""
    said = " ".join(n for n in [out.note, *notes] if n)
    cam = checked.camera
    subject = "the whole car" if cam.subject_width_m <= 0 else f"a {cam.subject_width_m:g} m detail"
    ideas = "".join(f"\n- {i}" for i in out.ideas)
    yield Event(message=f"Plan for {checked.location_label}: {checked.concept} ({cam.focal_mm:g} mm, "
                        f"{cam.elevation_deg:g}° up, azimuth {cam.azimuth_deg:g}°, {subject}; "
                        f"{checked.light.time_of_day}; livery {checked.livery or 'default'}), job {job_id}."
                        f"{used}{' ' + said if said else ''}"
                        f"{chr(10) + 'Other ideas for this place:' + ideas if ideas else ''}")
