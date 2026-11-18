"""review_guide (fn, human in the loop): show the Blender guide that passed the automatic gate and
wait for the user before any image is generated. approve -> plates; revise (note) -> repair,
which changes the set or camera and rebuilds the guide (then this card comes back); reject ->
finalize. Any later revise that changes the set or camera rebuilds the guide and returns here,
so the guide can always be edited again."""

from collections.abc import Iterator

from google.adk import Context, Event
from google.adk.events import RequestInput
from google.adk.workflow import node

from .. import state
from ..schemas import Approval
from ..shot import load, save


@node(rerun_on_resume=True)
def review_guide(ctx: Context) -> Iterator[Event | RequestInput]:
    shot = load(ctx.state)
    interrupt_id = f"guide_{shot.job_id}_a{shot.attempt}"
    answer = ctx.resume_inputs.get(interrupt_id)
    if not answer:
        state.set_run(ctx.state, status="awaiting_approval", node="review_guide", job_id=shot.job_id,
                      location=shot.location_key, attempt=shot.attempt)
        yield RequestInput(
            interrupt_id=interrupt_id,
            message=(f"{shot.plan.location_label}: guide for attempt {shot.attempt}, before any image is made. "
                     "Approve, revise with a note (camera, road, set), or reject?"),
            payload={"stage": "guide", "job_id": shot.job_id, "location": shot.location_key,
                     "final_key": shot.guides["keys"]["grey"], "attempt": shot.attempt,
                     "scores": shot.guides.get("checks", {}), "verdict": "pass"},
            response_schema=Approval,
        )
        return
    decision = Approval.model_validate(answer)
    shot.history.append({"attempt": shot.attempt, "event": "human_guide", "decision": decision.decision,
                         "note": decision.note})
    if decision.decision == "revise":
        shot.note = f"user (guide review): {decision.note}"
        shot.report = {"passed": False, "score": 0.0, "stage": "guides", "codes": ["user_revised_guide"],
                       "results": []}
        save(ctx.state, shot)
        yield Event(route="revise", output=decision.note)
        return
    save(ctx.state, shot)
    if decision.decision == "reject":
        yield Event(route="rejected", output={"decision": "reject"})
        return
    yield Event(route="approved", message="Guide approved; painting the scene.")
