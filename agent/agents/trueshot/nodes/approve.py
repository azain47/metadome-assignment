"""approve (fn, human in the loop) -> finalize (fn).

approve pauses the run with an ADK RequestInput (persisted: the stack can restart while it
waits) and resumes with the user's Approval: approve | revise (note -> repair) | reject.
finalize stores the result, exports the still with its passes, writes long-term memory on
approval, and moves on to the next queued location.
"""

import json
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any

from google.adk import Context, Event
from google.adk.events import RequestInput
from google.adk.workflow import node

from app.services import services

from .. import memory, state
from ..schemas import Approval
from ..shot import QUEUE, ShotState, load, remember_summary, save
from .guides import plate_depth_png_key


@node(rerun_on_resume=True)
def approve(ctx: Context) -> Iterator[Event | RequestInput]:
    shot = load(ctx.state)
    best = shot.best
    if best is None:
        yield Event(route="none", output={"decision": "none"})
        return
    interrupt_id = f"approve_{shot.job_id}_a{shot.attempt}"
    answer = ctx.resume_inputs.get(interrupt_id)
    if not answer:
        state.set_run(ctx.state, status="awaiting_approval", node="approve", job_id=shot.job_id,
                      location=shot.location_key, attempt=shot.attempt)
        below = not best.passed
        yield RequestInput(
            interrupt_id=interrupt_id,
            message=(f"{shot.plan.location_label}: attempt {best.attempt}, score {best.score:.2f}"
                     + (" (below the quality bar)" if below else "") + ". Approve, revise with a note, or reject?"),
            payload={"job_id": shot.job_id, "location": shot.location_key, "final_key": best.key,
                     "scores": {r["name"]: r["score"] for r in best.report}, "verdict": "best_below_bar" if below else "pass",
                     "reason": ", ".join(best.codes) or None, "attempt": best.attempt, "below_threshold": below},
            response_schema=Approval,
        )
        return
    decision = Approval.model_validate(answer)
    shot.history.append({"attempt": shot.attempt, "event": "human", "decision": decision.decision, "note": decision.note})
    if decision.decision == "revise":
        # The checks can't measure the user's request, so the next result replaces the old best.
        shot.note = f"user: {decision.note}"
        shot.best = None
        save(ctx.state, shot)
        yield Event(route="revise", output=decision.note)
        return
    save(ctx.state, shot)
    yield Event(route="approved" if decision.decision == "approve" else "rejected",
                output={"decision": decision.decision})


# Files exported per approved still: name in the export folder -> where its key lives.
def _export_items(shot: ShotState) -> dict[str, str]:
    g, c = shot.guides.get("keys", {}), shot.car.get("keys", {})
    plate = shot.best.plate_key if shot.best else shot.plate_key
    items = {"final.png": shot.best.key if shot.best else "", "plate.png": plate,
             "plate_depth.png": plate_depth_png_key(plate) if plate else "",
             "car_rgba.png": c.get("rgba_png", ""), "car.exr": c.get("exr", ""), "scene.blend": c.get("blend", ""),
             "guide_grey.png": g.get("grey", ""), "guide_lines.png": g.get("lines", ""),
             "guide_depth.png": g.get("depth_png", ""), "mask_road.png": g.get("road_mask", ""),
             "mask_car.png": g.get("car_mask", ""), "layout.json": g.get("layout", "")}
    return {name: key for name, key in items.items() if isinstance(key, str) and key}


async def _export(shot: ShotState) -> str:
    """Copy the still and its passes to outputs/<location>/<job>/ (mounted from the repo)."""
    store = services().store
    folder = os.path.join(services().settings.outputs_dir, shot.location_key, shot.job_id)
    os.makedirs(folder, exist_ok=True)
    for name, key in _export_items(shot).items():
        if await store.size(key) is None:  # e.g. plate_depth.png when the depth could not be measured
            continue
        with open(os.path.join(folder, name), "wb") as fh:
            fh.write(await store.get(key))
    meta = {"plan": shot.plan.model_dump(mode="json"), "report": shot.best.report if shot.best else [],
            "history": shot.history, "scene_ops": shot.scene_ops, "scene": shot.scene,
            "budget": shot.budget.model_dump()}
    with open(os.path.join(folder, "run.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    return folder


async def finalize(ctx: Context, node_input: Any) -> AsyncIterator[Event]:
    shot = load(ctx.state)
    decision = (node_input or {}).get("decision", "none") if isinstance(node_input, dict) else "none"
    db = services().db
    if decision == "approved" or decision == "approve":
        final_key = await services().store.put(f"jobs/{shot.job_id}/final.png", await services().store.get(shot.best.key))
        folder = await _export(shot)
        memory.remember_approval(ctx.state, shot, final_key)
        status, text = "approved", f"Approved {shot.job_id}. Exported the still and its passes to {folder}."
        state.add_gallery(ctx.state, kind="final", key=final_key, job_id=shot.job_id, location=shot.location_key,
                          attempt=shot.best.attempt, scores={r["name"]: r["score"] for r in shot.best.report},
                          verdict="pass" if shot.best.passed else "best_below_bar")
    elif decision == "reject":
        status, final_key, text = "rejected", "", f"Rejected {shot.job_id}; nothing exported."
    else:
        status, final_key, text = "failed", "", f"No usable result for {shot.job_id} within the budget."
    await db.update_job(shot.job_id, status=status, final_key=final_key or None,
                        verdict=(("pass" if shot.best.passed else "best_below_bar") if shot.best else None),
                        best_score=shot.best.score if shot.best else None, attempts=shot.attempt,
                        plan=shot.plan.model_dump(mode="json"), state=shot.model_dump(mode="json"))
    remember_summary(ctx.state, shot, status, final_key)
    queue = list(ctx.state.get(QUEUE) or [])
    if queue:
        ctx.state[QUEUE] = queue[1:]
        ctx.state["shot"] = None
        yield Event(message=f"{text} Next: {queue[0].splitlines()[0]}", route="next", output=queue[0])
        return
    state.set_run(ctx.state, status="done", node="finalize", job_id=shot.job_id, location=shot.location_key,
                  attempt=shot.attempt)
    yield Event(message=text)
