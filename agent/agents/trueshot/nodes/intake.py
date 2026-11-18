"""Entry of every chat turn: read the message, classify it, and route it.

intake_prep (fn) -> intake (LLM, Intent) -> route_intent (fn) -> plan | answer | dispatch (re-run)
"""

import json
from typing import Any, AsyncIterator

from google.adk import Agent, Context, Event
from google.genai import types

from app.services import services

from .. import memory, prompts, state
from ..llm import make_model
from ..schemas import Intent, ShotPlan
from ..shot import QUEUE, SHOT, SHOTS, ShotState, save

_RECENT_TURNS = 6


def message_text(node_input: Any) -> str:
    if isinstance(node_input, types.Content):
        return " ".join(p.text for p in node_input.parts or [] if p.text).strip()
    return str(node_input or "").strip()


def _recent_turns(ctx: Context) -> list[str]:
    turns: list[str] = []
    for event in ctx.session.events:
        if event.partial or not event.content or not event.content.parts:
            continue
        text = " ".join(p.text for p in event.content.parts if p.text and not p.thought).strip()
        if text and not text.startswith("{"):
            who = "user" if event.author == "user" else "agent"
            turns.append(f"{who}: {text[:400]}")
    return turns[-_RECENT_TURNS - 1:-1]  # the current message is the last one


def intake_prep(ctx: Context, node_input: Any) -> Event:
    """Give the single-turn intake LLM the message plus recent conversation."""
    memory.ensure_product_facts(ctx.state)
    text = message_text(node_input)
    state.set_run(ctx.state, status="running", node="intake", job_id="", location="", attempt=0)
    brief = {"message": text, "recent_conversation": _recent_turns(ctx)}
    return Event(output=json.dumps(brief, indent=1))


intake = Agent(
    name="intake",
    model=make_model(),
    static_instruction=prompts.STATIC_INTAKE,
    instruction=prompts.intake_context,
    output_schema=Intent,
)


def _plan_brief(location_request: str, intent: Intent) -> str:
    return f"Plan this shot: {location_request}\n\nFull user request: {intent.request}"


def stored_job_id(state: Any, ref: str) -> str:
    """The job a shot reference points to: a shot of this session (location key), an approved
    shot from long-term memory (any session: key, label or job id), else `ref` as a job id."""
    ref = ref.strip()
    shots = state.get(SHOTS) or {}
    if ref in shots:
        return shots[ref].get("job_id", ref)
    want = ref.lower()
    for key, entry in (state.get(memory.APPROVED) or {}).items():
        label = str((entry.get("plan") or {}).get("location_label", "")).lower()
        if want and (want in (key.lower(), str(entry.get("job_id", "")).lower()) or want in label):
            return str(entry["job_id"])
    return ref


def known_shots(state: Any) -> list[str]:
    return sorted(set(state.get(SHOTS) or {}) | set(state.get(memory.APPROVED) or {}))


async def route_intent(ctx: Context, node_input: dict) -> AsyncIterator[Event]:
    intent = Intent.model_validate(node_input)
    if intent.kind == "question":
        yield Event(route="question", output=intent.request)
        return

    if intent.kind == "new_shots":
        locations = intent.locations or [intent.request]
        ctx.state[QUEUE] = [_plan_brief(loc, intent) for loc in locations[1:]]
        ctx.state[SHOT] = None  # the planner sees "nothing being edited"
        yield Event(route="plan", output=_plan_brief(locations[0], intent))
        return

    # edit_shot / rerun: find the stored shot (this session, else approved in long-term memory)
    ref = intent.shot_ref.strip()
    job_id = stored_job_id(ctx.state, ref)
    job = await services().db.get_job(job_id) if job_id else None
    if not job or not job.get("state"):
        state.set_run(ctx.state, status="failed")
        yield Event(message=f"I couldn't find a stored shot for '{ref}'. Known shots: "
                            f"{', '.join(known_shots(ctx.state)) or 'none yet'}.")
        return
    stored = ShotState.model_validate(job["state"])
    if intent.kind == "rerun":
        # Same plan, same scene: rebuild the scene from its recorded Blender calls, then run
        # the pipeline again from the guides on.
        rerun = ShotState(job_id=await services().db.create_job(ctx.session.id, stored.location_key,
                                                                stored.plan.model_dump(mode="json")),
                          location_key=stored.location_key, plan=stored.plan, scene_ops=stored.scene_ops,
                          scene=stored.scene, stage="REPLAY", note=f"re-run of {stored.job_id}")
        save(ctx.state, rerun)
        yield Event(route="rerun", output=f"re-run {stored.job_id}")
        return
    ctx.state[SHOT] = stored.model_dump(mode="json")  # the planner edits this one
    ctx.state[QUEUE] = []
    yield Event(route="plan", output=f"Edit the stored shot {stored.job_id} ({stored.plan.location_label}): "
                                     f"{intent.request}")


answer = Agent(
    name="answer",
    model=make_model(),
    static_instruction=prompts.STATIC_ANSWER,
    instruction=prompts.answer_context,
)


def answered(ctx: Context) -> None:
    state.set_run(ctx.state, status="done", node="answer")


__all__ = ["intake_prep", "intake", "route_intent", "answer", "answered", "message_text", "ShotPlan"]
