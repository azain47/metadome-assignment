"""repair (LLM, RepairDecision) -> accept_repair (fn): decide which stage to redo and how the
plan changes, from the check report, the history and the user's note. Code moves the stage
earlier when the plan change needs it (scene or light changes -> SETUP)."""

from google.adk import Agent, Context, Event

from .. import prompts
from ..llm import make_model
from ..schemas import RepairDecision
from ..shot import earliest, load, save, stage_for_change

repair = Agent(
    name="repair",
    model=make_model(),
    static_instruction=prompts.STATIC_REPAIR,
    instruction=prompts.repair_context,
    output_schema=RepairDecision,
)


def accept_repair(ctx: Context, node_input: dict) -> Event:
    decision = RepairDecision.model_validate(node_input)
    shot = load(ctx.state)
    stage = earliest([decision.stage, stage_for_change(shot.plan, decision.plan) or decision.stage])
    shot.plan = decision.plan
    shot.stage = stage
    shot.note = decision.reason
    shot.history.append({"attempt": shot.attempt, "event": "repair", "stage": stage, "reason": decision.reason})
    save(ctx.state, shot)
    return Event(message=f"Repair: redo from {stage.lower()}: {decision.reason}")
