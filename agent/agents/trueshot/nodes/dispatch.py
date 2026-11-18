"""dispatch (fn, hub): every pass and every retry goes through here.

It clamps the wanted stage to the budget (images, loops, session spend), opens a new attempt
folder, and routes to the earliest stage that has to run: SETUP | PLATES | RENDER |
COMPOSITE, REPLAY (rebuild a stored scene first) or OUT (stop; best candidate to approval).
"""

from google.adk import Context, Event

from .. import state
from ..shot import load, next_stage, save


def dispatch(ctx: Context) -> Event:
    shot = load(ctx.state)
    spend = (ctx.state.get(state.SPEND) or {}).get("total_usd", 0.0)
    wanted = shot.stage
    stage, reason = ("REPLAY", "") if wanted == "REPLAY" else next_stage(shot, wanted, spend)
    if stage != "OUT":
        shot.budget.loops += 1
        shot.attempt += 1
        shot.prefix = f"jobs/{shot.job_id}/a{shot.attempt}/"
    shot.stage = stage
    shot.history.append({"attempt": shot.attempt, "stage": stage, "wanted": wanted, "reason": reason or shot.note})
    save(ctx.state, shot)
    state.set_run(ctx.state, status="running", node=stage.lower(), job_id=shot.job_id,
                  location=shot.location_key, attempt=shot.attempt)
    note = f" ({reason})" if reason else ""
    message = f"Attempt {shot.attempt}: {stage.lower()}{note}." if stage != "OUT" else f"Stopping retries{note}."
    return Event(route=stage, message=message, output=message)
