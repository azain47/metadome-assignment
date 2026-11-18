"""Writers for the session-state keys the UI reads. Only this module knows their shapes.

Contract (local://contracts.md, "Session state keys the UI reads"):
- run:     {status: idle|running|done|failed, node, job_id, location, attempt}
- gallery: [{id, job_id, location, attempt, kind, key, scores, verdict, reason}]
- spend:   {llm_usd, image_usd, total_usd, cap_usd}
Memory keys (`user:*`, `app:*`) are written by the memory node tomorrow.

`state` is an ADK `State` (ctx.state / tool_context.state): assigning a key records it in the
event's state delta, which persists it and streams it to the UI.
"""

import uuid
from typing import Any, Literal, MutableMapping

RUN = "run"
GALLERY = "gallery"
SPEND = "spend"

RunStatus = Literal["idle", "running", "done", "failed"]
GalleryKind = Literal["preview", "snapshot", "guide", "plate", "car", "composite", "final"]
Verdict = Literal["pass", "fail", "best_below_bar"]


def set_run(state: MutableMapping[str, Any], *, status: RunStatus | None = None, **fields: Any) -> dict[str, Any]:
    run = dict(state.get(RUN) or {"status": "idle"})
    if status is not None:
        run["status"] = status
    run.update(fields)
    state[RUN] = run
    return run


def add_gallery(state: MutableMapping[str, Any], *, kind: GalleryKind, key: str, job_id: str = "",
                location: str = "", attempt: int = 0, scores: dict[str, Any] | None = None,
                verdict: Verdict | None = None, reason: str | None = None) -> dict[str, Any]:
    item = {
        "id": uuid.uuid4().hex[:10], "job_id": job_id, "location": location, "attempt": attempt,
        "kind": kind, "key": key, "scores": scores or {}, "verdict": verdict, "reason": reason,
    }
    state[GALLERY] = [*(state.get(GALLERY) or []), item]
    return item


def add_spend(state: MutableMapping[str, Any], *, cap_usd: float, llm_usd: float = 0.0,
              image_usd: float = 0.0) -> dict[str, Any]:
    spend = dict(state.get(SPEND) or {"llm_usd": 0.0, "image_usd": 0.0})
    spend["llm_usd"] = round(spend.get("llm_usd", 0.0) + llm_usd, 4)
    spend["image_usd"] = round(spend.get("image_usd", 0.0) + image_usd, 4)
    spend["total_usd"] = round(spend["llm_usd"] + spend["image_usd"], 4)
    spend["cap_usd"] = cap_usd
    state[SPEND] = spend
    return spend
