"""Visible pipeline steps run by graph function nodes (not by the LLM).

Each step produces two ADK events with `custom_metadata.ts_step` (same id: "running", then
"ok"/"error"), which the UI upserts into its activity feed, and one row in the tool log.

Inside a function node (an async generator):

    async for item in tracked(ctx, "render_guides", {"job_prefix": p}, blender.call("render_guides", job_prefix=p)):
        if isinstance(item, Event):
            yield item
        else:
            result = item

`tracked` re-raises the work's exception after emitting the "error" event.
"""

import dataclasses
import json
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any, TypeVar

from google.adk import Context, Event

from app.services import services

T = TypeVar("T")
TS_STEP = "ts_step"
_MAX_RESULT_CHARS = 4000


def _node_name(ctx: Context) -> str:
    """'trueshot@1/setup_scene@2' -> 'setup_scene'."""
    last = (ctx.node_path or "").rsplit("/", 1)[-1]
    return last.split("@", 1)[0] or "workflow"


def _loggable(value: Any) -> Any:
    """A JSON-safe copy for the tool log. Image bytes become a size note: the database holds
    object keys, never image bytes (a PlateResult carries its PNGs)."""
    if isinstance(value, (bytes, bytearray)):
        return f"<{len(value)} bytes>"
    if hasattr(value, "shape") and hasattr(value, "dtype"):  # numpy arrays (depth maps, masks)
        return f"<array {tuple(value.shape)} {value.dtype}>"
    if isinstance(value, dict):
        return {str(k): _loggable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_loggable(v) for v in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {"type": type(value).__name__,
                **{f.name: _loggable(getattr(value, f.name)) for f in dataclasses.fields(value)}}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _clip(value: Any) -> Any:
    safe = _loggable(value)
    text = json.dumps(safe, default=str)
    return safe if len(text) <= _MAX_RESULT_CHARS else text[:_MAX_RESULT_CHARS] + "…"


class Step:
    def __init__(self, ctx: Context, name: str, args: dict[str, Any] | None = None) -> None:
        self._ctx = ctx
        self.id = uuid.uuid4().hex[:12]
        self.name = name
        self.node = _node_name(ctx)
        self.args = args or {}
        self._t0 = time.monotonic()

    def _event(self, status: str, summary: str = "", duration_ms: int | None = None) -> Event:
        payload = {"id": self.id, "node": self.node, "name": self.name, "status": status,
                   "args": self.args, "summary": summary}
        if duration_ms is not None:
            payload["duration_ms"] = duration_ms
        return Event(custom_metadata={TS_STEP: payload})

    def started(self) -> Event:
        self._t0 = time.monotonic()
        return self._event("running")

    async def _log(self, status: str, result: Any, duration_ms: int) -> None:
        await services().db.log_tool(
            tool=self.name, source="graph", status=status, args=self.args, result=_clip(result),
            duration_ms=duration_ms, session_id=self._ctx.session.id,
            invocation_id=self._ctx.invocation_id, node=self.node)

    async def finished(self, result: Any = None, summary: str = "") -> Event:
        duration_ms = int((time.monotonic() - self._t0) * 1000)
        await self._log("ok", result, duration_ms)
        return self._event("ok", summary, duration_ms)

    async def failed(self, error: BaseException | str) -> Event:
        duration_ms = int((time.monotonic() - self._t0) * 1000)
        message = str(error)
        await self._log("error", {"error": message}, duration_ms)
        return self._event("error", message[:500], duration_ms)


async def tracked(ctx: Context, name: str, args: dict[str, Any], work: Awaitable[T],
                  summary: Callable[[T], str] | None = None) -> AsyncIterator[Event | T]:
    """Run `work` as a visible step: yields the running event, the final event, then the result."""
    step = Step(ctx, name, args)
    yield step.started()
    try:
        result = await work
    except Exception as ex:
        yield await step.failed(ex)
        raise
    yield await step.finished(result, summary(result) if summary else "")
    yield result
