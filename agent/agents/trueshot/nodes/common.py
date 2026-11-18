"""Helpers shared by graph nodes that drive Blender directly."""

from typing import Any, AsyncIterator

from google.adk import Context, Event

from app.services import services

from ..steps import tracked

_READY = "temp:blender_ready"


async def ensure_blender_ready(ctx: Context) -> AsyncIterator[Event]:
    """Once per run: GPU render device (OptiX > CUDA > CPU), render display off, MD collections.
    Blender may have been restarted between runs, so this is cheap to repeat per invocation."""
    if ctx.state.get(_READY):
        return
    result: dict[str, Any] = {}
    async for item in tracked(ctx, "setup_blender", {"use_gpu": True},
                              services().blender.call("setup_blender", use_gpu=True),
                              lambda r: f"{r.get('device_type')} {', '.join(r.get('devices', []))}"):
        if isinstance(item, Event):
            yield item
        else:
            result = item
    ctx.state[_READY] = result.get("device_type", "ok")
