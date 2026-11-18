"""render_car (fn): the final Cycles render of the car alone (transparent film, shadow catcher,
spinning wheels), lit and reflecting the scene's HDRI, with the shot's exposure (one shutter
for wheels, body slip and the plate's rolling-shot blur; depth of field). No image generation
here, so this stage costs nothing but time."""

from typing import Any, AsyncIterator

from google.adk import Context, Event
from google.adk.workflow import RetryConfig, node

from app.services import services

from app.livery import material_assignments

from .. import memory, scene, state
from ..camera import car_motion_args
from ..shot import load, save
from ..steps import tracked
from .common import ensure_blender_ready

_BLENDER_RETRY = RetryConfig(max_attempts=2, initial_delay=3.0)
# Unknown spoke count: assume a dense wheel, which gives a short shutter. Spokes of 5-10 spoke
# wheels stay readable; the cost is less motion blur than ideal for a 5-spoke wheel.
_DEFAULT_SPOKES = 10


@node(retry_config=_BLENDER_RETRY)
async def render_car(ctx: Context) -> AsyncIterator[Event]:
    async for event in ensure_blender_ready(ctx):
        yield event
    shot = load(ctx.state)
    svc = services()
    facts = memory.product_facts(ctx.state)
    # Materials first: a livery or lamps change re-renders only the car (the plate is car-free).
    livery = {"livery": shot.plan.livery or facts.get("livery_default"), "lamps_on": shot.plan.light.lamps_on}
    assignments = material_assignments(facts, shot.plan.livery, lamps_on=shot.plan.light.lamps_on)
    async for item in tracked(ctx, "set_materials", livery, svc.blender.call("set_materials", assignments=assignments),
                              lambda r: f"{len(r.get('changed', []))} materials"):
        if isinstance(item, Event):
            yield item
    # The chosen plate is projected onto the set, so the car's lower body reflects and picks up
    # bounce light from the generated road and roadside; the HDRI covers what the camera can't see.
    args = {"job_prefix": shot.prefix, "samples": svc.settings.render_samples,
            "spoke_count": int(facts.get("wheel_spoke_count") or _DEFAULT_SPOKES),
            **({"plate_key": shot.plate_key} if shot.plate_key else {}),
            **car_motion_args(scene.exposure(shot.plan))}
    result: dict[str, Any] = {}
    async for item in tracked(ctx, "render_car", args, svc.blender.call("render_car", **args),
                              lambda r: f"{r.get('render_s', 0):.1f}s on {r.get('device', '?')}"):
        if isinstance(item, Event):
            yield item
        else:
            result = item
    blend: dict[str, Any] = {}
    async for item in tracked(ctx, "save_blend", {"job_prefix": shot.prefix, "name": "scene"},
                              svc.blender.call("save_blend", job_prefix=shot.prefix, name="scene"),
                              lambda r: f"{r.get('bytes', 0) / 1e6:.1f} MB"):
        if isinstance(item, Event):
            yield item
        else:
            blend = item
    shot.car = {"keys": {**result["keys"], "blend": blend.get("key", "")}, "wheels": result.get("wheels", []),
                "shutter_frames": result.get("shutter_frames"), "attempt": shot.attempt}
    save(ctx.state, shot)
    state.add_gallery(ctx.state, kind="car", key=result["keys"]["rgba_png"], job_id=shot.job_id,
                      location=shot.location_key, attempt=shot.attempt)
    yield Event(output="car rendered")
