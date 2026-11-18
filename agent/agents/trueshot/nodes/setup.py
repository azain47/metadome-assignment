"""setup_scene (fn): build the shot's scene in Blender from the plan, one visible step per tool
call (scene.setup_ops), then light it and save the .blend. replay_scene (fn): rebuild a stored
scene from its recorded calls (re-runs and edits), then light it from the current plan.

Both store the frame_shot camera, the car's placement and the derived sun geometry in
`shot.scene`, which the plates, render and composite stages read.
"""

from collections.abc import AsyncIterator
from typing import Any

from google.adk import Context, Event

from app import lighting
from app.hdri_pick import pick_hdri
from app.services import services

from .. import memory, scene, state
from .. import settings as user_settings
from ..camera import MIN_CAMERA_DISTANCE_M
from ..shot import ShotState, load, save
from ..steps import tracked
from .common import ensure_blender_ready

_SUMMARY = {
    "load_car": lambda r: f"{r.get('car_key')} dims {r.get('dims_m')} m",
    "build_set": lambda r: (f"road {r.get('road_length_m')} m, "
                            + (", ".join(f"{e.get('kind')} {e.get('side', '')} x{e.get('placed', '')}".strip()
                                         for e in r.get("elements", [])) or "no stand-ins")),
    "place_car_on_road": lambda r: f"at {r.get('location')}, heading {r.get('heading_deg')}°",
    "frame_shot": lambda r: (f"{r['camera']['lens_mm']:.0f} mm at {r['camera']['distance_m']:.1f} m, "
                             f"all in frame: {r.get('all_in_frame')}"),
    "find_hdris": lambda r: ", ".join(f"{h['id']} ({h.get('score')})" for h in r.get("results", [])[:3]),
    "set_lighting": lambda r: f"HDRI sun {r.get('hdri_sun')}, rotation {r.get('hdri_rotation_deg')}°",
    "save_blend": lambda r: f"{r.get('bytes', 0) / 1e6:.1f} MB",
}


async def _call(ctx: Context, tool: str, args: dict[str, Any], out: dict[str, Any]) -> AsyncIterator[Event]:
    """One visible Blender tool call; its result lands in out["result"]."""
    async for item in tracked(ctx, tool, args, services().blender.call(tool, **args), _SUMMARY.get(tool)):
        if isinstance(item, Event):
            yield item
        else:
            out["result"] = item


async def _build(ctx: Context, shot: ShotState, ops: list[dict[str, Any]]) -> AsyncIterator[Event]:
    """Run the scene ops, light the scene from the plan and save the .blend; fills shot.scene.
    A replay keeps the HDRI the shot found before when its query is unchanged."""
    previous_light = shot.scene.get("light") or {}
    shot.scene = {}
    out: dict[str, Any] = {}
    for op in ops:
        async for event in _call(ctx, op["tool"], dict(op.get("args") or {}), out):
            yield event
        if op["tool"] == "frame_shot":
            shot.scene["camera"] = out["result"]["camera"]
        elif op["tool"] == "place_car_on_road":
            shot.scene["place"] = {"location": out["result"]["location"], "heading_deg": out["result"]["heading_deg"]}
    camera = shot.scene["camera"]
    if shot.plan.camera.subject_width_m <= 0 and camera["distance_m"] < MIN_CAMERA_DISTANCE_M:
        yield Event(message=f"Warning: the camera is {camera['distance_m']:.1f} m from the car "
                            f"(< {MIN_CAMERA_DISTANCE_M} m); expect perspective distortion.")

    light = shot.plan.light
    reuse = previous_light.get("hdri_query") == light.hdri_query and previous_light.get("polyhaven_id")
    hdri_id, why = (previous_light["polyhaven_id"], previous_light.get("hdri_why", "")) if reuse else ("", "")
    if not hdri_id:
        async for event in _call(ctx, "find_hdris", {"query": light.hdri_query, "limit": scene.HDRI_CANDIDATES}, out):
            yield event
        found = out["result"].get("results") or []
        if not found:
            raise ValueError(f"find_hdris found nothing for {light.hdri_query!r}")
        # A changed query asks for a different sky: never hand back the map it replaces.
        replaced = previous_light.get("polyhaven_id")
        if replaced and len(found) > 1:
            found = [c for c in found if c["id"] != replaced] or found
        pick: dict[str, Any] = {}
        async for item in tracked(ctx, "pick_hdri", {"candidates": [c["id"] for c in found]},
                                  pick_hdri(found, shot.plan.scene_prompt, light.time_of_day,
                                            reasoning_effort=user_settings.reasoning_effort(ctx.state)),
                                  lambda p: f"{p['id']}: {p['why']}"):
            if isinstance(item, Event):
                yield item
            else:
                pick = item
        hdri_id, why = pick["id"], pick["why"]
        state.add_spend(ctx.state, cap_usd=services().settings.spend_cap_usd, llm_usd=float(pick.get("cost_usd", 0.0)))
    args = scene.light_args(light, hdri_id, camera, shot.scene["place"])
    async for event in _call(ctx, "set_lighting", args, out):
        yield event
    sun = out["result"]["hdri_sun"]
    shot.scene["light"] = {**args, "hdri_query": light.hdri_query, "hdri_why": why, "hdri_sun": sun,
                           "hdri_rotation_deg": out["result"].get("hdri_rotation_deg"),
                           "sun_words": lighting.sun_words(scene.sun_rel(light, camera, shot.scene["place"]),
                                                           sun["elevation_deg"])}

    async for event in _call(ctx, "save_blend", {"job_prefix": shot.prefix, "name": "scene"}, out):
        yield event
    shot.scene["blend_key"] = out["result"].get("key", "")


async def setup_scene(ctx: Context) -> AsyncIterator[Event]:
    async for event in ensure_blender_ready(ctx):
        yield event
    shot = load(ctx.state)
    ops = scene.setup_ops(shot.plan, memory.product_facts(ctx.state))
    async for event in _build(ctx, shot, ops):
        yield event
    shot.scene_ops = ops
    shot.plate_key, shot.car = "", {}  # a new scene invalidates earlier plates and car renders
    save(ctx.state, shot)
    yield Event(message=f"Scene set: {shot.plan.concept} Road of {len(shot.plan.set.road)} piece(s), "
                        f"{len(shot.plan.set.elements)} stand-in group(s), HDRI {shot.scene['light']['polyhaven_id']}. "
                        f"{shot.scene['light']['sun_words']}")


async def replay_scene(ctx: Context) -> AsyncIterator[Event]:
    async for event in ensure_blender_ready(ctx):
        yield event
    shot = load(ctx.state)
    async for event in _build(ctx, shot, shot.scene_ops):
        yield event
    resume = shot.resume_stage or "GUIDES"
    shot.resume_stage = ""
    save(ctx.state, shot)
    yield Event(route=resume, message=f"Scene rebuilt from {len(shot.scene_ops)} recorded Blender calls.")
