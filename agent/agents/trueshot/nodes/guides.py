"""guides (fn): render the layout guides with the car hidden, derive the line and depth guides
the image backends use, and approve the guide (the `guide` checks) before any image is spent:
route "ok" goes on to the plates, "bad" to decide (and from there to repair). plate_depth measures
a generated plate's own depth against these guides, for the plate checks and the composite's blur."""

import asyncio
from typing import Any, AsyncIterator

import numpy as np
from google.adk import Context, Event
from app.checks import CheckContext, run_checks
from app.imaging.depth import PlateDepth, estimate_plate_depth
from app.imaging.guides import depth_guide, line_guide
from app.imaging.io import decode_png, decode_rgb, encode_png, exr_layer, read_exr
from app.services import services

from .. import state
from ..shot import load, save
from ..steps import tracked
from .common import ensure_blender_ready


def depth_metres(exr: bytes) -> Any:
    """The guides' depth pass (metres, car hidden) as an HxW array."""
    layer = exr_layer(read_exr(exr), "Depth", "Z")
    return layer.get("Z", next(iter(layer.values())))


def _mask(png: bytes) -> np.ndarray:
    m = decode_png(png)
    return (m[..., 0] if m.ndim == 3 else m) > 127


async def depth_guides(keys: dict[str, str]) -> dict[str, np.ndarray]:
    """What a plate's depth is calibrated against, from the shot camera: Blender's depth
    (metres, car hidden) and the road mask."""
    store = services().store
    return {"depth": depth_metres(await store.get(keys["depth"])),
            "road_mask": _mask(await store.get(keys["road_mask"]))}


def plate_depth_png_key(plate_key: str) -> str:
    """Where a plate's viewable depth (inverse depth, near = bright) is stored: next to the plate."""
    return plate_key.rsplit(".", 1)[0] + "_depth.png"


async def _measure(plate: np.ndarray, guides: dict[str, np.ndarray], plate_key: str) -> PlateDepth:
    depth = await asyncio.to_thread(estimate_plate_depth, plate, guides["depth"], guides["road_mask"])
    await services().store.put(plate_depth_png_key(plate_key), encode_png(depth_guide(depth.metres)))
    return depth


async def plate_depth(ctx: Context, plate: np.ndarray, guides: dict[str, np.ndarray],
                      plate_key: str) -> AsyncIterator[Any]:
    """Measure the plate's own depth as the visible step `plate_depth`. Yields the step's events,
    then the PlateDepth, or None when it could not be measured: the step's error is in the feed
    and the composite blurs by Blender's depth instead."""
    try:
        async for item in tracked(ctx, "plate_depth", {"plate": plate_key}, _measure(plate, guides, plate_key),
                                  PlateDepth.summary):
            yield item
    except Exception:  # noqa: BLE001 - shown as the step's error; the shot carries on
        yield None


async def guides(ctx: Context) -> AsyncIterator[Event]:
    async for event in ensure_blender_ready(ctx):
        yield event
    shot = load(ctx.state)
    svc = services()
    result: dict[str, Any] = {}
    async for item in tracked(ctx, "render_guides", {"job_prefix": shot.prefix},
                              svc.blender.call("render_guides", job_prefix=shot.prefix),
                              lambda r: f"{len(r.get('keys', {}))} passes, horizon y={r['layout'].get('horizon_y_px')}"):
        if isinstance(item, Event):
            yield item
        else:
            result = item
    keys = dict(result["keys"])

    grey = decode_rgb(await svc.store.get(keys["grey"]))
    keys["lines"] = await svc.store.put(shot.prefix + "guide_lines.png", encode_png(line_guide(grey)))
    keys["depth_png"] = await svc.store.put(
        shot.prefix + "guide_depth.png", encode_png(depth_guide(depth_metres(await svc.store.get(keys["depth"])))))

    shot.guides = {"keys": keys, "layout": result["layout"]}
    check_ctx = CheckContext(plate=grey, guides={"lines": decode_png(await svc.store.get(keys["lines"])),
                                                 "road_mask": _mask(await svc.store.get(keys["road_mask"]))},
                             layout=result["layout"])
    checks: list[Any] = []
    async for item in tracked(ctx, "approve_guide", {"guide": keys["grey"]}, run_checks("guide", check_ctx),
                              lambda rs: "; ".join(r.detail for r in rs)):
        if isinstance(item, Event):
            yield item
        else:
            checks = item
    failed = [c for c in checks if c.blocking and not c.passed]
    shot.guides["checks"] = {c.name: round(c.score, 3) for c in checks}
    state.add_gallery(ctx.state, kind="guide", key=keys["grey"], job_id=shot.job_id,
                      location=shot.location_key, attempt=shot.attempt,
                      scores={c.name: round(c.score, 3) for c in checks},
                      verdict="fail" if failed else "pass", reason=", ".join(c.code for c in failed if c.code) or None)
    if failed:
        shot.report = {"passed": False, "score": 0.0, "stage": "guides",
                       "codes": sorted({c.code for c in failed if c.code}),
                       "results": [{"name": c.name, "passed": c.passed, "score": round(c.score, 3),
                                    "detail": c.detail, "blocking": c.blocking, "code": c.code} for c in checks]}
        save(ctx.state, shot)
        yield Event(route="bad", message=f"Guide rejected before generating: {failed[0].detail}.")
        return
    save(ctx.state, shot)
    yield Event(route="ok", output="guides ready")
