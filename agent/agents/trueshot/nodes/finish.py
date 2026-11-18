"""composite (fn) -> check (fn) -> decide (fn): put the exact car render over the chosen plate,
score the result with the registered checks, and decide what happens next.

decide routes:
- pass   -> approve
- fix    -> dispatch with a stage chosen by code (FIX_TABLE), no LLM
- repair -> repair (LLM) for failures that need judgement
- out    -> approve with the best candidate so far, flagged below threshold
"""

import asyncio
from typing import Any, AsyncIterator

import numpy as np
from google.adk import Context, Event

from app.checks import CheckContext, run_checks
from app.compositing import CarRender, Composite, load_car_render
from app.compositing import composite as composite_images
from app.imaging.depth import PlateDepth
from app.imaging.io import decode_png, decode_rgb, encode_png
from app.services import services

from .. import memory, scene, state
from ..schemas import GradePlan
from ..shot import Candidate, ShotState, earliest, load, next_stage, save
from ..steps import tracked
from .guides import depth_guides, depth_metres, plate_depth

# Failure code -> stage that fixes it without judgement. Codes not listed go to `repair`.
FIX_TABLE: dict[str, str] = {
    "plate_blank": "PLATES",
    "layout_mismatch": "PLATES",
    "generation_failed": "PLATES",
    "no_plates": "PLATES",
}
# Codes that mean a pipeline bug rather than a bad shot: stop instead of retrying.
FATAL_CODES = frozenset({"car_pixels_changed"})

# Photographic finish, as tuned in the lab (scripts/dev/lab.py comp).
LIGHT_WRAP, VIGNETTE, GRAIN = 0.5, 0.12, 0.008


async def _car(shot: ShotState) -> CarRender:
    """The exact car render, re-developed from its EXR at the plan's grade exposure. The final
    check compares the composite against this same layer."""
    store, keys = services().store, shot.car["keys"]
    grade = shot.plan.grade or GradePlan()
    return load_car_render(await store.get(keys["rgba_png"]), await store.get(keys["exr"]),
                           exposure_ev=grade.car_exposure_ev + grade.exposure_ev)


async def _composite(shot: ShotState, plate: np.ndarray, depth: PlateDepth | None) -> dict[str, Any]:
    """Plate blurred by the shot's rolling-shot tracking and depth of field, both driven by the
    plate's own depth (Blender's depth pass when it could not be measured), car shadow, exact car
    render (graded from its EXR), light wrap, vignette and grain."""
    store = services().store
    grade = shot.plan.grade or GradePlan()
    car = await _car(shot)
    depth_m = depth.metres if depth is not None else depth_metres(await store.get(shot.guides["keys"]["depth"]))
    exp, cam = scene.exposure(shot.plan), shot.scene["camera"]
    height, width = car.rgba.shape[:2]
    result = await asyncio.to_thread(
        composite_images, plate, car, depth_m=depth_m, lens=scene.lens(exp, cam),
        tracking=scene.tracking(exp, cam, shot.scene["place"]["heading_deg"], width, height),
        light_wrap=LIGHT_WRAP, vignette=VIGNETTE, grain=GRAIN, exposure_ev=grade.exposure_ev)
    final_key = await store.put(f"{shot.prefix}composite.png", encode_png(result.final))
    pre_lens_key = await store.put(f"{shot.prefix}composite_pre_lens.png", encode_png(result.pre_lens))
    return {"final": final_key, "pre_lens": pre_lens_key, "plate": shot.plate_key}


async def composite(ctx: Context) -> AsyncIterator[Event]:
    shot = load(ctx.state)
    plate = decode_rgb(await services().store.get(shot.plate_key))
    depth: PlateDepth | None = None
    async for item in plate_depth(ctx, plate, await depth_guides(shot.guides["keys"]), shot.plate_key):
        if isinstance(item, Event):
            yield item
        else:
            depth = item
    out: dict[str, Any] = {}
    async for item in tracked(ctx, "composite", {"plate": shot.plate_key, "car": shot.car["keys"]["rgba_png"],
                                                 "blur_depth": "plate" if depth is not None else "blender"},
                              _composite(shot, plate, depth)):
        if isinstance(item, Event):
            yield item
        else:
            out = item
    shot.composite_key = out["final"]
    shot.report = {"pre_lens_key": out["pre_lens"]}
    save(ctx.state, shot)
    yield Event(output="composited")


async def check(ctx: Context) -> AsyncIterator[Event]:
    shot = load(ctx.state)
    store = services().store
    car = await _car(shot)
    pre_lens = decode_rgb(await store.get(shot.report["pre_lens_key"]))
    final = decode_rgb(await store.get(shot.composite_key))

    check_ctx = CheckContext(
        plate=decode_rgb(await store.get(shot.plate_key)),
        guides={"grey": decode_rgb(await store.get(shot.guides["keys"]["grey"])),
                "lines": decode_png(await store.get(shot.guides["keys"]["lines"]))},
        layout=shot.guides.get("layout", {}), plan=shot.plan.model_dump(mode="json"),
        facts=memory.product_facts(ctx.state),
        composite=Composite(final=final, pre_lens=pre_lens, alpha=car.rgba[..., 3].astype(np.float32) / 255.0),
        car=car)
    step_results: list[Any] = []
    async for item in tracked(ctx, "checks", {"stage": "final"}, run_checks("final", check_ctx),
                              lambda rs: ", ".join(f"{r.name}={'ok' if r.passed else r.code}" for r in rs)):
        if isinstance(item, Event):
            yield item
        else:
            step_results = item
    plate_info = next((p for p in shot.plates if p.get("key") == shot.plate_key), {})
    results = [*plate_info.get("checks", []),
               *({"name": r.name, "passed": r.passed, "score": round(r.score, 3), "detail": r.detail,
                  "blocking": r.blocking, "code": r.code} for r in step_results)]
    codes = [*plate_info.get("codes", []), *(r.code for r in step_results if not r.passed and r.code)]
    passed = all(r.passed for r in step_results if r.blocking) and bool(plate_info.get("passed", True))
    score = round(sum(r["score"] for r in results) / max(1, len(results)), 3)
    shot.report = {"passed": passed, "score": score, "codes": codes, "results": results, "stage": "final"}
    candidate = Candidate(attempt=shot.attempt, key=shot.composite_key, plate_key=shot.plate_key, score=score,
                          passed=passed, codes=codes, report=results)
    if shot.best is None or (candidate.passed, candidate.score) > (shot.best.passed, shot.best.score):
        shot.best = candidate
    save(ctx.state, shot)
    state.add_gallery(ctx.state, kind="composite", key=shot.composite_key, job_id=shot.job_id,
                      location=shot.location_key, attempt=shot.attempt,
                      scores={r["name"]: r["score"] for r in results},
                      verdict="pass" if passed else "fail", reason=", ".join(codes) or None)
    await services().db.add_candidate(shot.job_id, shot.attempt, "composite", shot.composite_key,
                                      scores={r["name"]: r["score"] for r in results},
                                      verdict="pass" if passed else "fail", reason=", ".join(codes) or None,
                                      meta={"plate_key": shot.plate_key, "car": shot.car["keys"]})
    yield Event(output=shot.report)


def _layout_missed_before(shot: ShotState) -> bool:
    """The previous attempt's plates all failed layout_match too: a third try at the same guide
    won't fix it (ladakh-b68d97 burned 8 images that way); the plan has to change."""
    prev = [p for p in shot.plates if p.get("attempt") == shot.attempt - 1]
    return bool(prev) and all("layout_mismatch" in (p.get("codes") or []) for p in prev)


def decide(ctx: Context) -> Event:
    shot = load(ctx.state)
    report = shot.report
    if report.get("passed"):
        return Event(route="pass", message=f"Checks passed (score {report.get('score', 0):.2f}).")
    codes = list(report.get("codes") or ["unknown"])
    if FATAL_CODES & set(codes):
        shot.note = f"stopped: {', '.join(sorted(FATAL_CODES & set(codes)))} (pipeline bug)"
        save(ctx.state, shot)
        return Event(route="out", message=f"Stopping: {shot.note}.")
    spend = (ctx.state.get(state.SPEND) or {}).get("total_usd", 0.0)
    if all(code in FIX_TABLE for code in codes) and not ("layout_mismatch" in codes and _layout_missed_before(shot)):
        wanted = earliest([FIX_TABLE[c] for c in codes])
        stage, reason = next_stage(shot, wanted, spend)
        if stage == "OUT":
            shot.note = reason
            save(ctx.state, shot)
            return Event(route="out", message=f"Out of budget: {reason}.")
        shot.stage, shot.note = stage, f"automatic fix for {', '.join(codes)}"
        save(ctx.state, shot)
        return Event(route="fix", message=f"Failed {', '.join(codes)}; retrying from {stage.lower()}.")
    if next_stage(shot, "COMPOSITE" if shot.car else "PLATES", spend)[0] == "OUT":
        return Event(route="out", message="Out of budget for a repair.")
    return Event(route="repair", message=f"Failed {', '.join(codes)}; asking the repair step what to change.")
