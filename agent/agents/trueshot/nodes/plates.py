"""Plates: the empty scene, painted by the image model from the Blender guides.

plate_jobs (fn) -> plate_worker (fn, parallel, one per candidate) -> pick_plate (fn)

Best-of-N: every candidate is checked by the plate-stage checks (cheap, deterministic) before
any Blender time is spent on it; the best passing one is kept. Image generations are counted
against the shot's budget whether or not they pass.
"""

from typing import Any, AsyncIterator

from google.adk import Context, Event
from google.adk.workflow import RetryConfig, node

from app.checks import CheckContext, run_checks
from app.imagegen import ImageGenError, PlateRequest
from app.imaging.io import decode_png, decode_rgb
from app.services import services

from .. import memory, prompts, state
from .. import settings as user_settings
from ..shot import ShotState, load, save
from ..steps import tracked


def plate_jobs(ctx: Context) -> Event:
    shot = load(ctx.state)
    n = min(services().settings.plate_candidates, shot.budget.images_left)
    jobs = [{"index": i, "seed": shot.attempt * 1000 + i} for i in range(n)]
    return Event(output=jobs)


async def _guide_inputs(keys: dict[str, str]) -> dict[str, bytes]:
    store = services().store
    return {name: await store.get(keys[k]) for name, k in (("grey", "grey"), ("lines", "lines"), ("depth", "depth_png"))}


def plate_prompt(shot: ShotState, has_refs: bool) -> str:
    """The plan's scene words plus the real sun direction and camera of the built scene."""
    cam = shot.scene["camera"]
    return prompts.plate_prompt(shot.plan.model_dump(mode="json"), has_refs,
                                light_words=shot.scene["light"]["sun_words"],
                                camera=prompts.camera_words(cam["lens_mm"], cam["location"][2]))


async def _generate(prompt: str, guides: dict[str, bytes], refs: list[bytes], seed: int,
                    quality: str | None = None) -> Any:
    request = PlateRequest(
        guide_grey=guides["grey"], guide_lines=guides["lines"], guide_depth=guides["depth"],
        references=refs, prompt=prompt, n=1, seed=seed, quality=quality)
    return await services().images.generate(request)


@node(parallel_worker=True, max_parallel_workers=2,
      retry_config=RetryConfig(max_attempts=3, initial_delay=2.0, exceptions=[TimeoutError, ConnectionError]))
async def plate_worker(ctx: Context, node_input: dict) -> AsyncIterator[Any]:
    """One candidate: generate, upload, run the plate checks. Never raises for a bad plate."""
    shot = load(ctx.state)
    svc = services()
    index, seed = int(node_input["index"]), int(node_input["seed"])
    guides = await _guide_inputs(shot.guides["keys"])
    refs = [await svc.store.get(k) for k in shot.plan.reference_keys]
    outcome: dict[str, Any] = {"index": index, "seed": seed, "key": "", "cost_usd": 0.0, "generated": True}
    prompt = plate_prompt(shot, bool(refs))
    try:
        result = None
        async for item in tracked(ctx, "generate_plate", {"index": index, "seed": seed, "backend": svc.images.name,
                                                          "model": svc.images.model, "prompt": prompt},
                                  _generate(prompt, guides, refs, seed, user_settings.image_quality(ctx.state)),
                                  lambda r: f"{r.backend}/{r.model} ${r.cost_usd:.3f}"):
            if isinstance(item, Event):
                yield item
            else:
                result = item
    except ImageGenError as ex:
        outcome.update(passed=False, score=0.0, codes=["generation_failed"], detail=str(ex)[:300])
        yield outcome
        return
    outcome["cost_usd"] = result.cost_usd
    outcome["key"] = await svc.store.put(f"{shot.prefix}plate_{index}.png", result.images[0])

    check_ctx = CheckContext(
        plate=decode_rgb(result.images[0]),
        guides={"grey": decode_rgb(guides["grey"]), "lines": decode_png(guides["lines"])},
        layout=shot.guides.get("layout", {}), plan=shot.plan.model_dump(mode="json"),
        facts=memory.product_facts(ctx.state))
    checks = await run_checks("plate", check_ctx)
    blocking_failed = [c for c in checks if c.blocking and not c.passed]
    outcome.update(
        passed=not blocking_failed,
        score=round(sum(c.score for c in checks) / max(1, len(checks)), 3),
        codes=[c.code for c in checks if not c.passed and c.code],
        checks=[{"name": c.name, "passed": c.passed, "score": round(c.score, 3), "detail": c.detail} for c in checks])
    yield outcome


def pick_plate(ctx: Context, node_input: list[dict[str, Any]]) -> Event:
    shot = load(ctx.state)
    results = [r for r in (node_input or []) if r]
    shot.budget.images_used += sum(1 for r in results if r.get("generated"))
    image_usd = sum(float(r.get("cost_usd", 0.0)) for r in results)
    state.add_spend(ctx.state, cap_usd=services().settings.spend_cap_usd, image_usd=image_usd)
    for r in results:
        r["attempt"] = shot.attempt
        if r.get("key"):
            state.add_gallery(ctx.state, kind="plate", key=r["key"], job_id=shot.job_id, location=shot.location_key,
                              attempt=shot.attempt, scores={c["name"]: c["score"] for c in r.get("checks", [])},
                              verdict="pass" if r.get("passed") else "fail",
                              reason=", ".join(r.get("codes", [])) or None)
    shot.plates.extend(results)
    passing = sorted((r for r in results if r.get("passed")), key=lambda r: r["score"], reverse=True)
    if passing:
        shot.plate_key = passing[0]["key"]
        save(ctx.state, shot)
        return Event(route="ok", message=f"Plate {passing[0]['index']} kept (score {passing[0]['score']:.2f}).")
    codes = sorted({c for r in results for c in r.get("codes", [])}) or ["no_plates"]
    shot.report = {"passed": False, "score": 0.0, "codes": codes, "stage": "plates",
                   "results": [{"index": r["index"], "codes": r.get("codes"), "checks": r.get("checks")} for r in results]}
    save(ctx.state, shot)
    return Event(route="none", message=f"No usable plate this attempt ({', '.join(codes)}).")
