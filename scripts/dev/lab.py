"""Plate lab: Blender -> guides -> image model -> composite, driven by hand (no LLM, no graph).

Every step reuses the agent's own adapters (BlenderClient, S3Store, image backends, guide
helpers, compositor), so whatever works here moves into the graph unchanged. Files for a shot
live in outputs/lab/<shot>/ and each step can be re-run on its own:

    cd agent && uv run python ../scripts/dev/lab.py scene alps
    uv run python ../scripts/dev/lab.py guides alps
    uv run python ../scripts/dev/lab.py plates alps --n 2 [--refs] [--quality high]
    uv run python ../scripts/dev/lab.py car alps
    uv run python ../scripts/dev/lab.py comp alps            # every plate, or --plate <file>
    uv run python ../scripts/dev/lab.py call scene_info '{}'  # any MCP tool, JSON args

Talks to the dev-published MCP server (127.0.0.1:8765) and S3 (127.0.0.1:8333).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[2]
for _k, _v in dotenv_values(ROOT / ".env").items():
    if _v:
        os.environ.setdefault(_k, _v)
os.environ.setdefault("BLENDER_MCP_URL", "http://127.0.0.1:8765/")
os.environ.setdefault("S3_ENDPOINT", "http://127.0.0.1:8333")
sys.path[:0] = [str(ROOT / "agent"), str(ROOT / "agent" / "agents")]

from app import lighting  # noqa: E402
from app.blender import BlenderClient  # noqa: E402
from app.compositing import composite, load_car_render  # noqa: E402
from app.config import settings  # noqa: E402
from app.hdri_pick import pick_hdri  # noqa: E402
from app.imagegen import get_backend  # noqa: E402
from app.imagegen.base import PlateRequest  # noqa: E402
from app.imaging.depth import estimate_plate_depth  # noqa: E402
from app.imaging.guides import depth_guide, line_guide  # noqa: E402
from app.imaging.io import decode_png, decode_rgb, encode_png, exr_layer, read_exr  # noqa: E402
from app.imaging.lens import Intrinsics, Lens, camera_displacement  # noqa: E402
from app.livery import material_assignments  # noqa: E402
from app.storage import S3Store  # noqa: E402
from trueshot import prompts, scene  # noqa: E402
from trueshot.camera import MIN_CAMERA_DISTANCE_M, car_motion_args, exposure  # noqa: E402
from trueshot.schemas import ShotPlan  # noqa: E402

LAB = ROOT / "outputs" / "lab"
FACTS = json.loads((ROOT / "assets" / "product_facts.json").read_text(encoding="utf-8"))

# Shot presets: what the planner would produce (set, placement, camera, motion as in
# schemas.ShotPlan). `sun_rel_deg` is the sun relative to the camera's view (0 ahead, +90 right,
# 180 behind); the HDRI is rotated to put its sun there.
_ALPINE_PEAKS = [
    {"kind": "mountains", "count": 8, "distance_m": [700, 1600], "height_deg": [1.0, 2.4], "width_deg": [20, 45],
     "jagged": 0.5, "steepness": 1.4},
    {"kind": "mountains", "count": 16, "distance_m": [2500, 7000], "height_deg": [2.8, 5.8], "width_deg": [8, 20],
     "jagged": 0.7, "steepness": 1.1}]
SHOTS: dict[str, dict[str, Any]] = {
    "alps": {
        "concept": "low rolling shot out of an alpine hairpin, peaks behind",
        "set": {"road": [{"straight_m": 45}, {"turn_deg": 180, "radius_m": 15}, {"straight_m": 90}],
                "elements": [{"kind": "hillside", "side": "near", "grade": 0.3, "height_deg": [4.0, 5.0]},
                             {"kind": "hillside", "side": "far", "grade": 0.3, "height_deg": [1.0, 1.5]},
                             *_ALPINE_PEAKS], "seed": 3},
        "place": {"distance_m": 0.0, "lane": "near"},
        "camera": {"azimuth_deg": 32.0, "elevation_deg": 2.5, "focal_mm": 40.0, "car_width_frac": 0.62,
                   "frame_x": 0.47, "frame_y": 0.64, "target_height_m": 0.55, "fstop": 5.6},
        "motion": {"speed_kmh": 60.0, "shutter_s": 0.01},
        "hdri_query": "alpine mountain meadow green grass midday clear sky", "sun_rel_deg": 135.0,
        "time_of_day": "bright late-morning daylight, clear blue sky with a few small clouds",
        "scene": ("A mountain pass road high in the Swiss Alps: fresh dark asphalt with white edge lines, "
                  "a low stone wall and alpine meadow beside the road, pine forest on the lower slopes, "
                  "steep rocky peaks with snow fields behind."),
    },
    "ladakh": {
        "concept": "chase car behind and above on a desert highway, mesas ahead",
        "set": {"road": [{"straight_m": 40}, {"turn_deg": 60, "radius_m": 140}],
                "elements": [{"kind": "mountains", "count": 7, "distance_m": [1500, 3500], "height_deg": [0.8, 2.0],
                              "width_deg": [30, 70], "jagged": 0.15, "flat_top": 0.35, "steepness": 4.0},
                             {"kind": "mountains", "count": 9, "distance_m": [4000, 10000], "height_deg": [1.2, 2.8],
                              "width_deg": [25, 60], "jagged": 0.2, "flat_top": 0.2, "steepness": 3.0}], "seed": 1},
        "place": {"distance_m": 0.0, "lane": "near"},
        "camera": {"azimuth_deg": 155.0, "elevation_deg": 8.0, "focal_mm": 50.0, "car_width_frac": 0.4,
                   "frame_x": 0.66, "frame_y": 0.62, "fstop": 5.6},
        "motion": {"speed_kmh": 70.0, "shutter_s": 1 / 80},
        "hdri_query": "arid barren rocky mountain valley midday clear", "sun_rel_deg": 140.0,
        "time_of_day": "hard high-altitude midday sun, deep blue cloudless sky, crisp short shadows",
        "scene": ("A high-altitude desert highway in Ladakh, India: smooth asphalt with faded edge lines, "
                  "gravel shoulders, barren brown and ochre mountains with sharp ridges, snow on the highest "
                  "peaks, thin clean air, no vegetation."),
    },
    "coast": {
        "concept": "eye-level rear three-quarter along the sea wall, town on the hill",
        "set": {"road": [{"straight_m": 30}, {"turn_deg": -50, "radius_m": 160}],
                "elements": [{"kind": "sea", "side": "far", "distance_m": [25]},
                             {"kind": "wall", "side": "far", "distance_m": [0.5], "height_m": [0.9]},
                             {"kind": "buildings", "side": "near", "count": 180, "distance_m": [10, 260],
                              "size_m": [7, 16], "height_m": [5, 18], "height_deg": [12]},
                             {"kind": "mountains", "count": 9, "distance_m": [900, 2600], "height_deg": [1.5, 4.0],
                              "width_deg": [25, 55], "jagged": 0.3, "steepness": 2.4}], "seed": 2},
        "place": {"distance_m": 0.0, "lane": "near"},
        "camera": {"azimuth_deg": 140.0, "elevation_deg": 6.0, "focal_mm": 70.0, "car_width_frac": 0.5,
                   "frame_x": 0.5, "frame_y": 0.62, "fstop": 4.0},
        "motion": {"speed_kmh": 0.0},
        "hdri_query": "mediterranean coast sea town beach clear afternoon", "sun_rel_deg": 130.0,
        "time_of_day": "bright midday Mediterranean sun, clear blue sky",
        "scene": ("A coastal road through a Mediterranean seaside town: clean asphalt, a low stone sea wall, "
                  "pastel-painted houses with terracotta roofs climbing the hillside, the blue sea and a "
                  "small harbour beyond."),
    },
}


def _plan(shot: str, livery: str | None = None, lamps_on: bool = False) -> ShotPlan:
    s = SHOTS[shot]
    return ShotPlan.model_validate({
        "location_key": shot, "location_label": shot, "concept": s["concept"], "set": s["set"],
        "placement": s["place"], "camera": s["camera"], "motion": s["motion"], "livery": livery,
        "light": {"time_of_day": s["time_of_day"], "hdri_query": s["hdri_query"], "sun_rel_deg": s["sun_rel_deg"],
                  "lamps_on": lamps_on},
        "scene_prompt": s["scene"]})


def shot_dir(shot: str) -> Path:
    d = LAB / shot
    d.mkdir(parents=True, exist_ok=True)
    return d


def store() -> S3Store:
    return S3Store(endpoint=settings.s3_endpoint, public_endpoint=settings.s3_public_endpoint,
                   access_key=settings.s3_access_key, secret_key=settings.s3_secret_key, bucket=settings.s3_bucket)


def blender() -> BlenderClient:
    return BlenderClient(settings.blender_mcp_url)


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")


def log_spend(entry: dict) -> None:
    LAB.mkdir(parents=True, exist_ok=True)
    with open(LAB / "spend.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


async def timed(bl: BlenderClient, tool: str, **args: Any) -> dict:
    t0 = time.monotonic()
    result = await bl.call(tool, **args)
    print(f"  {tool:18s} {time.monotonic() - t0:6.1f}s  {json.dumps(result, default=str)[:220]}")
    return result


async def cmd_scene(a: argparse.Namespace) -> None:
    """The graph's own scene calls (trueshot.scene.setup_ops) for the preset, then the light."""
    s, bl = SHOTS[a.shot], blender()
    plan = _plan(a.shot, a.livery, a.lamps_on)
    out: dict[str, Any] = {"preset": s}
    out["setup"] = await timed(bl, "setup_blender")
    for op in scene.setup_ops(plan, FACTS):
        out[op["tool"]] = await timed(bl, op["tool"], **op["args"])
    out["frame"], out["place"] = out.pop("frame_shot"), out.pop("place_car_on_road")
    if plan.camera.subject_width_m <= 0 and out["frame"]["camera"]["distance_m"] < MIN_CAMERA_DISTANCE_M:
        print(f"  WARNING camera {out['frame']['camera']['distance_m']:.1f} m from the car (< {MIN_CAMERA_DISTANCE_M} m): "
              "expect perspective distortion; use a longer lens")
    out["light"] = await cmd_light_apply(bl, s, out)
    out["blend"] = await timed(bl, "save_blend", job_prefix=f"lab/{a.shot}/", name="scene")
    save_json(shot_dir(a.shot) / "scene.json", out)


async def cmd_light_apply(bl: BlenderClient, s: dict, scene: dict) -> dict:
    """Pick a Poly Haven HDRI for the surroundings (`hdri` id, else the best `hdri_query` match)
    and rotate it so its measured sun sits at sun_rel_deg from the camera's view."""
    hdri_id = s.get("hdri")
    if not hdri_id:
        found = await bl.call("find_hdris", query=s["hdri_query"], limit=6)
        pick = await pick_hdri(found["results"], s["scene"], s["time_of_day"])
        hdri_id = pick["id"]
        print("  hdri candidates:", ", ".join(f"{r['id']} ({r['score']})" for r in found["results"]))
        print(f"  picked {hdri_id}: {pick['why']}")
    heading = lighting.camera_heading(scene["frame"]["camera"]["location"], scene["place"]["location"])
    wanted = lighting.sun_azimuth(heading, s["sun_rel_deg"])
    result = await timed(bl, "set_lighting", sky="hdri", polyhaven_id=hdri_id, resolution="4k",
                         align_sun_azimuth_deg=wanted, sun_strength=0.0)
    words = lighting.sun_words(s["sun_rel_deg"], result["hdri_sun"]["elevation_deg"])
    print(f"  {hdri_id}: camera heading {heading:.1f}, sun azimuth {wanted:.1f}, sun {result['hdri_sun']}: {words}")
    return {**result, "hdri_id": hdri_id, "camera_heading_deg": heading, "sun_azimuth_deg": wanted,
            "sun_words": words}


async def _fetch(st: S3Store, keys: dict[str, str], d: Path, names: dict[str, str]) -> dict[str, bytes]:
    got = {}
    for k, fname in names.items():
        if keys.get(k):
            got[k] = await st.get(keys[k])
            (d / fname).write_bytes(got[k])
    return got


async def cmd_guides(a: argparse.Namespace) -> None:
    d, st = shot_dir(a.shot), store()
    r = await timed(blender(), "render_guides", job_prefix=f"lab/{a.shot}/")
    got = await _fetch(st, r["keys"], d, {"grey": "guide_grey.png", "depth": "depth.exr", "road_mask": "mask_road.png",
                                          "car_mask": "mask_car.png"})
    save_json(d / "layout.json", r.get("layout", {}))
    (d / "guide_lines.png").write_bytes(encode_png(line_guide(decode_png(got["grey"]))))
    ch = exr_layer(read_exr(got["depth"]), "Depth", "Z")
    z = ch.get("Z", next(iter(ch.values())))
    (d / "guide_depth.png").write_bytes(encode_png(depth_guide(z)))
    print("  guides ->", d)


def _refs(shot: str) -> list[bytes]:
    folder = ROOT / "assets" / "refs" / shot
    if not folder.is_dir():
        return []
    return [p.read_bytes() for p in sorted(folder.iterdir()) if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp")]


async def cmd_plates(a: argparse.Namespace) -> None:
    d, s = shot_dir(a.shot), SHOTS[a.shot]
    refs = _refs(a.shot)[: a.max_refs] if a.refs else []
    scene = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    cam = scene["frame"]["camera"]
    plan = {"scene_prompt": s["scene"], "light": {"time_of_day": s["time_of_day"]}}
    prompt = prompts.plate_prompt(plan, bool(refs), light_words=scene["light"]["sun_words"],
                                  camera=prompts.camera_words(cam["lens_mm"], cam["location"][2]))
    backend = get_backend(a.backend, a.model)
    if a.backend == "openai":
        backend.quality, backend.complexity = a.quality, (a.complexity or None)
    req = PlateRequest(guide_grey=(d / "guide_grey.png").read_bytes(), guide_lines=(d / "guide_lines.png").read_bytes(),
                       guide_depth=(d / "guide_depth.png").read_bytes(), references=refs, prompt=prompt, n=a.n)
    t0 = time.monotonic()
    res = await backend.generate(req)
    dt = time.monotonic() - t0
    pdir = d / "plates"
    pdir.mkdir(exist_ok=True)
    tag = f"{a.backend}_{(a.model or 'default').replace('/', '-')}_{a.quality}_r{len(refs)}_{int(time.time()) % 100000}"
    for i, png in enumerate(res.images):
        (pdir / f"{tag}_{i}.png").write_bytes(png)
    meta = {"shot": a.shot, "tag": tag, "prompt": prompt, "n_refs": len(refs), "seconds": round(dt, 1),
            "cost_usd": res.cost_usd, "model": res.model, "meta": res.meta}
    save_json(pdir / f"{tag}.json", meta)
    log_spend({"t": time.time(), "kind": "plate", **{k: meta[k] for k in ("shot", "tag", "cost_usd", "seconds", "model")}})
    print(f"  {len(res.images)} plate(s) {tag} in {dt:.1f}s, ${res.cost_usd:.4f} ({res.meta.get('cost_source')})")


def _exposure_args(sp: argparse.ArgumentParser) -> None:
    """The plan's motion overrides; pass the same values to `car` and `comp` (one exposure)."""
    sp.add_argument("--speed", type=float, default=None, help="km/h; 0 = parked (default: the recipe's)")
    sp.add_argument("--shutter", type=float, default=None, help="seconds, e.g. 0.02 (default: the recipe's)")
    sp.add_argument("--tracking", type=float, default=None, help="1 = car sharp, 0 = static camera")
    sp.add_argument("--fstop", type=float, default=None, help="0 = no depth of field (default: the recipe's)")
    sp.add_argument("--no-blur", action="store_true", help="no motion blur anywhere (frozen exposure)")


def _exposure(a: argparse.Namespace) -> dict[str, float]:
    plan = _plan(a.shot)
    motion = {k: v for k, v in (("speed_kmh", a.speed), ("shutter_s", a.shutter), ("tracking", a.tracking))
              if v is not None}
    if a.no_blur:
        motion["blur"] = False
    camera = plan.camera if a.fstop is None else plan.camera.model_copy(update={"fstop": a.fstop})
    return exposure(camera, plan.motion.model_copy(update=motion))


async def cmd_car(a: argparse.Namespace) -> None:
    """Render the car; with --plate (a file, or "all"), once per plate with that plate projected
    for reflections, saved as cars/<plate stem>_rgba.png / .exr."""
    d, st, bl = shot_dir(a.shot), store(), blender()
    plates = sorted((d / "plates").glob("*.png")) if a.plate == "all" else [Path(a.plate)] if a.plate else [None]
    for p in plates:
        extra, names = {}, {"exr": "car.exr", "rgba_png": "car_rgba.png"}
        if p is not None:
            key = f"lab/{a.shot}/plates/{p.name}"
            await st.put(key, p.read_bytes(), "image/png")
            extra = {"plate_key": key, "plate_strength": a.plate_strength}
            (d / "cars").mkdir(exist_ok=True)
            names = {"exr": f"cars/{p.stem}.exr", "rgba_png": f"cars/{p.stem}_rgba.png"}
        args = car_motion_args(_exposure(a))  # one exposure with the plate's blur
        r = await timed(bl, "render_car", job_prefix=f"lab/{a.shot}/", samples=a.samples, spoke_count=a.spokes,
                        probes=a.probes, **args, **extra)
        await _fetch(st, r["keys"], d, names)
        save_json(d / (f"cars/{p.stem}.json" if p else "car.json"), r)


async def cmd_lookdev(a: argparse.Namespace) -> None:
    """Render the car under several `hdri:sun_rel` variants over neutral grey, one labelled sheet."""
    import cv2
    import numpy as np

    d, bl, st = shot_dir(a.shot), blender(), store()
    scene = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    cells = []
    try:
        for spec in a.variants.split(","):
            hdri_id, rel = spec.split(":")
            light = await cmd_light_apply(bl, {**SHOTS[a.shot], "hdri": hdri_id, "sun_rel_deg": float(rel)}, scene)
            r = await timed(bl, "render_car", job_prefix=f"lab/{a.shot}/lookdev/{hdri_id}_{rel}/", samples=a.samples,
                            spoke_count=FACTS.get("wheel_spoke_count", 0), resolution_x=960, resolution_y=640)
            car = load_car_render(await st.get(r["keys"]["rgba_png"]), None)
            img = composite(np.full(car.rgba.shape[:2] + (3,), 128, np.uint8), car).final
            ys, xs = np.nonzero(car.rgba[..., 3] > 8)
            m = 24
            img = img[max(0, ys.min() - m):ys.max() + m, max(0, xs.min() - m):xs.max() + m]
            img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])), interpolation=cv2.INTER_AREA)
            label = f"{hdri_id} rel {rel}: {light['sun_words'].split(';')[0][8:]}"
            cv2.putText(img, label[:78], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(img)
    finally:
        await cmd_light_apply(bl, SHOTS[a.shot], scene)
    h = max(c.shape[0] for c in cells)
    cells = [np.pad(c, ((0, h - c.shape[0]), (0, 0), (0, 0)), constant_values=255) for c in cells]
    rows = [np.hstack(cells[i:i + 2] + [np.full_like(cells[0], 255)] * (2 - len(cells[i:i + 2])))
            for i in range(0, len(cells), 2)]
    out = d / "lookdev.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("  lookdev ->", out)


async def cmd_liveries(a: argparse.Namespace) -> None:
    """Research real colours for the car, then render every livery option on the exact car in
    this shot's light (over its first plate) as one labelled sheet with sources."""
    import cv2
    import numpy as np

    from app.livery_research import research_liveries

    d, bl, st = shot_dir(a.shot), blender(), store()
    researched = await research_liveries(FACTS["name"], a.n) if a.n > 0 else {}
    options = {**FACTS.get("liveries", {}), **researched}
    save_json(LAB / "liveries.json", options)
    facts = {**FACTS, "liveries": options}
    plates = sorted((d / "plates").glob("*.png"))
    cells = []
    try:
        for name in options:
            await timed(bl, "set_materials", assignments=material_assignments(facts, name, lamps_on=False))
            r = await timed(bl, "render_car", job_prefix=f"lab/{a.shot}/liveries/{name}/", samples=a.samples,
                            motion_blur=False, resolution_x=960, resolution_y=640)
            car = load_car_render(await st.get(r["keys"]["rgba_png"]), await st.get(r["keys"]["exr"]))
            plate = decode_rgb(plates[0].read_bytes()) if plates else np.full(car.rgba.shape[:2] + (3,), 128, np.uint8)
            img = cv2.resize(composite(plate, car).final, (640, 427), interpolation=cv2.INTER_AREA)
            src = options[name].get("source_url", "product facts")
            for k, text in enumerate((name, re.sub(r"^https?://(www\.)?", "", src)[:60])):
                cv2.putText(img, text, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55 - 0.1 * k, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, text, (10, 24 + 22 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.55 - 0.1 * k, (255, 255, 255), 1, cv2.LINE_AA)
            cells.append(img)
    finally:
        await timed(bl, "set_materials", assignments=material_assignments(FACTS, lamps_on=False))
    rows = [np.hstack(cells[i:i + 3] + [np.full_like(cells[0], 255)] * (3 - len(cells[i:i + 3])))
            for i in range(0, len(cells), 3)]
    out = LAB / "liveries.jpg"
    cv2.imwrite(str(out), cv2.cvtColor(np.vstack(rows), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print("  liveries ->", out, "|", ", ".join(options))


async def cmd_preview(a: argparse.Namespace) -> None:
    d = shot_dir(a.shot)
    r = await timed(blender(), "render_preview", job_prefix=f"lab/{a.shot}/", index=0, resolution_x=a.width)
    (d / "preview.png").write_bytes(await store().get(r["key"]))
    print("  preview ->", d / "preview.png")


async def cmd_comp(a: argparse.Namespace) -> None:
    d = shot_dir(a.shot)
    default_car = (d / "car_rgba.png", d / "car.exr")
    scene = json.loads((d / "scene.json").read_text(encoding="utf-8"))
    cam = scene["frame"]["camera"]
    exp = _exposure(a)
    lens = Lens(focal_mm=cam["lens_mm"], focus_m=cam["distance_m"], f_number=exp["fstop"]) if exp["fstop"] > 0 else None
    tracking = None
    if exp["speed_kmh"] * exp["tracking"] > 0:
        tracking = (Intrinsics.from_camera(cam, 1536, 1024),
                    camera_displacement(cam["rotation_deg"], scene["place"]["heading_deg"],
                                        exp["speed_kmh"] * exp["tracking"], exp["shutter_s"]))
    ch = exr_layer(read_exr((d / "depth.exr").read_bytes()), "Depth", "Z")
    blender_m = ch.get("Z", next(iter(ch.values())))
    road = decode_png((d / "mask_road.png").read_bytes()) > 127
    plates = [Path(a.plate)] if a.plate else sorted((d / "plates").glob("*.png"))
    cdir = d / "comps"
    cdir.mkdir(exist_ok=True)
    for p in plates:
        own = (d / "cars" / f"{p.stem}_rgba.png", d / "cars" / f"{p.stem}.exr")
        rgba, exr = own if own[0].exists() else default_car
        car = load_car_render(rgba.read_bytes(), exr.read_bytes())
        rgb = decode_rgb(p.read_bytes())
        depth = estimate_plate_depth(rgb, blender_m, road)
        (cdir / f"{p.stem}_depth.png").write_bytes(encode_png(depth_guide(depth.metres)))
        print(f"  plate depth {p.name}: {depth.summary()}")
        c = composite(rgb, car, depth_m=depth.metres, lens=lens, tracking=tracking, light_wrap=a.wrap,
                      vignette=a.vignette, grain=a.grain)
        (cdir / p.name).write_bytes(encode_png(c.final))
        print("  comp ->", cdir / p.name)


async def cmd_call(a: argparse.Namespace) -> None:
    print(json.dumps(await blender().call(a.tool, **json.loads(a.args or "{}")), indent=1, default=str))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("scene", "guides", "car", "comp", "plates", "preview", "lookdev", "liveries"):
        sp = sub.add_parser(name)
        sp.add_argument("shot", choices=sorted(SHOTS))
        if name == "plates":
            sp.add_argument("--backend", default="openai")
            sp.add_argument("--model", default="gpt-image-2.6-flare")
            sp.add_argument("--quality", default="low", help="low for tests, high for finals")
            sp.add_argument("--complexity", default="low")
            sp.add_argument("--n", type=int, default=1)
            sp.add_argument("--refs", action="store_true", help="send assets/refs/<shot>/* as references")
            sp.add_argument("--max-refs", type=int, default=3)
        if name == "car":
            sp.add_argument("--samples", type=int, default=128)
            _exposure_args(sp)
            sp.add_argument("--spokes", type=int, default=10)
            sp.add_argument("--probes", action="store_true", help="chrome + grey balls ahead of the nose (HDRI check)")
            sp.add_argument("--plate", help='plate file to project for reflections, or "all"')
            sp.add_argument("--plate-strength", type=float, default=1.0)
        if name == "comp":
            sp.add_argument("--plate")
            _exposure_args(sp)
            sp.add_argument("--wrap", type=float, default=0.5)
            sp.add_argument("--vignette", type=float, default=0.12)
            sp.add_argument("--grain", type=float, default=0.008)
        if name == "scene":
            sp.add_argument("--livery", default=None, help="livery name from product facts (default: livery_default)")
            sp.add_argument("--lamps-on", action="store_true", help="light the car's lamps (default off: daylight)")
        if name == "preview":
            sp.add_argument("--width", type=int, default=1536)
        if name == "lookdev":
            sp.add_argument("--variants", required=True, help="comma list of hdri_id:sun_rel_deg")
            sp.add_argument("--samples", type=int, default=32)
        if name == "liveries":
            sp.add_argument("--n", type=int, default=4, help="real colours to research (0 = only product facts)")
            sp.add_argument("--samples", type=int, default=48)
    sc = sub.add_parser("call")
    sc.add_argument("tool")
    sc.add_argument("args", nargs="?")
    a = p.parse_args()
    asyncio.run(globals()[f"cmd_{a.cmd}"](a))


if __name__ == "__main__":
    main()
