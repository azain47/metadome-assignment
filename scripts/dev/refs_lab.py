"""Reference-research lab: photos of a car and a location -> ratings, pose matches, camera recipes.

Runs the agent's own `app.references` step end to end, by hand (no graph):

    cd agent && uv run python ../scripts/dev/refs_lab.py "McLaren P1 GTR" "Ladakh road mountains"
    uv run python ../scripts/dev/refs_lab.py "<car>" "<location>" --load-car cars/<id>.glb --length-m 4.6

1. search Commons/Openverse for the car, download thumbnails (JPEG, long side <= 512), rate them
   with one batched VLM call;
2. render_pose_sheet in Blender (the loaded car; `--load-car` loads one first), match the top car
   photos against it with one VLM call, turn each into frame_shot args (`camera_recipe`);
3. search the location, rate its photos as plate references with one VLM call.

Writes outputs/lab/refs/<slug>/: car/ and location/ thumbnails, ratings.json, recipes.json,
pose_sheet.png/json, match_<n>.jpg (photo beside its matched render) and summary.json; spend is
appended to outputs/lab/spend.jsonl. Talks to BLENDER_MCP_URL (default 127.0.0.1:8765) and S3.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
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
sys.path[:0] = [str(ROOT / "agent")]

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app import references as refs  # noqa: E402
from app.blender import BlenderClient  # noqa: E402
from app.config import settings  # noqa: E402
from app.imaging.io import decode_rgb, encode_jpeg  # noqa: E402
from app.storage import S3Store  # noqa: E402

LAB = ROOT / "outputs" / "lab"


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")[:60]


def save_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=1, default=str), encoding="utf-8")


def side_by_side(photo: bytes, sheet: np.ndarray, cell: dict[str, Any], height: int = 300) -> bytes:
    """Photo next to its matched render (for eyeballing the pose match)."""
    a = decode_rgb(photo)
    x, y, w, h = (int(cell[k]) for k in ("x", "y", "w", "h"))
    b = sheet[y:y + h, x:x + w]
    tiles = [cv2.resize(t, (max(1, round(t.shape[1] * height / t.shape[0])), height), interpolation=cv2.INTER_AREA)
             for t in (a, b)]
    return encode_jpeg(np.concatenate([tiles[0], np.full((height, 6, 3), 255, np.uint8), tiles[1]], axis=1), 88)


async def gather_photos(query: str, limit: int, out_dir: Path, prefix: str) -> tuple[list[refs.RefPhoto], list[bytes]]:
    photos = await refs.search_photos(query, limit)
    thumbs = await refs.fetch_thumbs(photos)
    kept = [(p, t) for p, t in zip(photos, thumbs) if t is not None]
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (_p, t) in enumerate(kept):
        (out_dir / f"{prefix}_{i:02d}.jpg").write_bytes(t)
    print(f"  {query!r}: {len(photos)} found, {len(kept)} thumbnails")
    return [p for p, _ in kept], [t for _, t in kept]


async def pose_sheet(bl: BlenderClient, store: S3Store, out: Path, job_prefix: str) -> tuple[bytes, dict[str, Any]]:
    t0 = time.monotonic()
    res = await bl.call("render_pose_sheet", job_prefix=job_prefix)
    print(f"  render_pose_sheet {time.monotonic() - t0:.1f}s (blender {res.get('total_s')}s, "
          f"{len(res.get('cells', []))} cells)")
    png = await store.get(res["keys"]["sheet"])
    index = json.loads(await store.get(res["keys"]["index"]))
    (out / "pose_sheet.png").write_bytes(png)
    save_json(out / "pose_sheet.json", index)
    return png, index


async def run(a: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.WARNING)
    out = LAB / "refs" / slug(f"{a.car} {a.location}")
    out.mkdir(parents=True, exist_ok=True)
    vlm = refs.Vlm()
    print(f"model {vlm.model} -> {out}")

    print("car photos")
    car_photos, car_thumbs = await gather_photos(a.car, a.limit, out / "car", "car")
    car_ratings = await refs.rate_car_photos(car_thumbs, a.car, vlm=vlm)
    ranked = sorted(range(len(car_photos)), key=lambda i: -refs.car_score(car_ratings[i]))
    top = [i for i in ranked if refs.car_score(car_ratings[i]) > 0][: a.top]

    print("pose sheet")
    bl = BlenderClient(settings.blender_mcp_url)
    store = S3Store(endpoint=settings.s3_endpoint, public_endpoint=settings.s3_public_endpoint,
                    access_key=settings.s3_access_key, secret_key=settings.s3_secret_key, bucket=settings.s3_bucket)
    if a.load_car:
        await bl.call("load_car", car_key=a.load_car, length_m=a.length_m)
    sheet_png, index = await pose_sheet(bl, store, out, f"lab/refs/{out.name}")
    poses = await refs.match_poses([car_thumbs[i] for i in top], sheet_png, index["cells"], vlm=vlm)
    sheet = decode_rgb(sheet_png)
    cells = {c["index"]: c for c in index["cells"]}
    recipes = []
    for n, (i, pose) in enumerate(zip(top, poses)):
        recipe = refs.camera_recipe(car_ratings[i], pose)
        recipes.append({"photo": f"car/car_{i:02d}.jpg", "source": car_photos[i].to_dict(),
                        "rating": car_ratings[i].model_dump(), "pose": pose.model_dump() if pose else None,
                        "frame_shot": recipe})
        if pose is not None:
            (out / f"match_{n}.jpg").write_bytes(side_by_side(car_thumbs[i], sheet, cells[pose.cell_index]))

    print("location photos")
    loc_photos, loc_thumbs = await gather_photos(a.location, a.limit, out / "location", "loc")
    loc_ratings = await refs.rate_location_photos(loc_thumbs, a.location, vlm=vlm)
    loc_rank = sorted(range(len(loc_photos)), key=lambda i: -refs.location_score(loc_ratings[i]))
    loc_top = [i for i in loc_rank if refs.location_score(loc_ratings[i]) > 0][: a.top]

    save_json(out / "ratings.json", {
        "car": [{"photo": f"car/car_{i:02d}.jpg", "source": p.to_dict(), "rating": r.model_dump(),
                 "score": refs.car_score(r)} for i, (p, r) in enumerate(zip(car_photos, car_ratings))],
        "location": [{"photo": f"location/loc_{i:02d}.jpg", "source": p.to_dict(), "rating": r.model_dump(),
                      "usable": r.usable, "score": refs.location_score(r)}
                     for i, (p, r) in enumerate(zip(loc_photos, loc_ratings))],
    })
    save_json(out / "recipes.json", recipes)
    summary = {
        "car": a.car, "location": a.location, "model": vlm.model,
        "car_photos": len(car_photos), "car_is_the_car": sum(r.is_the_car for r in car_ratings),
        "location_photos": len(loc_photos), "location_usable": sum(r.usable for r in loc_ratings),
        "top_car": recipes,
        "top_location": [{"photo": f"location/loc_{i:02d}.jpg", "source": loc_photos[i].to_dict(),
                          "rating": loc_ratings[i].model_dump()} for i in loc_top],
        "usage": vlm.usage.summary(),
    }
    save_json(out / "summary.json", summary)
    LAB.mkdir(parents=True, exist_ok=True)
    with open(LAB / "spend.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": time.time(), "step": "refs_lab", "dir": out.name, "model": vlm.model,
                             **{k: v for k, v in vlm.usage.summary().items() if k != "per_call"}}) + "\n")

    print(f"\ncar: {summary['car_photos']} photos, {summary['car_is_the_car']} judged to be the car")
    for r in recipes:
        rt, ps = r["rating"], r["pose"]
        print(f"  {r['photo']}  q{rt['is_marketing_quality']} {rt['view']} {rt['lens_look']} {rt['camera_height']} "
              f"sun={rt['sun_position']}  pose={ps and (ps['azimuth_deg'], ps['elevation_deg'], ps['confidence'], ps['mirrored'])}"
              f"\n    frame_shot {r['frame_shot']}\n    {rt['why']}  [{r['source']['license']}, {r['source']['author'][:40]}]")
    print(f"location: {summary['location_photos']} photos, {summary['location_usable']} usable as plate refs")
    for t in summary["top_location"]:
        print(f"  {t['photo']}  score {t['rating']['score']} road={t['rating']['has_road']}  {t['rating']['why']}"
              f"  [{t['source']['license']}]")
    u = vlm.usage
    print(f"usage: {u.calls} calls, {u.input_tokens} in / {u.output_tokens} out tokens, ${u.cost_usd:.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("car", help="car name as people would search it, e.g. 'Porsche 911 GT3 RS'")
    p.add_argument("location", help="location search query, e.g. 'mountain pass road'")
    p.add_argument("--limit", type=int, default=30, help="photos per search")
    p.add_argument("--top", type=int, default=3, help="car photos to pose-match / location photos to keep")
    p.add_argument("--load-car", default="", help="S3 car key to load before the pose sheet")
    p.add_argument("--length-m", type=float, default=0.0)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
