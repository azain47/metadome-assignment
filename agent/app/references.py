"""Reference research: find photos of a car and a location, judge them with a vision model,
and turn the good car photos into `frame_shot` camera recipes.

Generic for any car and location; nothing here knows a particular model or place.

Flow (see scripts/dev/refs_lab.py):
    photos  = await search_photos("<car name>")            # Wikimedia Commons, then Openverse
    thumbs  = [await fetch_thumb(p) for p in photos]       # JPEG, long side <= 512
    ratings = await rate_car_photos(thumbs, "<car name>")  # one batched VLM call
    sheet   = Blender `render_pose_sheet` (grey renders of OUR car from many angles)
    poses   = await match_poses(best_thumbs, sheet_png, sheet_index["cells"])  # one VLM call
    recipe  = camera_recipe(rating, pose)                  # frame_shot args + sun_rel_deg
    plates  = await rate_location_photos(loc_thumbs, "<location description>")

Cost rule: every image is downscaled (long side <= 512 px, JPEG q85) and sent with
detail "low" before any model call; each judgement is one request with all images batched.
Token usage and dollars accumulate on the `Vlm` instance (`vlm.usage`) and are logged per call.

Network and model access are injectable: `fetch_json` / `fetch_bytes` callables for HTTP and a
`Vlm` (or anything with the same async `parse(prompt, images, schema)`) for the model.
"""

from __future__ import annotations

import asyncio
import base64
import html
import logging
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, TypeVar
from urllib.parse import unquote, urlsplit

import cv2
import numpy as np
from pydantic import BaseModel

from .config import settings
from .imaging.io import decode_rgb, encode_jpeg

log = logging.getLogger(__name__)

USER_AGENT = "trueshot-reference-research/0.1 (marketing stills research; contact: dev@trueshot.local)"
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
OPENVERSE_API = "https://api.openverse.org/v1/images/"
OPENVERSE_MAX_ANON = 20
MIN_SOURCE_SIDE = 600  # skip tiny originals: they are never useful as references
MAX_SIDE = 512  # cost rule: long side of every image sent to the model
JPEG_QUALITY = 85

FetchJson = Callable[[str, dict[str, Any]], Awaitable[Any]]
FetchBytes = Callable[[str], Awaitable[bytes]]


# ---------------------------------------------------------------------------
# Photo search


@dataclass
class RefPhoto:
    thumb_url: str
    page_url: str
    license: str
    author: str
    width: int
    height: int
    source: str  # "commons" | "openverse:<provider>"
    title: str = ""
    image_url: str = ""  # the original file (for dedupe and attribution)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


async def _http_json(url: str, params: dict[str, Any]) -> Any:
    import httpx

    async with httpx.AsyncClient(timeout=30.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as http:
        r = await http.get(url, params=params)
        r.raise_for_status()
        return r.json()


async def _http_bytes(url: str) -> bytes:
    import httpx

    async with httpx.AsyncClient(timeout=60.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True) as http:
        r = await http.get(url)
        r.raise_for_status()
        return r.content


def _plain(text: Any) -> str:
    """HTML fragment (Commons extmetadata) -> one line of plain text."""
    s = re.sub(r"<[^>]+>", " ", str(text or ""))
    return re.sub(r"\s+", " ", html.unescape(s)).strip()


def _file_key(url: str) -> str:
    """Dedupe key: the original file name, decoded and lower-cased."""
    name = unquote(urlsplit(url).path.rsplit("/", 1)[-1]).lower()
    return re.sub(r"[\s_]+", "_", name)


def commons_params(query: str, limit: int) -> dict[str, Any]:
    return {
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"{query} filetype:bitmap", "gsrnamespace": 6, "gsrlimit": min(50, max(1, limit)),
        "prop": "imageinfo", "iiprop": "url|size|mime|extmetadata", "iiurlwidth": 1024,
    }


def parse_commons(data: dict[str, Any]) -> list[RefPhoto]:
    """Commons `generator=search` + `imageinfo` response -> photos in search rank order."""
    pages = list(((data or {}).get("query") or {}).get("pages", {}).values())
    pages.sort(key=lambda p: p.get("index", 1 << 30))
    out = []
    for page in pages:
        info = (page.get("imageinfo") or [None])[0]
        if not info:
            continue
        if str(info.get("mime", "image/jpeg")) not in ("image/jpeg", "image/png", "image/webp", "image/tiff"):
            continue
        width, height = int(info.get("width") or 0), int(info.get("height") or 0)
        if max(width, height) < MIN_SOURCE_SIDE:
            continue
        meta = info.get("extmetadata") or {}
        out.append(RefPhoto(
            thumb_url=str(info.get("thumburl") or info.get("url") or ""),
            page_url=str(info.get("descriptionurl") or ""),
            license=_plain((meta.get("LicenseShortName") or {}).get("value")) or "unknown",
            author=_plain((meta.get("Artist") or {}).get("value")) or "unknown",
            width=width, height=height, source="commons",
            title=str(page.get("title", "")).removeprefix("File:"),
            image_url=str(info.get("url") or ""),
        ))
    return [p for p in out if p.thumb_url]


def parse_openverse(data: dict[str, Any]) -> list[RefPhoto]:
    out = []
    for item in (data or {}).get("results") or []:
        width, height = int(item.get("width") or 0), int(item.get("height") or 0)
        if width and height and max(width, height) < MIN_SOURCE_SIDE:
            continue
        lic = str(item.get("license") or "unknown").upper()
        if item.get("license_version"):
            lic = f"CC {lic} {item['license_version']}" if lic not in ("CC0", "PDM") else lic
        thumb = str(item.get("thumbnail") or item.get("url") or "")
        if not thumb:
            continue
        out.append(RefPhoto(
            thumb_url=thumb,
            page_url=str(item.get("foreign_landing_url") or ""),
            license=lic, author=str(item.get("creator") or "unknown"),
            width=width, height=height, source=f"openverse:{item.get('source') or item.get('provider') or '?'}",
            title=str(item.get("title") or ""), image_url=str(item.get("url") or ""),
        ))
    return out


def dedupe(photos: list[RefPhoto]) -> list[RefPhoto]:
    seen: set[str] = set()
    out = []
    for p in photos:
        keys = {k for k in (_file_key(p.image_url) if p.image_url else "", p.thumb_url, p.page_url) if k}
        if keys & seen:
            continue
        seen |= keys
        out.append(p)
    return out


async def search_photos(query: str, limit: int = 30, *, fetch_json: FetchJson | None = None) -> list[RefPhoto]:
    """Openly licensed photos for `query`: Wikimedia Commons first, then Openverse (Flickr and
    others; up to a third of `limit` is kept for it so a full Commons page cannot crowd it out).
    A failing source is logged and skipped. Results are deduped across sources."""
    fetch = fetch_json or _http_json
    commons: list[RefPhoto] = []
    openverse: list[RefPhoto] = []
    try:
        commons = dedupe(parse_commons(await fetch(COMMONS_API, commons_params(query, limit))))
    except Exception as ex:  # noqa: BLE001 - one source down must not stop the research
        log.warning("Commons search failed for %r: %s", query, ex)
    try:
        data = await fetch(OPENVERSE_API, {"q": query, "page_size": OPENVERSE_MAX_ANON, "mature": "false"})
        openverse = parse_openverse(data)
    except Exception as ex:  # noqa: BLE001
        log.warning("Openverse search failed for %r: %s", query, ex)
    merged = dedupe(commons + openverse)
    extra = [p for p in merged if p not in commons]
    keep_commons = max(limit - len(extra), limit - limit // 3)
    return (commons[:keep_commons] + extra)[:limit]


# ---------------------------------------------------------------------------
# Images


def downscale_jpeg(data: bytes, max_side: int = MAX_SIDE, quality: int = JPEG_QUALITY) -> bytes:
    """Any image bytes -> RGB JPEG whose long side is <= max_side (never upscaled)."""
    return encode_jpeg(_fit(decode_rgb(data), max_side), quality=quality)


def _fit(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1.0:
        return img
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


async def fetch_thumb(photo: RefPhoto, max_side: int = MAX_SIDE, *, fetch_bytes: FetchBytes | None = None) -> bytes:
    """The photo's thumbnail as a downscaled JPEG (long side <= max_side)."""
    return downscale_jpeg(await (fetch_bytes or _http_bytes)(photo.thumb_url), max_side)


async def fetch_thumbs(photos: list[RefPhoto], max_side: int = MAX_SIDE, *, concurrency: int = 4,
                       fetch_bytes: FetchBytes | None = None) -> list[bytes | None]:
    """`fetch_thumb` for many photos (None where a download or decode failed)."""
    sem = asyncio.Semaphore(concurrency)

    async def one(p: RefPhoto) -> bytes | None:
        async with sem:
            try:
                return await fetch_thumb(p, max_side, fetch_bytes=fetch_bytes)
            except Exception as ex:  # noqa: BLE001
                log.warning("thumbnail failed %s: %s", p.thumb_url, ex)
                return None

    return list(await asyncio.gather(*(one(p) for p in photos)))


def image_size(data: bytes) -> tuple[int, int]:
    h, w = decode_rgb(data).shape[:2]
    return w, h


# ---------------------------------------------------------------------------
# Vision model


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    per_call: list[dict[str, Any]] = field(default_factory=list)

    def add(self, label: str, usage: Any) -> dict[str, Any]:
        inp = int(getattr(usage, "input_tokens", 0) or 0)
        out = int(getattr(usage, "output_tokens", 0) or 0)
        details = getattr(usage, "input_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
        cost = ((inp - cached) * settings.llm_price_in + cached * settings.llm_price_cached
                + out * settings.llm_price_out) / 1e6
        self.calls += 1
        self.input_tokens += inp
        self.cached_tokens += cached
        self.output_tokens += out
        self.cost_usd += cost
        rec = {"label": label, "input_tokens": inp, "cached_tokens": cached, "output_tokens": out,
               "cost_usd": round(cost, 5)}
        self.per_call.append(rec)
        return rec

    def summary(self) -> dict[str, Any]:
        return {"calls": self.calls, "input_tokens": self.input_tokens, "cached_tokens": self.cached_tokens,
                "output_tokens": self.output_tokens, "cost_usd": round(self.cost_usd, 5), "per_call": self.per_call}


T = TypeVar("T", bound=BaseModel)


class VisionModel(Protocol):
    async def parse(self, prompt: str, images: list[tuple[str, bytes]], schema: type[T], *, label: str = "") -> T: ...


def vlm_model_name() -> str:
    """VLM_MODEL, else LLM_MODEL (read at call time so a changed environment applies)."""
    return os.environ.get("VLM_MODEL", "").strip() or os.environ.get("LLM_MODEL", "").strip() or settings.llm_model


class Vlm:
    """OpenAI Responses API with structured output. `images` are (caption, bytes) pairs: each
    image is downscaled (cost rule) and preceded by its caption, so answers can cite it."""

    def __init__(self, client: Any = None, model: str | None = None, *, reasoning_effort: str | None = "low",
                 max_side: int = MAX_SIDE) -> None:
        self._client = client
        self.model = model or vlm_model_name()
        self.reasoning_effort = reasoning_effort
        self.max_side = max_side
        self.usage = Usage()

    @property
    def client(self) -> Any:
        if self._client is None:
            import openai

            key = os.environ.get("OPENAI_API_KEY", "").strip() or settings.openai_api_key
            self._client = openai.AsyncOpenAI(api_key=key, base_url=settings.llm_api_base or None, max_retries=2,
                                              timeout=180.0)
        return self._client

    def build_input(self, prompt: str, images: list[tuple[str, bytes]]) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": prompt}]
        for caption, data in images:
            jpg = downscale_jpeg(data, self.max_side)
            content.append({"type": "input_text", "text": caption})
            content.append({"type": "input_image", "detail": "low",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")})
        return [{"role": "user", "content": content}]

    async def parse(self, prompt: str, images: list[tuple[str, bytes]], schema: type[T], *, label: str = "") -> T:
        kwargs: dict[str, Any] = {"model": self.model, "input": self.build_input(prompt, images), "text_format": schema}
        if self.reasoning_effort:
            kwargs["reasoning"] = {"effort": self.reasoning_effort}
        resp = await self.client.responses.parse(**kwargs)
        rec = self.usage.add(label or schema.__name__, getattr(resp, "usage", None))
        log.info("vlm %s: %d images, %s", label or schema.__name__, len(images), rec)
        parsed = resp.output_parsed
        if parsed is None:
            raise RuntimeError(f"{self.model} returned no parsable {schema.__name__}")
        return parsed


_default_vlm: Vlm | None = None


def default_vlm() -> Vlm:
    global _default_vlm
    if _default_vlm is None:
        _default_vlm = Vlm()
    return _default_vlm


# ---------------------------------------------------------------------------
# Car photo ratings

View = Literal["front34", "rear34", "side", "front", "rear", "top", "detail"]
SunPosition = Literal[
    "behind_camera", "behind_camera_left", "behind_camera_right", "left", "right",
    "front_left", "front_right", "backlit", "overhead", "no_direct_sun",
]


class CarShotRating(BaseModel):
    index: int
    is_the_car: bool
    is_marketing_quality: int  # 1 (snapshot) .. 5 (professional campaign image)
    view: View
    camera_height: Literal["low", "eye", "high"]
    lens_look: Literal["wide", "normal", "tele"]
    car_width_frac: float  # car's width in the frame / frame width
    frame_x: float  # car centre, fraction from the left
    frame_y: float  # car centre, fraction from the top
    sun_position: SunPosition  # relative to the camera
    light_hardness: Literal["hard", "medium", "soft"]
    light_words: str
    motion: Literal["static", "rolling", "panning"]
    why: str


class _CarRatings(BaseModel):
    photos: list[CarShotRating]


CAR_PROMPT = """You judge reference photos for a car marketing shoot of this exact car: {car}.
There are {n} photos, numbered 0..{last}. Return exactly one entry per photo, in order, with `index` set.
- is_the_car: true only if the main subject is clearly this exact model and variant (not a related model,
  a toy, a drawing, a render, or a crowd scene where the car is small).
- is_marketing_quality 1-5: 5 = professional campaign/press photo (clean composition, deliberate light,
  uncluttered background, whole car sharp); 1 = casual snapshot, cluttered, cropped car, bad light.
- view: front34 / rear34 / side / front / rear / top / detail (detail = close-up of a part or interior).
- camera_height: low (below the car's roof, near bumper height), eye (standing person), high (above).
- lens_look: wide (strong perspective, exaggerated near parts), normal, tele (flat, compressed perspective).
- car_width_frac: width of the car's bounding box as a fraction of the image width (0-1).
- frame_x, frame_y: centre of the car's bounding box as fractions of image width (from left) and height (from top).
- sun_position: where the main light/sun is relative to the camera; light_hardness; light_words: 3-8 words.
- motion: static, rolling (car moving, sharp background), panning (car sharp, background streaked).
- why: one short line."""


async def rate_car_photos(images: list[bytes], car_name: str, *, vlm: VisionModel | None = None) -> list[CarShotRating]:
    """One batched VLM call; ratings in the order of `images` (a missing answer is filled with a
    rejecting rating)."""
    if not images:
        return []
    model = vlm or default_vlm()
    prompt = CAR_PROMPT.format(car=car_name, n=len(images), last=len(images) - 1)
    res = await model.parse(prompt, [(f"Photo {i}:", b) for i, b in enumerate(images)], _CarRatings,
                            label="rate_car_photos")
    by_index = {r.index: r for r in res.photos if 0 <= r.index < len(images)}
    return [by_index.get(i) or _rejected_car(i) for i in range(len(images))]


def _rejected_car(i: int) -> CarShotRating:
    return CarShotRating(index=i, is_the_car=False, is_marketing_quality=1, view="detail", camera_height="eye",
                         lens_look="normal", car_width_frac=0.0, frame_x=0.5, frame_y=0.5,
                         sun_position="no_direct_sun", light_hardness="soft", light_words="",
                         motion="static", why="no answer from the model")


def car_score(r: CarShotRating) -> float:
    """Ranking key for camera references: the right car, whole-car views, quality first."""
    if not r.is_the_car or r.view in ("detail", "top"):
        return 0.0
    return float(r.is_marketing_quality) + (0.3 if r.view in ("front34", "rear34") else 0.0)


# ---------------------------------------------------------------------------
# Pose matching against the Blender pose sheet

NOSE = Literal["left", "right", "toward_camera", "away_from_camera"]


class PoseMatch(BaseModel):
    azimuth_deg: float
    elevation_deg: float
    confidence: float  # 0..1
    cell_index: int
    mirrored: bool  # the chosen render showed the car's other side; azimuth was mirrored
    nose_points: NOSE
    why: str = ""


class _PoseAnswer(BaseModel):
    photo: int
    nose_points: NOSE  # where the car's nose points in the photo
    side_visible: Literal["left", "right", "none"]  # the car's own left/right side
    camera_height: Literal["low", "eye", "high"]
    cell: int
    confidence: float
    why: str


class _PoseAnswers(BaseModel):
    matches: list[_PoseAnswer]


POSE_PROMPT = """The numbered grey renders show OUR 3D model of the car from known camera angles (white background;
each number is a render id). Grids are grouped by camera elevation (written above each grid).
Then come {n} photos of the real car, numbered 0..{last}.
For each photo pick the render id whose viewing direction matches the photo best:
1) nose_points: where the car's nose points in the photo frame (left / right / toward_camera / away_from_camera);
2) side_visible: which of the car's own sides (driver sitting facing forward: left or right) faces the camera;
3) how much front/rear versus side is visible (e.g. a 3/4 view showing mostly side vs mostly nose);
4) camera height: a photo looking down onto the roof/bonnet matches the higher elevation, a near-ground or
   level view matches the lower one.
The chosen render must have the nose pointing the same way in the frame as the photo. Ignore colour,
background, lens distortion and crop. confidence 0..1. Return one entry per photo with `photo` set."""


def _nose_of(az: float) -> str:
    """Where the car's nose points in a frame_shot render at azimuth `az` (0 = front view)."""
    a = az % 360.0
    if a < 15.0 or a > 345.0:
        return "toward_camera"
    if 165.0 < a < 195.0:
        return "away_from_camera"
    return "left" if a < 180.0 else "right"


def pose_grids(pose_sheet: bytes, cells: list[dict[str, Any]], *, cols: int = 4,
               tile_w: int = 128, label_px: int = 18) -> list[tuple[str, bytes]]:
    """Re-tile the sheet's cells into compact numbered grids, one per elevation, sized for the
    model's low-detail view (512 px wide). The sheet's own small labels are cropped off."""
    sheet = decode_rgb(pose_sheet)
    by_el: dict[float, list[dict[str, Any]]] = {}
    for c in cells:
        by_el.setdefault(float(c["elevation_deg"]), []).append(c)
    grids = []
    for el in sorted(by_el):
        group = by_el[el]
        tiles = []
        for c in group:
            x, y, w, h = (int(c[k]) for k in ("x", "y", "w", "h"))
            crop = sheet[y + label_px:y + h, x:x + w]
            tile_h = max(1, round(tile_w * crop.shape[0] / crop.shape[1]))
            tile = cv2.resize(crop, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            tile = np.ascontiguousarray(tile)
            cv2.putText(tile, str(c["index"]), (3, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 0, 0), 2, cv2.LINE_AA)
            cv2.rectangle(tile, (0, 0), (tile_w - 1, tile_h - 1), (150, 150, 150), 1)
            tiles.append(tile)
        th = max(t.shape[0] for t in tiles)
        rows = (len(tiles) + cols - 1) // cols
        grid = np.full((rows * th, cols * tile_w, 3), 255, dtype=np.uint8)
        for i, t in enumerate(tiles):
            r, cc = divmod(i, cols)
            grid[r * th:r * th + t.shape[0], cc * tile_w:(cc + 1) * tile_w] = t
        grids.append((f"Renders at camera elevation {el:g} degrees (ids "
                      f"{', '.join(str(c['index']) for c in group)}):", encode_jpeg(grid, quality=JPEG_QUALITY)))
    return grids


def resolve_pose(cell: dict[str, Any], nose_points: str, confidence: float, why: str = "") -> PoseMatch:
    """PoseMatch from the chosen cell. If the cell's nose direction contradicts what the model saw
    in the photo, the photo shows the car's other side: mirror the azimuth (az -> 360 - az)."""
    az = float(cell["azimuth_deg"]) % 360.0
    mirrored = False
    cell_nose = _nose_of(az)
    if {cell_nose, nose_points} == {"left", "right"}:
        az, mirrored = (360.0 - az) % 360.0, True
        confidence *= 0.8
    elif cell_nose != nose_points:
        confidence *= 0.6  # e.g. a 3/4 render for a photo the model called a straight front view
    return PoseMatch(azimuth_deg=az, elevation_deg=float(cell["elevation_deg"]),
                     confidence=round(max(0.0, min(1.0, confidence)), 3), cell_index=int(cell["index"]),
                     mirrored=mirrored, nose_points=nose_points, why=why)  # type: ignore[arg-type]


async def match_poses(photos: list[bytes], pose_sheet: bytes, cells: list[dict[str, Any]], *,
                      vlm: VisionModel | None = None) -> list[PoseMatch | None]:
    """One VLM call for all `photos`: the best matching pose-sheet cell for each (None when the
    model gave no valid answer). `cells` come from render_pose_sheet's index (index, x, y, w, h,
    azimuth_deg, elevation_deg)."""
    if not photos:
        return []
    model = vlm or default_vlm()
    by_index = {int(c["index"]): c for c in cells}
    images = pose_grids(pose_sheet, cells) + [(f"Photo {i}:", p) for i, p in enumerate(photos)]
    res = await model.parse(POSE_PROMPT.format(n=len(photos), last=len(photos) - 1), images, _PoseAnswers,
                            label="match_poses")
    out: list[PoseMatch | None] = [None] * len(photos)
    for a in res.matches:
        if 0 <= a.photo < len(photos) and a.cell in by_index and out[a.photo] is None:
            out[a.photo] = resolve_pose(by_index[a.cell], a.nose_points, a.confidence, a.why)
    return out


async def match_pose(photo: bytes, pose_sheet: bytes, cells: list[dict[str, Any]], *,
                     vlm: VisionModel | None = None) -> PoseMatch:
    match = (await match_poses([photo], pose_sheet, cells, vlm=vlm))[0]
    if match is None:
        raise RuntimeError("the model did not pick a valid pose-sheet cell")
    return match


# ---------------------------------------------------------------------------
# Camera recipe

FOCAL_MM = {"wide": 35.0, "normal": 50.0}
ELEVATION_OF_HEIGHT = {"low": 3.0, "eye": 8.0, "high": 20.0}
# sun_rel_deg convention (scripts/dev/lab.py): sun relative to the camera's view direction,
# 0 = ahead of the camera (backlit), +90 = to the right, 180 = behind the camera.
SUN_REL_DEG: dict[str, float | None] = {
    "backlit": 0.0, "front_right": 45.0, "right": 90.0, "behind_camera_right": 135.0, "behind_camera": 180.0,
    "behind_camera_left": 225.0, "left": 270.0, "front_left": 315.0, "overhead": None, "no_direct_sun": None,
}
VIEW_AZIMUTH = {"front": 0.0, "front34": 35.0, "side": 90.0, "rear34": 145.0, "rear": 180.0, "top": 35.0,
                "detail": 35.0}


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def tele_focal(rating: CarShotRating) -> float:
    """85 mm for tight tele looks, 135 mm for panning / compressed long-lens looks."""
    return 135.0 if rating.motion == "panning" or rating.car_width_frac < 0.45 else 85.0


def camera_recipe(rating: CarShotRating, pose: PoseMatch | None) -> dict[str, Any]:
    """frame_shot arguments reproducing the photo's camera, plus `sun_rel_deg` (None when the
    photo has no direct sun). The angle comes from the pose match; without one, from the view."""
    focal = FOCAL_MM.get(rating.lens_look) or tele_focal(rating)
    if pose is not None:
        az, el = pose.azimuth_deg, pose.elevation_deg
        if rating.camera_height == "low":
            el = min(el, ELEVATION_OF_HEIGHT["low"])
    else:
        az, el = VIEW_AZIMUTH[rating.view], ELEVATION_OF_HEIGHT[rating.camera_height]
    return {
        "azimuth_deg": round(az % 360.0, 1),
        "elevation_deg": round(el, 1),
        "focal_mm": focal,
        "car_width_frac": round(_clamp(rating.car_width_frac or 0.55, 0.25, 0.92), 3),
        "frame_x": round(_clamp(rating.frame_x, 0.2, 0.8), 3),
        "frame_y": round(_clamp(rating.frame_y, 0.25, 0.8), 3),
        "sun_rel_deg": SUN_REL_DEG.get(rating.sun_position),
    }


# ---------------------------------------------------------------------------
# Location photos (plate references)


class LocationRating(BaseModel):
    index: int
    matches_location: bool
    realistic_photo: bool  # a real photograph, not a painting/map/render/collage
    has_cars: bool
    has_people: bool
    has_text: bool  # captions, signs dominating, watermarks, borders
    has_road: bool
    landscape: bool  # wider than tall (overwritten from the pixels)
    score: int  # 1..5 usefulness as a background-plate reference
    why: str

    @property
    def usable(self) -> bool:
        return (self.matches_location and self.realistic_photo and not self.has_cars and not self.has_people
                and not self.has_text and self.landscape)


class _LocationRatings(BaseModel):
    photos: list[LocationRating]


LOCATION_PROMPT = """You judge reference photos for the BACKGROUND of a car advertisement (the car is added later).
Wanted location: {where}.
There are {n} photos, numbered 0..{last}. Return exactly one entry per photo, in order, with `index` set.
- matches_location: the photo plausibly shows this kind of place.
- realistic_photo: a real photograph (not a painting, map, render, collage or heavily filtered image).
- has_cars / has_people: any vehicles or people visible; has_text: visible captions, watermarks, borders or
  dominant signage; has_road: a drivable road is visible.
- score 1-5: usefulness as a backdrop reference for a car ad: 5 = clean, striking landscape with a road the
  car could stand on, good light, no clutter; 1 = useless.
- why: one short line."""


async def rate_location_photos(images: list[bytes], location_description: str, *,
                               vlm: VisionModel | None = None) -> list[LocationRating]:
    """One batched VLM call; ratings in the order of `images`. `landscape` is measured from the
    pixels, not asked."""
    if not images:
        return []
    model = vlm or default_vlm()
    prompt = LOCATION_PROMPT.format(where=location_description, n=len(images), last=len(images) - 1)
    res = await model.parse(prompt, [(f"Photo {i}:", b) for i, b in enumerate(images)], _LocationRatings,
                            label="rate_location_photos")
    by_index = {r.index: r for r in res.photos if 0 <= r.index < len(images)}
    out = []
    for i, data in enumerate(images):
        r = by_index.get(i) or LocationRating(index=i, matches_location=False, realistic_photo=False, has_cars=False,
                                              has_people=False, has_text=False, has_road=False, landscape=False,
                                              score=1, why="no answer from the model")
        w, h = image_size(data)
        out.append(r.model_copy(update={"landscape": w > h, "score": int(_clamp(r.score, 1, 5))}))
    return out


def location_score(r: LocationRating) -> float:
    return (float(r.score) + (0.5 if r.has_road else 0.0)) if r.usable else 0.0
