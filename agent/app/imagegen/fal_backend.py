"""fal.ai backend: queue API through `fal_client` (submit -> poll status -> fetch result).

Inputs are uploaded with `client.upload` (fal CDN) instead of data URIs. Retries (with backoff)
on 429/5xx/timeouts/connection errors; 4xx are not retried and a 422 content-policy refusal
raises `ContentPolicyError`. Outputs flagged by `has_nsfw_concepts` or all-black (the safety
checker's replacement image) are rejected.

Supporting a new endpoint = one builder function + one entry in `ENDPOINTS` (endpoint-id prefix
-> (builder, cost estimator)); unknown ids use the generic multi-image edit shape.
"""

from __future__ import annotations

import asyncio
import os
import re
import time
from collections.abc import Awaitable, Callable

import fal_client
import httpx

from .base import (
    ContentPolicyError,
    ImageGenError,
    MissingCredentialsError,
    PlateRequest,
    PlateResult,
    guide_at_size,
    normalize_output,
    reference_at_aspect,
)
from .openai_backend import IMAGE_IN_PER_M, IN_TOKENS_1536x1024, OUT_1536x1024

DEFAULT_MODEL = "openai/gpt-image-2.6/flare/edit"

Upload = Callable[[bytes, str, str], Awaitable[str]]  # (data, content_type, file_name) -> url
Builder = Callable[[PlateRequest, Upload, dict], Awaitable[dict]]
CostFn = Callable[[PlateRequest], float]


# ---------- shared pieces ----------
async def _edit_images(req: PlateRequest, upload: Upload) -> list[str]:
    """Image 1 = grey guide at the output size, then the references cropped to its aspect."""
    urls = [await upload(guide_at_size(req.guide_grey, req.width, req.height), "image/png", "guide_grey.png")]
    for i, ref in enumerate(req.references):
        urls.append(await upload(reference_at_aspect(ref, req.width, req.height), "image/jpeg", f"reference_{i + 1}.jpg"))
    return urls


_ASPECTS = ["21:9", "16:9", "3:2", "4:3", "5:4", "1:1", "4:5", "3:4", "2:3", "9:16"]


def _aspect_ratio(w: int, h: int) -> str:
    def val(a: str) -> float:
        x, y = a.split(":")
        return int(x) / int(y)

    return min(_ASPECTS, key=lambda a: abs(val(a) - w / h))


def _with_seed(args: dict, req: PlateRequest) -> dict:
    if req.seed is not None:
        args["seed"] = req.seed
    return args


# ---------- request builders (one per endpoint family) ----------
async def build_gpt_image_edit(req: PlateRequest, upload: Upload, opts: dict) -> dict:
    """`openai/gpt-image-*/edit`: same contract as OpenAI direct; `reasoning_effort` = `complexity`."""
    return {
        "prompt": req.prompt,
        "image_urls": await _edit_images(req, upload),
        "image_size": {"width": req.width, "height": req.height},
        "quality": opts.get("quality", "high"),
        "reasoning_effort": opts.get("reasoning_effort", "low"),
        "num_images": req.n,
        "output_format": "png",
    }


async def build_zimage_controlnet(req: PlateRequest, upload: Upload, opts: dict) -> dict:
    """`fal-ai/z-image/turbo/controlnet`: one control image used as-is, no references."""
    guide = opts.get("control_guide", "depth")
    data = req.guide_depth if guide == "depth" else req.guide_lines
    url = await upload(guide_at_size(data, req.width, req.height), "image/png", f"guide_{guide}.png")
    return _with_seed(
        {
            "prompt": req.prompt,
            "image_url": url,
            "preprocess": "none",
            "control_scale": opts.get("control_scale", 0.7),
            "control_start": 0.0,
            "control_end": opts.get("control_end", 0.8),
            "image_size": {"width": req.width, "height": req.height},
            "num_inference_steps": 8,
            "num_images": min(req.n, 4),
            "output_format": "png",
        },
        req,
    )


async def build_nano_banana_edit(req: PlateRequest, upload: Upload, opts: dict) -> dict:
    """`fal-ai/nano-banana*/edit`: size by `aspect_ratio` + `resolution`; output is resized after."""
    return _with_seed(
        {
            "prompt": req.prompt,
            "image_urls": await _edit_images(req, upload),
            "aspect_ratio": _aspect_ratio(req.width, req.height),
            "resolution": opts.get("resolution", "2K"),
            "num_images": req.n,
            "output_format": "png",
        },
        req,
    )


async def build_muse_edit(req: PlateRequest, upload: Upload, opts: dict) -> dict:
    """`meta/muse-image/edit`: `aspect_ratio`, no seed."""
    return {
        "prompt": req.prompt,
        "image_urls": await _edit_images(req, upload),
        "aspect_ratio": _aspect_ratio(req.width, req.height),
        "num_images": req.n,
    }


async def build_flux3_klein_edit(req: PlateRequest, upload: Upload, opts: dict) -> dict:
    args = await build_generic_edit(req, upload, opts)
    args["enable_prompt_expansion"] = False
    return args


async def build_generic_edit(req: PlateRequest, upload: Upload, opts: dict) -> dict:
    """Generic multi-image edit shape (Seedream, Qwen, FLUX.2 ...): guide first, then references."""
    return _with_seed(
        {
            "prompt": req.prompt,
            "image_urls": await _edit_images(req, upload),
            "image_size": {"width": req.width, "height": req.height},
            "num_images": req.n,
            "output_format": "png",
        },
        req,
    )


# ---------- cost estimates (fal list prices read 2026-11-16) ----------
def _mp(req: PlateRequest) -> float:
    return req.width * req.height / 1e6


def _cost_gpt_image(req: PlateRequest) -> float:
    area = req.width * req.height / (1536 * 1024)
    per_out = OUT_1536x1024["high"] * area
    inputs = (1 + len(req.references)) * IN_TOKENS_1536x1024 * area * IMAGE_IN_PER_M / 1e6
    return req.n * (per_out + inputs)  # fal bills inputs per output [INFERENCE: conservative]


ENDPOINTS: dict[str, tuple[Builder, CostFn]] = {
    "openai/gpt-image": (build_gpt_image_edit, _cost_gpt_image),
    "fal-ai/z-image/turbo/controlnet/lora": (build_zimage_controlnet, lambda r: min(r.n, 4) * 0.01 * _mp(r)),
    "fal-ai/z-image/turbo/controlnet": (build_zimage_controlnet, lambda r: min(r.n, 4) * 0.0065 * _mp(r)),
    "fal-ai/nano-banana-pro/edit": (build_nano_banana_edit, lambda r: r.n * 0.15),
    "fal-ai/nano-banana-2/edit": (build_nano_banana_edit, lambda r: r.n * 0.12),
    "meta/muse-image/edit": (build_muse_edit, lambda r: r.n * 0.01),
    "blackforestlabs/flux-3/klein/edit-image": (build_flux3_klein_edit, lambda r: r.n * 0.011),
    "bytedance/seedream/v5.5/edit": (build_generic_edit, lambda r: r.n * (0.03 + 0.012 * (1 + len(r.references)))),
}
GENERIC: tuple[Builder, CostFn] = (build_generic_edit, lambda r: r.n * 0.15)  # unknown price: assume the top of the range


def endpoint_for(model: str) -> tuple[Builder, CostFn, str]:
    """Longest matching prefix wins; returns (builder, cost fn, matched prefix or 'generic')."""
    for prefix in sorted(ENDPOINTS, key=len, reverse=True):
        if model.startswith(prefix):
            b, c = ENDPOINTS[prefix]
            return b, c, prefix
    return GENERIC[0], GENERIC[1], "generic"


def fal_model_id(model: str | None) -> str:
    """fal endpoint id; a bare OpenAI id like `gpt-image-2.6-flare` maps to `openai/gpt-image-2.6/flare/edit`."""
    if not model:
        return DEFAULT_MODEL
    m = re.fullmatch(r"gpt-image-(\d+(?:\.\d+)?)-([a-z]+)", model)
    return f"openai/gpt-image-{m[1]}/{m[2]}/edit" if m else model


def _status_code(e: Exception) -> int | None:
    return getattr(e, "status_code", None) or getattr(getattr(e, "response", None), "status_code", None)


class FalImageBackend:
    name = "fal"

    def __init__(
        self,
        model: str | None = None,
        *,
        client=None,
        control_guide: str = "depth",
        options: dict | None = None,
        max_attempts: int = 3,
        poll_interval_s: float = 1.0,
        timeout_s: float = 300.0,
        downloader: Callable[[str], Awaitable[bytes]] | None = None,
    ):
        self.model = fal_model_id(model)
        self.options = {"control_guide": control_guide, **(options or {})}
        self._client = client
        self.max_attempts = max_attempts
        self.poll_interval_s = poll_interval_s
        self.timeout_s = timeout_s
        self._download = downloader or _http_get

    @property
    def client(self):
        if self._client is None:
            key = os.environ.get("FAL_KEY", "").strip()
            if not key:
                raise MissingCredentialsError("FAL_KEY is not set")
            self._client = fal_client.AsyncClient(key=key)
        return self._client

    def estimate_cost(self, req: PlateRequest) -> float:
        return round(endpoint_for(self.model)[1](req), 5)

    async def build_arguments(self, req: PlateRequest) -> dict:
        builder, _, _ = endpoint_for(self.model)

        async def upload(data: bytes, content_type: str, file_name: str) -> str:
            return await self.client.upload(data, content_type, file_name=file_name)

        return await builder(req, upload, self.options)

    async def _run_once(self, arguments: dict) -> tuple[dict, dict]:
        handle = await self.client.submit(self.model, arguments=arguments)
        t0 = time.monotonic()
        while True:
            status = await handle.status()
            if isinstance(status, fal_client.Completed):
                if status.error:
                    raise ImageGenError(f"fal request failed: {status.error} ({status.error_type})", retryable=False)
                break
            if time.monotonic() - t0 > self.timeout_s:
                await _quiet(handle.cancel())
                raise ImageGenError(f"fal request timed out after {self.timeout_s:.0f} s", retryable=True)
            await asyncio.sleep(self.poll_interval_s)
        result = await handle.get()
        return result, {"request_id": getattr(handle, "request_id", None), "metrics": getattr(status, "metrics", None)}

    async def _run(self, arguments: dict) -> tuple[dict, dict]:
        last: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                result, info = await self._run_once(arguments)
                info["attempts"] = attempt + 1
                return result, info
            except ImageGenError as e:
                if not e.retryable:
                    raise
                last = e
            except (httpx.TransportError, fal_client.FalClientTimeoutError) as e:
                last = e
            except fal_client.FalClientHTTPError as e:
                code = _status_code(e)
                if code == 422:
                    text = str(e).lower()
                    if any(w in text for w in ("policy", "content", "safety", "nsfw", "moderation")):
                        raise ContentPolicyError(f"fal refused the request: {e}", status=422) from e
                    raise ImageGenError(f"fal rejected the request: {e}", retryable=False, status=422) from e
                if code is not None and code != 429 and code < 500:
                    raise ImageGenError(f"fal request failed: {e}", retryable=False, status=code) from e
                last = e
            if attempt + 1 < self.max_attempts:
                await asyncio.sleep(min(2**attempt, 8))
        raise ImageGenError(f"fal request failed after {self.max_attempts} attempts: {last}", retryable=True)

    async def generate(self, req: PlateRequest) -> PlateResult:
        arguments = await self.build_arguments(req)
        result, info = await self._run(arguments)
        flags = result.get("has_nsfw_concepts") or []
        images, rejected = [], []
        for i, item in enumerate(result.get("images") or []):
            if i < len(flags) and flags[i]:
                rejected.append({"index": i, "reason": "nsfw_flag"})
                continue
            url = item.get("url") if isinstance(item, dict) else item
            raw = await self._download(url)
            png, why = normalize_output(raw, req.width, req.height)
            if png is None:
                rejected.append({"index": i, "reason": why})
            else:
                images.append(png)
        meta = {
            "endpoint_family": endpoint_for(self.model)[2],
            "rejected": rejected,
            "seed": result.get("seed"),
            "cost_source": "estimate",
            **info,
        }
        if not images:
            raise ImageGenError(f"fal returned no usable images ({rejected})", retryable=True)
        # fal bills per generated image, rejected ones included
        return PlateResult(images=images, backend=self.name, model=self.model, cost_usd=self.estimate_cost(req), meta=meta)

    async def health(self) -> dict:
        ok = bool(os.environ.get("FAL_KEY", "").strip()) or self._client is not None
        detail = "FAL_KEY set (not verified: fal has no free auth probe)" if ok else "FAL_KEY is not set"
        return {"ok": ok, "name": self.name, "model": self.model, "detail": detail}


async def _quiet(aw) -> None:
    try:
        await aw
    except Exception:
        pass


async def _http_get(url: str) -> bytes:
    if url.startswith("data:"):
        import base64

        return base64.b64decode(url.split(",", 1)[1])
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as http:
        r = await http.get(url)
        r.raise_for_status()
        return r.content
