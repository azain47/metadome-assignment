"""OpenAI GPT Image edit backend (`POST /v1/images/edits`, several input images).

Request: image 1 = grey guide at exactly the output size, then the reference photos cropped to
the output aspect (long side <= 1536, JPEG). `size`, `n`, `quality` from the request/backend;
`complexity` (not yet an SDK argument) goes through `extra_body`. `input_fidelity` is omitted:
GPT Image 2+ always uses high fidelity.
Retries on 429/5xx/connection errors are the SDK's (`max_retries`); moderation blocks raise
`ContentPolicyError` and are not retried.
"""

from __future__ import annotations

import base64
import os

import openai

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

DEFAULT_MODEL = "gpt-image-2.6-flare"

# GPT Image 2.6 rates, USD per 1M tokens.
TEXT_IN_PER_M = 5.00
IMAGE_IN_PER_M = 8.00
IMAGE_OUT_PER_M = 30.00
# Output price per 1536x1024 image by quality (scaled by area for other sizes).
OUT_1536x1024 = {"low": 0.020, "medium": 0.034, "high": 0.041}
IN_TOKENS_1536x1024 = 1536


class OpenAIImageBackend:
    name = "openai"

    def __init__(
        self,
        model: str | None = None,
        *,
        client: openai.AsyncOpenAI | None = None,
        quality: str = "high",
        complexity: str | None = "low",
        max_retries: int = 3,
        timeout_s: float = 240.0,
    ):
        self.model = model or DEFAULT_MODEL
        self.quality = quality
        self.complexity = complexity
        self._client = client
        self._max_retries = max_retries
        self._timeout_s = timeout_s

    @property
    def client(self) -> openai.AsyncOpenAI:
        if self._client is None:
            key = os.environ.get("OPENAI_API_KEY", "").strip()
            if not key:
                raise MissingCredentialsError("OPENAI_API_KEY is not set")
            # Images always go to api.openai.com: LLM_API_BASE is for chat models only.
            self._client = openai.AsyncOpenAI(api_key=key, max_retries=self._max_retries, timeout=self._timeout_s)
        return self._client

    def build_request(self, req: PlateRequest) -> dict:
        """Keyword arguments for `client.images.edit` (no network)."""
        images = [("guide_grey.png", guide_at_size(req.guide_grey, req.width, req.height), "image/png")]
        for i, ref in enumerate(req.references):
            images.append((f"reference_{i + 1}.jpg", reference_at_aspect(ref, req.width, req.height), "image/jpeg"))
        kwargs: dict = {
            "model": self.model,
            "image": images,
            "prompt": req.prompt,
            "size": f"{req.width}x{req.height}",
            "n": req.n,
            "quality": self.quality_for(req),
            "output_format": "png",
        }
        if self.complexity:
            kwargs["extra_body"] = {"complexity": self.complexity}
        return kwargs

    def quality_for(self, req: PlateRequest) -> str:
        return req.quality or self.quality

    def estimate_cost(self, req: PlateRequest) -> float:
        area = req.width * req.height / (1536 * 1024)
        out = OUT_1536x1024.get(self.quality_for(req), OUT_1536x1024["high"]) * area * req.n
        n_in = 1 + len(req.references)
        inputs = n_in * IN_TOKENS_1536x1024 * area * IMAGE_IN_PER_M / 1e6
        text = (len(req.prompt) / 4) * TEXT_IN_PER_M / 1e6
        return round(out + inputs + text, 5)

    @staticmethod
    def cost_from_usage(usage) -> float | None:
        if usage is None:
            return None
        details = getattr(usage, "input_tokens_details", None)
        text_in = getattr(details, "text_tokens", None) if details else None
        image_in = getattr(details, "image_tokens", None) if details else None
        if text_in is None or image_in is None:
            text_in, image_in = 0, getattr(usage, "input_tokens", 0) or 0
        out = getattr(usage, "output_tokens", 0) or 0
        return round((text_in * TEXT_IN_PER_M + image_in * IMAGE_IN_PER_M + out * IMAGE_OUT_PER_M) / 1e6, 5)

    async def generate(self, req: PlateRequest) -> PlateResult:
        kwargs = self.build_request(req)
        try:
            resp = await self.client.images.edit(**kwargs)
        except openai.BadRequestError as e:
            code = getattr(e, "code", None) or ""
            if "moderation" in str(code) or "moderation_blocked" in str(e) or "safety" in str(code):
                raise ContentPolicyError(f"OpenAI refused the request: {e}", status=400) from e
            raise ImageGenError(f"OpenAI rejected the request: {e}", retryable=False, status=400) from e
        except (openai.RateLimitError, openai.InternalServerError, openai.APIConnectionError, openai.APITimeoutError) as e:
            raise ImageGenError(f"OpenAI image call failed after retries: {e}", retryable=True) from e
        except openai.APIStatusError as e:
            raise ImageGenError(f"OpenAI image call failed: {e}", retryable=False, status=e.status_code) from e

        images, rejected = [], []
        for i, item in enumerate(resp.data or []):
            raw = base64.b64decode(item.b64_json) if getattr(item, "b64_json", None) else None
            png, why = normalize_output(raw, req.width, req.height) if raw else (None, "empty")
            if png is None:
                rejected.append({"index": i, "reason": why})
            else:
                images.append(png)
        usage = getattr(resp, "usage", None)
        actual = self.cost_from_usage(usage)
        cost = actual if actual is not None else self.estimate_cost(req)
        meta = {
            "cost_source": "usage" if actual is not None else "estimate",
            "usage": usage.model_dump() if hasattr(usage, "model_dump") else None,
            "rejected": rejected,
            "quality": kwargs["quality"],
            "complexity": self.complexity,
            "size": kwargs["size"],
            "n_inputs": len(kwargs["image"]),
        }
        if not images:
            raise ImageGenError(f"OpenAI returned no usable images ({rejected})", retryable=True)
        return PlateResult(images=images, backend=self.name, model=self.model, cost_usd=cost, meta=meta)

    async def health(self) -> dict:
        if not os.environ.get("OPENAI_API_KEY", "").strip() and self._client is None:
            return {"ok": False, "name": self.name, "model": self.model, "detail": "OPENAI_API_KEY is not set"}
        try:
            await self.client.models.retrieve(self.model, timeout=10)
            return {"ok": True, "name": self.name, "model": self.model, "detail": "model visible"}
        except Exception as e:  # health must never raise
            return {"ok": False, "name": self.name, "model": self.model, "detail": f"{type(e).__name__}: {e}"[:300]}
