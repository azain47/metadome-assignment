"""Shared types for the plate (empty-scene) image-generation backends."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from app.imaging.io import decode_rgb, encode_jpeg, encode_png, fit_aspect


@dataclass
class PlateRequest:
    """Inputs for one plate generation call. Images are PNG/JPEG bytes.

    `prompt` is sent as given (the pipeline owns the prompt text). Image order for edit models:
    guide_grey first, then `references`. Layout-control models use guide_depth or guide_lines."""

    guide_grey: bytes
    guide_lines: bytes
    guide_depth: bytes
    references: list[bytes]
    prompt: str
    n: int
    width: int = 1536
    height: int = 1024
    seed: int | None = None
    quality: str | None = None  # low|medium|high; None = backend default (OpenAI only)


@dataclass
class PlateResult:
    images: list[bytes]  # PNG bytes, each exactly width x height
    backend: str
    model: str
    cost_usd: float
    meta: dict = field(default_factory=dict)


@runtime_checkable
class ImageBackend(Protocol):
    name: str
    model: str

    async def generate(self, req: PlateRequest) -> PlateResult: ...

    async def health(self) -> dict: ...

    def estimate_cost(self, req: PlateRequest) -> float: ...


class ImageGenError(RuntimeError):
    """Generation failed. `retryable` tells the caller whether trying again can help."""

    def __init__(self, message: str, *, retryable: bool = False, status: int | None = None):
        super().__init__(message)
        self.retryable = retryable
        self.status = status


class ContentPolicyError(ImageGenError):
    """The provider refused the input or output (moderation / safety). Never retried."""

    def __init__(self, message: str, *, status: int | None = None):
        super().__init__(message, retryable=False, status=status)


class MissingCredentialsError(ImageGenError):
    def __init__(self, message: str):
        super().__init__(message, retryable=False)


def is_black(img: np.ndarray) -> bool:
    """True for safety-checker outputs (all-black or near-black frames) and empty arrays."""
    if img.size == 0:
        return True
    return float(np.percentile(img, 99.5)) < 10.0


def guide_at_size(data: bytes, width: int, height: int) -> bytes:
    """Guide image as PNG at exactly width x height (centre crop to the aspect, then resize)."""
    img = decode_rgb(data)
    if img.shape[1] == width and img.shape[0] == height:
        return data
    return encode_png(fit_aspect(img, width, height))


def reference_at_aspect(data: bytes, width: int, height: int, max_side: int = 1536) -> bytes:
    """Reference photo centre-cropped to the output aspect, long side <= max_side, as JPEG."""
    img = decode_rgb(data)
    h, w = img.shape[:2]
    crop_w = min(w, h * width / height)  # width of the centre crop at the output aspect
    tw = min(crop_w, width, max_side * width / max(width, height))  # never upscale
    tw = max(16, int(round(tw)))
    th = max(16, int(round(tw * height / width)))
    return encode_jpeg(fit_aspect(img, tw, th))


def normalize_output(data: bytes, width: int, height: int) -> tuple[bytes | None, str | None]:
    """Provider output -> (PNG at width x height, None) or (None, reason) when it must be rejected."""
    try:
        img = decode_rgb(data)
    except ValueError:
        return None, "undecodable"
    if is_black(img):
        return None, "black"
    if img.shape[1] != width or img.shape[0] != height:
        img = fit_aspect(img, width, height)
    return encode_png(img), None
