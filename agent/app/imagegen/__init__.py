"""Plate image-generation backends. Add a backend: one module + one `BACKENDS` entry."""

from __future__ import annotations

import os
from collections.abc import Callable

from .base import (
    ContentPolicyError,
    ImageBackend,
    ImageGenError,
    MissingCredentialsError,
    PlateRequest,
    PlateResult,
)


def _openai(model: str | None) -> ImageBackend:
    from .openai_backend import OpenAIImageBackend

    return OpenAIImageBackend(model, quality=os.environ.get("IMAGE_QUALITY", "").strip() or "high")


def _fal(model: str | None) -> ImageBackend:
    from .fal_backend import FalImageBackend

    return FalImageBackend(model)


def _fake(model: str | None) -> ImageBackend:
    from .fake_backend import FakeImageBackend

    return FakeImageBackend(model)


BACKENDS: dict[str, Callable[[str | None], ImageBackend]] = {
    "openai": _openai,
    "fal": _fal,
    "fake": _fake,
}


def get_backend(name: str | None = None, model: str | None = None) -> ImageBackend:
    """Backend by name (`IMAGE_BACKEND` when None, default "openai") and model (`IMAGE_MODEL`
    when None; empty = the backend's default)."""
    if name is None:
        name = os.environ.get("IMAGE_BACKEND", "").strip() or "openai"
    if model is None:
        model = os.environ.get("IMAGE_MODEL", "").strip() or None
    try:
        factory = BACKENDS[name]
    except KeyError:
        raise ValueError(f"unknown image backend {name!r}; choose one of {sorted(BACKENDS)}") from None
    return factory(model)


__all__ = [
    "BACKENDS",
    "ContentPolicyError",
    "ImageBackend",
    "ImageGenError",
    "MissingCredentialsError",
    "PlateRequest",
    "PlateResult",
    "get_backend",
]
