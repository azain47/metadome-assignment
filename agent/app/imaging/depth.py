"""Depth of a generated plate, in metres, from the plate itself.

The image model paints its own scene, so Blender's depth pass (grey stand-ins) does not describe
the plate: a stand-in ridge sits where the plate shows peaks kilometres behind it, and blur
driven by that depth leaves a seam wherever a painted edge misses a stand-in edge. Depth Anything
V2 Small (Apache-2.0, ONNX on the CPU, about 1 s for 1536x1024) estimates relative inverse depth
d from the plate's pixels; sky comes out as exactly 0. It is converted to metres on the one
surface both images share exactly, the road, whose Blender depth is known:

    1 / z = scale * d + shift                  for d >= far_d (as far as the road reaches)
    1 / z = (scale * far_d + shift) * d / far_d    beyond the farthest road, down to sky at d = 0

The line is fitted on the road pixels robustly: it starts from the median ratio of inverse depth
to d (which a few painted markings or objects cannot move), then is re-fitted three times by least
squares on the 80 % of road pixels it fits best; far_d is the 2nd percentile of the road's d.
Beyond the road there is no metric evidence, so inverse depth falls straight
to 0 at the model's sky value: continuous, monotone, and the sky stays infinitely far. Extending
the road line instead put the sky about 50 m away on some plates (which would streak clouds in a
rolling shot); forcing it through d = 0 cost up to 9 % of road accuracy. Measured on the first
four real plates: median road error 0.3 to 2.4 %.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.imaging.depth_weights import ensure_weights

INPUT_SHORT_SIDE = 518  # the model's training resolution (its preprocessor_config.json)
PATCH = 14  # input sides must be multiples of the ViT patch size
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
FAR_INV = 1e-4  # inverse metres: anything beyond 10 km counts as sky (inf)
MIN_ROAD_PX = 500
TRIM_KEEP, TRIM_ROUNDS = 0.8, 3


@dataclass(frozen=True)
class PlateDepth:
    metres: np.ndarray  # HxW float32 at the plate's size, inf for sky and anything beyond 10 km
    scale: float  # inverse metres per model unit
    shift: float  # inverse metres
    far_d: float  # model value of the farthest fitted road; beyond it the far tail applies
    road_px: int  # road pixels the fit used
    road_err: float  # median |plate - Blender| / Blender over those pixels

    def summary(self) -> str:
        return f"road error {self.road_err:.1%} over {self.road_px} px"


@functools.lru_cache(maxsize=1)
def _session() -> Any:
    import onnxruntime as ort  # imported on first use: only plate depth needs it

    return ort.InferenceSession(ensure_weights(settings.models_dir), providers=["CPUExecutionProvider"])


def at_size(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """`arr` (HxW, or its first channel) at `shape`, nearest neighbour so masks stay masks and inf
    stays inf. Boolean masks come back as uint8 (0/1)."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = a[..., 0]
    if a.dtype == bool:
        a = a.astype(np.uint8)
    if a.shape == shape:
        return a
    return cv2.resize(a, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)


def relative_depth(rgb: np.ndarray) -> np.ndarray:
    """Depth Anything's relative inverse depth of an HxWx3 uint8 RGB image (HxW float32, sky = 0)."""
    h, w = rgb.shape[:2]
    s = INPUT_SHORT_SIDE / min(h, w)
    nh, nw = (max(PATCH, int(round(v * s / PATCH)) * PATCH) for v in (h, w))
    x = cv2.resize(np.ascontiguousarray(rgb[..., :3]), (nw, nh), interpolation=cv2.INTER_CUBIC)
    x = ((x.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
    d = _session().run(None, {"pixel_values": np.ascontiguousarray(x)})[0][0]
    return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def _affine(d: np.ndarray, t: np.ndarray) -> tuple[float, float]:
    """Least squares t = scale * d + shift."""
    (scale, shift), *_ = np.linalg.lstsq(np.stack([d, np.ones_like(d)], 1), t, rcond=None)
    return float(scale), float(shift)


def calibrate(disparity: np.ndarray, blender_m: np.ndarray, road: np.ndarray) -> PlateDepth:
    """Relative inverse depth -> metres (see the module docstring), fitted where `road` (the
    guides' road mask) has a finite Blender depth. Raises ValueError when there is too little
    road to fit on, or the fit runs the wrong way."""
    shape = disparity.shape
    z = at_size(blender_m, shape).astype(np.float64)
    with np.errstate(invalid="ignore"):
        valid = (at_size(road, shape) > 0) & np.isfinite(z) & (z > 0) & (z < 1.0 / FAR_INV) & (disparity > 0)
    n = int(valid.sum())
    if n < MIN_ROAD_PX:
        raise ValueError(f"only {n} road pixels to calibrate the plate depth on")
    d, t = disparity[valid].astype(np.float64), 1.0 / z[valid]
    scale, shift = float(np.median(t / d)), 0.0
    for _ in range(TRIM_ROUNDS):
        rel = np.abs(scale * d + shift - t) / t
        keep = rel <= np.quantile(rel, TRIM_KEEP)
        scale, shift = _affine(d[keep], t[keep])
    far_d = float(np.quantile(d, 0.02))
    inv_far = scale * far_d + shift
    if not (scale > 0 and inv_far > FAR_INV):
        raise ValueError(f"plate depth runs the wrong way along the road (1/z = {scale:.3g} d {shift:+.3g})")
    x = disparity.astype(np.float64)
    inv = np.where(x >= far_d, scale * x + shift, inv_far * x / far_d)
    metres = np.where(inv > FAR_INV, 1.0 / np.maximum(inv, FAR_INV), np.inf).astype(np.float32)
    road_err = float(np.median(np.abs(metres[valid] - z[valid]) / z[valid]))
    return PlateDepth(metres, scale, shift, far_d, n, road_err)


def estimate_plate_depth(rgb: np.ndarray, blender_m: np.ndarray, road: np.ndarray) -> PlateDepth:
    """The plate's own depth in metres (see the module docstring)."""
    return calibrate(relative_depth(rgb), blender_m, road)
