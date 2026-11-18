"""Scene-linear car render -> display sRGB, reproducing Blender's "Khronos PBR Neutral" view.

Blender bakes the view transform into `car_rgba.png` (8-bit, clipped). Going back to the
scene-linear EXR lets the compositor change exposure before the tone mapper, which recovers
highlight colour that PBR Neutral otherwise desaturates towards white.
"""

from __future__ import annotations

import numpy as np

from app.imaging.io import exr_layer, read_exr

_START_COMPRESSION = 0.8 - 0.04
_DESATURATION = 0.15


def pbr_neutral(rgb: np.ndarray) -> np.ndarray:
    """Khronos PBR Neutral tone mapper (reference GLSL, KhronosGroup/ToneMapping).
    HxWx3 (or ...x3) linear Rec.709 >= 0 -> linear display-referred Rec.709 in [0, 1]."""
    c = np.maximum(np.asarray(rgb, dtype=np.float32), 0.0)
    x = c.min(axis=-1, keepdims=True)
    offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04).astype(np.float32)
    c = c - offset

    peak = c.max(axis=-1, keepdims=True)
    d = 1.0 - _START_COMPRESSION
    new_peak = 1.0 - d * d / (peak + d - _START_COMPRESSION)
    scaled = c * (new_peak / np.maximum(peak, 1e-12))
    g = 1.0 - 1.0 / (_DESATURATION * (peak - new_peak) + 1.0)
    compressed = scaled * (1.0 - g) + new_peak * g
    return np.where(peak < _START_COMPRESSION, c, compressed).astype(np.float32)


def encode_srgb(linear: np.ndarray) -> np.ndarray:
    """Linear [0, 1] -> 8-bit sRGB (IEC 61966-2-1 piecewise OETF), rounded to nearest."""
    v = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    s = np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0.0031308), 1.0 / 2.4) - 0.055)
    return np.clip(np.rint(s * 255.0), 0, 255).astype(np.uint8)


def car_rgba_from_exr(exr: bytes, exposure_ev: float = 0.0) -> np.ndarray:
    """Blender multilayer car EXR (premultiplied scene-linear `Combined`) -> HxWx4 uint8, straight
    alpha, display sRGB through PBR Neutral: the same encoding as Blender's `car_rgba.png`.
    `exposure_ev` scales scene-linear light before the view transform."""
    layer = exr_layer(read_exr(exr), "Combined")
    if not all(k in layer for k in "RGB"):
        raise ValueError("EXR has no Combined RGB layer")
    rgb = np.stack([layer[k] for k in "RGB"], axis=-1).astype(np.float32)
    alpha = layer["A"].astype(np.float32) if "A" in layer else np.ones(rgb.shape[:2], np.float32)
    rgb = np.nan_to_num(rgb, nan=0.0, posinf=0.0, neginf=0.0)
    alpha = np.clip(np.nan_to_num(alpha, nan=0.0), 0.0, 1.0)

    rgb *= np.float32(2.0 ** float(exposure_ev))
    covered = alpha > 0.0
    rgb = np.where(covered[..., None], rgb / np.where(covered, alpha, 1.0)[..., None], 0.0)

    out = np.empty(rgb.shape[:2] + (4,), np.uint8)
    out[..., :3] = encode_srgb(pbr_neutral(rgb))
    out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    return out
