"""Guides derived from Blender's guide renders: line guide and inverse-depth guide."""

from __future__ import annotations

import cv2
import numpy as np

TONE_STEP = 6  # minimum 8-bit grey difference between neighbours that counts as a boundary
SKY_M = 1.0e4  # depths at or beyond this (or non-finite / <= 0) are sky


def _grey_u8(img: np.ndarray) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 3:
        a = cv2.cvtColor(np.ascontiguousarray(a[..., :3]), cv2.COLOR_RGB2GRAY) if a.shape[2] >= 3 else a[..., 0]
    if a.dtype == np.uint16:
        a = (a // 257).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(np.rint(a * 255.0), 0, 255).astype(np.uint8)
    return a


def line_guide(grey: np.ndarray) -> np.ndarray:
    """Workbench grey guide (HxW or HxWx3, flat tone per object) -> uint8 HxW, white (255) lines on
    black where the tone changes (road edges, horizon/skyline, stand-in silhouettes)."""
    g = cv2.medianBlur(_grey_u8(grey), 3).astype(np.int16)  # drops anti-aliasing speckle
    edges = np.zeros(g.shape, dtype=bool)
    edges[:, 1:] |= np.abs(g[:, 1:] - g[:, :-1]) >= TONE_STEP
    edges[1:, :] |= np.abs(g[1:, :] - g[:-1, :]) >= TONE_STEP
    out = edges.astype(np.uint8) * 255
    t = max(1, int(round(g.shape[1] / 1024)))  # about 2 px at 1536 wide
    if t > 1:
        out = cv2.dilate(out, np.ones((t, t), np.uint8))
    return out


def depth_guide(depth_m: np.ndarray) -> np.ndarray:
    """Camera depth in metres (HxW) -> uint8 HxW inverse depth: near = bright, sky = 0.

    Sky = non-finite, <= 0 or >= 10 km. Inverse depth is normalised between its 1st and 99th
    percentiles over non-sky pixels and mapped to 16..255, so the farthest ground never merges
    with the sky."""
    z = np.asarray(depth_m, dtype=np.float32)
    if z.ndim == 3:
        z = z[..., 0]
    valid = np.isfinite(z) & (z > 0) & (z < SKY_M)
    out = np.zeros(z.shape, np.uint8)
    if not valid.any():
        return out
    inv = np.zeros(z.shape, np.float32)
    inv[valid] = 1.0 / z[valid]
    lo, hi = np.percentile(inv[valid], [1, 99])
    v = np.clip((inv - lo) / max(float(hi - lo), 1e-12), 0.0, 1.0)
    out[valid] = np.rint(16 + 239 * v[valid]).astype(np.uint8)
    return out
