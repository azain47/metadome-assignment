"""Lens effects that make a composite read as one photograph: depth of field and rolling-shot
motion blur from the shot's real geometry, vignette and grain. Pure numpy/cv2, applied to
display-referred sRGB uint8 images.

Depth of field uses the thin-lens circle of confusion with Blender's depth pass (metres, car
hidden) and the shot camera: blur diameter on the sensor = f^2 / (N (s - f)) * |d - s| / d.
Tracking blur assumes the camera moves with the car (a rolling / chase shot): each pixel is lifted
to 3D with the depth pass, moved by the camera's displacement over the shutter, and the plate is
integrated along its image path. Near tarmac streaks, distant mountains barely move; the car (a
separate render that moves with the camera) stays sharp.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

SKY_M = 1.0e4  # depth at or beyond this, non-finite or <= 0 counts as infinitely far


@dataclass(frozen=True)
class Lens:
    focal_mm: float
    focus_m: float
    f_number: float = 4.0
    sensor_width_mm: float = 36.0


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole camera in pixels (image y down); horizontal sensor fit, Blender lens shift."""
    fx: float
    cx: float
    cy: float

    @classmethod
    def from_camera(cls, cam: dict, width: int, height: int) -> "Intrinsics":
        """`cam` = frame_shot's camera dict (lens_mm, sensor_width_mm, shift_x, shift_y)."""
        fx = cam["lens_mm"] / cam.get("sensor_width_mm", 36.0) * width
        return cls(fx=fx, cx=width / 2.0 - cam.get("shift_x", 0.0) * width,
                   cy=height / 2.0 + cam.get("shift_y", 0.0) * width)


def world_to_camera(rotation_deg) -> np.ndarray:
    """3x3 matrix taking world vectors into Blender camera space (x right, y up, looking -z),
    from the camera's XYZ Euler rotation in degrees."""
    rx, ry, rz = (math.radians(a) for a in rotation_deg)
    cxr, sxr, cyr, syr, czr, szr = math.cos(rx), math.sin(rx), math.cos(ry), math.sin(ry), math.cos(rz), math.sin(rz)
    rot_x = np.array([[1, 0, 0], [0, cxr, -sxr], [0, sxr, cxr]])
    rot_y = np.array([[cyr, 0, syr], [0, 1, 0], [-syr, 0, cyr]])
    rot_z = np.array([[czr, -szr, 0], [szr, czr, 0], [0, 0, 1]])
    return (rot_z @ rot_y @ rot_x).T  # camera->world is Rz Ry Rx; its inverse is the transpose


def _depth(depth_m: np.ndarray, shape: tuple[int, int], far: float) -> np.ndarray:
    z = np.asarray(depth_m, dtype=np.float32)
    if z.ndim == 3:
        z = z[..., 0]
    if z.shape != shape:
        z = cv2.resize(z, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return np.where(np.isfinite(z) & (z > 0) & (z < SKY_M), z, far).astype(np.float32)


def coc_px(depth_m: np.ndarray, lens: Lens, width_px: int) -> np.ndarray:
    """Blur-disc diameter in pixels for each depth (HxW float32)."""
    f = lens.focal_mm / 1000.0
    s = max(lens.focus_m, f * 1.01)
    d = _depth(depth_m, np.asarray(depth_m).shape[:2], SKY_M)
    c_m = (f * f / (lens.f_number * (s - f))) * np.abs(d - s) / d
    return (c_m * 1000.0 / lens.sensor_width_mm * width_px).astype(np.float32)


def defocus(img: np.ndarray, coc: np.ndarray, max_px: float = 24.0) -> np.ndarray:
    """Depth-varying blur: blend between a stack of Gaussian blurs by each pixel's CoC
    (disc of diameter c ~ Gaussian sigma c / 2.4). img HxWx3 uint8, coc HxW pixels."""
    if coc.shape != img.shape[:2]:
        coc = cv2.resize(coc, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
    c = np.clip(coc, 0.0, max_px)
    levels = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, max_px]
    src = img.astype(np.float32)
    stack = [src] + [cv2.GaussianBlur(src, (0, 0), sigmaX=lv / 2.4) for lv in levels[1:]]
    out = np.zeros_like(src)
    for i in range(len(levels) - 1):
        lo, hi = levels[i], levels[i + 1]
        t = np.clip((c - lo) / (hi - lo), 0.0, 1.0)[..., None]
        band = ((c >= lo) & ((c < hi) | (i == len(levels) - 2)))[..., None]
        out += band * (stack[i] * (1.0 - t) + stack[i + 1] * t)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def camera_displacement(rotation_deg, heading_deg: float, speed_kmh: float, shutter_s: float) -> np.ndarray:
    """Camera-space displacement over the shutter for a camera travelling with a car that
    drives at `speed_kmh` along compass `heading_deg` (0 = +Y, 90 = +X)."""
    h = math.radians(heading_deg)
    world = np.array([math.sin(h), math.cos(h), 0.0]) * (speed_kmh / 3.6) * shutter_s
    return world_to_camera(rotation_deg) @ world


def tracking_blur(img: np.ndarray, depth_m: np.ndarray, intr: Intrinsics, displacement_cam,
                  samples: int = 24) -> np.ndarray:
    """Motion blur of a static scene seen from a camera that moves by `displacement_cam`
    (camera-space metres, over the whole shutter). img HxWx3 uint8, depth_m HxW planar depth."""
    h, w = img.shape[:2]
    z = _depth(depth_m, (h, w), 1.0e6)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    px = (u - intr.cx) / intr.fx * z
    py = -(v - intr.cy) / intr.fx * z
    dx, dy, dz = (float(c) for c in displacement_cam)
    src = img.astype(np.float32)
    acc = np.zeros_like(src)
    for t in np.linspace(-0.5, 0.5, samples):
        qz = np.maximum(z + t * dz, 1.0e-3)  # camera-space z is -depth; moving by +t*d gives -(-z - t*dz)
        mu = intr.cx + intr.fx * (px - t * dx) / qz
        mv = intr.cy - intr.fx * (py - t * dy) / qz
        acc += cv2.remap(src, mu.astype(np.float32), mv.astype(np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)
    return np.clip(np.rint(acc / samples), 0, 255).astype(np.uint8)


def vignette(img: np.ndarray, strength: float = 0.15) -> np.ndarray:
    """Darken toward the corners: multiply by 1 - strength * r^2 (r = 1 at the corners)."""
    h, w = img.shape[:2]
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r2 = ((x - w / 2) / (w / 2)) ** 2 * 0.5 + ((y - h / 2) / (h / 2)) ** 2 * 0.5
    gain = (1.0 - strength * r2)[..., None]
    return np.clip(np.rint(img.astype(np.float32) * gain), 0, 255).astype(np.uint8)


def grain(img: np.ndarray, sigma: float = 0.01, seed: int = 0) -> np.ndarray:
    """Monochrome film grain, sigma as a fraction of full scale, slightly softened."""
    noise = np.random.default_rng(seed).normal(0.0, sigma * 255.0, img.shape[:2]).astype(np.float32)
    noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=0.6)[..., None]
    return np.clip(np.rint(img.astype(np.float32) + noise), 0, 255).astype(np.uint8)
