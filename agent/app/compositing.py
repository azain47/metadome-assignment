"""Compositor: plate defocused by the shot's depth of field, shadow-catcher multiply, straight-alpha
over of the exact car render, light wrap on the car's soft edges, then lens finish (vignette, grain).

Works in display space for the over (Blender's straight-alpha sRGB `car_rgba.png`) and in linear
light for the shadow multiply. Fully opaque car pixels are never changed before `pre_lens`: the
light wrap only touches partially transparent pixels (anti-aliased and motion-blurred edges).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.imaging import lens as lens_fx
from app.imaging.io import decode_png, exr_layer, fit_aspect, read_exr
from app.imaging.tonemap import car_rgba_from_exr


@dataclass
class CarRender:
    rgba: np.ndarray  # HxWx4 uint8, straight alpha, display-referred sRGB (car_rgba.png)
    shadow: np.ndarray | None  # HxW float32 multiplier in [0, 1], 1 = no shadow


@dataclass
class Composite:
    final: np.ndarray  # HxWx3 uint8
    pre_lens: np.ndarray  # HxWx3 uint8, before grain
    alpha: np.ndarray  # HxW float32


def _srgb_to_linear(v: np.ndarray) -> np.ndarray:
    return np.where(v <= 0.04045, v / 12.92, ((np.maximum(v, 0.04045) + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(np.maximum(v, 0.0031308), 1 / 2.4) - 0.055)


def _rgba_u8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint16:
        img = (img.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] == 3:
        img = np.concatenate([img, np.full(img.shape[:2] + (1,), 255, np.uint8)], axis=2)
    return np.ascontiguousarray(img[..., :4].astype(np.uint8))


def load_car_render(rgba_png: bytes, exr: bytes | None, exposure_ev: float = 0.0) -> CarRender:
    """`car_rgba.png` (+ optional `car.exr`) -> CarRender. The shadow is the mean RGB of the EXR's
    Shadow Catcher layer; missing layer or mismatched size -> shadow None. A non-zero
    `exposure_ev` re-develops the car from the EXR's scene-linear Combined pass (exposure before
    the PBR Neutral view transform), so lowering it recovers highlight colour the PNG has lost."""
    if exposure_ev and exr is not None:
        rgba = _rgba_u8(car_rgba_from_exr(exr, exposure_ev))
    else:
        rgba = _rgba_u8(decode_png(rgba_png))
    shadow = None
    if exr is not None:
        layer = exr_layer(read_exr(exr), "Shadow Catcher", "ShadowCatcher", exclude=("matte", "approx"))
        comps = [layer[c] for c in "RGB" if c in layer] or list(layer.values())
        if comps and comps[0].shape == rgba.shape[:2]:
            s = np.mean(np.stack(comps, axis=0), axis=0)
            shadow = np.clip(np.nan_to_num(s, nan=1.0, posinf=1.0, neginf=1.0), 0.0, 1.0).astype(np.float32)
    return CarRender(rgba=rgba, shadow=shadow)


def _light_wrap(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray, strength: float, radius_px: float) -> np.ndarray:
    """Bleed the background's light into the car's semi-transparent edge pixels."""
    soft = (alpha > 0.0) & (alpha < 0.98)
    if strength <= 0 or not soft.any():
        return fg
    inv = 1.0 - alpha
    spread = cv2.GaussianBlur(bg * inv[..., None], (0, 0), radius_px)
    weight = cv2.GaussianBlur(inv, (0, 0), radius_px)
    bg_near = spread / np.maximum(weight, 1e-4)[..., None]
    w = (strength * weight * soft)[..., None]
    return fg * (1.0 - w) + np.maximum(fg, bg_near) * w


def composite(plate: np.ndarray, car: CarRender, *, depth_m: np.ndarray | None = None,
              lens: lens_fx.Lens | None = None, tracking: tuple | None = None, light_wrap: float = 0.0,
              vignette: float = 0.0, grain: float = 0.0, seed: int = 0, exposure_ev: float = 0.0) -> Composite:
    """Plate (HxWx3 uint8 sRGB) + car -> Composite at the car's size.
    `exposure_ev` scales the plate in linear light (the car's exposure is set when it is loaded,
    from its EXR). `depth_m` (Blender depth with the car hidden) drives `tracking` = (Intrinsics,
    camera-space displacement over the shutter) rolling-shot motion blur and `lens` depth of field
    on the plate; `light_wrap` (0..1) bleeds the plate into the car's soft edges; `vignette` and
    `grain` (noise sigma as a fraction of full scale, e.g. 0.012) are applied after `pre_lens`."""
    h, w = car.rgba.shape[:2]
    plate = np.asarray(plate)[..., :3]
    if plate.shape[:2] != (h, w):
        plate = fit_aspect(plate, w, h)
    if exposure_ev:
        lin = _srgb_to_linear(plate.astype(np.float32) / 255.0) * np.float32(2.0 ** exposure_ev)
        plate = np.rint(_linear_to_srgb(lin) * 255.0).astype(np.uint8)
    if depth_m is not None and tracking is not None:
        plate = lens_fx.tracking_blur(plate, depth_m, *tracking)
    if depth_m is not None and lens is not None:
        plate = lens_fx.defocus(plate, lens_fx.coc_px(depth_m, lens, w))
    bg = plate.astype(np.float32) / 255.0
    if car.shadow is not None:
        bg = _linear_to_srgb(_srgb_to_linear(bg) * car.shadow[..., None])
    alpha = car.rgba[..., 3].astype(np.float32) / 255.0
    fg = _light_wrap(car.rgba[..., :3].astype(np.float32) / 255.0, bg, alpha, light_wrap, max(1.0, w / 512))
    a = alpha[..., None]
    out = fg * a + bg * (1.0 - a)
    pre_lens = np.clip(np.rint(out * 255.0), 0, 255).astype(np.uint8)
    final = pre_lens
    if vignette > 0:
        final = lens_fx.vignette(final, vignette)
    if grain > 0:
        final = lens_fx.grain(final, grain, seed)
    return Composite(final=final, pre_lens=pre_lens, alpha=alpha)
