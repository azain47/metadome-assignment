"""Offline, deterministic stand-in plate generator (cost 0).

Colours the grey Blender guide by its flat tones (sky gradient, asphalt, ground, rock ridges
with snow caps, buildings, trees, sea), adds texture and sensor-like noise and a slight layout
jitter per candidate (shift <= 0.6 %, scale <= 0.8 %, roll <= 0.3 deg), so ranking and
compositing code can run without credits. Same request -> same bytes.
"""

from __future__ import annotations

import hashlib

import cv2
import numpy as np

from app.imaging.io import decode_rgb, encode_png, fit_aspect

from .base import PlateRequest, PlateResult

# Guide tones from the scene conventions (Workbench flat object colour, 0..1).
TONES = {
    "sky": 0.92,
    "building": 0.68,
    "ground": 0.55,
    "ridge": 0.42,
    "tree": 0.35,
    "sea": 0.30,
    "road": 0.22,
}
BASE_RGB = {
    "building": (196, 186, 168),
    "ground": (104, 112, 66),
    "ridge": (112, 104, 96),
    "tree": (44, 70, 38),
    "road": (68, 68, 71),
}


def _srgb_encode(v: np.ndarray) -> np.ndarray:
    v = np.clip(v, 0, 1)
    return np.where(v <= 0.0031308, v * 12.92, 1.055 * np.power(v, 1 / 2.4) - 0.055)


def classify(grey_u8: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Label map (index into the returned names; -1 = unknown) by nearest guide tone. The PNG may
    hold the tones raw or sRGB-encoded; the encoding with the smaller total error is used."""
    names = list(TONES)
    raw = np.array([TONES[n] for n in names]) * 255
    srgb = _srgb_encode(np.array([TONES[n] for n in names])) * 255
    g = grey_u8.astype(np.float32)
    best = None
    for table in (raw, srgb):
        d = np.abs(g[..., None] - table[None, None, :])
        idx = d.argmin(axis=-1)
        err = float(d.min(axis=-1).mean())
        if best is None or err < best[0]:
            best = (err, idx, d.min(axis=-1))
    _, idx, dmin = best
    labels = np.where(dmin <= 20, idx, -1).astype(np.int16)
    return labels, names


def _lowfreq(rng: np.random.Generator, h: int, w: int, cells: int) -> np.ndarray:
    small = rng.standard_normal((max(2, cells * h // w), cells)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def paint(grey_rgb: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    grey = cv2.cvtColor(grey_rgb, cv2.COLOR_RGB2GRAY)
    labels, names = classify(grey)
    h, w = grey.shape
    out = np.zeros((h, w, 3), np.float32)
    rows = np.arange(h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
    texture = 0.6 * _lowfreq(rng, h, w, 12) + 0.3 * _lowfreq(rng, h, w, 48) + 0.25 * rng.standard_normal((h, w)).astype(np.float32)

    for i, name in enumerate(names):
        m = labels == i
        if not m.any():
            continue
        if name == "sky":
            bottom = max(1.0, float(rows[m].max()))
            t = np.clip(rows / bottom, 0, 1)[..., None]
            col = (1 - t) * np.array([58, 104, 178], np.float32) + t * np.array([196, 212, 228], np.float32)
            col = col + 6 * _lowfreq(rng, h, w, 6)[..., None]  # soft cloud variation
        elif name == "sea":
            top = float(rows[m].min())
            t = np.clip((rows - top) / max(1.0, h - top), 0, 1)[..., None]
            col = (1 - t) * np.array([120, 160, 185], np.float32) + t * np.array([32, 78, 112], np.float32)
            col = col + 8 * texture[..., None]
        else:
            col = np.array(BASE_RGB[name], np.float32)[None, None, :] * (1 + 0.12 * texture[..., None])
            if name == "ridge":
                # snow on the upper part of every ridge column
                ys = np.where(m, rows, np.inf).min(axis=0)
                ye = np.where(m, rows, -np.inf).max(axis=0)
                snow_line = ys + 0.35 * np.maximum(ye - ys, 0) + 6 * _lowfreq(rng, 1, w, 40)[0]
                snow = m & (rows < snow_line[None, :])
                col = np.where(snow[..., None], np.array([236, 238, 242], np.float32) * (1 + 0.03 * texture[..., None]), col)
            elif name == "building":
                win = ((np.arange(w) // 7) % 3 == 0)[None, :] & ((np.arange(h) // 9) % 2 == 0)[:, None]
                col = np.where(win[..., None], col * 0.55, col)
            elif name == "road":
                col = col + 5 * rng.standard_normal((h, w, 1)).astype(np.float32)
        out[m] = np.broadcast_to(col, (h, w, 3))[m]
    unknown = labels < 0
    if unknown.any():
        g = grey.astype(np.float32)[..., None]
        out[unknown] = (g * np.array([1.0, 0.97, 0.92], np.float32) + 10 * texture[..., None])[unknown]
    out = cv2.GaussianBlur(out, (0, 0), 0.7)
    out += 2.5 * rng.standard_normal(out.shape).astype(np.float32)  # sensor noise
    return np.clip(out, 0, 255).astype(np.uint8)


def jitter(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    h, w = img.shape[:2]
    dx, dy = rng.uniform(-0.006, 0.006) * w, rng.uniform(-0.006, 0.006) * h
    s = 1.0 + rng.uniform(-0.008, 0.008)
    rot = rng.uniform(-0.3, 0.3)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rot, s)
    m[:, 2] += (dx, dy)
    out = cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out, {"dx_px": round(dx, 2), "dy_px": round(dy, 2), "scale": round(s, 4), "rot_deg": round(rot, 3)}


class FakeImageBackend:
    name = "fake"

    def __init__(self, model: str | None = None):
        self.model = model or "fake-tones-v1"

    def estimate_cost(self, req: PlateRequest) -> float:
        return 0.0

    def _seed(self, req: PlateRequest, k: int) -> int:
        h = hashlib.sha256(req.guide_grey)
        h.update(req.prompt.encode())
        h.update(f"{req.seed}:{k}:{req.width}x{req.height}".encode())
        return int.from_bytes(h.digest()[:8], "little")

    async def generate(self, req: PlateRequest) -> PlateResult:
        guide = fit_aspect(decode_rgb(req.guide_grey), req.width, req.height)
        images, jitters = [], []
        for k in range(req.n):
            rng = np.random.default_rng(self._seed(req, k))
            plate, jit = jitter(paint(guide, rng), rng)
            images.append(encode_png(plate))
            jitters.append(jit)
        return PlateResult(images=images, backend=self.name, model=self.model, cost_usd=0.0, meta={"jitter": jitters})

    async def health(self) -> dict:
        return {"ok": True, "name": self.name, "model": self.model, "detail": "offline fake backend"}
