"""Image I/O: PNG encode/decode, EXR reading (Blender multilayer aware), aspect fitting.

Blender multilayer EXR channel names look like `ViewLayer.Combined.R` and
`ViewLayer.Shadow Catcher.R`; single-layer EXRs use bare `R/G/B/A`, depth files `V`, `Y` or `Z`.
`exr_layer` groups channels by layer, matching names case/space-insensitively.
"""

from __future__ import annotations

import io
import re

import cv2
import numpy as np
import OpenEXR


def decode_png(data: bytes) -> np.ndarray:
    """PNG/JPEG bytes -> RGB / RGBA / grey array, keeping alpha and 16-bit depth."""
    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("could not decode image bytes")
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def decode_rgb(data: bytes) -> np.ndarray:
    """Image bytes -> HxWx3 uint8 RGB (alpha dropped, grey expanded, 16-bit reduced)."""
    img = decode_png(data)
    if img.dtype == np.uint16:
        img = (img.astype(np.uint32) * 255 // 65535).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img.astype(np.float32) * 255.0, 0, 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    return np.ascontiguousarray(img[..., :3])


def encode_png(arr: np.ndarray) -> bytes:
    """HxW, HxWx3 (RGB) or HxWx4 (RGBA) array -> PNG bytes. uint8/uint16 kept; floats are 0..1."""
    a = np.asarray(arr)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif a.dtype not in (np.uint8, np.uint16):
        a = np.clip(np.rint(a.astype(np.float32) * 255.0), 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        a = cv2.cvtColor(a, cv2.COLOR_RGB2BGR)
    elif a.ndim == 3 and a.shape[2] == 4:
        a = cv2.cvtColor(a, cv2.COLOR_RGBA2BGRA)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = a[..., 0]
    ok, out = cv2.imencode(".png", a)
    if not ok:
        raise ValueError("PNG encoding failed")
    return out.tobytes()


def encode_jpeg(arr: np.ndarray, quality: int = 92) -> bytes:
    """HxWx3 RGB uint8 -> JPEG bytes."""
    a = np.asarray(arr)
    if a.ndim == 3:
        a = cv2.cvtColor(a[..., :3], cv2.COLOR_RGB2BGR)
    ok, out = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return out.tobytes()


def read_exr(data: bytes) -> dict[str, np.ndarray]:
    """EXR bytes -> {channel name: float32 HxW}. Multi-part files: names are kept; on a clash
    the part name is prefixed (`<part>.<channel>`)."""
    f = OpenEXR.File(io.BytesIO(data), separate_channels=True)
    out: dict[str, np.ndarray] = {}
    for idx, part in enumerate(f.parts):
        pname = part.name() if callable(getattr(part, "name", None)) else str(getattr(part, "name", "") or idx)
        for name, ch in part.channels.items():
            key = name if name not in out else f"{pname}.{name}"
            out[key] = np.asarray(ch.pixels, dtype=np.float32)
    return out


def write_exr(channels: dict[str, np.ndarray]) -> bytes:
    """{channel name: HxW float array} -> single-part EXR bytes (float32, ZIP). Used by tests/tools."""
    import os
    import tempfile

    header = {"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}
    chans = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in channels.items()}
    fd, path = tempfile.mkstemp(suffix=".exr")
    os.close(fd)
    try:
        OpenEXR.File(header, chans).write(path)
        with open(path, "rb") as fh:
            return fh.read()
    finally:
        os.unlink(path)


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _split(name: str) -> tuple[str, str]:
    """`ViewLayer.Combined.R` -> (`ViewLayer.Combined`, `R`); `R` -> (``, `R`)."""
    if "." in name:
        layer, comp = name.rsplit(".", 1)
        return layer, comp
    return "", name


def exr_layer(channels: dict[str, np.ndarray], *names: str, exclude: tuple[str, ...] = ()) -> dict[str, np.ndarray]:
    """Components (`R`, `G`, ...) of the first layer whose last path element matches one of `names`
    (case/space-insensitive), e.g. `exr_layer(ch, "Combined")` -> {"R": .., "G": .., "B": .., "A": ..}.
    An empty name matches bare (unprefixed) channels."""
    wanted = [_norm(n) for n in names]
    groups: dict[str, dict[str, np.ndarray]] = {}
    for full, arr in channels.items():
        layer, comp = _split(full)
        groups.setdefault(layer, {})[comp.upper()] = arr
    for want in wanted:
        # exact last element first, then substring match
        for exact in (True, False):
            for layer, comps in groups.items():
                last = _norm(layer.rsplit(".", 1)[-1]) if layer else ""
                if any(x in _norm(layer) for x in exclude):
                    continue
                if (exact and last == want) or (not exact and want and want in _norm(layer)):
                    return comps
    return {}


def fit_aspect(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Centre-crop `img` to width:height, then resize to exactly (height, width)."""
    h, w = img.shape[:2]
    target = width / height
    if abs(w / h - target) > 1e-4:
        if w / h > target:
            nw = max(1, int(round(h * target)))
            x0 = (w - nw) // 2
            img = img[:, x0 : x0 + nw]
        else:
            nh = max(1, int(round(w / target)))
            y0 = (h - nh) // 2
            img = img[y0 : y0 + nh]
    h, w = img.shape[:2]
    if (w, h) == (width, height):
        return np.ascontiguousarray(img)
    interp = cv2.INTER_AREA if w > width else cv2.INTER_LANCZOS4
    return cv2.resize(np.ascontiguousarray(img), (width, height), interpolation=interp)
