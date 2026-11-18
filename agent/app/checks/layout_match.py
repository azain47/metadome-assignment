"""Plate keeps the Blender layout: overlap of the plate's Canny edges with the line guide.

score = F1 of precision (plate edges lying within TOL of a guide line) and recall (guide-line
pixels with a plate edge within TOL), TOL = 1 % of the width. Plate texture lowers precision, so
real photos score lower than clean renders. PASS_F1 is UNCALIBRATED: set it from tonight's or
tomorrow's real plates.
"""

from __future__ import annotations

import asyncio

import cv2
import numpy as np

from app.imaging.guides import line_guide

from .base import CheckContext, CheckResult

PASS_F1 = 0.25  # uncalibrated
TOL_FRAC = 0.01  # edge tolerance as a fraction of the width


def _canny(grey: np.ndarray) -> np.ndarray:
    # blur scaled to the width (sigma 2 px at 768, 4 px at 1536) suppresses texture edges
    g = cv2.GaussianBlur(grey, (0, 0), max(1.0, grey.shape[1] / 384))
    mag = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    hi = max(float(np.percentile(mag, 95)), 8.0)  # keep roughly the strongest 5 % of gradients
    return cv2.Canny(g, hi * 0.5, hi, L2gradient=True) > 0


def edge_f1(plate: np.ndarray, lines: np.ndarray) -> tuple[float, float, float]:
    """(f1, precision, recall) of plate edges against a line guide (resized to the plate)."""
    h, w = plate.shape[:2]
    grey = cv2.cvtColor(np.ascontiguousarray(plate[..., :3]), cv2.COLOR_RGB2GRAY) if plate.ndim == 3 else plate
    lines = lines if lines.shape[:2] == (h, w) else cv2.resize(lines, (w, h), interpolation=cv2.INTER_NEAREST)
    guide = lines > 127
    edges = _canny(grey)
    r = max(1, int(round(TOL_FRAC * w)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    guide_d = cv2.dilate(guide.astype(np.uint8), k) > 0
    edges_d = cv2.dilate(edges.astype(np.uint8), k) > 0
    precision = float(guide_d[edges].mean()) if edges.any() else 0.0
    recall = float(edges_d[guide].mean()) if guide.any() else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return f1, precision, recall


class LayoutMatch:
    name = "layout_match"
    stage = "plate"
    blocking = True

    async def run(self, ctx: CheckContext) -> CheckResult:
        lines = ctx.guides.get("lines")
        if lines is None:
            if ctx.guides.get("grey") is None:
                raise ValueError("layout_match needs guides['lines'] or guides['grey']")
            lines = line_guide(ctx.guides["grey"])
        f1, p, r = await asyncio.to_thread(edge_f1, np.asarray(ctx.plate), np.asarray(lines))
        passed = f1 >= PASS_F1
        detail = f"edge F1={f1:.3f} (precision {p:.3f}, recall {r:.3f}; pass >= {PASS_F1}, uncalibrated)"
        return CheckResult(self.name, passed, round(f1, 4), self.blocking, None if passed else "layout_mismatch", detail)


CHECK = LayoutMatch()
