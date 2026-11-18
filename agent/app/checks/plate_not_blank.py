"""Plate is not a safety-checker black frame or a near-uniform image."""

from __future__ import annotations

import numpy as np

from .base import CheckContext, CheckResult

BLACK_P99_5 = 10.0  # 99.5th percentile below this = black frame
MIN_STD = 4.0  # grey-level std below this = near-uniform frame


class PlateNotBlank:
    name = "plate_not_blank"
    stage = "plate"
    blocking = True

    async def run(self, ctx: CheckContext) -> CheckResult:
        img = np.asarray(ctx.plate)
        if img.size == 0:
            return CheckResult(self.name, False, 0.0, self.blocking, "plate_blank", "empty image")
        p995 = float(np.percentile(img, 99.5))
        std = float(img.astype(np.float32).mean(axis=-1).std() if img.ndim == 3 else img.std())
        black = p995 < BLACK_P99_5
        uniform = std < MIN_STD
        passed = not (black or uniform)
        score = 0.0 if black else min(1.0, std / (4 * MIN_STD))
        detail = f"p99.5={p995:.1f} std={std:.2f}" + (" (black)" if black else " (near-uniform)" if uniform else "")
        return CheckResult(self.name, passed, score, self.blocking, None if passed else "plate_blank", detail)


CHECK = PlateNotBlank()
