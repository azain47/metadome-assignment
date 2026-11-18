"""Compositing assertion: opaque car pixels in `pre_lens` equal the car render."""

from __future__ import annotations

import numpy as np

from .base import CheckContext, CheckResult

ALPHA_MIN = 0.99
MAX_LEVEL_DIFF = 2  # per channel, 8-bit levels
MIN_FRACTION = 0.995


class CarPixelsUntouched:
    name = "car_pixels_untouched"
    stage = "final"
    blocking = True

    async def run(self, ctx: CheckContext) -> CheckResult:
        code = "car_pixels_changed"
        if ctx.composite is None or ctx.car is None:
            return CheckResult(self.name, False, 0.0, self.blocking, code, "missing composite or car render")
        pre = ctx.composite.pre_lens
        rgba = ctx.car.rgba
        if pre.shape[:2] != rgba.shape[:2]:
            return CheckResult(self.name, False, 0.0, self.blocking, code,
                               f"size mismatch: pre_lens {pre.shape[:2]} vs car {rgba.shape[:2]}")
        mask = rgba[..., 3].astype(np.float32) / 255.0 >= ALPHA_MIN
        n = int(mask.sum())
        if n == 0:
            return CheckResult(self.name, True, 1.0, self.blocking, None, "no opaque car pixels to compare")
        diff = np.abs(pre[..., :3].astype(np.int16) - rgba[..., :3].astype(np.int16)).max(axis=-1)
        frac = float((diff[mask] <= MAX_LEVEL_DIFF).mean())
        passed = frac >= MIN_FRACTION
        detail = f"{frac * 100:.2f} % of {n} opaque car pixels within ±{MAX_LEVEL_DIFF} levels (need {MIN_FRACTION * 100:.1f} %)"
        return CheckResult(self.name, passed, round(frac, 5), self.blocking, None if passed else code, detail)


CHECK = CarPixelsUntouched()
