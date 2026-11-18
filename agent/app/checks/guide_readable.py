"""Guide approval, before any image is generated: the Blender guide must show a scene the image
model can follow, i.e. visible road and lines spread over the frame, not just a horizon.

At the guide stage `CheckContext.plate` is the grey guide; `guides` holds "lines" and "road_mask".
Measured (2026-11-17): approved guides alps / ladakh / coast have line rows 0.65 / 1.00 / 0.41
and road 0.33 / 0.49 / 0.15 of the frame; the rejected ladakh-b68d97 guide (camera at road height
looking up) had 0.01 and 0.002, and all 8 plates generated from it failed layout_match.
"""

from __future__ import annotations

import numpy as np

from .base import CheckContext, CheckResult

MIN_LINE_ROWS = 0.15  # share of image rows holding any guide line
MIN_ROAD = 0.03  # share of the frame covered by road


def _bool(a: np.ndarray) -> np.ndarray:
    """A mask as booleans: accepts 8-bit images (white = on) or masks already thresholded."""
    a = a[..., 0] if a.ndim == 3 else a
    return a if a.dtype == bool else a > 127


class GuideReadable:
    name = "guide_readable"
    stage = "guide"
    blocking = True

    async def run(self, ctx: CheckContext) -> CheckResult:
        line_rows = float(_bool(ctx.guides["lines"]).any(axis=1).mean())
        road = float(_bool(ctx.guides["road_mask"]).mean())
        score = min(1.0, line_rows / MIN_LINE_ROWS, road / MIN_ROAD)
        passed = line_rows >= MIN_LINE_ROWS and road >= MIN_ROAD
        detail = f"lines span {line_rows:.0%} of the height (min {MIN_LINE_ROWS:.0%}), road covers {road:.1%} (min {MIN_ROAD:.0%})"
        return CheckResult(self.name, passed, round(score, 3), self.blocking, None if passed else "guide_unreadable", detail)


CHECK = GuideReadable()
