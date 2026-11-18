"""Check registry. Add a check: one module defining `CHECK` + one entry in `CHECKS`."""

from __future__ import annotations

from . import car_pixels_untouched, guide_readable, layout_match, plate_not_blank
from .base import Check, CheckContext, CheckResult, Stage

CHECKS: list[Check] = [
    guide_readable.CHECK,
    plate_not_blank.CHECK,
    layout_match.CHECK,
    car_pixels_untouched.CHECK,
]


async def run_checks(stage: Stage, ctx: CheckContext) -> list[CheckResult]:
    """Run the registered checks of `stage` in order. Never raises: an exception inside a check
    becomes a failed, non-blocking result with code "check_error"."""
    results: list[CheckResult] = []
    for check in CHECKS:
        if check.stage != stage:
            continue
        try:
            results.append(await check.run(ctx))
        except Exception as e:  # a broken check must not stop the pipeline
            results.append(CheckResult(check.name, False, 0.0, False, "check_error", f"{type(e).__name__}: {e}"[:500]))
    return results


__all__ = ["CHECKS", "Check", "CheckContext", "CheckResult", "Stage", "run_checks"]
