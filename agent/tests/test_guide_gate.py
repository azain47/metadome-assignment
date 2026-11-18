"""The guide gate rejects a guide that shows only a horizon (no image budget spent on it) and
approves one with a visible road."""

import asyncio

import numpy as np

from app.checks import CheckContext, run_checks

H, W = 512, 768


def _ctx(lines: np.ndarray, road: np.ndarray) -> CheckContext:
    return CheckContext(plate=np.zeros((H, W, 3), np.uint8), guides={"lines": lines, "road_mask": road})


def test_horizon_only_guide_is_rejected():
    # Camera at road height: one horizon line, the road a 2-px sliver (ladakh-b68d97).
    lines = np.zeros((H, W), np.uint8)
    lines[350:353] = 255
    road = np.zeros((H, W), np.uint8)
    road[352:354] = 255
    [result] = asyncio.run(run_checks("guide", _ctx(lines, road)))
    assert not result.passed and result.code == "guide_unreadable"


def test_guide_with_receding_road_is_approved():
    lines = np.zeros((H, W), np.uint8)
    road = np.zeros((H, W), np.uint8)
    for y in range(300, H):  # road widening towards the camera, edges drawn as lines
        half = (y - 300) * 1.5
        left, right = int(W / 2 - half), int(W / 2 + half)
        road[y, max(left, 0):min(right, W)] = 255
        lines[y, max(left, 0)] = lines[y, min(right, W - 1)] = 255
    # The guides node passes the road mask already thresholded (bool), the lines as 8-bit.
    [result] = asyncio.run(run_checks("guide", _ctx(lines, road > 127)))
    assert result.passed and result.code is None
