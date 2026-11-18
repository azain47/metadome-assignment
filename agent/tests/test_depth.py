"""Plate depth: calibration of relative inverse depth to metres on the road. Synthetic scene: a
ground plane seen by a level camera 1.5 m up with a 100 px focal length, horizon at row 40, sky
above it."""

import numpy as np
import pytest

from app.imaging.depth import at_size, calibrate

H, W, HORIZON = 120, 180, 40
SCALE, SHIFT = 0.015, -0.003


def _scene() -> tuple[np.ndarray, np.ndarray]:
    """(depth in metres with inf sky, road mask) of the synthetic scene."""
    rows = np.arange(H, dtype=np.float64)[:, None].repeat(W, 1)
    with np.errstate(divide="ignore"):
        z = np.where(rows > HORIZON, 150.0 / (rows - HORIZON), np.inf).astype(np.float32)
    road = np.zeros((H, W), bool)
    road[HORIZON + 1:, 50:130] = True
    return z, road


def _disparity(z: np.ndarray, scale: float = SCALE, shift: float = SHIFT) -> np.ndarray:
    """What the depth model reports: affine in inverse depth, never below 0, exactly 0 for sky."""
    with np.errstate(divide="ignore"):
        d = (1.0 / z - shift) / scale
    return np.where(np.isfinite(z), np.maximum(d, 0.0), 0.0).astype(np.float32)


def test_calibration_recovers_metres_and_keeps_the_sky_infinite():
    z, road = _scene()
    depth = calibrate(_disparity(z), z, road)
    assert depth.scale == pytest.approx(SCALE, rel=1e-3) and depth.shift == pytest.approx(SHIFT, rel=1e-2)
    ground = np.isfinite(z)
    np.testing.assert_allclose(depth.metres[ground & (z < 60)], z[ground & (z < 60)], rtol=1e-3)
    assert np.isinf(depth.metres[~ground]).all()
    assert depth.road_err < 1e-3


def test_calibration_ignores_paint_and_objects_on_the_road():
    z, road = _scene()
    d = _disparity(z)
    rng = np.random.default_rng(0)
    stray = road & (rng.random((H, W)) < 0.1)  # 10 % of the road reads 3x nearer
    d[stray] *= 3.0
    depth = calibrate(d, z, road)
    assert depth.scale == pytest.approx(SCALE, rel=0.02)
    assert depth.road_err < 0.01


def test_sky_stays_infinitely_far_where_the_road_line_would_put_it_nearby():
    z, road = _scene()
    depth = calibrate(_disparity(z, scale=0.02, shift=0.01), z, road)  # the line alone: sky at 100 m
    assert depth.shift == pytest.approx(0.01, rel=1e-2)
    assert np.isinf(depth.metres[~np.isfinite(z)]).all()
    np.testing.assert_allclose(depth.metres[road & (z < 50)], z[road & (z < 50)], rtol=1e-3)
    assert depth.road_err < 1e-3


def test_too_little_road_raises():
    z, road = _scene()
    tiny = np.zeros_like(road)
    tiny[100:105, 60:70] = True
    with pytest.raises(ValueError, match="road pixels"):
        calibrate(_disparity(z), z, tiny)


def test_at_size_resizes_boolean_masks():
    _, road = _scene()
    small = at_size(road, (H // 2, W // 2))
    assert small.shape == (H // 2, W // 2) and small.dtype == np.uint8 and small.max() == 1
