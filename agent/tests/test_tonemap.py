import numpy as np
import pytest

from app.imaging.io import write_exr
from app.imaging.tonemap import car_rgba_from_exr, encode_srgb, pbr_neutral


def _exr(rgb: np.ndarray, alpha: np.ndarray) -> bytes:
    ch = {f"ViewLayer.Combined.{c}": rgb[..., i] for i, c in enumerate("RGB")}
    ch["ViewLayer.Combined.A"] = alpha
    ch["ViewLayer.Shadow Catcher.R"] = np.ones_like(alpha)
    return write_exr(ch)


def test_pbr_neutral_below_compression_only_subtracts_toe_offset():
    # min channel >= 0.08 -> constant 0.04 offset; peak after offset < 0.76 -> no compression
    rgb = np.array([[0.5, 0.2, 0.1], [0.7, 0.7, 0.7], [0.3, 0.6, 0.79]], np.float32)
    np.testing.assert_allclose(pbr_neutral(rgb), rgb - 0.04, atol=1e-6)
    # toe: min channel x < 0.08 -> offset x - 6.25 x^2 applied to all channels
    rgb = np.array([[0.4, 0.04, 0.2]], np.float32)
    off = 0.04 - 6.25 * 0.04**2
    np.testing.assert_allclose(pbr_neutral(rgb), rgb - off, atol=1e-6)
    assert pbr_neutral(np.zeros((1, 3), np.float32)).max() == 0.0


def test_pbr_neutral_bright_input_approaches_white_without_exceeding_one():
    rgb = np.array([[1.0, 0.5, 0.2], [10.0, 5.0, 2.0], [1e4, 5e3, 2e3], [1e4, 1e4, 1e4]], np.float32)
    out = pbr_neutral(rgb)
    assert out.max() <= 1.0 and out.min() >= 0.0
    np.testing.assert_allclose(out[-1], 1.0, atol=1e-3)
    np.testing.assert_allclose(out[2], 1.0, atol=1e-2)  # hue desaturated towards white
    # brighter input -> peak rises and saturation (max - min) drops
    sat = out[:3].max(axis=1) - out[:3].min(axis=1)
    assert np.all(np.diff(out[:3].max(axis=1)) > 0)
    assert np.all(np.diff(sat) < 0)
    # continuous at the compression knee
    lo = pbr_neutral(np.array([[0.7999, 0.5, 0.5]], np.float32))
    hi = pbr_neutral(np.array([[0.8001, 0.5, 0.5]], np.float32))
    np.testing.assert_allclose(lo, hi, atol=1e-3)


def test_encode_srgb_piecewise_and_rounding():
    lin = np.array([0.0, 0.0031308, 0.002, 0.18, 0.5, 1.0, 2.0, -1.0], np.float32)
    out = encode_srgb(lin)
    assert out.dtype == np.uint8
    # 0.002 is in the linear segment: 0.002 * 12.92 * 255 = 6.59 -> 7 (gamma 2.2 would give 15)
    assert out.tolist() == [0, 10, 7, 118, 188, 255, 255, 0]


def test_car_rgba_from_exr_exposure_scales_scene_linear_before_tonemap():
    rgb = np.zeros((2, 3, 3), np.float32)
    rgb[0, 0] = [3.0, 1.2, 0.4]  # compressed highlight
    rgb[0, 1] = [0.6, 0.3, 0.12]
    rgb[1, 2] = [8.0, 8.0, 8.0]
    alpha = np.ones((2, 3), np.float32)
    exr = _exr(rgb, alpha)
    out = car_rgba_from_exr(exr, exposure_ev=-1.0)
    assert out.shape == (2, 3, 4) and out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., :3], encode_srgb(pbr_neutral(rgb * 0.5)))
    np.testing.assert_array_equal(out[..., 3], 255)
    base = car_rgba_from_exr(exr)
    np.testing.assert_array_equal(base[..., :3], encode_srgb(pbr_neutral(rgb)))
    # lowering exposure recovers highlight colour (more R-B spread than at 0 EV)
    spread = lambda px: int(px[0]) - int(px[2])
    assert spread(out[0, 0]) > spread(base[0, 0])


def test_car_rgba_from_exr_unpremultiplies_alpha():
    colour = np.array([0.6, 0.3, 0.12], np.float32)
    rgb = np.zeros((1, 3, 3), np.float32)
    rgb[0, 0] = colour  # opaque
    rgb[0, 1] = colour * 0.5  # premultiplied at half alpha
    alpha = np.array([[1.0, 0.5, 0.0]], np.float32)
    out = car_rgba_from_exr(_exr(rgb, alpha))
    np.testing.assert_array_equal(out[0, 1, :3], out[0, 0, :3])
    assert out[0, :, 3].tolist() == [255, 128, 0]
    np.testing.assert_array_equal(out[0, 2], [0, 0, 0, 0])


def test_car_rgba_from_exr_requires_combined_layer():
    exr = write_exr({"ViewLayer.Depth.Z": np.ones((2, 2), np.float32)})
    with pytest.raises(ValueError):
        car_rgba_from_exr(exr)
