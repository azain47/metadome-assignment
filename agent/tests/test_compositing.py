import numpy as np
import pytest

from app.compositing import CarRender, Composite, composite, load_car_render
from app.imaging.io import encode_png, write_exr


def _car(h=60, w=90):
    rgba = np.zeros((h, w, 4), np.uint8)
    rgba[20:40, 30:60, :3] = (200, 30, 40)
    rgba[20:40, 30:60, 3] = 255
    rgba[19, 30:60, :3] = (200, 30, 40)
    rgba[19, 30:60, 3] = 128  # soft edge
    return rgba


def _shadow(h=60, w=90):
    s = np.ones((h, w), np.float32)
    s[40:50, 25:65] = 0.4
    return s


def test_load_car_render_with_and_without_shadow():
    rgba = _car()
    s = _shadow()
    exr = write_exr({
        "ViewLayer.Combined.R": np.zeros_like(s), "ViewLayer.Combined.A": np.zeros_like(s),
        "ViewLayer.Shadow Catcher.R": s, "ViewLayer.Shadow Catcher.G": s, "ViewLayer.Shadow Catcher.B": s,
    })
    car = load_car_render(encode_png(rgba), exr)
    assert car.rgba.shape == (60, 90, 4) and car.rgba.dtype == np.uint8
    assert np.array_equal(car.rgba, rgba)
    assert car.shadow.dtype == np.float32 and car.shadow.min() == pytest.approx(0.4)

    assert load_car_render(encode_png(rgba), None).shadow is None
    no_shadow = write_exr({"ViewLayer.Combined.R": s})
    assert load_car_render(encode_png(rgba), no_shadow).shadow is None
    wrong_size = write_exr({"ViewLayer.Shadow Catcher.R": np.ones((10, 10), np.float32)})
    assert load_car_render(encode_png(rgba), wrong_size).shadow is None
    rgb_only = load_car_render(encode_png(rgba[..., :3]), None)
    assert rgb_only.rgba[..., 3].min() == 255


def test_composite_keeps_car_pixels_and_darkens_shadow():
    plate = np.full((60, 90, 3), 180, np.uint8)
    car = CarRender(rgba=_car(), shadow=_shadow())
    out = composite(plate, car)
    assert isinstance(out, Composite)
    assert out.final.shape == (60, 90, 3) and out.final.dtype == np.uint8
    assert out.alpha.dtype == np.float32 and out.alpha.shape == (60, 90)
    opaque = car.rgba[..., 3] == 255
    assert np.array_equal(out.pre_lens[opaque], car.rgba[..., :3][opaque])  # exact
    assert np.array_equal(out.final, out.pre_lens)  # grain off
    assert out.pre_lens[45, 27].max() < 140  # shadowed plate darker (linear 0.4 multiply)
    assert out.pre_lens[5, 5].tolist() == [180, 180, 180]  # unshadowed plate untouched
    edge = out.pre_lens[19, 40].astype(int)
    assert 180 > edge[1] > 30  # half-alpha edge is a blend


def test_composite_resizes_plate_and_adds_grain():
    plate = np.full((120, 200, 3), 100, np.uint8)  # 5:3, 2x size -> cropped + resized
    car = CarRender(rgba=_car(), shadow=None)
    out = composite(plate, car, grain=0.02, seed=3)
    assert out.final.shape == (60, 90, 3)
    assert not np.array_equal(out.final, out.pre_lens)
    again = composite(plate, car, grain=0.02, seed=3)
    assert np.array_equal(out.final, again.final)  # seeded
