import numpy as np
import pytest

from app.imaging.io import decode_png, decode_rgb, encode_png, exr_layer, fit_aspect, read_exr, write_exr


def test_png_round_trip_rgb_rgba_grey_16bit():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (20, 30, 3), dtype=np.uint8)
    rgba = rng.integers(0, 256, (20, 30, 4), dtype=np.uint8)
    grey = rng.integers(0, 256, (20, 30), dtype=np.uint8)
    deep = rng.integers(0, 65536, (20, 30, 3), dtype=np.uint16)
    for arr in (rgb, rgba, grey, deep):
        out = decode_png(encode_png(arr))
        assert out.dtype == arr.dtype and out.shape == arr.shape
        assert np.array_equal(out, arr)
    # channel order: pure red stays red
    red = np.zeros((4, 4, 3), np.uint8)
    red[..., 0] = 255
    assert decode_png(encode_png(red))[0, 0].tolist() == [255, 0, 0]
    assert decode_rgb(encode_png(rgba)).shape == (20, 30, 3)
    assert decode_rgb(encode_png(deep)).dtype == np.uint8


def test_float_png_is_scaled():
    out = decode_png(encode_png(np.full((2, 2), 0.5, np.float32)))
    assert out.dtype == np.uint8 and int(out[0, 0]) == 128


def _blender_like_exr():
    h, w = 16, 24
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    alpha = (xx > 8).astype(np.float32)
    ch = {
        "ViewLayer.Combined.R": 2.5 * alpha,  # premultiplied, HDR value
        "ViewLayer.Combined.G": 0.5 * alpha,
        "ViewLayer.Combined.B": 0.1 * alpha + 0.05 * (1 - alpha),  # glass-like: colour with alpha 0
        "ViewLayer.Combined.A": alpha,
        "ViewLayer.Shadow Catcher.R": 1 - 0.4 * (yy > 10),
        "ViewLayer.Shadow Catcher.G": 1 - 0.4 * (yy > 10),
        "ViewLayer.Shadow Catcher.B": 1 - 0.4 * (yy > 10),
        "ViewLayer.Depth.Z": 3.0 + xx,
    }
    return ch


def test_read_exr_multilayer_names_and_values():
    ch = _blender_like_exr()
    got = read_exr(write_exr(ch))
    assert set(got) == set(ch)
    for k, v in ch.items():
        assert got[k].dtype == np.float32 and got[k].shape == v.shape
        np.testing.assert_allclose(got[k], v, rtol=0, atol=1e-6)

    comb = exr_layer(got, "Combined")
    assert set(comb) == {"R", "G", "B", "A"}
    assert comb["R"].max() == pytest.approx(2.5)  # HDR survives
    shadow = exr_layer(got, "ShadowCatcher")  # spacing/case-insensitive
    assert set(shadow) == {"R", "G", "B"} and shadow["R"].min() == pytest.approx(0.6)
    assert "Z" in exr_layer(got, "depth")


def test_read_exr_half_and_bare_channels():
    a = np.linspace(0, 1, 12, dtype=np.float32).reshape(3, 4)
    got = read_exr(write_exr({"R": a, "G": a, "B": a, "A": a}))
    assert set(exr_layer(got, "")) == {"R", "G", "B", "A"}

    import OpenEXR, tempfile, os

    fd, p = tempfile.mkstemp(suffix=".exr")
    os.close(fd)
    OpenEXR.File({"compression": OpenEXR.ZIP_COMPRESSION, "type": OpenEXR.scanlineimage}, {"V": a.astype(np.float16)}).write(p)
    data = open(p, "rb").read()
    os.unlink(p)
    v = read_exr(data)["V"]
    assert v.dtype == np.float32
    np.testing.assert_allclose(v, a, atol=1e-3)


def test_fit_aspect_crops_centre_then_resizes():
    img = np.zeros((100, 300, 3), np.uint8)  # 3:1
    img[:, 75:225] = 200  # centre 150 columns = what a 3:2 crop keeps
    out = fit_aspect(img, 150, 100)
    assert out.shape == (100, 150, 3)
    assert out.min() == 200
    tall = np.zeros((400, 100), np.uint8)
    tall[150:250] = 255
    out = fit_aspect(tall, 30, 20)
    assert out.shape == (20, 30)
    assert out.mean() > 250  # the kept centre band is the white one
    same = np.ones((10, 15, 3), np.uint8)
    assert fit_aspect(same, 15, 10).shape == (10, 15, 3)
    assert fit_aspect(same, 30, 20).shape == (20, 30, 3)  # upscale
