import numpy as np
import pytest

from app.imagegen import BACKENDS, PlateRequest, get_backend
from app.imagegen.fake_backend import FakeImageBackend, classify
from app.imaging.io import decode_png, encode_png


def _srgb(v):
    return 1.055 * v ** (1 / 2.4) - 0.055


def synthetic_guide(w=768, h=512, encode=True) -> np.ndarray:
    """Grey guide with sky, ridges, ground and a road converging to the horizon."""
    tone = (lambda v: int(round(_srgb(v) * 255))) if encode else (lambda v: int(round(v * 255)))
    img = np.full((h, w), tone(0.55), np.uint8)
    hz = int(0.38 * h)
    img[:hz] = tone(0.92)
    yy, xx = np.mgrid[0:h, 0:w]
    for cx, top, half in ((0.2, 0.18, 0.18), (0.55, 0.12, 0.22), (0.85, 0.22, 0.15)):
        ridge = (yy < hz) & (yy > top * h + np.abs(xx - cx * w) * (hz - top * h) / (half * w))
        img[ridge] = tone(0.42)
    vx = 0.52 * w
    t = np.clip((yy - hz) / (h - hz), 0, 1)
    road = (yy > hz) & (np.abs(xx - vx) < t * 0.45 * w)
    img[road] = tone(0.22)
    return np.repeat(img[..., None], 3, axis=2)


def _req(n=2, seed=7, w=768, h=512):
    g = encode_png(synthetic_guide(w, h))
    return PlateRequest(guide_grey=g, guide_lines=g, guide_depth=g, references=[], prompt="alpine road", n=n,
                        width=w, height=h, seed=seed)


def test_registry_and_env(monkeypatch):
    assert set(BACKENDS) == {"openai", "fal", "fake"}
    monkeypatch.setenv("IMAGE_BACKEND", "fake")
    monkeypatch.setenv("IMAGE_MODEL", "")
    b = get_backend()
    assert b.name == "fake" and b.model == "fake-tones-v1"
    assert get_backend("fake", "x").model == "x"
    with pytest.raises(ValueError):
        get_backend("nope")


@pytest.mark.parametrize("encode", [True, False])
def test_classify_both_encodings(encode):
    labels, names = classify(synthetic_guide(encode=encode)[..., 0])
    found = {names[i] for i in np.unique(labels) if i >= 0}
    assert {"sky", "ridge", "ground", "road"} <= found
    assert (labels < 0).mean() < 0.01


async def test_fake_end_to_end_deterministic_and_jittered():
    b = FakeImageBackend()
    req = _req(n=3)
    assert b.estimate_cost(req) == 0.0
    r1 = await b.generate(req)
    r2 = await b.generate(req)
    assert r1.backend == "fake" and r1.cost_usd == 0.0 and len(r1.images) == 3
    assert r1.images == r2.images  # deterministic
    imgs = [decode_png(x) for x in r1.images]
    for im in imgs:
        assert im.shape == (512, 768, 3) and im.dtype == np.uint8
    assert not np.array_equal(imgs[0], imgs[1])  # candidates differ
    jit = r1.meta["jitter"]
    assert len(jit) == 3 and all(abs(j["dx_px"]) <= 0.006 * 768 + 1e-6 for j in jit)
    im = imgs[0].astype(int)
    sky, road = im[20, 384], im[480, 400]
    assert sky[2] > sky[0]  # bluish sky
    assert road.mean() < 110 and im[:150].mean() > road.mean()  # dark asphalt
    assert (await b.health())["ok"]


async def test_fake_resizes_to_requested_size():
    g = encode_png(synthetic_guide(400, 400))
    req = PlateRequest(guide_grey=g, guide_lines=g, guide_depth=g, references=[], prompt="p", n=1, width=300, height=200)
    res = await FakeImageBackend().generate(req)
    assert decode_png(res.images[0]).shape == (200, 300, 3)
