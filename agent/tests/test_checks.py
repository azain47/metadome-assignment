import sys
from pathlib import Path

import cv2
import numpy as np

from app.checks import CHECKS, CheckContext, CheckResult, run_checks
from app.checks import layout_match as layout_mod
from app.compositing import CarRender, composite
from app.imagegen import PlateRequest
from app.imagegen.fake_backend import FakeImageBackend
from app.imaging.guides import line_guide
from app.imaging.io import decode_png, encode_png

sys.path.insert(0, str(Path(__file__).parent))
from test_imagegen_fake import synthetic_guide  # noqa: E402

W, H = 768, 512


async def _plate(seed=1):
    g = encode_png(synthetic_guide(W, H))
    res = await FakeImageBackend().generate(PlateRequest(g, g, g, [], "x", 1, W, H, seed))
    return decode_png(res.images[0])


def _ctx(plate, **kw):
    grey = synthetic_guide(W, H)
    return CheckContext(plate=plate, guides={"grey": grey, "lines": line_guide(grey)}, layout={}, plan={}, facts={}, **kw)


def _by_name(results):
    return {r.name: r for r in results}


def test_registry():
    names = [c.name for c in CHECKS]
    assert names == ["guide_readable", "plate_not_blank", "layout_match", "car_pixels_untouched"]
    assert {c.name: c.stage for c in CHECKS} == {
        "guide_readable": "guide", "plate_not_blank": "plate", "layout_match": "plate", "car_pixels_untouched": "final"}
    assert all(c.blocking for c in CHECKS)


async def test_plate_stage_pass_on_good_plate():
    res = _by_name(await run_checks("plate", _ctx(await _plate())))
    assert set(res) == {"plate_not_blank", "layout_match"}
    assert res["plate_not_blank"].passed and res["plate_not_blank"].code is None
    lm = res["layout_match"]
    assert lm.passed and lm.score > 0.6, lm.detail


async def test_plate_blank_black_and_uniform():
    for img in (np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 128, np.uint8) + np.random.default_rng(0).integers(0, 2, (H, W, 3), dtype=np.uint8)):
        r = _by_name(await run_checks("plate", _ctx(img)))["plate_not_blank"]
        assert not r.passed and r.code == "plate_blank" and r.blocking


async def test_layout_mismatch_on_shifted_flipped_and_noise():
    good = await _plate()
    shifted = cv2.warpAffine(good, np.float32([[1, 0, 0], [0, 1, 0.06 * H]]), (W, H), borderMode=cv2.BORDER_REFLECT)
    noise = np.random.default_rng(0).integers(0, 256, (H, W, 3), dtype=np.uint8)
    for img in (shifted, good[::-1].copy(), noise):
        r = _by_name(await run_checks("plate", _ctx(img)))["layout_match"]
        assert not r.passed and r.code == "layout_mismatch", r.detail
        assert r.score < layout_mod.PASS_F1


async def test_layout_match_builds_lines_from_grey():
    plate = await _plate()
    ctx = CheckContext(plate=plate, guides={"grey": synthetic_guide(W, H)}, layout={}, plan={}, facts={})
    assert _by_name(await run_checks("plate", ctx))["layout_match"].passed


def _car():
    rgba = np.zeros((H, W, 4), np.uint8)
    rgba[250:400, 200:560, :3] = (30, 60, 200)
    rgba[250:400, 200:560, 3] = 255
    return CarRender(rgba=rgba, shadow=None)


async def test_car_pixels_untouched_pass_and_fail():
    plate = await _plate()
    car = _car()
    comp = composite(plate, car, grain=0.02)
    r = _by_name(await run_checks("final", _ctx(plate, composite=comp, car=car)))["car_pixels_untouched"]
    assert r.passed and r.score == 1.0 and r.code is None  # grain is after pre_lens

    comp.pre_lens[300:320, 300:340] += 10  # repaint a patch of the car
    r = _by_name(await run_checks("final", _ctx(plate, composite=comp, car=car)))["car_pixels_untouched"]
    assert not r.passed and r.code == "car_pixels_changed" and r.blocking

    r = _by_name(await run_checks("final", _ctx(plate)))["car_pixels_untouched"]
    assert not r.passed and r.code == "car_pixels_changed"


async def test_run_checks_never_raises(monkeypatch):
    class Boom:
        name = "boom"
        stage = "plate"
        blocking = True

        async def run(self, ctx):
            raise RuntimeError("kaput")

    monkeypatch.setattr("app.checks.CHECKS", [Boom(), *CHECKS])
    res = await run_checks("plate", _ctx(await _plate()))
    boom = res[0]
    assert isinstance(boom, CheckResult)
    assert boom.name == "boom" and not boom.passed and not boom.blocking and boom.code == "check_error"
    assert "kaput" in boom.detail
    assert len(res) == 3  # the other plate checks still ran
    # missing guides make layout_match raise inside -> check_error, not an exception
    bad = CheckContext(plate=await _plate(), guides={}, layout={}, plan={}, facts={})
    lm = _by_name(await run_checks("plate", bad))["layout_match"]
    assert lm.code == "check_error" and not lm.blocking
