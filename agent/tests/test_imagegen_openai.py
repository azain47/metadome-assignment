import base64
from types import SimpleNamespace

import httpx
import numpy as np
import openai
import pytest

from app.imagegen import ContentPolicyError, ImageGenError, MissingCredentialsError, PlateRequest, get_backend
from app.imagegen.openai_backend import OpenAIImageBackend
from app.imaging.io import decode_png, encode_png


def _png(h, w, value=128):
    rng = np.random.default_rng(value)
    return encode_png(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


class FakeImages:
    def __init__(self, data=None, exc=None):
        self.calls = []
        self.data = data
        self.exc = exc

    async def edit(self, **kwargs):
        self.calls.append(kwargs)
        if self.exc:
            raise self.exc
        return SimpleNamespace(
            data=self.data,
            usage=SimpleNamespace(
                input_tokens=4700,
                output_tokens=2706,
                input_tokens_details=SimpleNamespace(text_tokens=100, image_tokens=4600),
                model_dump=lambda: {"input_tokens": 4700},
            ),
        )


def _client(images):
    return SimpleNamespace(images=images)


def _req(**kw):
    base = dict(
        guide_grey=_png(512, 768),  # wrong size on purpose: must be resized to 1536x1024
        guide_lines=b"",
        guide_depth=b"",
        references=[_png(1200, 1600, 1), _png(3000, 2000, 2)],  # 4:3 and 2:3 photos
        prompt="PROMPT AS GIVEN",
        n=2,
    )
    base.update(kw)
    return PlateRequest(**base)


async def test_request_shape_and_result():
    out = [SimpleNamespace(b64_json=base64.b64encode(_png(1024, 1536, 5)).decode()),
           SimpleNamespace(b64_json=base64.b64encode(encode_png(np.zeros((1024, 1536, 3), np.uint8))).decode())]
    images = FakeImages(data=out)
    b = OpenAIImageBackend(client=_client(images))
    assert b.model == "gpt-image-2.6-flare"
    res = await b.generate(_req())
    (kw,) = images.calls
    assert kw["model"] == "gpt-image-2.6-flare"
    assert kw["prompt"] == "PROMPT AS GIVEN"  # passed through untouched
    assert kw["size"] == "1536x1024" and kw["n"] == 2 and kw["quality"] == "high"
    assert kw["output_format"] == "png" and kw["extra_body"] == {"complexity": "low"}
    assert "input_fidelity" not in kw
    names = [i[0] for i in kw["image"]]
    assert names == ["guide_grey.png", "reference_1.jpg", "reference_2.jpg"]  # guide first
    assert decode_png(kw["image"][0][1]).shape == (1024, 1536, 3)  # exactly the output size
    for _, data, ctype in kw["image"][1:]:
        h, w = decode_png(data).shape[:2]
        assert ctype == "image/jpeg" and abs(w / h - 1.5) < 0.01 and max(w, h) <= 1536
    # the all-black output is rejected, the other kept
    assert len(res.images) == 1 and res.meta["rejected"] == [{"index": 1, "reason": "black"}]
    assert res.backend == "openai" and res.meta["cost_source"] == "usage"
    assert res.cost_usd == pytest.approx((100 * 5 + 4600 * 8 + 2706 * 30) / 1e6, rel=1e-3)


def test_estimate_cost_matches_list_prices():
    b = OpenAIImageBackend(client=_client(FakeImages()))
    one = b.estimate_cost(_req(n=1, prompt="x" * 2400))
    assert one == pytest.approx(0.0809, abs=0.002)  # per-candidate figure from the list prices


async def test_moderation_is_content_policy_and_not_retryable():
    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/images/edits"))
    exc = openai.BadRequestError("blocked", response=resp, body={"code": "moderation_blocked", "message": "blocked"})
    b = OpenAIImageBackend(client=_client(FakeImages(exc=exc)))
    with pytest.raises(ContentPolicyError) as ei:
        await b.generate(_req())
    assert ei.value.retryable is False


async def test_server_error_is_retryable():
    resp = httpx.Response(503, request=httpx.Request("POST", "https://api.openai.com/v1/images/edits"))
    exc = openai.InternalServerError("down", response=resp, body=None)
    b = OpenAIImageBackend(client=_client(FakeImages(exc=exc)))
    with pytest.raises(ImageGenError) as ei:
        await b.generate(_req())
    assert ei.value.retryable is True


async def test_missing_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    b = get_backend("openai", None)
    with pytest.raises(MissingCredentialsError):
        b.client
    h = await b.health()
    assert h["ok"] is False and "OPENAI_API_KEY" in h["detail"]


async def test_per_request_quality_reaches_call_and_cost():
    out = [SimpleNamespace(b64_json=base64.b64encode(_png(1024, 1536, 5)).decode())]
    images = FakeImages(data=out)
    b = OpenAIImageBackend(client=_client(images), quality="high")
    res = await b.generate(_req(n=1, quality="low"))
    (kw,) = images.calls
    assert kw["quality"] == "low" and res.meta["quality"] == "low"
    low, high = b.estimate_cost(_req(n=1, quality="low")), b.estimate_cost(_req(n=1))
    assert high - low == pytest.approx(0.041 - 0.020, abs=1e-4)  # per-image output price by quality
