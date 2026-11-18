import fal_client
import httpx
import numpy as np
import pytest

from app.imagegen import ContentPolicyError, ImageGenError, MissingCredentialsError, PlateRequest
from app.imagegen.fal_backend import ENDPOINTS, FalImageBackend, endpoint_for, fal_model_id
from app.imaging.io import decode_png, encode_png


def _png(h, w, seed=1):
    return encode_png(np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8))


GREY, LINES, DEPTH = _png(1024, 1536, 1), _png(1024, 1536, 2), _png(1024, 1536, 3)


def _req(**kw):
    base = dict(guide_grey=GREY, guide_lines=LINES, guide_depth=DEPTH, references=[_png(900, 1600, 4)],
                prompt="PROMPT AS GIVEN", n=2, seed=11)
    base.update(kw)
    return PlateRequest(**base)


class Handle:
    def __init__(self, client):
        self.client = client
        self.request_id = "req-1"
        self.polls = 0

    async def status(self, with_logs=False):
        self.polls += 1
        if self.polls < 2:
            return fal_client.InProgress(logs=None)
        return fal_client.Completed(logs=None, metrics={"inference_time": 1.0})

    async def get(self):
        return self.client.result

    async def cancel(self):
        pass


class FakeFal:
    def __init__(self, result=None, errors=()):
        self.uploads = []
        self.submits = []
        self.result = result or {"images": [{"url": "https://cdn/out0.png"}, {"url": "https://cdn/out1.png"}], "seed": 11}
        self.errors = list(errors)

    async def upload(self, data, content_type, file_name=None):
        self.uploads.append((file_name, content_type, data))
        return f"https://cdn/{file_name}"

    async def submit(self, application, arguments):
        self.submits.append((application, arguments))
        if self.errors:
            raise self.errors.pop(0)
        return Handle(self)


def _http_error(code, msg="err"):
    resp = httpx.Response(code, request=httpx.Request("POST", "https://queue.fal.run/x"))
    return fal_client.FalClientHTTPError(msg, code, {}, resp)


OUTPUTS = {
    "https://cdn/out0.png": _png(1024, 1536, 5),
    "https://cdn/out1.png": encode_png(np.zeros((1024, 1536, 3), np.uint8)),  # safety-checker black frame
    "https://cdn/big.png": _png(1365, 2048, 6),
}


async def _download(url):
    return OUTPUTS[url]


def _backend(model, fal, **kw):
    b = FalImageBackend(model, client=fal, downloader=_download, poll_interval_s=0, **kw)
    return b


async def test_gpt_image_edit_shape_and_black_rejection():
    fal = FakeFal()
    b = _backend("openai/gpt-image-2.6/flare/edit", fal)
    res = await b.generate(_req())
    (app, args), = fal.submits
    assert app == "openai/gpt-image-2.6/flare/edit"
    assert args == {
        "prompt": "PROMPT AS GIVEN",
        "image_urls": ["https://cdn/guide_grey.png", "https://cdn/reference_1.jpg"],
        "image_size": {"width": 1536, "height": 1024},
        "quality": "high",
        "reasoning_effort": "low",
        "num_images": 2,
        "output_format": "png",
    }
    assert [u[0] for u in fal.uploads] == ["guide_grey.png", "reference_1.jpg"]
    ref = decode_png(fal.uploads[1][2])
    assert abs(ref.shape[1] / ref.shape[0] - 1.5) < 0.01
    assert len(res.images) == 1 and res.meta["rejected"] == [{"index": 1, "reason": "black"}]
    assert res.meta["endpoint_family"] == "openai/gpt-image" and res.meta["request_id"] == "req-1"
    assert res.cost_usd == pytest.approx(2 * (0.041 + 2 * 1536 * 8 / 1e6), rel=1e-3)


@pytest.mark.parametrize("guide,expected", [("depth", DEPTH), ("lines", LINES)])
async def test_zimage_controlnet_shape(guide, expected):
    fal = FakeFal(result={"images": [{"url": "https://cdn/out0.png"}]})
    b = _backend("fal-ai/z-image/turbo/controlnet", fal, control_guide=guide)
    res = await b.generate(_req(n=6))
    (_, args), = fal.submits
    assert args["image_url"] == f"https://cdn/guide_{guide}.png" and "image_urls" not in args
    assert fal.uploads[0][2] == expected and len(fal.uploads) == 1  # no references sent
    assert args["preprocess"] == "none" and args["control_scale"] == 0.7 and args["control_end"] == 0.8
    assert args["control_start"] == 0.0 and args["num_inference_steps"] == 8
    assert args["num_images"] == 4 and args["seed"] == 11 and args["prompt"] == "PROMPT AS GIVEN"
    assert args["image_size"] == {"width": 1536, "height": 1024}
    assert res.cost_usd == pytest.approx(4 * 0.0065 * 1.572864, rel=1e-3)


async def test_nano_banana_aspect_and_resize():
    fal = FakeFal(result={"images": [{"url": "https://cdn/big.png"}]})
    b = _backend("fal-ai/nano-banana-pro/edit", fal)
    res = await b.generate(_req(n=1))
    (_, args), = fal.submits
    assert args["aspect_ratio"] == "3:2" and args["resolution"] == "2K" and "image_size" not in args
    assert decode_png(res.images[0]).shape == (1024, 1536, 3)  # 2K output fitted to the request size
    assert res.cost_usd == pytest.approx(0.15)


async def test_generic_and_klein_shapes():
    fal = FakeFal(result={"images": [{"url": "https://cdn/out0.png"}]})
    await _backend("some-vendor/new-model/edit", fal).generate(_req(n=1, seed=None))
    (_, args), = fal.submits
    assert set(args) == {"prompt", "image_urls", "image_size", "num_images", "output_format"}
    fal2 = FakeFal(result={"images": [{"url": "https://cdn/out0.png"}]})
    await _backend("blackforestlabs/flux-3/klein/edit-image", fal2).generate(_req(n=1))
    assert fal2.submits[0][1]["enable_prompt_expansion"] is False


def test_endpoint_table_and_model_ids():
    assert endpoint_for("fal-ai/z-image/turbo/controlnet/lora")[2] == "fal-ai/z-image/turbo/controlnet/lora"
    assert endpoint_for("unknown/x")[2] == "generic"
    assert all(callable(b) and callable(c) for b, c in ENDPOINTS.values())
    assert fal_model_id("gpt-image-2.6-flare") == "openai/gpt-image-2.6/flare/edit"
    assert fal_model_id(None) == "openai/gpt-image-2.6/flare/edit"
    assert fal_model_id("fal-ai/z-image/turbo/controlnet") == "fal-ai/z-image/turbo/controlnet"


async def test_retries_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr("app.imagegen.fal_backend.asyncio.sleep", _no_sleep)
    fal = FakeFal(errors=[_http_error(503), httpx.ConnectError("boom")])
    res = await _backend("openai/gpt-image-2.6/flare/edit", fal).generate(_req())
    assert len(fal.submits) == 3 and res.meta["attempts"] == 3


async def test_422_content_policy_not_retried(monkeypatch):
    monkeypatch.setattr("app.imagegen.fal_backend.asyncio.sleep", _no_sleep)
    fal = FakeFal(errors=[_http_error(422, "content policy violation")])
    with pytest.raises(ContentPolicyError):
        await _backend("fal-ai/nano-banana-pro/edit", fal).generate(_req())
    assert len(fal.submits) == 1
    fal = FakeFal(errors=[_http_error(400, "bad field")])
    with pytest.raises(ImageGenError) as ei:
        await _backend("fal-ai/nano-banana-pro/edit", fal).generate(_req())
    assert ei.value.retryable is False and len(fal.submits) == 1


async def test_retries_exhausted(monkeypatch):
    monkeypatch.setattr("app.imagegen.fal_backend.asyncio.sleep", _no_sleep)
    fal = FakeFal(errors=[_http_error(500)] * 3)
    with pytest.raises(ImageGenError) as ei:
        await _backend("fal-ai/nano-banana-pro/edit", fal).generate(_req())
    assert ei.value.retryable is True and len(fal.submits) == 3


async def test_nsfw_flag_rejected():
    fal = FakeFal(result={"images": [{"url": "https://cdn/out0.png"}, {"url": "https://cdn/big.png"}],
                          "has_nsfw_concepts": [True, False]})
    res = await _backend("fal-ai/z-image/turbo/controlnet", fal).generate(_req())
    assert len(res.images) == 1 and res.meta["rejected"] == [{"index": 0, "reason": "nsfw_flag"}]


async def test_missing_key(monkeypatch):
    monkeypatch.delenv("FAL_KEY", raising=False)
    b = FalImageBackend("fal-ai/z-image/turbo/controlnet")
    with pytest.raises(MissingCredentialsError):
        b.client
    assert (await b.health())["ok"] is False


async def _no_sleep(*_a, **_k):
    return None
