"""The chat UI's `user:settings` reach the LLM request config, the HDRI pick and the plate request."""

import sys
from pathlib import Path
from types import SimpleNamespace

from google.adk.integrations.openai import OpenAIGenerateContentConfig
from google.adk.integrations.openai._openai_common import build_reasoning_effort
from google.adk.models.llm_request import LlmRequest
from google.genai import types
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "agents"))

from trueshot import plugins  # noqa: E402
from trueshot import settings as user_settings  # noqa: E402

from app.hdri_pick import pick_hdri  # noqa: E402


class Answer(BaseModel):
    text: str


def _state(**values):
    return {user_settings.SETTINGS: values}


def test_readers_accept_only_known_values():
    assert user_settings.image_quality(_state(image_quality="medium")) == "medium"
    assert user_settings.reasoning_effort(_state(reasoning_effort="high")) == "high"
    assert user_settings.image_quality(_state(image_quality="ultra")) is None
    assert user_settings.reasoning_effort({}) is None


def _request(config=None):
    return LlmRequest(model="gpt-5.6-terra", config=config) if config else LlmRequest(model="gpt-5.6-terra")


async def _run(plugin, state, request, monkeypatch, api_base=""):
    monkeypatch.setattr(plugins, "settings", SimpleNamespace(llm_api_base=api_base, llm_model="gpt-5.6-terra"))
    out = await plugin.before_model_callback(callback_context=SimpleNamespace(state=state), llm_request=request)
    assert out is None


async def test_effort_plugin_keeps_schema_and_instruction(monkeypatch):
    config = types.GenerateContentConfig(system_instruction="BE BRIEF", response_schema=Answer,
                                         response_mime_type="application/json", temperature=0.2,
                                         thinking_config=types.ThinkingConfig(include_thoughts=True))
    request = _request(config)
    await _run(plugins.ReasoningEffortPlugin(), _state(reasoning_effort="high"), request, monkeypatch)
    cfg = request.config
    assert isinstance(cfg, OpenAIGenerateContentConfig) and cfg.effort == "high"
    assert cfg.system_instruction == "BE BRIEF" and cfg.response_schema is Answer
    assert cfg.response_mime_type == "application/json" and cfg.temperature == 0.2
    assert build_reasoning_effort(cfg, "gpt-5.6-terra", "responses") == "high"


async def test_effort_plugin_without_setting_leaves_request_alone(monkeypatch):
    config = types.GenerateContentConfig(system_instruction="X")
    request = _request(config)
    await _run(plugins.ReasoningEffortPlugin(), {}, request, monkeypatch)
    assert request.config is config


async def test_effort_plugin_from_empty_config_and_unsupported_model(monkeypatch):
    request = _request()
    await _run(plugins.ReasoningEffortPlugin(), _state(reasoning_effort="low"), request, monkeypatch)
    assert request.config.effort == "low"
    # api.openai.com would reject an effort for a non-reasoning model: skipped, not sent
    plain = LlmRequest(model="gpt-4o", config=types.GenerateContentConfig())
    await _run(plugins.ReasoningEffortPlugin(), _state(reasoning_effort="low"), plain, monkeypatch)
    assert not isinstance(plain.config, OpenAIGenerateContentConfig)
    # compatible backends (LLM_API_BASE set) get it passed through
    local = LlmRequest(model="qwen3", config=types.GenerateContentConfig())
    await _run(plugins.ReasoningEffortPlugin(), _state(reasoning_effort="medium"), local, monkeypatch,
               api_base="http://ollama:11434/v1")
    assert local.config.effort == "medium"


def test_plugin_registered():
    assert any(isinstance(p, plugins.ReasoningEffortPlugin) for p in plugins.all_plugins())


class FakeVlm:
    def __init__(self):
        self.reasoning_effort = "low"
        self.seen = []

    async def parse(self, prompt, images, schema, *, label=""):
        self.seen.append(self.reasoning_effort)
        return schema(number=2, why="matches")


async def test_pick_hdri_uses_requested_effort():
    import numpy as np

    from app.imaging.io import encode_jpeg

    thumb = encode_jpeg(np.full((64, 128, 3), 120, np.uint8))

    async def fetch(url):
        return thumb

    cands = [{"id": "a", "thumbnail": "a"}, {"id": "b", "thumbnail": "b"}]
    vlm = FakeVlm()
    pick = await pick_hdri(cands, "a meadow", "noon", vlm=vlm, fetch_bytes=fetch, reasoning_effort="high")
    assert pick["id"] == "b" and vlm.seen == ["high"]
    assert vlm.reasoning_effort == "low"  # the shared model keeps its own default
    await pick_hdri(cands, "a meadow", "noon", vlm=vlm, fetch_bytes=fetch)
    assert vlm.seen == ["high", "low"]
