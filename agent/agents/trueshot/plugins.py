"""ADK plugins: cross-cutting behaviour applied to every node, registered in agent.py.

- ToolLogPlugin:   every LLM-initiated tool call -> tool log (graph steps log through steps.Step)
- LlmBudgetPlugin: spend meter for every LLM call
- ReasoningEffortPlugin: the user's `thinking` setting on every LLM call
- RunFailurePlugin: a run that ends in an exception is marked failed, with the reason in the chat
"""

import time
from typing import Any

from google.adk.agents.callback_context import CallbackContext
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.integrations.openai import OpenAIGenerateContentConfig
from google.adk.integrations.openai._openai_common import supported_efforts
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins.base_plugin import BasePlugin
from google.adk.tools.base_tool import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from app.blender import payload_from_dump
from app.config import settings
from app.services import services

from . import settings as user_settings
from . import state


class ToolLogPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="tool_log")
        self._started: dict[str, float] = {}

    async def before_tool_callback(self, *, tool: BaseTool, tool_args: dict[str, Any],
                                   tool_context: ToolContext) -> dict | None:
        self._started[tool_context.function_call_id or ""] = time.monotonic()
        return None

    async def after_tool_callback(self, *, tool: BaseTool, tool_args: dict[str, Any],
                                  tool_context: ToolContext, result: dict) -> dict | None:
        started = self._started.pop(tool_context.function_call_id or "", None)
        duration_ms = int((time.monotonic() - started) * 1000) if started else None
        payload = payload_from_dump(result)
        await services().db.log_tool(
            tool=tool.name, source="llm", status=str(payload.get("status", "ok")), args=tool_args,
            result=payload, duration_ms=duration_ms, session_id=tool_context.session.id,
            invocation_id=tool_context.invocation_id, node=tool_context.agent_name)
        return None


class LlmBudgetPlugin(BasePlugin):
    def __init__(self) -> None:
        super().__init__(name="llm_budget")

    async def after_model_callback(self, *, callback_context: CallbackContext,
                                   llm_response: LlmResponse) -> LlmResponse | None:
        usage = llm_response.usage_metadata
        if usage is None:
            return None
        prompt = usage.prompt_token_count or 0
        cached = usage.cached_content_token_count or 0
        output = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        usd = ((prompt - cached) * settings.llm_price_in + cached * settings.llm_price_cached
               + output * settings.llm_price_out) / 1e6
        state.add_spend(callback_context.state, cap_usd=settings.spend_cap_usd, llm_usd=usd)
        return None


class ReasoningEffortPlugin(BasePlugin):
    """Swaps the request config for an OpenAIGenerateContentConfig carrying the user's effort.

    No setting -> the request is untouched (model default). On api.openai.com (no LLM_API_BASE)
    ADK validates the tier against the model and raises for models without one, so an effort the
    model does not accept is skipped instead; compatible backends get it passed through."""

    def __init__(self) -> None:
        super().__init__(name="reasoning_effort")

    async def before_model_callback(self, *, callback_context: CallbackContext,
                                    llm_request: LlmRequest) -> LlmResponse | None:
        effort = user_settings.reasoning_effort(callback_context.state)
        if effort is None:
            return None
        if not settings.llm_api_base and effort not in supported_efforts(
                llm_request.model or settings.llm_model, "responses"):
            return None
        llm_request.config = with_effort(llm_request.config, effort)
        return None


def with_effort(config: Any, effort: str) -> OpenAIGenerateContentConfig:
    """`config` (any GenerateContentConfig or None) as an OpenAIGenerateContentConfig with `effort`.
    Fields are copied as objects (not dumped) so a pydantic `response_schema` class survives;
    `thinking_config` is dropped because the OpenAI config rejects it (and ADK ignores it)."""
    if isinstance(config, OpenAIGenerateContentConfig):
        return config.model_copy(update={"effort": effort})
    fields = {name: getattr(config, name) for name in type(config).model_fields
              if getattr(config, name) is not None and name != "thinking_config"} if config else {}
    return OpenAIGenerateContentConfig(**fields, effort=effort)


class RunFailurePlugin(BasePlugin):
    """An exception that ends a run (e.g. a Blender tool rejecting its arguments) would leave
    `run.status` at "running", which reads as an interrupted run. Record it as failed, with the
    reason, and say it in the chat."""

    def __init__(self) -> None:
        super().__init__(name="run_failure")

    async def on_run_error_callback(self, *, invocation_context: InvocationContext, error: Exception) -> None:
        session = invocation_context.session
        reason = f"{type(error).__name__}: {error}"[:400]
        run = {**(session.state.get(state.RUN) or {}), "status": "failed", "error": reason}
        event = Event(author="trueshot", invocation_id=invocation_context.invocation_id,
                      content=types.Content(role="model", parts=[types.Part(text=f"Run failed: {reason}")]),
                      actions=EventActions(state_delta={state.RUN: run}))
        await invocation_context.session_service.append_event(session, event)


def all_plugins() -> list[BasePlugin]:
    return [ToolLogPlugin(), LlmBudgetPlugin(), ReasoningEffortPlugin(), RunFailurePlugin()]
