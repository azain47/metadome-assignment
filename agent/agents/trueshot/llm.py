"""LLM endpoint, switchable by env (brief: OpenAI by default; Ollama, Docker Model Runner or
another OpenAI-compatible API also work).

- LLM_API_BASE empty  -> OpenAI Responses API, `LLM_MODEL`, sequential tool calls
  (parallel calls would race on the single Blender scene).
- LLM_API_BASE set    -> Chat Completions against that base URL (Ollama, DMR, vLLM, ...).
"""

from google.adk.models.base_llm import BaseLlm

from app.config import settings


def make_model() -> BaseLlm:
    from google.adk.integrations.openai import OpenAILlm, OpenAIResponsesLlm

    if settings.llm_api_base:
        return OpenAILlm(
            model=settings.llm_model,
            base_url=settings.llm_api_base,
            api_key=settings.openai_api_key or "not-needed",
            max_tokens=8192,
        )
    return OpenAIResponsesLlm(model=settings.llm_model, parallel_tool_calls=False)


