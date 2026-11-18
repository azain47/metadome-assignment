"""Per-user run settings chosen in the chat UI.

The UI sends `user:settings` as the run request's state delta on every message, so a change
applies to the next message without a restart; the `user:` prefix keeps it across sessions.
Missing or invalid values read as None: the caller keeps its own default.
"""

from typing import Any, Mapping

SETTINGS = "user:settings"
QUALITIES = ("low", "medium", "high")
EFFORTS = ("low", "medium", "high")


def _get(state: Mapping[str, Any], key: str, allowed: tuple[str, ...]) -> str | None:
    value = (state.get(SETTINGS) or {}).get(key)
    return value if value in allowed else None


def image_quality(state: Mapping[str, Any]) -> str | None:
    return _get(state, "image_quality", QUALITIES)


def reasoning_effort(state: Mapping[str, Any]) -> str | None:
    return _get(state, "reasoning_effort", EFFORTS)
