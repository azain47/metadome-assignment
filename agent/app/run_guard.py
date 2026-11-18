"""One run per session, and approval answers must match the pending approval.

ADK accepts any /run_sse request for a session. Two of them at once (two browser windows, or an
approval answered twice) make the second resume the invocation while the first is still running:
the workflow restarts from its replay, cancels the live steps and continues from a half-built
scene. This ASGI middleware refuses such requests with 409 and a reason the UI shows:

- a run is already streaming for the session (in this process);
- a RequestInput answer (`adk_request_input` functionResponse) whose id is not the session's
  unanswered pause, i.e. it was already answered elsewhere or belongs to an earlier pause.
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

REQUEST_INPUT = "adk_request_input"
RUN_PATHS = frozenset({"/run_sse", "/run"})

# (app_name, user_id, session_id) -> ids of adk_request_input calls with no answer yet,
# or None when the session does not exist (ADK then reports its own error).
PendingLookup = Callable[[str, str, str], Awaitable["set[str] | None"]]

Scope = dict[str, Any]
Message = dict[str, Any]
Receive = Callable[[], Awaitable[Message]]
Send = Callable[[Message], Awaitable[None]]
App = Callable[[Scope, Receive, Send], Awaitable[None]]


def pending_requests(events: list[Any]) -> set[str]:
    """Ids of `adk_request_input` calls in ADK session events that have no functionResponse."""
    calls: set[str] = set()
    for event in events:
        for part in getattr(getattr(event, "content", None), "parts", None) or []:
            call, resp = getattr(part, "function_call", None), getattr(part, "function_response", None)
            if call is not None and call.name == REQUEST_INPUT and call.id:
                calls.add(call.id)
            if resp is not None and resp.name == REQUEST_INPUT and resp.id:
                calls.discard(resp.id)
    return calls


def _answers(body: dict[str, Any]) -> list[str]:
    message = body.get("newMessage") or body.get("new_message") or {}
    out = []
    for part in message.get("parts") or []:
        resp = part.get("functionResponse") or part.get("function_response") or {}
        if resp.get("name") == REQUEST_INPUT:
            out.append(str(resp.get("id") or ""))
    return out


class RunGuard:
    def __init__(self, app: App, pending: PendingLookup) -> None:
        self.app = app
        self.pending = pending
        self._active: set[str] = set()

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in RUN_PATHS:
            await self.app(scope, receive, send)
            return
        raw = await _read_body(receive)
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            body = {}
        session_id = str(body.get("sessionId") or body.get("session_id") or "")
        if not session_id:
            await self.app(scope, _replay(raw, receive), send)
            return
        if session_id in self._active:
            await _reject(send, "A run is already in progress for this session (another window or tab?). "
                                "Wait for it to finish, then reload.")
            return
        self._active.add(session_id)  # no await between the check and the add: one winner per session
        try:
            answers = _answers(body)
            if answers:
                pending = await self.pending(str(body.get("appName") or body.get("app_name") or ""),
                                             str(body.get("userId") or body.get("user_id") or ""), session_id)
                stale = [a for a in answers if pending is not None and a not in pending]
                if stale:
                    await _reject(send, f"That approval ({stale[0]}) was already answered or is no longer "
                                        "pending. Reload the session to see where the run is.")
                    return
            await self.app(scope, _replay(raw, receive), send)
        finally:
            self._active.discard(session_id)


async def _read_body(receive: Receive) -> bytes:
    chunks = []
    while True:
        message = await receive()
        if message["type"] != "http.request":
            break
        chunks.append(message.get("body", b""))
        if not message.get("more_body"):
            break
    return b"".join(chunks)


def _replay(raw: bytes, receive: Receive) -> Receive:
    sent = False

    async def again() -> Message:
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": raw, "more_body": False}
        return await receive()  # later calls wait for http.disconnect, as ADK expects

    return again


async def _reject(send: Send, detail: str) -> None:
    body = json.dumps({"detail": detail}).encode()
    await send({"type": "http.response.start", "status": 409,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
    await send({"type": "http.response.body", "body": body})
