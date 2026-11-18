"""Two windows on one session: a second run or a stale approval answer is refused, not replayed."""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.adk.events import Event  # noqa: E402
from google.genai import types  # noqa: E402

from app.run_guard import REQUEST_INPUT, RunGuard, pending_requests  # noqa: E402


class Inner:
    """Stands in for ADK: records each body it receives and streams until released."""

    def __init__(self) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.release = asyncio.Event()

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        message = await receive()
        self.bodies.append(json.loads(message["body"]))
        await self.release.wait()
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"data: {}\n\n"})


async def _post(app: Any, body: dict[str, Any], path: str = "/run_sse") -> int:
    raw = json.dumps(body).encode()
    sent: list[dict] = []

    async def receive() -> dict:
        return {"type": "http.request", "body": raw, "more_body": False}

    async def send(message: dict) -> None:
        sent.append(message)

    await app({"type": "http", "method": "POST", "path": path}, receive, send)
    return sent[0]["status"]


def _run(session: str, **extra: Any) -> dict[str, Any]:
    return {"appName": "trueshot", "userId": "demo", "sessionId": session, **extra}


def _answer(session: str, interrupt_id: str, decision: str) -> dict[str, Any]:
    response = {"id": interrupt_id, "name": REQUEST_INPUT, "response": {"decision": decision, "note": ""}}
    return _run(session, newMessage={"role": "user", "parts": [{"functionResponse": response}]})


async def _pending(ids: set[str]) -> Any:
    async def lookup(app_name: str, user_id: str, session_id: str) -> set[str]:
        return ids
    return lookup


async def test_a_second_run_on_a_busy_session_is_refused_until_the_first_ends() -> None:
    inner = Inner()
    guard = RunGuard(inner, await _pending(set()))
    first = asyncio.create_task(_post(guard, _run("s1", invocationId="inv")))
    await asyncio.sleep(0.01)
    assert await _post(guard, _run("s1", invocationId="inv")) == 409
    other = asyncio.create_task(_post(guard, _run("s2", invocationId="inv2")))  # other sessions run
    await asyncio.sleep(0.01)
    inner.release.set()
    assert await first == 200 and await other == 200
    assert await _post(guard, _run("s1", invocationId="inv")) == 200
    assert [b["sessionId"] for b in inner.bodies] == ["s1", "s2", "s1"]


async def test_an_answer_to_an_approval_that_is_no_longer_pending_is_refused() -> None:
    inner = Inner()
    inner.release.set()
    guard = RunGuard(inner, await _pending({"approve_ladakh_a1"}))
    assert await _post(guard, _answer("s1", "approve_alps_a1", "revise")) == 409
    assert inner.bodies == []
    body = _answer("s1", "approve_ladakh_a1", "approve")
    assert await _post(guard, body) == 200
    assert inner.bodies == [body]  # ADK receives the request body unchanged


async def test_other_routes_and_plain_messages_pass_through() -> None:
    inner = Inner()
    inner.release.set()
    guard = RunGuard(inner, await _pending(set()))
    text = _run("s1", newMessage={"role": "user", "parts": [{"text": "hi"}]})
    assert await _post(guard, text) == 200
    assert await _post(guard, text, path="/apps/trueshot/users/demo/sessions") == 200


def test_pending_requests_are_calls_without_a_response() -> None:
    def call(i: str) -> Event:
        part = types.Part(function_call=types.FunctionCall(id=i, name=REQUEST_INPUT, args={}))
        return Event(author="trueshot", content=types.Content(role="model", parts=[part]))

    def answer(i: str) -> Event:
        part = types.Part(function_response=types.FunctionResponse(id=i, name=REQUEST_INPUT, response={}))
        return Event(author="user", content=types.Content(role="user", parts=[part]))

    assert pending_requests([call("a"), answer("a"), call("b"), Event(author="x")]) == {"b"}
