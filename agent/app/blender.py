"""Calls to the Blender MCP server from graph code (function nodes, health checks).

Every Blender step of the pipeline is a plain-code graph node calling the MCP server through
this client, so each call shows up in the tool log and the UI feed.

Interface: `await BlenderClient.call(tool, **args) -> dict` returns the tool's result dict when
its `status` is "ok"; raises `BlenderToolError` (with `.status` "error" | "busy" | "transport")
otherwise. `health()` never raises. `payload_from_dump()` reads the same payload out of the
CallToolResult dict that ADK's McpTool hands to callbacks.
"""

import json
import logging
from datetime import timedelta
from typing import Any, Protocol

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

log = logging.getLogger(__name__)


class BlenderToolError(RuntimeError):
    def __init__(self, tool: str, status: str, message: str, result: dict[str, Any] | None = None) -> None:
        super().__init__(f"{tool}: {status}: {message}")
        self.tool = tool
        self.status = status
        self.message = message
        self.result = result or {}


class Blender(Protocol):
    async def call(self, tool: str, **args: Any) -> dict[str, Any]: ...
    async def health(self) -> dict[str, Any]: ...


def payload_from_dump(dumped: Any) -> dict[str, Any]:
    """Tool payload (a dict with `status`) from a dumped CallToolResult dict."""
    if not isinstance(dumped, dict):
        return {"status": "ok", "result": dumped}
    if "structuredContent" not in dumped and "content" not in dumped:
        return dumped  # already a plain tool payload
    payload = dumped.get("structuredContent")
    if isinstance(payload, dict):
        # FastMCP wraps non-object returns as {"result": ...}
        if set(payload) == {"result"} and isinstance(payload["result"], dict):
            payload = payload["result"]
    else:
        text = "".join(str(b.get("text", "")) for b in dumped.get("content") or [] if isinstance(b, dict))
        try:
            payload = json.loads(text) if text else {}
        except json.JSONDecodeError:
            payload = {"message": text[:2000]}
        if not isinstance(payload, dict):
            payload = {"result": payload}
    payload = dict(payload)
    payload.setdefault("status", "error" if dumped.get("isError") else "ok")
    return payload


def _root_cause(ex: BaseException) -> BaseException:
    """anyio task groups wrap the real failure (e.g. ConnectError) in ExceptionGroups."""
    while isinstance(ex, BaseExceptionGroup) and ex.exceptions:
        ex = ex.exceptions[0]
    return ex


class BlenderClient:
    """MCP adapter. One short-lived MCP session per call: the server is stateless HTTP."""

    def __init__(self, url: str, *, connect_timeout_s: float = 10.0, call_timeout_s: float = 1800.0) -> None:
        self.url = url
        self._connect_timeout_s = connect_timeout_s
        self._call_timeout_s = call_timeout_s

    async def _call_raw(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        timeout = httpx.Timeout(self._connect_timeout_s, read=self._call_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as http:
            async with streamable_http_client(self.url, http_client=http) as (read, write, _sid):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(
                        tool, args, read_timeout_seconds=timedelta(seconds=self._call_timeout_s))
                    return payload_from_dump(result.model_dump(mode="json"))

    async def call(self, tool: str, **args: Any) -> dict[str, Any]:
        try:
            payload = await self._call_raw(tool, args)
        except Exception as ex:  # noqa: BLE001 - every transport failure maps to one error type
            cause = _root_cause(ex)
            raise BlenderToolError(tool, "transport", f"{type(cause).__name__}: {cause}") from ex
        status = str(payload.get("status", "ok"))
        if status != "ok":
            raise BlenderToolError(tool, status, str(payload.get("message", "")), payload)
        return payload

    async def health(self) -> dict[str, Any]:
        """Reachability of the MCP server and, through it, of Blender itself."""
        try:
            info = await self.call("scene_info")
            return {"ok": True, "detail": f"Blender reachable; rendering={info.get('is_rendering', False)}"}
        except BlenderToolError as ex:
            if ex.status == "busy":
                return {"ok": True, "detail": "Blender busy rendering"}
            return {"ok": False, "detail": ex.message[:200]}
