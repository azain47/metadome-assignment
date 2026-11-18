"""Trueshot additions to the official Blender MCP server.

- `call()`: run a tool-code string in Blender from an `async def` tool without blocking
  the MCP server's event loop (the official tools are sync and block it).
- Blender host discovery: try BLENDER_HOST, then BLENDER_HOST_CANDIDATES
  (`file:<path>` entries are read from a file written by the compose `hostip` helper).
- `run_job()`: start/poll pattern for long Blender work (renders), so no single socket
  call comes near the official 300 s timeout.
- S3 presigning for the Windows-side Blender, which reaches the object store at
  S3_PUBLIC_ENDPOINT (default http://127.0.0.1:8333).
- `pace()`: a short server-side pause after visible steps so judges can follow the viewport.
"""

__all__ = (
    "ToolCode",
    "ToolCall",
    "call",
    "run_job",
    "pace",
    "presign_get",
    "presign_put",
    "s3_head",
    "blender_host",
)

import asyncio
import hashlib
import os
import socket
import time
from collections.abc import Callable
from typing import Any

from blmcp.tools_helpers import connection, toolcode_load_from_filepath

_PORT = int(os.environ.get("BLENDER_PORT", "9876"))
_host_cache: str | None = None


# ---------------------------------------------------------------------------
# Blender connection

def _candidates() -> list[str]:
    hosts: list[str] = []
    if explicit := os.environ.get("BLENDER_HOST", "").strip():
        hosts.append(explicit)
    raw = os.environ.get("BLENDER_HOST_CANDIDATES", "file:/run/hostip/gw,host.docker.internal,localhost")
    for item in (s.strip() for s in raw.split(",")):
        if not item:
            continue
        if item.startswith("file:"):
            try:
                with open(item[5:], encoding="utf-8") as fh:
                    value = fh.read().strip()
            except OSError:
                value = ""
            if value:
                hosts.append(value)
        else:
            hosts.append(item)
    return list(dict.fromkeys(hosts))


def _probe() -> str | None:
    for host in _candidates():
        try:
            with socket.create_connection((host, _PORT), timeout=1.0):
                return host
        except OSError:
            continue
    return None


def blender_host() -> str:
    """Return the reachable Blender host, probing the candidates on first use."""
    global _host_cache
    if _host_cache is None:
        _host_cache = _probe()
        if _host_cache is None:
            raise ConnectionError(
                f"Blender is not reachable on port {_PORT}; tried {_candidates()}. "
                "Is Blender open with the MCP add-on server running?"
            )
    return _host_cache


def _send(code: str) -> dict[str, Any]:
    global _host_cache
    host = blender_host()
    os.environ["BLENDER_MCP_HOST"] = host
    os.environ["BLENDER_MCP_PORT"] = str(_PORT)
    try:
        return connection.send_code(code, True)
    except ConnectionError:
        # Re-probe only when another candidate answers now (e.g. the WSL address changed).
        # Never resend after a timeout on the same host: the code may already have run.
        _host_cache = None
        fresh = _probe()
        if fresh is None or fresh == host:
            raise
        _host_cache = fresh
        os.environ["BLENDER_MCP_HOST"] = fresh
        return connection.send_code(code, True)


class ToolCode:
    """A tool's `*_toolcode.py` (templates expanded), run in Blender through a cached copy.

    The add-on reads requests 4 KiB per timer tick, so sending 15-35 KB of tool code on every
    call costs seconds. Instead the source is installed once per Blender session into
    `bpy.app.driver_namespace["trueshot_code"]` (compiled, keyed by a hash of the source), and
    each call sends a ~0.5 KB stub with the `Params` repr. A Blender restart or a changed
    source simply misses the cache and installs again.
    """

    def __init__(self, tool_module_file: str) -> None:
        self.name = os.path.splitext(os.path.basename(tool_module_file))[0]
        self.source = toolcode_load_from_filepath(tool_module_file)
        self.digest = hashlib.sha1(self.source.encode("utf-8")).hexdigest()[:16]

    def format(self, params: object) -> "ToolCall":
        return ToolCall(self, repr(params))

    def install_code(self) -> str:
        return (
            "import bpy\n"
            f"bpy.app.driver_namespace.setdefault('trueshot_code', {{}})[{self.digest!r}] = "
            f"compile({self.source!r}, {'<ts:' + self.name + '>'!r}, 'exec')\n"
            "result = {'status': 'ok'}\n"
        )


_NEEDS_INSTALL = "__ts_install__"


class ToolCall:
    """One invocation of a cached tool: `call()` sends `stub()`, installing the code on a miss."""

    def __init__(self, tool: ToolCode, params_repr: str) -> None:
        self.tool = tool
        self.params_repr = params_repr

    def stub(self) -> str:
        # Same calling convention as the official footer: a NamedTuple result becomes a dict;
        # a callable becomes the add-on's deferred `check_is_finished`.
        return (
            "import bpy\n"
            f"_ts_code = bpy.app.driver_namespace.get('trueshot_code', {{}}).get({self.tool.digest!r})\n"
            "if _ts_code is None:\n"
            f"    result = {{'status': {_NEEDS_INSTALL!r}}}\n"
            "else:\n"
            f"    _ts_ns = {{'__name__': {'ts_' + self.tool.name!r}}}\n"
            "    exec(_ts_code, _ts_ns)\n"
            f"    _ts_rv = _ts_ns['main'](eval({self.params_repr!r}, _ts_ns))\n"
            "    if callable(_ts_rv):\n"
            "        check_is_finished = _ts_rv\n"
            "        result = {}\n"
            "    else:\n"
            "        result = _ts_rv._asdict() if hasattr(_ts_rv, '_asdict') else _ts_rv\n"
        )


async def _call_text(code: str) -> dict[str, Any]:
    try:
        response = await asyncio.to_thread(_send, code)
    except ConnectionError as ex:
        return {"status": "error", "message": str(ex)}
    if response.get("status") != "ok":
        return {
            "status": "error",
            "message": str(response.get("message", "unknown error"))[-4000:],
            "stderr": str(response.get("stderr", ""))[-2000:],
        }
    result = response.get("result")
    return result if isinstance(result, dict) else {"status": "ok", "result": result}


async def call(code: "str | ToolCall") -> dict[str, Any]:
    """Run tool code in Blender; return the tool's result dict.

    Pass `TOOL.format(params)` (cached path) or a raw code string that sets `result`.
    Add-on level failures (exceptions in tool code, unreachable Blender) come back as
    {"status": "error", "message": ...} so the agent always gets a dict with a status.
    """
    if not isinstance(code, ToolCall):
        return await _call_text(code)
    result = await _call_text(code.stub())
    if result.get("status") == _NEEDS_INSTALL:
        installed = await _call_text(code.tool.install_code())
        if installed.get("status") != "ok":
            return installed
        result = await _call_text(code.stub())
    return result


async def run_job(
    make_code: "Callable[[str], str | ToolCall]",
    *,
    poll_s: float = 1.0,
    timeout_s: float = float(os.environ.get("TS_JOB_TIMEOUT_S", "1500")),
) -> dict[str, Any]:
    """Start/poll protocol for long Blender work.

    `make_code(action)` returns tool code for action "start" or "status".
    The tool code returns {"status": "started"|"running"} while work continues, and its
    final result dict (status "ok"/"error"/"busy") when done.
    """
    started = time.monotonic()
    result = await call(make_code("start"))
    while result.get("status") in ("started", "running"):
        if time.monotonic() - started > timeout_s:
            return {"status": "error", "message": f"Blender job exceeded {timeout_s:.0f} s"}
        await asyncio.sleep(poll_s)
        result = await call(make_code("status"))
    result.setdefault("duration_s", round(time.monotonic() - started, 2))
    return result


async def pace() -> None:
    """Pause after a visible step (server side, so Blender keeps redrawing)."""
    await asyncio.sleep(float(os.environ.get("TS_STEP_PAUSE_S", "1.0")))


# ---------------------------------------------------------------------------
# Object storage (SeaweedFS S3)

_s3_clients: dict[str, Any] = {}


def _s3(endpoint_env: str, default: str) -> Any:
    endpoint = os.environ.get(endpoint_env, default)
    if endpoint not in _s3_clients:
        import boto3
        from botocore.config import Config

        _s3_clients[endpoint] = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
            aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
            region_name="us-east-1",
            config=Config(
                signature_version="s3v4",
                s3={"addressing_style": "path"},
                request_checksum_calculation="when_required",
                response_checksum_validation="when_required",
            ),
        )
    return _s3_clients[endpoint]


def _bucket() -> str:
    return os.environ.get("S3_BUCKET", "trueshot")


def presign_get(key: str, expires_s: int = 3600) -> str:
    """URL the Windows-side Blender can GET (signed for S3_PUBLIC_ENDPOINT)."""
    client = _s3("S3_PUBLIC_ENDPOINT", "http://127.0.0.1:8333")
    return client.generate_presigned_url(
        "get_object", Params={"Bucket": _bucket(), "Key": key}, ExpiresIn=expires_s)


def presign_put(key: str, content_type: str, expires_s: int = 3600) -> str:
    """URL the Windows-side Blender can PUT to; send the same Content-Type header."""
    client = _s3("S3_PUBLIC_ENDPOINT", "http://127.0.0.1:8333")
    return client.generate_presigned_url(
        "put_object",
        Params={"Bucket": _bucket(), "Key": key, "ContentType": content_type},
        ExpiresIn=expires_s,
    )


def s3_head(key: str) -> dict[str, Any] | None:
    """{"etag", "size"} for an object (via the in-network endpoint), or None if missing."""
    client = _s3("S3_ENDPOINT", "http://s3:8333")
    try:
        head = client.head_object(Bucket=_bucket(), Key=key)
    except client.exceptions.ClientError:
        return None
    return {"etag": str(head.get("ETag", "")).strip('"'), "size": int(head.get("ContentLength", 0))}


# ---------------------------------------------------------------------------
# Render tools (start/poll jobs that upload their outputs)

def job_key(job_prefix: str, filename: str) -> str:
    """S3 key of a per-attempt output: `jobs/<job_id>/a<n>/` + file name."""
    return job_prefix.rstrip("/") + "/" + filename


def presign_outputs(
    job_prefix: str, outputs: dict[str, tuple[str, str]],
) -> tuple[dict[str, str], tuple[tuple[str, str, str], ...]]:
    """`outputs` = {name: (file_name, content_type)} -> (keys {name: s3_key},
    uploads ((name, presigned_put_url, content_type), ...)) for the tool code's Params."""
    keys = {name: job_key(job_prefix, fname) for name, (fname, _ct) in outputs.items()}
    uploads = tuple((name, presign_put(keys[name], ct), ct) for name, (_f, ct) in outputs.items())
    return keys, uploads


async def run_render_job(
    tool: ToolCode, params_type: type, job_prefix: str, outputs: dict[str, tuple[str, str]], **fields: Any,
) -> dict[str, Any]:
    """Presign `outputs`, run the tool code's start/status chain, add `keys` on success."""
    keys, uploads = presign_outputs(job_prefix, outputs)
    result = await run_job(lambda action: tool.format(params_type(action=action, uploads=uploads, **fields)))
    if result.get("status") == "ok":
        result["keys"] = keys
    return result
