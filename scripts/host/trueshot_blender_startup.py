"""Blender startup script for the Trueshot stack (runs inside Blender on the host).

Install: copy into Blender's user `scripts/startup/` folder (scripts/host/setup-host.sh does it),
or pass it with `blender --python trueshot_blender_startup.py`.

What it does, once the UI is up and then every 15 s:
- enables the official Blender MCP add-on (`blender_mcp_addon`);
- binds its socket server to the address the Docker containers can reach:
  TRUESHOT_BLENDER_BIND if set, else the WSL vEthernet IPv4 on Windows (found with `ipconfig`),
  else "localhost" (Docker Desktop and native Linux reach that through host.docker.internal);
- uses port TRUESHOT_BLENDER_PORT (default 9876);
- restarts the server when the WSL address changes (it changes whenever WSL restarts);
- keeps final renders out of the judges' viewport (render_display_type = 'NONE').
"""

import os
import re
import subprocess
import sys

import bpy

_ADDON = "blender_mcp_addon"
_CHECK_INTERVAL_S = 15.0
_state = {"bind": None}


def _log(msg: str) -> None:
    print(f"[trueshot] {msg}", flush=True)


def _wsl_vethernet_ipv4() -> str | None:
    if sys.platform != "win32":
        return None
    try:
        out = subprocess.run(
            ["ipconfig"], capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    section = None
    for line in out.splitlines():
        if line and not line.startswith(" "):
            section = line
            continue
        if section and "vEthernet (WSL" in section:
            m = re.search(r"IPv4 Address[ .]*:\s*([\d.]+)", line)
            if m:
                return m.group(1)
    return None


def _bind_address() -> str:
    return os.environ.get("TRUESHOT_BLENDER_BIND") or _wsl_vethernet_ipv4() or "localhost"


def _addon_module() -> str | None:
    for name in bpy.context.preferences.addons.keys():
        if name == _ADDON or name.endswith("." + _ADDON):
            return name
    try:
        bpy.ops.preferences.addon_enable(module=_ADDON)
    except Exception as ex:  # noqa: BLE001
        _log(f"could not enable {_ADDON}: {ex}")
        return None
    return _ADDON if _ADDON in bpy.context.preferences.addons else None


def _server_module(addon_module: str):
    return sys.modules.get(addon_module + ".mcp_to_blender_server")


def _sync() -> float:
    module = _addon_module()
    if module is None:
        return _CHECK_INTERVAL_S
    prefs = bpy.context.preferences
    prefs.view.render_display_type = "NONE"
    addon_prefs = prefs.addons[module].preferences
    bind = _bind_address()
    port = int(os.environ.get("TRUESHOT_BLENDER_PORT", addon_prefs.port))
    server = _server_module(module)
    running = bool(server and server.is_running())
    if running and addon_prefs.host == bind and addon_prefs.port == port:
        return _CHECK_INTERVAL_S
    if not bpy.app.online_access:
        _log("Online Access is off; enable it in Preferences > System (or start with --online-mode)")
        return _CHECK_INTERVAL_S
    addon_prefs.host = bind
    addon_prefs.port = port
    try:
        if running:
            bpy.ops.blmcp.server_stop()
        bpy.ops.blmcp.server_start()
        _log(f"MCP bridge listening on {bind}:{port}")
    except Exception as ex:  # noqa: BLE001
        _log(f"MCP bridge start on {bind}:{port} failed: {ex}")
    return _CHECK_INTERVAL_S


def register() -> None:
    if bpy.app.background:
        return
    if not bpy.app.timers.is_registered(_sync):
        bpy.app.timers.register(_sync, first_interval=2.0, persistent=True)


def unregister() -> None:
    if bpy.app.timers.is_registered(_sync):
        bpy.app.timers.unregister(_sync)


if __name__ == "__main__":
    register()
