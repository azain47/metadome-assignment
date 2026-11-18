"""Dev launcher for the MCP server that skips tool modules failing to import.

Other agents edit their tool modules concurrently; a half-written file must not stop the
scene-tool tests. Same arguments as `blender-mcp`:

    cd mcp && .venv/bin/python ../scripts/dev/scene_mcp_server.py --transport http --port 8781
"""

import importlib
import sys
import types

_real_import_module = importlib.import_module


def _tolerant_import_module(name: str, package: str | None = None) -> types.ModuleType:
    try:
        return _real_import_module(name, package)
    except Exception as ex:  # pylint: disable=broad-exception-caught
        if not name.startswith("blmcp.tools."):
            raise
        print(f"[scene_mcp_server] skipping {name}: {type(ex).__name__}: {ex}", file=sys.stderr)
        return types.ModuleType(name)


importlib.import_module = _tolerant_import_module

import blmcp  # noqa: E402  pylint: disable=wrong-import-position

sys.exit(blmcp.main())
