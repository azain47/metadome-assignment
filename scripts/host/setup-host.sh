#!/usr/bin/env bash
# One-time host setup for running the stack with Blender on Windows and Docker in WSL2.
#
#   scripts/host/setup-host.sh                # install the MCP add-on + startup script into Blender 5.3
#   scripts/host/setup-host.sh --wslconfig    # also stop WSL from shutting Docker down when terminals close
#
# What it changes:
# - %APPDATA%\Blender Foundation\Blender\<ver>\scripts\addons\blender_mcp_addon  (official Blender Lab add-on)
# - %APPDATA%\Blender Foundation\Blender\<ver>\scripts\startup\trueshot_blender_startup.py
#   (enables the add-on, binds it to the WSL adapter address, rebinds when WSL restarts)
# - with --wslconfig: %USERPROFILE%\.wslconfig gets [general] instanceIdleTimeout=-1 (takes effect after `wsl --shutdown`)
# Nothing else in your Blender preferences is touched.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BLENDER_VERSION="${BLENDER_VERSION:-5.3}"
win() { cmd.exe /c "$1" 2>/dev/null | tr -d '\r'; }

APPDATA_WIN="$(win 'echo %APPDATA%')"
USERPROFILE_WIN="$(win 'echo %USERPROFILE%')"
[[ -n "$APPDATA_WIN" ]] || { echo "Not running under WSL with Windows interop; install the add-on manually (README)." >&2; exit 1; }

BLENDER_USER="$(wslpath "$APPDATA_WIN")/Blender Foundation/Blender/$BLENDER_VERSION"
mkdir -p "$BLENDER_USER/scripts/addons" "$BLENDER_USER/scripts/startup"
rm -rf "$BLENDER_USER/scripts/addons/blender_mcp_addon"
cp -r "$ROOT/blender_addon/blender_mcp_addon" "$BLENDER_USER/scripts/addons/"
cp "$ROOT/scripts/host/trueshot_blender_startup.py" "$BLENDER_USER/scripts/startup/"
echo "Installed the MCP add-on and startup script into: $BLENDER_USER/scripts"

if [[ "${1:-}" == "--wslconfig" ]]; then
  WSLCONFIG="$(wslpath "$USERPROFILE_WIN")/.wslconfig"
  touch "$WSLCONFIG"
  if grep -q '^instanceIdleTimeout' "$WSLCONFIG"; then
    sed -i 's/^instanceIdleTimeout.*/instanceIdleTimeout=-1/' "$WSLCONFIG"
  elif grep -q '^\[general\]' "$WSLCONFIG"; then
    sed -i '/^\[general\]/a instanceIdleTimeout=-1' "$WSLCONFIG"
  else
    printf '\n[general]\ninstanceIdleTimeout=-1\n' >>"$WSLCONFIG"
  fi
  echo "Updated $WSLCONFIG. Run 'wsl --shutdown' from PowerShell once for it to take effect (this stops all containers)."
fi

host_ip="$(ip route show default | awk '{print $3}')"
cat <<EOF

Next steps:
1. Start Blender $BLENDER_VERSION. In Preferences > System, turn on "Allow Online Access" (the add-on refuses to start without it).
2. Within a few seconds the Blender console shows: [trueshot] MCP bridge listening on $host_ip:9876
3. From WSL: docker compose up -d   then open http://localhost:8000
EOF
