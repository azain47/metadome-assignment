#!/usr/bin/env bash
# Launch a throwaway Blender GUI instance on the Windows host for development tests.
# It uses its own user profile (your real add-ons and preferences are untouched),
# enables the official MCP add-on and binds it to the WSL vEthernet address on PORT.
#
#   scripts/dev/blender_instance.sh start <name> <port>   # prints the Blender host:port when ready
#   scripts/dev/blender_instance.sh stop <name>
#
# Containers and WSL reach it at $(ip route show default | awk '{print $3}'):<port>.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
BLENDER_EXE="${BLENDER_EXE:-/mnt/c/Program Files/Blender Foundation/Blender 5.3/blender.exe}"
WIN_TEMP="$(wslpath "$(cmd.exe /c 'echo %TEMP%' 2>/dev/null | tr -d '\r')")"
cmd="${1:?start|stop}"
name="${2:?instance name}"
PROFILE="$WIN_TEMP/trueshot_blender/$name"
LOG="/tmp/trueshot_blender_$name.log"

case "$cmd" in
start)
  port="${3:?port}"
  rm -rf "$PROFILE"
  mkdir -p "$PROFILE/scripts/addons" "$PROFILE/config"
  cp -r "$ROOT/blender_addon/blender_mcp_addon" "$PROFILE/scripts/addons/"
  cp "$ROOT/scripts/host/trueshot_blender_startup.py" "$PROFILE/"
  export BLENDER_USER_SCRIPTS="$PROFILE/scripts" BLENDER_USER_CONFIG="$PROFILE/config"
  export TRUESHOT_BLENDER_PORT="$port"
  export WSLENV="BLENDER_USER_SCRIPTS/p:BLENDER_USER_CONFIG/p:TRUESHOT_BLENDER_PORT${WSLENV:+:$WSLENV}"
  # The tag in --python-expr lets `stop` find this exact Windows process.
  nohup "$BLENDER_EXE" --online-mode --python-expr "# trueshot:$name" \
    --python "$(wslpath -w "$PROFILE/trueshot_blender_startup.py")" >"$LOG" 2>&1 &
  host="$(ip route show default | awk '{print $3}')"
  for _ in $(seq 1 60); do
    if (exec 3<>"/dev/tcp/$host/$port") 2>/dev/null; then
      echo "$host:$port"
      exit 0
    fi
    sleep 1
  done
  echo "Blender did not open $host:$port within 60 s; log: $LOG" >&2
  tail -20 "$LOG" >&2
  exit 1
  ;;
stop)
  powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='blender.exe'\" | Where-Object { \$_.CommandLine -like '*trueshot:$name*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" >/dev/null 2>&1 || true
  ;;
*)
  echo "usage: $0 start <name> <port> | stop <name>" >&2
  exit 2
  ;;
esac
