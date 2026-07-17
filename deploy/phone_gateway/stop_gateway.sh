#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

RUNTIME_DIR="${XDG_RUNTIME_HOME:-$HOME/.local/run}/cashlog-gateway"
PID_FILE="$RUNTIME_DIR/gateway.pid"

if [ ! -s "$PID_FILE" ]; then
  echo "cashlog gateway is not managed by this deployment"
  exit 0
fi

pid="$(cat "$PID_FILE")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
  for _ in 1 2 3 4 5; do
    kill -0 "$pid" 2>/dev/null || break
    sleep 1
  done
fi
rm -f "$PID_FILE"
echo "cashlog gateway stopped"
