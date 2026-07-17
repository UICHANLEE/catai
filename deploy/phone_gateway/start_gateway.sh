#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

SERVICE_DIR="${CASHLOG_GATEWAY_DIR:-$HOME/services/cashlog-gateway}"
RUNTIME_DIR="${XDG_RUNTIME_HOME:-$HOME/.local/run}/cashlog-gateway"
LOG_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/cashlog-gateway"
PID_FILE="$RUNTIME_DIR/gateway.pid"
LOG_FILE="$LOG_DIR/gateway.log"

mkdir -p "$RUNTIME_DIR" "$LOG_DIR"
chmod 700 "$RUNTIME_DIR" "$LOG_DIR"

if [ -s "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "cashlog gateway is already running"
  exit 0
fi

termux-wake-lock >/dev/null 2>&1 || true
nohup "$SERVICE_DIR/run_gateway.sh" >>"$LOG_FILE" 2>&1 &
pid=$!
printf '%s\n' "$pid" >"$PID_FILE"

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "cashlog gateway failed to start; inspect $LOG_FILE" >&2
  exit 1
fi

echo "cashlog gateway started with pid $pid"
