#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

umask 077

SERVICE_DIR="${CASHLOG_GATEWAY_DIR:-$HOME/services/cashlog-gateway}"
PYTHON_BIN="${GATEWAY_PYTHON:-$SERVICE_DIR/.venv/bin/python}"
GATEWAY_KEY_FILE="${GATEWAY_KEY_FILE:-$HOME/.config/cashlog/api-key}"
MODEL_KEY_FILE="${MODEL_KEY_FILE:-$HOME/.config/cashlog-gateway/model-api-key}"

for secret_file in "$GATEWAY_KEY_FILE" "$MODEL_KEY_FILE"; do
  if [ ! -s "$secret_file" ]; then
    echo "required secret file is missing or empty: $secret_file" >&2
    exit 1
  fi
done

if [ ! -x "$PYTHON_BIN" ]; then
  echo "gateway Python is missing: $PYTHON_BIN" >&2
  exit 1
fi

export GATEWAY_API_KEY="$(tr -d '\r\n' < "$GATEWAY_KEY_FILE")"
export MODEL_API_KEY="$(tr -d '\r\n' < "$MODEL_KEY_FILE")"
export MODEL_BASE_URL="${MODEL_BASE_URL:-http://127.0.0.1:18010}"
export MAX_REQUEST_BYTES="${MAX_REQUEST_BYTES:-14680064}"
export MAX_RESPONSE_BYTES="${MAX_RESPONSE_BYTES:-2097152}"
export EXPOSE_HEALTH="${EXPOSE_HEALTH:-false}"

cd "$SERVICE_DIR"
exec "$PYTHON_BIN" -m uvicorn proxy_server:app \
  --host "${GATEWAY_BIND_HOST:-127.0.0.1}" \
  --port "${GATEWAY_PORT:-8000}" \
  --workers 1 \
  --no-access-log
