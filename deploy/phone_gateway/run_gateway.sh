#!/data/data/com.termux/files/usr/bin/bash
set -euo pipefail

umask 077

SERVICE_DIR="${CASHLOG_GATEWAY_DIR:-$HOME/services/cashlog-gateway}"
PYTHON_BIN="${GATEWAY_PYTHON:-$SERVICE_DIR/.venv/bin/python}"
GATEWAY_KEY_FILE="${GATEWAY_KEY_FILE:-$HOME/.config/cashlog/api-key}"
MODEL_KEY_FILE="${MODEL_KEY_FILE:-$HOME/.config/cashlog-gateway/model-api-key}"
BIND_HOST_FILE="${BIND_HOST_FILE:-$HOME/.config/cashlog-gateway/bind-host}"
MODEL_BASE_URL_FILE="${MODEL_BASE_URL_FILE:-$HOME/.config/cashlog-gateway/model-base-url}"

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
export MAX_REQUEST_BYTES="${MAX_REQUEST_BYTES:-14680064}"
export MAX_RESPONSE_BYTES="${MAX_RESPONSE_BYTES:-2097152}"
export EXPOSE_HEALTH="${EXPOSE_HEALTH:-false}"

default_bind_host="127.0.0.1"
if [ -s "$BIND_HOST_FILE" ]; then
  default_bind_host="$(tr -d '\r\n' < "$BIND_HOST_FILE")"
fi

default_model_base_url="http://127.0.0.1:18010"
if [ -s "$MODEL_BASE_URL_FILE" ]; then
  default_model_base_url="$(tr -d '\r\n' < "$MODEL_BASE_URL_FILE")"
fi
export MODEL_BASE_URL="${MODEL_BASE_URL:-$default_model_base_url}"

cd "$SERVICE_DIR"
exec "$PYTHON_BIN" -m uvicorn proxy_server:app \
  --host "${GATEWAY_BIND_HOST:-$default_bind_host}" \
  --port "${GATEWAY_PORT:-8000}" \
  --workers 1 \
  --no-access-log
