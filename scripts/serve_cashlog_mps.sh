#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${CATAI_ENV_FILE:-$ROOT/.runtime/cashlog-api.env}"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

export CATAI_CASHLOG_HYBRID_CONFIG="${CATAI_CASHLOG_HYBRID_CONFIG:-$ROOT/configs/cashlog/hybrid.serving.json}"
export CATAI_DEVICE=mps
export CATAI_EAGER_LOAD_MODEL="${CATAI_EAGER_LOAD_MODEL:-true}"
export CATAI_WARMUP_MODEL="${CATAI_WARMUP_MODEL:-true}"
export CATAI_MAX_CONCURRENT_INFERENCE="${CATAI_MAX_CONCURRENT_INFERENCE:-1}"
export CATAI_LOG_HEALTH_REQUESTS="${CATAI_LOG_HEALTH_REQUESTS:-false}"
export CATAI_REQUIRE_INTERNAL_API_KEY="${CATAI_REQUIRE_INTERNAL_API_KEY:-true}"
export CATAI_JSON_LOG_PATH="${CATAI_JSON_LOG_PATH:-$ROOT/logs/model-api.jsonl}"
export CATAI_LOG_MAX_BYTES="${CATAI_LOG_MAX_BYTES:-20971520}"
export CATAI_LOG_BACKUP_COUNT="${CATAI_LOG_BACKUP_COUNT:-5}"

if [[ "$CATAI_REQUIRE_INTERNAL_API_KEY" == "true" && -z "${CATAI_INTERNAL_API_KEY:-}" ]]; then
  echo "CATAI_INTERNAL_API_KEY is required in $ENV_FILE" >&2
  exit 1
fi

cd "$ROOT"
"$ROOT/.venv/bin/python" -c \
  "import torch; assert torch.backends.mps.is_available(), 'MPS is unavailable'"

exec "$ROOT/.venv/bin/python" -m uvicorn main:app \
  --host 127.0.0.1 \
  --port "${CASHLOG_API_PORT:-8010}" \
  --workers 1 \
  --no-access-log
