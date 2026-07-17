#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.cashlog.model-api"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
TEMPLATE="$ROOT/deploy/macos/$LABEL.plist.example"
LOG_DIR="$ROOT/logs"
ENV_FILE="$ROOT/.runtime/cashlog-api.env"
DOMAIN="gui/$(id -u)"

if [[ ! -x "$ROOT/scripts/serve_cashlog_mps.sh" ]]; then
  echo "scripts/serve_cashlog_mps.sh must be executable" >&2
  exit 1
fi
if [[ ! -s "$ENV_FILE" ]]; then
  echo "missing protected runtime environment: $ENV_FILE" >&2
  exit 1
fi
if [[ "$(stat -f '%Lp' "$ENV_FILE")" != "600" ]]; then
  echo "$ENV_FILE must have mode 600" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a
SERVICE_PORT="${CASHLOG_API_PORT:-8010}"

mkdir -p "$HOME/Library/LaunchAgents" "$LOG_DIR"
sed \
  -e "s|__PROJECT_ROOT__|$ROOT|g" \
  -e "s|__LOG_DIR__|$LOG_DIR|g" \
  "$TEMPLATE" > "$PLIST"
chmod 600 "$PLIST"
plutil -lint "$PLIST"

launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
for _ in {1..20}; do
  if ! launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done

bootstrap_ok=false
for attempt in {1..5}; do
  if launchctl bootstrap "$DOMAIN" "$PLIST"; then
    bootstrap_ok=true
    break
  fi
  if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
    bootstrap_ok=true
    break
  fi
  echo "launchctl bootstrap attempt $attempt failed; retrying" >&2
  sleep 2
done
if [[ "$bootstrap_ok" != "true" ]]; then
  echo "failed to bootstrap $LABEL after 5 attempts" >&2
  exit 1
fi

launchctl kickstart -k "$DOMAIN/$LABEL"

ready=false
for _ in {1..30}; do
  if curl -fs --max-time 2 "http://127.0.0.1:$SERVICE_PORT/health" >/dev/null; then
    ready=true
    break
  fi
  sleep 2
done
if [[ "$ready" != "true" ]]; then
  echo "$LABEL did not become healthy within 60 seconds" >&2
  tail -n 40 "$LOG_DIR/model-api.error.log" >&2 || true
  exit 1
fi

echo "installed=$PLIST"
echo "logs=$LOG_DIR/model-api.jsonl"
echo "health=ok"
launchctl print "$DOMAIN/$LABEL" | sed -n '1,35p'
