#!/usr/bin/env bash
set -euo pipefail

if [ -z "${GALAXY_SSH_HOST:-}" ] || [ -z "${GALAXY_SSH_USER:-}" ]; then
  echo "usage: GALAXY_SSH_USER=<termux-user> GALAXY_SSH_HOST=<tailscale-ip-or-name> $0" >&2
  exit 2
fi

GALAXY_SSH_PORT="${GALAXY_SSH_PORT:-8022}"
GALAXY_TUNNEL_BIND="${GALAXY_TUNNEL_BIND:-127.0.0.1}"
GALAXY_TUNNEL_PORT="${GALAXY_TUNNEL_PORT:-18010}"
MAC_MODEL_HOST="${MAC_MODEL_HOST:-127.0.0.1}"
MAC_MODEL_PORT="${MAC_MODEL_PORT:-8010}"
GALAXY_SSH_IDENTITY="${GALAXY_SSH_IDENTITY:-}"

identity_args=()
if [ -n "$GALAXY_SSH_IDENTITY" ]; then
  identity_args=(-i "$GALAXY_SSH_IDENTITY")
fi

exec ssh "${identity_args[@]}" -p "$GALAXY_SSH_PORT" -N -T \
  -o BatchMode=yes \
  -o IdentitiesOnly=yes \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -R "${GALAXY_TUNNEL_BIND}:${GALAXY_TUNNEL_PORT}:${MAC_MODEL_HOST}:${MAC_MODEL_PORT}" \
  "${GALAXY_SSH_USER}@${GALAXY_SSH_HOST}"
