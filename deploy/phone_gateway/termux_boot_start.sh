#!/data/data/com.termux/files/usr/bin/bash
set -u

termux-wake-lock >/dev/null 2>&1 || true
pgrep -x sshd >/dev/null 2>&1 || sshd
sleep 2
"$HOME/services/cashlog-gateway/start_gateway.sh"
