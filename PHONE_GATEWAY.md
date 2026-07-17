# Private Galaxy Relay

This optional deployment uses the Galaxy Note10+ only as a Tailscale-private
relay. It is not a public API gateway. Public requests terminate at Cloudflare,
Nginx, and NestJS on the home server before reaching this path.

## Topology

```text
React Native
  -> Cloudflare HTTPS
  -> Home server Nginx/NestJS
  -> Tailscale ACL
  -> Galaxy relay
  -> Galaxy 127.0.0.1:18010
  -> SSH reverse tunnel
  -> Mac/private worker 127.0.0.1:8010
```

No router port forwarding is used. Cloudflare must not point at the Galaxy relay
or model worker. A future on-phone ONNX worker can replace the reverse-tunnel
target without changing the NestJS-to-relay contract.

## 1. Lock Down Galaxy SSH

In Termux, install OpenSSH and use key-only authentication. In
`$PREFIX/etc/ssh/sshd_config`:

```text
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
AllowTcpForwarding yes
GatewayPorts no
```

The repository includes the stricter drop-in
`deploy/phone_gateway/sshd_config.d/90-cashlog-security.conf`. It also disables
agent, X11, and TUN forwarding and limits TCP forwarding to the reverse
direction required by the model tunnel. Run `sshd -t` before reloading it.

Restart `sshd` and verify a new key-authenticated session before closing the old
one. Restrict SSH to the MacBook identity in the Tailscale ACL.

## 2. Run the Private Model Worker

On the MacBook, start the production worker on loopback with internal auth:

```bash
export CATAI_REQUIRE_INTERNAL_API_KEY=true
export CATAI_INTERNAL_API_KEY='<model-worker-secret>'
docker compose -f docker-compose.prod.yml up -d cashlog-api
```

Open the loopback-only reverse tunnel:

```bash
GALAXY_SSH_USER=u0_a123 \
GALAXY_SSH_HOST=<galaxy-tailscale-name-or-ip> \
GALAXY_SSH_PORT=8022 \
bash scripts/open_phone_reverse_tunnel.sh
```

Galaxy `127.0.0.1:18010` now reaches Mac `127.0.0.1:8010`. It must not be
reachable through the Galaxy Wi-Fi or Tailscale address because `GatewayPorts`
is disabled.

## 3. Run the Galaxy Relay

Install `deploy/phone_gateway/requirements.txt`, generate two independent
secrets, and launch on the Galaxy Tailscale address only:

```bash
export GATEWAY_API_KEY='<nest-backend-to-galaxy-secret>'
export MODEL_API_KEY='<same-value-as-CATAI_INTERNAL_API_KEY>'
export MODEL_BASE_URL=http://127.0.0.1:18010
export MAX_REQUEST_BYTES=14680064
export MAX_RESPONSE_BYTES=2097152
export EXPOSE_HEALTH=false

python -m uvicorn proxy_server:app \
  --host <galaxy-tailscale-ip> \
  --port 8000 \
  --workers 1 \
  --no-access-log
```

The included `run_gateway.sh`, `start_gateway.sh`, and `stop_gateway.sh` read
the two secrets from mode-`600` files instead of command-line arguments. They
also acquire a Termux wake lock and keep the process ID and logs under private
runtime/state directories. `GATEWAY_BIND_HOST` remains `127.0.0.1` until the
Tailscale address and ACL are ready.

Termux Python 3.13 currently supplies its Android-compatible Pydantic build as
a platform package. Create the gateway environment with
`python -m venv --system-site-packages .venv`; the requirements deliberately
keep Pydantic on that compatible major version while updating the pure-Python
FastAPI and Starlette layers.

For reboot recovery, install the official Termux:Boot Android app and copy
`termux_boot_start.sh` to `~/.termux/boot/20-cashlog-gateway`. The script starts
key-only SSH if needed, acquires a wake lock, and starts the loopback gateway.
Android battery optimization must be disabled for Termux and Termux:Boot.

NestJS sends `X-API-Key: <GATEWAY_API_KEY>` over Tailscale. The relay verifies
that key, forwards the request to loopback, and adds
`X-Internal-API-Key: <MODEL_API_KEY>` for the model worker. Neither secret is
stored in React Native.

With `EXPOSE_HEALTH=false`, `/health` is available only when the same
`X-API-Key` header is present. Request bodies are streamed with a 14 MiB cap,
and model responses are rejected above 2 MiB.

## 4. Tailscale Policy

Allow only the home-server identity to connect to Galaxy TCP 8000 and only the
MacBook identity to connect to Galaxy SSH 8022. Deny other tailnet members by
default. Keep Galaxy Wi-Fi/LAN firewall exposure disabled and do not advertise
this relay through Funnel.

## 5. Failure Behavior

- Missing/wrong gateway key: `401`.
- Missing model-worker key configuration: `503`.
- Reverse tunnel or worker unavailable: `503`.
- Worker timeout: `504`.
- Oversized or unsupported request: `413` or `415`.

NestJS should map these to a retryable user state without logging image bytes,
OCR text, JWTs, or either internal key.
