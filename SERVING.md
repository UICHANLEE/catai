# CashLog Model Serving

The current worker serves `cashlog33-hybrid-v1`, covering the exact 33 category
leaves in `configs/cashlog/categories.json`.

## Private Runtime

Build and start the API-only CPU image:

```bash
export CATAI_REQUIRE_INTERNAL_API_KEY=true
export CATAI_INTERNAL_API_KEY='<backend-only-secret>'
docker compose -f docker-compose.prod.yml up -d --build cashlog-api
```

The default host binding is `127.0.0.1:8010`. For a separate worker device, bind
only to its Tailscale interface and allow only the NestJS backend. Do not expose
this service through Cloudflare Tunnel, a router port-forward, or a public IP.

Public traffic follows:

```text
React Native -> Cloudflare -> Nginx -> NestJS -> Tailscale -> Catai FastAPI
```

## Endpoints

```http
GET /health
POST /analyze-image
```

Supported image containers are JPEG, PNG, WebP, HEIC, and HEIF. The server checks
the file signature, declared MIME type, filename extension, safe decode, 10 MiB
decoded size limit, 14 MiB request limit, and 30 megapixel limit.

Example backend call:

```bash
curl --fail --silent --show-error \
  --header "X-Internal-API-Key: $CATAI_INTERNAL_API_KEY" \
  --form image=@receipt.jpg \
  http://127.0.0.1:8010/analyze-image
```

The response contains `recommended_category`, a normalized Top 3,
`need_user_check`, model/version information, OCR/vision evidence, and fallback
reasons. Current serving policy has `allow_auto_confirm=false`, so the app must
offer Top 3 or manual category selection.

## Artifact Integrity

`configs/cashlog/hybrid.serving.json` pins SHA-256 for SigLIP2, the visual head,
the text model, and Korean OCR model. Airflow writes only an isolated candidate
configuration. An operator promotes a versioned candidate only after all real
holdout gates pass and deployment approval is recorded.

The local production Compose file also mounts the generated model report read-only,
so retraining evidence can be reviewed without rebuilding the runtime image. This
mount does not include datasets, checkpoints, or secrets.

The API runs as uid/gid `10001`, with a read-only root filesystem, a bounded `/tmp`
tmpfs, all Linux capabilities dropped, `no-new-privileges`, and an internal Docker
healthcheck. The runtime is forced offline for Hugging Face/Transformers; only pinned
local model files are loaded.

## Documentation

- Architecture and I/O: `ml_docs/CASHLOG33_MODEL_DESIGN.md`
- Data provenance and privacy: `ml_docs/CASHLOG33_DATA_CARD.md`
- Runbooks, Jenkins, and security: `ml_docs/CASHLOG33_OPERATIONS.md`
- Metrics, failures, and blockers: `ml_docs/CASHLOG33_RUN_LOG.md`
