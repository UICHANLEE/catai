# CashLog 33-Leaf Operations

## Local Services

All development ports bind to loopback only:

| Service | URL | Purpose |
|---|---|---|
| Model API/report | `http://127.0.0.1:8010` | Private inference and model report |
| MLflow | `http://127.0.0.1:5500` | Runs, metrics, parameters, artifacts |
| Airflow | `http://127.0.0.1:8080` | Dataset-to-selection task graph |
| Jenkins | `http://127.0.0.1:8081` | Manual/CI orchestration |

`admin/admin` is a local Airflow bootstrap credential only. Replace it before any
non-loopback deployment and store the replacement in the secret manager.

Start infrastructure without replacing the production API image:

```bash
docker compose up -d mlflow airflow jenkins
```

Start the pinned inference worker:

```bash
CATAI_REQUIRE_INTERNAL_API_KEY=true \
CATAI_INTERNAL_API_KEY='<secret>' \
docker compose -f docker-compose.prod.yml up -d --build cashlog-api
```

Apple Silicon production-like testing uses native MPS instead of the CPU-only
Docker worker:

```bash
scripts/install_cashlog_mps_service.sh
curl --fail http://127.0.0.1:8010/health
tail -f logs/model-api.jsonl
tail -f logs/model-api.error.log
```

The owner-only `.runtime/cashlog-api.env` supplies the internal key and is ignored
by Git. `model-api.jsonl` contains rotating inference/request JSON events; library
and process errors go to `model-api.error.log`.

When CashLog calls the worker directly, set its server-only
`PRODUCT_ANALYZER_API_KEY` to the same value. The existing backend emits
`X-API-Key`; the worker also accepts the preferred `X-Internal-API-Key` during
the migration. Neither value belongs in React Native or a `VITE_` variable.

Health checks:

```bash
curl --fail http://127.0.0.1:5500/health
curl --fail http://127.0.0.1:8080/health
curl --fail http://127.0.0.1:8010/health
curl --fail --header "X-Internal-API-Key: $CATAI_INTERNAL_API_KEY" \
  http://127.0.0.1:8010/metrics
```

## Airflow Training

The DAG `cashlog33_training_pipeline` executes:

1. Validate source artifacts, model files, and the exact 33-label contract.
2. Rebuild the revision-pinned text dataset.
3. Re-score the visual proxy and train the SigLIP2 linear head.
4. Train and calibrate the text classifier.
5. Build a checksum-pinned isolated candidate config.
6. Generate deterministic Korean receipt fixtures.
7. Evaluate hybrid E2E behavior and log artifacts to MLflow.
8. Apply integration and production promotion gates.

It writes candidates to `checkpoints/cashlog33/airflow_latest`, reports to
`reports/cashlog33/airflow_latest`, and never modifies
`configs/cashlog/hybrid.serving.json`.

Manual trigger:

```bash
curl --user '<airflow-user>:<airflow-password>' \
  --header 'Content-Type: application/json' \
  --request POST \
  http://127.0.0.1:8080/api/v1/dags/cashlog33_training_pipeline/dagRuns \
  --data '{"dag_run_id":"manual-YYYYMMDD-HHMM","conf":{"source":"operator"}}'
```

Use the Airflow Grid view for task state and per-task logs. MLflow experiment
`cashlog33-hybrid-v2` stores component and E2E metrics. A failed training task retries
once after two minutes; the DAG has a four-hour timeout and only one active run.

## Jenkins Automation

Create one Jenkins username/password credential:

- ID: `airflow-local-basic`
- Username/password: the Airflow API account, not a personal reuse password.

Create a Pipeline job from this repository's `Jenkinsfile`. Jenkins then creates a
unique Airflow run, polls every 20 seconds, fails with the DAG, and archives the
trigger/status JSON plus `reports/cashlog33/airflow_latest`. Jenkins has no Docker
socket mount and cannot directly promote or replace the serving container.

## Promotion and Rollback

Run the real holdout validator and evaluator first. Then call the selector with real
metrics and `--require-production-ready`. A non-production candidate exits nonzero.
Promotion requires human approval after all gates pass:

1. Copy the selected checksum-pinned candidate to a versioned serving config.
2. Build an immutable API image tagged with the Git commit and model version.
3. Run health, invalid-input, auth, and CashLog canary checks.
4. Deploy only the backend-private worker.
5. Retain the previous image/config and roll back both together on regression.

Never point serving directly at `airflow_latest`; it is a mutable workspace.

## Production Security Checklist

- Cloudflare Tunnel terminates public traffic at Nginx/NestJS, not the model worker.
- Backend-to-worker traffic uses Tailscale and a rotated `X-Internal-API-Key`.
- Bind the worker to loopback or the Tailscale interface; do not port-forward it.
- Keep Supabase service role, Airflow, Jenkins, MLflow, and model report private.
- Use user JWT and authorization in NestJS; a shared mobile-app API key is not auth.
- Enforce request size, MIME/signature match, decode, pixel, timeout, and rate limits.
- Log request ID, model version, latency, result category, review flag, and error code;
  do not log raw image bytes, OCR PII, JWT, or internal keys.
- Mount model/checkpoint directories read-only and verify artifact SHA-256 at selection.
- Rotate development credentials before deployment and keep secrets out of Git/images.

## Monitoring and Alerts

Operational dashboards should track request count, HTTP errors, p50/p95 latency,
`need_user_check` rate, fallback rate, Top-3 selection rank, and correction rate by
leaf and model version. Alert on worker unavailability, p95 above three seconds,
unexpected fallback spikes, a leaf receiving no traffic, or a sudden correction-rate
increase. Accuracy and calibration are computed only after delayed manual truth is
available; online prediction confidence alone is not an accuracy metric.

Every inference emits a request ID, model version, device, category, confidence,
review flag, total latency, and stage latencies. The in-memory `/metrics` window is
bounded to 1,000 requests by default and resets on process restart.

For the public CashLog route, correlate the response `X-Request-ID` with
`model-api.jsonl`. `X-Cashlog-Read-Time-Ms`, `X-Cashlog-Optimize-Time-Ms`,
`X-Cashlog-Analyzer-Time-Ms`, and `X-Cashlog-Total-Time-Ms` separate Vercel work
from model latency without logging image or OCR contents.

Retraining is triggered by a reviewed data release, sustained per-leaf correction
drift, or a planned model change. It is not triggered automatically from unreviewed
user corrections.
