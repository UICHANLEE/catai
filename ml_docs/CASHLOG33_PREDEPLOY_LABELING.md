# CashLog 33-leaf pre-deployment labeling

## Purpose

This workflow fixes dataset labels and mines hard examples before the first
production deployment. It is separate from the post-deployment feedback loop in
`CASHLOG33_FEEDBACK_LOOP.md`.

The Open Images manifest contains 411 local images. The earlier zero-shot scorer
disagreed with the source-to-CashLog label on 196 of them. The currently trained
SigLIP2 linear head reduces Top-1 disagreements to 36; another 330 samples are
flagged as uncertain by the initial confidence/margin policy. Disagreement does
not prove that either side is correct, so every queued item supports three human
decisions:

- Confirm the dataset label: the model missed a valid hard example.
- Select another of the 33 leaves: the dataset mapping was wrong.
- Reject the sample: the image is ambiguous, irrelevant, unsafe, or unusable.

## Start the local tool

First build the queue from the currently trained vision-head artifact and its
embedding cache. This does not run image inference again.

```bash
.venv/bin/python -m catai.predeploy_queue
.venv/bin/python -m catai.predeploy_labeler
```

Open `http://127.0.0.1:8011`. The server refuses non-loopback hosts and origins.
It must not be exposed through Cloudflare Tunnel, Tailscale Funnel, a reverse
proxy, or router port forwarding.

## Actual CashLog image queue

Real CashLog feedback images are never mixed into the public/proxy source tree.
The daily feedback DAG materializes only non-rejected images with explicit model
improvement retention consent into this private tree:

```text
data/raw/cashlog33/actual/
  images/<sha256-prefix>/<sha256>.jpg
  manifest.jsonl
  import_audit.jsonl
  import_quarantine.jsonl
  import_summary.json
```

The importer removes EXIF/GPS metadata, normalizes the image to JPEG, replaces
the storage filename with a content hash, and omits user ID, object key, HMAC
group ID, expense ID, and event ID from the labeling manifest. Files use mode
`0600` and directories use `0700`. Supabase originals are not deleted by this
import job; deletion and consent withdrawal are separate retention operations.

Airflow runs the importer automatically after the de-identified export. To
materialize an existing release manually from private Supabase Storage:

```bash
SUPABASE_URL=https://PROJECT.supabase.co \
SUPABASE_SERVICE_ROLE_KEY=... \
.venv/bin/python scripts/sync_cashlog_actual.py \
  --release-dir data/feedback/releases/RELEASE_ID \
  --fail-on-quarantine
```

Keep the service role key in the backend environment, never in shell history,
the app, a manifest, or a command-line argument. For a local private-storage
mirror, use `--source-root /private/mirror`; validated source files are removed
only after the sanitized destination and manifest are durably committed.

Start the separate actual-only labeler:

```bash
.venv/bin/python -m catai.predeploy_labeler --actual
```

Open `http://127.0.0.1:8012`. This preset reads only
`data/raw/cashlog33/actual/manifest.jsonl`, starts on the unreviewed queue, and
writes decisions only to `data/processed/cashlog33/actual_review/v1`. The proxy
queue on port `8011` and all of its decisions remain untouched. Restart the
actual labeler after a daily import to load newly arrived samples.

To inspect another scored manifest:

```bash
.venv/bin/python -m catai.predeploy_labeler \
  --input-manifest data/raw/cashlog33/openverse_smoke/scored_manifest.jsonl \
  --output-dir data/processed/cashlog33/predeploy_review/smoke
```

When `current_model_scored_manifest.jsonl` exists, the labeler selects it by
default. Otherwise it falls back to the earlier zero-shot scored manifest and
prints that input path at startup. The queue is ordered by current-model mismatch,
automatic rejection, pending status, and uncertainty. Decisions are saved
immediately on the server; closing the browser does not lose progress.

## Durable outputs

The default output directory is
`data/processed/cashlog33/predeploy_review/v1`.

The actual-only preset uses
`data/processed/cashlog33/actual_review/v1` with the same file contract.

| File | Purpose |
| --- | --- |
| `decisions.jsonl` | Latest decision for each sample |
| `decision_audit.jsonl` | Append-only decision history |
| `corrected_manifest.jsonl` | Full input manifest with human decisions applied |
| `training_manifest.jsonl` | Rows eligible for the dataset builder |
| `human_verified_manifest.jsonl` | Human-approved rows only |
| `labeling_summary.json` | Counts, paths, source checksum, and evaluation policy |

Decision and export files are written with mode `0600`; the output directory is
mode `0700`. Every mutation uses optimistic revision checks and a process-local
token to prevent stale or cross-origin writes.

## Export and rebuild

Use the UI command or run:

```bash
.venv/bin/python -m catai.predeploy_labeler --export-only

.venv/bin/python scripts/build_cashlog33_dataset.py \
  --input-manifest data/processed/cashlog33/predeploy_review/v1/training_manifest.jsonl \
  --output-dir data/processed/cashlog33/human_corrected_v1 \
  --allow-incomplete
```

`--allow-incomplete` is appropriate while labeling is still in progress. It must
be removed for a promotion-candidate dataset build.

## Evaluation boundary

Every human-reviewed sample is assigned `split_lock=train`. These samples were
selected after observing current-model errors, so using them for validation or
test would inflate reported accuracy. Deployment metrics must use a separate,
frozen holdout that was never used for error mining, prompt changes, calibration,
or training.

The pre-deployment iteration is:

1. Score candidate training data with the current model.
2. Review mismatches and uncertain samples.
3. Export the corrected training manifest.
4. Rebuild and retrain the candidate model.
5. Re-score remaining training candidates.
6. Evaluate once against the untouched real-photo holdout.

After deployment, user confirmations and corrections enter the consent-based
Airflow/MLflow feedback loop instead. The two sources remain distinguishable by
`review_method` and are never silently mixed into the frozen holdout.

Actual rows remain `pending` until the operator confirms, corrects, or rejects
them in the actual-only tool. Only human-approved rows appear in its
`training_manifest.jsonl`; they are locked to `train` and cannot be used to
claim holdout accuracy.
