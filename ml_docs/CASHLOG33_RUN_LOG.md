# CashLog 33-Leaf Execution Log

Run date: 2026-07-17 (Asia/Seoul)
Taxonomy: `13.33.1`, exactly 33 leaf IDs
Selected serving family: `cashlog33-hybrid-v1`
Decision: `guarded_integration_candidate`, not production-promoted

## 1. Outcome

The complete dataset-to-selection DAG ran successfully after one bounded-memory
training fix. It covers all 33 CashLog leaf IDs, logs component and end-to-end runs
to MLflow, produces checksum-pinned candidate artifacts, and serves authenticated
Top-3 recommendations from a loopback-only Docker API.

This run does **not** establish production accuracy. There is no frozen, manually
labeled real-photo holdout yet. Synthetic receipts, weak text labels, and Open Images
object proxies are kept as separate scopes throughout the reports. Automatic category
confirmation remains disabled.

## 2. Orchestrated Run

| Field | Value |
|---|---|
| Airflow DAG | `cashlog33_training_pipeline` |
| DAG run ID | `codex-20260717T0411KST` |
| Start | `2026-07-16T19:11:16.693054+00:00` |
| End | `2026-07-16T19:26:24.352695+00:00` |
| Final state | `success` |
| Task result | 9 of 9 tasks `success` |
| DAG import errors after restart | 0 |

Task results:

| Task | Attempts | Result |
|---|---:|---|
| `validate_inputs` | 1 | success |
| `build_text_dataset` | 1 | success |
| `score_visual_proxy` | 1 | success |
| `train_text` | 1 | success |
| `train_vision_head` | 2 | success |
| `build_candidate_config` | 1 | success |
| `generate_e2e_fixtures` | 1 | success |
| `evaluate_candidate` | 1 | success |
| `select_candidate` | 1 | success |

The first vision-head attempt exited with code 137. The trainer retained 1,068
augmented PIL images at once and exceeded the Airflow container's practical memory
budget. `scripts/train_cashlog33_vision_head.py` was changed to stream bounded image
batches, release each batch, and collect garbage; the DAG batch size was reduced from
16 to 4. Attempt two completed at roughly 2.08 GiB resident container memory.

## 3. MLflow Runs

Experiment ID `3`, name `cashlog33-hybrid-v2`:

| Scope | Run ID | State |
|---|---|---|
| Airflow text training | `7733ae80f3b44e59a25c68657411efcf` | FINISHED |
| Airflow vision-head training | `9307a30f940a40a988342652e71997e3` | FINISHED |
| Airflow hybrid E2E | `139ab7ba17d84c05bc02c1c5246256c4` | FINISHED |
| Pinned serving-config E2E | `59a60db4e3d446fb97ab199d05b43291` | FINISHED |

MLflow stores parameters, metrics, status, and report artifacts. The service is bound
to `http://127.0.0.1:5500`; it is not an internet-facing production service.

## 4. Data Acquisition and Coverage

| Source | Accepted output | Coverage | Decision |
|---|---:|---:|---|
| Revision-pinned MIT synthetic bank transactions | 18,669 text rows | weak mapping where available | used as proxy training data |
| Deterministic CashLog templates | 15,840 text rows | 33/33 leaves | used for routing/training support |
| Open Images V7 validation | 411 images | 23/33 leaves | used as visual object proxy |
| Fixed Noto receipt renderer | 99 images | 3 x 33 leaves | E2E I/O test only |
| Openverse API | no selected rows | n/a | anonymous API returned 401/429; excluded |
| PD12M discovery | no selected rows | n/a | dataset API/index returned 500-class errors; excluded |

The text dataset has 34,509 rows: 26,794 train, 3,839 validation, and 3,876
test. Group-based deterministic splitting prevents variations of one source/template
from crossing splits. Open Images collection removed 4,058 ambiguous cross-leaf
mappings and recorded no accepted-image download failures.

Open Images intentionally covers only 23 leaves. Rent, utilities, internet/TV,
financial products, and unclassified/other meanings cannot be inferred reliably from
generic object labels. OCR/text and safe fallback are the primary evidence paths for
those categories until real CashLog photos exist.

## 5. Measured Results by Scope

| Evaluation scope | Samples / leaves | Top-1 | Top-3 | Macro F1 |
|---|---:|---:|---:|---:|
| SigLIP2 zero-shot, Open Images proxy holdout | 82 / 23 | 0.5976 | 0.7073 | 0.5995 |
| Trained SigLIP2 linear head, same holdout | 82 / 23 | 0.7927 | 0.9390 | 0.8094 |
| TF-IDF SGD, synthetic/weak text test | 3,876 / 33 | 0.9972 | 0.9995 | 0.9981 |
| Full hybrid, fixed Noto synthetic receipts | 99 / 33 | 0.9899 | 0.9899 | 0.9896 |

The fixed synthetic hybrid run has minimum leaf recall `0.6667`, ECE `0.2743`,
false auto-confirm rate `0.0`, fallback rate `0.0404`, and local Mac p95 latency
`0.3776` seconds. The same fixtures inside Airflow produced Top-1 `0.9899` and p95
`1.3453` seconds. Every prediction still requested user confirmation.

These are proxy and integration metrics, not real-app accuracy. In particular, the
high synthetic text score is expected to be optimistic and the ECE already fails the
production calibration threshold.

## 6. Architecture and I/O Decision

Inference members:

1. SigLIP2 base patch16 224 embedding and 33-leaf zero-shot prior.
2. A trained logistic linear head over frozen SigLIP2 embeddings.
3. Locally pinned RapidOCR detector, classifier, and Korean recognizer ONNX files.
4. Word/character TF-IDF SGD text classifier.
5. Normalized CashLog OCR lexicon and conservative unreadable-input fallback.

Input is multipart `image` or JSON `imageBase64`, limited to JPEG, PNG, WebP,
HEIC, and HEIF. The API checks declared size, decoded size, file signature/MIME,
decode success, and pixel count before inference. Output includes taxonomy version,
model version, one recommended leaf, Top 3 probabilities, OCR/evidence fields,
fallback reasons, and `need_user_check`.

Training output is isolated under `checkpoints/cashlog33/airflow_latest`; training
never overwrites `configs/cashlog/hybrid.serving.json`. Selection verifies SHA-256 for
the vision model, vision head, text model, and all three OCR ONNX files before a
candidate can pass the integration gate.

## 7. Model Selection

The hybrid is selected because expense category meaning frequently appears in Korean
merchant/item text rather than image shape. The trained visual head improved Top-1 by
about 19.5 percentage points over zero-shot on the identical proxy holdout, while OCR
and text evidence correctly route categories that generic visual data cannot model.

Rejected or superseded attempts:

| Attempt | Failure or limitation | Resolution |
|---|---|---|
| Old four-leaf EfficientNet run | heartbeat timeout after epoch 25; wrong taxonomy for this task | retained only as historical evidence; not selected |
| ConvNeXt candidate | process exit 137 under host memory pressure | rejected; frozen SigLIP2 embeddings plus linear head selected |
| Eager vision augmentation | Airflow attempt one exit 137 | bounded streaming batches; retry succeeded |
| Initial MLflow artifacts | server advertised a read-only workspace artifact URI | enabled the MLflow artifact proxy using `mlflow-artifacts:/` |
| One host-side E2E MLflow call | sandbox denied the local loopback socket and the client retried indefinitely | stopped that attempt; reran with local service access and finished |
| Apple-font synthetic fixtures | materially different OCR score from Linux | fixed Noto font and one immutable 99-image fixture set for cross-runtime comparison |

## 8. Serving Verification

The production API image was rebuilt from `Dockerfile.api` for Linux ARM64. Its
uncompressed Docker size is 592,340,355 bytes and digest is
`sha256:64e212fda1c2a030bc89dfdfe06407ba5c5660a8c1c2a1259cb46ef21d1041dd`.
Model/checkpoint directories are mounted read-only and are not baked into the image.
The runtime uses uid/gid `10001`, a read-only root filesystem, a bounded tmpfs,
all Linux capabilities dropped, `no-new-privileges`, and an offline model runtime.

Final local checks against `127.0.0.1:8010`:

| Check | Result |
|---|---|
| Health and runtime availability | HTTP 200, hybrid runtime available |
| Missing internal key | HTTP 401 |
| Invalid Base64 image with a valid key | HTTP 400 |
| `X-Internal-API-Key` compatibility | HTTP 200 |
| Existing CashLog `X-API-Key` compatibility | HTTP 200 |
| Real app cafe receipt canary | `meal_cafe`, confidence 0.8143 |
| Canary Top 3 | `meal_cafe`, `meal_drink`, `meal_dining` |
| Canary taxonomy/model | `13.33.1`, `cashlog33-hybrid-v1` |
| Canary decision | `need_user_check=true` |

The first authenticated request included cold model/OCR loading and took about 16.6
seconds end to end; the repeated warm request took about 2.0 seconds. Cold-start
preloading and real-device latency must be measured before production promotion.

## 9. Promotion Status and Remaining Inputs

The current selector result is `integration_ready=true`,
`production_eligible=false`, and `auto_confirm_enabled=false`. The mandatory missing
input is a frozen, consented, manually reviewed real-photo holdout:

- At least 330 photos and at least 10 independent photos for every one of 33 leaves.
- Correct `leaf_id`, SHA-256, de-identified group ID, consent status, and manual review.
- No train/threshold tuning reuse, no duplicate or near-duplicate split leakage.
- Redacted PII and a defined private-storage retention/deletion policy.

After that input exists, one frozen run must pass Top-1 `>=0.95`, Top-3 `>=0.95`,
macro F1 `>=0.75`, every leaf recall `>=0.60`, ECE `<=0.08`, false auto-confirm
`<=0.02`, and p95 latency `<=3s` before automatic confirmation can be considered.

Deployment-specific operator actions still required:

1. Set one rotated secret as worker `CATAI_INTERNAL_API_KEY` and CashLog backend
   `PRODUCT_ANALYZER_API_KEY`; never place it in React Native or a `VITE_` variable.
2. Create Jenkins credential `airflow-local-basic` and a Pipeline job using
   `Jenkinsfile`.
3. Add the actual Backend/Home Server as a Tailnet node and apply the documented
   ACL. The current Tailnet contains only the Mac model worker and Galaxy relay.
4. Replace the local Airflow bootstrap `admin/admin` before any non-loopback use.

## 10. Galaxy Deployment Verification

The private relay was deployed to the Galaxy under
`~/services/cashlog-gateway` on 2026-07-17. The device is Android/aarch64 with
Termux Python 3.13.13. The CashLog application repository on the phone was not
modified; the relay has an isolated service directory and Python environment.

The existing relay exposed a concrete authentication defect: it forwarded the
incoming gateway key but did not add `X-Internal-API-Key`, while the Mac model
worker required that header. `/health` therefore returned 200 even though actual
inference could fail upstream authentication. The deployed relay now forwards two
independent credentials, streams request bodies with a 14 MiB limit, caps JSON
responses at 2 MiB, disables redirects and access logs, and requires the gateway
key for health checks.

Deployment checks:

| Check | Result |
|---|---|
| Dedicated SSH identity | key login succeeded |
| SSH password / keyboard login | disabled and actively rejected |
| SSH forwarding policy | reverse only; `GatewayPorts no` |
| Secret files | mode `600`; values never printed or committed |
| Relay without or with wrong key | HTTP 401 |
| Authenticated relay health | HTTP 200 |
| Galaxy-to-Mac tunnel health | HTTP 200 |
| Existing cafe receipt through Galaxy | `meal_cafe`, confidence 0.8143 |
| Warm end-to-end relay latency | 2.01 seconds |
| Galaxy LAN ports 8000 and 18010 | connection refused |
| Direct Cloudflare Quick Tunnel | restored for the explicitly requested test phase; URL is ephemeral |
| Galaxy Tailscale-only gateway bind | authenticated HTTP 200; unauthenticated 401 |
| macOS reverse-tunnel LaunchAgent | running; automatic restart verified |
| Obsolete reverse tunnels | removed; only managed loopback tunnel remains |
| Final Tailscale image E2E | `meal_cafe`, confidence 0.8143, 2.28 seconds |

The Galaxy relay is bound only to its assigned Tailscale IPv4 address. The same
port on the Wi-Fi/LAN address refuses connections. A macOS user LaunchAgent keeps
the reverse tunnel alive over the Galaxy Tailscale address; forced restart changed
the SSH process and recovered model health to HTTP 200 automatically. Both the
gateway bind and selected loopback model URL are stored in owner-only runtime files,
not in the repository.

The private Mac-to-Galaxy model path is complete. During the current test phase an
ephemeral Cloudflare Quick Tunnel is active in front of the authenticated Galaxy
gateway and Vercel points to that URL. React Native still receives neither the
Galaxy address nor the gateway key. This test tunnel must be replaced by a named,
policy-controlled tunnel before production release.

## 11. Monitoring Locations

- Airflow: `http://127.0.0.1:8080`, DAG run `codex-20260717T0411KST`.
- MLflow: `http://127.0.0.1:5500`, experiment `cashlog33-hybrid-v2`.
- Jenkins: `http://127.0.0.1:8081` after local setup.
- Model report: `http://127.0.0.1:8010/report` or
  `reports/cashlog33/model_report/index.html`.
- Machine-readable selection: `reports/cashlog33/model_selection.json`.

The online service should monitor request/error count, warm and cold latency,
fallback rate, review rate, selected Top-3 rank, and correction rate by leaf/model.
Raw images, OCR text containing PII, JWTs, and internal API keys must not be logged.

## 12. MPS Accuracy, Latency, and Observability Update (2026-07-17)

The worker was moved from the Linux Docker CPU runtime to a loopback-only macOS
LaunchAgent because Docker Desktop cannot expose Apple MPS. The service now eagerly
loads and warms the model, runs SigLIP2 in FP16 on MPS, overlaps MPS vision with CPU
OCR, bounds large OCR inputs, and records stage timing without image/OCR contents.

The first aggressive OCR resize failed the new Top-1 95% gate at 93.94%. It was
rejected. Restoring a 736px OCR minimum while capping large inputs at 960px recovered
the fixed 33-leaf integration result to Top-1 98.99% and Top-3 98.99%.

| Measurement | Before | After |
|---|---:|---:|
| 33-leaf fixture p50 | 349ms | 233ms |
| 33-leaf fixture p95 | 378ms | 278ms |
| Repeated 1254px cafe image, local API | about 710ms CPU Docker | about 350ms native MPS |
| SigLIP2 vision stage p50 | not recorded | 44ms |
| OCR stage p50 on fixtures | not recorded | 224ms |

The UECFood meal specialist was retrained for two MPS epochs after removing double
class balancing. Its fixed 4,249-image validation reached Top-1 98.05% and Top-3
100%. Its scope is only `meal_dining` and `meal_cafe`; it is retained as a candidate
artifact and is not evidence of 33-leaf real-photo accuracy.

Observability is available through protected `GET /metrics`, response headers
`X-Request-ID` and `X-Process-Time-Ms`, `logs/model-api.jsonl`, MLflow experiment
`cashlog33-mps-specialists`, and per-run `progress.json`/`training.jsonl` files.

## 13. Public Path Latency Remediation (2026-07-17)

The model worker was not the main source of the multi-second app delay. A repeated
2.65 MiB PNG request through the original Vercel `iad1` function and Quick Tunnel
took 5.30 to 6.32 seconds, while the same image on the local MPS worker took about
0.35 seconds. A direct authenticated Quick Tunnel request took 0.74 to 1.12 seconds.

CashLog now runs only the analyzer and analyzer-status functions in Vercel `icn1`,
re-encodes large upstream images to a maximum 960px JPEG, and performs the same
best-effort reduction in the browser before upload. The server keeps the validated
original when optimization is unavailable or does not reduce size. The 2.65 MiB
canary was reduced to about 138 KiB before the Galaxy hop without changing its
`meal_cafe` result.

| Public measurement | Before | After |
|---|---:|---:|
| Original 2.65 MiB request, browser optimization bypassed | 5.30-6.32s | 2.98-4.10s |
| Pre-compressed 424 KiB request | 3.30-3.54s on `iad1` | 1.71-2.15s on `icn1` |
| Final 960px / 138 KiB client-sized request | not available | 0.94-1.12s |
| Model stage inside measured requests | not correlated | 0.34-0.59s |

The deployed client bundle contains the browser-side compressor. Vercel responses
include `X-Cashlog-Read-Time-Ms`, `X-Cashlog-Optimize-Time-Ms`,
`X-Cashlog-Analyzer-Time-Ms`, `X-Cashlog-Total-Time-Ms`, byte counts, and the same
`X-Request-ID` forwarded through Galaxy to `logs/model-api.jsonl`. Vercel also emits
an `image_analysis_completed` JSON event without image, OCR, or secret contents.

These results validate the current test route only. The ephemeral Quick Tunnel
still must be replaced by a named, policy-controlled tunnel before production.
## 2026-07-17 - Consent-based feedback collection

- Added versioned confirmation events for accepted Top-1, alternate Top-3, and manual leaf edits.
- Added separate opt-in image-retention consent in CashLog; ordinary photo storage is not treated as training consent.
- Added Supabase pending-only client RLS, event idempotency, review state, model/taxonomy, Top-3, and private image reference fields.
- Added HMAC de-identification, duplicate/path validation, quarantine, active-learning priority scoring, 33-leaf readiness gates, and restricted image indexes.
- Added daily Airflow curation and MLflow metrics/artifacts. Automatic training from user feedback remains disabled until a reviewed release is explicitly approved.

## 2026-07-18 - Pre-deployment hard-example labeling

- Added a loopback-only 33-leaf labeling server at `http://127.0.0.1:8011`.
- Added server-side, atomic label decisions with revision checks and an append-only audit log.
- Added confirm, correct, reject, filter, search, and current-model Top-3 inspection flows.
- Added a queue builder that reuses the trained vision head and embedding cache instead of rerunning image inference.
- Current trained-head queue: 411 samples, 36 Top-1 mismatches, 330 uncertain samples, and 45 confident matches.
- Human-reviewed error-mining rows are locked to `train`; they cannot establish deployment accuracy.
- Verified 26 Python tests, Python bytecode compilation, JavaScript syntax, loopback/API security, and desktop/mobile layouts.

## 2026-07-23 - Isolated actual-data labeling

- Added `data/raw/cashlog33/actual` as the private destination for consented
  CashLog images; it is separate from all public, proxy, and synthetic sources.
- Added an idempotent importer for restricted feedback releases. It blocks path
  traversal, strips EXIF/GPS by image re-encoding, uses SHA-256 filenames,
  de-identifies sample IDs, and writes files/directories as `0600`/`0700`.
- Local mirror imports remove the source only after the image and manifest are
  durably committed. Supabase imports never delete remote originals.
- Added an Airflow `materialize_actual_dataset` task after feedback export.
  Rejected or nonconsented images cannot enter its secure source index.
- Added `predeploy_labeler --actual` on loopback port `8012`, with a separate
  unreviewed queue and output directory
  `data/processed/cashlog33/actual_review/v1`.
- Actual samples remain ineligible for training until a human confirms or
  corrects their 33-leaf label. Approved actual rows are locked to `train`;
  deployment accuracy still requires a separate untouched holdout.

## 2026-07-26 - Actual-label incremental retraining

- Exported 2 human-approved actual rows, both `meal_dining`, as train-locked
  inputs. No actual image, OCR text, identifier, or prediction file was committed.
- Reused the frozen Open Images embedding cache and encoded 8 augmented actual
  views with SigLIP2 on MPS in 7.50 seconds.
- Trained candidate `cashlog33-hybrid-actual-v1-candidate`; MLflow experiment 3,
  run `ea44469a8f67442cbc214e9ca380d8d7`.
- The untouched 62-row validation and 82-row test metrics were exactly unchanged:
  validation Top-1 75.81%, Top-3 96.77%, macro-F1 72.60%; test Top-1 79.27%,
  Top-3 93.90%, macro-F1 80.94%.
- On the fixed 99-row synthetic E2E set, both serving and candidate models scored
  Top-1 98.99%, Top-3 98.99%, and macro-F1 98.96%. Candidate p50/p95 latency was
  243/292ms versus 238/262ms for the serving model in these single warm runs.
- On the two training examples, fit improved from 1/2 to 2/2 and repaired the
  observed `transit_car` to `meal_dining` error. This is not holdout evidence
  because both examples were used for training.
- Promotion decision: **keep `cashlog33-hybrid-v1.1-fast` serving**. The candidate
  fixed the known training example but did not improve any untouched accuracy
  metric, so `configs/cashlog/hybrid.serving.json` and its pinned hash were not
  changed.

## 2026-07-27 - Full merged-dataset retraining correction

- Replaced the implicit embedding append workflow with an explicit, versioned
  merged dataset. It contains all 411 existing Open Images rows plus 2 reviewed
  actual rows: 413 total, split into train 269 / validation 62 / test 82.
- Preserved every original split from
  `checkpoints/cashlog33/vision_head_v1/split_manifest.jsonl`. Both actual rows
  remain human-approved and locked to train.
- The merged manifest SHA-256 is
  `40e4213cea8eaaac58cf445de7a5554b2641d1def717a2b2a12721ecb9ff83a5`.
  Its versioned location is
  `data/processed/cashlog33/training/incremental_v1/manifest.jsonl`.
- Re-embedded the full merged dataset on MPS without the old embedding cache:
  1,076 augmented train views, 62 validation images, and 82 test images in
  34.88 seconds.
- MLflow run `c1e9f769ca6449758da120512e013e4a` records candidate
  `cashlog33-vision-head-merged-v1`.
- Validation remained Top-1 75.81%, Top-3 96.77%, macro-F1 72.60%; test remained
  Top-1 79.27%, Top-3 93.90%, macro-F1 80.94%.
- The 99-row synthetic E2E result also remained Top-1 98.99%, Top-3 98.99%, and
  macro-F1 98.96%. The two reviewed training rows scored 2/2.
- Promotion decision remains **no replacement**: the candidate learned the known
  rows but did not improve an untouched metric. The current serving config is
  unchanged.

## 2026-07-27 - Uncapped all-data MPS retraining

- Removed the text source cap and consumed all 60,000 expense-mappable source rows
  plus all 15,840 generated rows. The remaining 8,000 source rows describe income
  or transfers and have no valid target in the 33 expense leaves.
- Built a 75,840-row text manifest: 59,685 train, 8,067 validation, and 8,088
  test. MLflow run `7f6249fb766141de8c9487c8aacfd9d6` reached test Top-1
  `0.9984`, Top-3 `1.0000`, and macro-F1 `0.9983`.
- Built the general visual manifest from all 411 Open Images rows, all 61
  license-checked weak Openverse rows, and both approved actual rows: 474 source
  images. Weak rows are train-only; the fixed 62/82 proxy validation/test images
  remain untouched.
- MPS encoded all 1,320 augmented training views plus all validation/test images.
  MLflow run `de2293b95dc748c1881b84af6becdab0` reached test Top-1 `0.8049`,
  Top-3 `0.9390`, and macro-F1 `0.7903`. Top-1 improved over `0.7927`, while
  macro-F1 declined from `0.8094`; this proxy head was therefore required to pass
  the full hybrid regression gate before selection.
- Auditing the UECFood override found that substring matching treated `tea` as a
  match inside `steak` and `steamed`, and ingredient words incorrectly fabricated
  grocery labels from prepared dishes. Those runs were rejected. Overrides now use
  word-boundary matching and only map prepared-food dining/cafe semantics.
- The corrected mapping consumes all 31,395 UECFood images: 29,002 dining and
  2,393 cafe, with 26,686 train and 4,709 deterministic validation images. Existing
  specialist baseline run `ba17031e68274ec0a56357e78a4e05bd` scored Top-1
  `0.9620` and minimum leaf recall `0.9610`.
- The all-data MPS fine-tune consumed the complete train split in one epoch.
  MLflow runs `c4bcdab3b4db411e96f03227409bfc65` and
  `48ecc29e0a5d442cbddc3ab233efbe2e` scored Top-1 `0.9565` and minimum
  leaf recall `0.9443`. It passed the 95% gate but regressed, so it was not
  selected over the stronger existing specialist.
- On the fixed 99-receipt synthetic integration set, serving baseline run
  `a639c571ae6242cc8e75a698208a2a42` and all-data candidate run
  `d4fbefd6292443c2989be88cb70d0853` both scored Top-1/Top-3 `0.9899`
  and macro-F1 `0.9896`. Candidate p95 latency was `256.7ms` versus `274.6ms`.
- Promotion decision: select `cashlog33-all-data-mps-v1` with the all-data
  visual/text heads and the stronger existing meal specialist. Keep
  `allow_auto_confirm=false`; these proxy and synthetic results do not establish
  95% accuracy on a frozen real CashLog photo holdout.
- Restarted the loopback-only macOS LaunchAgent. `/health` reports MPS, the selected
  model version, and loaded/warmed artifacts. An authenticated multipart grocery
  fixture returned `meal_grocery`, the selected model version, and the full
  `siglip2+mobilenetv4+rapidocr+tfidf` member contract.
