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

After that input exists, one frozen run must pass Top-1 `>=0.80`, Top-3 `>=0.95`,
macro F1 `>=0.75`, every leaf recall `>=0.60`, ECE `<=0.08`, false auto-confirm
`<=0.02`, and p95 latency `<=3s` before automatic confirmation can be considered.

Deployment-specific operator actions still required:

1. Set one rotated secret as worker `CATAI_INTERNAL_API_KEY` and CashLog backend
   `PRODUCT_ANALYZER_API_KEY`; never place it in React Native or a `VITE_` variable.
2. Create Jenkins credential `airflow-local-basic` and a Pipeline job using
   `Jenkinsfile`.
3. Supply/operate the actual Tailscale ACL and Galaxy/home-server credentials. This
   repository provides the private relay and reverse-tunnel implementation but cannot
   deploy to an external phone without access.
4. Replace the local Airflow bootstrap `admin/admin` before any non-loopback use.

## 10. Monitoring Locations

- Airflow: `http://127.0.0.1:8080`, DAG run `codex-20260717T0411KST`.
- MLflow: `http://127.0.0.1:5500`, experiment `cashlog33-hybrid-v2`.
- Jenkins: `http://127.0.0.1:8081` after local setup.
- Model report: `http://127.0.0.1:8010/report` or
  `reports/cashlog33/model_report/index.html`.
- Machine-readable selection: `reports/cashlog33/model_selection.json`.

The online service should monitor request/error count, warm and cold latency,
fallback rate, review rate, selected Top-3 rank, and correction rate by leaf/model.
Raw images, OCR text containing PII, JWTs, and internal API keys must not be logged.
