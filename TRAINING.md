# Training

## Current CashLog 33-leaf target

The production-aligned workflow is Airflow DAG
`cashlog33_training_pipeline`. It rebuilds the 33-leaf text dataset, scores the
visual proxy, trains the SigLIP2 linear head and text classifier, builds an isolated
checksum-pinned candidate, evaluates fixed E2E fixtures, and applies promotion gates.

The verified run `codex-20260717T0411KST` finished successfully with 9 of 9 tasks.
Candidate files are written to `checkpoints/cashlog33/airflow_latest` and
`reports/cashlog33/airflow_latest`; training never changes
`configs/cashlog/hybrid.serving.json`.

```bash
docker compose up -d mlflow airflow jenkins

curl --user '<airflow-user>:<airflow-password>' \
  --header 'Content-Type: application/json' \
  --request POST \
  http://127.0.0.1:8080/api/v1/dags/cashlog33_training_pipeline/dagRuns \
  --data '{"dag_run_id":"manual-YYYYMMDD-HHMM","conf":{"source":"operator"}}'
```

Current model family: `cashlog33-hybrid-v1` (`SigLIP2 + linear head + Korean
RapidOCR + text SGD + lexicon`). All 33 IDs are trained/routed, but production
promotion remains blocked until a frozen real-photo holdout covers at least 10
samples per leaf and passes every accuracy, recall, calibration, safety, and latency
gate. Proxy and synthetic scores are not real-photo accuracy.

Primary records:

- `ml_docs/CASHLOG33_MODEL_DESIGN.md`
- `ml_docs/CASHLOG33_DATA_CARD.md`
- `ml_docs/CASHLOG33_OPERATIONS.md`
- `ml_docs/CASHLOG33_PREDEPLOY_LABELING.md`
- `ml_docs/CASHLOG33_RUN_LOG.md`
- `reports/cashlog33/model_report/index.html`

## Pre-deployment error review

Before the first deployment, inspect current-model mismatches against the scored
training candidate manifest with the loopback-only labeling tool:

```bash
.venv/bin/python -m catai.predeploy_queue
.venv/bin/python -m catai.predeploy_labeler
```

Open `http://127.0.0.1:8011`. Confirm a valid source label, correct it to any of
the 33 leaves, or reject the sample. Decisions are stored server-side and the
generated `training_manifest.jsonl` feeds `scripts/build_cashlog33_dataset.py`.
All reviewed error-mining samples are locked to `train`; production metrics still
require a separate untouched holdout. See
`ml_docs/CASHLOG33_PREDEPLOY_LABELING.md` for the complete I/O contract.

## MPS target-95 run (2026-07-17)

The UECFood meal specialist was initialized from the previous MobileNetV4
checkpoint and retrained natively on Apple MPS. The old pipeline used both a
balanced sampler and inverse-frequency loss weights, which double-corrected the
minority class. The current trainer uses exactly one balancing mechanism.

```bash
.venv/bin/python scripts/launch_training.py cashlog_meal2_mps_target95
```

Verified fixed validation result over 4,249 images:

| Scope | Epoch | Top-1 | Top-3 | Target |
|---|---:|---:|---:|---:|
| `meal_dining` / `meal_cafe` specialist | 2 | 98.05% | 100.00% | >=95% PASS |
| 33-leaf synthetic integration | n/a | 98.99% | 98.99% | >=95% PASS |

The first number is not 33-leaf or real-app accuracy. The 33-leaf number is an I/O
integration fixture score, not a real-photo holdout. A model cannot be marked
production-eligible until the manually labeled 33-leaf real holdout also reaches
Top-1 95%.

Monitoring locations:

- `checkpoints/cashlog_meal2_mps_target95/progress.json`
- `checkpoints/cashlog_meal2_mps_target95/training.jsonl`
- `checkpoints/cashlog_meal2_mps_target95/metrics.csv`
- MLflow experiment `cashlog33-mps-specialists`, run `meal2-mps-target95-v1`

## Historical UECFood experiments

Historical standalone training target:

- Model: `mobilenetv4_conv_small`
- Starting weights: `models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors`
- Dataset: `data/processed/classification/uecfood256/UECFOOD256`
- Classes: 256, mapped from `category.txt`
- Device for active document-aligned run: Apple MPS (`--device mps`, launched
  outside the sandbox so PyTorch can access Metal)

Completed run:

```bash
.venv/bin/python scripts/train_uecfood256_mobilenetv4.py \
  --epochs 1 \
  --batch-size 32 \
  --max-samples-per-class 10 \
  --log-interval 20
```

Result:

| Split | Loss | Top-1 | Top-5 |
| --- | ---: | ---: | ---: |
| train | 5.478942 | 2.4414 | 7.3730 |
| val | 5.045927 | 7.6172 | 21.6797 |

Artifacts:

- `checkpoints/uecfood256_mobilenetv4/best.pt`
- `checkpoints/uecfood256_mobilenetv4/last.pt`
- `checkpoints/uecfood256_mobilenetv4/labels.json`
- `checkpoints/uecfood256_mobilenetv4/metrics.csv`

Historical full UECFood256 training command:

```bash
.venv/bin/python scripts/train_uecfood256_mobilenetv4.py \
  --epochs 5 \
  --batch-size 32 \
  --log-interval 50
```

Historical target-90% run:

```bash
bash scripts/train_until_90_uecfood256.sh
```

This uses bbox crop, RandAugment, random erasing, mixup, label smoothing,
one frozen-backbone warmup epoch, and stops early when validation top-1 reaches
90%.

Historical long-run target:

```bash
bash scripts/train_until_90_plateau_uecfood256.sh
```

This uses all UECFood256 images, bbox crop, RandAugment, random erasing,
mixup, and label smoothing. It keeps training until validation top-1 reaches
90% and then stops only after 100 epochs without a further improvement of at
least 0.01 percentage points.

The historical long run was launched detached:

```bash
.venv/bin/python scripts/launch_training.py uecfood256_target90_plateau100
```

Status files:

- Log: `logs/uecfood256_target90_plateau100.log`
- PID: `logs/uecfood256_target90_plateau100.pid`
- Output: `checkpoints/uecfood256_mobilenetv4_target90_plateau100/`

Check progress:

```bash
tail -f logs/uecfood256_target90_plateau100.log
cat checkpoints/uecfood256_mobilenetv4_target90_plateau100/metrics.csv
```

Monitor progress/restarts:

```bash
tail -f logs/uecfood256_target90_plateau100.monitor.log
```

In this CPU-only session, the small run processed 2,560 images in 558.5 seconds.
The full dataset has 31,910 images, so one full epoch is expected to take roughly
1.8 to 2.0 hours on the same runtime. If MPS becomes available in a normal local
terminal, use `--device mps`.

Detection/segmentation training is not started yet because the downloaded assets
do not currently include a usable detection dataset. `ABO listings` contains
metadata only, and `UECFood256` is suitable for classification.

## Historical Cashlog B-plan alignment

The attached Cashlog B-plan document defines the MVP success metric at the
expense-category level:

- Top-1 Cashlog category accuracy: 70%+
- Top-3 Cashlog category accuracy: 90%+
- MVP pipeline: YOLO/YOLO11n detection -> crop classification or CLIP/VLM ->
  rule-based category mapper

The previous long run targeted 90% top-1 on 256 UECFood fine-grained food
classes. That is not the document metric, so it was stopped.

Historical document-aligned local training:

```bash
.venv/bin/python scripts/train_cashlog_category_from_uecfood.py \
  --output-dir checkpoints/cashlog_category_uecfood_mobilenetv4 \
  --epochs 20 \
  --batch-size 32 \
  --log-interval 50
```

Historical MPS-accelerated full run:

```bash
.venv/bin/python scripts/launch_training.py cashlog_category_uecfood_mps
```

Status files:

- Log: `logs/cashlog_category_uecfood_mps.log`
- PID: `logs/cashlog_category_uecfood_mps.pid`
- Output: `checkpoints/cashlog_category_uecfood_mps/`

This job uses all currently mapped UECFood256 images, bbox crop, RandAugment,
weighted sampling, class-weighted cross entropy, MobileNetV4 ImageNet weights,
30 epochs, batch size 32, and explicit `--device mps`.

## Legacy four-leaf UECFood pipeline

This section documents the first image-only experiment for reproducibility. It
is not the current CashLog model and it must not be used as evidence for the
33-leaf service.

The training labels now follow the Cashlog category tree in
`configs/cashlog/category_tree.json`. The flat leaf list used by the classifier
head lives in `configs/cashlog/categories.json`.

With the datasets currently downloaded, supervised image training coverage is
still limited to food-like UECFood256 images. The UECFood mapping rules in
`configs/cashlog/uecfood_category_overrides.json` currently produce these
trainable Cashlog leaf labels:

- `meal_grocery` (`식재료`)
- `meal_dining` (`외식·배달`)
- `meal_cafe` (`카페·디저트`)
- `meal_drink` (`술·음료`)

The script automatically trains only leaf labels that have at least
`--min-samples-per-leaf` samples. More categories can be enabled by adding
source datasets and extending the mapping rules, without changing the model
head code.

Current leaf-label smoke command:

```bash
.venv/bin/python scripts/train_cashlog_category_from_uecfood.py \
  --arch mobilenetv4_conv_small \
  --output-dir checkpoints/cashlog_leaf_smoke \
  --epochs 1 \
  --batch-size 4 \
  --max-samples-per-uec-class 2 \
  --log-interval 1 \
  --disable-mlflow
```

Smoke result from this command on CPU:

```text
labels=meal_grocery,meal_dining,meal_cafe,meal_drink
train top1=44.21 top3=77.08
val top1=6.49 top3=20.78
```

This smoke run is intentionally tiny and only proves the leaf-label pipeline,
artifact generation, and metric path. It is not a model-quality benchmark.

## Cashlog ensemble model

The Cashlog classifier supports a weighted-logit ensemble. Each member is
trained as a normal checkpoint, then a manifest makes the group load as one
classifier in the API.

Train candidate backbones:

```bash
.venv/bin/python scripts/train_cashlog_category_from_uecfood.py \
  --arch mobilenetv4_conv_small \
  --output-dir checkpoints/cashlog_leaf_mobilenetv4 \
  --epochs 30 \
  --batch-size 32

.venv/bin/python scripts/train_cashlog_category_from_uecfood.py \
  --arch efficientnet_b0 \
  --pretrained \
  --output-dir checkpoints/cashlog_leaf_efficientnet_b0 \
  --epochs 30 \
  --batch-size 32

.venv/bin/python scripts/train_cashlog_category_from_uecfood.py \
  --arch convnext_tiny \
  --pretrained \
  --output-dir checkpoints/cashlog_leaf_convnext_tiny \
  --epochs 30 \
  --batch-size 32
```

Build an ensemble manifest. If explicit `--weights` are omitted, the script
uses each checkpoint's `best_top1` or `val_top1` metric as its weight.

```bash
.venv/bin/python scripts/build_cashlog_ensemble.py \
  --output configs/cashlog/ensemble.json \
  checkpoints/cashlog_leaf_mobilenetv4/best.pt \
  checkpoints/cashlog_leaf_efficientnet_b0/best.pt \
  checkpoints/cashlog_leaf_convnext_tiny/best.pt
```

Serve or predict with the ensemble:

```bash
export CATAI_CASHLOG_ENSEMBLE_CONFIG=configs/cashlog/ensemble.json
catai-serve-cashlog

catai-predict-cashlog path/to/image.jpg \
  --ensemble-config configs/cashlog/ensemble.json \
  --pretty
```

Airflow can train any one backbone by passing DAG conf:

```json
{
  "arch": "efficientnet_b0",
  "pretrained": true,
  "output_dir": "checkpoints/cashlog_leaf_efficientnet_b0",
  "epochs": 30,
  "batch_size": 32
}
```

## CashLog 33-leaf implementation details

The selected pipeline trains and evaluates all 33 configured leaves. It combines
SigLIP2 visual evidence, a trained visual head, Korean RapidOCR, a text classifier,
and the CashLog domain lexicon. Architecture, data provenance, production gates,
and operations are maintained in:

- `ml_docs/CASHLOG33_MODEL_DESIGN.md`
- `ml_docs/CASHLOG33_DATA_CARD.md`
- `ml_docs/CASHLOG33_OPERATIONS.md`
- `ml_docs/CASHLOG33_RUN_LOG.md`

Run the full isolated candidate workflow with Airflow DAG
`cashlog33_training_pipeline`. Candidate artifacts are written under
`checkpoints/cashlog33/airflow_latest`; the DAG does not overwrite the serving
configuration.

## Docker, Airflow, MLflow, Jenkins

Build the training image:

```bash
docker compose build trainer
```

Run MLflow locally:

```bash
docker compose up mlflow
```

Run a Docker training job that logs to MLflow:

```bash
docker compose --profile train run --rm trainer
```

Run Airflow and trigger the DAG named `cashlog33_training_pipeline` from the
Airflow UI or REST API:

```bash
docker compose up airflow mlflow
```

Validate DAG loading:

```bash
docker compose exec airflow airflow dags list
docker compose exec airflow airflow dags list-import-errors
```

Useful local URLs:

- MLflow: `http://127.0.0.1:5500`
- Airflow: `http://127.0.0.1:8080` (`admin` / `admin`, local bootstrap only)
- Jenkins: `http://127.0.0.1:8081`

Jenkins automation is defined in `Jenkinsfile`. Configure a username/password
credential with ID `airflow-local-basic`; the job triggers Airflow, monitors the
run to completion, and archives the candidate decision. Jenkins cannot overwrite
the serving configuration or access the Docker socket.
