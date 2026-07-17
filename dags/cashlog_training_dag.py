from __future__ import annotations

from datetime import datetime

from airflow import DAG
from airflow.operators.bash import BashOperator


DEFAULT_ARGS = {
    "owner": "catai",
    "depends_on_past": False,
    "retries": 0,
}

with DAG(
    dag_id="cashlog_leaf_training",
    default_args=DEFAULT_ARGS,
    description="Train Cashlog leaf-category image classifier and log artifacts to MLflow.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["catai", "cashlog", "mlflow"],
) as dag:
    validate_inputs = BashOperator(
        task_id="validate_inputs",
        cwd="/workspace",
        bash_command="""
        set -euo pipefail
        test -d "${CATAI_DATASET_ROOT:-data/processed/classification/uecfood256/UECFOOD256}"
        test -f "${CATAI_CATEGORIES:-configs/cashlog/categories.json}"
        test -f "${CATAI_OVERRIDES:-configs/cashlog/uecfood_category_overrides.json}"
        if [ -n "${CATAI_WEIGHTS:-models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors}" ]; then
          test -f "${CATAI_WEIGHTS:-models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors}"
        fi
        """,
    )

    train_model = BashOperator(
        task_id="train_model",
        cwd="/workspace",
        bash_command="""
        set -euo pipefail
        OUTPUT_DIR_VALUE="{{ dag_run.conf.get('output_dir', '') if dag_run else '' }}"
        TRAIN_EPOCHS="{{ dag_run.conf.get('epochs', '') if dag_run else '' }}"
        TRAIN_BATCH_SIZE="{{ dag_run.conf.get('batch_size', '') if dag_run else '' }}"
        TRAIN_LOG_INTERVAL="{{ dag_run.conf.get('log_interval', '') if dag_run else '' }}"
        MAX_SAMPLES_VALUE="{{ dag_run.conf.get('max_samples_per_uec_class', '') if dag_run else '' }}"
        RESUME_VALUE="{{ dag_run.conf.get('resume_checkpoint', '') if dag_run else '' }}"
        ARCH_VALUE="{{ dag_run.conf.get('arch', '') if dag_run else '' }}"
        WEIGHTS_VALUE="{{ dag_run.conf.get('weights', '') if dag_run else '' }}"
        PRETRAINED_VALUE="{{ dag_run.conf.get('pretrained', '') if dag_run else '' }}"

        if [ -z "$OUTPUT_DIR_VALUE" ]; then
          OUTPUT_DIR_VALUE="${CATAI_OUTPUT_DIR:-checkpoints/cashlog_leaf_airflow}"
        fi
        if [ -z "$TRAIN_EPOCHS" ]; then
          TRAIN_EPOCHS="${CATAI_TRAIN_EPOCHS:-30}"
        fi
        if [ -z "$TRAIN_BATCH_SIZE" ]; then
          TRAIN_BATCH_SIZE="${CATAI_TRAIN_BATCH_SIZE:-32}"
        fi
        if [ -z "$TRAIN_LOG_INTERVAL" ]; then
          TRAIN_LOG_INTERVAL="${CATAI_LOG_INTERVAL:-50}"
        fi
        if [ -z "$MAX_SAMPLES_VALUE" ]; then
          MAX_SAMPLES_VALUE="${CATAI_MAX_SAMPLES_PER_UEC_CLASS:-}"
        fi
        if [ -z "$RESUME_VALUE" ]; then
          RESUME_VALUE="${CATAI_RESUME_CHECKPOINT:-}"
        fi
        if [ -z "$ARCH_VALUE" ]; then
          ARCH_VALUE="${CATAI_MODEL_ARCH:-mobilenetv4_conv_small}"
        fi
        if [ -z "$WEIGHTS_VALUE" ]; then
          if [ -n "${CATAI_WEIGHTS:-}" ]; then
            WEIGHTS_VALUE="$CATAI_WEIGHTS"
          elif [ "$ARCH_VALUE" = "mobilenetv4" ] || [ "$ARCH_VALUE" = "mobilenetv4_conv_small" ]; then
            WEIGHTS_VALUE="models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors"
          else
            WEIGHTS_VALUE=""
          fi
        fi
        if [ -z "$PRETRAINED_VALUE" ]; then
          PRETRAINED_VALUE="${CATAI_PRETRAINED:-false}"
        fi

        MAX_SAMPLES_ARGS=""
        if [ -n "$MAX_SAMPLES_VALUE" ]; then
          MAX_SAMPLES_ARGS="--max-samples-per-uec-class $MAX_SAMPLES_VALUE"
        fi
        RESUME_ARGS=""
        if [ -n "$RESUME_VALUE" ]; then
          RESUME_ARGS="--resume $RESUME_VALUE"
        fi
        WEIGHTS_ARGS=""
        if [ -n "$WEIGHTS_VALUE" ]; then
          WEIGHTS_ARGS="--weights $WEIGHTS_VALUE"
        fi
        PRETRAINED_ARGS=""
        if [ "$PRETRAINED_VALUE" = "true" ] || [ "$PRETRAINED_VALUE" = "True" ] || [ "$PRETRAINED_VALUE" = "1" ]; then
          PRETRAINED_ARGS="--pretrained"
        fi

        python scripts/train_cashlog_category_from_uecfood.py \
          --dataset-root "${CATAI_DATASET_ROOT:-data/processed/classification/uecfood256/UECFOOD256}" \
          --arch "$ARCH_VALUE" \
          --categories "${CATAI_CATEGORIES:-configs/cashlog/categories.json}" \
          --overrides "${CATAI_OVERRIDES:-configs/cashlog/uecfood_category_overrides.json}" \
          --output-dir "$OUTPUT_DIR_VALUE" \
          --epochs "$TRAIN_EPOCHS" \
          --batch-size "$TRAIN_BATCH_SIZE" \
          --log-interval "$TRAIN_LOG_INTERVAL" \
          --mlflow-tracking-uri "${MLFLOW_TRACKING_URI:-http://mlflow:5000}" \
          --mlflow-experiment "${MLFLOW_EXPERIMENT_NAME:-catai-cashlog-category}" \
          $WEIGHTS_ARGS \
          $PRETRAINED_ARGS \
          $MAX_SAMPLES_ARGS \
          $RESUME_ARGS
        """,
    )

    summarize_artifacts = BashOperator(
        task_id="summarize_artifacts",
        cwd="/workspace",
        bash_command="""
        set -euo pipefail
        OUTPUT_DIR="{{ dag_run.conf.get('output_dir', '') if dag_run else '' }}"
        if [ -z "$OUTPUT_DIR" ]; then
          OUTPUT_DIR="${CATAI_OUTPUT_DIR:-checkpoints/cashlog_leaf_airflow}"
        fi
        export CATAI_OUTPUT_DIR="$OUTPUT_DIR"
        test -f "$OUTPUT_DIR/best.pt"
        test -f "$OUTPUT_DIR/labels.json"
        test -f "$OUTPUT_DIR/metrics.csv"
        python - <<'PY'
import csv
import json
import os
from pathlib import Path

output_dir = Path(os.getenv("CATAI_OUTPUT_DIR", "checkpoints/cashlog_leaf_airflow"))
labels = json.loads((output_dir / "labels.json").read_text())
with (output_dir / "metrics.csv").open() as f:
    rows = list(csv.DictReader(f))
last = rows[-1] if rows else {}
print("trained_leaf_ids=", [label["id"] for label in labels])
print("last_metrics=", last)
print("artifacts=", sorted(path.name for path in output_dir.iterdir()))
PY
        """,
    )

    validate_inputs >> train_model >> summarize_artifacts
