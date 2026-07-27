from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


DEFAULT_ARGS = {
    "owner": "catai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
ENV = {
    "OMP_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4",
    "MKL_NUM_THREADS": "4",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}
TEXT_MANIFEST = "data/processed/cashlog33/text/all_500k_v2/manifest.jsonl"
CANDIDATE_DIR = "checkpoints/cashlog33/text_500k_v2_alpha1e5"
REPORT_DIR = "reports/cashlog33/500k"


with DAG(
    dag_id="cashlog33_500k_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Build and evaluate the leakage-checked 500k CashLog OCR-text candidate.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=8),
    tags=["catai", "cashlog33", "500k", "mlflow", "promotion-gated"],
) as dag:
    validate_inputs = BashOperator(
        task_id="validate_inputs",
        cwd="/workspace",
        env=ENV,
        append_env=True,
        bash_command="""
        set -euo pipefail
        test -f configs/cashlog/categories.json
        test -f data/processed/cashlog33/text/all_v1/manifest.jsonl
        test -f data/raw/cashlog33/cord_v2/manifest.jsonl
        test -f checkpoints/cashlog33/text_all_data_v1/text_model.joblib
        """,
    )

    build_500k_dataset = BashOperator(
        task_id="build_500k_dataset",
        cwd="/workspace",
        env=ENV,
        append_env=True,
        execution_timeout=timedelta(hours=6),
        bash_command="bash scripts/build_cashlog33_500k_dataset.sh",
    )

    evaluate_baseline = BashOperator(
        task_id="evaluate_baseline_on_v2_holdout",
        cwd="/workspace",
        env=ENV,
        append_env=True,
        bash_command=f"""
        set -euo pipefail
        python scripts/evaluate_cashlog33_text_model.py \
          --model checkpoints/cashlog33/text_all_data_v1/text_model.joblib \
          --manifest {TEXT_MANIFEST} \
          --output {REPORT_DIR}/baseline_text_v2_holdout.json \
          --mlflow-tracking-uri http://mlflow:5000 \
          --mlflow-experiment cashlog33-500k \
          --mlflow-run-name baseline-text-v2-holdout
        """,
    )

    train_regularized = BashOperator(
        task_id="train_regularized_text_candidate",
        cwd="/workspace",
        env=ENV,
        append_env=True,
        execution_timeout=timedelta(hours=2),
        bash_command=f"""
        set -euo pipefail
        python scripts/train_cashlog33_text.py \
          --manifest {TEXT_MANIFEST} \
          --output-dir {CANDIDATE_DIR} \
          --alpha 0.00001 \
          --max-iter 100 \
          --mlflow-tracking-uri http://mlflow:5000 \
          --mlflow-experiment cashlog33-500k \
          --mlflow-run-name text-500k-alpha1e5
        """,
    )

    evaluate_candidate = BashOperator(
        task_id="evaluate_candidate_on_v2_holdout",
        cwd="/workspace",
        env=ENV,
        append_env=True,
        bash_command=f"""
        set -euo pipefail
        python scripts/evaluate_cashlog33_text_model.py \
          --model {CANDIDATE_DIR}/text_model.joblib \
          --manifest {TEXT_MANIFEST} \
          --output {REPORT_DIR}/candidate_text_v2_holdout.json \
          --mlflow-tracking-uri http://mlflow:5000 \
          --mlflow-experiment cashlog33-500k \
          --mlflow-run-name candidate-text-v2-holdout
        """,
    )

    validate_inputs >> build_500k_dataset
    build_500k_dataset >> [evaluate_baseline, train_regularized]
    train_regularized >> evaluate_candidate
