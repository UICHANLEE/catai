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

RUNTIME_ENV = {
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
    "TOKENIZERS_PARALLELISM": "false",
}


with DAG(
    dag_id="cashlog_feedback_curation",
    default_args=DEFAULT_ARGS,
    description="Export, de-identify, validate, and rank reviewed CashLog feedback.",
    schedule="@daily",
    start_date=datetime(2026, 7, 17),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=30),
    tags=["catai", "cashlog33", "feedback", "privacy", "mlflow"],
) as dag:
    validate_secrets = BashOperator(
        task_id="validate_secrets",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command="""
        set -euo pipefail
        test -n "${SUPABASE_URL:-}"
        test -n "${SUPABASE_SERVICE_ROLE_KEY:-}"
        test -n "${CASHLOG_FEEDBACK_HMAC_KEY:-}"
        test "${#CASHLOG_FEEDBACK_HMAC_KEY}" -ge 32
        test -f configs/cashlog/categories.json
        """,
    )

    export_feedback = BashOperator(
        task_id="export_feedback",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        do_xcom_push=True,
        bash_command="""
        set -euo pipefail
        RELEASE_DIR="/workspace/data/feedback/releases/{{ ts_nodash }}"
        python scripts/export_cashlog_feedback.py \
          --output-dir "$RELEASE_DIR" \
          --status all \
          --reuse-existing
        """,
    )

    curate_feedback = BashOperator(
        task_id="curate_feedback",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command="""
        set -euo pipefail
        RELEASE_DIR="{{ ti.xcom_pull(task_ids='export_feedback') }}"
        python scripts/curate_cashlog_feedback.py \
          --events "$RELEASE_DIR/events.jsonl" \
          --output-dir "$RELEASE_DIR/curated" \
          --minimum-images-per-leaf 10 \
          --reuse-existing \
          --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
          --mlflow-run-name "cashlog-feedback-{{ ts_nodash }}"
        """,
    )

    publish_gate = BashOperator(
        task_id="publish_training_gate",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command="""
        set -euo pipefail
        RELEASE_DIR="{{ ti.xcom_pull(task_ids='export_feedback') }}"
        SUMMARY="$RELEASE_DIR/curated/curation_summary.json"
        test -f "$SUMMARY"
        python -c 'import json,sys; p=json.load(open(sys.argv[1])); print("ready_for_training=", p["ready_for_training"]); print("approved_image_candidates=", p["approved_image_candidates"]); assert p["auto_training_allowed"] is False' "$SUMMARY"
        """,
    )

    validate_secrets >> export_feedback >> curate_feedback >> publish_gate
