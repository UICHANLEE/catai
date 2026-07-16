from __future__ import annotations

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator


DEFAULT_ARGS = {
    "owner": "catai",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}

RUNTIME_ENV = {
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
    "MKL_NUM_THREADS": "2",
    "TOKENIZERS_PARALLELISM": "false",
    "MLFLOW_TRACKING_URI": "http://mlflow:5000",
}

TEXT_DIR = "data/processed/cashlog33/text/v2"
CANDIDATE_ROOT = "checkpoints/cashlog33/airflow_latest"
CANDIDATE_CONFIG = "configs/cashlog/hybrid.airflow-candidate.json"
CANDIDATE_REPORT = "reports/cashlog33/airflow_latest"


with DAG(
    dag_id="cashlog33_training_pipeline",
    default_args=DEFAULT_ARGS,
    description="Rebuild, evaluate, and gate the 33-leaf CashLog hybrid candidate.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=4),
    tags=["catai", "cashlog33", "mlflow", "promotion-gated"],
) as dag:
    validate_inputs = BashOperator(
        task_id="validate_inputs",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command="""
        set -euo pipefail
        test -f configs/cashlog/categories.json
        test -f configs/cashlog/leaf_semantics.json
        test -f configs/cashlog/ocr_lexicon.json
        test -f data/raw/cashlog33/text/us-bank-transactions-v2/transactions-synthetic.csv
        test -f data/raw/cashlog33/openimages_v7/manifest.jsonl
        test -f models/siglip2-base-patch16-224/model.safetensors
        test -f models/rapidocr/PP-OCRv6_det_small.onnx
        test -f models/rapidocr/ch_ppocr_mobile_v2.0_cls_mobile.onnx
        test -f models/rapidocr/korean_PP-OCRv5_rec_mobile.onnx
        python - <<'PY'
import json
from pathlib import Path
categories = json.loads(Path("configs/cashlog/categories.json").read_text())
semantics = json.loads(Path("configs/cashlog/leaf_semantics.json").read_text())
lexicon = json.loads(Path("configs/cashlog/ocr_lexicon.json").read_text())
ids = [row["id"] for row in categories]
assert len(ids) == len(set(ids)) == 33
assert set(ids) == set(semantics["leaves"]) == set(lexicon["leaves"])
assert semantics["taxonomy_version"] == lexicon["taxonomy_version"] == "13.33.1"
print("validated_leaf_count=33")
PY
        """,
    )

    build_text_dataset = BashOperator(
        task_id="build_text_dataset",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command=f"""
        set -euo pipefail
        python scripts/build_cashlog33_text_dataset.py \
          --output-dir {TEXT_DIR} \
          --max-source-per-leaf 1200 \
          --synthetic-per-leaf 480
        """,
    )

    score_visual_proxy = BashOperator(
        task_id="score_visual_proxy",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        execution_timeout=timedelta(hours=1),
        bash_command="""
        set -euo pipefail
        python scripts/score_cashlog33_candidates.py \
          --input-manifest data/raw/cashlog33/openimages_v7/manifest.jsonl \
          --output-manifest data/raw/cashlog33/openimages_v7/scored_manifest.jsonl \
          --model models/siglip2-base-patch16-224 \
          --device cpu \
          --batch-size 16 \
          --min-score 0.001 \
          --min-margin 0.0001
        """,
    )

    train_text = BashOperator(
        task_id="train_text",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command=f"""
        set -euo pipefail
        python scripts/train_cashlog33_text.py \
          --manifest {TEXT_DIR}/manifest.jsonl \
          --output-dir {CANDIDATE_ROOT}/text \
          --mlflow-tracking-uri http://mlflow:5000 \
          --mlflow-experiment cashlog33-hybrid-v2 \
          --mlflow-run-name cashlog33-airflow-text
        """,
    )

    train_vision_head = BashOperator(
        task_id="train_vision_head",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        execution_timeout=timedelta(hours=1),
        bash_command=f"""
        set -euo pipefail
        python scripts/train_cashlog33_vision_head.py \
          --manifest data/raw/cashlog33/openimages_v7/manifest.jsonl \
          --scored-manifest data/raw/cashlog33/openimages_v7/scored_manifest.jsonl \
          --vision-model models/siglip2-base-patch16-224 \
          --output-dir {CANDIDATE_ROOT}/vision \
          --device cpu \
          --batch-size 4 \
          --mlflow-tracking-uri http://mlflow:5000 \
          --mlflow-experiment cashlog33-hybrid-v2 \
          --mlflow-run-name cashlog33-airflow-vision-head
        """,
    )

    build_candidate_config = BashOperator(
        task_id="build_candidate_config",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command=f"""
        set -euo pipefail
        python scripts/build_cashlog33_hybrid_config.py \
          --template configs/cashlog/hybrid.serving.json \
          --vision-head {CANDIDATE_ROOT}/vision/vision_head.joblib \
          --text-model {CANDIDATE_ROOT}/text/text_model.joblib \
          --output {CANDIDATE_CONFIG} \
          --model-version cashlog33-hybrid-airflow-latest
        """,
    )

    generate_e2e_fixtures = BashOperator(
        task_id="generate_e2e_fixtures",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command="""
        set -euo pipefail
        python scripts/generate_cashlog33_e2e_fixtures.py \
          --output-dir data/processed/cashlog33/e2e_fixtures/v1 \
          --per-leaf 3
        """,
    )

    evaluate_candidate = BashOperator(
        task_id="evaluate_candidate",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        execution_timeout=timedelta(hours=1),
        bash_command=f"""
        set -euo pipefail
        python scripts/evaluate_cashlog33_hybrid.py \
          --config {CANDIDATE_CONFIG} \
          --manifest data/processed/cashlog33/e2e_fixtures/v1/manifest.jsonl \
          --output-dir {CANDIDATE_REPORT}/hybrid_e2e \
          --device cpu \
          --mlflow-tracking-uri http://mlflow:5000 \
          --mlflow-experiment cashlog33-hybrid-v2 \
          --mlflow-run-name cashlog33-airflow-hybrid-e2e
        """,
    )

    select_candidate = BashOperator(
        task_id="select_candidate",
        cwd="/workspace",
        env=RUNTIME_ENV,
        append_env=True,
        bash_command=f"""
        set -euo pipefail
        python scripts/select_cashlog33_candidate.py \
          --config {CANDIDATE_CONFIG} \
          --vision-metrics {CANDIDATE_ROOT}/vision/metrics.json \
          --text-metrics {CANDIDATE_ROOT}/text/metrics.json \
          --e2e-metrics {CANDIDATE_REPORT}/hybrid_e2e/metrics.json \
          --e2e-per-class {CANDIDATE_REPORT}/hybrid_e2e/per_class_metrics.csv \
          --output {CANDIDATE_REPORT}/model_selection.json
        """,
    )

    validate_inputs >> [build_text_dataset, score_visual_proxy, generate_e2e_fixtures]
    build_text_dataset >> train_text
    score_visual_proxy >> train_vision_head
    [train_text, train_vision_head] >> build_candidate_config
    [build_candidate_config, generate_e2e_fixtures] >> evaluate_candidate >> select_candidate
