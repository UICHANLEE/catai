#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OPENIMAGES_DIR="data/raw/cashlog33/openimages_v7"
OUTPUT_DIR="checkpoints/cashlog33/vision_head_originals_v2"
REPORT_DIR="reports/cashlog33/originals_v2"
REPORT="$REPORT_DIR/vision_head_ensemble_comparison.json"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5500}"
export MLFLOW_ALLOW_FILE_STORE="${MLFLOW_ALLOW_FILE_STORE:-true}"

.venv/bin/python scripts/collect_cashlog_abo.py \
  --per-leaf 5000 \
  --max-dimension 512 \
  --min-dimension 96

.venv/bin/python scripts/audit_cashlog33_original_sources.py

.venv/bin/python scripts/build_cashlog33_original_train_manifest.py

.venv/bin/python scripts/score_cashlog33_candidates.py \
  --input-manifest "$OPENIMAGES_DIR/manifest.jsonl" \
  --output-manifest "$OPENIMAGES_DIR/scored_manifest_originals_v2.jsonl" \
  --device mps \
  --batch-size 32

.venv/bin/python scripts/train_cashlog33_vision_head.py \
  --manifest "$OPENIMAGES_DIR/manifest.jsonl" \
  --scored-manifest "$OPENIMAGES_DIR/scored_manifest_originals_v2.jsonl" \
  --additional-train-manifest \
    data/processed/cashlog33/training/originals_v2_additional/manifest.jsonl \
  --output-dir "$OUTPUT_DIR" \
  --device mps \
  --batch-size 32 \
  --class-weight balanced \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment cashlog33-original-data \
  --mlflow-run-name cashlog33-originals-v2-siglip2-head

.venv/bin/python scripts/tune_cashlog33_original_source_weight.py \
  --embedding-cache "$OUTPUT_DIR/embedding_cache.npz" \
  --metrics "$OUTPUT_DIR/metrics.json" \
  --baseline-head checkpoints/cashlog33/vision_head_all_data_v1/vision_head.joblib \
  --input-head "$OUTPUT_DIR/vision_head.joblib" \
  --output-head "$OUTPUT_DIR/vision_head_weighted.joblib" \
  --output-report "$REPORT_DIR/source_weight_tuning.json"

.venv/bin/python scripts/build_cashlog33_vision_ensemble.py \
  --baseline-head checkpoints/cashlog33/vision_head_all_data_v1/vision_head.joblib \
  --candidate-head "$OUTPUT_DIR/vision_head_weighted.joblib" \
  --embedding-cache "$OUTPUT_DIR/embedding_cache.npz" \
  --output-head "$OUTPUT_DIR/vision_head_ensemble.joblib" \
  --output-report "$REPORT_DIR/ensemble_selection.json"

.venv/bin/python scripts/compare_cashlog33_vision_heads.py \
  --baseline-head checkpoints/cashlog33/vision_head_all_data_v1/vision_head.joblib \
  --candidate-head "$OUTPUT_DIR/vision_head_ensemble.joblib" \
  --embedding-cache "$OUTPUT_DIR/embedding_cache.npz" \
  --output "$REPORT"

.venv/bin/python scripts/log_cashlog33_originals_mlflow.py \
  --tracking-uri "$MLFLOW_TRACKING_URI" \
  --metrics "$OUTPUT_DIR/metrics.json" \
  --comparison "$REPORT" \
  --ensemble-selection "$REPORT_DIR/ensemble_selection.json" \
  --source-audit "$REPORT_DIR/source_audit.json" \
  --artifact "$OUTPUT_DIR/vision_head_ensemble.joblib"
