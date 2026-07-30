#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
OUTPUT_DIR="$ROOT/checkpoints/cashlog33/vision_head_siglip_expanded_v1"
REPORT_DIR="$ROOT/reports/cashlog33/siglip_expanded_v1"
LOG_DIR="$ROOT/logs/cashlog33-siglip-expanded-v1"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5500}"
mkdir -p "$OUTPUT_DIR" "$REPORT_DIR" "$LOG_DIR"

printf '{"단계":"동일_SigLIP2_재학습","상태":"시작","시각":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/pipeline_ko.jsonl"

"$PYTHON" scripts/train_cashlog33_vision_head.py \
  --manifest data/raw/cashlog33/openimages_v7/manifest.jsonl \
  --scored-manifest \
    data/raw/cashlog33/openimages_v7/scored_manifest_originals_v2.jsonl \
  --additional-train-manifest \
    data/raw/cashlog33/openimages_v7_train/manifest.jsonl \
  --base-embedding-cache \
    checkpoints/cashlog33/vision_head_originals_v2/embedding_cache.npz \
  --allow-cached-train-suffix \
  --additional-views original \
  --additional-shard-dir "$OUTPUT_DIR/embedding_shards" \
  --additional-rows-per-shard 4096 \
  --output-dir "$OUTPUT_DIR" \
  --device mps \
  --mps-float16 \
  --batch-size 128 \
  --class-weight balanced \
  --c-values 3.0 \
  --mlflow-tracking-uri "$MLFLOW_TRACKING_URI" \
  --mlflow-experiment cashlog33-same-siglip-expanded \
  --mlflow-run-name cashlog33-same-siglip-expanded-v1 \
  2>&1 | tee "$LOG_DIR/training.log"

printf '{"단계":"동일_SigLIP2_재학습","상태":"완료","시각":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/pipeline_ko.jsonl"

"$PYTHON" scripts/compare_cashlog33_vision_heads.py \
  --baseline-head \
    checkpoints/cashlog33/vision_head_all_data_v1/vision_head.joblib \
  --candidate-head "$OUTPUT_DIR/vision_head.joblib" \
  --embedding-cache "$OUTPUT_DIR/embedding_cache.npz" \
  --max-leaf-recall-regression 0.10 \
  --output "$REPORT_DIR/head_comparison.json" \
  2>&1 | tee "$LOG_DIR/head_comparison.log"

"$PYTHON" scripts/build_cashlog33_hybrid_config.py \
  --template configs/cashlog/hybrid.serving.json \
  --vision-head "$OUTPUT_DIR/vision_head.joblib" \
  --text-model checkpoints/cashlog33/text_all_data_v1/text_model.joblib \
  --meal-specialist-checkpoint checkpoints/cashlog_meal2_mps_target95/best.pt \
  --meal-specialist-labels checkpoints/cashlog_meal2_mps_target95/labels.json \
  --meal-specialist-blend-weight 0.65 \
  --model-version cashlog33-same-siglip-expanded-v1 \
  --output configs/cashlog/hybrid.siglip-expanded-candidate.json \
  2>&1 | tee "$LOG_DIR/candidate_config.log"

"$PYTHON" scripts/evaluate_cashlog33_serving_vision.py \
  --config configs/cashlog/hybrid.siglip-expanded-candidate.json \
  --output "$REPORT_DIR/candidate_serving_visual.json" \
  --device mps \
  2>&1 | tee "$LOG_DIR/candidate_serving_visual.log"

printf '{"단계":"동일_SigLIP2_후보평가","상태":"완료","시각":"%s"}\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/pipeline_ko.jsonl"
