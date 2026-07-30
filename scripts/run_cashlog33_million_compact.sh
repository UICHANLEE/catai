#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-$ROOT/.venv/bin/python}"
MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-http://127.0.0.1:5500}"
TRAIN_DEVICE="${CATAI_TRAIN_DEVICE:-mps}"
TRAIN_BATCH_SIZE="${CATAI_TRAIN_BATCH_SIZE:-128}"
TRAIN_WORKERS="${CATAI_TRAIN_WORKERS:-4}"
POSTTRAIN_ONLY="${CATAI_POSTTRAIN_ONLY:-false}"
LOG_DIR="$ROOT/logs/cashlog33-million-v1"
mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  printf '{"단계":"%s","상태":"시작","시각":"%s"}\n' \
    "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/pipeline_ko.jsonl"
  "$@" 2>&1 | tee "$LOG_DIR/$name.log"
  printf '{"단계":"%s","상태":"완료","시각":"%s"}\n' \
    "$name" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/pipeline_ko.jsonl"
}

if [[ "$POSTTRAIN_ONLY" != "true" ]]; then
  run_step 데이터셋_구축 \
    "$PYTHON" scripts/build_cashlog33_million_dataset.py

  run_step MPS_학습 \
    "$PYTHON" scripts/train_cashlog33_million_mobilenet.py \
      --device "$TRAIN_DEVICE" \
      --batch-size "$TRAIN_BATCH_SIZE" \
      --num-workers "$TRAIN_WORKERS" \
      --mlflow-tracking-uri "$MLFLOW_TRACKING_URI"
else
  test -f "$ROOT/checkpoints/cashlog33/mobilenet_million_v1/best.pt"
  test -f "$ROOT/checkpoints/cashlog33/mobilenet_million_v1/mlflow_run.json"
  printf '{"단계":"학습_재사용","상태":"완료","시각":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/pipeline_ko.jsonl"
fi

run_step ONNX_INT8_내보내기 \
  "$PYTHON" scripts/export_quantize_cashlog33_mobilenet.py

run_step 배포_자산_스테이징 \
  "$PYTHON" scripts/stage_cashlog33_compact_artifact.py

run_step 후보_설정_생성 \
  "$PYTHON" scripts/build_cashlog33_compact_serving_config.py

if ! "$PYTHON" -c \
  "import hashlib,json,pathlib; p=pathlib.Path('configs/cashlog/hybrid.serving.json'); r=pathlib.Path('reports/cashlog33/million_v1/current_serving_visual_baseline.json'); assert r.is_file() and json.load(r.open())['config_sha256'] == hashlib.sha256(p.read_bytes()).hexdigest()"; then
  run_step 현재_서빙_기준선 \
    "$PYTHON" scripts/evaluate_cashlog33_serving_vision.py \
      --device "$TRAIN_DEVICE"
fi

run_step 후보_actual_회귀평가 \
  "$PYTHON" scripts/evaluate_cashlog33_hybrid.py \
    --config configs/cashlog/hybrid.million-int8-candidate.json \
    --manifest data/raw/cashlog33/actual/manifest.jsonl \
    --output-dir reports/cashlog33/million_v1/candidate_actual \
    --device "$TRAIN_DEVICE" \
    --dataset-scope real_cashlog_holdout \
    --disable-mlflow

run_step 후보_비교 \
  "$PYTHON" scripts/evaluate_cashlog33_compact.py

if "$PYTHON" -c \
  "import json; assert json.load(open('reports/cashlog33/million_v1/compact_comparison.json'))['promotion_gate_passed']"; then
  run_step 모델_교체 \
    "$PYTHON" scripts/promote_cashlog33_compact.py
else
  printf '{"단계":"모델_교체","상태":"보류","사유":"성능_게이트_미통과","시각":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_DIR/pipeline_ko.jsonl"
fi
