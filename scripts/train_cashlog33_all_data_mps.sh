#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OUTPUT_DIR="checkpoints/cashlog33/meal2_uecfood_all_v1"

.venv/bin/python scripts/train_cashlog_category_from_uecfood.py \
  --device mps \
  --trainable-leaf-ids meal_dining meal_cafe \
  --output-dir "$OUTPUT_DIR" \
  --init-checkpoint checkpoints/cashlog_meal2_mps_target95/best.pt \
  --epochs 8 \
  --batch-size 64 \
  --lr 0.00002 \
  --head-lr 0.0001 \
  --weight-decay 0.0001 \
  --label-smoothing 0.05 \
  --val-ratio 0.15 \
  --num-workers 0 \
  --no-balanced-sampler \
  --class-weight-power 0.5 \
  --target-top1 95 \
  --require-target \
  --stop-on-target \
  --minimum-epochs 1 \
  --early-stopping-patience 3 \
  --mlflow-tracking-uri http://127.0.0.1:5500 \
  --mlflow-experiment cashlog33-all-data-mps \
  --mlflow-run-name meal2-uecfood-all-v1-warm-start \
  --log-interval 50

.venv/bin/python scripts/evaluate_cashlog_meal_specialist.py \
  --checkpoint "$OUTPUT_DIR/best.pt" \
  --labels "$OUTPUT_DIR/labels.json" \
  --output-dir "$OUTPUT_DIR/evaluation" \
  --device mps \
  --batch-size 128 \
  --mlflow-tracking-uri http://127.0.0.1:5500 \
  --mlflow-experiment cashlog33-all-data-mps \
  --mlflow-run-name meal2-uecfood-all-v1-evaluation

.venv/bin/python -c '
import json
from pathlib import Path
metrics = json.loads(
    Path("checkpoints/cashlog33/meal2_uecfood_all_v1/evaluation/metrics.json").read_text()
)
assert metrics["top1_accuracy"] >= 0.95
assert metrics["minimum_leaf_recall"] >= 0.60
print("meal_specialist_gate=passed")
'
