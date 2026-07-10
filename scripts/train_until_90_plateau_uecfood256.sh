#!/usr/bin/env bash
set -euo pipefail

.venv/bin/python scripts/train_uecfood256_mobilenetv4.py \
  --output-dir checkpoints/uecfood256_mobilenetv4_target90_plateau100 \
  --epochs 300 \
  --batch-size 32 \
  --lr 0.0003 \
  --weight-decay 0.0001 \
  --label-smoothing 0.1 \
  --bbox-padding 0.1 \
  --random-erasing 0.15 \
  --mixup-alpha 0.1 \
  --freeze-backbone-epochs 1 \
  --target-top1 90 \
  --patience-after-target 100 \
  --min-delta 0.01 \
  --log-interval 50
