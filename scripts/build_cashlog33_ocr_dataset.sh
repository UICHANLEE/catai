#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

.venv/bin/python scripts/collect_cashlog_cord.py
.venv/bin/python scripts/generate_cashlog33_ocr_category_dataset.py \
  --train-per-leaf 3200 \
  --validation-per-leaf 100 \
  --test-per-leaf 100 \
  --workers 8
.venv/bin/python scripts/export_cashlog33_ocr_recognition_crops.py \
  --train-limit 120000 \
  --validation-limit 5000 \
  --test-limit 5000
.venv/bin/python scripts/validate_cashlog33_ocr_category_dataset.py \
  --minimum-train-images 100000 \
  --minimum-train-per-leaf 3000 \
  --verify-all-hashes
