#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

.venv/bin/python scripts/generate_cashlog33_ocr_category_dataset.py \
  --output-dir data/processed/cashlog33/ocr_category/v2_500k \
  --dataset-version v2 \
  --train-per-leaf 15200 \
  --validation-per-leaf 100 \
  --test-per-leaf 100 \
  --workers 8 \
  --seed 500027

.venv/bin/python scripts/repair_cashlog33_ocr_boxes.py \
  --dataset-dir data/processed/cashlog33/ocr_category/v2_500k

.venv/bin/python scripts/build_cashlog33_ocr_text_manifest.py \
  --dataset-dir data/processed/cashlog33/ocr_category/v2_500k \
  --output data/processed/cashlog33/ocr_text/v2_500k/manifest.jsonl

.venv/bin/python scripts/merge_cashlog33_text_manifests.py \
  --manifest data/processed/cashlog33/text/all_v1/manifest.jsonl \
  --manifest data/processed/cashlog33/ocr_text/v2_500k/manifest.jsonl \
  --output data/processed/cashlog33/text/all_500k_v2/manifest.jsonl

.venv/bin/python scripts/validate_cashlog33_ocr_category_dataset.py \
  --synthetic-dir data/processed/cashlog33/ocr_category/v2_500k \
  --minimum-train-images 500000 \
  --minimum-train-per-leaf 15000 \
  --skip-recognition \
  --report reports/cashlog33/data/ocr_category_v2_500k_validation.json \
  --verify-all-hashes
