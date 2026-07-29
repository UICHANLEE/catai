#!/usr/bin/env python3
"""Launch long-running training detached from the current terminal."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

JOBS = {
    "uecfood256_target80_plateau100": [
        "bash",
        "scripts/train_until_80_plateau_uecfood256.sh",
    ],
    "uecfood256_target90_plateau100": [
        "bash",
        "scripts/train_until_90_plateau_uecfood256.sh",
    ],
    "monitor_uecfood256_target90_plateau100": [
        ".venv/bin/python",
        "scripts/monitor_uecfood256_training.py",
    ],
    "cashlog_category_uecfood_mps": [
        ".venv/bin/python",
        "scripts/train_cashlog_category_from_uecfood.py",
        "--device",
        "mps",
        "--output-dir",
        "checkpoints/cashlog_category_uecfood_mps",
        "--epochs",
        "30",
        "--batch-size",
        "32",
        "--log-interval",
        "50",
    ],
    "cashlog_meal2_mps_target95": [
        ".venv/bin/python",
        "scripts/train_cashlog_category_from_uecfood.py",
        "--device",
        "mps",
        "--trainable-leaf-ids",
        "meal_dining",
        "meal_cafe",
        "--init-checkpoint",
        "checkpoints/cashlog_category_uecfood_mps/best.pt",
        "--init-label-map",
        "food=meal_dining",
        "cafe_snack=meal_cafe",
        "--output-dir",
        "checkpoints/cashlog_meal2_mps_target95",
        "--epochs",
        "12",
        "--batch-size",
        "64",
        "--lr",
        "0.00005",
        "--target-top1",
        "95",
        "--require-target",
        "--stop-on-target",
        "--minimum-epochs",
        "2",
        "--early-stopping-patience",
        "4",
        "--log-interval",
        "50",
    ],
    "cashlog_meal2_all_mps_target95": [
        "bash",
        "scripts/train_cashlog33_all_data_mps.sh",
    ],
    "cashlog_ocr_category_112k": [
        "bash",
        "scripts/build_cashlog33_ocr_dataset.sh",
    ],
    "cashlog_ocr_category_500k": [
        "bash",
        "scripts/build_cashlog33_500k_dataset.sh",
    ],
    "cashlog_text_500k_alpha3e5": [
        ".venv/bin/python",
        "scripts/train_cashlog33_text.py",
        "--manifest",
        "data/processed/cashlog33/text/all_500k_v2/manifest.jsonl",
        "--output-dir",
        "checkpoints/cashlog33/text_500k_v2_alpha3e5",
        "--alpha",
        "0.00003",
        "--max-iter",
        "100",
        "--mlflow-tracking-uri",
        "http://127.0.0.1:5500",
        "--mlflow-experiment",
        "cashlog33-500k",
        "--mlflow-run-name",
        "text-500k-alpha3e5",
    ],
    "cashlog_text_500k_alpha1e5": [
        ".venv/bin/python",
        "scripts/train_cashlog33_text.py",
        "--manifest",
        "data/processed/cashlog33/text/all_500k_v2/manifest.jsonl",
        "--output-dir",
        "checkpoints/cashlog33/text_500k_v2_alpha1e5",
        "--alpha",
        "0.00001",
        "--max-iter",
        "100",
        "--mlflow-tracking-uri",
        "http://127.0.0.1:5500",
        "--mlflow-experiment",
        "cashlog33-500k",
        "--mlflow-run-name",
        "text-500k-alpha1e5",
    ],
    "cashlog_originals_v2_mps": [
        "bash",
        "scripts/train_cashlog33_originals_mps.sh",
    ],
    "cashlog_leaf_uecfood_auto": [
        ".venv/bin/python",
        "scripts/train_cashlog_category_from_uecfood.py",
        "--output-dir",
        "checkpoints/cashlog_leaf_uecfood_auto",
        "--epochs",
        "30",
        "--batch-size",
        "32",
        "--log-interval",
        "50",
    ],
    "cashlog_leaf_mobilenetv4": [
        ".venv/bin/python",
        "scripts/train_cashlog_category_from_uecfood.py",
        "--arch",
        "mobilenetv4_conv_small",
        "--output-dir",
        "checkpoints/cashlog_leaf_mobilenetv4",
        "--epochs",
        "30",
        "--batch-size",
        "32",
        "--log-interval",
        "50",
    ],
    "cashlog_leaf_efficientnet_b0": [
        ".venv/bin/python",
        "scripts/train_cashlog_category_from_uecfood.py",
        "--arch",
        "efficientnet_b0",
        "--pretrained",
        "--output-dir",
        "checkpoints/cashlog_leaf_efficientnet_b0",
        "--epochs",
        "30",
        "--batch-size",
        "32",
        "--log-interval",
        "50",
    ],
    "cashlog_leaf_convnext_tiny": [
        ".venv/bin/python",
        "scripts/train_cashlog_category_from_uecfood.py",
        "--arch",
        "convnext_tiny",
        "--pretrained",
        "--output-dir",
        "checkpoints/cashlog_leaf_convnext_tiny",
        "--epochs",
        "30",
        "--batch-size",
        "32",
        "--log-interval",
        "50",
    ],
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in JOBS:
        names = ", ".join(sorted(JOBS))
        raise SystemExit(f"usage: {sys.argv[0]} <job>\navailable: {names}")

    job = sys.argv[1]
    log_path = LOG_DIR / f"{job}.log"
    pid_path = LOG_DIR / f"{job}.pid"

    log_file = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        JOBS[job],
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path.write_text(f"{process.pid}\n")
    print(f"started {job} pid={process.pid}")
    print(f"log={log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
