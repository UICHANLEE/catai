#!/usr/bin/env python3
"""Evaluate RapidOCR and OCR-text category routing on the synthetic dataset."""

from __future__ import annotations

import argparse
import json
import statistics
import time
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from catai.cashlog_hybrid_classifier import CashlogHybridClassifier


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_MANIFEST = ROOT / "data/processed/cashlog33/ocr_category/v1/manifest.jsonl"
DEFAULT_OCR = ROOT / "data/processed/cashlog33/ocr_category/v1/ocr_annotations.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/cashlog33/data/ocr_category_v1_ocr_metrics.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return "".join(character for character in normalized if character.isalnum())


def edit_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1]
                    + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--ocr-annotations", type=Path, default=DEFAULT_OCR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--per-leaf", type=int, default=5)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--mlflow-tracking-uri")
    parser.add_argument("--mlflow-experiment", default="cashlog33-ocr-data")
    parser.add_argument("--mlflow-run-name", default="rapidocr-synthetic-quality")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.per_leaf <= 0:
        raise SystemExit("--per-leaf must be positive")
    manifests = load_jsonl(args.manifest)
    ocr_by_id = {
        str(row["sample_id"]): row for row in load_jsonl(args.ocr_annotations)
    }
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    for row in manifests:
        leaf_id = str(row["leaf_id"])
        if (
            row["split"] == args.split
            and selected_counts[leaf_id] < args.per_leaf
        ):
            selected.append(row)
            selected_counts[leaf_id] += 1
    if len(selected_counts) != 33:
        raise SystemExit(f"selected split covers {len(selected_counts)} leaves, expected 33")

    classifier = CashlogHybridClassifier.from_config(args.config, device=args.device)
    total_distance = 0
    total_characters = 0
    exact = 0
    category_correct = 0
    per_leaf: dict[str, Counter[str]] = defaultdict(Counter)
    latencies: list[float] = []
    for index, row in enumerate(selected, start=1):
        annotation = ocr_by_id[str(row["sample_id"])]
        expected_text = normalize_text(str(annotation["transcription"]))
        started = time.perf_counter()
        with Image.open(resolve_path(str(row["relative_path"]))) as image:
            image.load()
            result = classifier._extract_ocr(image.convert("RGB"))
        latency = time.perf_counter() - started
        predicted_text = normalize_text(str(result["text"]))
        distance = edit_distance(expected_text, predicted_text)
        expected_leaf = str(row["leaf_id"])
        text_scores = classifier._text_scores(str(result["text"]))
        predicted_leaf = max(text_scores, key=text_scores.get)
        total_distance += distance
        total_characters += max(1, len(expected_text))
        exact += int(expected_text == predicted_text)
        category_correct += int(predicted_leaf == expected_leaf)
        per_leaf[expected_leaf]["samples"] += 1
        per_leaf[expected_leaf]["category_correct"] += int(
            predicted_leaf == expected_leaf
        )
        latencies.append(latency)
        if index % 25 == 0 or index == len(selected):
            print(f"evaluated={index}/{len(selected)}", flush=True)

    metrics = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "scope": "synthetic OCR/category validation",
        "samples": len(selected),
        "leaf_count": len(selected_counts),
        "per_leaf_samples": args.per_leaf,
        "character_error_rate": total_distance / total_characters,
        "exact_transcription_rate": exact / len(selected),
        "ocr_text_category_top1": category_correct / len(selected),
        "latency_p50_seconds": statistics.median(latencies),
        "latency_p95_seconds": percentile(latencies, 0.95),
        "device": args.device,
        "per_leaf_category_accuracy": {
            leaf_id: counts["category_correct"] / counts["samples"]
            for leaf_id, counts in sorted(per_leaf.items())
        },
        "privacy": "OCR text and image contents are not stored in this report.",
        "scope_warning": (
            "Synthetic OCR metrics do not establish real receipt or production accuracy."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.mlflow_tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.log_params(
                {
                    "config": str(args.config),
                    "split": args.split,
                    "per_leaf": args.per_leaf,
                    "device": args.device,
                    "scope": metrics["scope"],
                }
            )
            mlflow.log_metrics(
                {
                    "character_error_rate": metrics["character_error_rate"],
                    "exact_transcription_rate": metrics[
                        "exact_transcription_rate"
                    ],
                    "ocr_text_category_top1": metrics[
                        "ocr_text_category_top1"
                    ],
                    "latency_p50_seconds": metrics["latency_p50_seconds"],
                    "latency_p95_seconds": metrics["latency_p95_seconds"],
                }
            )
            mlflow.log_artifact(str(args.output))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
