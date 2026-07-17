#!/usr/bin/env python3
"""Evaluate the complete hybrid I/O path on a labeled image manifest."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_MANIFEST = ROOT / "data/processed/cashlog33/e2e_fixtures/v1/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/cashlog33/hybrid_e2e"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_image_path(row: dict[str, Any]) -> Path:
    path = Path(str(row["relative_path"]))
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="cashlog33-hybrid-v2")
    parser.add_argument("--mlflow-run-name", default="cashlog33-hybrid-synthetic-e2e-v1")
    parser.add_argument(
        "--dataset-scope",
        choices=["synthetic_integration", "real_cashlog_holdout", "proxy"],
        default="synthetic_integration",
    )
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--target-top1", type=float, default=0.95)
    parser.add_argument("--require-target", action="store_true")
    return parser.parse_args()


def expected_calibration_error(
    actual: list[str], predicted: list[str], confidences: list[float], bins: int = 10
) -> float:
    correctness = np.asarray(
        [expected == prediction for expected, prediction in zip(actual, predicted, strict=True)],
        dtype=np.float64,
    )
    confidence = np.asarray(confidences, dtype=np.float64)
    total = max(1, len(confidence))
    ece = 0.0
    for lower in np.linspace(0.0, 1.0, bins + 1)[:-1]:
        upper = lower + 1.0 / bins
        mask = (confidence >= lower) & (
            confidence <= upper if upper >= 1.0 else confidence < upper
        )
        if mask.any():
            ece += float(mask.sum() / total) * abs(
                float(correctness[mask].mean()) - float(confidence[mask].mean())
            )
    return ece


def main() -> None:
    args = parse_args()
    from catai.cashlog_hybrid_classifier import CashlogHybridClassifier

    classifier = CashlogHybridClassifier(args.config, device=args.device)
    rows = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    predictions = []
    latencies = []
    for index, row in enumerate(rows, start=1):
        started = time.perf_counter()
        response = classifier.analyze(resolve_image_path(row))
        latency = time.perf_counter() - started
        timings_ms = response.pop("_timings_ms", {})
        latencies.append(latency)
        top3 = [str(value["category"]) for value in response["top_categories"]]
        predictions.append(
            {
                "sample_id": row["sample_id"],
                "expected": row["leaf_id"],
                "predicted": response["recommended_category"],
                "top3": top3,
                "confidence": response["confidence"],
                "need_user_check": response["need_user_check"],
                "latency_seconds": latency,
                "ocr_text": response["evidence"]["ocr"]["text"],
                "matched_terms": response["evidence"]["matched_terms"],
                "fallback_reasons": response["evidence"].get("fallback_reasons", []),
                "timings_ms": timings_ms,
            }
        )
        print(
            json.dumps(
                {
                    "event": "evaluation_sample_completed",
                    "sample": index,
                    "samples": len(rows),
                    "expected": row["leaf_id"],
                    "predicted": response["recommended_category"],
                    "need_user_check": response["need_user_check"],
                    "latency_ms": latency * 1000.0,
                    "timings_ms": timings_ms,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )

    actual = [str(row["expected"]) for row in predictions]
    predicted = [str(row["predicted"]) for row in predictions]
    category_ids = classifier.leaf_ids
    top3_accuracy = sum(
        expected in row["top3"] for expected, row in zip(actual, predictions, strict=True)
    ) / len(predictions)
    wrong_auto = [
        row for row in predictions if row["expected"] != row["predicted"] and not row["need_user_check"]
    ]
    correct_auto = [
        row for row in predictions if row["expected"] == row["predicted"] and not row["need_user_check"]
    ]
    report = classification_report(
        actual,
        predicted,
        labels=category_ids,
        output_dict=True,
        zero_division=0,
    )
    support_by_leaf = {
        leaf_id: sum(expected == leaf_id for expected in actual) for leaf_id in category_ids
    }
    minimum_recall = min(
        (float(report[leaf_id]["recall"]) for leaf_id in category_ids if support_by_leaf[leaf_id]),
        default=0.0,
    )
    top1_accuracy = float(accuracy_score(actual, predicted))
    stage_names = sorted(
        {name for row in predictions for name in row.get("timings_ms", {})}
    )
    stage_latency = {
        name: {
            "p50_ms": float(
                np.percentile(
                    [row["timings_ms"][name] for row in predictions if name in row["timings_ms"]],
                    50,
                )
            ),
            "p95_ms": float(
                np.percentile(
                    [row["timings_ms"][name] for row in predictions if name in row["timings_ms"]],
                    95,
                )
            ),
        }
        for name in stage_names
    }
    metrics = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "samples": len(predictions),
        "leaf_count": len(set(actual)),
        "minimum_per_leaf": min(support_by_leaf.values()),
        "top1_accuracy": top1_accuracy,
        "top3_accuracy": float(top3_accuracy),
        "macro_f1": float(
            f1_score(actual, predicted, labels=category_ids, average="macro", zero_division=0)
        ),
        "minimum_leaf_recall": minimum_recall,
        "ece_10_bin": expected_calibration_error(
            actual, predicted, [float(row["confidence"]) for row in predictions]
        ),
        "need_user_check_rate": float(
            sum(row["need_user_check"] for row in predictions) / len(predictions)
        ),
        "correct_auto_confirm_rate": float(len(correct_auto) / len(predictions)),
        "false_auto_confirm_rate": float(len(wrong_auto) / len(predictions)),
        "latency_p50_seconds": float(np.percentile(latencies, 50)),
        "latency_p95_seconds": float(np.percentile(latencies, 95)),
        "dataset_scope": args.dataset_scope,
        "device": str(classifier.device),
        "target_top1": args.target_top1,
        "target_met": top1_accuracy >= args.target_top1,
        "stage_latency": stage_latency,
        "safe_fallback_rate": float(
            sum(bool(row["fallback_reasons"]) for row in predictions) / len(predictions)
        ),
        "scope_warning": (
            "Synthetic rendered receipts validate I/O and 33-leaf routing only."
            if args.dataset_scope == "synthetic_integration"
            else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "predictions.jsonl"
    with predictions_path.open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    per_class_path = args.output_dir / "per_class_metrics.csv"
    with per_class_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["leaf_id", "precision", "recall", "f1", "support"],
            lineterminator="\n",
        )
        writer.writeheader()
        for leaf_id in category_ids:
            values = report[leaf_id]
            writer.writerow(
                {
                    "leaf_id": leaf_id,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "f1": values["f1-score"],
                    "support": int(values["support"]),
                }
            )
    confusion_path = args.output_dir / "confusion_matrix.csv"
    matrix = confusion_matrix(actual, predicted, labels=category_ids)
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["expected\\predicted", *category_ids])
        for leaf_id, values in zip(category_ids, matrix.tolist(), strict=True):
            writer.writerow([leaf_id, *values])
    if not args.disable_mlflow and args.mlflow_tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.log_params(
                {
                    "component": "hybrid_e2e",
                    "model_version": classifier.config["model_version"],
                    "samples": len(predictions),
                    "leaf_count": len(set(actual)),
                    "dataset_scope": args.dataset_scope,
                    "device": str(classifier.device),
                    "target_top1": args.target_top1,
                }
            )
            mlflow.log_metrics(
                {
                    "e2e_top1": metrics["top1_accuracy"],
                    "e2e_top3": metrics["top3_accuracy"],
                    "e2e_macro_f1": metrics["macro_f1"],
                    "e2e_false_auto_confirm_rate": metrics["false_auto_confirm_rate"],
                    "e2e_latency_p95": metrics["latency_p95_seconds"],
                    "e2e_ece_10_bin": metrics["ece_10_bin"],
                    "e2e_minimum_leaf_recall": metrics["minimum_leaf_recall"],
                    "e2e_target_met": float(metrics["target_met"]),
                }
            )
            for path in [metrics_path, predictions_path, per_class_path, confusion_path]:
                mlflow.log_artifact(str(path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if args.require_target and not metrics["target_met"]:
        raise SystemExit(
            f"Top-1 target not met for {args.dataset_scope}: "
            f"actual={metrics['top1_accuracy']:.6f} required={args.target_top1:.6f}"
        )


if __name__ == "__main__":
    main()
