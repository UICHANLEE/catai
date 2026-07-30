#!/usr/bin/env python3
"""Evaluate FP32/INT8 compact models against the frozen visual proxy baseline."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import mlflow
import onnxruntime as ort
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)

from catai.cashlog_hybrid_classifier import CashlogHybridClassifier

try:
    from scripts.train_cashlog33_million_mobilenet import make_transforms, read_jsonl
except ModuleNotFoundError:
    from train_cashlog33_million_mobilenet import make_transforms, read_jsonl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = ROOT / "checkpoints/cashlog33/mobilenet_million_v1/onnx"
DEFAULT_MODEL_DIR = (
    ROOT / "src/catai/assets/cashlog33_mobilenetv4_int8_v1"
)
DEFAULT_EXTERNAL_MANIFEST = ROOT / "data/raw/cashlog33/openimages_v7/manifest.jsonl"
DEFAULT_SPLIT = (
    ROOT / "checkpoints/cashlog33/vision_head_originals_v2/split_manifest.jsonl"
)
DEFAULT_BASELINE = (
    ROOT / "reports/cashlog33/million_v1/current_serving_visual_baseline.json"
)
DEFAULT_SERVING_CONFIG = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_OUTPUT = ROOT / "reports/cashlog33/million_v1/compact_comparison.json"
DEFAULT_BASELINE_ACTUAL = (
    ROOT / "reports/cashlog33/million_v1/current_actual/metrics.json"
)
DEFAULT_CANDIDATE_ACTUAL = (
    ROOT / "reports/cashlog33/million_v1/candidate_actual/metrics.json"
)
DEFAULT_TRAINING_MANIFEST = (
    ROOT
    / "data/processed/cashlog33/training/million_v1/original_manifest.jsonl"
)
DEFAULT_ACTUAL_MANIFEST = ROOT / "data/raw/cashlog33/actual/manifest.jsonl"
DEFAULT_MLFLOW_RUN = (
    ROOT / "checkpoints/cashlog33/mobilenet_million_v1/mlflow_run.json"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def artifact_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": repository_path(path),
        "bytes": path.stat().st_size,
        "megabytes": path.stat().st_size / (1024 * 1024),
        "sha256": sha256_file(path),
    }


def load_frozen_test_rows(
    manifest_path: Path, split_path: Path
) -> list[dict[str, Any]]:
    test_ids = {
        str(row["sample_id"])
        for row in read_jsonl(split_path)
        if row["split"] == "test"
    }
    rows = [
        row
        for row in read_jsonl(manifest_path)
        if str(row["sample_id"]) in test_ids
    ]
    if len(rows) != len(test_ids):
        missing = test_ids - {str(row["sample_id"]) for row in rows}
        raise RuntimeError(f"frozen test images are missing: {sorted(missing)[:5]}")
    return sorted(rows, key=lambda row: str(row["sample_id"]))


def load_batches(
    rows: list[dict[str, Any]],
    label_to_index: dict[str, int],
    image_size: int,
    batch_size: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    _, transform = make_transforms(image_size)
    for offset in range(0, len(rows), batch_size):
        batch_rows = rows[offset : offset + batch_size]
        tensors = []
        expected = []
        for row in batch_rows:
            with Image.open(ROOT / str(row["relative_path"])) as image:
                tensors.append(transform(image.convert("RGB")).numpy())
            expected.append(label_to_index[str(row["leaf_id"])])
        yield (
            np.stack(tensors).astype(np.float32, copy=False),
            np.asarray(expected, dtype=np.int64),
        )


def evaluate_session(
    model_path: Path,
    batches: Iterator[tuple[np.ndarray, np.ndarray]],
    labels: list[str],
    latency_image: Image.Image,
    image_size: int,
) -> tuple[dict[str, Any], np.ndarray]:
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    expected_parts = []
    logits_parts = []
    elapsed = 0.0
    single_image: np.ndarray | None = None
    for images, expected in batches:
        if single_image is None:
            single_image = images[:1]
        started = time.perf_counter()
        logits = session.run(None, {input_name: images})[0]
        elapsed += time.perf_counter() - started
        expected_parts.append(expected)
        logits_parts.append(logits)
    expected = np.concatenate(expected_parts)
    logits = np.concatenate(logits_parts)
    if single_image is None:
        raise RuntimeError("evaluation produced no input batches")
    for _ in range(5):
        session.run(None, {input_name: single_image})
    model_timings = []
    for _ in range(50):
        started = time.perf_counter()
        session.run(None, {input_name: single_image})
        model_timings.append((time.perf_counter() - started) * 1000.0)
    serving_timings = []
    for _ in range(50):
        started = time.perf_counter()
        serving_tensor = CashlogHybridClassifier._compact_image_tensor(
            latency_image, image_size
        )
        session.run(None, {input_name: serving_tensor})
        serving_timings.append((time.perf_counter() - started) * 1000.0)
    predicted = logits.argmax(axis=1)
    top3 = np.argpartition(logits, -min(3, len(labels)), axis=1)[
        :, -min(3, len(labels)) :
    ]
    recalls = recall_score(
        expected,
        predicted,
        labels=list(range(len(labels))),
        average=None,
        zero_division=0,
    )
    return (
        {
            "samples": len(expected),
            "top1_accuracy": float(np.mean(predicted == expected)),
            "top3_accuracy": float(
                np.mean(np.any(top3 == expected[:, None], axis=1))
            ),
            "macro_f1": float(
                f1_score(
                    expected,
                    predicted,
                    labels=list(range(len(labels))),
                    average="macro",
                    zero_division=0,
                )
            ),
            "per_leaf_recall": {
                leaf_id: float(recalls[index])
                for index, leaf_id in enumerate(labels)
            },
            "runtime": {
                "provider": "CPUExecutionProvider",
                "total_seconds": elapsed,
                "images_per_second": len(expected) / elapsed,
                "mean_model_ms_per_image": elapsed * 1000.0 / len(expected),
                "single_image_model_p50_ms": float(
                    np.percentile(model_timings, 50)
                ),
                "single_image_model_p95_ms": float(
                    np.percentile(model_timings, 95)
                ),
                "single_image_p50_ms": float(
                    np.percentile(serving_timings, 50)
                ),
                "single_image_p95_ms": float(
                    np.percentile(serving_timings, 95)
                ),
                "single_image_timing_scope": (
                    "PIL resize, crop, normalize, and ONNX Runtime inference"
                ),
            },
        },
        logits,
    )


def serving_visual_bytes(config_path: Path) -> int:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    base = config_path.parent
    compact = config.get("compact_vision", {})
    if compact.get("enabled", False):
        total = sum(
            (base / compact[key]).resolve().stat().st_size
            for key in ["model", "labels"]
        )
    else:
        model_dir = (base / config["vision_model"]).resolve()
        head = (base / config["vision_head"]).resolve()
        total = sum(
            path.stat().st_size
            for path in model_dir.rglob("*")
            if path.is_file()
        ) + head.stat().st_size
    specialist = config.get("meal_specialist", {})
    if specialist.get("enabled", False):
        total += (base / specialist["checkpoint"]).resolve().stat().st_size
        total += (base / specialist["labels"]).resolve().stat().st_size
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fp32",
        type=Path,
        default=DEFAULT_EXPORT_DIR / "cashlog33-visual-fp32.onnx",
    )
    parser.add_argument(
        "--int8", type=Path, default=DEFAULT_MODEL_DIR / "model.int8.onnx"
    )
    parser.add_argument("--labels", type=Path, default=DEFAULT_MODEL_DIR / "labels.json")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--serving-config", type=Path, default=DEFAULT_SERVING_CONFIG
    )
    parser.add_argument(
        "--baseline-actual", type=Path, default=DEFAULT_BASELINE_ACTUAL
    )
    parser.add_argument(
        "--candidate-actual", type=Path, default=DEFAULT_CANDIDATE_ACTUAL
    )
    parser.add_argument(
        "--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST
    )
    parser.add_argument(
        "--actual-manifest", type=Path, default=DEFAULT_ACTUAL_MANIFEST
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-recall-regression", type=float, default=0.10)
    parser.add_argument("--minimum-recall-gate-support", type=int, default=30)
    parser.add_argument("--max-quantized-top1-drift", type=float, default=0.005)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5500"),
    )
    parser.add_argument(
        "--mlflow-run-file", type=Path, default=DEFAULT_MLFLOW_RUN
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    category_rows = json.loads(args.labels.read_text(encoding="utf-8"))
    labels = [str(row["id"]) for row in category_rows]
    label_to_index = {leaf_id: index for index, leaf_id in enumerate(labels)}
    rows = load_frozen_test_rows(args.manifest, args.split)
    unexpected = sorted({str(row["leaf_id"]) for row in rows} - set(labels))
    if unexpected:
        raise RuntimeError(f"candidate model does not cover test labels: {unexpected}")
    with Image.open(ROOT / str(rows[0]["relative_path"])) as source:
        latency_image = source.convert("RGB")
    fp32, fp32_logits = evaluate_session(
        args.fp32,
        load_batches(rows, label_to_index, args.image_size, args.batch_size),
        labels,
        latency_image,
        args.image_size,
    )
    int8, int8_logits = evaluate_session(
        args.int8,
        load_batches(rows, label_to_index, args.image_size, args.batch_size),
        labels,
        latency_image,
        args.image_size,
    )

    baseline_report = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = baseline_report["baseline"]
    recall_deltas = {
        leaf_id: int8["per_leaf_recall"][leaf_id]
        - float(baseline["per_leaf_recall"][leaf_id])
        for leaf_id in labels
    }
    candidate_delta = {
        "top1_accuracy": int8["top1_accuracy"] - baseline["top1_accuracy"],
        "top3_accuracy": int8["top3_accuracy"] - baseline["top3_accuracy"],
        "macro_f1": int8["macro_f1"] - baseline["macro_f1"],
        "per_leaf_recall": recall_deltas,
    }
    quantization_delta = {
        "top1_accuracy": int8["top1_accuracy"] - fp32["top1_accuracy"],
        "top3_accuracy": int8["top3_accuracy"] - fp32["top3_accuracy"],
        "macro_f1": int8["macro_f1"] - fp32["macro_f1"],
        "top1_agreement": float(
            np.mean(int8_logits.argmax(axis=1) == fp32_logits.argmax(axis=1))
        ),
        "mean_abs_logit_error": float(np.mean(np.abs(int8_logits - fp32_logits))),
    }
    expected_indexes = np.asarray(
        [label_to_index[str(row["leaf_id"])] for row in rows],
        dtype=np.int64,
    )
    predicted_indexes = int8_logits.argmax(axis=1)
    top3_indexes = np.argsort(int8_logits, axis=1)[:, -3:][:, ::-1]
    predictions_path = args.output.parent / "int8_predictions.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with predictions_path.open("w", encoding="utf-8") as handle:
        for sample_index, (
            row,
            expected_index,
            predicted_index,
            top3,
        ) in enumerate(
            zip(
                rows,
                expected_indexes,
                predicted_indexes,
                top3_indexes,
                strict=True,
            )
        ):
            handle.write(
                json.dumps(
                    {
                        "sample_id": row["sample_id"],
                        "expected": labels[int(expected_index)],
                        "predicted": labels[int(predicted_index)],
                        "top3": [
                            {
                                "leaf_id": labels[int(index)],
                                "logit": float(
                                    int8_logits[sample_index, int(index)]
                                ),
                            }
                            for index in top3
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    matrix = confusion_matrix(
        expected_indexes,
        predicted_indexes,
        labels=list(range(len(labels))),
    )
    confusion_path = args.output.parent / "int8_confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["expected/predicted", *labels])
        for leaf_id, values in zip(labels, matrix, strict=True):
            writer.writerow([leaf_id, *values.tolist()])
    precision, recall, f1, support = precision_recall_fscore_support(
        expected_indexes,
        predicted_indexes,
        labels=list(range(len(labels))),
        zero_division=0,
    )
    per_leaf_path = args.output.parent / "int8_per_leaf_metrics.csv"
    with per_leaf_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "leaf_id",
                "support",
                "precision",
                "recall",
                "f1",
                "baseline_recall",
                "recall_delta",
            ]
        )
        for index, leaf_id in enumerate(labels):
            writer.writerow(
                [
                    leaf_id,
                    int(support[index]),
                    float(precision[index]),
                    float(recall[index]),
                    float(f1[index]),
                    float(baseline["per_leaf_recall"][leaf_id]),
                    float(recall_deltas[leaf_id]),
                ]
            )
    current_bytes = serving_visual_bytes(args.serving_config)
    baseline_p50_ms = float(
        baseline.get("runtime", {}).get(
            "p50_vision_ms",
            baseline.get("runtime", {}).get("mean_vision_ms", float("inf")),
        )
    )
    supports = {
        leaf_id: int(value)
        for leaf_id, value in baseline.get("per_leaf_support", {}).items()
    }
    recall_gate_leaves = [
        leaf_id
        for leaf_id in labels
        if supports.get(leaf_id, 0) >= args.minimum_recall_gate_support
    ]
    if not args.baseline_actual.is_file() or not args.candidate_actual.is_file():
        raise RuntimeError(
            "both current and candidate actual-holdout metrics are required"
        )
    baseline_actual = json.loads(args.baseline_actual.read_text(encoding="utf-8"))
    candidate_actual = json.loads(
        args.candidate_actual.read_text(encoding="utf-8")
    )
    training_hashes = {
        str(row["sha256"])
        for row in read_jsonl(args.training_manifest)
        if row.get("sha256")
    }
    actual_rows = read_jsonl(args.actual_manifest)
    actual_hashes = {
        str(row["sha256"]) for row in actual_rows if row.get("sha256")
    }
    feedback_replay_overlap = len(training_hashes & actual_hashes)
    gate_checks = {
        "int8_top1_beats_baseline": candidate_delta["top1_accuracy"] > 0.0,
        "int8_macro_f1_beats_baseline": candidate_delta["macro_f1"] > 0.0,
        "per_leaf_recall_regression_within_limit": (
            min(recall_deltas[leaf_id] for leaf_id in recall_gate_leaves)
            >= -args.max_recall_regression
            if recall_gate_leaves
            else True
        ),
        "int8_top1_drift_within_limit": (
            quantization_delta["top1_accuracy"]
            >= -args.max_quantized_top1_drift
        ),
        "int8_smaller_than_current_visual_stack": (
            args.int8.stat().st_size < current_bytes
        ),
        "int8_single_image_faster_than_current_visual_stack": (
            int8["runtime"]["single_image_p50_ms"] < baseline_p50_ms
        ),
    }
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "serving_config_sha256": sha256_file(args.serving_config),
        "evaluation_set": {
            "kind": "external_visual_proxy",
            "official_split": "Open Images validation",
            "selection": "frozen test rows from the existing split manifest",
            "samples": len(rows),
            "manifest_sha256": sha256_file(args.manifest),
            "split_sha256": sha256_file(args.split),
            "warning": (
                "This is a visual proxy benchmark, not a manually labeled "
                "CashLog production-photo holdout."
            ),
        },
        "baseline": baseline,
        "fp32": {**fp32, "artifact": artifact_metadata(args.fp32)},
        "int8": {**int8, "artifact": artifact_metadata(args.int8)},
        "candidate_delta_vs_baseline": candidate_delta,
        "quantization_delta": quantization_delta,
        "detail_artifacts": {
            "predictions": artifact_metadata(predictions_path),
            "confusion_matrix": artifact_metadata(confusion_path),
            "per_leaf_metrics": artifact_metadata(per_leaf_path),
        },
        "size_comparison": {
            "current_visual_stack_bytes": current_bytes,
            "current_visual_stack_megabytes": current_bytes / (1024 * 1024),
            "int8_bytes": args.int8.stat().st_size,
            "reduction_ratio": 1.0 - args.int8.stat().st_size / current_bytes,
        },
        "latency_comparison": {
            "current_visual_p50_ms": baseline_p50_ms,
            "int8_single_image_p50_ms": int8["runtime"][
                "single_image_p50_ms"
            ],
            "reduction_ratio": (
                1.0
                - int8["runtime"]["single_image_p50_ms"] / baseline_p50_ms
            ),
        },
        "actual_feedback_replay_smoke": {
            "warning": (
                "These human-verified feedback images overlap the training "
                "history. This is a replay smoke test only and is deliberately "
                "excluded from the promotion gate."
            ),
            "training_sha256_overlap": feedback_replay_overlap,
            "actual_samples": len(actual_rows),
            "baseline": {
                "samples": baseline_actual["samples"],
                "top1_accuracy": baseline_actual["top1_accuracy"],
            },
            "candidate": {
                "samples": candidate_actual["samples"],
                "top1_accuracy": candidate_actual["top1_accuracy"],
            },
        },
        "gate_thresholds": {
            "actual_feedback_replay_is_promotion_gate": False,
            "max_per_leaf_recall_regression": args.max_recall_regression,
            "minimum_per_leaf_support_for_recall_gate": (
                args.minimum_recall_gate_support
            ),
            "recall_gate_leaves": recall_gate_leaves,
            "max_quantized_top1_drift": args.max_quantized_top1_drift,
        },
        "gate_checks": gate_checks,
        "promotion_gate_passed": all(gate_checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.mlflow_run_file.is_file():
        run_info = json.loads(args.mlflow_run_file.read_text(encoding="utf-8"))
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        with mlflow.start_run(run_id=str(run_info["run_id"])):
            for path in [
                args.output,
                predictions_path,
                confusion_path,
                per_leaf_path,
            ]:
                mlflow.log_artifact(str(path), artifact_path="evaluation")
            mlflow.log_metrics(
                {
                    "external_int8_top1": int8["top1_accuracy"],
                    "external_int8_top3": int8["top3_accuracy"],
                    "external_int8_macro_f1": int8["macro_f1"],
                    "external_top1_delta_vs_serving": candidate_delta[
                        "top1_accuracy"
                    ],
                    "external_macro_f1_delta_vs_serving": candidate_delta[
                        "macro_f1"
                    ],
                    "promotion_gate_passed": float(
                        report["promotion_gate_passed"]
                    ),
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
