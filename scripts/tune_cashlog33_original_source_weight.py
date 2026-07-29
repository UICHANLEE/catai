#!/usr/bin/env python3
"""Tune the influence of train-only original sources from a cached embedding run."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from compare_cashlog33_vision_heads import evaluate, load_head


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_floats(value: str) -> list[float]:
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def metric_delta(candidate: dict, baseline: dict) -> dict:
    recall = {
        leaf_id: candidate["per_leaf_recall"][leaf_id] - baseline["per_leaf_recall"][leaf_id]
        for leaf_id in baseline["per_leaf_recall"]
    }
    return {
        "top1_accuracy": candidate["top1_accuracy"] - baseline["top1_accuracy"],
        "macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
        "minimum_leaf_recall": min(recall.values(), default=0.0),
        "per_leaf_recall": recall,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--baseline-head", type=Path, required=True)
    parser.add_argument("--input-head", type=Path, required=True)
    parser.add_argument("--output-head", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--additional-weights", default="0.02,0.05,0.10,0.20,0.35,0.50")
    parser.add_argument("--c-values", default="0.30,1.0,3.0")
    parser.add_argument("--max-leaf-recall-regression", type=float, default=0.02)
    args = parser.parse_args()

    run_metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    additional_samples = int(run_metrics["additional_train_augmented_samples"])
    total_train_samples = int(run_metrics["train_augmented_samples"])
    base_train_samples = total_train_samples - additional_samples
    if base_train_samples <= 0:
        raise ValueError("invalid base/additional training boundary")

    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        train = cache["train"]
        train_labels = [str(value) for value in cache["train_labels"].tolist()]
        val = cache["val"]
        val_labels = [str(value) for value in cache["val_labels"].tolist()]
    if len(train_labels) != total_train_samples or train.shape[0] != total_train_samples:
        raise ValueError("embedding cache does not match metrics sample counts")

    baseline = load_head(args.baseline_head)
    baseline_val = evaluate(baseline, val, val_labels)
    attempts: list[dict] = []
    fitted: list[tuple[tuple, LogisticRegression, dict]] = []
    for additional_weight in parse_floats(args.additional_weights):
        sample_weight = np.ones(total_train_samples, dtype=np.float32)
        sample_weight[base_train_samples:] = additional_weight
        for c_value in parse_floats(args.c_values):
            model = LogisticRegression(
                C=c_value,
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
                solver="lbfgs",
            )
            model.fit(train, train_labels, sample_weight=sample_weight)
            metrics = evaluate(model, val, val_labels)
            delta = metric_delta(metrics, baseline_val)
            gate = (
                delta["top1_accuracy"] > 0
                and delta["macro_f1"] > 0
                and delta["minimum_leaf_recall"] >= -args.max_leaf_recall_regression
            )
            attempt = {
                "additional_weight": additional_weight,
                "c_value": c_value,
                "validation": metrics,
                "delta_vs_baseline": delta,
                "validation_gate_passed": gate,
            }
            attempts.append(attempt)
            rank = (
                0 if gate else 1,
                max(0.0, -args.max_leaf_recall_regression - delta["minimum_leaf_recall"]),
                -metrics["macro_f1"],
                -metrics["top1_accuracy"],
            )
            fitted.append((rank, model, attempt))
            print(json.dumps(attempt, ensure_ascii=False), flush=True)

    fitted.sort(key=lambda item: item[0])
    _, selected, selected_attempt = fitted[0]
    artifact = joblib.load(args.input_head)
    if not isinstance(artifact, dict):
        artifact = {"model": selected}
    artifact["model"] = selected
    artifact["classes"] = [str(value) for value in selected.classes_]
    artifact["additional_source_weight"] = selected_attempt["additional_weight"]
    artifact["selected_c"] = selected_attempt["c_value"]
    artifact["source_weight_tuned_at"] = utc_now()
    args.output_head.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_head)

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "base_train_augmented_samples": base_train_samples,
        "additional_train_augmented_samples": additional_samples,
        "baseline_validation": baseline_val,
        "selected": selected_attempt,
        "attempts": attempts,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
