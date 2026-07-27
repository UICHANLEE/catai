#!/usr/bin/env python3
"""Evaluate a CashLog text artifact on a fixed manifest split."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, log_loss


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    values = np.exp(logits)
    return values / values.sum(axis=1, keepdims=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="cashlog33-500k")
    parser.add_argument("--mlflow-run-name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, Any]] = []
    with args.manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if str(row.get("split")) == args.split:
                rows.append(row)
    if not rows:
        raise SystemExit(f"manifest has no {args.split} rows")
    artifact = joblib.load(args.model)
    model = artifact["model"]
    temperature = float(artifact.get("temperature", 1.0))
    labels = [str(row["leaf_id"]) for row in rows]
    probabilities = apply_temperature(
        model.predict_proba([str(row["text"]) for row in rows]), temperature
    )
    classes = [str(value) for value in model.named_steps["classifier"].classes_]
    predicted = [classes[index] for index in probabilities.argmax(axis=1)]
    class_indexes = {leaf_id: index for index, leaf_id in enumerate(classes)}
    true_indexes = np.asarray([class_indexes[label] for label in labels])
    top3_indexes = np.argpartition(
        -probabilities, kth=min(2, probabilities.shape[1] - 1), axis=1
    )[:, :3]
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == true_indexes
    ece = 0.0
    for lower, upper in zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1.0, 10)):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    metrics = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "model": str(args.model.resolve()),
        "manifest": str(args.manifest.resolve()),
        "split": args.split,
        "samples": len(rows),
        "top1_accuracy": float(accuracy_score(labels, predicted)),
        "top3_accuracy": float(
            np.mean(
                [
                    truth in candidates
                    for truth, candidates in zip(
                        true_indexes, top3_indexes, strict=True
                    )
                ]
            )
        ),
        "macro_f1": float(
            f1_score(labels, predicted, labels=classes, average="macro", zero_division=0)
        ),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "ece_10_bin": ece,
        "scope_warning": "Synthetic/noisy proxy split; not real CashLog accuracy.",
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
                    "component": "ocr_text_fixed_holdout",
                    "model": str(args.model),
                    "manifest": str(args.manifest),
                    "split": args.split,
                    "samples": len(rows),
                }
            )
            mlflow.log_metrics(
                {
                    key: value
                    for key, value in metrics.items()
                    if key
                    in {
                        "top1_accuracy",
                        "top3_accuracy",
                        "macro_f1",
                        "log_loss",
                        "ece_10_bin",
                    }
                }
            )
            mlflow.log_artifact(str(args.output))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
