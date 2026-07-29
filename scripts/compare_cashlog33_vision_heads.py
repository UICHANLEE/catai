#!/usr/bin/env python3
"""Compare two CashLog vision heads on one frozen embedding holdout."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score


ROOT = Path(__file__).resolve().parents[1]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_head(path: Path):
    artifact = joblib.load(path)
    return artifact["model"] if isinstance(artifact, dict) and "model" in artifact else artifact


def evaluate(model, embeddings: np.ndarray, labels: list[str]) -> dict:
    probabilities = model.predict_proba(embeddings)
    classes = [str(value) for value in model.classes_]
    predicted = [classes[index] for index in probabilities.argmax(axis=1)]
    top_k = min(3, probabilities.shape[1])
    top_indexes = np.argpartition(-probabilities, kth=top_k - 1, axis=1)[:, :top_k]
    top3 = float(
        np.mean(
            [
                expected in [classes[index] for index in indexes]
                for expected, indexes in zip(labels, top_indexes, strict=True)
            ]
        )
    )
    leaf_order = sorted(set(labels))
    report = classification_report(
        labels,
        predicted,
        labels=leaf_order,
        output_dict=True,
        zero_division=0,
    )
    return {
        "samples": len(labels),
        "top1_accuracy": float(accuracy_score(labels, predicted)),
        "top3_accuracy": top3,
        "macro_f1": float(
            f1_score(labels, predicted, labels=leaf_order, average="macro", zero_division=0)
        ),
        "per_leaf_recall": {
            leaf_id: float(report[leaf_id]["recall"]) for leaf_id in leaf_order
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-head", type=Path, required=True)
    parser.add_argument("--candidate-head", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-leaf-recall-regression", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        embeddings = cache["test"]
        labels = [str(value) for value in cache["test_labels"].tolist()]
    baseline = evaluate(load_head(args.baseline_head), embeddings, labels)
    candidate = evaluate(load_head(args.candidate_head), embeddings, labels)
    recall_delta = {
        leaf_id: candidate["per_leaf_recall"][leaf_id] - baseline["per_leaf_recall"][leaf_id]
        for leaf_id in sorted(baseline["per_leaf_recall"])
    }
    proxy_gate = (
        candidate["top1_accuracy"] > baseline["top1_accuracy"]
        and candidate["macro_f1"] > baseline["macro_f1"]
        and min(recall_delta.values(), default=0.0) >= -args.max_leaf_recall_regression
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "baseline_head": str(args.baseline_head),
        "candidate_head": str(args.candidate_head),
        "embedding_cache": str(args.embedding_cache),
        "baseline": baseline,
        "candidate": candidate,
        "delta": {
            "top1_accuracy": candidate["top1_accuracy"] - baseline["top1_accuracy"],
            "top3_accuracy": candidate["top3_accuracy"] - baseline["top3_accuracy"],
            "macro_f1": candidate["macro_f1"] - baseline["macro_f1"],
            "per_leaf_recall": recall_delta,
        },
        "proxy_gate_passed": proxy_gate,
        "serving_promotion_allowed": False,
        "serving_block_reason": "A frozen, manually labeled real CashLog holdout is required.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
