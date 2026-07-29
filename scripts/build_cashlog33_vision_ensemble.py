#!/usr/bin/env python3
"""Select a conservative source-expanded vision ensemble on validation data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

from catai.probability_ensemble import ProbabilityBlendClassifier
from compare_cashlog33_vision_heads import evaluate, load_head


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-head", type=Path, required=True)
    parser.add_argument("--candidate-head", type=Path, required=True)
    parser.add_argument("--embedding-cache", type=Path, required=True)
    parser.add_argument("--output-head", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--candidate-weights", default="0.10,0.20,0.30,0.40,0.50")
    parser.add_argument("--max-leaf-recall-regression", type=float, default=0.02)
    args = parser.parse_args()

    baseline_artifact = joblib.load(args.baseline_head)
    candidate_artifact = joblib.load(args.candidate_head)
    baseline = load_head(args.baseline_head)
    candidate = load_head(args.candidate_head)
    with np.load(args.embedding_cache, allow_pickle=False) as cache:
        val = cache["val"]
        labels = [str(value) for value in cache["val_labels"].tolist()]
    baseline_metrics = evaluate(baseline, val, labels)

    attempts = []
    selected = None
    for weight in [float(value) for value in args.candidate_weights.split(",")]:
        ensemble = ProbabilityBlendClassifier(baseline, candidate, weight)
        metrics = evaluate(ensemble, val, labels)
        recall_delta = {
            leaf_id: metrics["per_leaf_recall"][leaf_id]
            - baseline_metrics["per_leaf_recall"][leaf_id]
            for leaf_id in baseline_metrics["per_leaf_recall"]
        }
        gate = (
            metrics["top1_accuracy"] > baseline_metrics["top1_accuracy"]
            and metrics["macro_f1"] > baseline_metrics["macro_f1"]
            and min(recall_delta.values(), default=0.0) >= -args.max_leaf_recall_regression
        )
        attempt = {
            "candidate_weight": weight,
            "validation": metrics,
            "minimum_leaf_recall_delta": min(recall_delta.values(), default=0.0),
            "validation_gate_passed": gate,
        }
        attempts.append(attempt)
        if selected is None and gate:
            selected = (ensemble, attempt)
    if selected is None:
        raise RuntimeError("no conservative ensemble passed the validation gate")

    ensemble, selected_attempt = selected
    artifact = dict(candidate_artifact)
    artifact["model"] = ensemble
    artifact["classes"] = [str(value) for value in ensemble.classes_]
    artifact["supported_leaves"] = [str(value) for value in ensemble.classes_]
    artifact["ensemble"] = {
        "baseline_head": str(args.baseline_head),
        "candidate_head": str(args.candidate_head),
        "candidate_weight": selected_attempt["candidate_weight"],
        "selection": "lowest validation weight passing aggregate and 2pp leaf regression gates",
    }
    artifact["created_at"] = utc_now()
    args.output_head.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output_head)

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "baseline_validation": baseline_metrics,
        "selected": selected_attempt,
        "attempts": attempts,
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
