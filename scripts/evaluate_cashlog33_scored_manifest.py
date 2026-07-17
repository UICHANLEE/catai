#!/usr/bin/env python3
"""Evaluate a scored image manifest without claiming CashLog production accuracy."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/cashlog33/openimages_v7/scored_manifest.jsonl"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_OUTPUT = ROOT / "reports/cashlog33/vision_proxy"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="cashlog33-hybrid-v2")
    parser.add_argument("--mlflow-run-name", default="cashlog33-siglip2-openimages-proxy-v1")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    category_ids = [str(row["id"]) for row in categories]
    rows = read_jsonl(args.input_manifest)
    usable = [row for row in rows if row.get("review_details", {}).get("top3")]
    if not usable:
        raise SystemExit("scored manifest has no usable predictions")
    actual = [str(row["leaf_id"]) for row in usable]
    predicted = [str(row["review_details"]["top3"][0]["leaf_id"]) for row in usable]
    top3_hits = [
        truth in {str(value["leaf_id"]) for value in row["review_details"]["top3"]}
        for truth, row in zip(actual, usable, strict=True)
    ]
    supported = [leaf_id for leaf_id in category_ids if leaf_id in set(actual)]
    report = classification_report(
        actual,
        predicted,
        labels=category_ids,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": "Open Images V7 validation human-verified source-label proxy",
        "samples": len(usable),
        "supported_leaf_count": len(supported),
        "taxonomy_leaf_count": len(category_ids),
        "supported_leaves": supported,
        "top1_accuracy": float(accuracy_score(actual, predicted)),
        "top3_accuracy": float(sum(top3_hits) / len(top3_hits)),
        "macro_f1_supported": float(
            f1_score(actual, predicted, labels=supported, average="macro", zero_division=0)
        ),
        "class_support": dict(sorted(Counter(actual).items())),
        "scope_warning": (
            "The source object labels are human verified, but their mapping to CashLog expense leaves "
            "is a project rule. This is a visual proxy metric, not real app accuracy."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
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
    matrix = confusion_matrix(actual, predicted, labels=category_ids)
    confusion_path = args.output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *category_ids])
        for leaf_id, values in zip(category_ids, matrix.tolist(), strict=True):
            writer.writerow([leaf_id, *values])

    if not args.disable_mlflow and args.mlflow_tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.log_params(
                {
                    "component": "siglip2_vision_zero_shot",
                    "dataset": "openimages_v7_validation_proxy",
                    "samples": len(usable),
                    "supported_leaf_count": len(supported),
                }
            )
            mlflow.log_metrics(
                {
                    "vision_proxy_top1": metrics["top1_accuracy"],
                    "vision_proxy_top3": metrics["top3_accuracy"],
                    "vision_proxy_macro_f1": metrics["macro_f1_supported"],
                }
            )
            for path in [metrics_path, per_class_path, confusion_path]:
                mlflow.log_artifact(str(path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
