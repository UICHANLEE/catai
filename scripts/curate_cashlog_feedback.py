#!/usr/bin/env python3
"""Build reviewed active-learning candidates from a de-identified feedback export."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.feedback import (  # noqa: E402
    curate_feedback_release,
    load_leaf_ids,
    read_jsonl,
)


DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--taxonomy-version", default="13.33.1")
    parser.add_argument("--minimum-images-per-leaf", type=int, default=10)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="cashlog33-feedback-curation")
    parser.add_argument("--mlflow-run-name", default="cashlog33-feedback-release")
    return parser.parse_args()


def log_to_mlflow(args: argparse.Namespace, summary: dict[str, object]) -> None:
    if not args.mlflow_tracking_uri:
        return
    import mlflow

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name=args.mlflow_run_name):
        mlflow.log_params(
            {
                "taxonomy_version": summary["taxonomy_version"],
                "minimum_images_per_leaf": summary["minimum_images_per_leaf"],
                "auto_training_allowed": summary["auto_training_allowed"],
            }
        )
        metrics = {
            "events": summary["events"],
            "pending_events": summary["pending_events"],
            "approved_events": summary["approved_events"],
            "rejected_events": summary["rejected_events"],
            "approved_image_candidates": summary["approved_image_candidates"],
            "metadata_only_approved": summary["metadata_only_approved"],
            "ready_for_training": int(bool(summary["ready_for_training"])),
        }
        if summary["correction_rate"] is not None:
            metrics["correction_rate"] = summary["correction_rate"]
        if summary["top3_coverage"] is not None:
            metrics["top3_coverage"] = summary["top3_coverage"]
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(args.output_dir), artifact_path="feedback_release")


def main() -> None:
    args = parse_args()
    if args.minimum_images_per_leaf < 1:
        raise SystemExit("minimum-images-per-leaf must be positive")
    existing_files = [
        args.output_dir / "training_candidates.jsonl",
        args.output_dir / "approved_metadata_feedback.jsonl",
        args.output_dir / "curation_summary.json",
        args.output_dir / "per_leaf_feedback.csv",
    ]
    if args.reuse_existing and all(path.is_file() for path in existing_files):
        summary = json.loads(existing_files[2].read_text(encoding="utf-8"))
        if summary.get("taxonomy_version") != args.taxonomy_version:
            raise SystemExit("existing curation uses a different taxonomy version")
        if summary.get("minimum_images_per_leaf") != args.minimum_images_per_leaf:
            raise SystemExit("existing curation uses a different leaf minimum")
    else:
        summary = curate_feedback_release(
            read_jsonl(args.events),
            output_dir=args.output_dir,
            leaf_ids=load_leaf_ids(args.categories),
            taxonomy_version=args.taxonomy_version,
            minimum_images_per_leaf=args.minimum_images_per_leaf,
        )
    log_to_mlflow(args, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
