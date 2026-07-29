#!/usr/bin/env python3
"""Log the finalized CashLog original-data ensemble and reports to MLflow."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking-uri", required=True)
    parser.add_argument("--experiment", default="cashlog33-original-data")
    parser.add_argument("--run-name", default="cashlog33-originals-v2-final-ensemble")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--ensemble-selection", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--artifact", action="append", type=Path, default=[])
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    import mlflow

    training = json.loads(args.metrics.read_text(encoding="utf-8"))
    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    selection = json.loads(args.ensemble_selection.read_text(encoding="utf-8"))
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "component": "probability_blend_vision_head",
                "original_train_rows": training["additional_train_rows"],
                "train_augmented_samples": training["train_augmented_samples"],
                "trained_leaf_count": training["trained_leaf_count"],
                "candidate_weight": selection["selected"]["candidate_weight"],
                "proxy_gate_passed": comparison["proxy_gate_passed"],
                "serving_promotion_allowed": comparison["serving_promotion_allowed"],
            }
        )
        mlflow.log_metrics(
            {
                "baseline_top1": comparison["baseline"]["top1_accuracy"],
                "candidate_top1": comparison["candidate"]["top1_accuracy"],
                "candidate_top3": comparison["candidate"]["top3_accuracy"],
                "candidate_macro_f1": comparison["candidate"]["macro_f1"],
                "top1_delta": comparison["delta"]["top1_accuracy"],
                "macro_f1_delta": comparison["delta"]["macro_f1"],
                "minimum_leaf_recall_delta": min(
                    comparison["delta"]["per_leaf_recall"].values()
                ),
            }
        )
        for path in [
            args.metrics,
            args.comparison,
            args.ensemble_selection,
            args.source_audit,
            *args.artifact,
        ]:
            mlflow.log_artifact(str(path))
        print(json.dumps({"run_id": run.info.run_id, "artifact_uri": run.info.artifact_uri}))


if __name__ == "__main__":
    main()
