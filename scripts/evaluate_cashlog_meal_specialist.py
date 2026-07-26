#!/usr/bin/env python3
"""Evaluate a meal specialist checkpoint on its deterministic UECFood split."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch.utils.data import DataLoader

from train_cashlog_category_from_uecfood import (
    ROOT,
    CashlogImageDataset,
    build_model,
    choose_device,
    collect_samples,
    load_categories,
    make_transforms,
    remap_samples,
    stratified_split,
)


MEAL_LEAVES = {
    "meal_grocery",
    "meal_dining",
    "meal_cafe",
    "meal_drink",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=ROOT / "data/processed/classification/uecfood256/UECFOOD256",
    )
    parser.add_argument(
        "--categories",
        type=Path,
        default=ROOT / "configs/cashlog/categories.json",
    )
    parser.add_argument(
        "--overrides",
        type=Path,
        default=ROOT / "configs/cashlog/uecfood_category_overrides.json",
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bbox-padding", type=float, default=0.10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="cashlog33-all-data-mps")
    parser.add_argument("--mlflow-run-name", default="meal2-uecfood-all-v1-evaluation")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = json.loads(args.labels.read_text(encoding="utf-8"))
    label_ids = [str(row["id"]) for row in labels]
    if len(label_ids) < 2 or not set(label_ids).issubset(MEAL_LEAVES):
        raise SystemExit(
            f"expected at least two CashLog meal leaves, got {label_ids}"
        )

    categories = load_categories(args.categories)
    all_label_to_index = {
        str(category["id"]): index for index, category in enumerate(categories)
    }
    overrides = json.loads(args.overrides.read_text(encoding="utf-8"))
    all_samples = collect_samples(
        args.dataset_root,
        label_to_index=all_label_to_index,
        overrides=overrides,
        max_samples_per_class=None,
        use_bbox=True,
    )
    index_remap = {
        all_label_to_index[leaf_id]: index
        for index, leaf_id in enumerate(label_ids)
    }
    samples = remap_samples(all_samples, index_remap)
    _, validation_samples = stratified_split(samples, args.val_ratio, args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_ids = [
        str(category["id"]) for category in checkpoint.get("categories", [])
    ]
    if checkpoint_ids and checkpoint_ids != label_ids:
        raise SystemExit(
            f"checkpoint labels do not match evaluation labels: "
            f"{checkpoint_ids} != {label_ids}"
        )
    arch = str(
        checkpoint.get("arch")
        or checkpoint.get("model_arch")
        or "mobilenetv4_conv_small"
    )
    device = choose_device(args.device)
    model = build_model(arch, None, len(label_ids), pretrained=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    _, validation_transform = make_transforms(224)
    loader = DataLoader(
        CashlogImageDataset(
            validation_samples,
            validation_transform,
            args.bbox_padding,
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    expected: list[int] = []
    predicted: list[int] = []
    top3_hits = 0
    with torch.inference_mode():
        for images, targets in loader:
            logits = model(images.to(device))
            predictions = logits.argmax(dim=1).cpu()
            top3 = logits.topk(min(3, len(label_ids)), dim=1).indices.cpu()
            expected.extend(int(value) for value in targets)
            predicted.extend(int(value) for value in predictions)
            top3_hits += int(
                (top3 == targets.unsqueeze(1)).any(dim=1).sum().item()
            )

    report = classification_report(
        expected,
        predicted,
        labels=list(range(len(label_ids))),
        target_names=label_ids,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        expected,
        predicted,
        labels=list(range(len(label_ids))),
    )
    metrics = {
        "schema_version": 1,
        "scope": "uecfood-derived deterministic validation",
        "device": str(device),
        "architecture": arch,
        "dataset_samples": len(samples),
        "validation_samples": len(expected),
        "top1_accuracy": float(accuracy_score(expected, predicted)),
        "top3_accuracy": float(top3_hits / len(expected)),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "minimum_leaf_recall": min(
            float(report[leaf_id]["recall"]) for leaf_id in label_ids
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "labels_sha256": sha256(args.labels),
        "per_leaf": {
            leaf_id: {
                "precision": float(report[leaf_id]["precision"]),
                "recall": float(report[leaf_id]["recall"]),
                "f1": float(report[leaf_id]["f1-score"]),
                "support": int(report[leaf_id]["support"]),
            }
            for leaf_id in label_ids
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    confusion_path = args.output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *label_ids])
        for leaf_id, row in zip(label_ids, matrix.tolist(), strict=True):
            writer.writerow([leaf_id, *row])

    if not args.disable_mlflow and args.mlflow_tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.log_params(
                {
                    "component": "uecfood_four_meal_specialist_evaluation",
                    "device": str(device),
                    "architecture": arch,
                    "dataset_samples": len(samples),
                    "validation_samples": len(expected),
                    "checkpoint_sha256": metrics["checkpoint_sha256"],
                }
            )
            mlflow.log_metrics(
                {
                    "top1_accuracy": metrics["top1_accuracy"],
                    "top3_accuracy": metrics["top3_accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "minimum_leaf_recall": metrics["minimum_leaf_recall"],
                }
            )
            for leaf_id in label_ids:
                mlflow.log_metric(
                    f"recall_{leaf_id}",
                    metrics["per_leaf"][leaf_id]["recall"],
                )
            mlflow.log_artifact(str(metrics_path))
            mlflow.log_artifact(str(confusion_path))

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
