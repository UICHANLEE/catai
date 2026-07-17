#!/usr/bin/env python3
"""Train a lightweight CashLog visual head on frozen SigLIP2 embeddings."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/raw/cashlog33/openimages_v7/manifest.jsonl"
DEFAULT_SCORED = ROOT / "data/raw/cashlog33/openimages_v7/scored_manifest.jsonl"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_MODEL = ROOT / "models/siglip2-base-patch16-224"
DEFAULT_OUTPUT = ROOT / "checkpoints/cashlog33/vision_head_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def resolve_image_path(row: dict[str, Any]) -> Path:
    path = Path(str(row["relative_path"]))
    return path if path.is_absolute() else ROOT / path


def stratified_split(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_leaf[str(row["leaf_id"])].append(row)
    output = {"train": [], "val": [], "test": []}
    for leaf_id, leaf_rows in sorted(by_leaf.items()):
        ordered = sorted(leaf_rows, key=lambda row: stable_key(seed, str(row["sample_id"])))
        count = len(ordered)
        test_count = max(1, round(count * 0.20))
        val_count = max(1, round(count * 0.15)) if count >= 3 else 0
        while count - test_count - val_count < 1:
            if val_count:
                val_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                break
        output["test"].extend(ordered[:test_count])
        output["val"].extend(ordered[test_count : test_count + val_count])
        output["train"].extend(ordered[test_count + val_count :])
    return output


def augmented_views(image: Image.Image) -> list[Image.Image]:
    width, height = image.size
    crop_x = max(1, round(width * 0.05))
    crop_y = max(1, round(height * 0.05))
    cropped = image.crop((crop_x, crop_y, width - crop_x, height - crop_y)).resize(
        image.size, Image.Resampling.BICUBIC
    )
    return [
        image,
        ImageOps.mirror(image),
        cropped,
        ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.08)).enhance(1.10),
    ]


def encode_batch(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    device: torch.device,
) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = model.get_image_features(**inputs)
        if not isinstance(features, torch.Tensor):
            features = features.pooler_output
        features = F.normalize(features.float(), dim=-1)
    return features.cpu().numpy()


def encode_rows(
    model: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    augment: bool,
    batch_size: int,
    device: torch.device,
    label: str,
) -> tuple[np.ndarray, list[str], list[str]]:
    batches: list[np.ndarray] = []
    labels: list[str] = []
    sample_ids: list[str] = []
    batch_images: list[Image.Image] = []
    total = len(rows) * (4 if augment else 1)
    processed = 0

    def flush() -> None:
        nonlocal processed
        if not batch_images:
            return
        batches.append(encode_batch(model, processor, batch_images, device))
        processed += len(batch_images)
        for image in batch_images:
            image.close()
        batch_images.clear()
        print(f"{label} embeddings {processed}/{total}", flush=True)

    for row in rows:
        with Image.open(resolve_image_path(row)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
        views = augmented_views(image) if augment else [image]
        for view_index, view in enumerate(views):
            batch_images.append(view)
            labels.append(str(row["leaf_id"]))
            sample_ids.append(f"{row['sample_id']}#view{view_index}")
            if len(batch_images) >= batch_size:
                flush()
    flush()
    return np.concatenate(batches, axis=0), labels, sample_ids


def top3_accuracy(labels: list[str], probabilities: np.ndarray, classes: list[str]) -> float:
    class_index = {leaf_id: index for index, leaf_id in enumerate(classes)}
    truth = [class_index[label] for label in labels]
    indexes = np.argpartition(-probabilities, kth=min(2, probabilities.shape[1] - 1), axis=1)[:, :3]
    return float(np.mean([expected in candidates for expected, candidates in zip(truth, indexes, strict=True)]))


def evaluate_model(
    model: LogisticRegression,
    embeddings: np.ndarray,
    labels: list[str],
    supported: list[str],
) -> tuple[dict[str, float], np.ndarray, dict[str, dict[str, Any]]]:
    probabilities = model.predict_proba(embeddings)
    classes = [str(value) for value in model.classes_]
    predicted = [classes[index] for index in probabilities.argmax(axis=1)]
    report = classification_report(
        labels, predicted, labels=supported, output_dict=True, zero_division=0
    )
    metrics = {
        "samples": len(labels),
        "top1_accuracy": float(accuracy_score(labels, predicted)),
        "top3_accuracy": top3_accuracy(labels, probabilities, classes),
        "macro_f1": float(
            f1_score(labels, predicted, labels=supported, average="macro", zero_division=0)
        ),
    }
    matrix = confusion_matrix(labels, predicted, labels=supported)
    per_class = {
        leaf_id: {
            "precision": float(report[leaf_id]["precision"]),
            "recall": float(report[leaf_id]["recall"]),
            "f1": float(report[leaf_id]["f1-score"]),
            "support": int(report[leaf_id]["support"]),
        }
        for leaf_id in supported
    }
    return metrics, matrix, per_class


def zero_shot_metrics(
    test_rows: list[dict[str, Any]], scored_rows: list[dict[str, Any]], supported: list[str]
) -> dict[str, float]:
    scored_by_id = {str(row["sample_id"]): row for row in scored_rows}
    actual: list[str] = []
    predicted: list[str] = []
    top3_hits: list[bool] = []
    for row in test_rows:
        scored = scored_by_id[str(row["sample_id"])]
        top3 = [str(value["leaf_id"]) for value in scored["review_details"]["top3"]]
        actual.append(str(row["leaf_id"]))
        predicted.append(top3[0])
        top3_hits.append(str(row["leaf_id"]) in top3)
    return {
        "samples": len(actual),
        "top1_accuracy": float(accuracy_score(actual, predicted)),
        "top3_accuracy": float(sum(top3_hits) / len(top3_hits)),
        "macro_f1": float(
            f1_score(actual, predicted, labels=supported, average="macro", zero_division=0)
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scored-manifest", type=Path, default=DEFAULT_SCORED)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--vision-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=250716)
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="cashlog33-hybrid-v2")
    parser.add_argument("--mlflow-run-name", default="cashlog33-siglip2-linear-head-v1")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.manifest)
    scored_rows = read_jsonl(args.scored_manifest)
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    category_order = [str(row["id"]) for row in categories]
    supported = [leaf_id for leaf_id in category_order if leaf_id in {str(row["leaf_id"]) for row in rows}]
    splits = stratified_split(rows, args.seed)
    split_counts = {
        split: dict(sorted(Counter(str(row["leaf_id"]) for row in split_rows).items()))
        for split, split_rows in splits.items()
    }
    if set(split_counts["train"]) != set(supported) or set(split_counts["test"]) != set(supported):
        raise RuntimeError("every supported visual leaf must have train and test examples")

    device = choose_device(args.device)
    vision_model = AutoModel.from_pretrained(args.vision_model, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.vision_model, local_files_only=True, use_fast=True
    )
    started = time.perf_counter()
    train_embeddings, train_labels, train_ids = encode_rows(
        vision_model, processor, splits["train"], True, args.batch_size, device, "train"
    )
    val_embeddings, val_labels, val_ids = encode_rows(
        vision_model, processor, splits["val"], False, args.batch_size, device, "val"
    )
    test_embeddings, test_labels, test_ids = encode_rows(
        vision_model, processor, splits["test"], False, args.batch_size, device, "test"
    )
    embedding_seconds = time.perf_counter() - started
    del vision_model, processor
    gc.collect()
    np.savez_compressed(
        args.output_dir / "embedding_cache.npz",
        train=train_embeddings,
        train_labels=np.asarray(train_labels),
        val=val_embeddings,
        val_labels=np.asarray(val_labels),
        test=test_embeddings,
        test_labels=np.asarray(test_labels),
    )

    candidates: list[tuple[float, float, LogisticRegression, dict[str, float]]] = []
    for c_value in [0.03, 0.10, 0.30, 1.0, 3.0, 10.0]:
        head = LogisticRegression(
            C=c_value,
            class_weight="balanced",
            max_iter=2000,
            random_state=args.seed,
            solver="lbfgs",
        )
        head.fit(train_embeddings, train_labels)
        if len(val_labels):
            val_metrics, _, _ = evaluate_model(head, val_embeddings, val_labels, supported)
        else:
            val_metrics, _, _ = evaluate_model(head, train_embeddings, train_labels, supported)
        candidates.append((val_metrics["macro_f1"], val_metrics["top1_accuracy"], head, val_metrics))
        print(f"C={c_value:.2f} val={val_metrics}", flush=True)
    candidates.sort(key=lambda item: (-item[0], -item[1]))
    selected_head = candidates[0][2]
    selected_c = float(selected_head.C)
    validation_metrics = candidates[0][3]
    test_metrics, test_matrix, per_class = evaluate_model(
        selected_head, test_embeddings, test_labels, supported
    )
    baseline_metrics = zero_shot_metrics(splits["test"], scored_rows, supported)
    selected = (
        test_metrics["top1_accuracy"] > baseline_metrics["top1_accuracy"]
        and test_metrics["macro_f1"] > baseline_metrics["macro_f1"]
    )
    model_sha256 = hashlib.sha256((args.vision_model / "model.safetensors").read_bytes()).hexdigest()
    artifact = {
        "schema_version": 1,
        "model": selected_head,
        "classes": [str(value) for value in selected_head.classes_],
        "supported_leaves": supported,
        "vision_model_sha256": model_sha256,
        "selected_c": selected_c,
        "created_at": utc_now(),
    }
    head_path = args.output_dir / "vision_head.joblib"
    joblib.dump(artifact, head_path, compress=3)
    metrics = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": "Open Images V7 validation source-label proxy",
        "supported_leaf_count": len(supported),
        "taxonomy_leaf_count": len(category_order),
        "split_counts": split_counts,
        "train_augmented_samples": len(train_labels),
        "embedding_seconds": embedding_seconds,
        "selected_c": selected_c,
        "validation": validation_metrics,
        "test_linear_head": test_metrics,
        "test_zero_shot_same_split": baseline_metrics,
        "selected_for_hybrid": selected,
        "selection_rule": "linear head must improve both Top-1 and macro-F1 on the same source-group holdout",
        "scope_warning": "Open Images object-to-expense mapping is a proxy, not real CashLog accuracy.",
    }
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
        for leaf_id in supported:
            writer.writerow({"leaf_id": leaf_id, **per_class[leaf_id]})
    confusion_path = args.output_dir / "confusion_matrix.csv"
    with confusion_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *supported])
        for leaf_id, values in zip(supported, test_matrix.tolist(), strict=True):
            writer.writerow([leaf_id, *values])
    split_path = args.output_dir / "split_manifest.jsonl"
    with split_path.open("w", encoding="utf-8") as handle:
        for split, split_rows in splits.items():
            for row in split_rows:
                handle.write(
                    json.dumps(
                        {"sample_id": row["sample_id"], "leaf_id": row["leaf_id"], "split": split},
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )

    if not args.disable_mlflow and args.mlflow_tracking_uri:
        import mlflow

        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=args.mlflow_run_name):
            mlflow.log_params(
                {
                    "component": "siglip2_linear_head",
                    "supported_leaf_count": len(supported),
                    "selected_c": selected_c,
                    "train_augmented_samples": len(train_labels),
                    "selected_for_hybrid": selected,
                }
            )
            mlflow.log_metrics(
                {
                    "vision_head_top1": test_metrics["top1_accuracy"],
                    "vision_head_top3": test_metrics["top3_accuracy"],
                    "vision_head_macro_f1": test_metrics["macro_f1"],
                    "vision_zero_shot_same_split_top1": baseline_metrics["top1_accuracy"],
                    "vision_zero_shot_same_split_macro_f1": baseline_metrics["macro_f1"],
                }
            )
            for path in [head_path, metrics_path, per_class_path, confusion_path, split_path]:
                mlflow.log_artifact(str(path))
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
