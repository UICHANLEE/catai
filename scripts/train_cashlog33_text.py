#!/usr/bin/env python3
"""Train and evaluate the 33-leaf OCR/transaction text classifier."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import FeatureUnion, Pipeline


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "data/processed/cashlog33/text/v1/manifest.jsonl"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_OUTPUT = ROOT / "checkpoints/cashlog33/text_sgd_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def update_progress(path: Path, stage: str, status: str, **details: Any) -> None:
    payload = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "stage": stage,
        "status": status,
        **details,
    }
    write_json(path, payload)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def build_pipeline(seed: int, max_word_features: int, max_char_features: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=max_word_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=max_char_features,
                    sublinear_tf=True,
                    dtype=np.float32,
                ),
            ),
        ]
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=150,
        tol=1e-5,
        class_weight="balanced",
        average=True,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("features", features), ("classifier", classifier)])


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    values = np.exp(shifted)
    return values / values.sum(axis=1, keepdims=True)


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
    return softmax(logits)


def fit_temperature(
    probabilities: np.ndarray, labels: list[str], classes: list[str]
) -> tuple[float, float]:
    candidates = np.linspace(0.50, 3.00, 101)
    losses = [
        log_loss(labels, apply_temperature(probabilities, float(value)), labels=classes)
        for value in candidates
    ]
    index = int(np.argmin(losses))
    return float(candidates[index]), float(losses[index])


def expected_calibration_error(
    probabilities: np.ndarray, true_indexes: np.ndarray, bins: int = 10
) -> float:
    confidence = probabilities.max(axis=1)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == true_indexes
    edges = np.linspace(0.0, 1.0, bins + 1)
    error = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        mask = (confidence > lower) & (confidence <= upper)
        if not mask.any():
            continue
        error += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return error


def evaluate(
    model: Pipeline,
    rows: list[dict[str, Any]],
    category_ids: list[str],
    temperature: float,
) -> tuple[dict[str, Any], np.ndarray, dict[str, dict[str, Any]]]:
    texts = [str(row["text"]) for row in rows]
    labels = [str(row["leaf_id"]) for row in rows]
    raw_probabilities = model.predict_proba(texts)
    classes = [str(value) for value in model.named_steps["classifier"].classes_]
    probabilities = apply_temperature(raw_probabilities, temperature)
    predicted_indexes = probabilities.argmax(axis=1)
    predictions = [classes[index] for index in predicted_indexes]
    class_index = {leaf_id: index for index, leaf_id in enumerate(classes)}
    true_indexes = np.asarray([class_index[label] for label in labels])
    top3_indexes = np.argpartition(-probabilities, kth=min(2, probabilities.shape[1] - 1), axis=1)[:, :3]
    top3 = float(np.mean([truth in candidates for truth, candidates in zip(true_indexes, top3_indexes, strict=True)]))
    matrix = confusion_matrix(labels, predictions, labels=category_ids)
    report = classification_report(
        labels,
        predictions,
        labels=category_ids,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "samples": len(rows),
        "top1_accuracy": float(accuracy_score(labels, predictions)),
        "top3_accuracy": top3,
        "macro_f1": float(f1_score(labels, predictions, labels=category_ids, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(labels, predictions, labels=category_ids, average="weighted", zero_division=0)
        ),
        "log_loss": float(log_loss(labels, probabilities, labels=classes)),
        "ece_10_bin": expected_calibration_error(probabilities, true_indexes),
    }
    per_class = {
        leaf_id: {
            "precision": float(report[leaf_id]["precision"]),
            "recall": float(report[leaf_id]["recall"]),
            "f1": float(report[leaf_id]["f1-score"]),
            "support": int(report[leaf_id]["support"]),
        }
        for leaf_id in category_ids
    }
    return metrics, matrix, per_class


def subgroup_metrics(
    model: Pipeline,
    rows: list[dict[str, Any]],
    category_ids: list[str],
    temperature: float,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for provenance in sorted({str(row["provenance_type"]) for row in rows}):
        subset = [row for row in rows if str(row["provenance_type"]) == provenance]
        metrics, _, _ = evaluate(model, subset, category_ids, temperature)
        output[provenance] = metrics
    return output


def write_per_class(path: Path, values: dict[str, dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["leaf_id", "precision", "recall", "f1", "support"],
            lineterminator="\n",
        )
        writer.writeheader()
        for leaf_id, metrics in values.items():
            writer.writerow({"leaf_id": leaf_id, **metrics})


def write_confusion(path: Path, matrix: np.ndarray, category_ids: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["actual\\predicted", *category_ids])
        for leaf_id, row in zip(category_ids, matrix.tolist(), strict=True):
            writer.writerow([leaf_id, *row])


def start_mlflow(args: argparse.Namespace) -> Any | None:
    if args.disable_mlflow or not args.mlflow_tracking_uri:
        return None
    import mlflow

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    return mlflow.start_run(run_name=args.mlflow_run_name or f"cashlog33-text-{int(time.time())}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=250716)
    parser.add_argument("--max-word-features", type=int, default=80000)
    parser.add_argument("--max-char-features", type=int, default=120000)
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default="cashlog33-hybrid")
    parser.add_argument("--mlflow-run-name")
    parser.add_argument("--disable-mlflow", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    update_progress(progress_path, "load", "running")
    categories = load_json(args.categories)
    category_ids = [str(row["id"]) for row in categories]
    rows = read_jsonl(args.manifest)
    by_split = {
        split: [row for row in rows if str(row.get("split")) == split]
        for split in ["train", "val", "test"]
    }
    for split, split_rows in by_split.items():
        covered = {str(row["leaf_id"]) for row in split_rows}
        if covered != set(category_ids):
            raise SystemExit(f"{split} does not cover all 33 leaves: {sorted(set(category_ids) - covered)}")
    input_sha256 = hashlib.sha256(args.manifest.read_bytes()).hexdigest()

    mlflow_run = start_mlflow(args)
    try:
        update_progress(
            progress_path,
            "fit",
            "running",
            train_samples=len(by_split["train"]),
            class_count=len(category_ids),
        )
        started = time.perf_counter()
        model = build_pipeline(args.seed, args.max_word_features, args.max_char_features)
        model.fit(
            [str(row["text"]) for row in by_split["train"]],
            [str(row["leaf_id"]) for row in by_split["train"]],
        )
        fit_seconds = time.perf_counter() - started

        update_progress(progress_path, "calibrate", "running", fit_seconds=fit_seconds)
        validation_raw = model.predict_proba([str(row["text"]) for row in by_split["val"]])
        classes = [str(value) for value in model.named_steps["classifier"].classes_]
        temperature, validation_calibrated_loss = fit_temperature(
            validation_raw,
            [str(row["leaf_id"]) for row in by_split["val"]],
            classes,
        )

        update_progress(progress_path, "evaluate", "running", temperature=temperature)
        validation_metrics, _, _ = evaluate(
            model, by_split["val"], category_ids, temperature
        )
        test_metrics, test_matrix, per_class = evaluate(
            model, by_split["test"], category_ids, temperature
        )
        subgroups = subgroup_metrics(model, by_split["test"], category_ids, temperature)
        metrics = {
            "schema_version": 1,
            "generated_at": utc_now(),
            "model_family": "tfidf_word_char_sgd_log_loss",
            "taxonomy_leaf_count": len(category_ids),
            "fit_seconds": fit_seconds,
            "temperature": temperature,
            "validation_calibrated_log_loss": validation_calibrated_loss,
            "validation": validation_metrics,
            "test": test_metrics,
            "test_by_provenance": subgroups,
            "split_counts": {split: len(split_rows) for split, split_rows in by_split.items()},
            "input_manifest": str(args.manifest.resolve()),
            "input_manifest_sha256": input_sha256,
            "test_scope": "synthetic/weak proxy only; not a real CashLog production holdout",
        }
        model_artifact = {
            "schema_version": 1,
            "model": model,
            "temperature": temperature,
            "categories": categories,
            "classes": classes,
            "input_manifest_sha256": input_sha256,
            "created_at": utc_now(),
        }
        model_path = args.output_dir / "text_model.joblib"
        joblib.dump(model_artifact, model_path, compress=3)
        write_json(args.output_dir / "metrics.json", metrics)
        write_per_class(args.output_dir / "per_class_metrics.csv", per_class)
        write_confusion(args.output_dir / "confusion_matrix.csv", test_matrix, category_ids)
        write_json(
            args.output_dir / "training_config.json",
            {
                "seed": args.seed,
                "max_word_features": args.max_word_features,
                "max_char_features": args.max_char_features,
                "class_counts_train": dict(
                    sorted(Counter(str(row["leaf_id"]) for row in by_split["train"]).items())
                ),
            },
        )

        if mlflow_run is not None:
            import mlflow

            mlflow.log_params(
                {
                    "component": "ocr_text",
                    "taxonomy_leaf_count": len(category_ids),
                    "seed": args.seed,
                    "train_samples": len(by_split["train"]),
                    "val_samples": len(by_split["val"]),
                    "test_samples": len(by_split["test"]),
                    "input_manifest_sha256": input_sha256,
                }
            )
            mlflow.log_metrics(
                {
                    "val_top1": validation_metrics["top1_accuracy"],
                    "val_macro_f1": validation_metrics["macro_f1"],
                    "test_top1": test_metrics["top1_accuracy"],
                    "test_top3": test_metrics["top3_accuracy"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "test_ece": test_metrics["ece_10_bin"],
                    "fit_seconds": fit_seconds,
                }
            )
            for artifact in [
                model_path,
                args.output_dir / "metrics.json",
                args.output_dir / "per_class_metrics.csv",
                args.output_dir / "confusion_matrix.csv",
                args.output_dir / "training_config.json",
            ]:
                mlflow.log_artifact(str(artifact))
            mlflow.end_run(status="FINISHED")
        update_progress(
            progress_path,
            "complete",
            "succeeded",
            model_path=str(model_path.resolve()),
            test_top1=test_metrics["top1_accuracy"],
            test_top3=test_metrics["top3_accuracy"],
            test_macro_f1=test_metrics["macro_f1"],
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        if mlflow_run is not None:
            import mlflow

            mlflow.set_tag("failure", f"{type(exc).__name__}: {exc}")
            mlflow.end_run(status="FAILED")
        update_progress(progress_path, "failed", "failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
