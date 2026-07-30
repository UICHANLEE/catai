#!/usr/bin/env python3
"""Export the compact CashLog visual model to FP32 and static INT8 ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import mlflow
import onnxruntime as ort
import torch
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
try:
    from scripts.train_cashlog33_million_mobilenet import (
        DEFAULT_DATASET,
        DEFAULT_OUTPUT,
        EvaluationDataset,
        build_model,
        make_transforms,
        read_jsonl,
    )
except ModuleNotFoundError:
    from train_cashlog33_million_mobilenet import (
        DEFAULT_DATASET,
        DEFAULT_OUTPUT,
        EvaluationDataset,
        build_model,
        make_transforms,
        read_jsonl,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT = DEFAULT_OUTPUT / "onnx"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def artifact_metadata(path: Path) -> dict[str, Any]:
    return {
        "path": repository_path(path),
        "bytes": path.stat().st_size,
        "megabytes": path.stat().st_size / (1024 * 1024),
        "sha256": sha256_file(path),
    }


def load_checkpoint_model(
    checkpoint_path: Path,
) -> tuple[torch.nn.Module, list[dict[str, Any]], dict[str, float]]:
    checkpoint = torch.load(
        checkpoint_path, map_location=torch.device("cpu"), weights_only=False
    )
    categories = list(checkpoint["categories"])
    model = build_model(
        Path(
            json.loads((checkpoint_path.parent / "config.json").read_text())[
                "weights"
            ]
        ),
        len(categories),
    )
    model.load_state_dict(
        checkpoint.get("ema_model") or checkpoint["model"]
    )
    probability_calibration = dict(
        checkpoint.get("validation", {}).get("probability_calibration", {})
    )
    temperature = float(probability_calibration.get("temperature", 1.0))
    if not np.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("checkpoint has an invalid serving temperature")
    probability_calibration["temperature"] = temperature
    return model.eval(), categories, probability_calibration


def load_calibration_arrays(
    dataset_dir: Path,
    categories: list[dict[str, Any]],
    image_size: int,
    sample_count: int,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, int]]:
    rows = [
        row
        for row in read_jsonl(dataset_dir / "original_manifest.jsonl")
        if row["partition"] == "validation"
    ]
    rng = np.random.default_rng(seed)
    rows_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_leaf[str(row["leaf_id"])].append(row)
    for leaf_rows in rows_by_leaf.values():
        rng.shuffle(leaf_rows)
    balanced_rows = []
    offsets = {leaf_id: 0 for leaf_id in rows_by_leaf}
    ordered_leaves = [
        str(category["id"])
        for category in categories
        if str(category["id"]) in rows_by_leaf
    ]
    while len(balanced_rows) < sample_count:
        added = False
        for leaf_id in ordered_leaves:
            offset = offsets[leaf_id]
            leaf_rows = rows_by_leaf[leaf_id]
            if offset < len(leaf_rows):
                balanced_rows.append(leaf_rows[offset])
                offsets[leaf_id] += 1
                added = True
                if len(balanced_rows) == sample_count:
                    break
        if not added:
            break
    rows = balanced_rows
    label_to_index = {
        str(category["id"]): index for index, category in enumerate(categories)
    }
    _, evaluation_transform = make_transforms(image_size)
    dataset = EvaluationDataset(rows, label_to_index, evaluation_transform)
    return (
        [
            dataset[index][0].numpy()[None, ...]
            for index in range(len(dataset))
        ],
        dict(sorted(Counter(str(row["leaf_id"]) for row in rows).items())),
    )


class ArrayCalibrationReader(CalibrationDataReader):
    def __init__(
        self, input_name: str, arrays: list[np.ndarray], batch_size: int
    ) -> None:
        self.input_name = input_name
        self.arrays = arrays
        self.batch_size = batch_size
        self._iterator: Iterator[dict[str, np.ndarray]] | None = None

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._iterator is None:
            self.rewind()
        return next(self._iterator, None)

    def rewind(self) -> None:
        self._iterator = iter(
            {
                self.input_name: np.concatenate(
                    self.arrays[offset : offset + self.batch_size], axis=0
                )
            }
            for offset in range(0, len(self.arrays), self.batch_size)
        )


def validate_exports(
    model: torch.nn.Module,
    fp32_path: Path,
    int8_path: Path,
    arrays: list[np.ndarray],
    repeats: int,
) -> dict[str, Any]:
    providers = ["CPUExecutionProvider"]
    fp32_session = ort.InferenceSession(str(fp32_path), providers=providers)
    int8_session = ort.InferenceSession(str(int8_path), providers=providers)
    input_name = fp32_session.get_inputs()[0].name
    validation = np.concatenate(arrays[: min(32, len(arrays))], axis=0)
    with torch.inference_mode():
        torch_logits = model(torch.from_numpy(validation)).numpy()
    fp32_logits = fp32_session.run(None, {input_name: validation})[0]
    int8_logits = int8_session.run(None, {input_name: validation})[0]

    sample = validation[:1]
    for _ in range(5):
        int8_session.run(None, {input_name: sample})
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        int8_session.run(None, {input_name: sample})
        timings.append((time.perf_counter() - started) * 1000.0)
    return {
        "validation_samples": len(validation),
        "pytorch_fp32_max_abs_error": float(
            np.max(np.abs(torch_logits - fp32_logits))
        ),
        "pytorch_fp32_top1_agreement": float(
            np.mean(torch_logits.argmax(axis=1) == fp32_logits.argmax(axis=1))
        ),
        "fp32_int8_top1_agreement": float(
            np.mean(fp32_logits.argmax(axis=1) == int8_logits.argmax(axis=1))
        ),
        "fp32_int8_mean_abs_error": float(
            np.mean(np.abs(fp32_logits - int8_logits))
        ),
        "int8_cpu_latency_ms": {
            "samples": len(timings),
            "mean": float(np.mean(timings)),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_OUTPUT / "best.pt")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--calibration-samples", type=int, default=512)
    parser.add_argument("--calibration-batch-size", type=int, default=32)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000_033)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5500"),
    )
    parser.add_argument(
        "--mlflow-run-file", type=Path, default=DEFAULT_OUTPUT / "mlflow_run.json"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model, categories, probability_calibration = load_checkpoint_model(
        args.checkpoint
    )
    labels_path = args.output_dir / "labels.json"
    labels_path.write_text(
        json.dumps(categories, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    arrays, calibration_leaf_counts = load_calibration_arrays(
        args.dataset_dir,
        categories,
        args.image_size,
        args.calibration_samples,
        args.seed,
    )
    if not arrays:
        raise SystemExit("no validation images are available for calibration")

    raw_fp32_path = args.output_dir / "cashlog33-visual-raw-fp32.onnx"
    fp32_path = args.output_dir / "cashlog33-visual-fp32.onnx"
    int8_path = args.output_dir / "cashlog33-visual-int8.onnx"
    dummy = torch.zeros(1, 3, args.image_size, args.image_size)
    torch.onnx.export(
        model,
        dummy,
        raw_fp32_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        dynamo=False,
    )
    quant_pre_process(
        input_model=raw_fp32_path,
        output_model_path=fp32_path,
        skip_optimization=False,
        skip_symbolic_shape=False,
        skip_onnx_shape=False,
    )
    raw_fp32_path.unlink()
    reader = ArrayCalibrationReader(
        "images", arrays, args.calibration_batch_size
    )
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=True,
        calibrate_method=CalibrationMethod.MinMax,
    )
    validation = validate_exports(
        model,
        fp32_path,
        int8_path,
        arrays,
        args.latency_repeats,
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "checkpoint": artifact_metadata(args.checkpoint),
        "labels": artifact_metadata(labels_path),
        "fp32": artifact_metadata(fp32_path),
        "int8": artifact_metadata(int8_path),
        "size_reduction_ratio": (
            1.0 - int8_path.stat().st_size / fp32_path.stat().st_size
        ),
        "probability_calibration": probability_calibration,
        "quantization": {
            "format": "QDQ",
            "activation": "QUInt8",
            "weight": "QInt8",
            "per_channel": True,
            "calibration_method": "MinMax",
            "calibration_samples": len(arrays),
            "calibration_leaf_counts": calibration_leaf_counts,
            "sampling": "class-balanced round-robin without replacement",
            "official_split": "Open Images train internal validation",
        },
        "validation": validation,
    }
    report_path = args.output_dir / "export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.mlflow_run_file.is_file():
        run_info = json.loads(args.mlflow_run_file.read_text(encoding="utf-8"))
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        with mlflow.start_run(run_id=str(run_info["run_id"])):
            mlflow.log_artifacts(str(args.output_dir), artifact_path="onnx")
            mlflow.log_metrics(
                {
                    "int8_model_megabytes": report["int8"]["megabytes"],
                    "onnx_size_reduction_ratio": report[
                        "size_reduction_ratio"
                    ],
                    "int8_cpu_latency_p50_ms": report["validation"][
                        "int8_cpu_latency_ms"
                    ]["p50"],
                    "fp32_int8_top1_agreement": report["validation"][
                        "fp32_int8_top1_agreement"
                    ],
                }
            )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
