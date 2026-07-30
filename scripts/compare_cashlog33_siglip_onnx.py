#!/usr/bin/env python3
"""Gate same-architecture SigLIP retraining and its quantized ONNX export."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = (
    ROOT / "reports/cashlog33/million_v1/current_serving_visual_baseline.json"
)
DEFAULT_NATIVE = (
    ROOT / "reports/cashlog33/siglip_expanded_v1/candidate_serving_visual.json"
)
DEFAULT_INT8 = (
    ROOT / "reports/cashlog33/siglip_expanded_v1/int8_serving_visual.json"
)
DEFAULT_EXPORT = (
    ROOT
    / "checkpoints/cashlog33/vision_head_siglip_expanded_v1/onnx/"
    "export_report.json"
)
DEFAULT_MODEL = (
    ROOT
    / "checkpoints/cashlog33/vision_head_siglip_expanded_v1/onnx/"
    "cashlog33-siglip-vision-int8.onnx"
)
DEFAULT_SERVING = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_OUTPUT = (
    ROOT / "reports/cashlog33/siglip_expanded_v1/final_comparison.json"
)


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


def metrics(report: dict[str, Any]) -> dict[str, Any]:
    return dict(report["baseline"])


def build_comparison(
    baseline_report: dict[str, Any],
    native_report: dict[str, Any],
    int8_report: dict[str, Any],
    export_report: dict[str, Any],
    *,
    serving_config_sha256: str,
    model_path: Path,
    max_recall_regression: float,
    minimum_recall_support: int,
    max_quantized_top1_drift: float,
    max_quantized_macro_f1_drift: float,
) -> dict[str, Any]:
    baseline = metrics(baseline_report)
    native = metrics(native_report)
    int8 = metrics(int8_report)
    recall_leaves = [
        leaf_id
        for leaf_id, support in baseline["per_leaf_support"].items()
        if int(support) >= minimum_recall_support
    ]
    native_recall_delta = {
        leaf_id: float(native["per_leaf_recall"][leaf_id])
        - float(baseline["per_leaf_recall"][leaf_id])
        for leaf_id in recall_leaves
    }
    int8_recall_delta = {
        leaf_id: float(int8["per_leaf_recall"][leaf_id])
        - float(baseline["per_leaf_recall"][leaf_id])
        for leaf_id in recall_leaves
    }
    quantized_top1_drift = (
        float(int8["top1_accuracy"]) - float(native["top1_accuracy"])
    )
    quantized_macro_f1_drift = (
        float(int8["macro_f1"]) - float(native["macro_f1"])
    )
    gate_checks = {
        "native_top1_beats_current": (
            native["top1_accuracy"] > baseline["top1_accuracy"]
        ),
        "native_macro_f1_beats_current": (
            native["macro_f1"] > baseline["macro_f1"]
        ),
        "native_leaf_recall_regression_within_limit": (
            min(native_recall_delta.values(), default=0.0)
            >= -max_recall_regression
        ),
        "int8_top1_beats_current": (
            int8["top1_accuracy"] > baseline["top1_accuracy"]
        ),
        "int8_macro_f1_beats_current": (
            int8["macro_f1"] > baseline["macro_f1"]
        ),
        "int8_leaf_recall_regression_within_limit": (
            min(int8_recall_delta.values(), default=0.0)
            >= -max_recall_regression
        ),
        "int8_top1_drift_within_limit": (
            quantized_top1_drift >= -max_quantized_top1_drift
        ),
        "int8_macro_f1_drift_within_limit": (
            quantized_macro_f1_drift >= -max_quantized_macro_f1_drift
        ),
        "int8_smaller_than_current_visual_stack": (
            model_path.stat().st_size < 1_569_777_120
        ),
        "int8_faster_than_current_visual_path": (
            int8["runtime"]["p50_vision_ms"]
            < baseline["runtime"]["p50_vision_ms"]
        ),
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "experiment_contract": (
            "same SigLIP2 vision architecture and frozen external holdout; "
            "only train data changes before ONNX quantization"
        ),
        "serving_config_sha256": serving_config_sha256,
        "evaluation_samples": baseline["samples"],
        "baseline": baseline,
        "native": native,
        "int8": {
            **int8,
            "artifact": {
                "path": repository_path(model_path),
                "bytes": model_path.stat().st_size,
                "sha256": sha256_file(model_path),
            },
        },
        "export": export_report,
        "native_delta_vs_current": {
            "top1_accuracy": (
                native["top1_accuracy"] - baseline["top1_accuracy"]
            ),
            "top3_accuracy": (
                native["top3_accuracy"] - baseline["top3_accuracy"]
            ),
            "macro_f1": native["macro_f1"] - baseline["macro_f1"],
            "per_leaf_recall": native_recall_delta,
        },
        "int8_delta_vs_current": {
            "top1_accuracy": int8["top1_accuracy"] - baseline["top1_accuracy"],
            "top3_accuracy": int8["top3_accuracy"] - baseline["top3_accuracy"],
            "macro_f1": int8["macro_f1"] - baseline["macro_f1"],
            "per_leaf_recall": int8_recall_delta,
        },
        "quantization_delta": {
            "top1_accuracy": quantized_top1_drift,
            "top3_accuracy": (
                int8["top3_accuracy"] - native["top3_accuracy"]
            ),
            "macro_f1": quantized_macro_f1_drift,
        },
        "gate_thresholds": {
            "max_leaf_recall_regression": max_recall_regression,
            "minimum_leaf_recall_support": minimum_recall_support,
            "max_quantized_top1_drift": max_quantized_top1_drift,
            "max_quantized_macro_f1_drift": max_quantized_macro_f1_drift,
        },
        "gate_checks": gate_checks,
        "promotion_gate_passed": all(gate_checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--native", type=Path, default=DEFAULT_NATIVE)
    parser.add_argument("--int8", type=Path, default=DEFAULT_INT8)
    parser.add_argument("--export-report", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--serving-config", type=Path, default=DEFAULT_SERVING)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-recall-regression", type=float, default=0.10)
    parser.add_argument("--minimum-recall-support", type=int, default=30)
    parser.add_argument("--max-quantized-top1-drift", type=float, default=0.005)
    parser.add_argument(
        "--max-quantized-macro-f1-drift", type=float, default=0.01
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_comparison(
        json.loads(args.baseline.read_text(encoding="utf-8")),
        json.loads(args.native.read_text(encoding="utf-8")),
        json.loads(args.int8.read_text(encoding="utf-8")),
        json.loads(args.export_report.read_text(encoding="utf-8")),
        serving_config_sha256=sha256_file(args.serving_config),
        model_path=args.model,
        max_recall_regression=args.max_recall_regression,
        minimum_recall_support=args.minimum_recall_support,
        max_quantized_top1_drift=args.max_quantized_top1_drift,
        max_quantized_macro_f1_drift=args.max_quantized_macro_f1_drift,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
