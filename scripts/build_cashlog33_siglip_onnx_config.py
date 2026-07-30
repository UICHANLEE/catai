#!/usr/bin/env python3
"""Build a checksum-pinned serving candidate for quantized SigLIP2 ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs/cashlog/hybrid.siglip-expanded-candidate.json"
DEFAULT_EXPORT = (
    ROOT
    / "checkpoints/cashlog33/vision_head_siglip_expanded_v1/onnx"
)
DEFAULT_OUTPUT = ROOT / "configs/cashlog/hybrid.siglip-onnx-candidate.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_path(path: Path, parent: Path) -> str:
    return Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix()


def build_config(
    base_path: Path,
    model_path: Path,
    labels_path: Path,
    report_path: Path,
    output_path: Path,
    model_version: str,
) -> dict:
    config = json.loads(base_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256_file(model_path) != str(report["int8"]["sha256"]):
        raise ValueError("INT8 model does not match export report")
    if sha256_file(labels_path) != str(report["labels"]["sha256"]):
        raise ValueError("labels do not match export report")
    for key in [
        "categories",
        "semantics",
        "ocr_lexicon",
        "vision_model",
        "vision_head",
        "text_model",
        "ocr_detector_model",
        "ocr_classifier_model",
        "ocr_model",
    ]:
        if key in config:
            path = Path(str(config[key]))
            if not path.is_absolute():
                path = base_path.parent / path
            config[key] = relative_path(path, output_path.parent)
    specialist = config.get("meal_specialist", {})
    for key in ["checkpoint", "labels"]:
        if key in specialist:
            path = Path(str(specialist[key]))
            if not path.is_absolute():
                path = base_path.parent / path
            specialist[key] = relative_path(path, output_path.parent)
    config["model_version"] = model_version
    config["compact_vision"] = {
        "enabled": True,
        "runtime": "onnxruntime-int8",
        "member_name": "siglip2_int8_onnx",
        "model": relative_path(model_path, output_path.parent),
        "model_sha256": sha256_file(model_path),
        "labels": relative_path(labels_path, output_path.parent),
        "labels_sha256": sha256_file(labels_path),
        "providers": ["CPUExecutionProvider"],
        "input_size": 224,
        "preprocess": "siglip_resize",
        "output_kind": "probabilities",
        "temperature": 1.0,
        "quantization": report["quantization"],
    }
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_EXPORT / "cashlog33-siglip-vision-int8.onnx",
    )
    parser.add_argument(
        "--labels", type=Path, default=DEFAULT_EXPORT / "labels.json"
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=DEFAULT_EXPORT / "export_report.json",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-version", default="cashlog33-same-siglip-expanded-int8-v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config(
        args.base_config,
        args.model,
        args.labels,
        args.export_report,
        args.output,
        args.model_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
