#!/usr/bin/env python3
"""Build a rollback-safe CashLog serving candidate for compact ONNX vision."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_MODEL_DIR = (
    ROOT / "src/catai/assets/cashlog33_mobilenetv4_int8_v1"
)
DEFAULT_OUTPUT = ROOT / "configs/cashlog/hybrid.million-int8-candidate.json"


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
    output_path: Path,
    model_version: str,
) -> dict:
    config = json.loads(base_path.read_text(encoding="utf-8"))
    path_keys = [
        "categories",
        "semantics",
        "ocr_lexicon",
        "vision_model",
        "vision_head",
        "text_model",
        "ocr_detector_model",
        "ocr_classifier_model",
        "ocr_model",
    ]
    for key in path_keys:
        if key in config:
            source_path = Path(config[key])
            if not source_path.is_absolute():
                source_path = base_path.parent / source_path
            config[key] = relative_path(source_path, output_path.parent)
    specialist = config.get("meal_specialist", {})
    for key in ["checkpoint", "labels"]:
        if key in specialist:
            source_path = Path(specialist[key])
            if not source_path.is_absolute():
                source_path = base_path.parent / source_path
            specialist[key] = relative_path(source_path, output_path.parent)
    config["model_version"] = model_version
    provenance_path = model_path.parent / "provenance.json"
    if not provenance_path.is_file():
        raise RuntimeError(f"missing compact model provenance: {provenance_path}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    temperature = float(
        provenance["probability_calibration"]["temperature"]
    )
    if not temperature > 0:
        raise RuntimeError("compact model temperature must be positive")
    config["compact_vision"] = {
        "enabled": True,
        "runtime": "onnxruntime-int8",
        "model": relative_path(model_path, output_path.parent),
        "model_sha256": sha256_file(model_path),
        "labels": relative_path(labels_path, output_path.parent),
        "labels_sha256": sha256_file(labels_path),
        "input_size": 224,
        "temperature": temperature,
        "providers": ["CPUExecutionProvider"],
    }
    if "meal_specialist" in config:
        config["meal_specialist"]["enabled"] = False
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE)
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_DIR / "model.int8.onnx",
    )
    parser.add_argument(
        "--labels", type=Path, default=DEFAULT_MODEL_DIR / "labels.json"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--model-version", default="cashlog33-million-mobilenetv4-int8-v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.base_config, args.model, args.labels]:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    config = build_config(
        args.base_config,
        args.model,
        args.labels,
        args.output,
        args.model_version,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
