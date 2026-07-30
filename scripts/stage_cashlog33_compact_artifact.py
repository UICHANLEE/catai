#!/usr/bin/env python3
"""Stage the evaluated compact model as a small, versioned serving asset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPORT_DIR = ROOT / "checkpoints/cashlog33/mobilenet_million_v1/onnx"
DEFAULT_OUTPUT_DIR = (
    ROOT / "src/catai/assets/cashlog33_mobilenetv4_int8_v1"
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


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def stage_artifacts(
    model_source: Path,
    labels_source: Path,
    export_report_source: Path,
    output_dir: Path,
    model_version: str,
) -> dict[str, Any]:
    export_report = json.loads(
        export_report_source.read_text(encoding="utf-8")
    )
    model_sha = sha256_file(model_source)
    labels_sha = sha256_file(labels_source)
    if model_sha != str(export_report["int8"]["sha256"]):
        raise RuntimeError("INT8 source does not match the export report")
    if labels_sha != str(export_report["labels"]["sha256"]):
        raise RuntimeError("labels source does not match the export report")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_destination = output_dir / "model.int8.onnx"
    labels_destination = output_dir / "labels.json"
    export_report_destination = output_dir / "export_report.json"
    atomic_copy(model_source, model_destination)
    atomic_copy(labels_source, labels_destination)
    atomic_copy(export_report_source, export_report_destination)
    provenance = {
        "schema_version": 1,
        "staged_at": datetime.now(timezone.utc).isoformat(),
        "model_version": model_version,
        "runtime": "onnxruntime-int8",
        "quantization": export_report["quantization"],
        "probability_calibration": export_report[
            "probability_calibration"
        ],
        "source": {
            "model": repository_path(model_source),
            "labels": repository_path(labels_source),
            "export_report": repository_path(export_report_source),
        },
        "artifacts": {
            "model": {
                "path": repository_path(model_destination),
                "bytes": model_destination.stat().st_size,
                "sha256": sha256_file(model_destination),
            },
            "labels": {
                "path": repository_path(labels_destination),
                "bytes": labels_destination.stat().st_size,
                "sha256": sha256_file(labels_destination),
            },
            "export_report": {
                "path": repository_path(export_report_destination),
                "bytes": export_report_destination.stat().st_size,
                "sha256": sha256_file(export_report_destination),
            },
        },
    }
    provenance_path = output_dir / "provenance.json"
    temporary = provenance_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, provenance_path)
    return provenance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_EXPORT_DIR / "cashlog33-visual-int8.onnx",
    )
    parser.add_argument(
        "--labels", type=Path, default=DEFAULT_EXPORT_DIR / "labels.json"
    )
    parser.add_argument(
        "--export-report",
        type=Path,
        default=DEFAULT_EXPORT_DIR / "export_report.json",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--model-version", default="cashlog33-million-mobilenetv4-int8-v1"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.model, args.labels, args.export_report]:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    result = stage_artifacts(
        args.model,
        args.labels,
        args.export_report,
        args.output_dir,
        args.model_version,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
