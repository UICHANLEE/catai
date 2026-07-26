#!/usr/bin/env python3
"""Build a checksum-pinned hybrid candidate configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEMPLATE = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_OUTPUT = ROOT / "configs/cashlog/hybrid.airflow-candidate.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_from(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def relative_to(path: Path, parent: Path) -> str:
    try:
        return Path(os.path.relpath(path.resolve(), parent.resolve())).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--vision-head", type=Path, required=True)
    parser.add_argument("--text-model", type=Path, required=True)
    parser.add_argument("--meal-specialist-checkpoint", type=Path)
    parser.add_argument("--meal-specialist-labels", type=Path)
    parser.add_argument("--meal-specialist-blend-weight", type=float, default=0.65)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-version", default="cashlog33-hybrid-airflow-candidate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template_path = args.template.resolve()
    output_path = args.output.resolve()
    config: dict[str, Any] = json.loads(template_path.read_text(encoding="utf-8"))
    vision_head = args.vision_head.resolve()
    text_model = args.text_model.resolve()
    for path in (vision_head, text_model):
        if not path.is_file():
            raise SystemExit(f"candidate artifact not found: {path}")
    if bool(args.meal_specialist_checkpoint) != bool(args.meal_specialist_labels):
        raise SystemExit(
            "meal specialist checkpoint and labels must be provided together"
        )

    path_keys = [
        "categories",
        "semantics",
        "ocr_lexicon",
        "vision_model",
        "ocr_detector_model",
        "ocr_classifier_model",
        "ocr_model",
    ]
    resolved = {key: resolve_from(template_path, str(config[key])) for key in path_keys}
    config.update(
        {
            "model_version": args.model_version,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_template": relative_to(template_path, output_path.parent),
            "vision_head": relative_to(vision_head, output_path.parent),
            "vision_head_sha256": sha256(vision_head),
            "text_model": relative_to(text_model, output_path.parent),
            "text_model_sha256": sha256(text_model),
        }
    )
    for key, path in resolved.items():
        config[key] = relative_to(path, output_path.parent)
    model_file = resolved["vision_model"] / "model.safetensors"
    config["vision_model_sha256"] = sha256(model_file)
    config["ocr_detector_model_sha256"] = sha256(resolved["ocr_detector_model"])
    config["ocr_classifier_model_sha256"] = sha256(resolved["ocr_classifier_model"])
    config["ocr_model_sha256"] = sha256(resolved["ocr_model"])
    if args.meal_specialist_checkpoint and args.meal_specialist_labels:
        specialist_checkpoint = args.meal_specialist_checkpoint.resolve()
        specialist_labels = args.meal_specialist_labels.resolve()
        for path in (specialist_checkpoint, specialist_labels):
            if not path.is_file():
                raise SystemExit(f"meal specialist artifact not found: {path}")
        if not 0.0 <= args.meal_specialist_blend_weight <= 1.0:
            raise SystemExit("meal specialist blend weight must be between 0 and 1")
        config["meal_specialist"] = {
            "enabled": True,
            "checkpoint": relative_to(specialist_checkpoint, output_path.parent),
            "checkpoint_sha256": sha256(specialist_checkpoint),
            "labels": relative_to(specialist_labels, output_path.parent),
            "labels_sha256": sha256(specialist_labels),
            "blend_weight": args.meal_specialist_blend_weight,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(output_path)
    print(
        json.dumps(
            {
                "config": str(output_path),
                "model_version": config["model_version"],
                "vision_head_sha256": config["vision_head_sha256"],
                "text_model_sha256": config["text_model_sha256"],
                "meal_specialist_checkpoint_sha256": (
                    config.get("meal_specialist", {}).get("checkpoint_sha256")
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
