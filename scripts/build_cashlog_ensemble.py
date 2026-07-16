#!/usr/bin/env python3
"""Build a Cashlog weighted-logit ensemble manifest from trained checkpoints."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]


def read_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"unsupported checkpoint format: {path}")
    return checkpoint


def metric_weight(checkpoint: dict[str, Any]) -> float:
    metrics = checkpoint.get("metrics", {})
    for key in ["best_top1", "val_top1"]:
        try:
            value = float(metrics.get(key, 0.0))
        except (TypeError, ValueError):
            value = 0.0
        if value > 0:
            return value
    return 1.0


def relative_to_manifest(path: Path, manifest_path: Path) -> str:
    absolute = path if path.is_absolute() else (ROOT / path).resolve()
    return os.path.relpath(absolute, start=manifest_path.parent.resolve())


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a Cashlog ensemble manifest.")
    parser.add_argument("checkpoints", nargs="+", type=Path, help="Checkpoint paths, usually */best.pt.")
    parser.add_argument("--output", type=Path, default=ROOT / "configs/cashlog/ensemble.json")
    parser.add_argument("--weights", nargs="*", type=float, help="Optional weights aligned with checkpoints.")
    parser.add_argument("--image-size", type=int, default=224)
    args = parser.parse_args()

    if args.weights and len(args.weights) != len(args.checkpoints):
        raise SystemExit("--weights length must match the number of checkpoints")

    output_path = args.output if args.output.is_absolute() else (ROOT / args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    members = []
    label_ids: list[str] | None = None
    for index, checkpoint_path in enumerate(args.checkpoints):
        checkpoint_abs = checkpoint_path if checkpoint_path.is_absolute() else (ROOT / checkpoint_path)
        checkpoint = read_checkpoint(checkpoint_abs)
        categories = checkpoint.get("categories", [])
        if not categories:
            raise SystemExit(f"checkpoint has no categories metadata: {checkpoint_path}")
        checkpoint_label_ids = [str(category["id"]) for category in categories]
        if label_ids is None:
            label_ids = checkpoint_label_ids
        elif checkpoint_label_ids != label_ids:
            raise SystemExit(f"label order mismatch in checkpoint: {checkpoint_path}")
        weight = args.weights[index] if args.weights else metric_weight(checkpoint)
        members.append(
            {
                "checkpoint": relative_to_manifest(checkpoint_abs, output_path),
                "arch": checkpoint.get("arch") or checkpoint.get("model_arch") or "mobilenetv4_conv_small",
                "weight": weight,
            }
        )

    manifest = {
        "type": "cashlog_weighted_logit_ensemble",
        "image_size": args.image_size,
        "combine": "weighted_logit_average",
        "members": members,
    }
    output_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {output_path.relative_to(ROOT)}")
    print("members=", len(members))


if __name__ == "__main__":
    main()
