#!/usr/bin/env python3
"""Clip OCR polygons to image bounds and rebuild Paddle detection labels."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from itertools import zip_longest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/processed/cashlog33/ocr_category/v2_500k"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset_dir / "manifest.jsonl"
    annotation_path = args.dataset_dir / "ocr_annotations.jsonl"
    temporary = annotation_path.with_suffix(".jsonl.tmp")
    paddle_paths = {
        split: args.dataset_dir / f"paddle_det_{split}.txt"
        for split in ("train", "validation", "test")
    }
    paddle_temporary = {
        split: path.with_suffix(".txt.tmp") for split, path in paddle_paths.items()
    }
    handles = {
        split: path.open("w", encoding="utf-8")
        for split, path in paddle_temporary.items()
    }
    corrected_points = 0
    corrected_samples: Counter[str] = Counter()
    try:
        with manifest_path.open(encoding="utf-8") as manifest_handle, annotation_path.open(
            encoding="utf-8"
        ) as annotation_handle, temporary.open("w", encoding="utf-8") as output_handle:
            for index, pair in enumerate(
                zip_longest(manifest_handle, annotation_handle), start=1
            ):
                manifest_line, annotation_line = pair
                if manifest_line is None or annotation_line is None:
                    raise SystemExit("manifest and OCR annotation row counts differ")
                manifest = json.loads(manifest_line)
                annotation = json.loads(annotation_line)
                if manifest["sample_id"] != annotation["sample_id"]:
                    raise SystemExit(f"sample mismatch at row {index}")
                width = int(manifest["width"])
                height = int(manifest["height"])
                sample_corrected = False
                for line in annotation.get("lines") or []:
                    clipped = []
                    for x, y in line["points"]:
                        point = [
                            max(0, min(width - 1, int(x))),
                            max(0, min(height - 1, int(y))),
                        ]
                        corrected_points += int(point != [x, y])
                        sample_corrected = sample_corrected or point != [x, y]
                        clipped.append(point)
                    line["points"] = clipped
                if sample_corrected:
                    corrected_samples[str(manifest["split"])] += 1
                output_handle.write(
                    json.dumps(annotation, ensure_ascii=False, sort_keys=True) + "\n"
                )
                paddle = [
                    {"transcription": line["text"], "points": line["points"]}
                    for line in annotation.get("lines") or []
                ]
                handles[str(manifest["split"])].write(
                    f"{manifest['relative_path']}\t"
                    f"{json.dumps(paddle, ensure_ascii=False, separators=(',', ':'))}\n"
                )
                if index % 100000 == 0:
                    print(f"repaired_scan={index}", flush=True)
    finally:
        for handle in handles.values():
            handle.close()
    temporary.replace(annotation_path)
    for split, path in paddle_paths.items():
        paddle_temporary[split].replace(path)
    print(
        json.dumps(
            {
                "samples_corrected": sum(corrected_samples.values()),
                "corrected_points": corrected_points,
                "corrected_samples_by_split": dict(corrected_samples),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
