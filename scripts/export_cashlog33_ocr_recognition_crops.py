#!/usr/bin/env python3
"""Export deterministic OCR line crops in PaddleOCR recognition format."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANNOTATIONS = ROOT / "data/processed/cashlog33/ocr_category/v1/ocr_annotations.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/ocr_recognition/v1"
DEFAULT_LIMITS = {"train": 120000, "validation": 5000, "test": 5000}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def stable_rank(sample_id: str, line_index: int, seed: int) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}:{line_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def select_lines(path: Path, limits: dict[str, int], seed: int) -> dict[str, list[dict[str, Any]]]:
    heaps: dict[str, list[tuple[int, int, dict[str, Any]]]] = {
        split: [] for split in limits
    }
    serial = 0
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            row = json.loads(raw_line)
            split = str(row["split"])
            if split not in heaps:
                continue
            for line_index, line in enumerate(row.get("lines") or []):
                text = " ".join(str(line.get("text") or "").replace("\t", " ").split())
                points = line.get("points") or []
                if not text or len(points) != 4:
                    continue
                record = {
                    "sample_id": str(row["sample_id"]),
                    "relative_path": str(row["relative_path"]),
                    "leaf_id": str(row["leaf_id"]),
                    "line_index": line_index,
                    "text": text,
                    "points": points,
                }
                rank = stable_rank(record["sample_id"], line_index, seed)
                item = (-rank, serial, record)
                serial += 1
                heap = heaps[split]
                if len(heap) < limits[split]:
                    heapq.heappush(heap, item)
                elif rank < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    return {
        split: [
            item[2]
            for item in sorted(heap, key=lambda value: (-value[0], value[1]))
        ]
        for split, heap in heaps.items()
    }


def crop_line(image: np.ndarray, points: list[list[int]]) -> np.ndarray:
    source = np.asarray(points, dtype=np.float32)
    top = np.linalg.norm(source[1] - source[0])
    bottom = np.linalg.norm(source[2] - source[3])
    left = np.linalg.norm(source[3] - source[0])
    right = np.linalg.norm(source[2] - source[1])
    width = max(8, int(round(max(top, bottom))))
    height = max(8, int(round(max(left, right))))
    destination = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, default=DEFAULT_ANNOTATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--train-limit", type=int, default=DEFAULT_LIMITS["train"])
    parser.add_argument("--validation-limit", type=int, default=DEFAULT_LIMITS["validation"])
    parser.add_argument("--test-limit", type=int, default=DEFAULT_LIMITS["test"])
    parser.add_argument("--seed", type=int, default=250716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    limits = {
        "train": args.train_limit,
        "validation": args.validation_limit,
        "test": args.test_limit,
    }
    if any(value < 0 for value in limits.values()) or limits["train"] < 100000:
        raise SystemExit("train-limit must be at least 100000 and all limits must be non-negative")
    selected = select_lines(args.annotations, limits, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    leaf_counts: dict[str, Counter[str]] = {
        split: Counter() for split in limits
    }
    byte_count = 0
    for split, rows in selected.items():
        rows = sorted(
            rows,
            key=lambda row: (
                str(row["relative_path"]),
                int(row["line_index"]),
            ),
        )
        label_path = args.output_dir / f"rec_{split}.txt"
        manifest_path = args.output_dir / f"manifest_{split}.jsonl"
        cache_path: Path | None = None
        cache_image: np.ndarray | None = None
        with label_path.open("w", encoding="utf-8") as label_handle, manifest_path.open(
            "w", encoding="utf-8"
        ) as manifest_handle:
            for index, row in enumerate(rows):
                source_path = resolve_path(row["relative_path"])
                if source_path != cache_path:
                    cache_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
                    cache_path = source_path
                if cache_image is None:
                    raise ValueError(f"unable to decode source image: {source_path}")
                crop = crop_line(cache_image, row["points"])
                crop_id = hashlib.sha256(
                    f"{row['sample_id']}:{row['line_index']}".encode()
                ).hexdigest()[:24]
                output_path = args.output_dir / "images" / split / f"{crop_id}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                success, encoded = cv2.imencode(
                    ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 92]
                )
                if not success:
                    raise ValueError(f"unable to encode OCR crop: {crop_id}")
                payload = encoded.tobytes()
                output_path.write_bytes(payload)
                relative_path = relative_to_root(output_path)
                label_handle.write(f"{relative_path}\t{row['text']}\n")
                manifest_handle.write(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "sample_id": f"cashlog-ocr-rec-v1:{split}:{crop_id}",
                            "source_sample_id": row["sample_id"],
                            "source_line_index": row["line_index"],
                            "leaf_id": row["leaf_id"],
                            "relative_path": relative_path,
                            "text": row["text"],
                            "split": split,
                            "license": "project-generated",
                            "sha256": hashlib.sha256(payload).hexdigest(),
                            "width": int(crop.shape[1]),
                            "height": int(crop.shape[0]),
                            "bytes": len(payload),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    + "\n"
                )
                counts[split] += 1
                leaf_counts[split][row["leaf_id"]] += 1
                byte_count += len(payload)
                if counts[split] % 5000 == 0 or counts[split] == len(rows):
                    print(
                        f"split={split} crops={counts[split]}/{len(rows)}",
                        flush=True,
                    )
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset_version": "cashlog33-ocr-recognition-v1",
        "counts": dict(counts),
        "limits": limits,
        "leaf_counts": {
            split: dict(sorted(counter.items()))
            for split, counter in leaf_counts.items()
        },
        "bytes": byte_count,
        "gib": byte_count / 2**30,
        "annotations": relative_to_root(args.annotations),
        "annotations_sha256": hashlib.sha256(args.annotations.read_bytes()).hexdigest(),
        "format": "PaddleOCR recognition path-tab-transcription",
        "scope_warning": "Synthetic line crops are not real OCR accuracy evidence.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
