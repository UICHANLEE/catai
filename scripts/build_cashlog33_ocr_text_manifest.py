#!/usr/bin/env python3
"""Build a leakage-aware OCR text classification manifest from page annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/processed/cashlog33/ocr_category/v2_500k"
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/ocr_text/v2_500k/manifest.jsonl"
OCR_SUBSTITUTIONS = str.maketrans(
    {
        "0": "O",
        "1": "I",
        "5": "S",
        "8": "B",
        "원": "윈",
        "결": "걸",
        "승": "숭",
    }
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def corrupt_text(text: str, sample_id: str, split: str) -> tuple[str, str]:
    rng = random.Random(f"ocr-noise-v2:{sample_id}")
    level_roll = rng.random()
    if level_roll < 0.20:
        return text, "clean"
    level = "light" if level_roll < 0.65 else "medium"
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) > 4 and rng.random() < (0.20 if level == "light" else 0.45):
        lines.pop(rng.randrange(1, len(lines) - 1))
    value = "\n".join(lines)
    output: list[str] = []
    drop_probability = 0.006 if level == "light" else 0.018
    substitute_probability = 0.008 if level == "light" else 0.022
    for character in value:
        if not character.isspace() and rng.random() < drop_probability:
            continue
        if rng.random() < substitute_probability:
            character = character.translate(OCR_SUBSTITUTIONS)
        output.append(character)
        if character == " " and rng.random() < 0.08:
            output.append(" ")
    corrupted = "".join(output)
    if level == "medium" and rng.random() < 0.25:
        corrupted = corrupted.replace(",", "").replace(":", " ")
    return corrupted or text, level


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.dataset_dir / "manifest.jsonl"
    ocr_path = args.dataset_dir / "ocr_annotations.jsonl"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    leaf_counts: dict[str, Counter[str]] = defaultdict(Counter)
    noise_counts: Counter[str] = Counter()
    source_groups: dict[str, set[str]] = defaultdict(set)
    with manifest_path.open(encoding="utf-8") as manifest_handle, ocr_path.open(
        encoding="utf-8"
    ) as ocr_handle, args.output.open("w", encoding="utf-8") as output_handle:
        for index, pair in enumerate(
            zip_longest(manifest_handle, ocr_handle), start=1
        ):
            manifest_line, ocr_line = pair
            if manifest_line is None or ocr_line is None:
                raise SystemExit("manifest and OCR annotations have different row counts")
            manifest = json.loads(manifest_line)
            ocr = json.loads(ocr_line)
            if manifest["sample_id"] != ocr["sample_id"]:
                raise SystemExit(f"row {index} sample_id mismatch")
            split = "val" if manifest["split"] == "validation" else str(manifest["split"])
            text, noise_level = corrupt_text(
                str(ocr["transcription"]), str(manifest["sample_id"]), split
            )
            group_key = str(manifest["source_text_group_key"])
            source_groups[group_key].add(split)
            row: dict[str, Any] = {
                "schema_version": 1,
                "sample_id": manifest["sample_id"],
                "group_key": group_key,
                "leaf_id": manifest["leaf_id"],
                "text": text,
                "split": split,
                "provenance_type": "synthetic_ocr_document_v2",
                "noise_level": noise_level,
                "document_style": manifest["document_style"],
                "clean_text_sha256": hashlib.sha256(
                    str(ocr["transcription"]).encode("utf-8")
                ).hexdigest(),
            }
            output_handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            counts[split] += 1
            leaf_counts[split][str(manifest["leaf_id"])] += 1
            noise_counts[noise_level] += 1
            if index % 50000 == 0:
                print(f"converted={index}", flush=True)
    leaking_groups = sorted(
        group_key for group_key, splits in source_groups.items() if len(splits) > 1
    )
    if leaking_groups:
        raise SystemExit(f"source groups cross splits: {leaking_groups[:10]}")
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "rows": sum(counts.values()),
        "split_counts": dict(counts),
        "leaf_counts": {
            split: dict(sorted(values.items()))
            for split, values in sorted(leaf_counts.items())
        },
        "noise_counts": dict(noise_counts),
        "source_group_count": len(source_groups),
        "source_group_leakage": 0,
        "input_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "input_ocr_sha256": hashlib.sha256(ocr_path.read_bytes()).hexdigest(),
        "output_manifest_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "scope_warning": "Synthetic OCR text is not real CashLog holdout evidence.",
    }
    summary_path = args.output.with_name("summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
