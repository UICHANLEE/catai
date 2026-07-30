#!/usr/bin/env python3
"""Build a deduplicated original-image dataset and exact one-million-view schedule."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    ROOT / "data/raw/cashlog33/openimages_v7_train/manifest.jsonl",
    ROOT
    / "data/processed/cashlog33/training/originals_v2_additional/manifest.jsonl",
]
DEFAULT_EXTERNAL_TEST = ROOT / "data/raw/cashlog33/openimages_v7/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/training/million_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def deduplicate_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_hash = 0
    for row in rows:
        digest = str(row.get("sha256") or "")
        if not digest:
            missing_hash += 1
            continue
        by_hash[digest].append(row)

    accepted = []
    duplicate_rows = 0
    conflicting_rows = 0
    for digest, matching in by_hash.items():
        leaves = {str(row["leaf_id"]) for row in matching}
        if len(leaves) != 1:
            conflicting_rows += len(matching)
            continue
        matching.sort(key=lambda row: (str(row.get("source")), str(row["sample_id"])))
        accepted.append(dict(matching[0]))
        duplicate_rows += len(matching) - 1
    accepted.sort(key=lambda row: (str(row["leaf_id"]), str(row["sample_id"])))
    return accepted, {
        "missing_hash_rows_dropped": missing_hash,
        "duplicate_rows_dropped": duplicate_rows,
        "cross_leaf_conflict_rows_dropped": conflicting_rows,
    }


def assign_partitions(
    rows: list[dict[str, Any]], validation_ratio: float, seed: int
) -> list[dict[str, Any]]:
    openimages_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)
    output = []
    for row in rows:
        if str(row.get("source")) == "openimages_v7_train":
            openimages_by_leaf[str(row["leaf_id"])].append(row)
        else:
            copied = dict(row)
            copied["partition"] = "train"
            output.append(copied)
    for leaf_id, leaf_rows in sorted(openimages_by_leaf.items()):
        ordered = sorted(
            leaf_rows, key=lambda row: stable_key(seed, str(row["sample_id"]))
        )
        validation_count = max(1, round(len(ordered) * validation_ratio))
        if validation_count >= len(ordered):
            validation_count = max(0, len(ordered) - 1)
        for index, row in enumerate(ordered):
            copied = dict(row)
            copied["partition"] = (
                "validation" if index < validation_count else "train"
            )
            output.append(copied)
    output.sort(
        key=lambda row: (
            str(row["partition"]),
            str(row["leaf_id"]),
            str(row["sample_id"]),
        )
    )
    return output


def build_schedule(
    rows: list[dict[str, Any]],
    target_views: int,
    minimum_originals_per_leaf: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train_indexes_by_leaf: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row["partition"] == "train":
            train_indexes_by_leaf[str(row["leaf_id"])].append(index)
    leaves = sorted(
        leaf_id
        for leaf_id, indexes in train_indexes_by_leaf.items()
        if len(indexes) >= minimum_originals_per_leaf
    )
    if len(leaves) < 2:
        raise ValueError("at least two leaves need enough original images")

    rng = np.random.default_rng(seed)
    original_counts = {
        leaf_id: len(train_indexes_by_leaf[leaf_id]) for leaf_id in leaves
    }
    eligible_originals = sum(original_counts.values())
    if target_views < eligible_originals:
        raise ValueError(
            "target views must be at least the number of eligible train originals"
        )

    low = min(original_counts.values())
    high = target_views
    water_level = low
    while low <= high:
        middle = (low + high) // 2
        proposed = sum(
            max(count, middle) for count in original_counts.values()
        )
        if proposed <= target_views:
            water_level = middle
            low = middle + 1
        else:
            high = middle - 1
    target_counts = {
        leaf_id: max(count, water_level)
        for leaf_id, count in original_counts.items()
    }
    remainder = target_views - sum(target_counts.values())
    for leaf_id in sorted(
        leaves, key=lambda value: (target_counts[value], stable_key(seed, value))
    )[:remainder]:
        target_counts[leaf_id] += 1

    scheduled_parts = []
    for leaf_id in leaves:
        source_indexes = np.asarray(
            train_indexes_by_leaf[leaf_id], dtype=np.uint32
        )
        extra_count = target_counts[leaf_id] - len(source_indexes)
        extras = (
            rng.choice(source_indexes, size=extra_count, replace=True)
            if extra_count
            else np.asarray([], dtype=np.uint32)
        )
        scheduled_parts.append(np.concatenate([source_indexes, extras]))
    row_indexes = np.concatenate(scheduled_parts)
    rng.shuffle(row_indexes)
    augmentation_seeds = rng.integers(
        0, np.iinfo(np.uint32).max, size=target_views, dtype=np.uint32
    )
    return row_indexes, augmentation_seeds, leaves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, dest="manifests")
    parser.add_argument("--external-test", type=Path, default=DEFAULT_EXTERNAL_TEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target-views", type=int, default=1_000_000)
    parser.add_argument("--validation-ratio", type=float, default=0.20)
    parser.add_argument("--minimum-originals-per-leaf", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1_000_033)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = args.manifests or DEFAULT_MANIFESTS
    for path in [*manifests, args.external_test]:
        if not path.is_file():
            raise SystemExit(f"missing manifest: {path}")

    source_rows = [row for path in manifests for row in read_jsonl(path)]
    deduplicated, dedup_stats = deduplicate_rows(source_rows)
    external_rows = read_jsonl(args.external_test)
    external_hashes = {str(row.get("sha256")) for row in external_rows}
    overlap = [row for row in deduplicated if str(row["sha256"]) in external_hashes]
    if overlap:
        raise RuntimeError(
            f"training sources overlap the frozen external test: {len(overlap)}"
        )
    external_openimages_ids = {
        str(row.get("source_id"))
        for row in external_rows
        if row.get("source_id")
    }
    source_id_overlap = [
        row
        for row in deduplicated
        if str(row.get("source")) == "openimages_v7_train"
        and str(row.get("source_id")) in external_openimages_ids
    ]
    if source_id_overlap:
        raise RuntimeError(
            "Open Images source ids overlap official train and validation: "
            f"{len(source_id_overlap)}"
        )

    rows = assign_partitions(deduplicated, args.validation_ratio, args.seed)
    missing_files = [
        row["relative_path"]
        for row in rows
        if not (ROOT / str(row["relative_path"])).is_file()
    ]
    if missing_files:
        raise RuntimeError(f"missing source images: {missing_files[:5]}")

    row_indexes, augmentation_seeds, trainable_leaves = build_schedule(
        rows,
        args.target_views,
        args.minimum_originals_per_leaf,
        args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "original_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    schedule_path = args.output_dir / "million_view_schedule.npz"
    np.savez_compressed(
        schedule_path,
        row_indexes=row_indexes,
        augmentation_seeds=augmentation_seeds,
        trainable_leaves=np.asarray(trainable_leaves),
    )

    train_rows = [row for row in rows if row["partition"] == "train"]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    scheduled_leaf_counts = dict(
        sorted(Counter(str(rows[index]["leaf_id"]) for index in row_indexes).items())
    )
    eligible_train_indexes = {
        index
        for index, row in enumerate(rows)
        if row["partition"] == "train"
        and str(row["leaf_id"]) in set(trainable_leaves)
    }
    scheduled_unique_indexes = set(int(index) for index in row_indexes)
    augmentation = {
        "logical_view_count": args.target_views,
        "epochs": 4,
        "views_per_epoch": args.target_views // 4,
        "operations": [
            {
                "name": "RandomResizedCrop",
                "scale": [0.65, 1.0],
                "ratio": [0.75, 1.3333],
                "interpolation": "bicubic",
            },
            {"name": "RandomHorizontalFlip", "probability": 0.5},
            {
                "name": "ColorJitter",
                "brightness": 0.20,
                "contrast": 0.20,
                "saturation": 0.15,
                "hue": 0.02,
            },
            {"name": "RandAugment", "num_ops": 2, "magnitude": 7},
            {
                "name": "RandomErasing",
                "probability": 0.10,
                "scale": [0.02, 0.10],
            },
        ],
        "evaluation_preprocess": {
            "resize_shorter_side": 256,
            "center_crop": 224,
            "interpolation": "bicubic",
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "note": (
            "One logical view means one scheduled original-image exposure with a "
            "recorded augmentation seed; it is not an additional original image."
        ),
    }
    (args.output_dir / "augmentation_plan.json").write_text(
        json.dumps(augmentation, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "input_manifests": [
            {"path": str(path), "sha256": sha256_file(path)} for path in manifests
        ],
        "external_test_manifest": {
            "path": str(args.external_test),
            "sha256": sha256_file(args.external_test),
            "rows": len(external_rows),
        },
        "source_rows": len(source_rows),
        "unique_originals": len(rows),
        "train_originals": len(train_rows),
        "validation_originals": len(validation_rows),
        "validation_ratio_for_openimages_train": args.validation_ratio,
        "logical_training_views": args.target_views,
        "original_reuse_factor": args.target_views / max(1, len(train_rows)),
        "trainable_leaves": trainable_leaves,
        "trainable_leaf_count": len(trainable_leaves),
        "source_counts": dict(
            sorted(Counter(str(row.get("source")) for row in rows).items())
        ),
        "train_leaf_counts": dict(
            sorted(Counter(str(row["leaf_id"]) for row in train_rows).items())
        ),
        "validation_leaf_counts": dict(
            sorted(Counter(str(row["leaf_id"]) for row in validation_rows).items())
        ),
        "deduplication": dedup_stats,
        "external_leakage_checks": {
            "sha256_overlap": 0,
            "openimages_source_id_overlap": 0,
        },
        "schedule": {
            "path": str(schedule_path),
            "sha256": sha256_file(schedule_path),
            "rows": len(row_indexes),
            "strategy": "all-originals-once-then-water-fill-rare-leaves",
            "eligible_unique_train_originals": len(eligible_train_indexes),
            "scheduled_unique_train_originals": len(
                eligible_train_indexes & scheduled_unique_indexes
            ),
            "all_eligible_train_originals_used": (
                eligible_train_indexes <= scheduled_unique_indexes
            ),
            "scheduled_leaf_counts": scheduled_leaf_counts,
        },
        "augmentation_plan": augmentation,
    }
    (args.output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
