#!/usr/bin/env python3
"""Validate a frozen, manually labeled real-photo CashLog holdout manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def resolve_image(manifest: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest.parent / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-per-leaf", type=int, default=10)
    parser.add_argument("--minimum-dimension", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    rows = read_jsonl(manifest)
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    leaf_ids = [str(row["id"]) for row in categories]
    if len(leaf_ids) != 33 or len(set(leaf_ids)) != 33:
        raise SystemExit("category contract must contain exactly 33 unique leaves")

    required = {
        "sample_id",
        "leaf_id",
        "relative_path",
        "sha256",
        "group_id",
        "review_status",
        "consent_status",
    }
    errors: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    hash_rows: dict[str, list[str]] = defaultdict(list)
    sample_ids: set[str] = set()
    group_labels: dict[str, set[str]] = defaultdict(set)
    for index, row in enumerate(rows, start=1):
        missing = sorted(required - set(row))
        if missing:
            errors.append({"row": index, "error": "missing_fields", "fields": missing})
            continue
        sample_id = str(row["sample_id"])
        leaf_id = str(row["leaf_id"])
        if sample_id in sample_ids:
            errors.append({"row": index, "sample_id": sample_id, "error": "duplicate_sample_id"})
        sample_ids.add(sample_id)
        if leaf_id not in leaf_ids:
            errors.append({"row": index, "sample_id": sample_id, "error": "unknown_leaf", "leaf_id": leaf_id})
            continue
        if str(row["review_status"]) != "approved":
            errors.append({"row": index, "sample_id": sample_id, "error": "label_not_approved"})
        if str(row["consent_status"]) not in {"owner_approved", "licensed"}:
            errors.append({"row": index, "sample_id": sample_id, "error": "invalid_consent_status"})
        image_path = resolve_image(manifest, str(row["relative_path"]))
        if not image_path.is_file():
            errors.append({"row": index, "sample_id": sample_id, "error": "missing_image", "path": str(image_path)})
            continue
        payload = image_path.read_bytes()
        actual_hash = hashlib.sha256(payload).hexdigest()
        if actual_hash != str(row["sha256"]).lower():
            errors.append({"row": index, "sample_id": sample_id, "error": "sha256_mismatch"})
        try:
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if min(image.size) < args.minimum_dimension:
                    errors.append({"row": index, "sample_id": sample_id, "error": "dimension_too_small", "dimensions": list(image.size)})
        except Exception as exc:
            errors.append({"row": index, "sample_id": sample_id, "error": "invalid_image", "detail": str(exc)})
            continue
        counts[leaf_id] += 1
        hash_rows[actual_hash].append(sample_id)
        group_labels[str(row["group_id"])].add(leaf_id)

    for digest, ids in hash_rows.items():
        if len(ids) > 1:
            errors.append({"error": "duplicate_image_sha256", "sha256": digest, "sample_ids": ids})
    for group_id, labels in group_labels.items():
        if len(labels) > 1:
            errors.append({"error": "cross_label_group", "group_id": group_id, "leaf_ids": sorted(labels)})
    below = {
        leaf_id: counts[leaf_id]
        for leaf_id in leaf_ids
        if counts[leaf_id] < args.minimum_per_leaf
    }
    if below:
        errors.append({"error": "minimum_per_leaf_not_met", "counts": below})

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest),
        "samples": len(rows),
        "leaf_count": sum(counts[leaf_id] > 0 for leaf_id in leaf_ids),
        "minimum_per_leaf": min((counts[leaf_id] for leaf_id in leaf_ids), default=0),
        "counts": {leaf_id: counts[leaf_id] for leaf_id in leaf_ids},
        "valid": not errors,
        "errors": errors,
    }
    output = args.output or manifest.with_name("quality_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
