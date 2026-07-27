#!/usr/bin/env python3
"""Merge frozen, reviewed, and licensed weak-label visual training data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MANIFEST = ROOT / "data/raw/cashlog33/openimages_v7/manifest.jsonl"
DEFAULT_BASE_SPLITS = ROOT / "checkpoints/cashlog33/vision_head_v1/split_manifest.jsonl"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/training/incremental_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_image_path(row: dict[str, Any]) -> Path:
    value = row.get("relative_path")
    if not value:
        raise ValueError(f"sample {row.get('sample_id')} is missing relative_path")
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def index_unique(rows: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if not sample_id:
            raise ValueError(f"{label} row is missing sample_id")
        if sample_id in indexed:
            raise ValueError(f"duplicate {label} sample_id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def validate_image_hash(row: dict[str, Any]) -> None:
    sample_id = str(row["sample_id"])
    expected = str(row.get("sha256") or "").lower()
    if len(expected) != 64:
        raise ValueError(f"sample {sample_id} has an invalid sha256")
    image_path = resolve_image_path(row)
    if not image_path.is_file():
        raise ValueError(f"sample {sample_id} image is missing")
    if file_sha256(image_path) != expected:
        raise ValueError(f"sample {sample_id} image sha256 does not match")


def build_incremental_dataset(
    *,
    base_manifest: Path,
    base_split_manifest: Path,
    additional_manifests: list[Path],
    weak_additional_manifests: list[Path] | None = None,
    categories_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    weak_additional_manifests = weak_additional_manifests or []
    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    category_ids = [str(row["id"]) for row in categories]
    if len(category_ids) != 33 or len(set(category_ids)) != 33:
        raise ValueError("categories must contain exactly 33 unique leaf ids")
    allowed_leaves = set(category_ids)

    base_rows = read_jsonl(base_manifest)
    base_by_id = index_unique(base_rows, "base")
    split_rows = read_jsonl(base_split_manifest)
    split_by_id = index_unique(split_rows, "base split")
    if set(base_by_id) != set(split_by_id):
        missing = sorted(set(base_by_id) - set(split_by_id))
        unexpected = sorted(set(split_by_id) - set(base_by_id))
        raise ValueError(
            f"base split contract mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
        )

    ordered_base: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    seen_hashes: dict[str, str] = {}
    for split_row in split_rows:
        sample_id = str(split_row["sample_id"])
        split = str(split_row.get("split") or "")
        if split not in ordered_base:
            raise ValueError(f"invalid base split for {sample_id}: {split}")
        row = base_by_id[sample_id]
        leaf_id = str(row.get("leaf_id") or "")
        if leaf_id != str(split_row.get("leaf_id") or ""):
            raise ValueError(f"base split leaf mismatch for {sample_id}")
        if leaf_id not in allowed_leaves:
            raise ValueError(f"base sample {sample_id} uses an unknown taxonomy leaf")
        sha256 = str(row.get("sha256") or "").lower()
        if sha256:
            seen_hashes[sha256] = sample_id
        ordered_base[split].append({**row, "split": split})

    additional_rows = [
        row
        for manifest in additional_manifests
        for row in read_jsonl(manifest)
    ]
    additional_by_id = index_unique(additional_rows, "additional")
    overlap = sorted(set(base_by_id) & set(additional_by_id))
    if overlap:
        raise ValueError(f"additional sample_id already exists in base dataset: {overlap[0]}")

    reviewed_train: list[dict[str, Any]] = []
    for sample_id, row in additional_by_id.items():
        leaf_id = str(row.get("leaf_id") or "")
        if leaf_id not in allowed_leaves:
            raise ValueError(f"additional sample {sample_id} uses an unknown taxonomy leaf")
        if str(row.get("split_lock") or "") != "train":
            raise ValueError(f"additional sample {sample_id} is not train-locked")
        if str(row.get("review_status") or "") != "approved":
            raise ValueError(f"additional sample {sample_id} is not human-approved")
        validate_image_hash(row)
        sha256 = str(row["sha256"]).lower()
        if sha256 in seen_hashes:
            raise ValueError(
                f"additional sample {sample_id} duplicates image from {seen_hashes[sha256]}"
            )
        seen_hashes[sha256] = sample_id
        reviewed_train.append({**row, "split": "train"})

    weak_rows = [
        row
        for manifest in weak_additional_manifests
        for row in read_jsonl(manifest)
    ]
    weak_by_id = index_unique(weak_rows, "weak additional")
    overlap = sorted((set(base_by_id) | set(additional_by_id)) & set(weak_by_id))
    if overlap:
        raise ValueError(
            f"weak additional sample_id already exists in another dataset: {overlap[0]}"
        )

    weak_ids_by_hash: dict[str, list[str]] = {}
    for sample_id, row in weak_by_id.items():
        validate_image_hash(row)
        weak_ids_by_hash.setdefault(str(row["sha256"]).lower(), []).append(sample_id)
    ambiguous_weak_ids: set[str] = set()
    duplicate_weak_ids: set[str] = set()
    for sample_ids in weak_ids_by_hash.values():
        if len(sample_ids) < 2:
            continue
        leaves = {str(weak_by_id[sample_id].get("leaf_id") or "") for sample_id in sample_ids}
        if len(leaves) > 1:
            ambiguous_weak_ids.update(sample_ids)
        else:
            duplicate_weak_ids.update(sorted(sample_ids)[1:])

    weak_train: list[dict[str, Any]] = []
    allowed_weak_sources = {
        "openverse",
        "openfoodfacts-grocery",
        "openfoodfacts-beverages",
        "openbeautyfacts",
        "openpetfoodfacts",
    }
    allowed_weak_licenses = {"by", "cc0", "pdm", "cc-by-sa-3.0"}
    for sample_id, row in weak_by_id.items():
        if sample_id in ambiguous_weak_ids or sample_id in duplicate_weak_ids:
            continue
        leaf_id = str(row.get("leaf_id") or "")
        if leaf_id not in allowed_leaves:
            raise ValueError(
                f"weak additional sample {sample_id} uses an unknown taxonomy leaf"
            )
        source = str(row.get("source") or "")
        if source not in allowed_weak_sources:
            raise ValueError(
                f"weak additional sample {sample_id} has an unsupported source"
            )
        if source == "openverse" and str(row.get("status") or "") != "accepted":
            raise ValueError(
                f"weak additional sample {sample_id} was not accepted by collection"
            )
        if str(row.get("license") or "").casefold() not in allowed_weak_licenses:
            raise ValueError(
                f"weak additional sample {sample_id} has an unsupported license"
            )
        if source != "openverse" and (
            str(row.get("provenance_type") or "") != "api_product_image"
            or str(row.get("review_status") or "") != "product_type_weak"
            or str(row.get("status") or "accepted") != "accepted"
        ):
            raise ValueError(
                f"weak product sample {sample_id} lacks API product provenance"
            )
        validate_image_hash(row)
        sha256 = str(row["sha256"]).lower()
        if sha256 in seen_hashes:
            raise ValueError(
                f"weak additional sample {sample_id} duplicates image from "
                f"{seen_hashes[sha256]}"
            )
        seen_hashes[sha256] = sample_id
        weak_train.append(
            {
                **row,
                "split": "train",
                "split_lock": "train",
                "review_status": "weak_label",
                "label_method": (
                    "openverse_query_leaf_v1"
                    if source == "openverse"
                    else "product_opener_product_type_v1"
                ),
                "weak_label": True,
            }
        )

    merged_rows = [
        *ordered_base["train"],
        *sorted(reviewed_train, key=lambda row: str(row["sample_id"])),
        *sorted(weak_train, key=lambda row: str(row["sample_id"])),
        *ordered_base["val"],
        *ordered_base["test"],
    ]
    output_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(output_dir, 0o700)
    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in merged_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(manifest_path, 0o600)

    split_counts = Counter(str(row["split"]) for row in merged_rows)
    source_counts = Counter(str(row.get("source") or "unknown") for row in merged_rows)
    leaf_counts = Counter(str(row["leaf_id"]) for row in merged_rows)
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "taxonomy_leaf_count": len(category_ids),
        "base_rows": len(base_rows),
        "additional_train_rows": len(reviewed_train),
        "weak_additional_train_rows": len(weak_train),
        "weak_ambiguous_rows_excluded": len(ambiguous_weak_ids),
        "weak_duplicate_rows_excluded": len(duplicate_weak_ids),
        "merged_rows": len(merged_rows),
        "split_counts": dict(sorted(split_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "leaf_counts": {leaf_id: leaf_counts.get(leaf_id, 0) for leaf_id in category_ids},
        "inputs": {
            "base_manifest": str(base_manifest.resolve()),
            "base_manifest_sha256": file_sha256(base_manifest),
            "base_split_manifest": str(base_split_manifest.resolve()),
            "base_split_manifest_sha256": file_sha256(base_split_manifest),
            "additional_manifests": [
                {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
                for path in additional_manifests
            ],
            "weak_additional_manifests": [
                {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }
                for path in weak_additional_manifests
            ],
        },
        "output_manifest": str(manifest_path),
        "output_manifest_sha256": file_sha256(manifest_path),
    }
    summary_path = output_dir / "dataset_summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.chmod(summary_path, 0o600)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-manifest", type=Path, default=DEFAULT_BASE_MANIFEST)
    parser.add_argument("--base-split-manifest", type=Path, default=DEFAULT_BASE_SPLITS)
    parser.add_argument(
        "--additional-train-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--optional-additional-train-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--weak-additional-train-manifest",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    optional_manifests = [
        path
        for path in args.optional_additional_train_manifest
        if path.is_file() and path.stat().st_size > 0
    ]
    summary = build_incremental_dataset(
        base_manifest=args.base_manifest,
        base_split_manifest=args.base_split_manifest,
        additional_manifests=[
            *args.additional_train_manifest,
            *optional_manifests,
        ],
        weak_additional_manifests=args.weak_additional_train_manifest,
        categories_path=args.categories,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
