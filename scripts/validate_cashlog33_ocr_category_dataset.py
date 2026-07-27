#!/usr/bin/env python3
"""Validate the large CashLog OCR/category dataset and its real CORD supplement."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SYNTHETIC = ROOT / "data/processed/cashlog33/ocr_category/v1"
DEFAULT_RECOGNITION = ROOT / "data/processed/cashlog33/ocr_recognition/v1"
DEFAULT_CORD = ROOT / "data/raw/cashlog33/cord_v2"
DEFAULT_REPORT = ROOT / "reports/cashlog33/data/ocr_category_v1_validation.json"
SPLITS = ("train", "validation", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-dir", type=Path, default=DEFAULT_SYNTHETIC)
    parser.add_argument("--recognition-dir", type=Path, default=DEFAULT_RECOGNITION)
    parser.add_argument("--cord-dir", type=Path, default=DEFAULT_CORD)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--minimum-train-images", type=int, default=100000)
    parser.add_argument("--minimum-train-per-leaf", type=int, default=3000)
    parser.add_argument("--verify-all-hashes", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthetic_manifest = load_jsonl(args.synthetic_dir / "manifest.jsonl")
    synthetic_ocr = load_jsonl(args.synthetic_dir / "ocr_annotations.jsonl")
    synthetic_summary = json.loads(
        (args.synthetic_dir / "summary.json").read_text(encoding="utf-8")
    )
    cord_manifest = load_jsonl(args.cord_dir / "manifest.jsonl")
    cord_ocr = load_jsonl(args.cord_dir / "ocr_annotations.jsonl")
    recognition_summary = json.loads(
        (args.recognition_dir / "summary.json").read_text(encoding="utf-8")
    )

    errors: list[str] = []
    source_text_path = resolve_path(str(synthetic_summary.get("text_manifest") or ""))
    if not source_text_path.is_file():
        errors.append("synthetic source text manifest is missing")
    elif (
        hashlib.sha256(source_text_path.read_bytes()).hexdigest()
        != synthetic_summary.get("text_manifest_sha256")
    ):
        errors.append("synthetic dataset is stale relative to the source text manifest")
    sample_ids: set[str] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    split_counts: Counter[str] = Counter()
    leaf_counts: dict[str, Counter[str]] = defaultdict(Counter)
    manifest_by_id: dict[str, dict[str, Any]] = {}

    for index, row in enumerate(synthetic_manifest):
        sample_id = str(row.get("sample_id") or "")
        relative_path = str(row.get("relative_path") or "")
        split = str(row.get("split") or "")
        leaf_id = str(row.get("leaf_id") or "")
        sha256 = str(row.get("sha256") or "")
        if sample_id in sample_ids:
            errors.append(f"duplicate synthetic sample_id: {sample_id}")
        if relative_path in paths:
            errors.append(f"duplicate synthetic path: {relative_path}")
        if sha256 in hashes:
            errors.append(f"duplicate synthetic image hash: {sha256}")
        sample_ids.add(sample_id)
        paths.add(relative_path)
        hashes.add(sha256)
        manifest_by_id[sample_id] = row
        split_counts[split] += 1
        leaf_counts[split][leaf_id] += 1
        path = resolve_path(relative_path)
        if not path.is_file():
            errors.append(f"missing synthetic image: {relative_path}")
            continue
        if int(row.get("bytes") or -1) != path.stat().st_size:
            errors.append(f"synthetic byte count mismatch: {relative_path}")
        should_hash = args.verify_all_hashes or index % 100 == 0
        if should_hash and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            errors.append(f"synthetic hash mismatch: {relative_path}")

    if split_counts["train"] < args.minimum_train_images:
        errors.append(
            f"synthetic train count {split_counts['train']} < {args.minimum_train_images}"
        )
    if len(leaf_counts["train"]) != 33:
        errors.append(f"synthetic train leaf coverage is {len(leaf_counts['train'])}, expected 33")
    for leaf_id, count in leaf_counts["train"].items():
        if count < args.minimum_train_per_leaf:
            errors.append(
                f"synthetic train leaf {leaf_id} has {count} < {args.minimum_train_per_leaf}"
            )

    ocr_ids: set[str] = set()
    line_count = 0
    for row in synthetic_ocr:
        sample_id = str(row.get("sample_id") or "")
        manifest = manifest_by_id.get(sample_id)
        if manifest is None:
            errors.append(f"OCR annotation has no synthetic manifest row: {sample_id}")
            continue
        if sample_id in ocr_ids:
            errors.append(f"duplicate synthetic OCR sample_id: {sample_id}")
        ocr_ids.add(sample_id)
        width = int(manifest["width"])
        height = int(manifest["height"])
        for line in row.get("lines") or []:
            text = str(line.get("text") or "").strip()
            points = line.get("points") or []
            if not text or len(points) != 4:
                errors.append(f"invalid OCR line: {sample_id}")
                continue
            for point in points:
                if (
                    len(point) != 2
                    or not -2 <= int(point[0]) <= width + 2
                    or not -2 <= int(point[1]) <= height + 2
                ):
                    errors.append(f"OCR point outside image: {sample_id}")
                    break
            line_count += 1
    if ocr_ids != sample_ids:
        errors.append(
            f"synthetic manifest/OCR ID mismatch: manifest={len(sample_ids)} ocr={len(ocr_ids)}"
        )

    cord_ids: set[str] = set()
    cord_split_counts: Counter[str] = Counter()
    for row in cord_manifest:
        sample_id = str(row.get("sample_id") or "")
        cord_ids.add(sample_id)
        cord_split_counts[str(row.get("split") or "")] += 1
        if row.get("eligible_for_category_training") is not False:
            errors.append(f"CORD row is incorrectly category-eligible: {sample_id}")
        if str(row.get("license") or "").lower() != "cc-by-4.0":
            errors.append(f"CORD license mismatch: {sample_id}")
        path = resolve_path(str(row.get("relative_path") or ""))
        if not path.is_file():
            errors.append(f"missing CORD image: {sample_id}")
    cord_ocr_ids = {str(row.get("sample_id") or "") for row in cord_ocr}
    if cord_ids != cord_ocr_ids:
        errors.append(
            f"CORD manifest/OCR ID mismatch: manifest={len(cord_ids)} ocr={len(cord_ocr_ids)}"
        )
    recognition_annotation_sha = hashlib.sha256(
        (args.synthetic_dir / "ocr_annotations.jsonl").read_bytes()
    ).hexdigest()
    if recognition_summary.get("annotations_sha256") != recognition_annotation_sha:
        errors.append("OCR recognition crops are stale relative to synthetic annotations")
    recognition_counts = recognition_summary.get("counts") or {}
    if int(recognition_counts.get("train") or 0) < args.minimum_train_images:
        errors.append("OCR recognition train crop count is below the required minimum")

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "status": "passed" if not errors else "failed",
        "synthetic": {
            "samples": len(synthetic_manifest),
            "split_counts": dict(split_counts),
            "train_leaf_counts": dict(sorted(leaf_counts["train"].items())),
            "unique_hashes": len(hashes),
            "ocr_lines": line_count,
        },
        "cord": {
            "samples": len(cord_manifest),
            "split_counts": dict(cord_split_counts),
            "ocr_lines": sum(len(row.get("lines") or []) for row in cord_ocr),
            "category_training_enabled": False,
        },
        "recognition": {
            "counts": recognition_counts,
            "format": recognition_summary.get("format"),
        },
        "combined_full_page_images": len(synthetic_manifest) + len(cord_manifest),
        "combined_with_recognition_crops": (
            len(synthetic_manifest)
            + len(cord_manifest)
            + sum(int(value) for value in recognition_counts.values())
        ),
        "errors": errors[:100],
        "error_count": len(errors),
        "scope_warning": (
            "Synthetic and OCR-only data cannot be reported as real CashLog accuracy."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
