#!/usr/bin/env python3
"""Validate, deduplicate, and split the 33-leaf CashLog dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_SEMANTICS = ROOT / "configs/cashlog/leaf_semantics.json"
DEFAULT_INPUT = ROOT / "data/raw/cashlog33/openverse/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
                row["_manifest_path"] = str(path)
                row["_line_number"] = line_number
                rows.append(row)
    return rows


def resolve_image_path(row: dict[str, Any]) -> Path:
    value = row.get("relative_path") or row.get("path")
    if not value:
        raise ValueError("missing relative_path")
    path = Path(str(value))
    return path if path.is_absolute() else ROOT / path


def difference_hash(image: Image.Image) -> int:
    sample = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(sample.get_flattened_data())
    value = 0
    for y in range(8):
        offset = y * 9
        for x in range(8):
            value = (value << 1) | int(pixels[offset + x] > pixels[offset + x + 1])
    return value


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def creator_group(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or row.get("source") or "unknown").strip().casefold()
    creator = str(row.get("creator") or "").strip().casefold()
    if not creator:
        return str(row["sample_id"])
    return f"{provider}:{creator}"


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row.get(key)
        for key in [
            "sample_id",
            "leaf_id",
            "relative_path",
            "source",
            "provider",
            "source_name",
            "source_id",
            "source_url",
            "title",
            "creator",
            "license",
            "sha256",
            "review_status",
            "review_method",
        ]
    }


def validate_rows(
    rows: list[dict[str, Any]],
    category_ids: set[str],
    allowed_licenses: set[str],
    allowed_review_statuses: set[str],
    min_dimension: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    valid: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    required = {"sample_id", "leaf_id", "relative_path", "source", "license", "sha256"}
    for row in rows:
        reasons: list[str] = []
        missing = sorted(key for key in required if not row.get(key))
        if missing:
            reasons.append(f"missing required fields: {missing}")
        leaf_id = str(row.get("leaf_id") or "")
        if leaf_id not in category_ids:
            reasons.append(f"unknown leaf_id: {leaf_id}")
        license_id = str(row.get("license") or "").lower()
        if license_id not in allowed_licenses:
            reasons.append(f"license not allowed: {license_id}")
        review_status = str(row.get("review_status") or "").lower()
        if review_status not in allowed_review_statuses:
            reasons.append(f"review status not allowed: {review_status or 'missing'}")
        try:
            path = resolve_image_path(row)
            payload = path.read_bytes()
            actual_sha = hashlib.sha256(payload).hexdigest()
            if actual_sha != row.get("sha256"):
                reasons.append("sha256 mismatch")
            with Image.open(path) as source:
                source.load()
                image = ImageOps.exif_transpose(source).convert("RGB")
            width, height = image.size
            if min(width, height) < min_dimension:
                reasons.append(f"image smaller than {min_dimension}px: {width}x{height}")
            row["width"] = width
            row["height"] = height
            row["dhash"] = f"{difference_hash(image):016x}"
            row["creator_group"] = creator_group(row)
            row["metadata_text"] = " ".join(
                [str(row.get("title") or ""), *[str(tag) for tag in row.get("tags") or []]]
            ).strip()
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            reasons.append(f"invalid image: {type(exc).__name__}: {exc}")
        if reasons:
            rejected.append({"row": compact_row(row), "reasons": reasons})
        else:
            valid.append(row)
    return valid, rejected


def remove_exact_duplicates(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_sha: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_sha[str(row["sha256"])].append(row)
    kept: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    for sha256, group in sorted(by_sha.items()):
        labels = sorted({str(row["leaf_id"]) for row in group})
        ordered = sorted(group, key=lambda row: str(row["sample_id"]))
        if len(labels) > 1:
            conflicts.append(
                {
                    "sha256": sha256,
                    "leaf_ids": labels,
                    "samples": [compact_row(row) for row in ordered],
                }
            )
            continue
        kept.append(ordered[0])
        for duplicate in ordered[1:]:
            duplicates.append(
                {
                    "kept_sample_id": ordered[0]["sample_id"],
                    "dropped_sample_id": duplicate["sample_id"],
                    "leaf_id": labels[0],
                    "sha256": sha256,
                }
            )
    return kept, duplicates, conflicts


def remove_near_duplicates(
    rows: list[dict[str, Any]], distance: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(rows, key=lambda row: str(row["sample_id"]))
    dropped: set[str] = set()
    same_label: list[dict[str, Any]] = []
    cross_label: list[dict[str, Any]] = []
    hashes = [int(str(row["dhash"]), 16) for row in ordered]
    for left_index, left in enumerate(ordered):
        if str(left["sample_id"]) in dropped:
            continue
        for right_index in range(left_index + 1, len(ordered)):
            right = ordered[right_index]
            if str(right["sample_id"]) in dropped:
                continue
            measured = hamming_distance(hashes[left_index], hashes[right_index])
            if measured > distance:
                continue
            finding = {
                "distance": measured,
                "left": compact_row(left),
                "right": compact_row(right),
            }
            if left["leaf_id"] == right["leaf_id"]:
                dropped.add(str(right["sample_id"]))
                same_label.append(finding)
            else:
                dropped.add(str(left["sample_id"]))
                dropped.add(str(right["sample_id"]))
                cross_label.append(finding)
                break
    return [row for row in ordered if str(row["sample_id"]) not in dropped], same_label, cross_label


def assign_splits(
    rows: list[dict[str, Any]],
    category_ids: list[str],
    seed: int,
    val_ratio: float,
    test_ratio: float,
    min_val: int,
    min_test: int,
) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["leaf_id"])].append(row)
    output: list[dict[str, Any]] = []
    for leaf_id in category_ids:
        label_rows = by_label.get(leaf_id, [])
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in label_rows:
            groups[str(row["creator_group"])].append(row)
        ordered_groups = sorted(groups.items(), key=lambda pair: stable_key(seed, f"{leaf_id}:{pair[0]}"))
        target_test = max(min_test, round(len(label_rows) * test_ratio)) if label_rows else 0
        target_val = max(min_val, round(len(label_rows) * val_ratio)) if label_rows else 0
        assigned_test = sum(
            len(group_rows)
            for _, group_rows in ordered_groups
            if {str(row.get("split_lock") or "") for row in group_rows} == {"test"}
        )
        assigned_val = sum(
            len(group_rows)
            for _, group_rows in ordered_groups
            if {str(row.get("split_lock") or "") for row in group_rows} == {"val"}
        )
        for _, group_rows in ordered_groups:
            split_locks = {
                str(row.get("split_lock") or "")
                for row in group_rows
                if str(row.get("split_lock") or "")
            }
            invalid_locks = split_locks - {"train", "val", "test"}
            if invalid_locks:
                raise ValueError(f"invalid split_lock for {leaf_id}: {sorted(invalid_locks)}")
            if len(split_locks) > 1:
                raise ValueError(
                    f"creator group has conflicting split_lock values for {leaf_id}: {sorted(split_locks)}"
                )
            if split_locks:
                split = next(iter(split_locks))
            elif assigned_test < target_test:
                split = "test"
                assigned_test += len(group_rows)
            elif assigned_val < target_val:
                split = "val"
                assigned_val += len(group_rows)
            else:
                split = "train"
            for row in group_rows:
                output.append({**row, "split": split})
    return sorted(output, key=lambda row: (str(row["split"]), str(row["leaf_id"]), str(row["sample_id"])))


def split_counts(rows: list[dict[str, Any]], category_ids: list[str]) -> dict[str, dict[str, int]]:
    counts = {leaf_id: {"train": 0, "val": 0, "test": 0, "total": 0} for leaf_id in category_ids}
    for row in rows:
        leaf_id = str(row["leaf_id"])
        split = str(row["split"])
        counts[leaf_id][split] += 1
        counts[leaf_id]["total"] += 1
    return counts


def creator_leakage(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_creator: dict[str, set[str]] = defaultdict(set)
    samples: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        group = str(row["creator_group"])
        by_creator[group].add(str(row["split"]))
        samples[group].append(str(row["sample_id"]))
    return [
        {"creator_group": group, "splits": sorted(splits), "sample_ids": sorted(samples[group])}
        for group, splits in sorted(by_creator.items())
        if len(splits) > 1
    ]


def write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    private_fields = {"_manifest_path", "_line_number"}
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            output = {key: value for key, value in row.items() if key not in private_fields}
            handle.write(json.dumps(output, ensure_ascii=False, sort_keys=True) + "\n")


def write_counts_csv(path: Path, counts: dict[str, dict[str, int]], display_names: dict[str, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["leaf_id", "display_name", "train", "val", "test", "total"],
            lineterminator="\n",
        )
        writer.writeheader()
        for leaf_id, values in counts.items():
            writer.writerow({"leaf_id": leaf_id, "display_name": display_names[leaf_id], **values})


def write_audit_markdown(path: Path, report: dict[str, Any], display_names: dict[str, str]) -> None:
    lines = [
        "# CashLog 33-leaf dataset quality audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Decision",
        "",
        f"- Ready for training: **{str(report['ready_for_training']).lower()}**",
        f"- Accepted samples: **{report['accepted_samples']}** / input {report['input_rows']}",
        f"- Covered leaves: **{report['covered_leaf_count']} / 33**",
        f"- Exact label conflicts: **{len(report['exact_cross_label_conflicts'])}**",
        f"- Near label conflicts: **{len(report['near_cross_label_conflicts'])}**",
        f"- Rejected rows: **{len(report['rejected_rows'])}**",
        "",
        "## Per-leaf counts",
        "",
        "| Leaf | Name | Train | Val | Test | Total |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for leaf_id, values in report["counts"].items():
        lines.append(
            f"| `{leaf_id}` | {display_names[leaf_id]} | {values['train']} | {values['val']} | "
            f"{values['test']} | {values['total']} |"
        )
    lines.extend(
        [
            "",
            "## Blocking checks",
            "",
            f"- Missing leaves: {report['missing_leaves'] or 'none'}",
            f"- Leaves below minimum split counts: {report['leaves_below_minimum'] or 'none'}",
            f"- Creator groups crossing splits: {len(report['creator_leakage'])}",
            "",
            "`misc_uncat` remains both a learned hard-negative bucket and the serving confidence fallback. "
            "Production quality must therefore also be measured on real CashLog corrections.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, action="append", dest="input_manifests")
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=250716)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--min-dimension", type=int, default=180)
    parser.add_argument("--near-duplicate-distance", type=int, default=2)
    parser.add_argument(
        "--allowed-review-status",
        action="append",
        dest="allowed_review_statuses",
        choices=["approved", "auto_approved", "source_mapped", "trusted", "pending"],
        help="Review status allowed into the dataset. Repeat to allow more than one.",
    )
    parser.add_argument("--min-train-per-leaf", type=int, default=20)
    parser.add_argument("--min-val-per-leaf", type=int, default=4)
    parser.add_argument("--min-test-per-leaf", type=int, default=4)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    args.input_manifests = args.input_manifests or [DEFAULT_INPUT]
    args.allowed_review_statuses = set(
        args.allowed_review_statuses or ["approved", "auto_approved", "source_mapped", "trusted"]
    )
    if args.val_ratio <= 0 or args.test_ratio <= 0 or args.val_ratio + args.test_ratio >= 1:
        parser.error("val/test ratios must be positive and sum to less than one")
    return args


def main() -> None:
    args = parse_args()
    categories = load_json(args.categories)
    category_ids = [str(row["id"]) for row in categories]
    if len(category_ids) != 33 or len(set(category_ids)) != 33:
        raise SystemExit("categories.json must contain exactly 33 unique leaf ids")
    display_names = {str(row["id"]): str(row["display_name"]) for row in categories}
    semantics = load_json(args.semantics)
    allowed_licenses = {str(value).lower() for value in semantics["allowed_training_licenses"]}

    rows = read_jsonl([path.resolve() for path in args.input_manifests])
    valid, rejected = validate_rows(
        rows,
        set(category_ids),
        allowed_licenses,
        args.allowed_review_statuses,
        args.min_dimension,
    )
    exact_kept, exact_duplicates, exact_conflicts = remove_exact_duplicates(valid)
    near_kept, near_duplicates, near_conflicts = remove_near_duplicates(
        exact_kept, args.near_duplicate_distance
    )
    split_rows = assign_splits(
        near_kept,
        category_ids,
        args.seed,
        args.val_ratio,
        args.test_ratio,
        args.min_val_per_leaf,
        args.min_test_per_leaf,
    )
    counts = split_counts(split_rows, category_ids)
    missing = [leaf_id for leaf_id, values in counts.items() if values["total"] == 0]
    below_minimum = [
        leaf_id
        for leaf_id, values in counts.items()
        if values["train"] < args.min_train_per_leaf
        or values["val"] < args.min_val_per_leaf
        or values["test"] < args.min_test_per_leaf
    ]
    leakage = creator_leakage(split_rows)
    ready = not missing and not below_minimum and not exact_conflicts and not near_conflicts
    source_counts = Counter(str(row.get("source") or "unknown") for row in split_rows)
    review_counts = Counter(str(row.get("review_status") or "missing") for row in split_rows)
    license_counts = Counter(str(row.get("license") or "unknown") for row in split_rows)

    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "taxonomy_version": semantics["taxonomy_version"],
        "input_manifests": [str(path.resolve()) for path in args.input_manifests],
        "input_rows": len(rows),
        "accepted_samples": len(split_rows),
        "covered_leaf_count": sum(1 for values in counts.values() if values["total"] > 0),
        "ready_for_training": ready,
        "thresholds": {
            "min_dimension": args.min_dimension,
            "near_duplicate_distance": args.near_duplicate_distance,
            "min_train_per_leaf": args.min_train_per_leaf,
            "min_val_per_leaf": args.min_val_per_leaf,
            "min_test_per_leaf": args.min_test_per_leaf,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
        },
        "counts": counts,
        "source_counts": dict(sorted(source_counts.items())),
        "review_status_counts": dict(sorted(review_counts.items())),
        "allowed_review_statuses": sorted(args.allowed_review_statuses),
        "license_counts": dict(sorted(license_counts.items())),
        "missing_leaves": missing,
        "leaves_below_minimum": below_minimum,
        "rejected_rows": rejected,
        "exact_duplicates": exact_duplicates,
        "exact_cross_label_conflicts": exact_conflicts,
        "near_duplicates": near_duplicates,
        "near_cross_label_conflicts": near_conflicts,
        "creator_leakage": leakage,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    report_path = args.output_dir / "quality_report.json"
    counts_path = args.output_dir / "class_counts.csv"
    markdown_path = args.output_dir / "DATA_QUALITY.md"
    write_manifest(manifest_path, split_rows)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_counts_csv(counts_path, counts, display_names)
    write_audit_markdown(markdown_path, report, display_names)
    print(
        json.dumps(
            {
                "ready_for_training": ready,
                "input_rows": len(rows),
                "accepted_samples": len(split_rows),
                "covered_leaf_count": report["covered_leaf_count"],
                "missing_leaves": missing,
                "leaves_below_minimum": below_minimum,
                "exact_cross_label_conflicts": len(exact_conflicts),
                "near_cross_label_conflicts": len(near_conflicts),
                "output_manifest": str(manifest_path),
                "quality_report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if not ready and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
