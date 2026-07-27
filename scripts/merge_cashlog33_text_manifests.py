#!/usr/bin/env python3
"""Merge text manifests while preserving every row and enforcing split isolation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/text/all_500k_v2/manifest.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_ids: set[str] = set()
    group_splits: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    leaf_counts: dict[str, Counter[str]] = defaultdict(Counter)
    inputs: list[dict[str, Any]] = []
    with args.output.open("w", encoding="utf-8") as output_handle:
        for manifest_path in args.manifest:
            inputs.append(
                {
                    "path": str(manifest_path.resolve()),
                    "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                }
            )
            with manifest_path.open(encoding="utf-8") as input_handle:
                for line_number, line in enumerate(input_handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample_id = str(row.get("sample_id") or "")
                    split = str(row.get("split") or "")
                    group_key = str(row.get("group_key") or sample_id)
                    leaf_id = str(row.get("leaf_id") or "")
                    if split == "validation":
                        split = "val"
                        row["split"] = split
                    if split not in {"train", "val", "test"}:
                        raise SystemExit(
                            f"{manifest_path}:{line_number} has invalid split {split}"
                        )
                    if not sample_id or sample_id in sample_ids:
                        raise SystemExit(f"duplicate/empty sample_id: {sample_id}")
                    sample_ids.add(sample_id)
                    group_splits[group_key].add(split)
                    split_counts[split] += 1
                    leaf_counts[split][leaf_id] += 1
                    provenance_counts[
                        str(row.get("provenance_type") or "unknown")
                    ] += 1
                    output_handle.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                    )
    leaking_groups = sorted(
        group_key for group_key, splits in group_splits.items() if len(splits) > 1
    )
    if leaking_groups:
        args.output.unlink(missing_ok=True)
        raise SystemExit(f"group leakage across splits: {leaking_groups[:10]}")
    missing_leaves = {
        split: 33 - len(counts) for split, counts in leaf_counts.items()
    }
    if any(missing_leaves.values()):
        raise SystemExit(f"merged manifest lacks leaf coverage: {missing_leaves}")
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "rows": len(sample_ids),
        "split_counts": dict(split_counts),
        "provenance_counts": dict(provenance_counts),
        "leaf_counts": {
            split: dict(sorted(counts.items()))
            for split, counts in sorted(leaf_counts.items())
        },
        "group_count": len(group_splits),
        "group_leakage": 0,
        "inputs": inputs,
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    args.output.with_name("quality_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
