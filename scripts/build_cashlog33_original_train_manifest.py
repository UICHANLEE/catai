#!/usr/bin/env python3
"""Merge train-only original-image sources without contaminating the proxy holdout."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ROOT / "data/raw/cashlog33/abo_v1/manifest.jsonl",
    ROOT / "data/processed/cashlog33/training/all_v3_500k/manifest.jsonl",
]
DEFAULT_OUTPUT = (
    ROOT / "data/processed/cashlog33/training/originals_v2_additional/manifest.jsonl"
)
EXCLUDED_SOURCES = {"openimages_v7_validation"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def merge(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if str(row.get("source") or "") not in EXCLUDED_SOURCES
        and str(row.get("split_lock") or "") == "train"
    ]
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_hash = 0
    for row in eligible:
        digest = str(row.get("sha256") or "")
        if not digest:
            missing_hash += 1
            continue
        by_hash[digest].append(row)

    merged: list[dict[str, Any]] = []
    duplicates = 0
    conflicts = 0
    for digest, group in sorted(by_hash.items()):
        leaves = {str(row["leaf_id"]) for row in group}
        if len(leaves) != 1:
            conflicts += 1
            continue
        ordered = sorted(group, key=lambda row: str(row["sample_id"]))
        selected = dict(ordered[0])
        selected["split"] = "train"
        selected["split_lock"] = "train"
        merged.append(selected)
        duplicates += len(ordered) - 1
    merged.sort(key=lambda row: (str(row["leaf_id"]), str(row["sample_id"])))
    return merged, {
        "input_rows": len(rows),
        "eligible_train_locked_rows": len(eligible),
        "accepted": len(merged),
        "missing_sha256": missing_hash,
        "duplicate_rows_dropped": duplicates,
        "cross_leaf_hash_conflicts_dropped": conflicts,
        "source_counts": dict(
            sorted(Counter(str(row.get("source") or "unknown") for row in merged).items())
        ),
        "leaf_counts": dict(sorted(Counter(str(row["leaf_id"]) for row in merged).items())),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", action="append", type=Path, dest="inputs")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs = args.inputs or DEFAULT_INPUTS
    rows = [row for path in inputs for row in read_jsonl(path)]
    merged, metrics = merge(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in merged:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "input_manifests": [str(path) for path in inputs],
        "excluded_sources": sorted(EXCLUDED_SOURCES),
        **metrics,
        "output_manifest": str(args.output),
    }
    args.output.with_name("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
