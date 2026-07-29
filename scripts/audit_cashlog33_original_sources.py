#!/usr/bin/env python3
"""Audit original-image provenance and leaf coverage across CashLog manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFESTS = [
    ROOT / "data/raw/cashlog33/actual/manifest.jsonl",
    ROOT / "data/raw/cashlog33/openimages_v7/manifest.jsonl",
    ROOT / "data/raw/cashlog33/openverse_smoke/manifest.jsonl",
    ROOT / "data/raw/cashlog33/open_products_api/manifest.jsonl",
    ROOT / "data/raw/cashlog33/open_products_api_250/manifest.jsonl",
    ROOT / "data/raw/cashlog33/cord_v2/manifest.jsonl",
    ROOT / "data/raw/cashlog33/abo_v1/manifest.jsonl",
]
DEFAULT_OUTPUT = ROOT / "reports/cashlog33/originals_v2/source_audit.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit(manifests: list[Path]) -> dict[str, Any]:
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_reports: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for path in manifests:
        rows = read_jsonl(path)
        all_rows.extend(rows)
        for row in rows:
            digest = str(row.get("sha256") or "")
            if digest:
                by_hash[digest].append(row)
        source_reports.append(
            {
                "manifest": str(path),
                "rows": len(rows),
                "unique_hashes": len({str(row.get("sha256")) for row in rows if row.get("sha256")}),
                "source_counts": dict(
                    sorted(Counter(str(row.get("source") or "unknown") for row in rows).items())
                ),
                "leaf_counts": dict(
                    sorted(Counter(str(row.get("leaf_id") or "ocr_only") for row in rows).items())
                ),
                "license_counts": dict(
                    sorted(Counter(str(row.get("license") or "unknown") for row in rows).items())
                ),
            }
        )
    conflicts = []
    duplicate_hashes = 0
    for digest, rows in by_hash.items():
        if len(rows) < 2:
            continue
        duplicate_hashes += 1
        leaves = sorted({str(row.get("leaf_id") or "ocr_only") for row in rows})
        if len(leaves) > 1:
            conflicts.append(
                {
                    "sha256": digest,
                    "leaf_ids": leaves,
                    "sample_ids": sorted(str(row.get("sample_id")) for row in rows),
                }
            )
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "manifest_count": len(manifests),
        "rows": len(all_rows),
        "unique_hashes": len(by_hash),
        "duplicate_hashes": duplicate_hashes,
        "cross_leaf_hash_conflicts": conflicts,
        "leaf_counts_before_cross_source_dedup": dict(
            sorted(Counter(str(row.get("leaf_id") or "ocr_only") for row in all_rows).items())
        ),
        "source_counts_before_cross_source_dedup": dict(
            sorted(Counter(str(row.get("source") or "unknown") for row in all_rows).items())
        ),
        "sources": source_reports,
        "scope_warning": "Source labels and OCR-only images are not equivalent to human-verified CashLog labels.",
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# CashLog33 원본 데이터 감사",
        "",
        f"- manifest: {report['manifest_count']}개",
        f"- row: {report['rows']:,}개",
        f"- 고유 SHA-256: {report['unique_hashes']:,}개",
        f"- 중복 hash: {report['duplicate_hashes']:,}개",
        f"- cross-leaf hash 충돌: {len(report['cross_leaf_hash_conflicts']):,}개",
        "",
        "## 출처별 수량",
        "",
        "| manifest | rows | unique hashes |",
        "|---|---:|---:|",
    ]
    for source in report["sources"]:
        lines.append(
            f"| `{source['manifest']}` | {source['rows']:,} | {source['unique_hashes']:,} |"
        )
    lines.extend(
        [
            "",
            "## 주의",
            "",
            "API·객체·상품 타입 기반 라벨은 weak label이다. 실제 CashLog 사람이 확인한",
            "라벨과 OCR-only 영수증을 같은 정확도 근거로 합산하면 안 된다.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", type=Path, dest="manifests")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifests = args.manifests or DEFAULT_MANIFESTS
    missing = [path for path in manifests if not path.is_file()]
    if missing:
        raise SystemExit(f"missing manifests: {missing}")
    report = audit(manifests)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_markdown(args.output.with_suffix(".md"), report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
