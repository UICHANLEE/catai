#!/usr/bin/env python3
"""Promote an improved native SigLIP2 head with an audited user override."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/cashlog33/siglip_expanded_v1/final_comparison.json"
DEFAULT_CANDIDATE = ROOT / "configs/cashlog/hybrid.siglip-expanded-candidate.json"
DEFAULT_SERVING = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_ROLLBACK = ROOT / "configs/cashlog/hybrid.serving.previous.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote(
    report_path: Path,
    candidate_path: Path,
    serving_path: Path,
    rollback_path: Path,
    *,
    approved_recall_regression: bool,
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    baseline = report["baseline"]
    native = report["native"]
    if not (
        float(native["top1_accuracy"]) > float(baseline["top1_accuracy"])
        and float(native["macro_f1"]) > float(baseline["macro_f1"])
    ):
        raise RuntimeError("native candidate does not improve Top-1 and Macro-F1")
    recall_passed = bool(
        report.get("gate_checks", {}).get(
            "native_leaf_recall_regression_within_limit", False
        )
    )
    if not recall_passed and not approved_recall_regression:
        raise RuntimeError("native leaf recall gate requires explicit approval")
    if candidate.get("compact_vision", {}).get("enabled"):
        raise RuntimeError("native promotion candidate cannot enable compact vision")

    expected_serving_sha = str(report.get("serving_config_sha256") or "")
    if not expected_serving_sha or sha256_file(serving_path) != expected_serving_sha:
        raise RuntimeError("serving config changed after candidate evaluation")
    head_path = (
        candidate_path.parent / str(candidate["vision_head"])
    ).resolve()
    if not head_path.is_file():
        raise RuntimeError("candidate vision head is missing")
    head_sha = sha256_file(head_path)
    if head_sha != str(candidate.get("vision_head_sha256") or ""):
        raise RuntimeError("candidate vision head checksum mismatch")
    if head_sha != str(report.get("export", {}).get("vision_head_sha256") or ""):
        raise RuntimeError("candidate vision head was not the evaluated artifact")

    serving_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(serving_path, rollback_path)
    temporary = serving_path.with_suffix(serving_path.suffix + ".tmp")
    shutil.copy2(candidate_path, temporary)
    os.replace(temporary, serving_path)
    return {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "model_version": candidate["model_version"],
        "vision_head_sha256": head_sha,
        "approved_recall_regression": not recall_passed,
        "serving_config": str(serving_path),
        "rollback_config": str(rollback_path),
        "evaluation_report": str(report_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_CANDIDATE)
    parser.add_argument("--serving", type=Path, default=DEFAULT_SERVING)
    parser.add_argument("--rollback", type=Path, default=DEFAULT_ROLLBACK)
    parser.add_argument("--approved-recall-regression", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = promote(
        args.report,
        args.candidate,
        args.serving,
        args.rollback,
        approved_recall_regression=args.approved_recall_regression,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
