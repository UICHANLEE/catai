#!/usr/bin/env python3
"""Promote a compact candidate only when its recorded evaluation gate passes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import hashlib
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / "reports/cashlog33/million_v1/compact_comparison.json"
DEFAULT_CANDIDATE = ROOT / "configs/cashlog/hybrid.million-int8-candidate.json"
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
) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not report.get("promotion_gate_passed"):
        failed = [
            name
            for name, passed in report.get("gate_checks", {}).items()
            if not passed
        ]
        raise RuntimeError(f"promotion gate failed: {failed}")
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    compact = candidate.get("compact_vision", {})
    if not compact.get("enabled") or compact.get("runtime") != "onnxruntime-int8":
        raise RuntimeError("candidate is not an enabled INT8 ONNX configuration")
    expected_serving_sha = str(report.get("serving_config_sha256") or "")
    if not expected_serving_sha or sha256_file(serving_path) != expected_serving_sha:
        raise RuntimeError("serving config changed after candidate evaluation")
    candidate_base = candidate_path.parent
    model_path = (candidate_base / str(compact["model"])).resolve()
    labels_path = (candidate_base / str(compact["labels"])).resolve()
    if not model_path.is_file() or not labels_path.is_file():
        raise RuntimeError("candidate model or labels are missing")
    model_sha = sha256_file(model_path)
    labels_sha = sha256_file(labels_path)
    if model_sha != str(compact.get("model_sha256")):
        raise RuntimeError("candidate INT8 model checksum mismatch")
    if labels_sha != str(compact.get("labels_sha256")):
        raise RuntimeError("candidate labels checksum mismatch")
    evaluated_model_sha = str(
        report.get("int8", {}).get("artifact", {}).get("sha256") or ""
    )
    if not evaluated_model_sha or evaluated_model_sha != model_sha:
        raise RuntimeError("candidate INT8 model was not the evaluated artifact")

    serving_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(serving_path, rollback_path)
    temporary = serving_path.with_suffix(serving_path.suffix + ".tmp")
    shutil.copy2(candidate_path, temporary)
    os.replace(temporary, serving_path)
    return {
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "model_version": candidate["model_version"],
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in [args.report, args.candidate, args.serving]:
        if not path.is_file():
            raise SystemExit(f"missing input: {path}")
    result = promote(args.report, args.candidate, args.serving, args.rollback)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
