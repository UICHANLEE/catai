#!/usr/bin/env python3
"""Apply explicit integration and production promotion gates to a hybrid candidate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_VISION = ROOT / "checkpoints/cashlog33/vision_head_v1/metrics.json"
DEFAULT_TEXT = ROOT / "checkpoints/cashlog33/text_sgd_v2/metrics.json"
DEFAULT_E2E = ROOT / "reports/cashlog33/hybrid_e2e/metrics.json"
DEFAULT_E2E_PER_CLASS = ROOT / "reports/cashlog33/hybrid_e2e/per_class_metrics.csv"
DEFAULT_OUTPUT = ROOT / "reports/cashlog33/model_selection.json"


PRODUCTION_THRESHOLDS = {
    "minimum_samples": 330,
    "minimum_per_leaf": 10,
    "top1_accuracy": 0.95,
    "top3_accuracy": 0.95,
    "macro_f1": 0.75,
    "minimum_leaf_recall": 0.60,
    "maximum_ece": 0.08,
    "maximum_false_auto_confirm_rate": 0.02,
    "maximum_latency_p95_seconds": 3.0,
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def gate(name: str, passed: bool, actual: Any, required: Any, scope: str) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "required": required,
        "scope": scope,
    }


def minimum_recall(path: Path) -> float | None:
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    values = [float(row["recall"]) for row in rows if int(float(row["support"])) > 0]
    return min(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--vision-metrics", type=Path, default=DEFAULT_VISION)
    parser.add_argument("--text-metrics", type=Path, default=DEFAULT_TEXT)
    parser.add_argument("--e2e-metrics", type=Path, default=DEFAULT_E2E)
    parser.add_argument("--e2e-per-class", type=Path, default=DEFAULT_E2E_PER_CLASS)
    parser.add_argument("--real-holdout-metrics", type=Path)
    parser.add_argument("--real-holdout-per-class", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--airflow-dag-run-id", default=os.getenv("AIRFLOW_CTX_DAG_RUN_ID")
    )
    parser.add_argument("--airflow-dag-id", default=os.getenv("AIRFLOW_CTX_DAG_ID"))
    parser.add_argument("--require-production-ready", action="store_true")
    return parser.parse_args()


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# CashLog 33 Model Selection",
        "",
        f"- Generated: `{result['generated_at']}`",
        f"- Candidate: `{result['selected_candidate']}`",
        f"- Decision: **{result['decision']}**",
        f"- Production eligible: **{str(result['production_eligible']).lower()}**",
        "",
        "## Gate Results",
        "",
        "| Gate | Result | Actual | Required | Scope |",
        "|---|---:|---:|---:|---|",
    ]
    for item in result["gates"]:
        lines.append(
            f"| {item['name']} | {'PASS' if item['passed'] else 'FAIL'} | "
            f"`{item['actual']}` | `{item['required']}` | {item['scope']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Note",
            "",
            result["decision_note"],
            "",
            "Proxy and synthetic metrics validate components and integration only. They do not "
            "replace a frozen, manually labeled CashLog photo holdout.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    vision = load_json(args.vision_metrics)
    text = load_json(args.text_metrics)
    e2e = load_json(args.e2e_metrics)

    categories = load_json(resolve(config_path, str(config["categories"])))
    artifacts = {
        "vision_model": resolve(config_path, str(config["vision_model"])) / "model.safetensors",
        "vision_head": resolve(config_path, str(config["vision_head"])),
        "text_model": resolve(config_path, str(config["text_model"])),
        "ocr_detector_model": resolve(config_path, str(config["ocr_detector_model"])),
        "ocr_classifier_model": resolve(config_path, str(config["ocr_classifier_model"])),
        "ocr_model": resolve(config_path, str(config["ocr_model"])),
    }
    expected_hashes = {
        "vision_model": config["vision_model_sha256"],
        "vision_head": config["vision_head_sha256"],
        "text_model": config["text_model_sha256"],
        "ocr_detector_model": config["ocr_detector_model_sha256"],
        "ocr_classifier_model": config["ocr_classifier_model_sha256"],
        "ocr_model": config["ocr_model_sha256"],
    }
    actual_hashes = {
        key: sha256(path) if path.is_file() else None for key, path in artifacts.items()
    }
    artifact_hashes_valid = all(
        actual_hashes[key] == expected_hashes[key] for key in expected_hashes
    )

    integration_gates = [
        gate("taxonomy_leaf_count", len(categories) == 33, len(categories), 33, "contract"),
        gate(
            "artifact_sha256",
            artifact_hashes_valid,
            artifact_hashes_valid,
            True,
            "supply-chain",
        ),
        gate(
            "vision_head_beats_zero_shot_top1",
            vision["test_linear_head"]["top1_accuracy"]
            > vision["test_zero_shot_same_split"]["top1_accuracy"],
            vision["test_linear_head"]["top1_accuracy"],
            f"> {vision['test_zero_shot_same_split']['top1_accuracy']}",
            "Open Images proxy",
        ),
        gate(
            "synthetic_33_leaf_coverage",
            e2e["leaf_count"] == 33,
            e2e["leaf_count"],
            33,
            "synthetic I/O",
        ),
        gate(
            "synthetic_top1",
            e2e["top1_accuracy"] >= 0.95,
            e2e["top1_accuracy"],
            ">= 0.95",
            "synthetic I/O",
        ),
        gate(
            "synthetic_top3",
            e2e["top3_accuracy"] >= 0.90,
            e2e["top3_accuracy"],
            ">= 0.90",
            "synthetic I/O",
        ),
        gate(
            "synthetic_false_auto_confirm",
            e2e["false_auto_confirm_rate"] <= 0.02,
            e2e["false_auto_confirm_rate"],
            "<= 0.02",
            "synthetic I/O",
        ),
        gate(
            "synthetic_latency_p95",
            e2e["latency_p95_seconds"] <= PRODUCTION_THRESHOLDS["maximum_latency_p95_seconds"],
            e2e["latency_p95_seconds"],
            f"<= {PRODUCTION_THRESHOLDS['maximum_latency_p95_seconds']}",
            f"local {e2e.get('device', 'unknown')} synthetic I/O",
        ),
    ]

    real = load_json(args.real_holdout_metrics) if args.real_holdout_metrics else None
    real_recall = (
        minimum_recall(args.real_holdout_per_class)
        if args.real_holdout_per_class
        else None
    )
    if real:
        production_gates = [
            gate("real_samples", real["samples"] >= 330, real["samples"], ">= 330", "real holdout"),
            gate("real_leaf_count", real["leaf_count"] == 33, real["leaf_count"], 33, "real holdout"),
            gate("real_minimum_per_leaf", real["minimum_per_leaf"] >= 10, real["minimum_per_leaf"], ">= 10", "real holdout"),
            gate("real_top1", real["top1_accuracy"] >= 0.95, real["top1_accuracy"], ">= 0.95", "real holdout"),
            gate("real_top3", real["top3_accuracy"] >= 0.95, real["top3_accuracy"], ">= 0.95", "real holdout"),
            gate("real_macro_f1", real["macro_f1"] >= 0.75, real["macro_f1"], ">= 0.75", "real holdout"),
            gate("real_minimum_recall", real_recall is not None and real_recall >= 0.60, real_recall, ">= 0.60", "real holdout"),
            gate("real_ece", real["ece_10_bin"] <= 0.08, real["ece_10_bin"], "<= 0.08", "real holdout"),
            gate("real_false_auto_confirm", real["false_auto_confirm_rate"] <= 0.02, real["false_auto_confirm_rate"], "<= 0.02", "real holdout"),
        ]
    else:
        production_gates = [
            gate(
                "real_cashlog_holdout",
                False,
                "missing",
                "frozen manual set: >=10 photos x 33 leaves",
                "real holdout",
            )
        ]

    integration_ready = all(item["passed"] for item in integration_gates)
    production_eligible = integration_ready and all(
        item["passed"] for item in production_gates
    )
    if production_eligible:
        decision = "production_candidate"
        note = "All configured gates passed. Promotion still requires an explicit deployment approval."
    elif integration_ready:
        decision = "guarded_integration_candidate"
        note = (
            "The hybrid is selected for guarded Top-3 recommendation serving. Auto-confirm stays "
            "disabled until the frozen real-photo holdout passes every production gate."
        )
    else:
        decision = "rejected"
        note = "At least one component or integration gate failed; do not deploy this candidate."

    result = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "selected_candidate": config["model_version"],
        "serving_config": portable_path(config_path),
        "decision": decision,
        "integration_ready": integration_ready,
        "production_eligible": production_eligible,
        "auto_confirm_enabled": bool(config["decision"].get("allow_auto_confirm", False)),
        "decision_note": note,
        "orchestration": {
            "airflow_dag_id": args.airflow_dag_id,
            "airflow_dag_run_id": args.airflow_dag_run_id,
        },
        "thresholds": PRODUCTION_THRESHOLDS,
        "gates": [*integration_gates, *production_gates],
        "artifact_sha256": actual_hashes,
        "evidence": {
            "vision_metrics": portable_path(args.vision_metrics),
            "text_metrics": portable_path(args.text_metrics),
            "e2e_metrics": portable_path(args.e2e_metrics),
            "real_holdout_metrics": (
                portable_path(args.real_holdout_metrics) if args.real_holdout_metrics else None
            ),
            "text_test_scope": text.get("test_scope"),
            "e2e_scope": e2e.get("scope_warning"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.output.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.require_production_ready and not production_eligible:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
