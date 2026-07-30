#!/usr/bin/env python3
"""Score the currently served visual ensemble on the frozen proxy test."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, recall_score

from catai.cashlog_hybrid_classifier import CashlogHybridClassifier

try:
    from scripts.evaluate_cashlog33_compact import (
        DEFAULT_EXTERNAL_MANIFEST,
        DEFAULT_SPLIT,
        load_frozen_test_rows,
    )
except ModuleNotFoundError:
    from evaluate_cashlog33_compact import (
        DEFAULT_EXTERNAL_MANIFEST,
        DEFAULT_SPLIT,
        load_frozen_test_rows,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/cashlog/hybrid.serving.json"
DEFAULT_OUTPUT = (
    ROOT / "reports/cashlog33/million_v1/current_serving_visual_baseline.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_EXTERNAL_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_frozen_test_rows(args.manifest, args.split)
    if args.limit:
        rows = rows[: args.limit]
    visual_labels = sorted({str(row["leaf_id"]) for row in rows})
    label_to_index = {
        leaf_id: index for index, leaf_id in enumerate(visual_labels)
    }
    classifier = CashlogHybridClassifier(args.config, device=args.device)
    expected: list[int] = []
    predicted: list[int] = []
    top3_hits = 0
    timings = []
    for index, row in enumerate(rows, start=1):
        with Image.open(ROOT / str(row["relative_path"])) as source:
            image = source.convert("RGB")
        started = time.perf_counter()
        scores = classifier._vision_result(image)["scores"]
        timings.append((time.perf_counter() - started) * 1000.0)
        ranked = sorted(scores, key=scores.get, reverse=True)
        expected_leaf = str(row["leaf_id"])
        predicted_leaf = ranked[0]
        expected.append(label_to_index[expected_leaf])
        predicted.append(label_to_index.get(predicted_leaf, -1))
        top3_hits += int(expected_leaf in ranked[:3])
        if index % args.log_interval == 0 or index == len(rows):
            print(
                json.dumps(
                    {
                        "event": "baseline_progress",
                        "completed": index,
                        "total": len(rows),
                        "mean_vision_ms": float(np.mean(timings)),
                    }
                ),
                flush=True,
            )
    recalls = recall_score(
        expected,
        predicted,
        labels=list(range(len(visual_labels))),
        average=None,
        zero_division=0,
    )
    supports = Counter(str(row["leaf_id"]) for row in rows)
    baseline = {
        "samples": len(expected),
        "top1_accuracy": float(
            np.mean(np.asarray(expected) == np.asarray(predicted))
        ),
        "top3_accuracy": top3_hits / max(1, len(expected)),
        "macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=list(range(len(visual_labels))),
                average="macro",
                zero_division=0,
            )
        ),
        "per_leaf_recall": {
            leaf_id: float(recalls[index])
            for index, leaf_id in enumerate(visual_labels)
        },
        "per_leaf_support": {
            leaf_id: supports[leaf_id] for leaf_id in visual_labels
        },
        "runtime": {
            "device": args.device,
            "mean_vision_ms": float(np.mean(timings)),
            "p50_vision_ms": float(np.percentile(timings, 50)),
            "p95_vision_ms": float(np.percentile(timings, 95)),
        },
    }
    report: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_version": classifier.config["model_version"],
        "config": repository_path(args.config),
        "config_sha256": sha256_file(args.config),
        "evaluation_set": {
            "kind": "external_visual_proxy",
            "official_split": "Open Images validation",
            "samples": len(rows),
            "manifest_sha256": sha256_file(args.manifest),
            "split_sha256": sha256_file(args.split),
        },
        "baseline": baseline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if classifier._executor is not None:
        classifier._executor.shutdown(wait=False)


if __name__ == "__main__":
    main()
