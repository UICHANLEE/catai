#!/usr/bin/env python3
"""Mirror resumable SigLIP embedding progress into a live MLflow run."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import mlflow


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROGRESS = (
    ROOT
    / "checkpoints/cashlog33/vision_head_siglip_expanded_v1/"
    "embedding_shards/progress.json"
)
DEFAULT_OUTPUT = (
    ROOT / "checkpoints/cashlog33/vision_head_siglip_expanded_v1"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress", type=Path, default=DEFAULT_PROGRESS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tracking-uri", default="http://127.0.0.1:5500"
    )
    parser.add_argument(
        "--experiment", default="cashlog33-same-siglip-expanded"
    )
    parser.add_argument(
        "--run-name", default="cashlog33-same-siglip-expanded-v1-monitor"
    )
    parser.add_argument("--interval", type=float, default=30.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stop = False

    def request_stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name) as run:
        run_file = args.output_dir / "mlflow_monitor_run.json"
        run_file.parent.mkdir(parents=True, exist_ok=True)
        run_file.write_text(
            json.dumps(
                {
                    "tracking_uri": args.tracking_uri,
                    "experiment": args.experiment,
                    "run_id": run.info.run_id,
                    "run_name": args.run_name,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mlflow.log_params(
            {
                "component": "siglip2_embedding_monitor",
                "architecture": "siglip2-base-patch16-224",
                "new_original_rows": 469001,
                "embedding_precision": "float16-mps",
                "additional_views": "original",
                "resumable_shards": True,
            }
        )
        last_step = -1
        while not stop:
            if args.progress.is_file():
                progress = json.loads(
                    args.progress.read_text(encoding="utf-8")
                )
                step = int(progress["완료_shard"])
                if step != last_step:
                    mlflow.log_metrics(
                        {
                            "completed_shards": step,
                            "completed_originals": float(
                                progress["완료_원본"]
                            ),
                            "completed_embeddings": float(
                                progress["완료_임베딩"]
                            ),
                            "originals_per_second": float(
                                progress["초당_원본"]
                            ),
                            "elapsed_seconds": float(progress["경과_초"]),
                        },
                        step=step,
                    )
                    print(
                        json.dumps(progress, ensure_ascii=False),
                        flush=True,
                    )
                    last_step = step
            metrics_path = args.output_dir / "metrics.json"
            if metrics_path.is_file():
                mlflow.log_artifact(str(metrics_path), artifact_path="final")
                break
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
