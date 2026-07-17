from __future__ import annotations

import json
import logging
import os
import sys
import threading
from collections import Counter, deque
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def configure_json_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    log_path = os.getenv("CATAI_JSON_LOG_PATH", "").strip()
    if log_path:
        path = Path(log_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        max_bytes = max(
            1024,
            int(os.getenv("CATAI_LOG_MAX_BYTES", str(20 * 1024 * 1024))),
        )
        backup_count = max(1, int(os.getenv("CATAI_LOG_BACKUP_COUNT", "5")))
        handler: logging.Handler = RotatingFileHandler(
            path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8",
        )
    else:
        handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(os.getenv("CATAI_LOG_LEVEL", "INFO").upper())
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    payload = {
        "timestamp": utc_now(),
        "event": event,
        **fields,
    }
    logger.info(json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str))


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class InferenceTelemetry:
    """Bounded, process-local inference telemetry without image or OCR contents."""

    def __init__(self, max_records: int = 1000) -> None:
        self._records: deque[dict[str, Any]] = deque(maxlen=max_records)
        self._lock = threading.Lock()
        self._total_requests = 0
        self._total_errors = 0

    def record(
        self,
        *,
        status: str,
        total_ms: float,
        stages_ms: dict[str, float] | None = None,
        model: str | None = None,
        device: str | None = None,
    ) -> None:
        record = {
            "status": status,
            "total_ms": float(total_ms),
            "stages_ms": {
                str(key): float(value) for key, value in (stages_ms or {}).items()
            },
            "model": model,
            "device": device,
        }
        with self._lock:
            self._records.append(record)
            self._total_requests += 1
            if status != "ok":
                self._total_errors += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            records = list(self._records)
            total_requests = self._total_requests
            total_errors = self._total_errors

        totals = [float(record["total_ms"]) for record in records]
        stage_names = sorted(
            {name for record in records for name in record["stages_ms"]}
        )
        stages = {}
        for name in stage_names:
            values = [
                float(record["stages_ms"][name])
                for record in records
                if name in record["stages_ms"]
            ]
            stages[name] = {
                "samples": len(values),
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "max_ms": max(values) if values else None,
            }

        return {
            "generated_at": utc_now(),
            "process_window_samples": len(records),
            "total_requests": total_requests,
            "total_errors": total_errors,
            "status_counts": dict(Counter(str(record["status"]) for record in records)),
            "latency": {
                "p50_ms": _percentile(totals, 0.50),
                "p95_ms": _percentile(totals, 0.95),
                "max_ms": max(totals) if totals else None,
            },
            "stages": stages,
            "models": dict(Counter(str(record["model"]) for record in records if record["model"])),
            "devices": dict(
                Counter(str(record["device"]) for record in records if record["device"])
            ),
        }
