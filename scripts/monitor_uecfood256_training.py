#!/usr/bin/env python3
"""Monitor long UECFood256 training and restart it if it stalls below target."""

from __future__ import annotations

import csv
import os
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOB = "uecfood256_target90_plateau100"
LOG = ROOT / "logs" / f"{JOB}.log"
MONITOR_LOG = ROOT / "logs" / f"{JOB}.monitor.log"
PID = ROOT / "logs" / f"{JOB}.pid"
OUTPUT = ROOT / "checkpoints" / "uecfood256_mobilenetv4_target90_plateau100"
METRICS = OUTPUT / "metrics.csv"
LAST = OUTPUT / "last.pt"
TARGET_TOP1 = 90.0
CHECK_SECONDS = 300
STALL_SECONDS = 14400


BASE_COMMAND = [
    ".venv/bin/python",
    "scripts/train_uecfood256_mobilenetv4.py",
    "--output-dir",
    "checkpoints/uecfood256_mobilenetv4_target90_plateau100",
    "--epochs",
    "300",
    "--batch-size",
    "32",
    "--lr",
    "0.0003",
    "--weight-decay",
    "0.0001",
    "--label-smoothing",
    "0.1",
    "--bbox-padding",
    "0.1",
    "--random-erasing",
    "0.15",
    "--mixup-alpha",
    "0.1",
    "--freeze-backbone-epochs",
    "1",
    "--target-top1",
    "90",
    "--patience-after-target",
    "100",
    "--min-delta",
    "0.01",
    "--log-interval",
    "50",
]


def note(message: str) -> None:
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n"
    with MONITOR_LOG.open("a") as f:
        f.write(line)
    print(line, end="", flush=True)


def pid_alive() -> bool:
    if not PID.exists():
        return False
    try:
        pid = int(PID.read_text().strip())
    except ValueError:
        return False
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:
        return log_is_fresh()
    except ProcessLookupError:
        return False


def log_is_fresh() -> bool:
    if not LOG.exists():
        return False
    return time.time() - LOG.stat().st_mtime < STALL_SECONDS


def best_metrics() -> tuple[float, int, int]:
    if not METRICS.exists() or METRICS.stat().st_size == 0:
        return -1.0, 0, 0
    with METRICS.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return -1.0, 0, 0
    last = rows[-1]
    return (
        float(last.get("best_top1") or last.get("val_top1") or -1.0),
        int(last.get("epoch") or 0),
        int(last.get("epochs_since_improvement") or 0),
    )


def last_log_has_traceback() -> bool:
    if not LOG.exists():
        return False
    tail = LOG.read_text(errors="replace")[-8000:]
    return "Traceback (most recent call last)" in tail or "Error" in tail or "ValueError" in tail


def launch() -> None:
    command = BASE_COMMAND.copy()
    if LAST.exists():
        command.extend(["--resume", str(LAST.relative_to(ROOT))])
    LOG.parent.mkdir(exist_ok=True)
    log_file = LOG.open("ab", buffering=0)
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    PID.write_text(f"{process.pid}\n")
    note(f"launched pid={process.pid} resume={LAST.exists()}")


def main() -> None:
    note("monitor started")
    while True:
        best, epoch, no_improve = best_metrics()
        fresh = log_is_fresh()
        alive = pid_alive()
        traceback = last_log_has_traceback()
        note(f"status alive={alive} fresh_log={fresh} epoch={epoch} best_top1={best:.4f} no_improve={no_improve}")

        if best >= TARGET_TOP1 and no_improve >= 100:
            note("target reached and plateau patience satisfied; monitor exiting")
            return

        if (not alive or traceback) and best < TARGET_TOP1:
            note("training appears stopped/stalled below target; relaunching")
            launch()
        elif alive and not fresh and best < TARGET_TOP1:
            note("training process is alive but log is stale; not relaunching to avoid duplicate workers")

        time.sleep(CHECK_SECONDS)


if __name__ == "__main__":
    main()
