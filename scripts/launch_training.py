#!/usr/bin/env python3
"""Launch long-running training detached from the current terminal."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

JOBS = {
    "uecfood256_target80_plateau100": [
        "bash",
        "scripts/train_until_80_plateau_uecfood256.sh",
    ],
    "uecfood256_target90_plateau100": [
        "bash",
        "scripts/train_until_90_plateau_uecfood256.sh",
    ],
    "monitor_uecfood256_target90_plateau100": [
        ".venv/bin/python",
        "scripts/monitor_uecfood256_training.py",
    ],
    "cashlog_category_uecfood_mps": [
        ".venv/bin/python",
        "scripts/train_cashlog_category_from_uecfood.py",
        "--device",
        "mps",
        "--output-dir",
        "checkpoints/cashlog_category_uecfood_mps",
        "--epochs",
        "30",
        "--batch-size",
        "32",
        "--log-interval",
        "50",
    ],
}


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in JOBS:
        names = ", ".join(sorted(JOBS))
        raise SystemExit(f"usage: {sys.argv[0]} <job>\navailable: {names}")

    job = sys.argv[1]
    log_path = LOG_DIR / f"{job}.log"
    pid_path = LOG_DIR / f"{job}.pid"

    log_file = log_path.open("ab", buffering=0)
    process = subprocess.Popen(
        JOBS[job],
        cwd=ROOT,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid_path.write_text(f"{process.pid}\n")
    print(f"started {job} pid={process.pid}")
    print(f"log={log_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
