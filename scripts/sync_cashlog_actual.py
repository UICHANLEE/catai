#!/usr/bin/env python3
"""CLI wrapper for the private CashLog actual-dataset materializer."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catai.actual_dataset import main  # noqa: E402


if __name__ == "__main__":
    main()
