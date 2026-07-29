#!/usr/bin/env python3
"""Download a large HTTP asset with resumable byte-range segments."""

from __future__ import annotations

import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def remote_size(url: str, timeout: float) -> int:
    request = Request(url, method="HEAD", headers={"User-Agent": "CashlogDatasetBuilder/0.1"})
    with urlopen(request, timeout=timeout) as response:
        if str(response.headers.get("Accept-Ranges") or "").lower() != "bytes":
            raise RuntimeError("server does not advertise byte-range support")
        value = response.headers.get("Content-Length")
        if not value:
            raise RuntimeError("server did not provide Content-Length")
        return int(value)


def segment_ranges(total: int, segments: int) -> list[tuple[int, int]]:
    if total <= 0 or segments <= 0:
        raise ValueError("total and segments must be positive")
    width = (total + segments - 1) // segments
    return [
        (start, min(total - 1, start + width - 1))
        for start in range(0, total, width)
    ]


def download_part(
    url: str,
    path: Path,
    start: int,
    end: int,
    timeout: float,
    retries: int,
) -> int:
    expected = end - start + 1
    path.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return expected
        if current > expected:
            path.unlink()
            current = 0
        request_start = start + current
        request = Request(
            url,
            headers={
                "Range": f"bytes={request_start}-{end}",
                "User-Agent": "CashlogDatasetBuilder/0.1",
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, received {response.status}")
                with path.open("ab") as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
        except (HTTPError, URLError, TimeoutError, OSError, RuntimeError):
            if attempt + 1 == retries:
                raise
            time.sleep(min(30.0, 2.0**attempt))
    actual = path.stat().st_size
    if actual != expected:
        raise RuntimeError(f"part size mismatch for {path}: {actual} != {expected}")
    return actual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--segments", type=int, default=24)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--retries", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    total = remote_size(args.url, args.timeout)
    ranges = segment_ranges(total, args.segments)
    parts_dir = args.output.with_name(args.output.name + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    first_part = parts_dir / "part-00000"
    if args.output.exists() and args.output.stat().st_size < total and not first_part.exists():
        first_expected = ranges[0][1] - ranges[0][0] + 1
        if args.output.stat().st_size <= first_expected:
            args.output.replace(first_part)

    with ThreadPoolExecutor(max_workers=min(args.workers, len(ranges))) as pool:
        futures = {
            pool.submit(
                download_part,
                args.url,
                parts_dir / f"part-{index:05d}",
                start,
                end,
                args.timeout,
                args.retries,
            ): index
            for index, (start, end) in enumerate(ranges)
        }
        completed = 0
        downloaded = 0
        for future in as_completed(futures):
            downloaded += future.result()
            completed += 1
            print(
                f"segments {completed}/{len(ranges)} "
                f"completed_bytes={downloaded}/{total}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as destination:
        for index, (start, end) in enumerate(ranges):
            part = parts_dir / f"part-{index:05d}"
            expected = end - start + 1
            if part.stat().st_size != expected:
                raise RuntimeError(f"cannot assemble incomplete part: {part}")
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, length=8 * 1024 * 1024)
    if args.output.stat().st_size != total:
        raise RuntimeError(f"assembled size mismatch: {args.output.stat().st_size} != {total}")
    print(f"downloaded={args.output} bytes={total}", flush=True)


if __name__ == "__main__":
    main()
