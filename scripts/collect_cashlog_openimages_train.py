#!/usr/bin/env python3
"""Collect a large, attribution-preserving Open Images train subset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import requests
from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / "configs/cashlog/openimages_leaf_mapping.json"
DEFAULT_OUTPUT = ROOT / "data/raw/cashlog33/openimages_v7_train"
CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
LABEL_URL = (
    "https://storage.googleapis.com/openimages/v7/"
    "oidv7-train-annotations-human-imagelabels.csv"
)
METADATA_URL = (
    "https://storage.googleapis.com/openimages/v6/"
    "oidv6-train-images-with-labels-with-rotation.csv"
)
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg"
CC_BY_20 = "https://creativecommons.org/licenses/by/2.0/"
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
THREAD_LOCAL = threading.local()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_large_file(url: str, path: Path, timeout: float) -> None:
    if path.is_file():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {
        "User-Agent": "CashlogDatasetBuilder/0.2 (https://github.com/UICHANLEE/catai)"
    }
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = Request(url, headers=headers)
    with urlopen(request, timeout=timeout) as response:
        mode = "ab" if offset and response.status == 206 else "wb"
        with partial.open(mode) as output:
            while chunk := response.read(8 * 1024 * 1024):
                output.write(chunk)
    os.replace(partial, path)


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def load_class_mapping(
    class_path: Path, mapping: dict[str, Any]
) -> tuple[dict[str, str], dict[str, str]]:
    with class_path.open(encoding="utf-8-sig", newline="") as handle:
        display_by_mid = {
            str(row["LabelName"]): str(row["DisplayName"])
            for row in csv.DictReader(handle)
        }
    mid_by_display = {display: mid for mid, display in display_by_mid.items()}
    configured = {name for names in mapping["leaves"].values() for name in names}
    missing = sorted(configured - set(mid_by_display))
    if missing:
        raise ValueError(f"mapping contains unknown Open Images classes: {missing}")
    leaf_by_mid = {
        mid_by_display[name]: leaf_id
        for leaf_id, names in mapping["leaves"].items()
        for name in names
    }
    return display_by_mid, leaf_by_mid


def prepare_candidates(
    label_path: Path,
    display_by_mid: dict[str, str],
    leaf_by_mid: dict[str, str],
    per_leaf: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    labels_by_image: dict[str, set[str]] = defaultdict(set)
    with label_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mid = str(row.get("LabelName") or "")
            if mid in leaf_by_mid and str(row.get("Confidence")) == "1":
                labels_by_image[str(row["ImageID"])].add(mid)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ambiguous = 0
    for image_id, mids in labels_by_image.items():
        leaves = {leaf_by_mid[mid] for mid in mids}
        if len(leaves) != 1:
            ambiguous += 1
            continue
        leaf_id = next(iter(leaves))
        candidates[leaf_id].append(
            {
                "image_id": image_id,
                "leaf_id": leaf_id,
                "source_label_ids": sorted(mids),
                "source_label_names": sorted(display_by_mid[mid] for mid in mids),
            }
        )
    selected = []
    for _, rows in sorted(candidates.items()):
        ordered = sorted(
            rows, key=lambda item: stable_key(seed, item["image_id"])
        )
        selected.extend(ordered[:per_leaf] if per_leaf is not None else ordered)
    return selected, {
        "mapped_unique_images": len(labels_by_image),
        "ambiguous_cross_leaf_images_dropped": ambiguous,
        "selected_before_metadata_validation": len(selected),
    }


def read_metadata(
    metadata_path: Path, selected_ids: set[str]
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    with metadata_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row.get("ImageID") or "")
            if image_id in selected_ids:
                output[image_id] = {key: str(value or "") for key, value in row.items()}
    return output


def request_image(url: str, timeout: float, retries: int = 6) -> bytes:
    session = getattr(THREAD_LOCAL, "http_session", None)
    if session is None:
        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "CashlogDatasetBuilder/0.2 "
                    "(https://github.com/UICHANLEE/catai)"
                )
            }
        )
        THREAD_LOCAL.http_session = session
    for attempt in range(retries):
        try:
            with session.get(url, timeout=timeout, stream=True) as response:
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise requests.HTTPError(
                        f"retryable HTTP {response.status_code}",
                        response=response,
                    )
                response.raise_for_status()
                content_length = int(response.headers.get("content-length", 0))
                if content_length > MAX_IMAGE_BYTES:
                    raise ValueError("image exceeds size limit")
                output = bytearray()
                for chunk in response.iter_content(chunk_size=256 * 1024):
                    output.extend(chunk)
                    if len(output) > MAX_IMAGE_BYTES:
                        raise ValueError("image exceeds size limit")
                payload = bytes(output)
            if len(payload) > MAX_IMAGE_BYTES:
                raise ValueError("image exceeds size limit")
            return payload
        except requests.HTTPError as exc:
            response = exc.response
            retryable = (
                response is not None
                and response.status_code in {429, 500, 502, 503, 504}
            )
            if not retryable or attempt + 1 == retries:
                raise
        except requests.RequestException:
            if attempt + 1 == retries:
                raise
        time.sleep((2**attempt) + random.random() * 0.25)
    raise RuntimeError("unreachable retry state")


def encode_image(
    image_id: str, timeout: float, max_dimension: int, min_dimension: int
) -> tuple[bytes, int, int, str]:
    payload = request_image(IMAGE_URL.format(image_id=image_id), timeout)
    try:
        with Image.open(io.BytesIO(payload)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise ValueError("decoded image exceeds pixel limit")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("downloaded object is not an image") from exc
    if min(image.size) < min_dimension:
        raise ValueError(f"minimum dimension is below {min_dimension}: {image.size}")
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=86, optimize=True)
    jpeg = output.getvalue()
    return jpeg, image.width, image.height, hashlib.sha256(jpeg).hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def source_record(url: str, path: Path) -> dict[str, Any]:
    return {
        "url": url,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_jsonl_if_exists(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    partial = path.with_suffix(path.suffix + ".tmp")
    with partial.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.replace(partial, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--per-leaf",
        type=int,
        default=0,
        help="Optional per-leaf cap; zero keeps every unambiguous mapped image.",
    )
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-dimension", type=int, default=384)
    parser.add_argument("--min-dimension", type=int, default=96)
    parser.add_argument("--seed", type=int, default=1_000_033)
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_dir / "metadata"
    class_path = metadata_dir / "class-descriptions-boxable.csv"
    label_path = metadata_dir / "train-human-imagelabels.csv"
    metadata_path = metadata_dir / "train-images-with-labels-with-rotation.csv"
    for url, path in [
        (CLASS_URL, class_path),
        (LABEL_URL, label_path),
        (METADATA_URL, metadata_path),
    ]:
        print(f"metadata {path.name}", flush=True)
        download_large_file(url, path, args.timeout)

    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    display_by_mid, leaf_by_mid = load_class_mapping(class_path, mapping)
    selected, candidate_stats = prepare_candidates(
        label_path,
        display_by_mid,
        leaf_by_mid,
        args.per_leaf if args.per_leaf > 0 else None,
        args.seed,
    )
    selected_ids = {row["image_id"] for row in selected}
    metadata = read_metadata(metadata_path, selected_ids)
    licensed = [
        row
        for row in selected
        if metadata.get(row["image_id"], {}).get("License") == CC_BY_20
    ]
    candidate_summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": "Open Images V7 human-verified train image labels",
        "per_leaf_cap": args.per_leaf if args.per_leaf > 0 else None,
        **candidate_stats,
        "metadata_rows_found": len(metadata),
        "cc_by_2_rows": len(licensed),
        "mapping": {
            "path": str(args.mapping.resolve()),
            "sha256": sha256_file(args.mapping),
            "schema_version": mapping.get("schema_version"),
        },
        "leaf_counts_before_download": dict(
            sorted(Counter(row["leaf_id"] for row in licensed).items())
        ),
    }
    (args.output_dir / "candidate_summary.json").write_text(
        json.dumps(candidate_summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(candidate_summary, ensure_ascii=False, indent=2), flush=True)
    if args.prepare_only:
        return

    manifest_path = args.output_dir / "manifest.jsonl"
    partial_manifest_path = args.output_dir / "manifest.partial.jsonl"
    recovered_rows = [
        *read_jsonl_if_exists(manifest_path),
        *read_jsonl_if_exists(partial_manifest_path),
    ]
    rows_by_id = {str(row["source_id"]): row for row in recovered_rows}
    existing_ids = set(rows_by_id)
    work = [row for row in licensed if row["image_id"] not in existing_ids]
    failures: list[dict[str, Any]] = []
    partial_manifest = partial_manifest_path.open("a", encoding="utf-8")
    partial_failures_path = args.output_dir / "failures.partial.jsonl"
    partial_failures = partial_failures_path.open("a", encoding="utf-8")
    work_iterator = iter(work)

    def submit_next(
        pool: ThreadPoolExecutor,
        futures: dict[Future[tuple[bytes, int, int, str]], dict[str, Any]],
    ) -> bool:
        try:
            row = next(work_iterator)
        except StopIteration:
            return False
        future = pool.submit(
            encode_image,
            row["image_id"],
            args.timeout,
            args.max_dimension,
            args.min_dimension,
        )
        futures[future] = row
        return True

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures: dict[
                Future[tuple[bytes, int, int, str]], dict[str, Any]
            ] = {}
            for _ in range(args.workers * 4):
                if not submit_next(pool, futures):
                    break
            completed = 0
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    candidate = futures.pop(future)
                    image_id = candidate["image_id"]
                    info = metadata[image_id]
                    try:
                        jpeg, width, height, digest = future.result()
                        output_path = (
                            args.output_dir
                            / "images"
                            / candidate["leaf_id"]
                            / f"{image_id}.jpg"
                        )
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(jpeg)
                        record = {
                            "schema_version": 1,
                            "sample_id": f"openimages-v7-train:{image_id}",
                            "leaf_id": candidate["leaf_id"],
                            "relative_path": relative_to_root(output_path),
                            "source": "openimages_v7_train",
                            "provider": "Google Open Images",
                            "source_name": "human-verified train image labels",
                            "source_id": image_id,
                            "source_url": info.get("OriginalLandingURL"),
                            "download_url": IMAGE_URL.format(image_id=image_id),
                            "title": info.get("Title"),
                            "creator": info.get("Author"),
                            "creator_url": info.get("AuthorProfileURL"),
                            "license": "by",
                            "license_version": "2.0",
                            "license_url": info.get("License"),
                            "attribution": (
                                f"{info.get('Title') or image_id} "
                                f"by {info.get('Author') or 'unknown'}"
                            ),
                            "source_label_ids": candidate["source_label_ids"],
                            "source_label_names": candidate["source_label_names"],
                            "sha256": digest,
                            "width": width,
                            "height": height,
                            "bytes": len(jpeg),
                            "retrieved_at": utc_now(),
                            "status": "accepted",
                            "review_status": "source_mapped",
                            "review_method": (
                                "openimages_human_verified_train_mapping_v1"
                            ),
                            "official_split": "train",
                        }
                        rows_by_id[image_id] = record
                        partial_manifest.write(
                            json.dumps(record, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                    except Exception as exc:
                        failure = {
                            "leaf_id": candidate["leaf_id"],
                            "source_id": image_id,
                            "error": f"{type(exc).__name__}: {exc}",
                            "recorded_at": utc_now(),
                        }
                        failures.append(failure)
                        partial_failures.write(
                            json.dumps(failure, ensure_ascii=False, sort_keys=True)
                            + "\n"
                        )
                    completed += 1
                    submit_next(pool, futures)
                    if completed % 250 == 0 or completed == len(work):
                        partial_manifest.flush()
                        partial_failures.flush()
                        print(f"downloaded {completed}/{len(work)}", flush=True)
    finally:
        partial_manifest.close()
        partial_failures.close()

    rows = sorted(
        rows_by_id.values(),
        key=lambda row: (str(row["leaf_id"]), str(row["source_id"])),
    )
    atomic_write_jsonl(manifest_path, rows)
    if partial_manifest_path.exists():
        partial_manifest_path.unlink()
    atomic_write_jsonl(args.output_dir / "failures.jsonl", failures)
    if partial_failures_path.exists():
        partial_failures_path.unlink()

    summary = {
        **candidate_summary,
        "accepted": len(rows),
        "download_failures": len(failures),
        "leaf_counts": dict(sorted(Counter(row["leaf_id"] for row in rows).items())),
        "source_files": [
            source_record(CLASS_URL, class_path),
            source_record(LABEL_URL, label_path),
            source_record(METADATA_URL, metadata_path),
        ],
        "license_note": (
            "Annotations are CC BY 4.0; every selected image metadata row is "
            "checked for CC BY 2.0."
        ),
        "scope_warning": (
            "Object labels are weak CashLog expense labels and do not prove "
            "purchase intent."
        ),
    }
    (args.output_dir / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
