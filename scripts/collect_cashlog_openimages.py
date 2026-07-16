#!/usr/bin/env python3
"""Collect a small, attribution-preserving Open Images visual proxy set."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import random
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAPPING = ROOT / "configs/cashlog/openimages_leaf_mapping.json"
DEFAULT_OUTPUT = ROOT / "data/raw/cashlog33/openimages_v7"
CLASS_URL = "https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv"
LABEL_URL = "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv"
METADATA_URL = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv"
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
CC_BY_20 = "https://creativecommons.org/licenses/by/2.0/"
MAX_METADATA_BYTES = 24 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_bytes(url: str, limit: int, timeout: float, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            request = Request(
                url,
                headers={"User-Agent": "CashlogDatasetBuilder/0.1 (https://github.com/UICHANLEE/catai)"},
            )
            with urlopen(request, timeout=timeout) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > limit:
                    raise ValueError(f"download exceeds limit: {declared} bytes")
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError(f"download exceeds {limit} bytes")
            return payload
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
        except (TimeoutError, URLError):
            if attempt + 1 == retries:
                raise
        time.sleep((2**attempt) + random.random() * 0.25)
    raise RuntimeError("unreachable retry state")


def cached_download(url: str, path: Path, limit: int, timeout: float) -> tuple[bytes, dict[str, Any]]:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = path.read_bytes() if path.exists() else request_bytes(url, limit, timeout)
    if not path.exists():
        path.write_bytes(payload)
    return payload, {
        "url": url,
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def parse_csv(payload: bytes) -> csv.DictReader:
    return csv.DictReader(io.StringIO(payload.decode("utf-8-sig")))


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def prepare_candidates(
    class_payload: bytes,
    label_payload: bytes,
    mapping: dict[str, Any],
    per_leaf: int,
    seed: int,
) -> tuple[dict[str, str], dict[str, list[dict[str, Any]]], dict[str, list[str]]]:
    display_by_mid = {
        str(row["LabelName"]): str(row["DisplayName"]) for row in parse_csv(class_payload)
    }
    mid_by_display = {display: mid for mid, display in display_by_mid.items()}
    configured_names = {
        name for names in mapping["leaves"].values() for name in names
    }
    missing_names = sorted(configured_names - set(mid_by_display))
    if missing_names:
        raise ValueError(f"mapping contains unknown Open Images classes: {missing_names}")
    leaf_by_mid = {
        mid_by_display[display]: leaf_id
        for leaf_id, names in mapping["leaves"].items()
        for display in names
    }

    positive_by_image: dict[str, set[str]] = defaultdict(set)
    for row in parse_csv(label_payload):
        if str(row.get("Confidence")) != "1":
            continue
        mid = str(row.get("LabelName") or "")
        if mid in leaf_by_mid:
            positive_by_image[str(row["ImageID"])].add(mid)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ambiguous: dict[str, list[str]] = {}
    for image_id, mids in positive_by_image.items():
        leaves = sorted({leaf_by_mid[mid] for mid in mids})
        if len(leaves) != 1:
            ambiguous[image_id] = leaves
            continue
        leaf_id = leaves[0]
        candidates[leaf_id].append(
            {
                "image_id": image_id,
                "mids": sorted(mids),
                "display_names": sorted(display_by_mid[mid] for mid in mids),
            }
        )
    selected = {
        leaf_id: sorted(rows, key=lambda row: stable_key(seed, row["image_id"]))[:per_leaf]
        for leaf_id, rows in candidates.items()
    }
    return display_by_mid, selected, ambiguous


def read_selected_metadata(
    metadata_payload: bytes, selected_ids: set[str]
) -> dict[str, dict[str, str]]:
    output: dict[str, dict[str, str]] = {}
    for row in parse_csv(metadata_payload):
        image_id = str(row.get("ImageID") or "")
        if image_id in selected_ids:
            output[image_id] = {key: str(value or "") for key, value in row.items()}
    return output


def encode_image(image_id: str, timeout: float, max_dimension: int) -> tuple[bytes, int, int, str]:
    payload = request_bytes(IMAGE_URL.format(image_id=image_id), MAX_IMAGE_BYTES, timeout)
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("downloaded object is not an image") from exc
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=91, optimize=True)
    jpeg = output.getvalue()
    return jpeg, image.width, image.height, hashlib.sha256(jpeg).hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-leaf", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-dimension", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=250716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir = args.output_dir / "metadata"
    class_payload, class_source = cached_download(
        CLASS_URL, metadata_dir / "class-descriptions-boxable.csv", MAX_METADATA_BYTES, args.timeout
    )
    label_payload, label_source = cached_download(
        LABEL_URL, metadata_dir / "validation-human-image-labels.csv", MAX_METADATA_BYTES, args.timeout
    )
    metadata_payload, image_source = cached_download(
        METADATA_URL, metadata_dir / "validation-images-with-rotation.csv", MAX_METADATA_BYTES, args.timeout
    )
    _, selected, ambiguous = prepare_candidates(
        class_payload, label_payload, mapping, args.per_leaf, args.seed
    )
    selected_ids = {row["image_id"] for rows in selected.values() for row in rows}
    metadata = read_selected_metadata(metadata_payload, selected_ids)

    existing_manifest = args.output_dir / "manifest.jsonl"
    existing_rows = []
    if existing_manifest.exists():
        existing_rows = [json.loads(line) for line in existing_manifest.read_text().splitlines() if line]
    existing_ids = {str(row["source_id"]) for row in existing_rows}
    manifest_rows = list(existing_rows)
    failures: list[dict[str, Any]] = []
    work = [
        (leaf_id, candidate)
        for leaf_id, candidates in selected.items()
        for candidate in candidates
        if candidate["image_id"] not in existing_ids
    ]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(encode_image, candidate["image_id"], args.timeout, args.max_dimension): (
                leaf_id,
                candidate,
            )
            for leaf_id, candidate in work
        }
        completed = 0
        for future in as_completed(futures):
            leaf_id, candidate = futures[future]
            image_id = candidate["image_id"]
            info = metadata.get(image_id)
            try:
                if not info:
                    raise ValueError("image metadata missing")
                if info.get("License") != CC_BY_20:
                    raise ValueError(f"unexpected image license: {info.get('License')}")
                jpeg, width, height, sha256 = future.result()
                output_path = args.output_dir / "images" / leaf_id / f"{image_id}.jpg"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(jpeg)
                manifest_rows.append(
                    {
                        "schema_version": 1,
                        "sample_id": f"openimages-v7-validation:{image_id}",
                        "leaf_id": leaf_id,
                        "relative_path": relative_to_root(output_path),
                        "source": "openimages_v7_validation",
                        "provider": "Google Open Images",
                        "source_name": "human-verified image labels",
                        "source_id": image_id,
                        "source_url": info.get("OriginalLandingURL"),
                        "download_url": IMAGE_URL.format(image_id=image_id),
                        "title": info.get("Title"),
                        "creator": info.get("Author"),
                        "creator_url": info.get("AuthorProfileURL"),
                        "license": "by",
                        "license_version": "2.0",
                        "license_url": info.get("License"),
                        "attribution": f"{info.get('Title') or image_id} by {info.get('Author') or 'unknown'}",
                        "source_label_ids": candidate["mids"],
                        "source_label_names": candidate["display_names"],
                        "sha256": sha256,
                        "width": width,
                        "height": height,
                        "bytes": len(jpeg),
                        "retrieved_at": utc_now(),
                        "status": "accepted",
                        "review_status": "source_mapped",
                        "review_method": "openimages_human_verified_label_mapping_v1",
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "leaf_id": leaf_id,
                        "source_id": image_id,
                        "error": f"{type(exc).__name__}: {exc}",
                        "recorded_at": utc_now(),
                    }
                )
            completed += 1
            if completed % 25 == 0 or completed == len(work):
                print(f"downloaded {completed}/{len(work)}", flush=True)

    manifest_rows.sort(key=lambda row: (str(row["leaf_id"]), str(row["source_id"])))
    with existing_manifest.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    failures_path = args.output_dir / "failures.jsonl"
    with failures_path.open("w", encoding="utf-8") as handle:
        for row in failures:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts = Counter(str(row["leaf_id"]) for row in manifest_rows)
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset_version": mapping["dataset_version"],
        "target_per_leaf": args.per_leaf,
        "covered_leaf_count": len(counts),
        "counts": {leaf_id: counts.get(leaf_id, 0) for leaf_id in mapping["leaves"]},
        "excluded_from_visual_proxy": mapping["excluded_from_visual_proxy"],
        "download_failures": len(failures),
        "ambiguous_cross_leaf_images_dropped": len(ambiguous),
        "source_files": [class_source, label_source, image_source],
        "license_note": "Annotations are CC BY 4.0; selected images are individually checked for CC BY 2.0 metadata.",
        "evaluation_note": "These are source-label proxy examples, not manually annotated CashLog expenses.",
    }
    (args.output_dir / "collection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
