#!/usr/bin/env python3
"""Collect the CC-BY CORD receipt OCR dataset through the Hugging Face Hub API."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from huggingface_hub import hf_hub_download
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/raw/cashlog33/cord_v2"
REPO_ID = "Heliosoph/CORD"
PINNED_REVISION = "51b7e932f2a07a883d77487b34aa5a33f76fa713"
SPLITS = ("train", "validation", "test")
FILES = {
    split: {
        "images": f"cord-{split}-images.zip",
        "annotations": f"cord-{split}.jsonl.gz",
    }
    for split in SPLITS
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def safe_member(name: str) -> str:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or len(path.parts) != 1:
        raise ValueError(f"unsafe zip member: {name}")
    return path.name


def quad_points(quad: dict[str, Any]) -> list[list[int]]:
    return [
        [int(round(float(quad[f"x{index}"]))), int(round(float(quad[f"y{index}"])))]
        for index in range(1, 5)
    ]


def flatten_lines(ground_truth: dict[str, Any]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for line in ground_truth.get("valid_line") or []:
        category = str(line.get("category") or "")
        for word in line.get("words") or []:
            text = str(word.get("text") or "").strip()
            quad = word.get("quad")
            if not text or not isinstance(quad, dict):
                continue
            try:
                points = quad_points(quad)
            except (KeyError, TypeError, ValueError):
                continue
            output.append(
                {
                    "text": text,
                    "points": points,
                    "semantic_category": category,
                }
            )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default=PINNED_REVISION)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    download_dir = args.output_dir / ".downloads"
    resolved: dict[str, dict[str, Path]] = {}
    source_files: list[dict[str, Any]] = []
    for split, names in FILES.items():
        resolved[split] = {}
        for role, filename in names.items():
            path = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    repo_type="dataset",
                    filename=filename,
                    revision=args.revision,
                    local_dir=download_dir,
                )
            )
            resolved[split][role] = path
            payload = path.read_bytes()
            source_files.append(
                {
                    "split": split,
                    "role": role,
                    "filename": filename,
                    "bytes": len(payload),
                    "sha256": sha256_bytes(payload),
                }
            )

    manifest_path = args.output_dir / "manifest.jsonl"
    ocr_path = args.output_dir / "ocr_annotations.jsonl"
    split_counts: Counter[str] = Counter()
    line_counts: Counter[str] = Counter()
    with manifest_path.open("w", encoding="utf-8") as manifest_handle, ocr_path.open(
        "w", encoding="utf-8"
    ) as ocr_handle:
        for split in SPLITS:
            annotation_rows = [
                json.loads(line)
                for line in gzip.open(
                    resolved[split]["annotations"], "rt", encoding="utf-8"
                )
                if line.strip()
            ]
            with zipfile.ZipFile(resolved[split]["images"]) as archive:
                members = {safe_member(info.filename): info for info in archive.infolist()}
                for index, row in enumerate(annotation_rows):
                    image_name = safe_member(str(row["image_file"]))
                    info = members.get(image_name)
                    if info is None:
                        raise ValueError(f"annotation image missing from zip: {image_name}")
                    payload = archive.read(info)
                    try:
                        with Image.open(io.BytesIO(payload)) as image:
                            image.load()
                            width, height = image.size
                            if image.format not in {"JPEG", "PNG"}:
                                raise ValueError(f"unexpected image format: {image.format}")
                    except (OSError, UnidentifiedImageError) as exc:
                        raise ValueError(f"invalid CORD image: {image_name}") from exc
                    output_path = args.output_dir / "images" / split / image_name
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_bytes(payload)
                    ground_truth = row.get("ground_truth") or {}
                    lines = flatten_lines(ground_truth)
                    sample_id = f"cord-v2:{split}:{index:04d}"
                    relative_path = relative_to_root(output_path)
                    manifest = {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "relative_path": relative_path,
                        "source": "cord_v2_huggingface_mirror",
                        "upstream": "naver-clova-ix/cord-v2",
                        "repository": REPO_ID,
                        "revision": args.revision,
                        "split": split,
                        "license": "cc-by-4.0",
                        "provenance_type": "real_receipt_ocr",
                        "language": "id",
                        "eligible_for_ocr_training": split == "train",
                        "eligible_for_category_training": False,
                        "sha256": sha256_bytes(payload),
                        "width": width,
                        "height": height,
                        "bytes": len(payload),
                        "line_count": len(lines),
                    }
                    ocr = {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "relative_path": relative_path,
                        "split": split,
                        "language": "id",
                        "transcription": "\n".join(item["text"] for item in lines),
                        "lines": lines,
                    }
                    manifest_handle.write(
                        json.dumps(manifest, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    ocr_handle.write(
                        json.dumps(ocr, ensure_ascii=False, sort_keys=True) + "\n"
                    )
                    split_counts[split] += 1
                    line_counts[split] += len(lines)

    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": "CORD v2",
        "repository": REPO_ID,
        "upstream": "naver-clova-ix/cord-v2",
        "revision": args.revision,
        "license": "CC-BY-4.0",
        "samples": sum(split_counts.values()),
        "split_counts": dict(split_counts),
        "ocr_line_counts": dict(line_counts),
        "source_files": source_files,
        "manifest": relative_to_root(manifest_path),
        "ocr_annotations": relative_to_root(ocr_path),
        "category_policy": (
            "CORD is OCR-only because Indonesian receipt domain labels are not CashLog truth."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
