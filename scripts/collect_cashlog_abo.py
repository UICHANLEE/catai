#!/usr/bin/env python3
"""Build a license-pinned CashLog visual manifest from ABO catalog images."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
import tarfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LISTINGS = ROOT / "data/raw/classification/abo/abo-listings.tar"
DEFAULT_IMAGES = ROOT / "data/raw/classification/abo/abo-images-small.tar"
DEFAULT_MAPPING = ROOT / "configs/cashlog/abo_product_type_mapping.json"
DEFAULT_OUTPUT = ROOT / "data/raw/cashlog33/abo_v1"
LICENSE_MEMBER = "LICENSE-CC-BY-4.0.txt"
IMAGE_METADATA_MEMBER = "images/metadata/images.csv.gz"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def sha256_path(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def product_types(row: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for item in row.get("product_type") or []:
        value = item.get("value") if isinstance(item, dict) else item
        if value:
            values.add(str(value).strip().upper())
    return values


def localized_values(row: dict[str, Any], field: str) -> list[str]:
    output: list[str] = []
    for item in row.get(field) or []:
        value = item.get("value") if isinstance(item, dict) else item
        if value:
            output.append(str(value).strip())
    return output


def load_candidates(
    listing_tar: Path,
    mapping: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    leaf_by_type: dict[str, str] = {}
    duplicate_types: dict[str, list[str]] = defaultdict(list)
    for leaf_id, types in mapping["leaves"].items():
        for product_type in types:
            normalized = str(product_type).upper()
            if normalized in leaf_by_type:
                duplicate_types[normalized].extend([leaf_by_type[normalized], leaf_id])
            leaf_by_type[normalized] = str(leaf_id)
    if duplicate_types:
        raise ValueError(f"product types mapped to multiple leaves: {dict(duplicate_types)}")

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    listing_rows = 0
    with tarfile.open(listing_tar) as archive:
        members = sorted(
            (member for member in archive.getmembers() if member.name.endswith(".json.gz")),
            key=lambda member: member.name,
        )
        for member in members:
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"cannot read {member.name}")
            with gzip.GzipFile(fileobj=source) as handle:
                for line in handle:
                    listing_rows += 1
                    row = json.loads(line)
                    image_id = str(row.get("main_image_id") or "")
                    if not image_id:
                        continue
                    mapped_leaves = sorted(
                        {leaf_by_type[value] for value in product_types(row) if value in leaf_by_type}
                    )
                    if not mapped_leaves:
                        continue
                    by_image[image_id].append(
                        {
                            "image_id": image_id,
                            "leaf_ids": mapped_leaves,
                            "item_id": str(row.get("item_id") or ""),
                            "domain_name": str(row.get("domain_name") or ""),
                            "product_types": sorted(product_types(row)),
                            "item_names": localized_values(row, "item_name"),
                            "brand_names": localized_values(row, "brand"),
                        }
                    )

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    conflicting_images = 0
    duplicate_listing_images = 0
    for image_id, rows in by_image.items():
        leaves = sorted({leaf for row in rows for leaf in row["leaf_ids"]})
        if len(leaves) != 1:
            conflicting_images += 1
            continue
        if len(rows) > 1:
            duplicate_listing_images += 1
        selected = sorted(
            rows,
            key=lambda row: (row["domain_name"], row["item_id"]),
        )[0]
        selected["leaf_id"] = leaves[0]
        candidates[leaves[0]].append(selected)
    return candidates, {
        "listing_rows": listing_rows,
        "mapped_unique_images": sum(len(rows) for rows in candidates.values()),
        "conflicting_images_dropped": conflicting_images,
        "duplicate_listing_images_collapsed": duplicate_listing_images,
    }


def select_candidates(
    candidates: dict[str, list[dict[str, Any]]],
    per_leaf: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for leaf_id, rows in sorted(candidates.items()):
        ordered = sorted(rows, key=lambda row: stable_key(seed, row["image_id"]))
        selected.extend(ordered[:per_leaf])
    return selected


def image_members(archive: tarfile.TarFile, selected_ids: set[str]) -> dict[str, tarfile.TarInfo]:
    metadata_member = archive.getmember(IMAGE_METADATA_MEMBER)
    metadata_source = archive.extractfile(metadata_member)
    if metadata_source is None:
        raise RuntimeError(f"cannot read {IMAGE_METADATA_MEMBER}")
    paths: dict[str, str] = {}
    with gzip.GzipFile(fileobj=metadata_source) as compressed:
        text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        for row in csv.DictReader(text):
            image_id = str(row.get("image_id") or "")
            if image_id in selected_ids:
                paths[image_id] = str(row["path"])

    image_id_by_member = {
        f"images/small/{relative_path}": image_id for image_id, relative_path in paths.items()
    }
    matched: dict[str, tarfile.TarInfo] = {}
    for member in archive.getmembers():
        image_id = image_id_by_member.get(member.name)
        if (
            image_id is not None
            and member.isfile()
            and Path(member.name).suffix.lower() in ALLOWED_IMAGE_SUFFIXES
        ):
            matched[image_id] = member
    return matched


def encode_image(source: BinaryIO, max_dimension: int) -> tuple[bytes, int, int, str]:
    payload = source.read()
    try:
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("ABO archive member is not a valid image") from exc
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=92, optimize=True)
    encoded = output.getvalue()
    return encoded, image.width, image.height, hashlib.sha256(encoded).hexdigest()


def extract_selected(
    image_tar: Path,
    selected: list[dict[str, Any]],
    output_dir: Path,
    max_dimension: int,
    min_dimension: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_id = {row["image_id"]: row for row in selected}
    manifest: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with tarfile.open(image_tar) as archive:
        members = image_members(archive, set(by_id))
        for index, image_id in enumerate(sorted(by_id), start=1):
            candidate = by_id[image_id]
            member = members.get(image_id)
            try:
                if member is None:
                    raise FileNotFoundError("selected image id is missing from image archive")
                source = archive.extractfile(member)
                if source is None:
                    raise OSError("cannot read image archive member")
                encoded, width, height, digest = encode_image(source, max_dimension)
                if min(width, height) < min_dimension:
                    raise ValueError(
                        f"image is smaller than minimum dimension: {width}x{height}"
                    )
                destination = output_dir / "images" / candidate["leaf_id"] / f"{image_id}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encoded)
                title = candidate["item_names"][0] if candidate["item_names"] else image_id
                manifest.append(
                    {
                        "schema_version": 1,
                        "sample_id": f"abo:{image_id}",
                        "leaf_id": candidate["leaf_id"],
                        "relative_path": relative_to_root(destination),
                        "source": "amazon_berkeley_objects",
                        "provider": "Amazon.com",
                        "source_name": "ABO catalog main image",
                        "source_id": image_id,
                        "source_url": "https://amazon-berkeley-objects.s3.amazonaws.com/index.html",
                        "title": title,
                        "creator": "Amazon.com",
                        "license": "by",
                        "license_version": "4.0",
                        "license_url": "https://creativecommons.org/licenses/by/4.0/",
                        "attribution": "Amazon Berkeley Objects (c) by Amazon.com",
                        "source_product_types": candidate["product_types"],
                        "source_item_id": candidate["item_id"],
                        "source_domain_name": candidate["domain_name"],
                        "brand_names": candidate["brand_names"],
                        "sha256": digest,
                        "width": width,
                        "height": height,
                        "bytes": len(encoded),
                        "retrieved_at": utc_now(),
                        "status": "accepted",
                        "review_status": "source_mapped",
                        "review_method": "abo_product_type_mapping_v1",
                        "label_strength": "deterministic_weak",
                        "split": "train",
                        "split_lock": "train",
                    }
                )
            except Exception as exc:
                failures.append(
                    {
                        "image_id": image_id,
                        "leaf_id": candidate["leaf_id"],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            if index % 1000 == 0 or index == len(by_id):
                print(f"processed {index}/{len(by_id)}", flush=True)
    return manifest, failures


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listings-tar", type=Path, default=DEFAULT_LISTINGS)
    parser.add_argument("--images-tar", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--mapping", type=Path, default=DEFAULT_MAPPING)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-leaf", type=int, default=5000)
    parser.add_argument("--max-dimension", type=int, default=512)
    parser.add_argument("--min-dimension", type=int, default=96)
    parser.add_argument("--seed", type=int, default=250729)
    parser.add_argument("--skip-archive-hash", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mapping = json.loads(args.mapping.read_text(encoding="utf-8"))
    candidates, candidate_stats = load_candidates(args.listings_tar, mapping)
    selected = select_candidates(candidates, args.per_leaf, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(args.listings_tar) as archive:
        license_source = archive.extractfile(LICENSE_MEMBER)
        if license_source is None:
            raise RuntimeError(f"{LICENSE_MEMBER} is missing from listings archive")
        license_payload = license_source.read()
    (args.output_dir / LICENSE_MEMBER).write_bytes(license_payload)

    manifest, failures = extract_selected(
        args.images_tar,
        selected,
        args.output_dir,
        args.max_dimension,
        args.min_dimension,
    )
    manifest.sort(key=lambda row: (row["leaf_id"], row["source_id"]))
    write_jsonl(args.output_dir / "manifest.jsonl", manifest)
    write_jsonl(args.output_dir / "failures.jsonl", failures)

    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset": "Amazon Berkeley Objects",
        "mapping_version": mapping["mapping_version"],
        "license": "CC BY 4.0",
        "license_sha256": hashlib.sha256(license_payload).hexdigest(),
        "per_leaf_cap": args.per_leaf,
        "minimum_dimension": args.min_dimension,
        "selected_before_archive_validation": len(selected),
        "accepted": len(manifest),
        "failures": len(failures),
        "leaf_counts": dict(sorted(Counter(row["leaf_id"] for row in manifest).items())),
        "candidate_stats": candidate_stats,
        "listing_archive": {
            "path": str(args.listings_tar),
            "sha256": None if args.skip_archive_hash else sha256_path(args.listings_tar),
        },
        "image_archive": {
            "path": str(args.images_tar),
            "sha256": None if args.skip_archive_hash else sha256_path(args.images_tar),
        },
        "scope_warning": "ABO product types are weak CashLog labels and do not prove purchase intent.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
