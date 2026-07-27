#!/usr/bin/env python3
"""Collect a bounded, attributed product-image sample through Product Opener APIs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/raw/cashlog33/open_products_api"
USER_AGENT = "CashlogDatasetBuilder/0.2 (https://github.com/UICHANLEE/catai)"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
MAX_IMAGE_BYTES = 12 * 1024 * 1024
SOURCES = (
    {
        "id": "openfoodfacts-grocery",
        "base_url": "https://world.openfoodfacts.org",
        "leaf_id": "meal_grocery",
        "category": "en:snacks",
        "license": "cc-by-sa-3.0",
    },
    {
        "id": "openfoodfacts-beverages",
        "base_url": "https://world.openfoodfacts.org",
        "leaf_id": "meal_drink",
        "category": "en:beverages",
        "license": "cc-by-sa-3.0",
    },
    {
        "id": "openbeautyfacts",
        "base_url": "https://world.openbeautyfacts.org",
        "leaf_id": "fashion_beauty",
        "category": None,
        "license": "cc-by-sa-3.0",
    },
    {
        "id": "openpetfoodfacts",
        "base_url": "https://world.openpetfoodfacts.org",
        "leaf_id": "family_pet",
        "category": None,
        "license": "cc-by-sa-3.0",
    },
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def request_bytes(url: str, limit: int, timeout: float, retries: int = 4) -> bytes:
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json,image/*"})
            with urlopen(request, timeout=timeout) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > limit:
                    raise ValueError(f"response exceeds byte limit: {declared}")
                payload = response.read(limit + 1)
            if len(payload) > limit:
                raise ValueError(f"response exceeds byte limit: {limit}")
            return payload
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
        except (TimeoutError, URLError):
            if attempt + 1 == retries:
                raise
        time.sleep((2**attempt) + random.random() * 0.25)
    raise RuntimeError("unreachable retry state")


def search_products(source: dict[str, Any], count: int, timeout: float) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    page = 1
    while len(output) < count:
        params = {
            "page": page,
            "page_size": min(100, count - len(output)),
            "fields": "code,product_name,categories_tags,image_front_url,brands",
            "sort_by": "unique_scans_n",
        }
        if source["category"]:
            params["categories_tags"] = source["category"]
        url = f"{source['base_url']}/api/v2/search?{urlencode(params)}"
        payload = json.loads(request_bytes(url, MAX_RESPONSE_BYTES, timeout))
        products = payload.get("products") or []
        if not products:
            break
        output.extend(
            product
            for product in products
            if product.get("code") and product.get("image_front_url")
        )
        page += 1
        if len(products) < int(params["page_size"]):
            break
        time.sleep(6.2)
    return output[:count]


def encode_image(url: str, timeout: float, max_dimension: int) -> tuple[bytes, int, int]:
    payload = request_bytes(url, MAX_IMAGE_BYTES, timeout)
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise ValueError("downloaded product object is not an image") from exc
    if max(image.size) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=90, optimize=True)
    return output.getvalue(), image.width, image.height


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-source", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--max-dimension", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.per_source <= 250:
        raise SystemExit("--per-source must be between 1 and 250; use official bulk exports above that")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    seen_codes: set[tuple[str, str]] = set()
    for source in SOURCES:
        try:
            products = search_products(source, args.per_source, args.timeout)
        except Exception as exc:
            failures.append(
                {
                    "source": source["id"],
                    "source_id": None,
                    "stage": "search_api",
                    "error": f"{type(exc).__name__}: {exc}",
                    "fallback": "official Product Opener export plus AWS Open Data",
                }
            )
            continue
        for product in products:
            code = str(product["code"])
            key = (str(source["id"]), code)
            if key in seen_codes:
                continue
            seen_codes.add(key)
            try:
                image_url = str(product["image_front_url"])
                jpeg, width, height = encode_image(
                    image_url, args.timeout, args.max_dimension
                )
                sha256 = hashlib.sha256(jpeg).hexdigest()
                output_path = (
                    args.output_dir
                    / "images"
                    / str(source["leaf_id"])
                    / f"{source['id']}-{code}.jpg"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_bytes(jpeg)
                manifest_rows.append(
                    {
                        "schema_version": 1,
                        "sample_id": f"{source['id']}:{code}",
                        "leaf_id": source["leaf_id"],
                        "relative_path": relative_to_root(output_path),
                        "source": source["id"],
                        "provider": "Open Food Facts Product Opener",
                        "source_id": code,
                        "source_url": f"{source['base_url']}/product/{code}",
                        "download_url": image_url,
                        "product_name": product.get("product_name"),
                        "brands": product.get("brands"),
                        "source_categories": product.get("categories_tags") or [],
                        "license": source["license"],
                        "license_url": "https://creativecommons.org/licenses/by-sa/3.0/",
                        "provenance_type": "api_product_image",
                        "status": "accepted",
                        "review_status": "product_type_weak",
                        "label_strength": "weak",
                        "split": "train",
                        "sha256": sha256,
                        "width": width,
                        "height": height,
                        "bytes": len(jpeg),
                        "retrieved_at": utc_now(),
                    }
                )
                if len(manifest_rows) % 25 == 0:
                    print(
                        f"accepted={len(manifest_rows)} source={source['id']}",
                        flush=True,
                    )
            except Exception as exc:
                failures.append(
                    {
                        "source": source["id"],
                        "source_id": code,
                        "stage": "image_download",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    manifest_rows.sort(key=lambda row: (str(row["leaf_id"]), str(row["sample_id"])))
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in manifest_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "samples": len(manifest_rows),
        "leaf_counts": dict(Counter(str(row["leaf_id"]) for row in manifest_rows)),
        "source_counts": dict(Counter(str(row["source"]) for row in manifest_rows)),
        "failure_count": len(failures),
        "failures": failures,
        "manifest": relative_to_root(manifest_path),
        "usage_policy": (
            "This bounded API collector is for smoke-scale acquisition. Use official "
            "Product Opener exports and AWS Open Data for larger downloads."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
