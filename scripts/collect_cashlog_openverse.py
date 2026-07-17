#!/usr/bin/env python3
"""Collect license-traceable CashLog training images from the Openverse API."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image, ImageOps, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
API_URL = "https://api.openverse.org/v1/images/"
DEFAULT_OUTPUT_DIR = ROOT / "data/raw/cashlog33/openverse"
DEFAULT_CONFIG = ROOT / "configs/cashlog/leaf_semantics.json"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024


@dataclass(frozen=True)
class DownloadedImage:
    candidate: dict[str, Any]
    query: str
    jpeg: bytes
    sha256: str
    width: int
    height: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def request_bytes(url: str, user_agent: str, timeout: float, retries: int = 4) -> bytes:
    for attempt in range(retries):
        request = Request(
            url,
            headers={
                "Accept": "application/json,image/*;q=0.9,*/*;q=0.1",
                "User-Agent": user_agent,
            },
        )
        try:
            with urlopen(request, timeout=timeout) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > MAX_DOWNLOAD_BYTES:
                    raise ValueError(f"response too large: {declared} bytes")
                payload = response.read(MAX_DOWNLOAD_BYTES + 1)
                if len(payload) > MAX_DOWNLOAD_BYTES:
                    raise ValueError("response exceeded maximum download size")
                return payload
        except HTTPError as exc:
            if exc.code not in {429, 500, 502, 503, 504} or attempt + 1 == retries:
                raise
            retry_after = exc.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else 2**attempt
        except (TimeoutError, URLError):
            if attempt + 1 == retries:
                raise
            delay = 2**attempt
        time.sleep(delay + random.random() * 0.25)
    raise RuntimeError("unreachable retry state")


def search_openverse(
    query: str,
    page: int,
    page_size: int,
    licenses: list[str],
    user_agent: str,
    timeout: float,
) -> list[dict[str, Any]]:
    params = {
        "q": query,
        "page": page,
        "page_size": page_size,
        "license": ",".join(licenses),
        "license_type": "commercial,modification",
        "categories": "photograph",
    }
    payload = request_bytes(f"{API_URL}?{urlencode(params)}", user_agent, timeout)
    body = json.loads(payload)
    return list(body.get("results", []))


def tag_names(candidate: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for tag in candidate.get("tags") or []:
        if isinstance(tag, dict) and tag.get("name"):
            values.append(str(tag["name"]))
        elif isinstance(tag, str):
            values.append(tag)
    return values


def metadata_score(candidate: dict[str, Any], semantics: dict[str, Any]) -> int:
    haystack = " ".join(
        [str(candidate.get("title") or ""), str(candidate.get("category") or ""), *tag_names(candidate)]
    ).casefold()
    positive = sum(1 for term in semantics.get("positive_terms", []) if term.casefold() in haystack)
    negative = sum(1 for term in semantics.get("negative_terms", []) if term.casefold() in haystack)
    return positive - (2 * negative)


def encode_candidate(
    candidate: dict[str, Any],
    query: str,
    user_agent: str,
    timeout: float,
    min_dimension: int,
    max_dimension: int,
) -> DownloadedImage:
    image_url = str(candidate.get("thumbnail") or candidate.get("url") or "")
    if not image_url.startswith(("http://", "https://")):
        raise ValueError("candidate has no HTTP image URL")
    payload = request_bytes(image_url, user_agent, timeout)
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("download is not a readable image") from exc

    width, height = image.size
    if min(width, height) < min_dimension:
        raise ValueError(f"image is too small: {width}x{height}")
    if max(width, height) > max_dimension:
        image.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
    width, height = image.size
    output = io.BytesIO()
    image.save(output, format="JPEG", quality=91, optimize=True)
    jpeg = output.getvalue()
    return DownloadedImage(
        candidate=candidate,
        query=query,
        jpeg=jpeg,
        sha256=hashlib.sha256(jpeg).hexdigest(),
        width=width,
        height=height,
    )


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def build_manifest_row(leaf_id: str, item: DownloadedImage, output_path: Path) -> dict[str, Any]:
    candidate = item.candidate
    source_id = str(candidate.get("id") or "")
    return {
        "schema_version": 1,
        "sample_id": f"openverse:{source_id}",
        "leaf_id": leaf_id,
        "relative_path": relative_to_root(output_path),
        "source": "openverse",
        "provider": candidate.get("provider"),
        "source_name": candidate.get("source"),
        "source_id": source_id,
        "source_url": candidate.get("foreign_landing_url") or candidate.get("detail_url"),
        "download_url": candidate.get("thumbnail") or candidate.get("url"),
        "title": candidate.get("title"),
        "tags": tag_names(candidate),
        "creator": candidate.get("creator"),
        "creator_url": candidate.get("creator_url"),
        "license": str(candidate.get("license") or "").lower(),
        "license_version": candidate.get("license_version"),
        "license_url": candidate.get("license_url"),
        "attribution": candidate.get("attribution"),
        "query": item.query,
        "sha256": item.sha256,
        "width": item.width,
        "height": item.height,
        "bytes": len(item.jpeg),
        "retrieved_at": utc_now(),
        "status": "accepted",
        # Search metadata is only a candidate label. A model score or a human
        # review must promote this row before the dataset builder can use it.
        "review_status": "pending",
        "review_method": None,
        "reviewed_at": None,
    }


def collect_leaf(
    leaf_id: str,
    semantics: dict[str, Any],
    args: argparse.Namespace,
    manifest_path: Path,
    failures_path: Path,
    source_ids: dict[str, str],
    content_hashes: dict[str, str],
    accepted_count: int,
) -> int:
    needed = max(0, args.per_leaf - accepted_count)
    if needed == 0:
        print(f"[{leaf_id}] already has {accepted_count}/{args.per_leaf}", flush=True)
        return accepted_count

    seen_candidates: set[str] = set(source_ids)
    print(f"[{leaf_id}] collecting {needed} image(s)", flush=True)
    for query in semantics["queries"]:
        for page in range(1, args.max_pages + 1):
            if accepted_count >= args.per_leaf:
                return accepted_count
            try:
                candidates = search_openverse(
                    query=query,
                    page=page,
                    page_size=args.page_size,
                    licenses=args.licenses,
                    user_agent=args.user_agent,
                    timeout=args.timeout,
                )
            except Exception as exc:
                append_jsonl(
                    failures_path,
                    {
                        "leaf_id": leaf_id,
                        "query": query,
                        "page": page,
                        "stage": "search",
                        "error": f"{type(exc).__name__}: {exc}",
                        "recorded_at": utc_now(),
                    },
                )
                print(f"[{leaf_id}] search failed for {query!r}: {exc}", flush=True)
                continue

            ranked: list[tuple[int, dict[str, Any]]] = []
            for candidate in candidates:
                source_id = str(candidate.get("id") or "")
                license_id = str(candidate.get("license") or "").lower()
                if not source_id or source_id in seen_candidates:
                    continue
                if license_id not in args.licenses or candidate.get("mature") is True:
                    continue
                score = metadata_score(candidate, semantics)
                if score <= 0 and not args.allow_zero_metadata_score:
                    continue
                ranked.append((score, candidate))
                seen_candidates.add(source_id)
            ranked.sort(key=lambda row: (-row[0], str(row[1].get("id") or "")))

            remaining = args.per_leaf - accepted_count
            selected = [candidate for _, candidate in ranked[: max(remaining * 3, remaining)]]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {
                    pool.submit(
                        encode_candidate,
                        candidate,
                        query,
                        args.user_agent,
                        args.timeout,
                        args.min_dimension,
                        args.max_dimension,
                    ): candidate
                    for candidate in selected
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    source_id = str(candidate.get("id") or "")
                    if accepted_count >= args.per_leaf:
                        continue
                    try:
                        item = future.result()
                        prior_leaf = content_hashes.get(item.sha256)
                        if prior_leaf is not None:
                            raise ValueError(f"exact duplicate already assigned to {prior_leaf}")
                        output_path = args.output_dir / "images" / leaf_id / f"{item.sha256[:24]}.jpg"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        output_path.write_bytes(item.jpeg)
                        row = build_manifest_row(leaf_id, item, output_path)
                        append_jsonl(manifest_path, row)
                        source_ids[source_id] = leaf_id
                        content_hashes[item.sha256] = leaf_id
                        accepted_count += 1
                        print(
                            f"[{leaf_id}] {accepted_count}/{args.per_leaf} "
                            f"score={metadata_score(candidate, semantics)} title={candidate.get('title')!r}",
                            flush=True,
                        )
                    except Exception as exc:
                        append_jsonl(
                            failures_path,
                            {
                                "leaf_id": leaf_id,
                                "query": query,
                                "source_id": source_id,
                                "source_url": candidate.get("foreign_landing_url"),
                                "stage": "download",
                                "error": f"{type(exc).__name__}: {exc}",
                                "recorded_at": utc_now(),
                            },
                        )
            if candidates:
                time.sleep(args.request_delay)
    return accepted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-leaf", type=int, default=40)
    parser.add_argument("--leaf", action="append", dest="leaves")
    parser.add_argument("--page-size", type=int, default=80)
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--request-delay", type=float, default=0.35)
    parser.add_argument("--min-dimension", type=int, default=180)
    parser.add_argument("--max-dimension", type=int, default=1280)
    parser.add_argument("--licenses", nargs="+", default=["cc0", "pdm", "by"])
    parser.add_argument("--allow-zero-metadata-score", action="store_true")
    parser.add_argument(
        "--user-agent",
        default=os.environ.get(
            "OPENVERSE_USER_AGENT",
            "CashlogDatasetBuilder/0.1 (https://github.com/UICHANLEE/catai)",
        ),
    )
    args = parser.parse_args()
    if args.per_leaf < 1:
        parser.error("--per-leaf must be positive")
    if not 1 <= args.page_size <= 200:
        parser.error("--page-size must be between 1 and 200")
    if not 1 <= args.workers <= 16:
        parser.error("--workers must be between 1 and 16")
    args.licenses = [value.lower() for value in args.licenses]
    return args


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    semantics_by_leaf: dict[str, dict[str, Any]] = config["leaves"]
    categories = load_json(args.categories)
    category_ids = [row["id"] for row in categories]
    if len(category_ids) != 33 or len(set(category_ids)) != 33:
        raise SystemExit("categories.json must contain exactly 33 unique leaf ids")
    if set(category_ids) != set(semantics_by_leaf):
        missing = sorted(set(category_ids) - set(semantics_by_leaf))
        extra = sorted(set(semantics_by_leaf) - set(category_ids))
        raise SystemExit(f"semantic config mismatch: missing={missing} extra={extra}")

    selected = args.leaves or category_ids
    unknown = sorted(set(selected) - set(category_ids))
    if unknown:
        raise SystemExit(f"unknown --leaf values: {unknown}")

    args.output_dir = args.output_dir.resolve()
    manifest_path = args.output_dir / "manifest.jsonl"
    failures_path = args.output_dir / "failures.jsonl"
    existing = read_jsonl(manifest_path)
    counts = {leaf_id: 0 for leaf_id in category_ids}
    source_ids: dict[str, str] = {}
    content_hashes: dict[str, str] = {}
    for row in existing:
        if row.get("status") != "accepted":
            continue
        path = ROOT / row["relative_path"] if not Path(row["relative_path"]).is_absolute() else Path(row["relative_path"])
        if not path.exists():
            continue
        leaf_id = row["leaf_id"]
        counts[leaf_id] = counts.get(leaf_id, 0) + 1
        source_ids[str(row.get("source_id") or row.get("sample_id"))] = leaf_id
        content_hashes[str(row["sha256"])] = leaf_id

    started_at = utc_now()
    for leaf_id in selected:
        counts[leaf_id] = collect_leaf(
            leaf_id=leaf_id,
            semantics=semantics_by_leaf[leaf_id],
            args=args,
            manifest_path=manifest_path,
            failures_path=failures_path,
            source_ids=source_ids,
            content_hashes=content_hashes,
            accepted_count=counts[leaf_id],
        )

    summary = {
        "schema_version": 1,
        "started_at": started_at,
        "finished_at": utc_now(),
        "taxonomy_version": config["taxonomy_version"],
        "target_per_leaf": args.per_leaf,
        "selected_leaves": selected,
        "counts": counts,
        "complete_selected_leaves": [leaf_id for leaf_id in selected if counts[leaf_id] >= args.per_leaf],
        "incomplete_selected_leaves": [leaf_id for leaf_id in selected if counts[leaf_id] < args.per_leaf],
        "manifest": relative_to_root(manifest_path),
        "failures": relative_to_root(failures_path),
        "licenses": args.licenses,
    }
    summary_path = args.output_dir / "collection_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["incomplete_selected_leaves"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
