#!/usr/bin/env python3
"""Build a reproducible 33-leaf OCR/transaction text training dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "DoDataThings/us-bank-transaction-categories-v2"
DATASET_API = f"https://huggingface.co/api/datasets/{DATASET_ID}"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_SEMANTICS = ROOT / "configs/cashlog/leaf_semantics.json"
DEFAULT_OCR_LEXICON = ROOT / "configs/cashlog/ocr_lexicon.json"
DEFAULT_RAW_DIR = ROOT / "data/raw/cashlog33/text/us-bank-transactions-v2"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/cashlog33/text/v1"
MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_bytes(url: str, timeout: float = 60.0) -> bytes:
    request = Request(
        url,
        headers={"User-Agent": "CashlogDatasetBuilder/0.1 (https://github.com/UICHANLEE/catai)"},
    )
    with urlopen(request, timeout=timeout) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > MAX_DOWNLOAD_BYTES:
            raise ValueError(f"download too large: {declared} bytes")
        payload = response.read(MAX_DOWNLOAD_BYTES + 1)
    if len(payload) > MAX_DOWNLOAD_BYTES:
        raise ValueError("download exceeded size limit")
    return payload


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_bucket(value: str) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 100


def assign_split(group_key: str) -> str:
    bucket = stable_bucket(group_key)
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def contains(text: str, *terms: str) -> bool:
    return any(term in text for term in terms)


def map_transaction(description: str, source_category: str) -> str | None:
    text = re.sub(r"\s+", " ", description.casefold())
    category = source_category.casefold()

    if category in {"income", "transfer"}:
        return None
    if category in {"rent", "mortgage"}:
        return "housing_rent"
    if category == "insurance":
        return "finance_insure"
    if category == "fees":
        return "finance_fee"
    if category == "groceries":
        if contains(text, "liquor", "wine", "beer", "beverage", "bottle shop"):
            return "meal_drink"
        return "meal_grocery"
    if category == "restaurants":
        if contains(text, "coffee", "cafe", "café", "bakery", "donut", "starbucks"):
            return "meal_cafe"
        if contains(text, "bar ", "pub ", "brewery", "liquor"):
            return "meal_drink"
        return "meal_dining"
    if category == "personal care":
        return "fashion_beauty"
    if category == "healthcare":
        if contains(text, "gym", "fitness", "yoga", "pilates"):
            return "health_gym"
        return "health_med"
    if category == "education":
        if contains(text, "book", "stationery", "office depot", "supplies"):
            return "edu_book"
        return "edu_class"
    if category == "travel":
        return "leisure_trip"
    if category == "utilities":
        if contains(text, "internet", "broadband", "xfinity", "comcast", "cable", "spectrum"):
            return "comm_internet"
        if contains(text, "mobile", "wireless", "cellular", "tmobile", "verizon", "at&t"):
            return "comm_mobile"
        return "housing_utility"
    if category == "transportation":
        if contains(text, "repair", "tire", "mechanic", "oil change", "service center"):
            return "transit_maintain"
        if contains(text, "uber", "lyft", "taxi", "parking", "fuel", "gas station", "shell "):
            return "transit_car"
        return "transit_public"
    if category == "entertainment":
        if contains(text, "cinema", "movie", "theater", "theatre", "museum", "concert", "ticket"):
            return "leisure_show"
        return "leisure_hobby"
    if category == "subscription":
        if contains(text, "internet", "broadband", "cable"):
            return "comm_internet"
        if contains(text, "fitness", "gym"):
            return "health_gym"
        return "leisure_hobby"
    if category == "shopping":
        if contains(text, "baby", "diaper", "children", "kids", "toys r us"):
            return "family_kids"
        if contains(text, "pet", "veterinary", "petsmart", "petco"):
            return "family_pet"
        if contains(text, "book", "stationery", "office depot", "staples"):
            return "edu_book"
        if contains(text, "clothing", "apparel", "fashion", "shoe", "nike", "zara", "uniqlo"):
            return "fashion_clothes"
        if contains(text, "beauty", "cosmetic", "salon", "sephora", "ulta"):
            return "fashion_beauty"
        if contains(text, "furniture", "appliance", "ikea", "best buy", "home depot"):
            return "life_appliance"
        if contains(text, "gift", "hallmark", "flowers"):
            return "gift_present"
        if contains(text, "craft", "music", "game", "hobby", "art supply"):
            return "leisure_hobby"
        return "life_goods"
    return None


def fetch_source(raw_dir: Path, offline_csv: Path | None) -> tuple[Path, dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = raw_dir / "source_metadata.json"
    if offline_csv is None and metadata_path.exists():
        cached_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        cached_name = str(cached_metadata.get("filename") or "")
        cached_path = raw_dir / cached_name
        if cached_name and cached_path.exists():
            payload = cached_path.read_bytes()
            if hashlib.sha256(payload).hexdigest() == cached_metadata.get("sha256"):
                return cached_path, cached_metadata
    if offline_csv:
        payload = offline_csv.read_bytes()
        destination = raw_dir / offline_csv.name
        if destination.resolve() != offline_csv.resolve():
            destination.write_bytes(payload)
        metadata = {
            "dataset_id": DATASET_ID,
            "revision": "offline-input",
            "license": "mit",
            "source_url": str(offline_csv.resolve()),
            "retrieved_at": utc_now(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return destination, metadata

    api_payload = request_bytes(DATASET_API)
    api_metadata = json.loads(api_payload)
    revision = str(api_metadata.get("sha") or "main")
    siblings = [str(item.get("rfilename")) for item in api_metadata.get("siblings", [])]
    csv_names = [name for name in siblings if name.lower().endswith(".csv")]
    if not csv_names:
        raise RuntimeError(f"{DATASET_ID} has no CSV file in revision {revision}")
    filename = next((name for name in csv_names if "transaction" in name.casefold()), csv_names[0])
    source_url = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{revision}/{filename}"
    payload = request_bytes(source_url)
    destination = raw_dir / Path(filename).name
    destination.write_bytes(payload)
    metadata = {
        "dataset_id": DATASET_ID,
        "revision": revision,
        "license": str(api_metadata.get("cardData", {}).get("license") or "mit").lower(),
        "source_url": source_url,
        "dataset_page": f"https://huggingface.co/datasets/{DATASET_ID}",
        "retrieved_at": utc_now(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
        "filename": filename,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return destination, metadata


def source_rows(csv_path: Path, metadata: dict[str, Any], max_per_leaf: int) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    output: list[dict[str, Any]] = []
    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if set(reader.fieldnames or []) != {"description", "category"}:
            raise ValueError(f"unexpected source columns: {reader.fieldnames}")
        for index, row in enumerate(reader):
            text = str(row["description"]).strip()
            source_category = str(row["category"]).strip()
            leaf_id = map_transaction(text, source_category)
            if not leaf_id or counts[leaf_id] >= max_per_leaf:
                continue
            record_hash = hashlib.sha256(
                f"{index}\0{text}\0{source_category}".encode()
            ).hexdigest()
            group_key = f"transaction:{record_hash}"
            output.append(
                {
                    "schema_version": 1,
                    "sample_id": f"usbankv2:{record_hash[:24]}",
                    "leaf_id": leaf_id,
                    "text": text,
                    "source": "hf_us_bank_transaction_categories_v2",
                    "source_category": source_category,
                    "source_revision": metadata["revision"],
                    "license": metadata["license"],
                    "provenance_type": "synthetic_external",
                    "label_method": "source_category_and_keyword_rule_v1",
                    "review_status": "weak_label",
                    "group_key": group_key,
                    "split": assign_split(group_key),
                }
            )
            counts[leaf_id] += 1
    return output


def synthetic_rows(
    categories: list[dict[str, Any]],
    semantics: dict[str, Any],
    ocr_lexicon: dict[str, Any],
    per_leaf: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    templates = [
        "[debit] {term} PAYMENT {amount}",
        "CARD PURCHASE {term} {code}",
        "RECEIPT {term} TOTAL {amount}",
        "{date} {term} 승인 {amount}원",
        "카드결제 {term} 합계 {amount}원",
        "영수증 {term} 결제금액 {amount}원",
        "CashLog OCR {term} {code}",
        "{term} 자동이체 {amount}원",
    ]
    output: list[dict[str, Any]] = []
    for category in categories:
        leaf_id = str(category["id"])
        leaf = semantics["leaves"][leaf_id]
        terms = list(
            dict.fromkeys(
                [
                    *[str(value) for value in leaf.get("positive_terms", [])],
                    *[str(value) for value in ocr_lexicon["leaves"].get(leaf_id, [])],
                ]
            )
        )
        if not terms:
            terms = [str(category["display_name"]), leaf_id]
        for index in range(per_leaf):
            term = terms[(index // len(templates)) % len(terms)]
            template_index = index % len(templates)
            split = "val" if template_index == 6 else "test" if template_index == 7 else "train"
            amount = rng.randrange(1, 900) * 100
            text = templates[template_index].format(
                term=term,
                amount=f"{amount:,}",
                code=f"{rng.randrange(100000, 999999)}",
                date=f"2026-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}",
            )
            group_key = f"template:{leaf_id}:{term}:{template_index}"
            sample_hash = hashlib.sha256(f"{leaf_id}\0{index}\0{text}".encode()).hexdigest()
            output.append(
                {
                    "schema_version": 1,
                    "sample_id": f"generated:{sample_hash[:24]}",
                    "leaf_id": leaf_id,
                    "text": text,
                    "source": "cashlog_template_generator_v1",
                    "source_category": leaf_id,
                    "source_revision": "13.33.1",
                    "license": "project-generated",
                    "provenance_type": "synthetic_internal",
                    "label_method": "deterministic_template_v1",
                    "review_status": "generated",
                    "group_key": group_key,
                    "split": split,
                }
            )
    return output


def ensure_split_coverage(rows: list[dict[str, Any]], leaf_ids: list[str]) -> None:
    covered = {(str(row["leaf_id"]), str(row["split"])) for row in rows}
    missing = [
        f"{leaf_id}:{split}"
        for leaf_id in leaf_ids
        for split in ["train", "val", "test"]
        if (leaf_id, split) not in covered
    ]
    if missing:
        raise RuntimeError(f"dataset lacks leaf/split coverage: {missing}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--ocr-lexicon", type=Path, default=DEFAULT_OCR_LEXICON)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--offline-csv", type=Path)
    parser.add_argument("--max-source-per-leaf", type=int, default=1200)
    parser.add_argument("--synthetic-per-leaf", type=int, default=240)
    parser.add_argument("--seed", type=int, default=250716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.synthetic_per_leaf < 8:
        raise SystemExit("--synthetic-per-leaf must be at least 8 to cover every split")
    categories = load_json(args.categories)
    semantics = load_json(args.semantics)
    ocr_lexicon = load_json(args.ocr_lexicon)
    leaf_ids = [str(row["id"]) for row in categories]
    if len(leaf_ids) != 33 or set(leaf_ids) != set(semantics["leaves"]):
        raise SystemExit("category and semantic configs must contain the same 33 leaves")
    if set(leaf_ids) != set(ocr_lexicon["leaves"]):
        raise SystemExit("OCR lexicon must contain the same 33 leaves")
    csv_path, source_metadata = fetch_source(args.raw_dir.resolve(), args.offline_csv)
    rows = source_rows(csv_path, source_metadata, args.max_source_per_leaf)
    rows.extend(
        synthetic_rows(categories, semantics, ocr_lexicon, args.synthetic_per_leaf, args.seed)
    )
    ensure_split_coverage(rows, leaf_ids)
    rows.sort(key=lambda row: (str(row["split"]), str(row["leaf_id"]), str(row["sample_id"])))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    counts: dict[str, dict[str, int]] = {
        leaf_id: {"train": 0, "val": 0, "test": 0, "total": 0} for leaf_id in leaf_ids
    }
    source_counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row["leaf_id"])][str(row["split"])] += 1
        counts[str(row["leaf_id"])]["total"] += 1
        source_counts[str(row["source"])] += 1
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "taxonomy_version": semantics["taxonomy_version"],
        "samples": len(rows),
        "leaf_count": len([leaf_id for leaf_id, values in counts.items() if values["total"]]),
        "source_counts": dict(sorted(source_counts.items())),
        "counts": counts,
        "source_metadata": source_metadata,
        "limitations": [
            "External transaction labels are synthetic and weakly mapped from broad categories.",
            "Internal templates guarantee 33-leaf coverage but cannot establish real-world accuracy.",
            "A manually labeled CashLog holdout remains required for a production promotion decision.",
        ],
    }
    (args.output_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "class_counts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["leaf_id", "train", "val", "test", "total"],
            lineterminator="\n",
        )
        writer.writeheader()
        for leaf_id in leaf_ids:
            writer.writerow({"leaf_id": leaf_id, **counts[leaf_id]})
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
