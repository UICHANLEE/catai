#!/usr/bin/env python3
"""Generate a large Korean OCR and 33-leaf receipt/document training dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_TEXT_MANIFEST = ROOT / "data/processed/cashlog33/text/all_v1/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/ocr_category/v1"
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
]
SPLITS = ("train", "validation", "test")
DOCUMENT_TITLES = (
    "결제 영수증",
    "거래 명세서",
    "이용 내역",
    "승인 내역",
    "전자 영수증",
    "납부 확인서",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def find_font(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else FONT_CANDIDATES
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError("Korean font not found; pass --font")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def wrap_text(text: str, width: int) -> list[str]:
    normalized = " ".join(text.split())
    if len(normalized) <= width:
        return [normalized]
    words = normalized.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.extend(word[offset : offset + width] for offset in range(0, len(word), width))
            continue
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines[:3]


def transform_points(points: list[list[float]], matrix: np.ndarray) -> list[list[int]]:
    source = np.asarray(points, dtype=np.float32).reshape(1, -1, 2)
    transformed = cv2.perspectiveTransform(source, matrix)[0]
    return [[int(round(x)), int(round(y))] for x, y in transformed]


def build_lines(
    text: str,
    display_name: str,
    leaf_id: str,
    rng: random.Random,
) -> list[str]:
    amount = rng.randrange(10, 5000) * 100
    date = f"202{rng.randrange(3, 7)}-{rng.randrange(1, 13):02d}-{rng.randrange(1, 29):02d}"
    lines = [
        rng.choice(DOCUMENT_TITLES),
        f"거래일시 {date} {rng.randrange(0, 24):02d}:{rng.randrange(0, 60):02d}",
        f"승인번호 {rng.randrange(100000, 999999)}",
    ]
    if leaf_id == "misc_uncat":
        lines.extend(["결제 승인", f"합계 {amount:,}원", "이용해 주셔서 감사합니다"])
        return lines
    lines.extend(wrap_text(text, rng.randrange(18, 27)))
    if rng.random() < 0.12:
        lines.append(f"분류 메모 {display_name}")
    lines.extend(
        [
            f"공급가액 {amount:,}원",
            f"부가세 {amount // 10:,}원",
            f"결제금액 {amount + amount // 10:,}원",
            rng.choice(("신용카드 결제", "체크카드 결제", "계좌이체 완료", "승인 완료")),
        ]
    )
    return lines[:11]


def draw_document(
    lines: list[str],
    rng: random.Random,
    font_path: Path,
    width: int,
    height: int,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    paper = rng.randrange(238, 256)
    image = Image.new("RGB", (width, height), (paper, paper, max(232, paper - rng.randrange(0, 6))))
    draw = ImageDraw.Draw(image)
    margin_x = rng.randrange(28, 54)
    y = rng.randrange(28, 52)
    annotations: list[dict[str, Any]] = []
    for line_index, line in enumerate(lines):
        font_size = rng.randrange(20, 29)
        if line_index == 0:
            font_size = rng.randrange(28, 35)
        font = ImageFont.truetype(str(font_path), font_size)
        box = draw.textbbox((0, 0), line, font=font)
        text_width = box[2] - box[0]
        text_height = box[3] - box[1]
        if text_width > width - margin_x * 2:
            scale = max(14, int(font_size * (width - margin_x * 2) / max(text_width, 1)))
            font = ImageFont.truetype(str(font_path), scale)
            box = draw.textbbox((0, 0), line, font=font)
            text_width = box[2] - box[0]
            text_height = box[3] - box[1]
        if y + text_height + 12 >= height - 28:
            break
        x = margin_x + rng.randrange(-4, 9)
        if line_index == 0:
            x = max(margin_x, (width - text_width) // 2)
        ink = rng.randrange(10, 55)
        draw.text((x, y), line, font=font, fill=(ink, ink, ink))
        pad = 3
        annotations.append(
            {
                "text": line,
                "points": [
                    [x - pad, y - pad],
                    [x + text_width + pad, y - pad],
                    [x + text_width + pad, y + text_height + pad],
                    [x - pad, y + text_height + pad],
                ],
            }
        )
        y += text_height + rng.randrange(15, 28)
        if rng.random() < 0.08:
            draw.line((margin_x, y, width - margin_x, y), fill=(120, 120, 120), width=1)
            y += rng.randrange(8, 16)
    for _ in range(rng.randrange(4, 15)):
        x = rng.randrange(width)
        shade = rng.randrange(210, 246)
        draw.line((x, 0, x, height), fill=(shade, shade, shade), width=1)
    return image, annotations


def distort_document(
    image: Image.Image,
    annotations: list[dict[str, Any]],
    rng: random.Random,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    array = np.asarray(image)
    height, width = array.shape[:2]
    margin = max(4, int(min(width, height) * rng.uniform(0.0, 0.035)))
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    destination = np.float32(
        [
            [rng.randrange(0, margin + 1), rng.randrange(0, margin + 1)],
            [width - 1 - rng.randrange(0, margin + 1), rng.randrange(0, margin + 1)],
            [width - 1 - rng.randrange(0, margin + 1), height - 1 - rng.randrange(0, margin + 1)],
            [rng.randrange(0, margin + 1), height - 1 - rng.randrange(0, margin + 1)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        array,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(232, 232, 228),
    )
    transformed = [
        {"text": row["text"], "points": transform_points(row["points"], matrix)}
        for row in annotations
    ]
    output = Image.fromarray(warped)
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.72, 1.22))
    output = ImageEnhance.Brightness(output).enhance(rng.uniform(0.80, 1.12))
    if rng.random() < 0.35:
        output = output.filter(ImageFilter.GaussianBlur(rng.uniform(0.25, 1.15)))
    if rng.random() < 0.20:
        noise = np.random.default_rng(rng.randrange(2**32)).normal(
            0, rng.uniform(1.0, 4.5), np.asarray(output).shape
        )
        noisy = np.clip(np.asarray(output, dtype=np.float32) + noise, 0, 255).astype(np.uint8)
        output = Image.fromarray(noisy)
    return output, transformed


def render_one(task: dict[str, Any]) -> tuple[str, str, str]:
    leaf_id = str(task["leaf_id"])
    split = str(task["split"])
    index = int(task["index"])
    rng = random.Random(f"{task['seed']}:{split}:{leaf_id}:{index}")
    width = rng.choice((448, 480, 512, 544))
    height = rng.choice((640, 704, 768))
    lines = build_lines(str(task["text"]), str(task["display_name"]), leaf_id, rng)
    image, annotations = draw_document(lines, rng, Path(task["font_path"]), width, height)
    image, annotations = distort_document(image, annotations, rng)
    output_path = Path(task["output_root"]) / "images" / split / leaf_id / f"{index:05d}.jpg"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quality = rng.randrange(68, 92)
    image.save(output_path, format="JPEG", quality=quality, optimize=True)
    payload = output_path.read_bytes()
    sample_id = f"cashlog-ocr-synth-v1:{split}:{leaf_id}:{index:05d}"
    relative_path = relative_to_root(output_path)
    manifest = {
        "schema_version": 1,
        "sample_id": sample_id,
        "leaf_id": leaf_id,
        "relative_path": relative_path,
        "source": "cashlog_ocr_category_synthetic_v1",
        "source_text_sample_id": task["source_sample_id"],
        "source_text_group_key": task["source_group_key"],
        "source_text_split": split,
        "license": "project-generated",
        "provenance_type": "synthetic_internal_training",
        "label_strength": "deterministic_weak",
        "split": split,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "width": image.width,
        "height": image.height,
        "bytes": len(payload),
        "line_count": len(annotations),
    }
    ocr = {
        "schema_version": 1,
        "sample_id": sample_id,
        "relative_path": relative_path,
        "split": split,
        "leaf_id": leaf_id,
        "transcription": "\n".join(row["text"] for row in annotations),
        "lines": annotations,
    }
    paddle = [
        {"transcription": row["text"], "points": row["points"]}
        for row in annotations
    ]
    return (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True),
        json.dumps(ocr, ensure_ascii=False, sort_keys=True),
        f"{relative_path}\t{json.dumps(paddle, ensure_ascii=False, separators=(',', ':'))}",
    )


def task_stream(
    categories: list[dict[str, Any]],
    text_by_split_leaf: dict[tuple[str, str], list[dict[str, Any]]],
    counts: dict[str, int],
    output_root: Path,
    font_path: Path,
    seed: int,
) -> Iterable[dict[str, Any]]:
    for split in SPLITS:
        for category in categories:
            leaf_id = str(category["id"])
            source_rows = text_by_split_leaf[(split, leaf_id)]
            for index in range(counts[split]):
                source = source_rows[index % len(source_rows)]
                yield {
                    "leaf_id": leaf_id,
                    "display_name": category["display_name"],
                    "split": split,
                    "index": index,
                    "text": source["text"],
                    "source_sample_id": source["sample_id"],
                    "source_group_key": source["group_key"],
                    "font_path": str(font_path),
                    "output_root": str(output_root),
                    "seed": seed,
                }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--text-manifest", type=Path, default=DEFAULT_TEXT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--train-per-leaf", type=int, default=3200)
    parser.add_argument("--validation-per-leaf", type=int, default=100)
    parser.add_argument("--test-per-leaf", type=int, default=100)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--seed", type=int, default=250716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    counts = {
        "train": args.train_per_leaf,
        "validation": args.validation_per_leaf,
        "test": args.test_per_leaf,
    }
    if any(value < 0 for value in counts.values()) or counts["train"] == 0:
        raise SystemExit("split counts must be non-negative and train-per-leaf must be positive")
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    if len(categories) != 33 or len({row["id"] for row in categories}) != 33:
        raise SystemExit("categories must contain exactly 33 unique leaves")
    text_rows = load_jsonl(args.text_manifest)
    text_by_split_leaf: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in text_rows:
        split = str(row["split"])
        if split == "val":
            split = "validation"
        if split in SPLITS:
            text_by_split_leaf[(split, str(row["leaf_id"]))].append(row)
    missing = [
        f"{split}:{category['id']}"
        for split in SPLITS
        for category in categories
        if not text_by_split_leaf[(split, str(category["id"]))]
    ]
    if missing:
        raise SystemExit(f"text manifest lacks required split/leaf rows: {missing[:10]}")
    font_path = find_font(args.font)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    ocr_path = args.output_dir / "ocr_annotations.jsonl"
    paddle_paths = {
        split: args.output_dir / f"paddle_det_{split}.txt"
        for split in SPLITS
    }
    handles = {
        split: path.open("w", encoding="utf-8")
        for split, path in paddle_paths.items()
    }
    total = 33 * sum(counts.values())
    completed = 0
    bytes_total = 0
    line_counts: Counter[str] = Counter()
    tasks = task_stream(
        categories,
        text_by_split_leaf,
        counts,
        args.output_dir,
        font_path,
        args.seed,
    )
    try:
        try:
            pool = ProcessPoolExecutor(max_workers=args.workers)
        except (OSError, PermissionError):
            print("process executor unavailable; using thread executor", flush=True)
            pool = ThreadPoolExecutor(max_workers=args.workers)
        with manifest_path.open("w", encoding="utf-8") as manifest_handle, ocr_path.open(
            "w", encoding="utf-8"
        ) as ocr_handle, pool:
            for manifest_line, ocr_line, paddle_line in pool.map(
                render_one, tasks, chunksize=16
            ):
                manifest_handle.write(manifest_line + "\n")
                ocr_handle.write(ocr_line + "\n")
                manifest = json.loads(manifest_line)
                split = str(manifest["split"])
                handles[split].write(paddle_line + "\n")
                completed += 1
                bytes_total += int(manifest["bytes"])
                line_counts[split] += int(manifest["line_count"])
                if completed % 1000 == 0 or completed == total:
                    print(
                        f"generated={completed}/{total} size_gib={bytes_total / 2**30:.2f}",
                        flush=True,
                    )
    finally:
        for handle in handles.values():
            handle.close()
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "dataset_version": "cashlog33-ocr-category-v1",
        "samples": completed,
        "leaf_count": 33,
        "counts_per_leaf": counts,
        "split_counts": {split: count * 33 for split, count in counts.items()},
        "ocr_line_counts": dict(line_counts),
        "bytes": bytes_total,
        "gib": bytes_total / 2**30,
        "font": str(font_path),
        "seed": args.seed,
        "text_manifest": relative_to_root(args.text_manifest),
        "text_manifest_sha256": hashlib.sha256(args.text_manifest.read_bytes()).hexdigest(),
        "manifest": relative_to_root(manifest_path),
        "ocr_annotations": relative_to_root(ocr_path),
        "scope_warning": (
            "Synthetic OCR/category images are training data, not real-photo accuracy evidence."
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
