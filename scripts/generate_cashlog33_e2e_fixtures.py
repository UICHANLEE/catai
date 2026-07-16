#!/usr/bin/env python3
"""Generate deterministic Korean receipt fixtures for all 33 CashLog leaves."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_LEXICON = ROOT / "configs/cashlog/ocr_lexicon.json"
DEFAULT_OUTPUT = ROOT / "data/processed/cashlog33/e2e_fixtures/v1"
FONT_CANDIDATES = [
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def find_font(explicit: Path | None) -> Path:
    candidates = [explicit] if explicit else FONT_CANDIDATES
    for path in candidates:
        if path and path.exists():
            return path
    raise FileNotFoundError("Korean font not found; pass --font")


def render_receipt(
    leaf_id: str,
    display_name: str,
    alias: str,
    index: int,
    seed: int,
    font_path: Path,
) -> Image.Image:
    rng = random.Random(f"{seed}:{leaf_id}:{index}")
    width, height = 720, 960
    image = Image.new("RGB", (width, height), (248, 248, 244))
    draw = ImageDraw.Draw(image)
    title_font = ImageFont.truetype(str(font_path), 38)
    body_font = ImageFont.truetype(str(font_path), 30)
    small_font = ImageFont.truetype(str(font_path), 24)
    amount = rng.randrange(1, 500) * 100
    receipt_number = rng.randrange(100000, 999999)
    lines = [
        ("CASHLOG TEST", title_font),
        ("결제 영수증", body_font),
        (f"승인번호  {receipt_number}", small_font),
        (f"분류항목  {display_name}", body_font),
        (f"상품/상호  {alias}", body_font),
        ("--------------------------", small_font),
        (f"공급가액  {amount:,}원", body_font),
        (f"부가세    {amount // 10:,}원", body_font),
        (f"결제금액  {amount + amount // 10:,}원", title_font),
        ("신용카드 결제 완료", body_font),
        ("감사합니다", small_font),
    ]
    y = 72
    for text, font in lines:
        box = draw.textbbox((0, 0), text, font=font)
        text_width = box[2] - box[0]
        x = max(32, (width - text_width) // 2 + rng.randrange(-8, 9))
        draw.text((x, y), text, fill=(25, 25, 25), font=font)
        y += (box[3] - box[1]) + rng.randrange(24, 38)
    for _ in range(20):
        x = rng.randrange(width)
        gray = rng.randrange(210, 240)
        draw.line((x, 0, x, height), fill=(gray, gray, gray), width=1)
    image = image.rotate(rng.uniform(-1.2, 1.2), resample=Image.Resampling.BICUBIC, fillcolor=(245, 245, 240))
    if leaf_id == "misc_uncat":
        image = image.filter(ImageFilter.GaussianBlur(radius=8.0))
    elif index % 3 == 2:
        image = image.filter(ImageFilter.GaussianBlur(radius=0.6))
    return image


def relative_to_root(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--font", type=Path)
    parser.add_argument("--per-leaf", type=int, default=3)
    parser.add_argument("--seed", type=int, default=250716)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    categories = json.loads(args.categories.read_text(encoding="utf-8"))
    lexicon = json.loads(args.lexicon.read_text(encoding="utf-8"))
    leaf_ids = [str(row["id"]) for row in categories]
    if len(leaf_ids) != 33 or set(leaf_ids) != set(lexicon["leaves"]):
        raise SystemExit("fixture generator requires the same 33 leaves in categories and lexicon")
    font_path = find_font(args.font)
    rows = []
    for category in categories:
        leaf_id = str(category["id"])
        aliases = [str(value) for value in lexicon["leaves"][leaf_id]]
        for index in range(args.per_leaf):
            alias = aliases[index % len(aliases)]
            image = render_receipt(
                leaf_id,
                str(category["display_name"]),
                alias,
                index,
                args.seed,
                font_path,
            )
            output_path = args.output_dir / "images" / leaf_id / f"fixture-{index:02d}.jpg"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(output_path, format="JPEG", quality=92, optimize=True)
            payload = output_path.read_bytes()
            rows.append(
                {
                    "schema_version": 1,
                    "sample_id": f"cashlog-e2e:{leaf_id}:{index:02d}",
                    "leaf_id": leaf_id,
                    "relative_path": relative_to_root(output_path),
                    "source": "cashlog_e2e_fixture_generator_v1",
                    "license": "project-generated",
                    "provenance_type": "synthetic_internal_e2e",
                    "alias": alias,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "split": "test",
                    "generated_at": utc_now(),
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "samples": len(rows),
        "leaf_count": len(leaf_ids),
        "per_leaf": args.per_leaf,
        "font": str(font_path),
        "scope_warning": "Synthetic OCR integration fixtures are not real-world accuracy evidence.",
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
