#!/usr/bin/env python3
"""Download model weights and public datasets for the mobile product classifier."""

from __future__ import annotations

import argparse
import subprocess
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Asset:
    slug: str
    name: str
    url: str
    output: Path
    group: str
    size_hint: str


ASSETS = [
    Asset(
        "yolo11n-seg",
        "YOLO11n-seg pretrained weights",
        "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt",
        ROOT / "models/detection/yolo11n-seg.pt",
        "models",
        "small",
    ),
    Asset(
        "mobilenetv4-small-weights",
        "MobileNetV4 Conv Small safetensors",
        "https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/model.safetensors",
        ROOT / "models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors",
        "models",
        "small",
    ),
    Asset(
        "mobilenetv4-small-config",
        "MobileNetV4 Conv Small config",
        "https://huggingface.co/timm/mobilenetv4_conv_small.e2400_r224_in1k/resolve/main/config.json",
        ROOT / "models/classification/mobilenetv4_conv_small.e2400_r224_in1k.config.json",
        "models",
        "small",
    ),
    Asset(
        "food101",
        "Food-101",
        "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
        ROOT / "data/raw/classification/food101/food-101.tar.gz",
        "public-datasets",
        "5 GB",
    ),
    Asset(
        "abo-listings",
        "ABO product listings",
        "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-listings.tar",
        ROOT / "data/raw/classification/abo/abo-listings.tar",
        "public-datasets",
        "83 MB",
    ),
    Asset(
        "abo-images-small",
        "ABO downscaled catalog images",
        "https://amazon-berkeley-objects.s3.amazonaws.com/archives/abo-images-small.tar",
        ROOT / "data/raw/classification/abo/abo-images-small.tar",
        "public-datasets",
        "3 GB",
    ),
    Asset(
        "sku110k",
        "SKU110K",
        "https://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz",
        ROOT / "data/raw/detection/sku110k/SKU110K_fixed.tar.gz",
        "public-datasets",
        "large",
    ),
    Asset(
        "uecfood256",
        "UECFood256",
        "http://foodcam.mobi/dataset256.zip",
        ROOT / "data/raw/classification/uecfood256/dataset256.zip",
        "best-effort",
        "unknown",
    ),
]


def run(asset: Asset) -> None:
    asset.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "curl",
        "-fL",
        "--continue-at",
        "-",
        "--retry",
        "3",
        "--retry-delay",
        "5",
        "--output",
        str(asset.output),
        asset.url,
    ]
    print(f"\n==> {asset.name} ({asset.size_hint})")
    print(f"    {asset.output.relative_to(ROOT)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "selection",
        choices=["models", "public-datasets", "best-effort", "all-public", "asset"],
        help="Which asset group to download.",
    )
    parser.add_argument("--slug", help="Asset slug to download when selection is 'asset'.")
    args = parser.parse_args()

    if args.selection == "asset" and not args.slug:
        parser.error("--slug is required when selection is 'asset'")

    for asset in ASSETS:
        if args.selection == "asset":
            selected = asset.slug == args.slug
        elif args.selection == "all-public":
            selected = asset.group in {"models", "public-datasets", "best-effort"}
        else:
            selected = asset.group == args.selection
        if selected:
            run(asset)


if __name__ == "__main__":
    main()
