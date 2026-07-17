#!/usr/bin/env python3
"""Train a Cashlog category classifier from the currently available UECFood data.

This is aligned with the Cashlog B-plan document: evaluate final expense
categories, not 256 fine-grained food names. UECFood only covers food-like
categories, so this script trains on the subset it can honestly label:
`식비` and `카페/간식`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from PIL import Image, ImageFile
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MOBILENETV4_WEIGHTS = ROOT / "models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors"
ARCH_ALIASES = {
    "mobilenetv4": "mobilenetv4_conv_small",
    "mobilenetv4_conv_small": "mobilenetv4_conv_small",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    source_label: str
    bbox: tuple[int, int, int, int] | None


def load_categories(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def read_uec_categories(dataset_root: Path) -> dict[int, str]:
    rows: dict[int, str] = {}
    with (dataset_root / "category.txt").open() as f:
        next(f)
        for line in f:
            if not line.strip():
                continue
            raw_id, name = line.rstrip("\n").split("\t", 1)
            rows[int(raw_id)] = name
    return rows


def read_bboxes(class_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    bbox_path = class_dir / "bb_info.txt"
    if not bbox_path.exists():
        return {}
    boxes: dict[str, tuple[int, int, int, int]] = {}
    with bbox_path.open() as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            image_id, x1, y1, x2, y2 = parts
            boxes[image_id] = (int(x1), int(y1), int(x2), int(y2))
    return boxes


def infer_cashlog_category(name: str, overrides: dict) -> str:
    lowered = name.lower()
    for rule in overrides.get("leaf_keyword_rules", []):
        leaf_id = rule["leaf_id"]
        for keyword in rule.get("keywords", []):
            if keyword in lowered:
                return leaf_id

    # Backward compatible support for the old two-label mapping file.
    for keyword in overrides.get("cafe_snack_keywords", []):
        if keyword in lowered:
            return "meal_cafe"
    for keyword in overrides.get("food_keywords", []):
        if keyword in lowered:
            return "meal_dining"
    return overrides.get("default_leaf_id", "meal_dining")


def crop_bbox(image: Image.Image, bbox: tuple[int, int, int, int], padding: float) -> Image.Image:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    pad_x = int(round(box_w * padding))
    pad_y = int(round(box_h * padding))
    left = max(0, x1 - pad_x)
    top = max(0, y1 - pad_y)
    right = min(width, x2 + pad_x)
    bottom = min(height, y2 + pad_y)
    if right <= left or bottom <= top:
        return image
    return image.crop((left, top, right, bottom))


def collect_samples(
    dataset_root: Path,
    label_to_index: dict[str, int],
    overrides: dict,
    max_samples_per_class: int | None,
    use_bbox: bool,
) -> list[Sample]:
    uec_categories = read_uec_categories(dataset_root)
    samples: list[Sample] = []
    for class_id, source_name in sorted(uec_categories.items()):
        cashlog_id = infer_cashlog_category(source_name, overrides)
        if cashlog_id not in label_to_index:
            continue
        class_dir = dataset_root / str(class_id)
        boxes = read_bboxes(class_dir) if use_bbox else {}
        image_paths = sorted(class_dir.glob("*.jpg"))
        if max_samples_per_class is not None:
            image_paths = image_paths[:max_samples_per_class]
        for path in image_paths:
            samples.append(
                Sample(
                    path=path,
                    label=label_to_index[cashlog_id],
                    source_label=source_name,
                    bbox=boxes.get(path.stem),
                )
            )
    return samples


def stratified_split(samples: list[Sample], val_ratio: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    by_label: dict[int, list[Sample]] = {}
    for sample in samples:
        by_label.setdefault(sample.label, []).append(sample)
    rng = random.Random(seed)
    train: list[Sample] = []
    val: list[Sample] = []
    for label_samples in by_label.values():
        rng.shuffle(label_samples)
        val_count = max(1, int(round(len(label_samples) * val_ratio)))
        val.extend(label_samples[:val_count])
        train.extend(label_samples[val_count:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


class CashlogImageDataset(Dataset):
    def __init__(self, samples: list[Sample], transform: transforms.Compose, bbox_padding: float) -> None:
        self.samples = samples
        self.transform = transform
        self.bbox_padding = bbox_padding

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        with Image.open(sample.path) as image:
            image = image.convert("RGB")
            if sample.bbox is not None:
                image = crop_bbox(image, sample.bbox, self.bbox_padding)
            return self.transform(image), sample.label


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.70, 1.0), ratio=(0.80, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3)),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return train_tf, val_tf


def choose_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_arch(name: str) -> str:
    if name not in ARCH_ALIASES:
        choices = ", ".join(sorted(ARCH_ALIASES))
        raise SystemExit(f"unsupported arch: {name}. available: {choices}")
    return ARCH_ALIASES[name]


def build_model(arch: str, weights_path: Path | None, num_classes: int, pretrained: bool) -> nn.Module:
    model_name = resolve_arch(arch)
    if weights_path is not None:
        model = timm.create_model(model_name, pretrained=False, num_classes=1000)
        state_dict = load_file(str(weights_path)) if weights_path.suffix == ".safetensors" else torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state_dict, strict=True)
        model.reset_classifier(num_classes)
        return model
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    return model


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...]) -> list[float]:
    maxk = min(max(topk), output.size(1))
    _, pred = output.topk(maxk, dim=1)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    values = []
    for k in topk:
        safe_k = min(k, output.size(1))
        values.append(correct[:safe_k].reshape(-1).float().sum().item() * 100.0 / target.numel())
    return values


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    log_interval: int,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_top1 = 0.0
    total_top3 = 0.0
    total_seen = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = labels.size(0)
        top1, top3 = accuracy(outputs.detach(), labels, topk=(1, 3))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top3 += top3 * batch_size
        total_seen += batch_size
        if train and log_interval and step % log_interval == 0:
            print(f"  step {step:04d}/{len(loader)} loss={loss.item():.4f} top1={top1:.2f}", flush=True)

    return {
        "loss": total_loss / total_seen,
        "top1": total_top1 / total_seen,
        "top3": total_top3 / total_seen,
    }


def class_weights(samples: list[Sample], num_classes: int) -> torch.Tensor:
    counts = [0 for _ in range(num_classes)]
    for sample in samples:
        counts[sample.label] += 1
    total = sum(counts)
    return torch.tensor([total / max(1, count) for count in counts], dtype=torch.float32)


def balanced_sampler(samples: list[Sample], num_classes: int) -> WeightedRandomSampler:
    weights = class_weights(samples, num_classes)
    sample_weights = [float(weights[sample.label]) for sample in samples]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def remap_samples(samples: list[Sample], index_remap: dict[int, int]) -> list[Sample]:
    return [
        Sample(
            path=sample.path,
            label=index_remap[sample.label],
            source_label=sample.source_label,
            bbox=sample.bbox,
        )
        for sample in samples
        if sample.label in index_remap
    ]


def maybe_start_mlflow_run(args: argparse.Namespace):
    if args.disable_mlflow:
        return None
    if not args.mlflow_tracking_uri:
        return None

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError("MLflow logging requested, but mlflow is not installed.") from exc

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    run_name = args.mlflow_run_name or f"cashlog-category-{int(time.time())}"
    return mlflow.start_run(run_name=run_name)


def log_mlflow_artifacts(output_dir: Path) -> None:
    try:
        import mlflow
    except ImportError:
        return

    for name in ["best.pt", "last.pt", "labels.json", "metrics.csv", "config.json"]:
        path = output_dir / name
        if path.exists():
            mlflow.log_artifact(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/processed/classification/uecfood256/UECFOOD256")
    parser.add_argument("--arch", default="mobilenetv4_conv_small", choices=sorted(ARCH_ALIASES))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--pretrained", action="store_true", help="Ask timm to load pretrained weights. Requires network/cache when weights are not supplied.")
    parser.add_argument("--categories", type=Path, default=ROOT / "configs/cashlog/categories.json")
    parser.add_argument("--overrides", type=Path, default=ROOT / "configs/cashlog/uecfood_category_overrides.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/cashlog_category_uecfood_mobilenetv4")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bbox-padding", type=float, default=0.10)
    parser.add_argument("--max-samples-per-uec-class", type=int)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--trainable-leaf-ids", nargs="*", help="Cashlog leaf ids to train. Defaults to leaves with available samples.")
    parser.add_argument("--min-samples-per-leaf", type=int, default=2)
    parser.add_argument("--mlflow-tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI"))
    parser.add_argument("--mlflow-experiment", default=os.getenv("MLFLOW_EXPERIMENT_NAME", "catai-cashlog-category"))
    parser.add_argument("--mlflow-run-name", default=os.getenv("MLFLOW_RUN_NAME"))
    parser.add_argument("--disable-mlflow", action="store_true")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    resolved_arch = resolve_arch(args.arch)
    if args.weights is None and resolved_arch == "mobilenetv4_conv_small" and DEFAULT_MOBILENETV4_WEIGHTS.exists():
        args.weights = DEFAULT_MOBILENETV4_WEIGHTS

    all_categories = load_categories(args.categories)
    all_label_to_index = {category["id"]: i for i, category in enumerate(all_categories)}
    overrides = json.loads(args.overrides.read_text())

    all_samples = collect_samples(
        args.dataset_root,
        label_to_index=all_label_to_index,
        overrides=overrides,
        max_samples_per_class=args.max_samples_per_uec_class,
        use_bbox=True,
    )
    all_class_counts: dict[int, int] = {}
    for sample in all_samples:
        all_class_counts[sample.label] = all_class_counts.get(sample.label, 0) + 1

    if args.trainable_leaf_ids:
        selected_ids = args.trainable_leaf_ids
    else:
        selected_ids = [
            category["id"]
            for index, category in enumerate(all_categories)
            if all_class_counts.get(index, 0) >= args.min_samples_per_leaf
        ]
    selected_ids = [leaf_id for leaf_id in selected_ids if leaf_id in all_label_to_index]
    if len(selected_ids) < 2:
        raise SystemExit(
            "at least two trainable leaf labels are required; "
            f"available={[(all_categories[i]['id'], count) for i, count in sorted(all_class_counts.items())]}"
        )

    trainable_categories = [category for category in all_categories if category["id"] in set(selected_ids)]
    index_remap = {all_label_to_index[category["id"]]: i for i, category in enumerate(trainable_categories)}
    samples = remap_samples(all_samples, index_remap)
    train_samples, val_samples = stratified_split(samples, args.val_ratio, args.seed)

    train_tf, val_tf = make_transforms(args.image_size)
    sampler = None if args.no_balanced_sampler else balanced_sampler(train_samples, len(trainable_categories))
    train_loader = DataLoader(
        CashlogImageDataset(train_samples, train_tf, args.bbox_padding),
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        CashlogImageDataset(val_samples, val_tf, args.bbox_padding),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = choose_device(args.device)
    model = build_model(resolved_arch, args.weights, len(trainable_categories), args.pretrained).to(device)
    loss_weights = class_weights(train_samples, len(trainable_categories)).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    start_epoch = 1
    best_top1 = -1.0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_categories = checkpoint.get("categories", [])
        checkpoint_ids = [category["id"] for category in checkpoint_categories]
        trainable_ids = [category["id"] for category in trainable_categories]
        if checkpoint_ids and checkpoint_ids != trainable_ids:
            raise SystemExit(f"resume labels do not match current labels: {checkpoint_ids} != {trainable_ids}")
        checkpoint_arch = checkpoint.get("arch") or checkpoint.get("model_arch") or "mobilenetv4_conv_small"
        if resolve_arch(checkpoint_arch) != resolved_arch:
            raise SystemExit(f"resume arch does not match current arch: {checkpoint_arch} != {resolved_arch}")
        model.load_state_dict(checkpoint["model"], strict=True)
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        metrics = checkpoint.get("metrics", {})
        best_top1 = float(metrics.get("best_top1", metrics.get("val_top1", -1.0)))

    (args.output_dir / "labels.json").write_text(json.dumps(trainable_categories, ensure_ascii=False, indent=2))
    config = vars(args).copy()
    for key in ["dataset_root", "weights", "categories", "overrides", "output_dir"]:
        config[key] = str(config[key]) if config[key] is not None else None
    config["resume"] = str(config["resume"]) if config["resume"] else None
    config["resolved_arch"] = resolved_arch
    config["trainable_leaf_ids"] = [category["id"] for category in trainable_categories]
    (args.output_dir / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2))

    class_counts: dict[int, int] = {}
    for sample in samples:
        class_counts[sample.label] = class_counts.get(sample.label, 0) + 1
    class_counts_by_id = {
        trainable_categories[index]["id"]: count for index, count in sorted(class_counts.items())
    }
    print(f"device={device}", flush=True)
    weights_label = str(args.weights) if args.weights else ("timm-pretrained" if args.pretrained else "random-init")
    print(f"arch={resolved_arch} weights={weights_label}", flush=True)
    print(f"cashlog_leaf_ids={[c['id'] for c in trainable_categories]}", flush=True)
    print(f"cashlog_categories={[c['display_name'] for c in trainable_categories]}", flush=True)
    print(f"train={len(train_samples)} val={len(val_samples)} counts={class_counts_by_id}", flush=True)
    print("document_target=Top-1>=70 Top-3>=90 on Cashlog leaf category level", flush=True)

    metrics_path = args.output_dir / "metrics.csv"
    mlflow_run = maybe_start_mlflow_run(args)
    if mlflow_run is not None:
        import mlflow

        mlflow.log_params(
            {
                "model": resolved_arch,
                "weights": str(args.weights) if args.weights else None,
                "pretrained": args.pretrained,
                "dataset_root": str(args.dataset_root),
                "num_classes": len(trainable_categories),
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "label_smoothing": args.label_smoothing,
                "image_size": args.image_size,
                "val_ratio": args.val_ratio,
                "balanced_sampler": not args.no_balanced_sampler,
                "trainable_leaf_ids": ",".join(category["id"] for category in trainable_categories),
            }
        )
        mlflow.log_dict(class_counts_by_id, "class_counts.json")

    append_metrics = args.resume is not None and metrics_path.exists()
    with metrics_path.open("a" if append_metrics else "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_top1", "train_top3", "val_loss", "val_top1", "val_top3", "best_top1", "seconds"],
        )
        if not append_metrics:
            writer.writeheader()
        try:
            for epoch in range(start_epoch, args.epochs + 1):
                start = time.time()
                print(f"\nepoch {epoch}/{args.epochs}", flush=True)
                train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.log_interval)
                val_metrics = run_epoch(model, val_loader, criterion, None, device, 0)
                scheduler.step()
                elapsed = time.time() - start
                improved = val_metrics["top1"] >= best_top1
                best_top1 = max(best_top1, val_metrics["top1"])
                row = {
                    "epoch": epoch,
                    "train_loss": f"{train_metrics['loss']:.6f}",
                    "train_top1": f"{train_metrics['top1']:.4f}",
                    "train_top3": f"{train_metrics['top3']:.4f}",
                    "val_loss": f"{val_metrics['loss']:.6f}",
                    "val_top1": f"{val_metrics['top1']:.4f}",
                    "val_top3": f"{val_metrics['top3']:.4f}",
                    "best_top1": f"{best_top1:.4f}",
                    "seconds": f"{elapsed:.2f}",
                }
                writer.writerow(row)
                f.flush()
                payload = {
                    "epoch": epoch,
                    "arch": resolved_arch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "categories": trainable_categories,
                    "metrics": row,
                }
                torch.save(payload, args.output_dir / "last.pt")
                if improved:
                    torch.save(payload, args.output_dir / "best.pt")
                if mlflow_run is not None:
                    mlflow.log_metrics(
                        {
                            "train_loss": train_metrics["loss"],
                            "train_top1": train_metrics["top1"],
                            "train_top3": train_metrics["top3"],
                            "val_loss": val_metrics["loss"],
                            "val_top1": val_metrics["top1"],
                            "val_top3": val_metrics["top3"],
                            "best_top1": best_top1,
                            "epoch_seconds": elapsed,
                        },
                        step=epoch,
                    )
                print(
                    f"  train top1={train_metrics['top1']:.2f} top3={train_metrics['top3']:.2f} "
                    f"val top1={val_metrics['top1']:.2f} top3={val_metrics['top3']:.2f} "
                    f"best={best_top1:.2f} seconds={elapsed:.1f}",
                    flush=True,
                )
        finally:
            if mlflow_run is not None:
                log_mlflow_artifacts(args.output_dir)
                mlflow.end_run()


if __name__ == "__main__":
    main()
