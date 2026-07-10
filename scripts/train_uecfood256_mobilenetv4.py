#!/usr/bin/env python3
"""Fine-tune MobileNetV4 on the downloaded UECFood256 dataset."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import timm
import torch
from PIL import Image, ImageFile
from safetensors.torch import load_file
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Sample:
    path: Path
    label: int
    bbox: tuple[int, int, int, int] | None = None


def read_categories(dataset_root: Path) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    with (dataset_root / "category.txt").open() as f:
        next(f)
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_id, name = line.split("\t", 1)
            rows.append({"uec_id": int(raw_id), "label": int(raw_id) - 1, "name": name})
    return rows


def read_bboxes(class_dir: Path) -> dict[str, tuple[int, int, int, int]]:
    boxes: dict[str, tuple[int, int, int, int]] = {}
    bbox_path = class_dir / "bb_info.txt"
    if not bbox_path.exists():
        return boxes

    with bbox_path.open() as f:
        next(f)
        for line in f:
            parts = line.split()
            if len(parts) != 5:
                continue
            image_id, x1, y1, x2, y2 = parts
            boxes[image_id] = (int(x1), int(y1), int(x2), int(y2))
    return boxes


def collect_samples(dataset_root: Path, max_samples_per_class: int | None, use_bbox: bool) -> list[Sample]:
    samples: list[Sample] = []
    for class_dir in sorted((p for p in dataset_root.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        label = int(class_dir.name) - 1
        boxes = read_bboxes(class_dir) if use_bbox else {}
        image_paths = sorted(class_dir.glob("*.jpg"))
        if max_samples_per_class is not None:
            image_paths = image_paths[:max_samples_per_class]
        samples.extend(Sample(path=p, label=label, bbox=boxes.get(p.stem)) for p in image_paths)
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


class ImageListDataset(Dataset):
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


def make_transforms(image_size: int, randaugment: bool, random_erasing: float) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps: list[object] = [
        transforms.RandomResizedCrop(image_size, scale=(0.72, 1.0), ratio=(0.80, 1.25)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02),
    ]
    if randaugment:
        train_steps.append(transforms.RandAugment(num_ops=2, magnitude=7))
    train_steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    if random_erasing > 0:
        train_steps.append(transforms.RandomErasing(p=random_erasing, scale=(0.02, 0.12), ratio=(0.3, 3.3)))

    train_tf = transforms.Compose(
        train_steps
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


def build_model(weights_path: Path, num_classes: int) -> nn.Module:
    model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=1000)
    state = load_file(str(weights_path))
    model.load_state_dict(state, strict=True)
    model.reset_classifier(num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.parameters():
        param.requires_grad = trainable
    classifier = model.get_classifier() if hasattr(model, "get_classifier") else None
    if classifier is not None:
        for param in classifier.parameters():
            param.requires_grad = True


def mixup_batch(images: torch.Tensor, labels: torch.Tensor, alpha: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = random.betavariate(alpha, alpha)
    index = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1.0 - lam) * images[index]
    return mixed, labels, labels[index], lam


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: tuple[int, ...] = (1, 5)) -> list[float]:
    with torch.no_grad():
        maxk = max(topk)
        _, pred = output.topk(maxk, dim=1)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        return [correct[:k].reshape(-1).float().sum().item() * 100.0 / target.numel() for k in topk]


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    log_interval: int,
    mixup_alpha: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_seen = 0

    for step, (images, labels) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            if train and mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
            else:
                labels_a = labels
                labels_b = labels
                lam = 1.0
            outputs = model(images)
            loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        top1, top5 = accuracy(outputs.detach(), labels, topk=(1, 5))
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_seen += batch_size

        if train and log_interval and step % log_interval == 0:
            print(f"  step {step:04d}/{len(loader)} loss={loss.item():.4f} top1={top1:.2f}", flush=True)

    return {
        "loss": total_loss / total_seen,
        "top1": total_top1 / total_seen,
        "top5": total_top5 / total_seen,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "data/processed/classification/uecfood256/UECFOOD256")
    parser.add_argument("--weights", type=Path, default=ROOT / "models/classification/mobilenetv4_conv_small.e2400_r224_in1k.safetensors")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "checkpoints/uecfood256_mobilenetv4")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--label-smoothing", type=float, default=0.10)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-samples-per-class", type=int)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--no-bbox-crop", action="store_true")
    parser.add_argument("--bbox-padding", type=float, default=0.10)
    parser.add_argument("--no-randaugment", action="store_true")
    parser.add_argument("--random-erasing", type=float, default=0.15)
    parser.add_argument("--mixup-alpha", type=float, default=0.10)
    parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
    parser.add_argument("--target-top1", type=float, default=90.0)
    parser.add_argument("--patience-after-target", type=int, default=0)
    parser.add_argument("--min-delta", type=float, default=0.01)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    categories = read_categories(args.dataset_root)
    samples = collect_samples(args.dataset_root, args.max_samples_per_class, use_bbox=not args.no_bbox_crop)
    train_samples, val_samples = stratified_split(samples, args.val_ratio, args.seed)

    with (args.output_dir / "labels.json").open("w") as f:
        json.dump(categories, f, ensure_ascii=False, indent=2)

    train_tf, val_tf = make_transforms(args.image_size, randaugment=not args.no_randaugment, random_erasing=args.random_erasing)
    train_loader = DataLoader(
        ImageListDataset(train_samples, train_tf, bbox_padding=args.bbox_padding),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        ImageListDataset(val_samples, val_tf, bbox_padding=args.bbox_padding),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = choose_device(args.device)
    model = build_model(args.weights, num_classes=len(categories)).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    start_epoch = 1
    best_top1 = -1.0
    epochs_since_improvement = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0)) + 1
        best_top1 = float(checkpoint.get("best_top1", -1.0))
        epochs_since_improvement = int(checkpoint.get("epochs_since_improvement", 0))

    print(f"device={device}", flush=True)
    print(f"classes={len(categories)} train={len(train_samples)} val={len(val_samples)}", flush=True)
    print(
        f"bbox_crop={not args.no_bbox_crop} randaugment={not args.no_randaugment} "
        f"mixup_alpha={args.mixup_alpha} freeze_backbone_epochs={args.freeze_backbone_epochs}",
        flush=True,
    )
    print(
        f"target_top1={args.target_top1} patience_after_target={args.patience_after_target} "
        f"min_delta={args.min_delta}",
        flush=True,
    )
    print(f"output={args.output_dir}", flush=True)

    config = vars(args).copy()
    config["dataset_root"] = str(config["dataset_root"])
    config["weights"] = str(config["weights"])
    config["output_dir"] = str(config["output_dir"])
    config["resume"] = str(config["resume"]) if config["resume"] else None
    with (args.output_dir / "config.json").open("w") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if args.eval_only:
        val_metrics = run_epoch(model, val_loader, criterion, None, device, 0, 0.0)
        print(f"eval val loss={val_metrics['loss']:.4f} top1={val_metrics['top1']:.2f} top5={val_metrics['top5']:.2f}", flush=True)
        return

    metrics_path = args.output_dir / "metrics.csv"
    append_metrics = args.resume is not None and metrics_path.exists()
    with metrics_path.open("a" if append_metrics else "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_top1",
                "train_top5",
                "val_loss",
                "val_top1",
                "val_top5",
                "best_top1",
                "epochs_since_improvement",
                "seconds",
            ],
        )
        if not append_metrics:
            writer.writeheader()

        for epoch in range(start_epoch, args.epochs + 1):
            start = time.time()
            print(f"\nepoch {epoch}/{args.epochs}", flush=True)
            set_backbone_trainable(model, trainable=epoch > args.freeze_backbone_epochs)
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  trainable_params={trainable}/{total_params}", flush=True)
            train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, args.log_interval, args.mixup_alpha)
            val_metrics = run_epoch(model, val_loader, criterion, None, device, 0, 0.0)
            scheduler.step()
            elapsed = time.time() - start

            row = {
                "epoch": epoch,
                "train_loss": f"{train_metrics['loss']:.6f}",
                "train_top1": f"{train_metrics['top1']:.4f}",
                "train_top5": f"{train_metrics['top5']:.4f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_top1": f"{val_metrics['top1']:.4f}",
                "val_top5": f"{val_metrics['top5']:.4f}",
                "best_top1": "",
                "epochs_since_improvement": "",
                "seconds": f"{elapsed:.2f}",
            }

            improved = val_metrics["top1"] > best_top1 + args.min_delta
            if improved:
                best_top1 = val_metrics["top1"]
                epochs_since_improvement = 0
            else:
                epochs_since_improvement += 1

            row["best_top1"] = f"{best_top1:.4f}"
            row["epochs_since_improvement"] = str(epochs_since_improvement)
            writer.writerow(row)
            f.flush()
            print(
                "  "
                f"train loss={train_metrics['loss']:.4f} top1={train_metrics['top1']:.2f} "
                f"val loss={val_metrics['loss']:.4f} top1={val_metrics['top1']:.2f} "
                f"top5={val_metrics['top5']:.2f} best={best_top1:.2f} "
                f"no_improve={epochs_since_improvement} seconds={elapsed:.1f}",
                flush=True,
            )

            last_payload = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "categories": categories,
                "metrics": row,
                "best_top1": best_top1,
                "epochs_since_improvement": epochs_since_improvement,
            }
            torch.save(last_payload, args.output_dir / "last.pt")
            if improved:
                torch.save(last_payload, args.output_dir / "best.pt")
            if best_top1 >= args.target_top1 and epochs_since_improvement >= args.patience_after_target:
                print(
                    f"stopping: best_top1={best_top1:.2f} >= {args.target_top1:.2f} "
                    f"and no improvement for {epochs_since_improvement} epochs",
                    flush=True,
                )
                break


if __name__ == "__main__":
    main()
