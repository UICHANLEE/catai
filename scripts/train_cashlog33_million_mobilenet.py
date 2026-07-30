#!/usr/bin/env python3
"""Train a compact CashLog visual classifier on an exact million-view schedule."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import timm
import torch
from timm.utils import ModelEmaV3
from PIL import Image, ImageFile
from safetensors.torch import load_file
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


ImageFile.LOAD_TRUNCATED_IMAGES = True
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data/processed/cashlog33/training/million_v1"
DEFAULT_OUTPUT = ROOT / "checkpoints/cashlog33/mobilenet_million_v1"
DEFAULT_WEIGHTS = (
    ROOT
    / "models/classification/"
    "mobilenetv4_conv_small.e2400_r224_in1k.safetensors"
)
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
ARCH = "mobilenetv4_conv_small"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def make_transforms(image_size: int) -> tuple[transforms.Compose, transforms.Compose]:
    train = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.65, 1.0),
                ratio=(0.75, 1.3333),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.20, contrast=0.20, saturation=0.15, hue=0.02
            ),
            transforms.RandAugment(num_ops=2, magnitude=7),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
            transforms.RandomErasing(
                p=0.10, scale=(0.02, 0.10), ratio=(0.3, 3.3)
            ),
        ]
    )
    evaluation = transforms.Compose(
        [
            transforms.Resize(
                round(image_size / 0.875),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
            ),
        ]
    )
    return train, evaluation


class ScheduledViewDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        row_indexes: np.ndarray,
        seeds: np.ndarray,
        label_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        if len(row_indexes) != len(seeds):
            raise ValueError("row index and seed schedule lengths differ")
        self.paths = [
            str(ROOT / str(row["relative_path"])) for row in rows
        ]
        self.labels = np.asarray(
            [
                label_to_index.get(str(row["leaf_id"]), -1)
                for row in rows
            ],
            dtype=np.int16,
        )
        self.row_indexes = row_indexes
        self.seeds = seeds
        self.transform = transform

    def __len__(self) -> int:
        return len(self.row_indexes)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row_index = int(self.row_indexes[index])
        seed = int(self.seeds[index])
        random.seed(seed)
        torch.manual_seed(seed)
        with Image.open(self.paths[row_index]) as image:
            tensor = self.transform(image.convert("RGB"))
        label = int(self.labels[row_index])
        if label < 0:
            raise RuntimeError("training schedule references an unsupported leaf")
        return tensor, label


class EvaluationDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, Any]],
        label_to_index: dict[str, int],
        transform: transforms.Compose,
    ) -> None:
        filtered_rows = [
            row for row in rows if str(row["leaf_id"]) in label_to_index
        ]
        self.paths = [
            str(ROOT / str(row["relative_path"])) for row in filtered_rows
        ]
        self.labels = np.asarray(
            [
                label_to_index[str(row["leaf_id"])]
                for row in filtered_rows
            ],
            dtype=np.int16,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(self.labels[index])


def build_model(weights: Path, num_classes: int) -> nn.Module:
    model = timm.create_model(ARCH, pretrained=False, num_classes=1000)
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.reset_classifier(num_classes)
    return model


def build_optimizer(
    model: nn.Module, backbone_lr: float, head_lr: float, weight_decay: float
) -> torch.optim.AdamW:
    classifier_ids = {id(parameter) for parameter in model.get_classifier().parameters()}
    backbone = [
        parameter for parameter in model.parameters() if id(parameter) not in classifier_ids
    ]
    head = list(model.get_classifier().parameters())
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": backbone_lr},
            {"params": head, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def negative_log_likelihood(
    logits: np.ndarray, expected: np.ndarray, temperature: float
) -> float:
    scaled = logits.astype(np.float64, copy=False) / temperature
    maximum = scaled.max(axis=1)
    log_partition = maximum + np.log(
        np.exp(scaled - maximum[:, None]).sum(axis=1)
    )
    return float(
        np.mean(log_partition - scaled[np.arange(len(expected)), expected])
    )


def expected_calibration_error(
    logits: np.ndarray,
    expected: np.ndarray,
    temperature: float,
    bins: int = 15,
) -> float:
    scaled = logits.astype(np.float64, copy=False) / temperature
    scaled -= scaled.max(axis=1, keepdims=True)
    probabilities = np.exp(scaled)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == expected
    error = 0.0
    boundaries = np.linspace(0.0, 1.0, bins + 1)
    for index in range(bins):
        included = (confidence > boundaries[index]) & (
            confidence <= boundaries[index + 1]
        )
        if included.any():
            error += float(
                included.mean()
                * abs(confidence[included].mean() - correct[included].mean())
            )
    return error


def fit_temperature(
    logits: np.ndarray, expected: np.ndarray
) -> dict[str, float]:
    coarse = np.geomspace(0.25, 16.0, num=49)
    coarse_losses = np.asarray(
        [
            negative_log_likelihood(logits, expected, float(value))
            for value in coarse
        ]
    )
    best_index = int(coarse_losses.argmin())
    lower = coarse[max(0, best_index - 1)]
    upper = coarse[min(len(coarse) - 1, best_index + 1)]
    refined = np.geomspace(lower, upper, num=41)
    refined_losses = np.asarray(
        [
            negative_log_likelihood(logits, expected, float(value))
            for value in refined
        ]
    )
    temperature = float(refined[int(refined_losses.argmin())])
    return {
        "temperature": temperature,
        "nll_uncalibrated": negative_log_likelihood(logits, expected, 1.0),
        "nll_calibrated": negative_log_likelihood(
            logits, expected, temperature
        ),
        "ece_uncalibrated": expected_calibration_error(
            logits, expected, 1.0
        ),
        "ece_calibrated": expected_calibration_error(
            logits, expected, temperature
        ),
    }


def inverse_schedule_class_weights(
    rows: list[dict[str, Any]],
    row_indexes: np.ndarray,
    label_to_index: dict[str, int],
) -> tuple[np.ndarray, dict[str, int]]:
    scheduled_labels = np.fromiter(
        (
            label_to_index[str(rows[int(row_index)]["leaf_id"])]
            for row_index in row_indexes
        ),
        dtype=np.int64,
        count=len(row_indexes),
    )
    counts = np.bincount(scheduled_labels, minlength=len(label_to_index))
    if np.any(counts == 0):
        raise ValueError("every model class must appear in the training schedule")
    weights = len(row_indexes) / (len(label_to_index) * counts.astype(np.float64))
    weights /= weights.mean()
    labels = list(label_to_index)
    return weights.astype(np.float32), {
        labels[index]: int(count) for index, count in enumerate(counts)
    }


def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, class_count: int
) -> dict[str, Any]:
    model.eval()
    expected: list[int] = []
    predicted: list[int] = []
    logit_parts: list[np.ndarray] = []
    top3_hits = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            total_loss += criterion(logits, labels).item() * labels.size(0)
            predictions = logits.argmax(dim=1)
            top3 = logits.topk(min(3, class_count), dim=1).indices
            top3_hits += int((top3 == labels[:, None]).any(dim=1).sum().item())
            expected.extend(labels.cpu().tolist())
            predicted.extend(predictions.cpu().tolist())
            logit_parts.append(logits.float().cpu().numpy())
    expected_array = np.asarray(expected, dtype=np.int64)
    predicted_array = np.asarray(predicted, dtype=np.int64)
    calibration = fit_temperature(
        np.concatenate(logit_parts, axis=0), expected_array
    )
    return {
        "samples": len(expected),
        "loss": total_loss / max(1, len(expected)),
        "top1": float(np.mean(expected_array == predicted_array)),
        "top3": top3_hits / max(1, len(expected)),
        "macro_f1": float(
            f1_score(
                expected,
                predicted,
                labels=list(range(class_count)),
                average="macro",
                zero_division=0,
            )
        ),
        "probability_calibration": calibration,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--head-lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--class-weighting",
        choices=["inverse_schedule", "none"],
        default="inverse_schedule",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--ema-decay", type=float, default=0.9997)
    parser.add_argument("--disable-ema", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--mlflow-tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5500"),
    )
    parser.add_argument("--mlflow-experiment", default="cashlog33-million-compact")
    parser.add_argument("--mlflow-run-name", default="mobilenetv4-million-v1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(1_000_033)
    random.seed(1_000_033)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.dataset_dir / "original_manifest.jsonl"
    schedule_path = args.dataset_dir / "million_view_schedule.npz"
    summary_path = args.dataset_dir / "dataset_summary.json"
    augmentation_path = args.dataset_dir / "augmentation_plan.json"
    rows = read_jsonl(manifest_path)
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    augmentation_plan = json.loads(
        augmentation_path.read_text(encoding="utf-8")
    )
    with np.load(schedule_path, allow_pickle=False) as schedule:
        row_indexes = schedule["row_indexes"]
        seeds = schedule["augmentation_seeds"]
        trainable_leaves = [str(value) for value in schedule["trainable_leaves"]]
    if len(row_indexes) % args.epochs:
        raise SystemExit("logical training views must divide evenly across epochs")

    category_rows = json.loads(args.categories.read_text(encoding="utf-8"))
    categories = [
        row for row in category_rows if str(row["id"]) in set(trainable_leaves)
    ]
    label_to_index = {
        str(category["id"]): index for index, category in enumerate(categories)
    }
    if list(label_to_index) != [
        leaf_id for leaf_id in [str(row["id"]) for row in category_rows] if leaf_id in set(trainable_leaves)
    ]:
        raise RuntimeError("category ordering mismatch")
    class_weights, scheduled_class_counts = inverse_schedule_class_weights(
        rows, row_indexes, label_to_index
    )
    if args.class_weighting == "none":
        class_weights = np.ones_like(class_weights)

    train_transform, evaluation_transform = make_transforms(args.image_size)
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    validation_loader = DataLoader(
        EvaluationDataset(validation_rows, label_to_index, evaluation_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )
    device = torch.device(args.device)
    model = build_model(args.weights, len(categories)).to(device)
    model_ema = (
        None
        if args.disable_ema
        else ModelEmaV3(
            model,
            decay=args.ema_decay,
            use_warmup=True,
            device=device,
        )
    )
    optimizer = build_optimizer(
        model, args.backbone_lr, args.head_lr, args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights).to(device),
        label_smoothing=args.label_smoothing,
    )
    start_epoch = 1
    best_macro_f1 = -1.0
    views_seen = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        if model_ema is not None and checkpoint.get("ema_model"):
            model_ema.module.load_state_dict(checkpoint["ema_model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_macro_f1 = float(checkpoint["best_macro_f1"])
        views_seen = int(checkpoint["views_seen"])

    config = {
        **vars(args),
        "dataset_manifest_sha256": sha256_file(manifest_path),
        "schedule_sha256": sha256_file(schedule_path),
        "dataset_summary_sha256": sha256_file(summary_path),
        "augmentation_plan_sha256": sha256_file(augmentation_path),
        "pretrained_weights_sha256": sha256_file(args.weights),
        "torch_version": torch.__version__,
        "timm_version": timm.__version__,
        "logical_training_views": len(row_indexes),
        "trainable_leaves": list(label_to_index),
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "labels.json").write_text(
        json.dumps(categories, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    class_weight_report = {
        leaf_id: {
            "scheduled_views": scheduled_class_counts[leaf_id],
            "loss_weight": float(class_weights[index]),
        }
        for index, leaf_id in enumerate(label_to_index)
    }
    (args.output_dir / "class_weights.json").write_text(
        json.dumps(class_weight_report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metrics_path = args.output_dir / "metrics.csv"
    korean_log_path = args.output_dir / "run_log_ko.jsonl"
    if not metrics_path.exists():
        with metrics_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(
                [
                    "epoch",
                    "views_seen",
                    "train_loss",
                    "train_top1",
                    "validation_loss",
                    "validation_top1",
                    "validation_top3",
                    "validation_macro_f1",
                    "temperature",
                    "validation_nll_calibrated",
                    "validation_ece_calibrated",
                    "seconds",
                ]
            )

    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment)
    views_per_epoch = len(row_indexes) // args.epochs
    with korean_log_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "시각": utc_now(),
                    "단계": "학습_시작",
                    "모델": ARCH,
                    "장치": args.device,
                    "고유_원본_수": dataset_summary["unique_originals"],
                    "학습_원본_수": dataset_summary["train_originals"],
                    "검증_원본_수": dataset_summary["validation_originals"],
                    "논리_학습_뷰": len(row_indexes),
                    "증강_기법": [
                        operation["name"]
                        for operation in augmentation_plan["operations"]
                    ],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    with mlflow.start_run(run_name=args.mlflow_run_name) as active_run:
        mlflow_run_path = args.output_dir / "mlflow_run.json"
        mlflow_run_path.write_text(
            json.dumps(
                {
                    "tracking_uri": args.mlflow_tracking_uri,
                    "experiment": args.mlflow_experiment,
                    "run_id": active_run.info.run_id,
                    "run_name": args.mlflow_run_name,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        mlflow.log_params(
            {
                "arch": ARCH,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "logical_training_views": len(row_indexes),
                "unique_originals": dataset_summary["unique_originals"],
                "train_originals": dataset_summary["train_originals"],
                "validation_originals": dataset_summary["validation_originals"],
                "all_eligible_train_originals_used": dataset_summary["schedule"][
                    "all_eligible_train_originals_used"
                ],
                "validation_ratio": 0.20,
                "trainable_leaf_count": len(categories),
                "class_weighting": args.class_weighting,
                "pretrained_weights_sha256": config[
                    "pretrained_weights_sha256"
                ],
                "torch_version": config["torch_version"],
                "timm_version": config["timm_version"],
            }
        )
        mlflow.log_artifact(str(summary_path), artifact_path="dataset")
        mlflow.log_artifact(str(augmentation_path), artifact_path="dataset")
        for epoch in range(start_epoch, args.epochs + 1):
            offset = (epoch - 1) * views_per_epoch
            epoch_dataset = ScheduledViewDataset(
                rows,
                row_indexes[offset : offset + views_per_epoch],
                seeds[offset : offset + views_per_epoch],
                label_to_index,
                train_transform,
            )
            train_loader = DataLoader(
                epoch_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                drop_last=False,
            )
            model.train()
            started = time.perf_counter()
            total_loss = 0.0
            total_correct = 0
            total_seen = 0
            for step, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                if model_ema is not None:
                    model_ema.update(model)
                total_loss += loss.item() * labels.size(0)
                total_correct += int((logits.argmax(dim=1) == labels).sum().item())
                total_seen += labels.size(0)
                if step % args.log_interval == 0:
                    print(
                        json.dumps(
                            {
                                "event": "train_progress",
                                "epoch": epoch,
                                "step": step,
                                "steps": len(train_loader),
                                "views_seen": views_seen + total_seen,
                                "loss": total_loss / total_seen,
                                "top1": total_correct / total_seen,
                            }
                        ),
                        flush=True,
                    )
            views_seen += total_seen
            validation_model = (
                model_ema.module if model_ema is not None else model
            )
            validation = evaluate(
                validation_model, validation_loader, device, len(categories)
            )
            seconds = time.perf_counter() - started
            train_metrics = {
                "loss": total_loss / total_seen,
                "top1": total_correct / total_seen,
            }
            improved = validation["macro_f1"] > best_macro_f1
            best_macro_f1 = max(best_macro_f1, validation["macro_f1"])
            scheduler.step()
            checkpoint = {
                "schema_version": 1,
                "arch": ARCH,
                "model": model.state_dict(),
                "ema_model": (
                    model_ema.module.state_dict()
                    if model_ema is not None
                    else None
                ),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "categories": categories,
                "epoch": epoch,
                "views_seen": views_seen,
                "best_macro_f1": best_macro_f1,
                "train": train_metrics,
                "validation": validation,
                "dataset_manifest_sha256": config["dataset_manifest_sha256"],
                "schedule_sha256": config["schedule_sha256"],
                "pretrained_weights_sha256": config[
                    "pretrained_weights_sha256"
                ],
                "class_weights": class_weight_report,
                "created_at": utc_now(),
            }
            torch.save(checkpoint, args.output_dir / "last.pt")
            if improved:
                torch.save(checkpoint, args.output_dir / "best.pt")
            event = {
                "timestamp": utc_now(),
                "event": "epoch_complete",
                "epoch": epoch,
                "views_seen": views_seen,
                "train": train_metrics,
                "validation": validation,
                "seconds": seconds,
                "best": improved,
            }
            with (args.output_dir / "training.jsonl").open(
                "a", encoding="utf-8"
            ) as handle:
                handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            with korean_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "시각": event["timestamp"],
                            "단계": "에폭_완료",
                            "에폭": epoch,
                            "누적_학습_뷰": views_seen,
                            "학습_손실": train_metrics["loss"],
                            "학습_Top1": train_metrics["top1"],
                            "검증_손실": validation["loss"],
                            "검증_Top1": validation["top1"],
                            "검증_Top3": validation["top3"],
                            "검증_Macro_F1": validation["macro_f1"],
                            "확률_보정_temperature": validation[
                                "probability_calibration"
                            ]["temperature"],
                            "검증_NLL_보정후": validation[
                                "probability_calibration"
                            ]["nll_calibrated"],
                            "검증_ECE_보정후": validation[
                                "probability_calibration"
                            ]["ece_calibrated"],
                            "소요_초": seconds,
                            "최고_모델_갱신": improved,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerow(
                    [
                        epoch,
                        views_seen,
                        train_metrics["loss"],
                        train_metrics["top1"],
                        validation["loss"],
                        validation["top1"],
                        validation["top3"],
                        validation["macro_f1"],
                        validation["probability_calibration"]["temperature"],
                        validation["probability_calibration"]["nll_calibrated"],
                        validation["probability_calibration"]["ece_calibrated"],
                        seconds,
                    ]
                )
            mlflow.log_metrics(
                {
                    "train_loss": train_metrics["loss"],
                    "train_top1": train_metrics["top1"],
                    "validation_loss": validation["loss"],
                    "validation_top1": validation["top1"],
                    "validation_top3": validation["top3"],
                    "validation_macro_f1": validation["macro_f1"],
                    "validation_temperature": validation[
                        "probability_calibration"
                    ]["temperature"],
                    "validation_nll_calibrated": validation[
                        "probability_calibration"
                    ]["nll_calibrated"],
                    "validation_ece_calibrated": validation[
                        "probability_calibration"
                    ]["ece_calibrated"],
                    "views_seen": views_seen,
                },
                step=epoch,
            )
            mlflow.log_artifact(
                str(args.output_dir / "last.pt"),
                artifact_path=f"checkpoints/epoch-{epoch}",
            )
            print(json.dumps(event, ensure_ascii=False), flush=True)
        for name in [
            "best.pt",
            "last.pt",
            "config.json",
            "labels.json",
            "metrics.csv",
            "training.jsonl",
            "run_log_ko.jsonl",
            "class_weights.json",
            "mlflow_run.json",
        ]:
            mlflow.log_artifact(str(args.output_dir / name))


if __name__ == "__main__":
    main()
