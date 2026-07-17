from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageFile, ImageOps
from torch import nn

try:
    from pillow_heif import register_heif_opener

    register_heif_opener()
except ImportError:
    pass


ImageFile.LOAD_TRUNCATED_IMAGES = True

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = Path(__file__).resolve().parent / "assets/cashlog_category_uecfood_mps"
DEFAULT_CHECKPOINT = ASSET_ROOT / "best.pt"
DEFAULT_LABELS = ASSET_ROOT / "labels.json"
DEFAULT_ENSEMBLE_CONFIG = ASSET_ROOT / "ensemble.json"

ARCH_ALIASES = {
    "mobilenetv4": "mobilenetv4_conv_small",
    "mobilenetv4_conv_small": "mobilenetv4_conv_small",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
}

LEGACY_CASHLOG_LEAF_BY_MODEL_ID = {
    "food": "meal_dining",
    "cafe_snack": "meal_cafe",
}
CASHLOG_LEAF_BY_MODEL_ID = LEGACY_CASHLOG_LEAF_BY_MODEL_ID


@dataclass(frozen=True)
class CategoryPrediction:
    model_id: str
    display_name: str
    cashlog_leaf_id: str
    confidence: float


@dataclass(frozen=True)
class EnsembleMember:
    checkpoint_path: Path
    labels_path: Path
    arch: str
    weight: float


def choose_device(name: str = "auto") -> torch.device:
    if name != "auto":
        if name == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS device was requested, but torch.backends.mps.is_available() is false.")
        if name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device was requested, but torch.cuda.is_available() is false.")
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _read_image(image: Image.Image | bytes | str | Path) -> Image.Image:
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    if isinstance(image, bytes):
        with Image.open(io.BytesIO(image)) as source:
            source.load()
            return ImageOps.exif_transpose(source).convert("RGB")
    with Image.open(image) as source:
        source.load()
        return ImageOps.exif_transpose(source).convert("RGB")


def crop_bbox(image: Image.Image, bbox: tuple[int, int, int, int], padding: float = 0.10) -> Image.Image:
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


def _load_labels(labels_path: Path, checkpoint: dict[str, Any]) -> list[dict[str, Any]]:
    categories = checkpoint.get("categories")
    if isinstance(categories, list) and categories:
        return categories
    return json.loads(labels_path.read_text())


def resolve_arch(name: str | None) -> str:
    if not name:
        return "mobilenetv4_conv_small"
    if name not in ARCH_ALIASES:
        choices = ", ".join(sorted(ARCH_ALIASES))
        raise ValueError(f"unsupported model arch: {name}. available: {choices}")
    return ARCH_ALIASES[name]


def build_timm_classifier(arch: str, num_classes: int) -> nn.Module:
    import timm

    return timm.create_model(resolve_arch(arch), pretrained=False, num_classes=num_classes)


def make_inference_transform(image_size: int) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


def prediction_from_probs(
    categories: list[dict[str, Any]],
    probs: torch.Tensor,
    top_k: int,
) -> list[CategoryPrediction]:
    safe_k = min(top_k, probs.numel())
    confidences, indexes = probs.topk(safe_k)
    predictions: list[CategoryPrediction] = []
    for confidence, index in zip(confidences.tolist(), indexes.tolist(), strict=True):
        category = categories[index]
        model_id = str(category["id"])
        cashlog_leaf_id = (
            model_id
            if "_" in model_id
            else LEGACY_CASHLOG_LEAF_BY_MODEL_ID.get(model_id, "misc_uncat")
        )
        predictions.append(
            CategoryPrediction(
                model_id=model_id,
                display_name=str(category.get("display_name") or model_id),
                cashlog_leaf_id=cashlog_leaf_id,
                confidence=float(confidence),
            )
        )
    return predictions


class CashlogCategoryClassifier:
    """Single-backbone Cashlog leaf-category classifier trained from product images."""

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        device: str = "auto",
        image_size: int = 224,
    ) -> None:
        checkpoint_env = os.getenv("CATAI_CASHLOG_CHECKPOINT")
        labels_env = os.getenv("CATAI_CASHLOG_LABELS")
        self.checkpoint_path = Path(checkpoint_path or checkpoint_env or DEFAULT_CHECKPOINT)
        self.labels_path = Path(labels_path or labels_env or DEFAULT_LABELS)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint not found: {self.checkpoint_path}")

        self.device = choose_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(f"unsupported checkpoint format: {self.checkpoint_path}")

        self.categories = _load_labels(self.labels_path, checkpoint)
        self.arch = resolve_arch(checkpoint.get("arch") or checkpoint.get("model_arch"))
        self.model = build_timm_classifier(self.arch, len(self.categories))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = make_inference_transform(image_size)

    @staticmethod
    def default_checkpoint_path() -> Path:
        return Path(os.getenv("CATAI_CASHLOG_CHECKPOINT") or DEFAULT_CHECKPOINT)

    @staticmethod
    def default_checkpoint_exists() -> bool:
        return CashlogCategoryClassifier.default_checkpoint_path().exists()

    @classmethod
    def from_env(cls) -> "CashlogCategoryClassifier":
        return cls(
            checkpoint_path=os.getenv("CATAI_CASHLOG_CHECKPOINT"),
            labels_path=os.getenv("CATAI_CASHLOG_LABELS"),
            device=os.getenv("CATAI_DEVICE", "auto"),
        )

    def predict(
        self,
        image: Image.Image | bytes | str | Path,
        top_k: int = 3,
        bbox: tuple[int, int, int, int] | None = None,
        bbox_padding: float = 0.10,
    ) -> list[CategoryPrediction]:
        pil_image = _read_image(image)
        if bbox is not None:
            pil_image = crop_bbox(pil_image, bbox, bbox_padding)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0].detach().cpu()

        return prediction_from_probs(self.categories, probs, top_k)

    def analyze(
        self,
        image: Image.Image | bytes | str | Path,
        low_confidence_threshold: float = 0.65,
        bbox: tuple[int, int, int, int] | None = None,
        bbox_padding: float = 0.10,
    ) -> dict[str, Any]:
        predictions = self.predict(image, bbox=bbox, bbox_padding=bbox_padding)
        best = predictions[0]
        top_categories = [
            {"category": prediction.cashlog_leaf_id, "confidence": prediction.confidence}
            for prediction in predictions
        ]
        need_user_check = best.confidence < low_confidence_threshold or best.cashlog_leaf_id == "misc_uncat"
        return {
            "success": True,
            "recommended_category": best.cashlog_leaf_id,
            "confidence": best.confidence,
            "reason": f"로컬 {self.arch} 모델이 상품 사진을 '{best.display_name}' 범주로 분류했습니다.",
            "items": [
                {
                    "name": best.model_id,
                    "display_name": best.display_name,
                    "category": best.cashlog_leaf_id,
                    "confidence": best.confidence,
                    "top_categories": top_categories,
                }
            ],
            "need_user_check": need_user_check,
            "model": f"{self.arch}_cashlog_leaf",
            "engine": "catai-local",
            **({"error_code": "LOW_CONFIDENCE"} if need_user_check else {}),
        }


class CashlogEnsembleClassifier:
    """Weighted-logit ensemble exposed through the same analyze/predict API."""

    def __init__(
        self,
        ensemble_config: str | Path,
        device: str = "auto",
        image_size: int = 224,
    ) -> None:
        self.config_path = Path(ensemble_config)
        manifest = json.loads(self.config_path.read_text())
        self.device = choose_device(device)
        self.image_size = int(manifest.get("image_size", image_size))
        self.transform = make_inference_transform(self.image_size)
        self.members = self._load_members(manifest)
        if not self.members:
            raise ValueError(f"ensemble has no members: {self.config_path}")
        self.categories = self.members[0]["categories"]
        self.label_ids = [str(category["id"]) for category in self.categories]

    @classmethod
    def from_env(cls) -> "CashlogEnsembleClassifier":
        config = os.getenv("CATAI_CASHLOG_ENSEMBLE_CONFIG") or DEFAULT_ENSEMBLE_CONFIG
        return cls(config, device=os.getenv("CATAI_DEVICE", "auto"))

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (self.config_path.parent / path).resolve()

    def _load_members(self, manifest: dict[str, Any]) -> list[dict[str, Any]]:
        members = []
        for index, raw_member in enumerate(manifest.get("members", []), start=1):
            checkpoint_path = self._resolve_path(raw_member["checkpoint"])
            labels_path = self._resolve_path(raw_member.get("labels", DEFAULT_LABELS))
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            categories = _load_labels(labels_path, checkpoint)
            arch = resolve_arch(
                raw_member.get("arch")
                or checkpoint.get("arch")
                or checkpoint.get("model_arch")
            )
            model = build_timm_classifier(arch, len(categories))
            model.load_state_dict(checkpoint["model"], strict=True)
            model.to(self.device)
            model.eval()
            member_label_ids = [str(category["id"]) for category in categories]
            if members and member_label_ids != [
                str(category["id"]) for category in members[0]["categories"]
            ]:
                raise ValueError(
                    f"ensemble member {index} label order does not match the first member"
                )
            members.append(
                {
                    "checkpoint_path": checkpoint_path,
                    "categories": categories,
                    "arch": arch,
                    "weight": float(raw_member.get("weight", 1.0)),
                    "model": model,
                }
            )
        return members

    def predict(
        self,
        image: Image.Image | bytes | str | Path,
        top_k: int = 3,
        bbox: tuple[int, int, int, int] | None = None,
        bbox_padding: float = 0.10,
    ) -> list[CategoryPrediction]:
        pil_image = _read_image(image)
        if bbox is not None:
            pil_image = crop_bbox(pil_image, bbox, bbox_padding)
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        total_weight = sum(max(0.0, member["weight"]) for member in self.members)
        if total_weight <= 0:
            raise ValueError("ensemble weights must sum to a positive value")
        with torch.inference_mode():
            combined_logits = None
            for member in self.members:
                logits = member["model"](tensor) * (member["weight"] / total_weight)
                combined_logits = logits if combined_logits is None else combined_logits + logits
            probs = torch.softmax(combined_logits, dim=1)[0].detach().cpu()
        return prediction_from_probs(self.categories, probs, top_k)

    def analyze(
        self,
        image: Image.Image | bytes | str | Path,
        low_confidence_threshold: float = 0.65,
        bbox: tuple[int, int, int, int] | None = None,
        bbox_padding: float = 0.10,
    ) -> dict[str, Any]:
        predictions = self.predict(image, bbox=bbox, bbox_padding=bbox_padding)
        best = predictions[0]
        top_categories = [
            {"category": prediction.cashlog_leaf_id, "confidence": prediction.confidence}
            for prediction in predictions
        ]
        need_user_check = (
            best.confidence < low_confidence_threshold
            or best.cashlog_leaf_id == "misc_uncat"
        )
        arches = [member["arch"] for member in self.members]
        return {
            "success": True,
            "recommended_category": best.cashlog_leaf_id,
            "confidence": best.confidence,
            "reason": f"로컬 앙상블 모델({', '.join(arches)})이 상품 사진을 '{best.display_name}' 범주로 분류했습니다.",
            "items": [
                {
                    "name": "ensemble",
                    "display_name": best.display_name,
                    "category": best.cashlog_leaf_id,
                    "confidence": best.confidence,
                    "top_categories": top_categories,
                }
            ],
            "need_user_check": need_user_check,
            "model": "cashlog_leaf_weighted_logit_ensemble",
            "engine": "catai-local",
            "members": [
                {
                    "arch": member["arch"],
                    "weight": member["weight"],
                    "checkpoint": str(member["checkpoint_path"]),
                }
                for member in self.members
            ],
            **({"error_code": "LOW_CONFIDENCE"} if need_user_check else {}),
        }


def load_cashlog_classifier_from_env() -> Any:
    hybrid_config = os.getenv("CATAI_CASHLOG_HYBRID_CONFIG")
    if hybrid_config:
        from .cashlog_hybrid_classifier import CashlogHybridClassifier

        return CashlogHybridClassifier.from_config(
            hybrid_config, device=os.getenv("CATAI_DEVICE", "auto")
        )
    ensemble_config = os.getenv("CATAI_CASHLOG_ENSEMBLE_CONFIG")
    if ensemble_config:
        return CashlogEnsembleClassifier.from_env()
    return CashlogCategoryClassifier.from_env()


def analyze_base64(image_base64: str, classifier: Any) -> dict[str, Any]:
    return classifier.analyze(base64.b64decode(image_base64))
