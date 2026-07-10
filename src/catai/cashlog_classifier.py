from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import timm
import torch
from PIL import Image, ImageFile
from torch import nn
from torchvision import transforms


ImageFile.LOAD_TRUNCATED_IMAGES = True

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "checkpoints/cashlog_category_uecfood_mps/best.pt"
DEFAULT_LABELS = PACKAGE_ROOT / "checkpoints/cashlog_category_uecfood_mps/labels.json"

CASHLOG_LEAF_BY_MODEL_ID = {
    "food": "meal_dining",
    "cafe_snack": "meal_cafe",
}


@dataclass(frozen=True)
class CategoryPrediction:
    model_id: str
    display_name: str
    cashlog_leaf_id: str
    confidence: float


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
        return image.convert("RGB")
    if isinstance(image, bytes):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


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


class CashlogCategoryClassifier:
    """MobileNetV4 Cashlog category classifier trained from UECFood256.

    The current checkpoint covers the two Cashlog categories available from the
    downloaded food data: 식비 and 카페/간식. The API response maps them to
    Cashlog leaf categories `meal_dining` and `meal_cafe`.
    """

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
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        if not isinstance(checkpoint, dict) or "model" not in checkpoint:
            raise ValueError(f"unsupported checkpoint format: {self.checkpoint_path}")

        self.categories = _load_labels(self.labels_path, checkpoint)
        self.model = self._build_model(len(self.categories))
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.14)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def _build_model(num_classes: int) -> nn.Module:
        model = timm.create_model("mobilenetv4_conv_small", pretrained=False, num_classes=num_classes)
        return model

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

        safe_k = min(top_k, probs.numel())
        confidences, indexes = probs.topk(safe_k)
        predictions: list[CategoryPrediction] = []
        for confidence, index in zip(confidences.tolist(), indexes.tolist(), strict=True):
            category = self.categories[index]
            model_id = str(category["id"])
            predictions.append(
                CategoryPrediction(
                    model_id=model_id,
                    display_name=str(category.get("display_name") or model_id),
                    cashlog_leaf_id=CASHLOG_LEAF_BY_MODEL_ID.get(model_id, "misc_uncat"),
                    confidence=float(confidence),
                )
            )
        return predictions

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
            "reason": f"로컬 MobileNetV4 모델이 상품 사진을 '{best.display_name}' 범주로 분류했습니다.",
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
            "model": "mobilenetv4_conv_small_cashlog_uecfood",
            "engine": "catai-local",
            **({"error_code": "LOW_CONFIDENCE"} if need_user_check else {}),
        }


def analyze_base64(image_base64: str, classifier: CashlogCategoryClassifier) -> dict[str, Any]:
    return classifier.analyze(base64.b64decode(image_base64))
