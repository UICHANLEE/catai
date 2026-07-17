from __future__ import annotations

import hashlib
import json
import math
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any

import joblib
import cv2
import numpy as np
import torch
from PIL import Image
from rapidocr import LangRec, ModelType, OCRVersion, RapidOCR
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor

from .cashlog_classifier import CategoryPrediction, _read_image, choose_device


class CashlogHybridClassifier:
    """Fuse SigLIP2 vision, Korean OCR text, and an auditable OCR lexicon."""

    def __init__(self, config_path: str | Path, device: str = "auto") -> None:
        self.config_path = Path(config_path).resolve()
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.device = choose_device(device)
        self.categories = self._load_json_path("categories")
        self.semantics = self._load_json_path("semantics")
        self.ocr_lexicon = self._load_json_path("ocr_lexicon")
        self.leaf_ids = [str(row["id"]) for row in self.categories]
        self.category_by_id = {str(row["id"]): row for row in self.categories}
        self._validate_contract()

        self.vision_model_path = self._resolve_path(self.config["vision_model"])
        self.vision_head_path = self._resolve_path(self.config["vision_head"])
        self.text_model_path = self._resolve_path(self.config["text_model"])
        self.ocr_detector_model_path = self._resolve_path(self.config["ocr_detector_model"])
        self.ocr_classifier_model_path = self._resolve_path(
            self.config["ocr_classifier_model"]
        )
        self.ocr_model_path = self._resolve_path(self.config["ocr_model"])
        self._verify_file(self.vision_model_path / "model.safetensors", "vision_model_sha256")
        self._verify_file(self.vision_head_path, "vision_head_sha256")
        self._verify_file(self.text_model_path, "text_model_sha256")
        self._verify_file(self.ocr_detector_model_path, "ocr_detector_model_sha256")
        self._verify_file(self.ocr_classifier_model_path, "ocr_classifier_model_sha256")
        self._verify_file(self.ocr_model_path, "ocr_model_sha256")

        self.vision_model = AutoModel.from_pretrained(
            self.vision_model_path, local_files_only=True
        ).to(self.device).eval()
        self.vision_processor = AutoProcessor.from_pretrained(
            self.vision_model_path, local_files_only=True, use_fast=True
        )
        self.prompts, self.prompt_indexes = self._build_prompts()
        self.text_features = self._encode_prompt_features()
        vision_head_artifact = joblib.load(self.vision_head_path)
        self.vision_head = vision_head_artifact["model"]
        self.vision_head_classes = [str(value) for value in vision_head_artifact["classes"]]
        if vision_head_artifact["vision_model_sha256"] != self.config["vision_model_sha256"]:
            raise ValueError("vision head was trained with a different SigLIP2 checkpoint")
        if not set(self.vision_head_classes).issubset(set(self.leaf_ids)):
            raise ValueError("vision head contains labels outside the 33-leaf taxonomy")

        text_artifact = joblib.load(self.text_model_path)
        self.text_model = text_artifact["model"]
        self.text_temperature = float(text_artifact["temperature"])
        self.text_classes = [str(value) for value in text_artifact["classes"]]
        if set(self.text_classes) != set(self.leaf_ids):
            raise ValueError("text model classes do not match the 33-leaf taxonomy")

        self.ocr = RapidOCR(
            params={
                "Det.model_path": str(self.ocr_detector_model_path),
                "Cls.model_path": str(self.ocr_classifier_model_path),
                "Rec.lang_type": LangRec.KOREAN,
                "Rec.ocr_version": OCRVersion.PPOCRV5,
                "Rec.model_type": ModelType.MOBILE,
                "Rec.model_path": str(self.ocr_model_path),
            }
        )

    @classmethod
    def from_config(cls, path: str | Path, device: str = "auto") -> "CashlogHybridClassifier":
        return cls(path, device=device)

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.config_path.parent / path).resolve()

    def _load_json_path(self, key: str) -> Any:
        return json.loads(self._resolve_path(self.config[key]).read_text(encoding="utf-8"))

    def _verify_file(self, path: Path, sha_key: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"hybrid model dependency not found: {path}")
        expected = str(self.config[sha_key]).lower()
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"SHA-256 mismatch for {path}: expected {expected}, got {actual}")

    def _validate_contract(self) -> None:
        if len(self.leaf_ids) != 33 or len(set(self.leaf_ids)) != 33:
            raise ValueError("hybrid classifier requires exactly 33 unique leaf ids")
        expected = set(self.leaf_ids)
        for name, values in [
            ("semantics", self.semantics["leaves"]),
            ("ocr_lexicon", self.ocr_lexicon["leaves"]),
        ]:
            if set(values) != expected:
                raise ValueError(f"{name} does not match the 33-leaf taxonomy")
        if self.config["taxonomy_version"] != self.semantics["taxonomy_version"]:
            raise ValueError("taxonomy versions do not match")

    def _build_prompts(self) -> tuple[list[str], dict[str, list[int]]]:
        prompts: list[str] = []
        indexes: dict[str, list[int]] = defaultdict(list)
        for category in self.categories:
            leaf_id = str(category["id"])
            leaf = self.semantics["leaves"][leaf_id]
            values = list(leaf["queries"])
            positive = [str(value) for value in leaf.get("positive_terms", [])]
            if positive:
                values.append(
                    f"CashLog {category['display_name']} expense: " + ", ".join(positive[:8])
                )
            for value in values:
                indexes[leaf_id].append(len(prompts))
                prompts.append(str(value))
        return prompts, indexes

    @staticmethod
    def _feature_tensor(value: Any) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        if hasattr(value, "pooler_output"):
            return value.pooler_output
        raise TypeError(f"unsupported feature output: {type(value)!r}")

    def _encode_prompt_features(self) -> torch.Tensor:
        inputs = self.vision_processor(
            text=self.prompts,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            features = self._feature_tensor(self.vision_model.get_text_features(**inputs))
        return F.normalize(features.float(), dim=-1)

    @staticmethod
    def _normalize(values: dict[str, float], leaf_ids: list[str]) -> dict[str, float]:
        total = sum(max(0.0, float(values.get(leaf_id, 0.0))) for leaf_id in leaf_ids)
        if total <= 0:
            uniform = 1.0 / len(leaf_ids)
            return {leaf_id: uniform for leaf_id in leaf_ids}
        return {
            leaf_id: max(0.0, float(values.get(leaf_id, 0.0))) / total for leaf_id in leaf_ids
        }

    def _vision_scores(self, image: Image.Image) -> dict[str, float]:
        inputs = self.vision_processor(images=[image], return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            image_features = self._feature_tensor(self.vision_model.get_image_features(**inputs))
            image_features = F.normalize(image_features.float(), dim=-1)
            logits = image_features @ self.text_features.T
            logits = (
                logits * self.vision_model.logit_scale.exp().float()
                + self.vision_model.logit_bias.float()
            )
            prompt_scores = torch.sigmoid(logits)[0].detach().cpu()
        zero_shot_raw = {
            leaf_id: float(prompt_scores[self.prompt_indexes[leaf_id]].mean().item())
            for leaf_id in self.leaf_ids
        }
        # `misc_uncat` is a decision fallback, not a visual semantic class.
        # A receipt-looking image otherwise matches the word "receipt" in the
        # unreadable prompt even when OCR proves that it is perfectly legible.
        zero_shot_raw["misc_uncat"] = 0.0
        zero_shot_raw["misc_other"] *= 0.25
        zero_shot = self._normalize(zero_shot_raw, self.leaf_ids)
        head_probabilities = self.vision_head.predict_proba(image_features.cpu().numpy())[0]
        head = {leaf_id: 0.0 for leaf_id in self.leaf_ids}
        for index, leaf_id in enumerate(self.vision_head_classes):
            head[leaf_id] = float(head_probabilities[index])
        blend = self.config["vision_blend"]
        return self._normalize(
            {
                leaf_id: (
                    float(blend["zero_shot"]) * zero_shot[leaf_id]
                    + float(blend["linear_head"]) * head[leaf_id]
                )
                for leaf_id in self.leaf_ids
            },
            self.leaf_ids,
        )

    def _extract_ocr(self, image: Image.Image) -> dict[str, Any]:
        result = self.ocr(np.asarray(image))
        texts = list(result.txts or [])
        scores = [float(value) for value in (result.scores or [])]
        threshold = float(self.config["decision"]["ocr_min_line_score"])
        accepted = [
            (str(text).strip(), score)
            for text, score in zip(texts, scores, strict=True)
            if str(text).strip() and score >= threshold
        ]
        return {
            "text": " ".join(text for text, _ in accepted),
            "lines": [{"text": text, "confidence": score} for text, score in accepted],
            "mean_confidence": (
                sum(score for _, score in accepted) / len(accepted) if accepted else 0.0
            ),
        }

    @staticmethod
    def _temperature_scale(probabilities: np.ndarray, temperature: float) -> np.ndarray:
        logits = np.log(np.clip(probabilities, 1e-9, 1.0)) / temperature
        logits -= logits.max(axis=1, keepdims=True)
        values = np.exp(logits)
        return values / values.sum(axis=1, keepdims=True)

    def _text_scores(self, text: str) -> dict[str, float]:
        if not text.strip():
            return self._normalize({}, self.leaf_ids)
        raw = self.text_model.predict_proba([text])
        probabilities = self._temperature_scale(raw, self.text_temperature)[0]
        by_class = {
            leaf_id: float(probabilities[index])
            for index, leaf_id in enumerate(self.text_classes)
        }
        return {leaf_id: by_class[leaf_id] for leaf_id in self.leaf_ids}

    def _lexicon_scores(self, text: str) -> tuple[dict[str, float], dict[str, list[str]]]:
        normalized_text = self._canonical_text(text)
        matched: dict[str, list[str]] = {}
        raw: dict[str, float] = {}
        for leaf_id in self.leaf_ids:
            category_name = str(self.category_by_id[leaf_id]["display_name"])
            candidates = [*self.ocr_lexicon["leaves"][leaf_id], category_name]
            terms = []
            for term in candidates:
                value = str(term)
                canonical_term = self._canonical_text(value)
                if canonical_term and canonical_term in normalized_text and value not in terms:
                    terms.append(value)
            if terms:
                matched[leaf_id] = terms
                raw[leaf_id] = sum(1.0 + math.log1p(len(term)) for term in terms)
        return self._normalize(raw, self.leaf_ids), matched

    @staticmethod
    def _canonical_text(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold()
        return "".join(character for character in normalized if character.isalnum())

    def _image_quality(self, image: Image.Image) -> dict[str, Any]:
        grayscale = np.asarray(image.convert("L").resize((256, 256)), dtype=np.uint8)
        laplacian_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
        grayscale_std = float(grayscale.std())
        decision = self.config["decision"]
        low_information = (
            laplacian_variance
            < float(decision["low_information_laplacian_variance"])
            and grayscale_std < float(decision["low_information_grayscale_std"])
        )
        return {
            "laplacian_variance": laplacian_variance,
            "grayscale_std": grayscale_std,
            "low_information": low_information,
        }

    def _apply_safe_fallback(
        self,
        scores: dict[str, float],
        ocr: dict[str, Any],
        matched_terms: dict[str, list[str]],
        image_quality: dict[str, Any],
    ) -> tuple[dict[str, float], list[str]]:
        decision = self.config["decision"]
        text = str(ocr["text"])
        canonical_text = self._canonical_text(text)
        generic_receipt_terms = [
            self._canonical_text(value)
            for value in decision["generic_receipt_terms"]
        ]
        generic_term_count = sum(
            term in canonical_text for term in generic_receipt_terms if term
        )
        reasons: list[str] = []
        if not text.strip() and bool(image_quality["low_information"]):
            reasons.append("low_information_image")
        if (
            not matched_terms
            and generic_term_count >= int(decision["generic_receipt_min_terms"])
            and max(scores.values()) < float(decision["semantic_fallback_threshold"])
        ):
            reasons.append("receipt_without_semantic_evidence")
        if not reasons:
            return scores, reasons

        fallback_confidence = float(decision["fallback_confidence"])
        remaining = max(0.0, 1.0 - fallback_confidence)
        non_fallback_total = sum(
            score for leaf_id, score in scores.items() if leaf_id != "misc_uncat"
        )
        fallback_scores = {
            leaf_id: (
                fallback_confidence
                if leaf_id == "misc_uncat"
                else remaining * score / max(non_fallback_total, 1e-12)
            )
            for leaf_id, score in scores.items()
        }
        return fallback_scores, reasons

    @staticmethod
    def _ranked(scores: dict[str, float], top_k: int = 3) -> list[tuple[str, float]]:
        return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]

    def _fuse(
        self,
        vision: dict[str, float],
        text: dict[str, float],
        lexicon: dict[str, float],
        matched_terms: dict[str, list[str]],
    ) -> tuple[dict[str, float], dict[str, float]]:
        key = "with_lexicon" if matched_terms else "without_lexicon"
        weights = {
            name: float(value) for name, value in self.config["fusion"][key].items()
        }
        fused = {
            leaf_id: (
                weights["vision"] * vision[leaf_id]
                + weights["text"] * text[leaf_id]
                + weights["lexicon"] * lexicon[leaf_id]
            )
            for leaf_id in self.leaf_ids
        }
        vision_top = self._ranked(vision, 1)[0][0]
        text_top = self._ranked(text, 1)[0][0]
        if vision_top == text_top:
            fused[vision_top] += float(self.config["fusion"]["agreement_bonus"])
        return self._normalize(fused, self.leaf_ids), weights

    def infer(self, image: Image.Image | bytes | str | Path) -> dict[str, Any]:
        pil_image = _read_image(image)
        image_quality = self._image_quality(pil_image)
        vision = self._vision_scores(pil_image)
        ocr = self._extract_ocr(pil_image)
        text = self._text_scores(str(ocr["text"]))
        lexicon, matched_terms = self._lexicon_scores(str(ocr["text"]))
        fused, weights = self._fuse(vision, text, lexicon, matched_terms)
        fused, fallback_reasons = self._apply_safe_fallback(
            fused, ocr, matched_terms, image_quality
        )
        ranked = self._ranked(fused, 3)
        margin = ranked[0][1] - ranked[1][1]
        decision = self.config["decision"]
        need_user_check = (
            not bool(decision.get("allow_auto_confirm", False))
            or ranked[0][0] == "misc_uncat"
            or ranked[0][1] < float(decision["auto_confirm_threshold"])
            or margin < float(decision["minimum_margin"])
        )
        return {
            "ranked": ranked,
            "margin": margin,
            "need_user_check": need_user_check,
            "vision_scores": vision,
            "text_scores": text,
            "lexicon_scores": lexicon,
            "matched_terms": matched_terms,
            "ocr": ocr,
            "image_quality": image_quality,
            "fallback_reasons": fallback_reasons,
            "fusion_weights": weights,
        }

    def predict(
        self, image: Image.Image | bytes | str | Path, top_k: int = 3, **_: Any
    ) -> list[CategoryPrediction]:
        result = self.infer(image)
        predictions: list[CategoryPrediction] = []
        for leaf_id, confidence in result["ranked"][:top_k]:
            category = self.category_by_id[leaf_id]
            predictions.append(
                CategoryPrediction(
                    model_id=leaf_id,
                    display_name=str(category["display_name"]),
                    cashlog_leaf_id=leaf_id,
                    confidence=float(confidence),
                )
            )
        return predictions

    def analyze(
        self,
        image: Image.Image | bytes | str | Path,
        low_confidence_threshold: float | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        result = self.infer(image)
        ranked = result["ranked"]
        best_id, best_score = ranked[0]
        if low_confidence_threshold is not None:
            result["need_user_check"] = (
                result["need_user_check"] or best_score < low_confidence_threshold
            )
        best_category = self.category_by_id[best_id]
        top_categories = [
            {"category": leaf_id, "confidence": float(score)} for leaf_id, score in ranked
        ]
        response = {
            "success": True,
            "recommended_category": best_id,
            "confidence": float(best_score),
            "reason": (
                f"시각·한국어 OCR 앙상블이 '{best_category['display_name']}'로 추천했습니다."
            ),
            "items": [
                {
                    "name": "cashlog33_hybrid",
                    "display_name": str(best_category["display_name"]),
                    "category": best_id,
                    "confidence": float(best_score),
                    "top_categories": top_categories,
                }
            ],
            "top_categories": top_categories,
            "need_user_check": bool(result["need_user_check"]),
            "model": str(self.config["model_version"]),
            "taxonomy_version": str(self.config["taxonomy_version"]),
            "engine": "siglip2+rapidocr+tfidf",
            "members": [
                {"name": "siglip2_vision_head", "weight": result["fusion_weights"]["vision"]},
                {"name": "rapidocr_text_sgd", "weight": result["fusion_weights"]["text"]},
                {"name": "cashlog_ocr_lexicon", "weight": result["fusion_weights"]["lexicon"]},
            ],
            "evidence": {
                "ocr": result["ocr"],
                "image_quality": result["image_quality"],
                "matched_terms": result["matched_terms"],
                "fallback_reasons": result["fallback_reasons"],
                "vision_top3": [
                    {"category": leaf_id, "confidence": float(score)}
                    for leaf_id, score in self._ranked(result["vision_scores"], 3)
                ],
                "text_top3": [
                    {"category": leaf_id, "confidence": float(score)}
                    for leaf_id, score in self._ranked(result["text_scores"], 3)
                ],
                "decision_margin": float(result["margin"]),
            },
        }
        if result["need_user_check"]:
            response["error_code"] = "LOW_CONFIDENCE"
        return response
