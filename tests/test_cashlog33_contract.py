from __future__ import annotations

import hashlib
import json
import os
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

from catai.cashlog_api import validate_image, verify_internal_api_key
from catai.cashlog_hybrid_classifier import CashlogHybridClassifier


ROOT = Path(__file__).resolve().parents[1]


class Cashlog33ContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.categories = json.loads(
            (ROOT / "configs/cashlog/categories.json").read_text(encoding="utf-8")
        )

    def test_taxonomy_configs_contain_the_same_33_leaves(self) -> None:
        semantics = json.loads(
            (ROOT / "configs/cashlog/leaf_semantics.json").read_text(encoding="utf-8")
        )
        lexicon = json.loads(
            (ROOT / "configs/cashlog/ocr_lexicon.json").read_text(encoding="utf-8")
        )
        category_tree = json.loads(
            (ROOT / "configs/cashlog/category_tree.json").read_text(encoding="utf-8")
        )
        leaf_ids = [str(row["id"]) for row in self.categories]
        tree_leaf_ids = [
            str(leaf["id"])
            for group in category_tree
            for leaf in group["leaves"]
        ]
        self.assertEqual(33, len(leaf_ids))
        self.assertEqual(33, len(set(leaf_ids)))
        self.assertEqual(leaf_ids, tree_leaf_ids)
        self.assertEqual(set(leaf_ids), set(semantics["leaves"]))
        self.assertEqual(set(leaf_ids), set(lexicon["leaves"]))
        self.assertEqual("13.33.1", semantics["taxonomy_version"])
        self.assertEqual(semantics["taxonomy_version"], lexicon["taxonomy_version"])

    def test_airflow_candidate_uses_portable_relative_paths(self) -> None:
        config = json.loads(
            (ROOT / "configs/cashlog/hybrid.airflow-candidate.json").read_text(
                encoding="utf-8"
            )
        )
        for key in [
            "categories",
            "semantics",
            "ocr_lexicon",
            "vision_model",
            "vision_head",
            "text_model",
            "ocr_detector_model",
            "ocr_classifier_model",
            "ocr_model",
        ]:
            self.assertFalse(Path(config[key]).is_absolute(), (key, config[key]))

    def test_serving_artifact_hashes_are_pinned(self) -> None:
        config_path = ROOT / "configs/cashlog/hybrid.serving.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        pairs = {
            "vision_head_sha256": config["vision_head"],
            "text_model_sha256": config["text_model"],
            "ocr_detector_model_sha256": config["ocr_detector_model"],
            "ocr_classifier_model_sha256": config["ocr_classifier_model"],
            "ocr_model_sha256": config["ocr_model"],
        }
        for key, value in pairs.items():
            path = (config_path.parent / value).resolve()
            self.assertTrue(path.is_file(), path)
            self.assertEqual(config[key], hashlib.sha256(path.read_bytes()).hexdigest())
        model = (config_path.parent / config["vision_model"] / "model.safetensors").resolve()
        self.assertEqual(
            config["vision_model_sha256"], hashlib.sha256(model.read_bytes()).hexdigest()
        )

    def test_ocr_matching_normalization_removes_spacing_and_punctuation(self) -> None:
        canonical = CashlogHybridClassifier._canonical_text
        self.assertEqual(canonical("LG: 유플러스 인터넷"), canonical("LG유플러스인터넷"))
        self.assertEqual(canonical("학원 · 강의"), canonical("학원강의"))

    def test_generic_receipt_without_semantics_uses_safe_fallback(self) -> None:
        classifier = CashlogHybridClassifier.__new__(CashlogHybridClassifier)
        classifier.leaf_ids = [str(row["id"]) for row in self.categories]
        classifier.config = {
            "decision": {
                "generic_receipt_terms": ["영수증", "결제", "승인번호"],
                "generic_receipt_min_terms": 2,
                "semantic_fallback_threshold": 0.25,
                "fallback_confidence": 0.55,
            }
        }
        scores = {leaf_id: 1 / 33 for leaf_id in classifier.leaf_ids}
        fused, reasons = classifier._apply_safe_fallback(
            scores,
            {"text": "결제 영수증 승인번호 123456"},
            {},
            {"low_information": False},
        )
        self.assertEqual(["receipt_without_semantic_evidence"], reasons)
        self.assertAlmostEqual(0.55, fused["misc_uncat"])

    def test_clear_object_with_no_ocr_keeps_visual_decision(self) -> None:
        classifier = CashlogHybridClassifier.__new__(CashlogHybridClassifier)
        classifier.leaf_ids = [str(row["id"]) for row in self.categories]
        classifier.config = {
            "decision": {
                "generic_receipt_terms": ["영수증", "결제"],
                "generic_receipt_min_terms": 2,
                "semantic_fallback_threshold": 0.25,
                "fallback_confidence": 0.55,
            }
        }
        scores = {leaf_id: 0.0 for leaf_id in classifier.leaf_ids}
        scores["meal_grocery"] = 1.0
        fused, reasons = classifier._apply_safe_fallback(
            scores, {"text": ""}, {}, {"low_information": False}
        )
        self.assertEqual([], reasons)
        self.assertEqual(scores, fused)

    def test_image_validation_decodes_real_payload(self) -> None:
        image_path = ROOT / "data/processed/cashlog33/e2e_fixtures/v1/images/meal_cafe/fixture-00.jpg"
        validate_image(image_path.read_bytes(), "image/jpeg", "receipt.jpg")
        with self.assertRaises(HTTPException) as context:
            validate_image(b"\xff\xd8\xffnot-a-jpeg", "image/jpeg", "receipt.jpg")
        self.assertEqual(400, context.exception.status_code)

    def test_internal_api_key_is_constant_time_guarded_when_configured(self) -> None:
        with patch.dict(
            os.environ,
            {"CATAI_REQUIRE_INTERNAL_API_KEY": "true", "CATAI_INTERNAL_API_KEY": "test-secret"},
            clear=False,
        ):
            verify_internal_api_key("test-secret")
            with self.assertRaises(HTTPException) as context:
                verify_internal_api_key("wrong")
            self.assertEqual(401, context.exception.status_code)

    def test_internal_api_key_fails_closed_when_required_but_missing(self) -> None:
        with patch.dict(
            os.environ,
            {"CATAI_REQUIRE_INTERNAL_API_KEY": "true"},
            clear=True,
        ):
            with self.assertRaises(HTTPException) as context:
                verify_internal_api_key(None)
            self.assertEqual(503, context.exception.status_code)


if __name__ == "__main__":
    unittest.main()
