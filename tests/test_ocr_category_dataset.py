from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_cashlog_cord import flatten_lines
from scripts.build_cashlog33_ocr_text_manifest import corrupt_text
from scripts.generate_cashlog33_ocr_category_dataset import find_font, render_one


ROOT = Path(__file__).resolve().parents[1]


class OcrCategoryDatasetTests(unittest.TestCase):
    def test_source_catalog_declares_more_than_100k_balanced_images(self) -> None:
        catalog = json.loads(
            (ROOT / "configs/cashlog/data_sources.json").read_text(encoding="utf-8")
        )
        synthetic = next(
            row
            for row in catalog["sources"]
            if row["id"] == "cashlog_ocr_category_synthetic_v1"
        )
        self.assertEqual(112200, synthetic["expected_images"])
        self.assertGreaterEqual(synthetic["expected_train_images"], 100000)
        self.assertEqual("enabled", synthetic["status"])
        expanded = next(
            row
            for row in catalog["sources"]
            if row["id"] == "cashlog_ocr_category_synthetic_v2_500k"
        )
        self.assertGreaterEqual(expanded["expected_train_images"], 500000)
        self.assertEqual(508200, expanded["expected_images"])

    def test_synthetic_renderer_emits_image_manifest_and_ocr_boxes(self) -> None:
        try:
            font = find_font(None)
        except FileNotFoundError:
            self.skipTest("Korean font is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            manifest_line, ocr_line, paddle_line = render_one(
                {
                    "leaf_id": "meal_cafe",
                    "display_name": "카페·디저트",
                    "split": "train",
                    "index": 0,
                    "text": "카페라떼 결제 5,500원",
                    "source_sample_id": "source-1",
                    "source_group_key": "group-1",
                    "font_path": str(font),
                    "output_root": directory,
                    "seed": 250716,
                }
            )
            manifest = json.loads(manifest_line)
            ocr = json.loads(ocr_line)
            image_path = Path(manifest["relative_path"])
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            self.assertTrue(image_path.is_file())
            self.assertEqual("meal_cafe", manifest["leaf_id"])
            self.assertGreater(len(ocr["lines"]), 3)
            self.assertIn("\t[", paddle_line)
            for line in ocr["lines"]:
                self.assertEqual(4, len(line["points"]))

    def test_v2_renderer_removes_explicit_category_hint(self) -> None:
        try:
            font = find_font(None)
        except FileNotFoundError:
            self.skipTest("Korean font is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            _, ocr_line, _ = render_one(
                {
                    "leaf_id": "meal_cafe",
                    "display_name": "카페·디저트",
                    "split": "train",
                    "index": 0,
                    "text": "아메리카노 4,500원",
                    "source_sample_id": "source-1",
                    "source_group_key": "group-1",
                    "font_paths": [str(font)],
                    "output_root": directory,
                    "seed": 500027,
                    "sample_prefix": "cashlog-ocr-synth-v2",
                    "source_id": "cashlog_ocr_category_synthetic_v2",
                    "allow_label_hint": False,
                }
            )
            transcription = json.loads(ocr_line)["transcription"]
            self.assertNotIn("분류 메모", transcription)
            self.assertNotIn("카페·디저트", transcription)

    def test_ocr_corruption_is_deterministic(self) -> None:
        first = corrupt_text("결제 영수증\n아메리카노 4,500원", "sample-1", "train")
        second = corrupt_text("결제 영수증\n아메리카노 4,500원", "sample-1", "train")
        self.assertEqual(first, second)

    def test_500k_airflow_dag_is_manual_and_promotion_gated(self) -> None:
        source = (ROOT / "dags/cashlog33_500k_training_dag.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('dag_id="cashlog33_500k_training_pipeline"', source)
        self.assertIn("schedule=None", source)
        self.assertIn("build_cashlog33_500k_dataset.sh", source)
        self.assertIn("--alpha 0.00001", source)
        self.assertNotIn("hybrid.serving.json", source)

    def test_cord_boxes_are_flattened_without_cashlog_labels(self) -> None:
        lines = flatten_lines(
            {
                "valid_line": [
                    {
                        "category": "menu.nm",
                        "words": [
                            {
                                "text": "LATTE",
                                "quad": {
                                    "x1": 1,
                                    "y1": 2,
                                    "x2": 10,
                                    "y2": 2,
                                    "x3": 10,
                                    "y3": 8,
                                    "x4": 1,
                                    "y4": 8,
                                },
                            }
                        ],
                    }
                ]
            }
        )
        self.assertEqual("LATTE", lines[0]["text"])
        self.assertEqual("menu.nm", lines[0]["semantic_category"])
        self.assertEqual([[1, 2], [10, 2], [10, 8], [1, 8]], lines[0]["points"])


if __name__ == "__main__":
    unittest.main()
