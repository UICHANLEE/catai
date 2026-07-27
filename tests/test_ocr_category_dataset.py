from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.collect_cashlog_cord import flatten_lines
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
