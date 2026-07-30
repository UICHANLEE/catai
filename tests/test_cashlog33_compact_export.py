from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from catai.cashlog_hybrid_classifier import CashlogHybridClassifier
from scripts.evaluate_cashlog33_compact import load_frozen_test_rows
from scripts.export_quantize_cashlog33_mobilenet import ArrayCalibrationReader
from scripts.train_cashlog33_million_mobilenet import make_transforms


class CompactExportTest(unittest.TestCase):
    def test_calibration_reader_batches_and_rewinds(self) -> None:
        arrays = [
            np.full((1, 3, 2, 2), value, dtype=np.float32)
            for value in range(5)
        ]
        reader = ArrayCalibrationReader("images", arrays, batch_size=2)
        self.assertEqual(reader.get_next()["images"].shape, (2, 3, 2, 2))
        self.assertEqual(reader.get_next()["images"].shape, (2, 3, 2, 2))
        self.assertEqual(reader.get_next()["images"].shape, (1, 3, 2, 2))
        self.assertIsNone(reader.get_next())
        reader.rewind()
        self.assertEqual(reader.get_next()["images"].shape, (2, 3, 2, 2))

    def test_frozen_test_rows_follow_split_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            split = root / "split.jsonl"
            manifest.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "leaf_id": "meal_cafe",
                            "relative_path": f"{sample_id}.jpg",
                        }
                    )
                    for sample_id in ["b", "a"]
                )
                + "\n",
                encoding="utf-8",
            )
            split.write_text(
                "\n".join(
                    [
                        json.dumps({"sample_id": "a", "split": "test"}),
                        json.dumps({"sample_id": "b", "split": "train"}),
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            rows = load_frozen_test_rows(manifest, split)
            self.assertEqual([row["sample_id"] for row in rows], ["a"])

    def test_serving_preprocess_matches_training_evaluation(self) -> None:
        pixels = np.random.default_rng(33).integers(
            0, 256, (333, 517, 3), dtype=np.uint8
        )
        image = Image.fromarray(pixels)
        _, evaluation = make_transforms(224)
        serving = CashlogHybridClassifier._compact_image_tensor(image, 224)
        training = evaluation(image).numpy()[None, ...]
        np.testing.assert_array_equal(serving, training)


if __name__ == "__main__":
    unittest.main()
