from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from catai.predeploy_queue import build_predeploy_queue


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ROOT / "configs/cashlog/categories.json"
OPTIONAL_RUNTIME = all(
    importlib.util.find_spec(name) is not None for name in ["joblib", "numpy"]
)


class FakeProbabilityModel:
    def __init__(self) -> None:
        import numpy as np

        self.classes_ = np.asarray(["meal_grocery", "meal_cafe", "health_med"])

    def predict_proba(self, embeddings):
        import numpy as np

        rows = []
        for embedding in embeddings:
            marker = int(embedding[0])
            if marker == 0:
                rows.append([0.80, 0.10, 0.10])
            elif marker == 1:
                rows.append([0.10, 0.80, 0.10])
            else:
                rows.append([0.40, 0.35, 0.25])
        return np.asarray(rows)


@unittest.skipUnless(OPTIONAL_RUNTIME, "predeploy queue requires joblib and numpy")
class PredeployQueueTests(unittest.TestCase):
    def test_builds_current_model_mismatch_and_uncertainty_queue(self) -> None:
        import joblib
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.jsonl"
            samples = [
                {"sample_id": "train-1", "leaf_id": "meal_grocery", "relative_path": "a.jpg"},
                {"sample_id": "val-1", "leaf_id": "meal_grocery", "relative_path": "b.jpg"},
                {"sample_id": "test-1", "leaf_id": "meal_grocery", "relative_path": "c.jpg"},
            ]
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in samples), encoding="utf-8"
            )
            artifact_dir = root / "vision"
            artifact_dir.mkdir()
            split_rows = [
                {"sample_id": "train-1", "leaf_id": "meal_grocery", "split": "train"},
                {"sample_id": "val-1", "leaf_id": "meal_grocery", "split": "val"},
                {"sample_id": "test-1", "leaf_id": "meal_grocery", "split": "test"},
            ]
            (artifact_dir / "split_manifest.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in split_rows), encoding="utf-8"
            )
            np.savez_compressed(
                artifact_dir / "embedding_cache.npz",
                train=np.asarray([[0.0, 0.0]] * 4),
                train_labels=np.asarray(["meal_grocery"] * 4),
                val=np.asarray([[1.0, 0.0]]),
                val_labels=np.asarray(["meal_grocery"]),
                test=np.asarray([[2.0, 0.0]]),
                test_labels=np.asarray(["meal_grocery"]),
            )
            model = FakeProbabilityModel()
            joblib.dump(
                {"model": model, "classes": [str(value) for value in model.classes_]},
                artifact_dir / "vision_head.joblib",
            )
            output = root / "queue" / "scored.jsonl"

            summary = build_predeploy_queue(
                manifest_path=manifest,
                categories_path=CATEGORIES,
                artifact_dir=artifact_dir,
                output_path=output,
                confidence_threshold=0.60,
                margin_threshold=0.15,
            )
            rows = {
                row["sample_id"]: row
                for row in [json.loads(line) for line in output.read_text().splitlines()]
            }

            self.assertEqual(
                {"model_match": 1, "model_mismatch": 1, "model_uncertain": 1},
                summary["status_counts"],
            )
            self.assertEqual("model_mismatch", rows["val-1"]["model_review_status"])
            self.assertEqual("meal_cafe", rows["val-1"]["model_review_details"]["top3"][0]["leaf_id"])
            self.assertEqual("model_uncertain", rows["test-1"]["model_review_status"])
            self.assertEqual("train", rows["train-1"]["dataset_split"])


if __name__ == "__main__":
    unittest.main()
