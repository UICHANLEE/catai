from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_cashlog33_incremental_dataset import build_incremental_dataset


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ROOT / "configs/cashlog/categories.json"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


class IncrementalDatasetTests(unittest.TestCase):
    def test_training_dag_builds_and_trains_from_the_merged_visual_manifest(self) -> None:
        source = (ROOT / "dags/cashlog33_pipeline_dag.py").read_text(encoding="utf-8")
        self.assertIn('task_id="build_visual_dataset"', source)
        self.assertIn("scripts/build_cashlog33_incremental_dataset.py", source)
        self.assertIn("--optional-additional-train-manifest", source)
        self.assertIn("--manifest {VISUAL_DIR}/manifest.jsonl", source)

    def test_merges_reviewed_rows_into_train_and_preserves_base_splits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base_image = root / "base.jpg"
            base_image.write_bytes(b"base-image")
            actual_image = root / "actual.jpg"
            actual_image.write_bytes(b"actual-image")
            base_manifest = root / "base.jsonl"
            write_jsonl(
                base_manifest,
                [
                    {
                        "sample_id": "base:1",
                        "leaf_id": "meal_dining",
                        "relative_path": str(base_image),
                        "sha256": hashlib.sha256(base_image.read_bytes()).hexdigest(),
                        "source": "base",
                    }
                ],
            )
            base_splits = root / "splits.jsonl"
            write_jsonl(
                base_splits,
                [
                    {
                        "sample_id": "base:1",
                        "leaf_id": "meal_dining",
                        "split": "test",
                    }
                ],
            )
            additions = root / "actual.jsonl"
            write_jsonl(
                additions,
                [
                    {
                        "sample_id": "actual:1",
                        "leaf_id": "meal_dining",
                        "relative_path": str(actual_image),
                        "sha256": hashlib.sha256(actual_image.read_bytes()).hexdigest(),
                        "source": "cashlog_actual",
                        "split_lock": "train",
                        "review_status": "approved",
                    }
                ],
            )

            summary = build_incremental_dataset(
                base_manifest=base_manifest,
                base_split_manifest=base_splits,
                additional_manifests=[additions],
                categories_path=CATEGORIES,
                output_dir=root / "output",
            )

            rows = [
                json.loads(line)
                for line in (root / "output/manifest.jsonl").read_text().splitlines()
            ]
            self.assertEqual(2, summary["merged_rows"])
            self.assertEqual({"test": 1, "train": 1}, summary["split_counts"])
            self.assertEqual("train", rows[0]["split"])
            self.assertEqual("actual:1", rows[0]["sample_id"])
            self.assertEqual("test", rows[1]["split"])

    def test_rejects_additional_rows_that_are_not_train_locked(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "image.jpg"
            image.write_bytes(b"image")
            base_manifest = root / "base.jsonl"
            write_jsonl(base_manifest, [])
            base_splits = root / "splits.jsonl"
            write_jsonl(base_splits, [])
            additions = root / "actual.jsonl"
            write_jsonl(
                additions,
                [
                    {
                        "sample_id": "actual:1",
                        "leaf_id": "meal_dining",
                        "relative_path": str(image),
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "source": "cashlog_actual",
                        "split_lock": "val",
                        "review_status": "approved",
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "not train-locked"):
                build_incremental_dataset(
                    base_manifest=base_manifest,
                    base_split_manifest=base_splits,
                    additional_manifests=[additions],
                    categories_path=CATEGORIES,
                    output_dir=root / "output",
                )


if __name__ == "__main__":
    unittest.main()
