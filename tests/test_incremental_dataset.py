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
        self.assertIn("--weak-additional-train-manifest", source)
        self.assertIn("--manifest {VISUAL_DIR}/manifest.jsonl", source)
        self.assertIn("--max-source-per-leaf 0", source)
        self.assertIn("--class-weight none", source)
        self.assertIn('task_id="validate_mps_meal_specialist"', source)
        self.assertIn('task_id="validate_ocr_category_dataset"', source)
        self.assertIn('task_id="evaluate_ocr_baseline"', source)
        self.assertIn("open_products_api/manifest.jsonl", source)
        self.assertIn("--meal-specialist-checkpoint", source)

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

    def test_marks_licensed_openverse_rows_as_weak_train_only_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "weak.jpg"
            image.write_bytes(b"weak-image")
            base_manifest = root / "base.jsonl"
            base_splits = root / "splits.jsonl"
            weak_manifest = root / "weak.jsonl"
            write_jsonl(base_manifest, [])
            write_jsonl(base_splits, [])
            write_jsonl(
                weak_manifest,
                [
                    {
                        "sample_id": "openverse:1",
                        "leaf_id": "meal_grocery",
                        "relative_path": str(image),
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "source": "openverse",
                        "status": "accepted",
                        "license": "by",
                    }
                ],
            )

            summary = build_incremental_dataset(
                base_manifest=base_manifest,
                base_split_manifest=base_splits,
                additional_manifests=[],
                weak_additional_manifests=[weak_manifest],
                categories_path=CATEGORIES,
                output_dir=root / "output",
            )
            rows = [
                json.loads(line)
                for line in (root / "output/manifest.jsonl").read_text().splitlines()
            ]

        self.assertEqual(1, summary["weak_additional_train_rows"])
        self.assertEqual("train", rows[0]["split"])
        self.assertEqual("train", rows[0]["split_lock"])
        self.assertEqual("weak_label", rows[0]["review_status"])
        self.assertTrue(rows[0]["weak_label"])

    def test_marks_product_opener_rows_as_weak_train_only_data(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "product.jpg"
            image.write_bytes(b"product-image")
            base_manifest = root / "base.jsonl"
            base_splits = root / "splits.jsonl"
            weak_manifest = root / "product.jsonl"
            write_jsonl(base_manifest, [])
            write_jsonl(base_splits, [])
            write_jsonl(
                weak_manifest,
                [
                    {
                        "sample_id": "openbeautyfacts:1",
                        "leaf_id": "fashion_beauty",
                        "relative_path": str(image),
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                        "source": "openbeautyfacts",
                        "status": "accepted",
                        "license": "cc-by-sa-3.0",
                        "provenance_type": "api_product_image",
                        "review_status": "product_type_weak",
                    }
                ],
            )

            build_incremental_dataset(
                base_manifest=base_manifest,
                base_split_manifest=base_splits,
                additional_manifests=[],
                weak_additional_manifests=[weak_manifest],
                categories_path=CATEGORIES,
                output_dir=root / "output",
            )
            rows = [
                json.loads(line)
                for line in (root / "output/manifest.jsonl").read_text().splitlines()
            ]

        self.assertEqual("train", rows[0]["split"])
        self.assertEqual("train", rows[0]["split_lock"])
        self.assertTrue(rows[0]["weak_label"])
        self.assertEqual(
            "product_opener_product_type_v1", rows[0]["label_method"]
        )

    def test_excludes_cross_leaf_weak_image_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "ambiguous.jpg"
            image.write_bytes(b"same-product-image")
            image_hash = hashlib.sha256(image.read_bytes()).hexdigest()
            base_manifest = root / "base.jsonl"
            base_splits = root / "splits.jsonl"
            weak_manifest = root / "weak.jsonl"
            write_jsonl(base_manifest, [])
            write_jsonl(base_splits, [])
            write_jsonl(
                weak_manifest,
                [
                    {
                        "sample_id": f"openfoodfacts-{source}:1",
                        "leaf_id": leaf_id,
                        "relative_path": str(image),
                        "sha256": image_hash,
                        "source": f"openfoodfacts-{source}",
                        "license": "cc-by-sa-3.0",
                        "provenance_type": "api_product_image",
                        "review_status": "product_type_weak",
                    }
                    for source, leaf_id in (
                        ("grocery", "meal_grocery"),
                        ("beverages", "meal_drink"),
                    )
                ],
            )

            summary = build_incremental_dataset(
                base_manifest=base_manifest,
                base_split_manifest=base_splits,
                additional_manifests=[],
                weak_additional_manifests=[weak_manifest],
                categories_path=CATEGORIES,
                output_dir=root / "output",
            )

        self.assertEqual(0, summary["merged_rows"])
        self.assertEqual(2, summary["weak_ambiguous_rows_excluded"])


if __name__ == "__main__":
    unittest.main()
