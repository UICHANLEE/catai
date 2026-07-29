import unittest

from scripts.build_cashlog33_original_train_manifest import merge


class BuildCashlogOriginalTrainManifestTests(unittest.TestCase):
    def test_excludes_proxy_holdout_and_deduplicates_train_rows(self) -> None:
        rows = [
            {
                "sample_id": "holdout",
                "source": "openimages_v7_validation",
                "leaf_id": "life_goods",
                "sha256": "one",
                "split_lock": "train",
            },
            {
                "sample_id": "train-a",
                "source": "amazon_berkeley_objects",
                "leaf_id": "life_goods",
                "sha256": "two",
                "split_lock": "train",
            },
            {
                "sample_id": "train-b",
                "source": "other",
                "leaf_id": "life_goods",
                "sha256": "two",
                "split_lock": "train",
            },
        ]
        merged, metrics = merge(rows)
        self.assertEqual([row["sample_id"] for row in merged], ["train-a"])
        self.assertEqual(metrics["duplicate_rows_dropped"], 1)

    def test_drops_cross_leaf_hash_conflicts(self) -> None:
        rows = [
            {
                "sample_id": "a",
                "source": "one",
                "leaf_id": "life_goods",
                "sha256": "same",
                "split_lock": "train",
            },
            {
                "sample_id": "b",
                "source": "two",
                "leaf_id": "fashion_beauty",
                "sha256": "same",
                "split_lock": "train",
            },
        ]
        merged, metrics = merge(rows)
        self.assertEqual(merged, [])
        self.assertEqual(metrics["cross_leaf_hash_conflicts_dropped"], 1)


if __name__ == "__main__":
    unittest.main()
