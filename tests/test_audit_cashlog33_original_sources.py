import json
import tempfile
import unittest
from pathlib import Path

from scripts.audit_cashlog33_original_sources import audit


class AuditCashlogOriginalSourcesTests(unittest.TestCase):
    def test_detects_duplicate_and_cross_leaf_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.jsonl"
            second = Path(directory) / "second.jsonl"
            first.write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "leaf_id": "meal_grocery",
                        "source": "one",
                        "sha256": "same",
                    }
                )
                + "\n"
            )
            second.write_text(
                json.dumps(
                    {
                        "sample_id": "b",
                        "leaf_id": "life_goods",
                        "source": "two",
                        "sha256": "same",
                    }
                )
                + "\n"
            )
            report = audit([first, second])
        self.assertEqual(report["unique_hashes"], 1)
        self.assertEqual(report["duplicate_hashes"], 1)
        self.assertEqual(len(report["cross_leaf_hash_conflicts"]), 1)


if __name__ == "__main__":
    unittest.main()
