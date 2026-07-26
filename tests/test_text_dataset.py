from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from scripts.build_cashlog33_text_dataset import source_rows


class TextDatasetTests(unittest.TestCase):
    def test_zero_limit_contract_keeps_every_mapped_source_row(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "transactions.csv"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["description", "category"]
                )
                writer.writeheader()
                writer.writerows(
                    [
                        {"description": "GROCERY ONE", "category": "Groceries"},
                        {"description": "GROCERY TWO", "category": "Groceries"},
                        {"description": "PAYROLL", "category": "Income"},
                    ]
                )

            metadata = {"revision": "test", "license": "test"}
            unlimited = source_rows(csv_path, metadata, max_per_leaf=None)
            limited = source_rows(csv_path, metadata, max_per_leaf=1)

        self.assertEqual(2, len(unlimited))
        self.assertEqual(1, len(limited))
        self.assertEqual(
            {"meal_grocery"}, {str(row["leaf_id"]) for row in unlimited}
        )


if __name__ == "__main__":
    unittest.main()
