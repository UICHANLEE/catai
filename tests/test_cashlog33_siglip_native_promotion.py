from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.promote_cashlog33_siglip_native import promote


class SiglipNativePromotionTest(unittest.TestCase):
    def test_recall_regression_requires_explicit_approval(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            serving = root / "serving.json"
            candidate = root / "candidate.json"
            report = root / "report.json"
            rollback = root / "rollback.json"
            head = root / "head.joblib"
            serving.write_text(
                json.dumps({"model_version": "old"}), encoding="utf-8"
            )
            head.write_bytes(b"new-head")
            head_sha = hashlib.sha256(head.read_bytes()).hexdigest()
            candidate.write_text(
                json.dumps(
                    {
                        "model_version": "new",
                        "vision_head": "head.joblib",
                        "vision_head_sha256": head_sha,
                    }
                ),
                encoding="utf-8",
            )
            report.write_text(
                json.dumps(
                    {
                        "serving_config_sha256": hashlib.sha256(
                            serving.read_bytes()
                        ).hexdigest(),
                        "baseline": {
                            "top1_accuracy": 0.68,
                            "macro_f1": 0.51,
                        },
                        "native": {
                            "top1_accuracy": 0.75,
                            "macro_f1": 0.62,
                        },
                        "gate_checks": {
                            "native_leaf_recall_regression_within_limit": False
                        },
                        "export": {"vision_head_sha256": head_sha},
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(RuntimeError):
                promote(
                    report,
                    candidate,
                    serving,
                    rollback,
                    approved_recall_regression=False,
                )
            result = promote(
                report,
                candidate,
                serving,
                rollback,
                approved_recall_regression=True,
            )
            self.assertTrue(result["approved_recall_regression"])
            self.assertEqual(
                json.loads(serving.read_text())["model_version"], "new"
            )
            self.assertEqual(
                json.loads(rollback.read_text())["model_version"], "old"
            )


if __name__ == "__main__":
    unittest.main()
