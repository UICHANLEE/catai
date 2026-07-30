from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_cashlog33_compact_serving_config import build_config
from scripts.promote_cashlog33_compact import promote
from scripts.stage_cashlog33_compact_artifact import stage_artifacts


class CompactPromotionTest(unittest.TestCase):
    def test_stage_artifacts_requires_exported_checksums(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "source.onnx"
            labels = root / "source-labels.json"
            export_report = root / "export-report.json"
            output = root / "assets"
            model.write_bytes(b"quantized-model")
            labels.write_text("[]", encoding="utf-8")
            export_report.write_text(
                json.dumps(
                    {
                        "int8": {
                            "sha256": hashlib.sha256(
                                model.read_bytes()
                            ).hexdigest()
                        },
                        "labels": {
                            "sha256": hashlib.sha256(
                                labels.read_bytes()
                            ).hexdigest()
                        },
                        "quantization": {"format": "QDQ"},
                        "probability_calibration": {"temperature": 1.7},
                    }
                ),
                encoding="utf-8",
            )

            result = stage_artifacts(
                model, labels, export_report, output, "compact-version"
            )

            self.assertEqual(result["model_version"], "compact-version")
            self.assertEqual(
                (output / "model.int8.onnx").read_bytes(), b"quantized-model"
            )
            self.assertTrue((output / "provenance.json").is_file())

    def test_build_config_uses_int8_and_disables_specialist(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base.json"
            model = root / "model.onnx"
            labels = root / "labels.json"
            output = root / "candidate.json"
            (root / "provenance.json").write_text(
                json.dumps(
                    {
                        "probability_calibration": {
                            "temperature": 1.7
                        }
                    }
                ),
                encoding="utf-8",
            )
            base.write_text(
                json.dumps(
                    {
                        "model_version": "old",
                        "categories": "categories.json",
                        "meal_specialist": {"enabled": True},
                    }
                ),
                encoding="utf-8",
            )
            model.write_bytes(b"int8")
            labels.write_text("[]", encoding="utf-8")
            config = build_config(
                base, model, labels, output, "compact-version"
            )
            self.assertEqual(config["model_version"], "compact-version")
            self.assertEqual(
                config["compact_vision"]["model_sha256"],
                hashlib.sha256(b"int8").hexdigest(),
            )
            self.assertFalse(config["meal_specialist"]["enabled"])
            self.assertEqual(config["categories"], "categories.json")
            self.assertEqual(config["compact_vision"]["temperature"], 1.7)

    def test_promotion_requires_gate_and_preserves_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            report = root / "report.json"
            candidate = root / "candidate.json"
            serving = root / "serving.json"
            rollback = root / "rollback.json"
            model = root / "model.onnx"
            labels = root / "labels.json"
            model.write_bytes(b"quantized-model")
            labels.write_text("[]", encoding="utf-8")
            model_sha = hashlib.sha256(model.read_bytes()).hexdigest()
            labels_sha = hashlib.sha256(labels.read_bytes()).hexdigest()
            candidate.write_text(
                json.dumps(
                    {
                        "model_version": "new",
                        "compact_vision": {
                            "enabled": True,
                            "runtime": "onnxruntime-int8",
                            "model": "model.onnx",
                            "model_sha256": model_sha,
                            "labels": "labels.json",
                            "labels_sha256": labels_sha,
                        },
                    }
                ),
                encoding="utf-8",
            )
            serving.write_text(
                json.dumps({"model_version": "old"}), encoding="utf-8"
            )
            report.write_text(
                json.dumps({"promotion_gate_passed": False, "gate_checks": {}}),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                promote(report, candidate, serving, rollback)
            self.assertFalse(rollback.exists())

            report.write_text(
                json.dumps(
                    {
                        "promotion_gate_passed": True,
                        "gate_checks": {},
                        "serving_config_sha256": hashlib.sha256(
                            serving.read_bytes()
                        ).hexdigest(),
                        "int8": {"artifact": {"sha256": model_sha}},
                    }
                ),
                encoding="utf-8",
            )
            result = promote(report, candidate, serving, rollback)
            self.assertEqual(result["model_version"], "new")
            self.assertEqual(
                json.loads(rollback.read_text())["model_version"], "old"
            )
            self.assertEqual(
                json.loads(serving.read_text())["model_version"], "new"
            )


if __name__ == "__main__":
    unittest.main()
