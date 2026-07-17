from __future__ import annotations

import io
import logging
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from catai.cashlog_api import app
from catai.telemetry import InferenceTelemetry, configure_json_logger, log_event
from scripts.train_cashlog_category_from_uecfood import (
    Sample,
    loss_weights_for_training,
)


class DummyClassifier:
    device = "mps"

    def analyze(self, _: bytes) -> dict:
        return {
            "success": True,
            "recommended_category": "meal_cafe",
            "confidence": 0.97,
            "need_user_check": False,
            "model": "test-model",
            "_timings_ms": {"vision": 12.0, "ocr": 20.0, "inference_total": 22.0},
        }


class ObservabilityTests(unittest.TestCase):
    def test_json_log_file_is_structured_and_rotating(self) -> None:
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ,
            {
                "CATAI_JSON_LOG_PATH": str(Path(directory) / "inference.jsonl"),
                "CATAI_LOG_MAX_BYTES": "1024",
                "CATAI_LOG_BACKUP_COUNT": "2",
            },
            clear=False,
        ):
            logger = configure_json_logger("catai.test.rotating")
            log_event(logger, "test_event", request_id="safe-id")
            for handler in logger.handlers:
                handler.flush()
                handler.close()
            logger.handlers.clear()
            content = (Path(directory) / "inference.jsonl").read_text(encoding="utf-8")
        self.assertIn('"event":"test_event"', content)
        self.assertIn('"request_id":"safe-id"', content)

    def test_telemetry_reports_latency_percentiles_without_payload_data(self) -> None:
        telemetry = InferenceTelemetry(max_records=3)
        telemetry.record(status="ok", total_ms=10, stages_ms={"vision": 4}, model="m", device="mps")
        telemetry.record(status="ok", total_ms=20, stages_ms={"vision": 8}, model="m", device="mps")
        snapshot = telemetry.snapshot()
        self.assertEqual(2, snapshot["total_requests"])
        self.assertEqual(15, snapshot["latency"]["p50_ms"])
        self.assertEqual(6, snapshot["stages"]["vision"]["p50_ms"])
        self.assertNotIn("records", snapshot)

    def test_api_adds_request_id_and_keeps_stage_timings_internal(self) -> None:
        image = io.BytesIO()
        Image.new("RGB", (32, 32), "white").save(image, format="JPEG")
        with patch("catai.cashlog_api.get_classifier", return_value=DummyClassifier()), patch.dict(
            os.environ,
            {"CATAI_REQUIRE_INTERNAL_API_KEY": "false", "CATAI_INCLUDE_PERFORMANCE_IN_RESPONSE": "false"},
            clear=False,
        ):
            with TestClient(app) as client:
                response = client.post(
                    "/analyze-image",
                    headers={"X-Request-ID": "test-request-1"},
                    files={"image": ("image.jpg", image.getvalue(), "image/jpeg")},
                )
        self.assertEqual(200, response.status_code)
        self.assertEqual("test-request-1", response.headers["x-request-id"])
        self.assertIn("x-process-time-ms", response.headers)
        self.assertNotIn("performance", response.json())
        self.assertNotIn("_timings_ms", response.json())

    def test_balanced_sampler_does_not_double_apply_class_weights(self) -> None:
        samples = [
            Sample(Path("a"), 0, "a", None),
            Sample(Path("b"), 0, "a", None),
            Sample(Path("c"), 1, "b", None),
        ]
        self.assertIsNone(loss_weights_for_training(samples, 2, balanced_sampling=True))
        weights = loss_weights_for_training(samples, 2, balanced_sampling=False)
        self.assertIsNotNone(weights)
        assert weights is not None
        self.assertGreater(float(weights[1]), float(weights[0]))


if __name__ == "__main__":
    unittest.main()
