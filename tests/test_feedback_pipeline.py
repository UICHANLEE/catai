from __future__ import annotations

import json
import stat
import tempfile
import unittest
from pathlib import Path

from catai.feedback import (
    curate_feedback_release,
    export_feedback_release,
    load_leaf_ids,
    normalize_feedback_rows,
)


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ROOT / "configs/cashlog/categories.json"
HMAC_KEY = b"cashlog-feedback-test-key-32-bytes-minimum"
USER_ID = "10000000-0000-4000-8000-000000000001"


def feedback_row(
    index: int,
    leaf_id: str,
    *,
    model_category: str | None = None,
    status: str = "approved",
    source: str = "accepted_prediction",
    consent: bool = True,
) -> dict[str, object]:
    model_leaf = model_category or leaf_id
    candidates = [{"category": model_leaf, "confidence": 0.72}]
    if leaf_id != model_leaf:
        candidates.append({"category": leaf_id, "confidence": 0.2})
    return {
        "schema_version": 2,
        "event_id": f"20000000-0000-4000-8000-{index:012d}",
        "sample_id": f"30000000-0000-4000-8000-{index:012d}",
        "user_id": USER_ID,
        "expense_id": f"private-expense-{index}",
        "request_id": f"private-request-{index}",
        "model_version": "cashlog33-hybrid-v1.1-fast",
        "taxonomy_version": "13.33.1",
        "model_category": model_leaf,
        "predicted_top3": candidates,
        "selected_leaf_id": leaf_id,
        "occurred_at": "2026-07-17T00:00:00+00:00",
        "source": source,
        "image_retention_consent": consent,
        "image_object_key": f"{USER_ID}/expense-{index}.jpg" if consent else None,
        "review_status": status,
        "reviewed_at": "2026-07-17T01:00:00+00:00" if status != "pending" else None,
    }


class FeedbackPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.leaf_ids = load_leaf_ids(CATEGORIES)

    def test_export_deidentifies_rows_and_separates_restricted_image_paths(self) -> None:
        rows = [
            feedback_row(1, "meal_cafe"),
            feedback_row(
                2,
                "meal_dining",
                model_category="meal_cafe",
                source="top3_selection",
            ),
            feedback_row(3, "health_med", status="pending", consent=False),
        ]
        invalid = feedback_row(4, "meal_cafe")
        invalid["image_object_key"] = "another-user/private.jpg"
        rows.append(invalid)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "release"
            summary = export_feedback_release(
                rows,
                output_dir=output,
                leaf_ids=self.leaf_ids,
                hmac_key=HMAC_KEY,
            )
            event_text = (output / "events.jsonl").read_text(encoding="utf-8")
            secure_text = (output / "secure_image_index.jsonl").read_text(
                encoding="utf-8"
            )

            self.assertEqual(3, summary["valid_events"])
            self.assertEqual(1, summary["quarantined_events"])
            self.assertNotIn(USER_ID, event_text)
            self.assertNotIn("private-expense", event_text)
            self.assertNotIn("private-request", event_text)
            self.assertIn(USER_ID, secure_text)
            self.assertEqual(
                0o600,
                stat.S_IMODE((output / "secure_image_index.jsonl").stat().st_mode),
            )
            self.assertEqual(0o700, stat.S_IMODE(output.stat().st_mode))

    def test_invalid_or_duplicate_events_are_quarantined_without_raw_rows(self) -> None:
        duplicate = feedback_row(1, "meal_cafe")
        events, secure, quarantine = normalize_feedback_rows(
            [duplicate, duplicate],
            leaf_ids=set(self.leaf_ids),
            hmac_key=HMAC_KEY,
        )
        self.assertEqual(1, len(events))
        self.assertEqual(1, len(secure))
        self.assertEqual("duplicate event_id", quarantine[0]["error"])
        self.assertNotIn("user_id", quarantine[0])

        invalid_consent = feedback_row(2, "meal_cafe")
        invalid_consent["image_retention_consent"] = "false"
        _, _, invalid_rows = normalize_feedback_rows(
            [invalid_consent],
            leaf_ids=set(self.leaf_ids),
            hmac_key=HMAC_KEY,
        )
        self.assertEqual("image_retention_consent must be boolean", invalid_rows[0]["error"])

    def test_curation_requires_reviewed_consented_images_for_all_33_leaves(self) -> None:
        rows = [feedback_row(index, leaf_id) for index, leaf_id in enumerate(self.leaf_ids, 1)]
        events, _, quarantine = normalize_feedback_rows(
            rows,
            leaf_ids=set(self.leaf_ids),
            hmac_key=HMAC_KEY,
        )
        self.assertEqual([], quarantine)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "curated"
            summary = curate_feedback_release(
                events,
                output_dir=output,
                leaf_ids=self.leaf_ids,
                taxonomy_version="13.33.1",
                minimum_images_per_leaf=1,
            )
            candidates = [
                json.loads(line)
                for line in (output / "training_candidates.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]

            self.assertTrue(summary["ready_for_training"])
            self.assertFalse(summary["auto_training_allowed"])
            self.assertEqual(33, len(candidates))
            self.assertNotIn("image_object_key", candidates[0])

    def test_airflow_collects_daily_without_triggering_unreviewed_training(self) -> None:
        dag_source = (ROOT / "dags/cashlog_feedback_dag.py").read_text(encoding="utf-8")
        self.assertIn('dag_id="cashlog_feedback_curation"', dag_source)
        self.assertIn('schedule="@daily"', dag_source)
        self.assertIn("SUPABASE_SERVICE_ROLE_KEY", dag_source)
        self.assertIn("auto_training_allowed", dag_source)
        self.assertNotIn("TriggerDagRunOperator", dag_source)


if __name__ == "__main__":
    unittest.main()
