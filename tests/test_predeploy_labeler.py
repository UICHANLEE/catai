from __future__ import annotations

import base64
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from catai.predeploy_labeler import (
    DecisionPayload,
    ReviewRepository,
    RevisionConflict,
    create_app,
)


ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ROOT / "configs/cashlog/categories.json"
PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def manifest_row(
    sample_id: str,
    image_path: Path,
    original_leaf: str,
    predicted_leaf: str,
    status: str,
) -> dict[str, object]:
    second = "misc_other" if predicted_leaf != "misc_other" else "misc_uncat"
    return {
        "sample_id": sample_id,
        "leaf_id": original_leaf,
        "relative_path": str(image_path),
        "source": "test",
        "license": "cc0",
        "sha256": hashlib.sha256(PNG_BYTES).hexdigest(),
        "review_status": status,
        "title": f"title {sample_id}",
        "review_details": {
            "top3": [
                {"leaf_id": predicted_leaf, "score": 0.72},
                {"leaf_id": second, "score": 0.21},
            ]
        },
    }


class PredeployLabelerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.directory = Path(self.temporary.name)
        self.images = self.directory / "images"
        self.images.mkdir()
        paths = [self.images / f"sample-{index}.png" for index in range(3)]
        for path in paths:
            path.write_bytes(PNG_BYTES)
        rows = [
            manifest_row(
                "sample:wrong",
                paths[0],
                "meal_grocery",
                "meal_cafe",
                "auto_rejected",
            ),
            manifest_row(
                "sample:match",
                paths[1],
                "meal_cafe",
                "meal_cafe",
                "auto_approved",
            ),
            manifest_row(
                "sample:correct",
                paths[2],
                "health_med",
                "life_goods",
                "pending",
            ),
        ]
        self.manifest = self.directory / "scored_manifest.jsonl"
        self.manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        self.output = self.directory / "review"
        self.repository = ReviewRepository(
            input_manifest=self.manifest,
            categories_path=CATEGORIES,
            output_dir=self.output,
            image_roots=[self.directory],
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prioritizes_mismatches_and_persists_human_decisions(self) -> None:
        state = self.repository.state()
        self.assertEqual(3, state["summary"]["total"])
        self.assertEqual(2, state["summary"]["mismatches"])
        self.assertTrue(state["samples"][0]["mismatch"])

        confirmed = self.repository.decide(
            "sample:wrong",
            DecisionPayload(
                decision="approved",
                selected_leaf_id="meal_grocery",
                expected_revision=0,
            ),
        )
        self.assertEqual("approved", confirmed["decision"])
        self.assertEqual(0o600, self.repository.decisions_path.stat().st_mode & 0o777)

        reloaded = ReviewRepository(
            input_manifest=self.manifest,
            categories_path=CATEGORIES,
            output_dir=self.output,
            image_roots=[self.directory],
        )
        self.assertEqual(
            "meal_grocery",
            reloaded.serialize_sample("sample:wrong")["decision"]["selected_leaf_id"],
        )

    def test_rejects_stale_mutations(self) -> None:
        self.repository.decide(
            "sample:wrong",
            DecisionPayload(
                decision="approved",
                selected_leaf_id="meal_grocery",
                expected_revision=0,
            ),
        )
        with self.assertRaises(RevisionConflict):
            self.repository.decide(
                "sample:wrong",
                DecisionPayload(
                    decision="rejected",
                    expected_revision=0,
                ),
            )

    def test_export_applies_corrections_and_locks_reviewed_rows_to_train(self) -> None:
        self.repository.decide(
            "sample:wrong",
            DecisionPayload(
                decision="approved",
                selected_leaf_id="meal_grocery",
                expected_revision=0,
            ),
        )
        self.repository.decide(
            "sample:correct",
            DecisionPayload(
                decision="approved",
                selected_leaf_id="fashion_beauty",
                note="human correction",
                expected_revision=0,
            ),
        )
        self.repository.decide(
            "sample:match",
            DecisionPayload(decision="rejected", expected_revision=0),
        )

        summary = self.repository.export()
        corrected = {
            row["sample_id"]: row
            for row in [
                json.loads(line)
                for line in Path(summary["corrected_manifest"])
                .read_text(encoding="utf-8")
                .splitlines()
            ]
        }
        training = [
            json.loads(line)
            for line in Path(summary["training_manifest"])
            .read_text(encoding="utf-8")
            .splitlines()
        ]

        self.assertEqual("fashion_beauty", corrected["sample:correct"]["leaf_id"])
        self.assertEqual("health_med", corrected["sample:correct"]["predeploy_original_leaf_id"])
        self.assertEqual("train", corrected["sample:correct"]["split_lock"])
        self.assertEqual("rejected", corrected["sample:match"]["review_status"])
        self.assertNotIn("split_lock", corrected["sample:match"])
        self.assertEqual({"sample:wrong", "sample:correct"}, {row["sample_id"] for row in training})
        self.assertEqual(2, summary["human_verified_rows"])
        self.assertEqual(1, summary["decisions"]["corrected"])

    def test_api_is_loopback_only_and_requires_token_for_mutation(self) -> None:
        app = create_app(self.repository)
        with TestClient(app, base_url="http://127.0.0.1") as client:
            state = client.get("/api/state")
            self.assertEqual(200, state.status_code)
            payload = state.json()
            sample = next(row for row in payload["samples"] if row["sample_id"] == "sample:wrong")
            self.assertEqual(200, client.get(sample["image_url"]).status_code)
            mutation = {
                "decision": "approved",
                "selected_leaf_id": "meal_grocery",
                "expected_revision": 0,
            }
            self.assertEqual(
                401,
                client.put("/api/decisions/sample%3Awrong", json=mutation).status_code,
            )
            saved = client.put(
                "/api/decisions/sample%3Awrong",
                json=mutation,
                headers={"x-labeler-token": payload["mutation_token"]},
            )
            self.assertEqual(200, saved.status_code)

        with TestClient(app, base_url="http://labeler.example") as remote_client:
            self.assertEqual(403, remote_client.get("/api/state").status_code)


if __name__ == "__main__":
    unittest.main()
