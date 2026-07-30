import csv
from pathlib import Path

import requests

from scripts.collect_cashlog_openimages_train import (
    THREAD_LOCAL,
    prepare_candidates,
    request_image,
)


def write_labels(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["ImageID", "Source", "LabelName", "Confidence"]
        )
        writer.writeheader()
        writer.writerows(rows)


def test_prepare_candidates_drops_cross_leaf_images_and_caps_deterministically(tmp_path):
    path = tmp_path / "labels.csv"
    write_labels(
        path,
        [
            {"ImageID": "a", "Source": "verification", "LabelName": "m1", "Confidence": "1"},
            {"ImageID": "b", "Source": "verification", "LabelName": "m1", "Confidence": "1"},
            {"ImageID": "c", "Source": "verification", "LabelName": "m1", "Confidence": "1"},
            {"ImageID": "x", "Source": "verification", "LabelName": "m1", "Confidence": "1"},
            {"ImageID": "x", "Source": "verification", "LabelName": "m2", "Confidence": "1"},
            {"ImageID": "n", "Source": "verification", "LabelName": "m1", "Confidence": "0"},
        ],
    )

    selected, stats = prepare_candidates(
        path,
        {"m1": "one", "m2": "two"},
        {"m1": "leaf_a", "m2": "leaf_b"},
        per_leaf=2,
        seed=7,
    )

    assert len(selected) == 2
    assert {row["leaf_id"] for row in selected} == {"leaf_a"}
    assert stats == {
        "mapped_unique_images": 4,
        "ambiguous_cross_leaf_images_dropped": 1,
        "selected_before_metadata_validation": 2,
    }

    uncapped, uncapped_stats = prepare_candidates(
        path,
        {"m1": "one", "m2": "two"},
        {"m1": "leaf_a", "m2": "leaf_b"},
        per_leaf=None,
        seed=7,
    )
    assert len(uncapped) == 3
    assert uncapped_stats["selected_before_metadata_validation"] == 3


def test_request_image_retries_chunked_connection_reset(monkeypatch):
    class Response:
        status_code = 200
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size > 0
            yield b"recovered"

    class Session:
        def __init__(self):
            self.calls = 0

        def get(self, *_args, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise requests.exceptions.ChunkedEncodingError("reset")
            return Response()

    session = Session()
    monkeypatch.setattr(
        THREAD_LOCAL, "http_session", session, raising=False
    )
    monkeypatch.setattr(
        "scripts.collect_cashlog_openimages_train.time.sleep", lambda _: None
    )

    assert request_image("https://example.test/image.jpg", timeout=1) == b"recovered"
    assert session.calls == 2
