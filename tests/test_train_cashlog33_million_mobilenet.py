import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_cashlog33_million_mobilenet import (
    ScheduledViewDataset,
    evaluate,
    fit_temperature,
    inverse_schedule_class_weights,
    make_transforms,
)


def test_scheduled_dataset_uses_exact_schedule_and_labels(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "scripts.train_cashlog33_million_mobilenet.ROOT", tmp_path
    )
    image_path = tmp_path / "image.jpg"
    Image.new("RGB", (256, 256), "red").save(image_path)
    train_transform, _ = make_transforms(224)
    dataset = ScheduledViewDataset(
        [{"relative_path": "image.jpg", "leaf_id": "leaf"}],
        np.asarray([0, 0], dtype=np.uint32),
        np.asarray([7, 8], dtype=np.uint32),
        {"leaf": 0},
        train_transform,
    )

    first, label = dataset[0]

    assert len(dataset) == 2
    assert first.shape == (3, 224, 224)
    assert label == 0


def test_inverse_schedule_weights_balance_uneven_views():
    rows = [
        {"relative_path": "a.jpg", "leaf_id": "large"},
        {"relative_path": "b.jpg", "leaf_id": "small"},
    ]
    weights, counts = inverse_schedule_class_weights(
        rows,
        np.asarray([0, 0, 0, 1], dtype=np.uint32),
        {"large": 0, "small": 1},
    )

    assert counts == {"large": 3, "small": 1}
    assert weights[1] > weights[0]
    assert np.isclose(weights.mean(), 1.0)


def test_temperature_fitting_reduces_overconfident_nll():
    logits = np.asarray(
        [[8.0, 0.0], [8.0, 0.0], [8.0, 0.0], [0.0, 8.0]],
        dtype=np.float32,
    )
    expected = np.asarray([0, 0, 1, 1], dtype=np.int64)

    calibration = fit_temperature(logits, expected)

    assert calibration["temperature"] > 1.0
    assert (
        calibration["nll_calibrated"]
        < calibration["nll_uncalibrated"]
    )


def test_evaluate_records_probability_calibration():
    logits = torch.tensor(
        [[8.0, 0.0], [8.0, 0.0], [8.0, 0.0], [0.0, 8.0]]
    )
    labels = torch.tensor([0, 0, 1, 1])
    loader = DataLoader(TensorDataset(logits, labels), batch_size=2)

    metrics = evaluate(
        torch.nn.Identity(), loader, torch.device("cpu"), class_count=2
    )

    assert metrics["samples"] == 4
    assert metrics["probability_calibration"]["temperature"] > 1.0
