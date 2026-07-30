from scripts.compare_cashlog33_siglip_onnx import build_comparison


def metric(top1, macro_f1, recall, p50):
    return {
        "samples": 100,
        "top1_accuracy": top1,
        "top3_accuracy": top1 + 0.1,
        "macro_f1": macro_f1,
        "per_leaf_recall": {"leaf": recall},
        "per_leaf_support": {"leaf": 100},
        "runtime": {"p50_vision_ms": p50},
    }


def test_same_architecture_gate_requires_native_and_int8_improvement(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"int8")
    report = build_comparison(
        {"baseline": metric(0.70, 0.60, 0.70, 50.0)},
        {"baseline": metric(0.75, 0.65, 0.72, 50.0)},
        {"baseline": metric(0.748, 0.645, 0.71, 20.0)},
        {"quantization": {"method": "dynamic-weight-only"}},
        serving_config_sha256="serving-sha",
        model_path=model,
        max_recall_regression=0.10,
        minimum_recall_support=30,
        max_quantized_top1_drift=0.005,
        max_quantized_macro_f1_drift=0.01,
    )

    assert report["promotion_gate_passed"]
    assert report["native_delta_vs_current"]["top1_accuracy"] > 0
    assert report["int8_delta_vs_current"]["top1_accuracy"] > 0


def test_quantization_regression_blocks_promotion(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"int8")
    report = build_comparison(
        {"baseline": metric(0.70, 0.60, 0.70, 50.0)},
        {"baseline": metric(0.75, 0.65, 0.72, 50.0)},
        {"baseline": metric(0.72, 0.61, 0.70, 20.0)},
        {},
        serving_config_sha256="serving-sha",
        model_path=model,
        max_recall_regression=0.10,
        minimum_recall_support=30,
        max_quantized_top1_drift=0.005,
        max_quantized_macro_f1_drift=0.01,
    )

    assert not report["promotion_gate_passed"]
    assert not report["gate_checks"]["int8_top1_drift_within_limit"]
