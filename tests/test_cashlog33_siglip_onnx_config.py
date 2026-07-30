import hashlib
import json

from scripts.build_cashlog33_siglip_onnx_config import build_config


def test_siglip_onnx_config_pins_quantized_probability_model(tmp_path):
    base = tmp_path / "base.json"
    model = tmp_path / "model.onnx"
    labels = tmp_path / "labels.json"
    report = tmp_path / "report.json"
    output = tmp_path / "candidate.json"
    dependency = tmp_path / "dependency.json"
    dependency.write_text("{}", encoding="utf-8")
    model.write_bytes(b"siglip-int8")
    labels.write_text("[]", encoding="utf-8")
    report.write_text(
        json.dumps(
            {
                "int8": {
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest()
                },
                "labels": {
                    "sha256": hashlib.sha256(labels.read_bytes()).hexdigest()
                },
                "quantization": {"method": "dynamic-weight-only"},
            }
        ),
        encoding="utf-8",
    )
    base.write_text(
        json.dumps(
            {
                "model_version": "old",
                "categories": "dependency.json",
                "meal_specialist": {
                    "enabled": True,
                    "checkpoint": "dependency.json",
                    "labels": "dependency.json",
                },
            }
        ),
        encoding="utf-8",
    )

    config = build_config(
        base, model, labels, report, output, "new-int8"
    )

    assert config["model_version"] == "new-int8"
    assert config["compact_vision"]["runtime"] == "onnxruntime-int8"
    assert config["compact_vision"]["preprocess"] == "siglip_resize"
    assert config["compact_vision"]["output_kind"] == "probabilities"
    assert config["compact_vision"]["model_sha256"] == hashlib.sha256(
        model.read_bytes()
    ).hexdigest()
