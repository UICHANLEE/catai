#!/usr/bin/env python3
"""Export the trained SigLIP2 vision path and linear head to quantized ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import onnx
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor
from onnxruntime.quantization import QuantType, quantize_dynamic


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/siglip2-base-patch16-224"
DEFAULT_HEAD = (
    ROOT
    / "checkpoints/cashlog33/vision_head_siglip_expanded_v1/"
    "vision_head.joblib"
)
DEFAULT_CONFIG = ROOT / "configs/cashlog/hybrid.siglip-expanded-candidate.json"
DEFAULT_OUTPUT = (
    ROOT
    / "checkpoints/cashlog33/vision_head_siglip_expanded_v1/onnx"
)
DEFAULT_TEST_MANIFEST = ROOT / "data/raw/cashlog33/openimages_v7/manifest.jsonl"
DEFAULT_SPLIT = (
    ROOT
    / "checkpoints/cashlog33/vision_head_originals_v2/split_manifest.jsonl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def feature_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    raise TypeError(f"unsupported feature output: {type(value)!r}")


def assignment_matrix(
    source_labels: list[str], target_labels: list[str]
) -> np.ndarray:
    target_index = {label: index for index, label in enumerate(target_labels)}
    matrix = np.zeros(
        (len(source_labels), len(target_labels)), dtype=np.float32
    )
    for source_index, label in enumerate(source_labels):
        if label not in target_index:
            raise ValueError(f"unknown target label: {label}")
        matrix[source_index, target_index[label]] = 1.0
    return matrix


def build_prompts(
    categories: list[dict[str, Any]], semantics: dict[str, Any]
) -> tuple[list[str], dict[str, list[int]]]:
    prompts: list[str] = []
    indexes: dict[str, list[int]] = defaultdict(list)
    for category in categories:
        leaf_id = str(category["id"])
        leaf = semantics["leaves"][leaf_id]
        values = [str(value) for value in leaf["queries"]]
        positive = [str(value) for value in leaf.get("positive_terms", [])]
        if positive:
            values.append(
                f"CashLog {category['display_name']} expense: "
                + ", ".join(positive[:8])
            )
        for value in values:
            indexes[leaf_id].append(len(prompts))
            prompts.append(value)
    return prompts, indexes


def prompt_assignment_matrix(
    prompt_indexes: dict[str, list[int]],
    labels: list[str],
    prompt_count: int,
) -> np.ndarray:
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((prompt_count, len(labels)), dtype=np.float32)
    for leaf_id, indexes in prompt_indexes.items():
        weight = 1.0 / len(indexes)
        for prompt_index in indexes:
            matrix[prompt_index, label_index[leaf_id]] = weight
    return matrix


class SiglipCashlogVision(nn.Module):
    def __init__(
        self,
        vision_model: nn.Module,
        prompt_features: np.ndarray,
        prompt_assignment: np.ndarray,
        head_coef: np.ndarray,
        head_intercept: np.ndarray,
        head_assignment: np.ndarray,
        logit_scale: float,
        logit_bias: float,
        zero_shot_weight: float,
        linear_head_weight: float,
        visual_adjustment: np.ndarray,
    ) -> None:
        super().__init__()
        self.vision_model = vision_model
        self.register_buffer(
            "prompt_features",
            torch.from_numpy(prompt_features.astype(np.float32)),
        )
        self.register_buffer(
            "prompt_assignment",
            torch.from_numpy(prompt_assignment.astype(np.float32)),
        )
        self.register_buffer(
            "head_coef",
            torch.from_numpy(head_coef.astype(np.float32)),
        )
        self.register_buffer(
            "head_intercept",
            torch.from_numpy(head_intercept.astype(np.float32)),
        )
        self.register_buffer(
            "head_assignment",
            torch.from_numpy(head_assignment.astype(np.float32)),
        )
        self.register_buffer(
            "visual_adjustment",
            torch.from_numpy(visual_adjustment.astype(np.float32)),
        )
        self.logit_scale = float(logit_scale)
        self.logit_bias = float(logit_bias)
        self.zero_shot_weight = float(zero_shot_weight)
        self.linear_head_weight = float(linear_head_weight)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.vision_model(pixel_values=pixel_values)
        features = F.normalize(output.pooler_output.float(), dim=-1)
        prompt_logits = (
            features @ self.prompt_features.T
        ) * self.logit_scale + self.logit_bias
        zero_shot = torch.sigmoid(prompt_logits) @ self.prompt_assignment
        zero_shot = zero_shot * self.visual_adjustment
        zero_shot = zero_shot / zero_shot.sum(dim=1, keepdim=True).clamp_min(
            1e-9
        )
        head_logits = features @ self.head_coef.T + self.head_intercept
        head = torch.softmax(head_logits, dim=1) @ self.head_assignment
        scores = (
            self.zero_shot_weight * zero_shot
            + self.linear_head_weight * head
        )
        return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-9)


def frozen_test_rows(manifest: Path, split: Path, limit: int) -> list[dict[str, Any]]:
    rows_by_id = {str(row["sample_id"]): row for row in read_jsonl(manifest)}
    test_ids = [
        str(row["sample_id"])
        for row in read_jsonl(split)
        if str(row.get("split")) == "test"
        and str(row["sample_id"]) in rows_by_id
    ]
    rows = [rows_by_id[sample_id] for sample_id in test_ids]
    return rows[:limit]


def artifact(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display_path = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        display_path = str(resolved)
    return {
        "path": display_path,
        "bytes": path.stat().st_size,
        "megabytes": path.stat().st_size / (1024 * 1024),
        "sha256": sha256_file(path),
    }


def encoder_layer_index(node_name: str) -> int | None:
    match = re.search(r"/vision_model/encoder/layers\.(\d+)/", node_name)
    return int(match.group(1)) if match else None


def quantization_exclusions(
    model_path: Path,
    keep_first_encoder_layers_fp32: int,
    keep_last_encoder_layers_fp32: int,
) -> list[str]:
    if not 0 <= keep_first_encoder_layers_fp32 <= 12:
        raise ValueError(
            "--keep-first-encoder-layers-fp32 must be between 0 and 12"
        )
    if not 0 <= keep_last_encoder_layers_fp32 <= 12:
        raise ValueError(
            "--keep-last-encoder-layers-fp32 must be between 0 and 12"
        )
    if (
        keep_first_encoder_layers_fp32 + keep_last_encoder_layers_fp32
        > 12
    ):
        raise ValueError("protected first and last encoder layers overlap")
    if (
        keep_first_encoder_layers_fp32 == 0
        and keep_last_encoder_layers_fp32 == 0
    ):
        return []
    first_fp32_layer = 12 - keep_last_encoder_layers_fp32
    model = onnx.load(str(model_path), load_external_data=False)
    return [
        node.name
        for node in model.graph.node
        if (layer_index := encoder_layer_index(node.name)) is not None
        and (
            layer_index < keep_first_encoder_layers_fp32
            or layer_index >= first_fp32_layer
        )
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--vision-head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--test-manifest", type=Path, default=DEFAULT_TEST_MANIFEST)
    parser.add_argument("--split", type=Path, default=DEFAULT_SPLIT)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--reuse-exported", action="store_true")
    parser.add_argument(
        "--keep-first-encoder-layers-fp32",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--keep-last-encoder-layers-fp32",
        type=int,
        default=0,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    categories_path = (args.config.parent / config["categories"]).resolve()
    semantics_path = (args.config.parent / config["semantics"]).resolve()
    categories = json.loads(categories_path.read_text(encoding="utf-8"))
    semantics = json.loads(semantics_path.read_text(encoding="utf-8"))
    labels = [str(category["id"]) for category in categories]

    model = AutoModel.from_pretrained(
        args.vision_model, local_files_only=True
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.vision_model, local_files_only=True, use_fast=True
    )
    prompts, prompt_indexes = build_prompts(categories, semantics)
    text_inputs = processor(
        text=prompts,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.inference_mode():
        prompt_features = feature_tensor(
            model.get_text_features(**text_inputs)
        )
        prompt_features = F.normalize(prompt_features.float(), dim=-1).numpy()

    head_artifact = joblib.load(args.vision_head)
    head = head_artifact["model"]
    if not all(hasattr(head, name) for name in ["coef_", "intercept_", "classes_"]):
        raise ValueError("ONNX export requires one fitted LogisticRegression head")
    head_classes = [str(value) for value in head.classes_]
    visual_adjustment = np.ones(len(labels), dtype=np.float32)
    visual_adjustment[labels.index("misc_uncat")] = 0.0
    visual_adjustment[labels.index("misc_other")] = 0.25
    blend = config["vision_blend"]
    wrapper = SiglipCashlogVision(
        model.vision_model,
        prompt_features,
        prompt_assignment_matrix(prompt_indexes, labels, len(prompts)),
        np.asarray(head.coef_, dtype=np.float32),
        np.asarray(head.intercept_, dtype=np.float32),
        assignment_matrix(head_classes, labels),
        float(model.logit_scale.exp().item()),
        float(model.logit_bias.item()),
        float(blend["zero_shot"]),
        float(blend["linear_head"]),
        visual_adjustment,
    ).eval()

    fp32_path = args.output_dir / "cashlog33-siglip-vision-fp32.onnx"
    int8_path = args.output_dir / "cashlog33-siglip-vision-int8.onnx"
    dummy = torch.zeros((1, 3, 224, 224), dtype=torch.float32)
    if not args.reuse_exported or not fp32_path.is_file():
        torch.onnx.export(
            wrapper,
            (dummy,),
            fp32_path,
            input_names=["pixel_values"],
            output_names=["vision_scores"],
            dynamic_axes={
                "pixel_values": {0: "batch"},
                "vision_scores": {0: "batch"},
            },
            opset_version=args.opset,
            do_constant_folding=True,
            dynamo=False,
        )
    onnx.checker.check_model(str(fp32_path))
    excluded_nodes = quantization_exclusions(
        fp32_path,
        args.keep_first_encoder_layers_fp32,
        args.keep_last_encoder_layers_fp32,
    )
    if (
        not args.reuse_exported
        or not int8_path.is_file()
        or args.keep_first_encoder_layers_fp32
        or args.keep_last_encoder_layers_fp32
    ):
        quantize_dynamic(
            model_input=str(fp32_path),
            model_output=str(int8_path),
            per_channel=True,
            reduce_range=False,
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
            nodes_to_exclude=excluded_nodes,
            use_external_data_format=False,
        )
    onnx.checker.check_model(str(int8_path))

    rows = frozen_test_rows(
        args.test_manifest, args.split, args.validation_samples
    )
    images = []
    for row in rows:
        with Image.open(ROOT / str(row["relative_path"])) as source:
            images.append(source.convert("RGB"))
    pixel_values = processor(images=images, return_tensors="pt")[
        "pixel_values"
    ]
    with torch.inference_mode():
        native = wrapper(pixel_values).numpy()
    fp32_session = ort.InferenceSession(
        str(fp32_path), providers=["CPUExecutionProvider"]
    )
    int8_session = ort.InferenceSession(
        str(int8_path), providers=["CPUExecutionProvider"]
    )
    fp32 = fp32_session.run(
        None, {"pixel_values": pixel_values.numpy()}
    )[0]
    int8 = int8_session.run(
        None, {"pixel_values": pixel_values.numpy()}
    )[0]
    timings = []
    sample = pixel_values[:1].numpy()
    for _ in range(10):
        int8_session.run(None, {"pixel_values": sample})
    for _ in range(50):
        started = time.perf_counter()
        int8_session.run(None, {"pixel_values": sample})
        timings.append((time.perf_counter() - started) * 1000.0)

    labels_path = args.output_dir / "labels.json"
    labels_path.write_text(
        json.dumps(categories, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "architecture": "siglip2-base-patch16-224-vision+cashlog-linear-head",
        "vision_model_sha256": sha256_file(
            args.vision_model / "model.safetensors"
        ),
        "vision_head_sha256": sha256_file(args.vision_head),
        "input": {
            "name": "pixel_values",
            "dtype": "float32",
            "shape": ["N", 3, 224, 224],
            "resize": [224, 224],
            "mean": [0.5, 0.5, 0.5],
            "std": [0.5, 0.5, 0.5],
        },
        "output": {
            "name": "vision_scores",
            "dtype": "float32",
            "shape": ["N", len(labels)],
            "kind": "probabilities",
            "labels": labels,
        },
        "quantization": {
            "method": "dynamic-weight-only",
            "format": "QOperator",
            "weight_type": "QInt8",
            "per_channel": True,
            "operators": ["MatMul", "Gemm"],
            "keep_first_encoder_layers_fp32": (
                args.keep_first_encoder_layers_fp32
            ),
            "keep_last_encoder_layers_fp32": (
                args.keep_last_encoder_layers_fp32
            ),
            "excluded_node_count": len(excluded_nodes),
        },
        "fp32": artifact(fp32_path),
        "int8": artifact(int8_path),
        "labels": artifact(labels_path),
        "validation": {
            "samples": len(rows),
            "native_fp32_top1_agreement": float(
                np.mean(native.argmax(axis=1) == fp32.argmax(axis=1))
            ),
            "fp32_int8_top1_agreement": float(
                np.mean(fp32.argmax(axis=1) == int8.argmax(axis=1))
            ),
            "native_fp32_mean_absolute_error": float(
                np.mean(np.abs(native - fp32))
            ),
            "fp32_int8_mean_absolute_error": float(
                np.mean(np.abs(fp32 - int8))
            ),
        },
        "runtime": {
            "provider": "CPUExecutionProvider",
            "single_image_p50_ms": float(np.percentile(timings, 50)),
            "single_image_p95_ms": float(np.percentile(timings, 95)),
            "scope": "model-only with preprocessed pixel_values",
        },
    }
    report_path = args.output_dir / "export_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
