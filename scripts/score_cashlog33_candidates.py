#!/usr/bin/env python3
"""Score weakly labeled CashLog candidates with SigLIP2 before training.

This is a triage step, not ground truth annotation. Automatically accepted rows
are locked to the training split so they cannot inflate validation/test metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.nn import functional as F
from transformers import AutoModel, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_SEMANTICS = ROOT / "configs/cashlog/leaf_semantics.json"
DEFAULT_MODEL = ROOT / "models/siglip2-base-patch16-224"
DEFAULT_INPUT = ROOT / "data/raw/cashlog33/openverse_v1/manifest.jsonl"
DEFAULT_OUTPUT = ROOT / "data/raw/cashlog33/openverse_v1/scored_manifest.jsonl"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def resolve_image_path(row: dict[str, Any]) -> Path:
    path = Path(str(row.get("relative_path") or row.get("path") or ""))
    if not str(path):
        raise ValueError("candidate has no image path")
    return path if path.is_absolute() else ROOT / path


def build_prompts(
    categories: list[dict[str, Any]], semantics: dict[str, Any]
) -> tuple[list[str], dict[str, list[int]]]:
    prompts: list[str] = []
    indexes: dict[str, list[int]] = defaultdict(list)
    for category in categories:
        leaf_id = str(category["id"])
        leaf = semantics["leaves"][leaf_id]
        candidates = list(leaf["queries"])
        positive_terms = [str(value) for value in leaf.get("positive_terms", [])]
        if positive_terms:
            candidates.append(
                f"CashLog {category['display_name']} expense: " + ", ".join(positive_terms[:8])
            )
        for prompt in candidates:
            indexes[leaf_id].append(len(prompts))
            prompts.append(str(prompt))
    return prompts, indexes


def pooled_features(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "pooler_output"):
        return value.pooler_output
    raise TypeError(f"unsupported model feature output: {type(value)!r}")


def encode_text(
    model: Any,
    processor: Any,
    prompts: list[str],
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(text=prompts, padding="max_length", truncation=True, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        features = pooled_features(model.get_text_features(**inputs))
    return F.normalize(features.float(), dim=-1)


def load_images(rows: list[dict[str, Any]]) -> tuple[list[Image.Image], list[str | None]]:
    images: list[Image.Image] = []
    errors: list[str | None] = []
    for row in rows:
        try:
            with Image.open(resolve_image_path(row)) as source:
                source.load()
                images.append(ImageOps.exif_transpose(source).convert("RGB"))
            errors.append(None)
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            images.append(Image.new("RGB", (224, 224), "black"))
            errors.append(f"{type(exc).__name__}: {exc}")
    return images, errors


def score_batch(
    model: Any,
    processor: Any,
    images: list[Image.Image],
    text_features: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.inference_mode():
        image_features = pooled_features(model.get_image_features(**inputs))
        image_features = F.normalize(image_features.float(), dim=-1)
        logits = image_features @ text_features.T
        logits = logits * model.logit_scale.exp().float() + model.logit_bias.float()
    return torch.sigmoid(logits).cpu()


def leaf_scores(
    prompt_scores: torch.Tensor,
    leaf_ids: list[str],
    prompt_indexes: dict[str, list[int]],
) -> dict[str, float]:
    return {
        leaf_id: float(prompt_scores[prompt_indexes[leaf_id]].mean().item())
        for leaf_id in leaf_ids
    }


def classify_review(
    row: dict[str, Any],
    scores: dict[str, float],
    min_score: float,
    min_margin: float,
    max_expected_rank: int,
) -> tuple[str, dict[str, Any]]:
    scores = dict(scores)
    scores["misc_uncat"] = 0.0
    scores["misc_other"] *= 0.25
    expected = str(row["leaf_id"])
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    expected_rank = next(index for index, item in enumerate(ranked, start=1) if item[0] == expected)
    expected_score = scores[expected]
    best_other_score = max(score for leaf_id, score in ranked if leaf_id != expected)
    margin = expected_score - best_other_score
    top3 = [{"leaf_id": leaf_id, "score": score} for leaf_id, score in ranked[:3]]

    existing = str(row.get("review_status") or "pending")
    if existing in {"approved", "trusted"}:
        status = existing
    elif (
        expected_rank <= max_expected_rank
        and expected_score >= min_score
        and margin >= min_margin
        and expected not in {"misc_uncat", "misc_other"}
    ):
        status = "auto_approved"
    elif expected_rank > 3 or expected_score < min_score / 2:
        status = "auto_rejected"
    else:
        status = "pending"
    details = {
        "expected_leaf_id": expected,
        "expected_rank": expected_rank,
        "expected_score": expected_score,
        "best_other_score": best_other_score,
        "expected_margin": margin,
        "top3": top3,
        "scores": scores,
    }
    return status, details


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--semantics", type=Path, default=DEFAULT_SEMANTICS)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--min-score", type=float, default=0.04)
    parser.add_argument("--min-margin", type=float, default=0.01)
    parser.add_argument("--max-expected-rank", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if not 1 <= args.max_expected_rank <= 3:
        parser.error("--max-expected-rank must be between 1 and 3")
    return args


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    categories = load_json(args.categories)
    semantics = load_json(args.semantics)
    leaf_ids = [str(row["id"]) for row in categories]
    if len(leaf_ids) != 33 or set(leaf_ids) != set(semantics["leaves"]):
        raise SystemExit("category and semantic configs must contain the same 33 leaves")
    rows = read_jsonl(args.input_manifest.resolve())
    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise SystemExit("input manifest has no rows")

    device = choose_device(args.device)
    model = AutoModel.from_pretrained(args.model.resolve(), local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        args.model.resolve(), local_files_only=True, use_fast=True
    )
    prompts, prompt_indexes = build_prompts(categories, semantics)
    text_features = encode_text(model, processor, prompts, device)
    model_sha256 = hashlib.sha256((args.model / "model.safetensors").read_bytes()).hexdigest()

    output: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    leaf_status_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scored_at = utc_now()
    for start in range(0, len(rows), args.batch_size):
        batch_rows = rows[start : start + args.batch_size]
        images, image_errors = load_images(batch_rows)
        batch_scores = score_batch(model, processor, images, text_features, device)
        for row, prompt_scores, image_error in zip(
            batch_rows, batch_scores, image_errors, strict=True
        ):
            if image_error:
                status = "auto_rejected"
                details = {"error": image_error, "top3": []}
            else:
                scores = leaf_scores(prompt_scores, leaf_ids, prompt_indexes)
                status, details = classify_review(
                    row,
                    scores,
                    args.min_score,
                    args.min_margin,
                    args.max_expected_rank,
                )
            scored_row = {
                **row,
                "review_status": status,
                "review_method": "siglip2_prompt_ensemble_v1",
                "reviewed_at": scored_at,
                "review_details": details,
            }
            if status == "auto_approved":
                scored_row["split_lock"] = "train"
            elif str(row.get("split_lock") or "") == "train" and status != "approved":
                scored_row.pop("split_lock", None)
            output.append(scored_row)
            status_counts[status] += 1
            leaf_status_counts[str(row["leaf_id"])][status] += 1
        print(f"scored {min(start + args.batch_size, len(rows))}/{len(rows)}", flush=True)

    write_jsonl(args.output_manifest.resolve(), output)
    summary = {
        "schema_version": 1,
        "generated_at": scored_at,
        "input_manifest": str(args.input_manifest.resolve()),
        "output_manifest": str(args.output_manifest.resolve()),
        "model": str(args.model.resolve()),
        "model_sha256": model_sha256,
        "device": str(device),
        "samples": len(output),
        "thresholds": {
            "min_score": args.min_score,
            "min_margin": args.min_margin,
            "max_expected_rank": args.max_expected_rank,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "leaf_status_counts": {
            leaf_id: dict(sorted(leaf_status_counts[leaf_id].items())) for leaf_id in leaf_ids
        },
        "evaluation_policy": "auto_approved rows are locked to train and never establish holdout accuracy",
    }
    summary_path = args.output_manifest.with_name("scoring_summary.json").resolve()
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
