from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "data/raw/cashlog33/openimages_v7/manifest.jsonl"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_ARTIFACT_DIR = ROOT / "checkpoints/cashlog33/airflow_latest/vision"
DEFAULT_OUTPUT = (
    ROOT / "data/processed/cashlog33/predeploy_review/current_model_scored_manifest.jsonl"
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(value)
    return rows


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_taxonomy(path: Path) -> list[str]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    leaf_ids = [str(row["id"]) for row in rows]
    if len(leaf_ids) != 33 or len(set(leaf_ids)) != 33:
        raise ValueError("categories must contain exactly 33 unique leaf ids")
    return leaf_ids


def original_embeddings_by_split(
    cache: Any,
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    import numpy as np

    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in split_rows:
        split = str(row.get("split") or "")
        if split not in {"train", "val", "test"}:
            raise ValueError(f"invalid dataset split: {split}")
        rows_by_split[split].append(row)

    output: dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        rows = rows_by_split[split]
        embeddings = cache[split]
        labels = cache[f"{split}_labels"]
        if not rows:
            if len(embeddings):
                raise ValueError(f"embedding cache contains unexpected {split} rows")
            output[split] = embeddings
            continue
        if len(embeddings) % len(rows):
            raise ValueError(
                f"{split} embeddings cannot be aligned to split manifest: "
                f"{len(embeddings)} embeddings for {len(rows)} samples"
            )
        views_per_sample = len(embeddings) // len(rows)
        if views_per_sample < 1:
            raise ValueError(f"{split} has no embeddings")
        label_matrix = np.asarray(labels).reshape(len(rows), views_per_sample)
        for index, row in enumerate(rows):
            expected = str(row["leaf_id"])
            if any(str(value) != expected for value in label_matrix[index]):
                raise ValueError(f"embedding label mismatch for {row['sample_id']}")
        output[split] = np.asarray(embeddings)[::views_per_sample]
    return output


def build_predeploy_queue(
    *,
    manifest_path: Path = DEFAULT_MANIFEST,
    categories_path: Path = DEFAULT_CATEGORIES,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    output_path: Path = DEFAULT_OUTPUT,
    confidence_threshold: float = 0.60,
    margin_threshold: float = 0.15,
) -> dict[str, Any]:
    import joblib
    import numpy as np

    manifest_path = manifest_path.resolve()
    categories_path = categories_path.resolve()
    artifact_dir = artifact_dir.resolve()
    output_path = output_path.resolve()
    if output_path == manifest_path:
        raise ValueError("output manifest must not overwrite the source manifest")
    if not 0.0 <= confidence_threshold <= 1.0:
        raise ValueError("confidence_threshold must be between 0 and 1")
    if not 0.0 <= margin_threshold <= 1.0:
        raise ValueError("margin_threshold must be between 0 and 1")

    taxonomy = load_taxonomy(categories_path)
    taxonomy_set = set(taxonomy)
    manifest_rows = read_jsonl(manifest_path)
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in manifest_rows:
        sample_id = str(row.get("sample_id") or "")
        leaf_id = str(row.get("leaf_id") or "")
        if not sample_id or sample_id in rows_by_id:
            raise ValueError(f"missing or duplicate sample_id: {sample_id}")
        if leaf_id not in taxonomy_set:
            raise ValueError(f"unknown leaf_id for {sample_id}: {leaf_id}")
        rows_by_id[sample_id] = row

    split_path = artifact_dir / "split_manifest.jsonl"
    cache_path = artifact_dir / "embedding_cache.npz"
    head_path = artifact_dir / "vision_head.joblib"
    split_rows = read_jsonl(split_path)
    split_ids = [str(row.get("sample_id") or "") for row in split_rows]
    if len(split_ids) != len(set(split_ids)):
        raise ValueError("split manifest contains duplicate sample ids")
    if set(split_ids) != set(rows_by_id):
        missing = sorted(set(rows_by_id) - set(split_ids))[:5]
        unexpected = sorted(set(split_ids) - set(rows_by_id))[:5]
        raise ValueError(
            f"manifest and split sample ids differ; missing={missing}, unexpected={unexpected}"
        )
    for row in split_rows:
        sample_id = str(row["sample_id"])
        if str(row["leaf_id"]) != str(rows_by_id[sample_id]["leaf_id"]):
            raise ValueError(f"split label differs from manifest for {sample_id}")

    artifact = joblib.load(head_path)
    model = artifact.get("model")
    classes = [str(value) for value in artifact.get("classes") or []]
    if model is None or classes != [str(value) for value in model.classes_]:
        raise ValueError("vision head artifact has an invalid model/classes contract")
    if not classes or not set(classes).issubset(taxonomy_set):
        raise ValueError("vision head classes must be a non-empty subset of the 33 leaves")

    with np.load(cache_path, allow_pickle=False) as cache:
        embeddings_by_split = original_embeddings_by_split(cache, split_rows)

    scored_by_id: dict[str, dict[str, Any]] = {}
    status_counts: Counter[str] = Counter()
    split_counts: dict[str, Counter[str]] = defaultdict(Counter)
    generated_at = utc_now()
    for split in ["train", "val", "test"]:
        current_rows = [row for row in split_rows if row["split"] == split]
        embeddings = embeddings_by_split[split]
        probabilities = model.predict_proba(embeddings)
        if probabilities.shape != (len(current_rows), len(classes)):
            raise ValueError(f"unexpected probability shape for {split}: {probabilities.shape}")
        top_count = min(3, len(classes))
        top_indexes = np.argsort(-probabilities, axis=1)[:, :top_count]
        for row, probability_row, indexes in zip(
            current_rows, probabilities, top_indexes, strict=True
        ):
            sample_id = str(row["sample_id"])
            expected = str(row["leaf_id"])
            top3 = [
                {"leaf_id": classes[int(index)], "score": float(probability_row[int(index)])}
                for index in indexes
            ]
            predicted = top3[0]["leaf_id"]
            confidence = top3[0]["score"]
            margin = confidence - (top3[1]["score"] if len(top3) > 1 else 0.0)
            expected_index = classes.index(expected) if expected in classes else None
            expected_score = (
                float(probability_row[expected_index]) if expected_index is not None else None
            )
            expected_rank = (
                int(np.where(np.argsort(-probability_row) == expected_index)[0][0]) + 1
                if expected_index is not None
                else None
            )
            mismatch = predicted != expected
            uncertain = confidence < confidence_threshold or margin < margin_threshold
            if mismatch:
                status = "model_mismatch"
            elif uncertain:
                status = "model_uncertain"
            else:
                status = "model_match"
            output = {
                **rows_by_id[sample_id],
                "dataset_split": split,
                "model_version": f"siglip2-linear-head:{sha256_file(head_path)[:16]}",
                "model_review_status": status,
                "model_review_method": "trained_siglip2_linear_head_v1",
                "model_scored_at": generated_at,
                "model_review_details": {
                    "expected_leaf_id": expected,
                    "expected_rank": expected_rank,
                    "expected_score": expected_score,
                    "top1_confidence": confidence,
                    "top1_margin": margin,
                    "top3": top3,
                },
                "need_user_check": mismatch or uncertain,
            }
            scored_by_id[sample_id] = output
            status_counts[status] += 1
            split_counts[split][status] += 1

    output_rows = [scored_by_id[str(row["sample_id"])] for row in manifest_rows]
    atomic_write_jsonl(output_path, output_rows)
    summary = {
        "schema_version": 1,
        "generated_at": generated_at,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "vision_head": str(head_path),
        "vision_head_sha256": sha256_file(head_path),
        "embedding_cache": str(cache_path),
        "output_manifest": str(output_path),
        "samples": len(output_rows),
        "supported_leaf_count": len(classes),
        "taxonomy_leaf_count": len(taxonomy),
        "thresholds": {
            "confidence": confidence_threshold,
            "margin": margin_threshold,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "split_status_counts": {
            split: dict(sorted(values.items())) for split, values in sorted(split_counts.items())
        },
        "review_policy": (
            "Human-reviewed error-mining samples must be locked to train; deployment evaluation "
            "requires an untouched holdout."
        ),
    }
    summary_path = output_path.with_name("current_model_scoring_summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a human-review queue from the currently trained CashLog vision head."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-manifest", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--confidence-threshold", type=float, default=0.60)
    parser.add_argument("--margin-threshold", type=float, default=0.15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = build_predeploy_queue(
        manifest_path=args.manifest,
        categories_path=args.categories,
        artifact_dir=args.artifact_dir,
        output_path=args.output_manifest,
        confidence_threshold=args.confidence_threshold,
        margin_threshold=args.margin_threshold,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
