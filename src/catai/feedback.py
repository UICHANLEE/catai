"""Privacy-preserving CashLog feedback export and active-learning curation."""

from __future__ import annotations

import csv
import hashlib
import hmac
import json
import math
import os
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ALLOWED_SOURCES = {"accepted_prediction", "top3_selection", "manual_edit"}
ALLOWED_REVIEW_STATUSES = {"pending", "approved", "rejected"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    os.chmod(path, mode)


def write_json(path: Path, payload: dict[str, Any], mode: int = 0o600) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.chmod(path, mode)


def load_leaf_ids(categories_path: Path) -> list[str]:
    rows = json.loads(categories_path.read_text(encoding="utf-8"))
    leaf_ids = [str(row["id"]) for row in rows]
    if len(leaf_ids) != 33 or len(set(leaf_ids)) != 33:
        raise ValueError("CashLog taxonomy must contain exactly 33 unique leaves")
    return leaf_ids


def _required_text(row: dict[str, Any], key: str, maximum: int = 256) -> str:
    value = str(row.get(key) or "").strip()
    if not value:
        raise ValueError(f"{key} is required")
    if len(value) > maximum:
        raise ValueError(f"{key} exceeds {maximum} characters")
    return value


def _canonical_uuid(value: str, key: str) -> str:
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as error:
        raise ValueError(f"{key} must be a UUID") from error


def _parse_timestamp(value: str, key: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as error:
        raise ValueError(f"{key} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{key} must include a timezone")
    return parsed.astimezone(timezone.utc).isoformat()


def _normalize_top3(value: Any, leaf_ids: set[str]) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not 1 <= len(value) <= 3:
        raise ValueError("predicted_top3 must contain one to three rows")
    output: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in value:
        if not isinstance(candidate, dict):
            raise ValueError("predicted_top3 rows must be objects")
        leaf_id = str(candidate.get("category") or candidate.get("leaf_id") or "")
        if leaf_id not in leaf_ids:
            raise ValueError(f"unknown predicted category: {leaf_id}")
        if leaf_id in seen:
            raise ValueError(f"duplicate predicted category: {leaf_id}")
        try:
            confidence = float(candidate["confidence"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("candidate confidence must be numeric") from error
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("candidate confidence must be between zero and one")
        seen.add(leaf_id)
        output.append({"category": leaf_id, "confidence": confidence})
    return sorted(output, key=lambda row: (-row["confidence"], row["category"]))


def normalize_feedback_row(
    row: dict[str, Any],
    *,
    leaf_ids: set[str],
    hmac_key: bytes,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Validate a Supabase row and return de-identified and restricted records."""

    if len(hmac_key) < 32:
        raise ValueError("feedback HMAC key must contain at least 32 bytes")
    schema_version = int(row.get("schema_version") or 1)
    if schema_version not in {1, 2}:
        raise ValueError(f"unsupported schema_version: {schema_version}")
    event_id = _canonical_uuid(_required_text(row, "event_id"), "event_id")
    sample_id = _canonical_uuid(_required_text(row, "sample_id"), "sample_id")
    user_id = _canonical_uuid(_required_text(row, "user_id"), "user_id")
    model_version = _required_text(row, "model_version", maximum=128)
    taxonomy_version = _required_text(row, "taxonomy_version", maximum=32)
    selected_leaf_id = _required_text(row, "selected_leaf_id", maximum=64)
    if selected_leaf_id not in leaf_ids:
        raise ValueError(f"unknown selected category: {selected_leaf_id}")
    top3 = _normalize_top3(row.get("predicted_top3"), leaf_ids)
    model_category = str(row.get("model_category") or top3[0]["category"])
    if model_category not in leaf_ids:
        raise ValueError(f"unknown model category: {model_category}")
    source = _required_text(row, "source", maximum=32)
    if source not in ALLOWED_SOURCES:
        raise ValueError(f"unsupported source: {source}")
    predicted_ids = [str(candidate["category"]) for candidate in top3]
    if source == "accepted_prediction" and selected_leaf_id != model_category:
        raise ValueError("accepted_prediction must keep the model category")
    if source == "top3_selection" and selected_leaf_id not in predicted_ids:
        raise ValueError("top3_selection must select a predicted candidate")
    review_status = str(row.get("review_status") or "pending")
    if review_status not in ALLOWED_REVIEW_STATUSES:
        raise ValueError(f"unsupported review_status: {review_status}")
    occurred_at = _parse_timestamp(
        _required_text(row, "occurred_at"), "occurred_at"
    )
    reviewed_at_value = row.get("reviewed_at")
    reviewed_at = (
        _parse_timestamp(str(reviewed_at_value), "reviewed_at")
        if reviewed_at_value
        else None
    )
    if review_status != "pending" and not reviewed_at:
        raise ValueError("reviewed_at is required after a review decision")
    if review_status == "pending" and reviewed_at:
        raise ValueError("pending feedback cannot have reviewed_at")

    image_consent_value = row.get("image_retention_consent")
    if not isinstance(image_consent_value, bool):
        raise ValueError("image_retention_consent must be boolean")
    image_consent = image_consent_value
    image_object_key = str(row.get("image_object_key") or "").strip() or None
    if not image_consent and image_object_key:
        raise ValueError("image path is forbidden without image retention consent")
    if image_object_key and not image_object_key.startswith(f"{user_id}/"):
        raise ValueError("image path does not belong to the event owner")

    group_id = hmac.new(hmac_key, user_id.encode(), hashlib.sha256).hexdigest()
    selected_rank = (
        predicted_ids.index(selected_leaf_id) + 1
        if selected_leaf_id in predicted_ids
        else None
    )
    safe_event = {
        "schema_version": 2,
        "event_id": event_id,
        "sample_id": sample_id,
        "group_id": group_id,
        "model_version": model_version,
        "taxonomy_version": taxonomy_version,
        "model_category": model_category,
        "predicted_top3": top3,
        "selected_leaf_id": selected_leaf_id,
        "selected_rank": selected_rank,
        "occurred_at": occurred_at,
        "source": source,
        "is_correction": selected_leaf_id != model_category,
        "image_retention_consent": image_consent,
        "has_retained_image": bool(image_consent and image_object_key),
        "review_status": review_status,
        "reviewed_at": reviewed_at,
    }
    secure_image = None
    if review_status != "rejected" and image_consent and image_object_key:
        secure_image = {
            "event_id": event_id,
            "sample_id": sample_id,
            "group_id": group_id,
            "leaf_id": selected_leaf_id,
            "image_object_key": image_object_key,
            "review_status": review_status,
        }
    return safe_event, secure_image


def normalize_feedback_rows(
    rows: Iterable[dict[str, Any]],
    *,
    leaf_ids: set[str],
    hmac_key: bytes,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    secure_images: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    sample_ids: set[str] = set()
    for index, row in enumerate(rows, start=1):
        try:
            event, secure_image = normalize_feedback_row(
                row, leaf_ids=leaf_ids, hmac_key=hmac_key
            )
            if event["event_id"] in event_ids:
                raise ValueError("duplicate event_id")
            if event["sample_id"] in sample_ids:
                raise ValueError("duplicate sample_id")
            event_ids.add(str(event["event_id"]))
            sample_ids.add(str(event["sample_id"]))
            events.append(event)
            if secure_image:
                secure_images.append(secure_image)
        except (TypeError, ValueError) as error:
            quarantine.append(
                {
                    "row": index,
                    "event_id": str(row.get("event_id") or "")[:64],
                    "error": str(error),
                }
            )
    events.sort(key=lambda row: (str(row["occurred_at"]), str(row["event_id"])))
    secure_images.sort(key=lambda row: str(row["event_id"]))
    return events, secure_images, quarantine


def export_feedback_release(
    rows: list[dict[str, Any]],
    *,
    output_dir: Path,
    leaf_ids: list[str],
    hmac_key: bytes,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"release directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    events, secure_images, quarantine = normalize_feedback_rows(
        rows, leaf_ids=set(leaf_ids), hmac_key=hmac_key
    )
    write_jsonl(output_dir / "events.jsonl", events)
    write_jsonl(output_dir / "secure_image_index.jsonl", secure_images)
    write_jsonl(output_dir / "quarantine.jsonl", quarantine)
    status_counts = Counter(str(row["review_status"]) for row in events)
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "source_rows": len(rows),
        "valid_events": len(events),
        "quarantined_events": len(quarantine),
        "consented_image_references": len(secure_images),
        "approved_image_references": sum(
            row["review_status"] == "approved" for row in secure_images
        ),
        "pending_image_references": sum(
            row["review_status"] == "pending" for row in secure_images
        ),
        "review_status_counts": dict(sorted(status_counts.items())),
        "contains_user_ids": False,
        "secure_image_index_mode": "0600",
    }
    write_json(output_dir / "export_summary.json", summary)
    return summary


def curate_feedback_release(
    events: list[dict[str, Any]],
    *,
    output_dir: Path,
    leaf_ids: list[str],
    taxonomy_version: str,
    minimum_images_per_leaf: int,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"curation directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(output_dir, 0o700)
    approved = [row for row in events if row.get("review_status") == "approved"]
    eligible = [
        row
        for row in approved
        if row.get("taxonomy_version") == taxonomy_version
        and bool(row.get("has_retained_image"))
    ]
    image_counts = Counter(str(row["selected_leaf_id"]) for row in eligible)
    max_count = max(image_counts.values(), default=1)
    candidates: list[dict[str, Any]] = []
    for row in eligible:
        top3 = list(row["predicted_top3"])
        top1 = float(top3[0]["confidence"])
        top2 = float(top3[1]["confidence"]) if len(top3) > 1 else 0.0
        margin = max(0.0, min(1.0, top1 - top2))
        scarcity = 1.0 - (image_counts[str(row["selected_leaf_id"])] / max_count)
        priority = (
            0.35 * float(bool(row["is_correction"]))
            + 0.25 * (1.0 - margin)
            + 0.20 * (1.0 - top1)
            + 0.20 * scarcity
        )
        candidates.append(
            {
                "schema_version": 1,
                "event_id": row["event_id"],
                "sample_id": row["sample_id"],
                "group_id": row["group_id"],
                "leaf_id": row["selected_leaf_id"],
                "model_version": row["model_version"],
                "source": row["source"],
                "owner_approved": True,
                "review_status": "approved",
                "priority_score": round(priority, 6),
                "predicted_top3": top3,
            }
        )
    candidates.sort(
        key=lambda row: (-float(row["priority_score"]), str(row["leaf_id"]), str(row["event_id"]))
    )
    underfilled = {
        leaf_id: image_counts.get(leaf_id, 0)
        for leaf_id in leaf_ids
        if image_counts.get(leaf_id, 0) < minimum_images_per_leaf
    }
    correction_count = sum(bool(row["is_correction"]) for row in approved)
    top3_hits = sum(
        row["selected_leaf_id"]
        in {candidate["category"] for candidate in row["predicted_top3"]}
        for row in approved
    )
    by_model: dict[str, dict[str, int]] = defaultdict(
        lambda: {"confirmed": 0, "corrections": 0}
    )
    for row in approved:
        values = by_model[str(row["model_version"])]
        values["confirmed"] += 1
        values["corrections"] += int(bool(row["is_correction"]))
    model_metrics = {
        model: {
            **values,
            "correction_rate": values["corrections"] / values["confirmed"],
        }
        for model, values in sorted(by_model.items())
    }
    summary = {
        "schema_version": 1,
        "generated_at": utc_now(),
        "taxonomy_version": taxonomy_version,
        "events": len(events),
        "pending_events": sum(row.get("review_status") == "pending" for row in events),
        "approved_events": len(approved),
        "rejected_events": sum(row.get("review_status") == "rejected" for row in events),
        "approved_image_candidates": len(candidates),
        "metadata_only_approved": sum(not row.get("has_retained_image") for row in approved),
        "correction_rate": correction_count / len(approved) if approved else None,
        "top3_coverage": top3_hits / len(approved) if approved else None,
        "minimum_images_per_leaf": minimum_images_per_leaf,
        "per_leaf_approved_images": {
            leaf_id: image_counts.get(leaf_id, 0) for leaf_id in leaf_ids
        },
        "underfilled_leaves": underfilled,
        "ready_for_training": not underfilled and bool(candidates),
        "model_metrics": model_metrics,
        "auto_training_allowed": False,
    }
    write_jsonl(output_dir / "training_candidates.jsonl", candidates)
    write_jsonl(output_dir / "approved_metadata_feedback.jsonl", approved)
    write_json(output_dir / "curation_summary.json", summary)
    per_leaf_path = output_dir / "per_leaf_feedback.csv"
    with per_leaf_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["leaf_id", "approved_events", "approved_images", "corrections"],
        )
        writer.writeheader()
        for leaf_id in leaf_ids:
            leaf_rows = [row for row in approved if row["selected_leaf_id"] == leaf_id]
            writer.writerow(
                {
                    "leaf_id": leaf_id,
                    "approved_events": len(leaf_rows),
                    "approved_images": image_counts.get(leaf_id, 0),
                    "corrections": sum(bool(row["is_correction"]) for row in leaf_rows),
                }
            )
    os.chmod(per_leaf_path, 0o600)
    return summary
