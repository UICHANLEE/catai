from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import threading
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
PACKAGE_DIR = Path(__file__).resolve().parent
LABELER_ASSET_DIR = PACKAGE_DIR / "labeler"
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_INPUT = ROOT / "data/raw/cashlog33/openimages_v7/scored_manifest.jsonl"
CURRENT_MODEL_INPUT = (
    ROOT / "data/processed/cashlog33/predeploy_review/current_model_scored_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/cashlog33/predeploy_review/v1"
ACTUAL_INPUT = ROOT / "data/raw/cashlog33/actual/manifest.jsonl"
ACTUAL_OUTPUT_DIR = ROOT / "data/processed/cashlog33/actual_review/v1"
ALLOWED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}
TRAINING_REVIEW_STATUSES = {"approved", "auto_approved", "source_mapped", "trusted"}
LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def append_audit_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


def load_categories(path: Path) -> list[dict[str, str]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("categories must be a JSON array")
    categories: list[dict[str, str]] = []
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("category rows must be objects")
        categories.append(
            {
                "id": str(row.get("id") or ""),
                "group_id": str(row.get("group_id") or ""),
                "group_name": str(row.get("group_name") or ""),
                "display_name": str(row.get("display_name") or ""),
            }
        )
    leaf_ids = [row["id"] for row in categories]
    if len(leaf_ids) != 33 or len(set(leaf_ids)) != 33 or any(not value for value in leaf_ids):
        raise ValueError("categories must contain exactly 33 unique leaf ids")
    return categories


def parse_top3(row: dict[str, Any], leaf_ids: set[str]) -> list[dict[str, Any]]:
    details = row.get("model_review_details") or row.get("review_details")
    details = details if isinstance(details, dict) else {}
    raw_top3 = details.get("top3") or row.get("top3") or []
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw_top3):
        if isinstance(value, str):
            leaf_id = value
            score = row.get("confidence") if index == 0 else None
        elif isinstance(value, dict):
            leaf_id = str(value.get("leaf_id") or value.get("category") or "")
            score = value.get("score", value.get("confidence"))
        else:
            continue
        if leaf_id not in leaf_ids or leaf_id in seen:
            continue
        try:
            normalized_score = float(score) if score is not None else None
        except (TypeError, ValueError):
            normalized_score = None
        candidates.append({"leaf_id": leaf_id, "score": normalized_score})
        seen.add(leaf_id)
        if len(candidates) == 3:
            break
    if not candidates:
        predicted = str(row.get("predicted") or "")
        if predicted in leaf_ids:
            try:
                confidence = float(row.get("confidence"))
            except (TypeError, ValueError):
                confidence = None
            candidates.append({"leaf_id": predicted, "score": confidence})
    return candidates


def priority_score(sample: dict[str, Any]) -> float:
    score = 0.0
    if sample["mismatch"]:
        score += 1000.0
    status = sample["review_status"]
    if status == "auto_rejected":
        score += 500.0
    elif status == "pending":
        score += 300.0
    if sample["need_user_check"]:
        score += 100.0
    margin = sample.get("margin")
    if margin is not None:
        score += max(0.0, 1.0 - float(margin)) * 10.0
    confidence = sample.get("confidence")
    if confidence is not None:
        score += max(0.0, 1.0 - float(confidence))
    return score


class DecisionPayload(BaseModel):
    decision: Literal["approved", "rejected", "cleared"]
    selected_leaf_id: str | None = None
    note: str = Field(default="", max_length=1000)
    expected_revision: int = Field(default=0, ge=0)


class RevisionConflict(RuntimeError):
    pass


class ReviewRepository:
    def __init__(
        self,
        *,
        input_manifest: Path = DEFAULT_INPUT,
        categories_path: Path = DEFAULT_CATEGORIES,
        output_dir: Path = DEFAULT_OUTPUT_DIR,
        image_roots: list[Path] | None = None,
        dataset_kind: Literal["proxy", "actual"] = "proxy",
    ) -> None:
        self.input_manifest = input_manifest.resolve()
        self.categories_path = categories_path.resolve()
        self.output_dir = output_dir.resolve()
        if dataset_kind not in {"proxy", "actual"}:
            raise ValueError("dataset_kind must be proxy or actual")
        self.dataset_kind = dataset_kind
        if self.output_dir == self.input_manifest.parent:
            raise ValueError("output directory must not overwrite the source manifest directory")
        self.decisions_path = self.output_dir / "decisions.jsonl"
        self.audit_path = self.output_dir / "decision_audit.jsonl"
        self.manifest_sha256 = sha256_file(self.input_manifest)
        self.categories = load_categories(self.categories_path)
        self.category_by_id = {row["id"]: row for row in self.categories}
        self.leaf_ids = set(self.category_by_id)
        self.image_roots = [path.resolve() for path in (image_roots or [ROOT])]
        self._rows = read_jsonl(self.input_manifest)
        self._lock = threading.RLock()
        self._samples: dict[str, dict[str, Any]] = {}
        self._sample_key_to_id: dict[str, str] = {}
        self._build_samples()
        self._decisions = self._load_decisions()

    def _resolve_image_path(self, row: dict[str, Any]) -> Path | None:
        raw_path = row.get("relative_path") or row.get("image_path") or row.get("path")
        if not raw_path:
            return None
        path = Path(str(raw_path))
        resolved = (path if path.is_absolute() else ROOT / path).resolve()
        if resolved.suffix.lower() not in ALLOWED_IMAGE_SUFFIXES:
            return None
        if not any(resolved.is_relative_to(root) for root in self.image_roots):
            return None
        return resolved

    def _build_samples(self) -> None:
        for row in self._rows:
            sample_id = str(row.get("sample_id") or "").strip()
            original_leaf_id = str(row.get("leaf_id") or "").strip()
            if not sample_id:
                raise ValueError("every manifest row must contain sample_id")
            if sample_id in self._samples:
                raise ValueError(f"duplicate sample_id: {sample_id}")
            if original_leaf_id not in self.leaf_ids:
                raise ValueError(f"unknown leaf_id for {sample_id}: {original_leaf_id}")
            top3 = parse_top3(row, self.leaf_ids)
            predicted_leaf_id = top3[0]["leaf_id"] if top3 else None
            confidence = top3[0]["score"] if top3 else None
            margin = None
            if len(top3) >= 2 and top3[0]["score"] is not None and top3[1]["score"] is not None:
                margin = float(top3[0]["score"]) - float(top3[1]["score"])
            details = row.get("model_review_details") or row.get("review_details")
            details = details if isinstance(details, dict) else {}
            expected_margin = details.get("expected_margin")
            try:
                expected_margin = float(expected_margin) if expected_margin is not None else margin
            except (TypeError, ValueError):
                expected_margin = margin
            mismatch = predicted_leaf_id is not None and predicted_leaf_id != original_leaf_id
            review_status = str(
                row.get("model_review_status") or row.get("review_status") or "pending"
            )
            image_path = self._resolve_image_path(row)
            sample_key = hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]
            if sample_key in self._sample_key_to_id:
                raise ValueError(f"sample key collision: {sample_id}")
            sample = {
                "sample_id": sample_id,
                "sample_key": sample_key,
                "original_leaf_id": original_leaf_id,
                "predicted_leaf_id": predicted_leaf_id,
                "top3": top3,
                "confidence": confidence,
                "margin": expected_margin,
                "mismatch": mismatch,
                "review_status": review_status,
                "need_user_check": bool(row.get("need_user_check"))
                or mismatch
                or review_status in {"pending", "auto_rejected"},
                "image_path": image_path,
                "image_sha256": str(row.get("sha256") or ""),
                "source": str(row.get("source") or row.get("provider") or "unknown"),
                "title": str(row.get("title") or ""),
                "query": str(row.get("query") or ""),
                "license": str(row.get("license") or ""),
                "tags": [str(value) for value in (row.get("tags") or [])[:12]],
                "metadata_text": str(row.get("metadata_text") or "")[:1000],
                "dataset_split": str(row.get("dataset_split") or ""),
                "model_version": str(row.get("model_version") or ""),
                "row": row,
            }
            sample["priority"] = priority_score(sample)
            self._samples[sample_id] = sample
            self._sample_key_to_id[sample_key] = sample_id

    def _load_decisions(self) -> dict[str, dict[str, Any]]:
        if not self.decisions_path.exists():
            return {}
        decisions: dict[str, dict[str, Any]] = {}
        for row in read_jsonl(self.decisions_path):
            sample_id = str(row.get("sample_id") or "")
            if sample_id not in self._samples:
                raise ValueError(
                    f"decision references a sample not present in the input manifest: {sample_id}"
                )
            if sample_id in decisions:
                raise ValueError(f"duplicate decision for sample: {sample_id}")
            decision = str(row.get("decision") or "")
            selected_leaf_id = row.get("selected_leaf_id")
            if decision not in {"approved", "rejected"}:
                raise ValueError(f"invalid decision for {sample_id}: {decision}")
            if decision == "approved" and selected_leaf_id not in self.leaf_ids:
                raise ValueError(f"invalid selected leaf for {sample_id}: {selected_leaf_id}")
            current_sha = self._samples[sample_id]["image_sha256"]
            decision_sha = str(row.get("image_sha256") or "")
            if current_sha and decision_sha and not secrets.compare_digest(current_sha, decision_sha):
                raise ValueError(f"image content changed for reviewed sample: {sample_id}")
            decisions[sample_id] = row
        return decisions

    def _display_top3(self, sample: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                **candidate,
                "display_name": self.category_by_id[candidate["leaf_id"]]["display_name"],
                "group_name": self.category_by_id[candidate["leaf_id"]]["group_name"],
            }
            for candidate in sample["top3"]
        ]

    def serialize_sample(self, sample_id: str) -> dict[str, Any]:
        sample = self._samples[sample_id]
        decision = self._decisions.get(sample_id)
        original = self.category_by_id[sample["original_leaf_id"]]
        predicted = (
            self.category_by_id[sample["predicted_leaf_id"]]
            if sample["predicted_leaf_id"]
            else None
        )
        return {
            "sample_id": sample_id,
            "sample_key": sample["sample_key"],
            "original_leaf_id": sample["original_leaf_id"],
            "original_display_name": original["display_name"],
            "original_group_name": original["group_name"],
            "predicted_leaf_id": sample["predicted_leaf_id"],
            "predicted_display_name": predicted["display_name"] if predicted else None,
            "predicted_group_name": predicted["group_name"] if predicted else None,
            "top3": self._display_top3(sample),
            "confidence": sample["confidence"],
            "margin": sample["margin"],
            "mismatch": sample["mismatch"],
            "review_status": sample["review_status"],
            "need_user_check": sample["need_user_check"],
            "priority": sample["priority"],
            "image_url": f"/api/images/{sample['sample_key']}"
            if sample["image_path"] and sample["image_path"].is_file()
            else None,
            "source": sample["source"],
            "title": sample["title"],
            "query": sample["query"],
            "license": sample["license"],
            "tags": sample["tags"],
            "metadata_text": sample["metadata_text"],
            "dataset_split": sample["dataset_split"],
            "model_version": sample["model_version"],
            "decision": decision,
        }

    def summary(self) -> dict[str, Any]:
        decisions = list(self._decisions.values())
        approved = [row for row in decisions if row["decision"] == "approved"]
        corrected = [
            row
            for row in approved
            if row["selected_leaf_id"] != self._samples[row["sample_id"]]["original_leaf_id"]
        ]
        mismatch_ids = {
            sample_id for sample_id, sample in self._samples.items() if sample["mismatch"]
        }
        reviewed_ids = set(self._decisions)
        status_counts = Counter(sample["review_status"] for sample in self._samples.values())
        return {
            "total": len(self._samples),
            "mismatches": len(mismatch_ids),
            "mismatch_remaining": len(mismatch_ids - reviewed_ids),
            "reviewed": len(decisions),
            "approved": len(approved),
            "confirmed": len(approved) - len(corrected),
            "corrected": len(corrected),
            "rejected": sum(row["decision"] == "rejected" for row in decisions),
            "remaining": len(self._samples) - len(decisions),
            "source_status_counts": dict(sorted(status_counts.items())),
        }

    def state(self) -> dict[str, Any]:
        ordered = sorted(
            self._samples,
            key=lambda sample_id: (-self._samples[sample_id]["priority"], sample_id),
        )
        return {
            "schema_version": 1,
            "dataset_kind": self.dataset_kind,
            "default_mode": "unreviewed" if self.dataset_kind == "actual" else "errors",
            "ui_title": (
                "실데이터 라벨 검수"
                if self.dataset_kind == "actual"
                else "배포 전 데이터 검수"
            ),
            "input_manifest": str(self.input_manifest),
            "manifest_sha256": self.manifest_sha256,
            "output_dir": str(self.output_dir),
            "categories": self.categories,
            "summary": self.summary(),
            "samples": [self.serialize_sample(sample_id) for sample_id in ordered],
        }

    def image_path_for_key(self, sample_key: str) -> Path:
        sample_id = self._sample_key_to_id.get(sample_key)
        if sample_id is None:
            raise KeyError(sample_key)
        path = self._samples[sample_id]["image_path"]
        if path is None or not path.is_file():
            raise FileNotFoundError(sample_id)
        return path

    def decide(self, sample_id: str, payload: DecisionPayload) -> dict[str, Any] | None:
        if sample_id not in self._samples:
            raise KeyError(sample_id)
        note = payload.note.strip()
        with self._lock:
            current = self._decisions.get(sample_id)
            current_revision = int(current.get("revision", 0)) if current else 0
            if payload.expected_revision != current_revision:
                raise RevisionConflict(
                    f"expected revision {payload.expected_revision}, current revision is {current_revision}"
                )
            if payload.decision == "approved" and payload.selected_leaf_id not in self.leaf_ids:
                raise ValueError("approved decisions require a valid selected_leaf_id")
            if payload.decision != "approved" and payload.selected_leaf_id is not None:
                raise ValueError("only approved decisions may include selected_leaf_id")
            revision = current_revision + 1
            event = {
                "schema_version": 1,
                "event_id": str(uuid.uuid4()),
                "manifest_sha256": self.manifest_sha256,
                "sample_id": sample_id,
                "image_sha256": self._samples[sample_id]["image_sha256"],
                "decision": payload.decision,
                "original_leaf_id": self._samples[sample_id]["original_leaf_id"],
                "selected_leaf_id": payload.selected_leaf_id,
                "note": note,
                "revision": revision,
                "reviewed_at": utc_now(),
            }
            if payload.decision == "cleared":
                self._decisions.pop(sample_id, None)
                result = None
            else:
                self._decisions[sample_id] = event
                result = event
            atomic_write_jsonl(
                self.decisions_path,
                [self._decisions[key] for key in sorted(self._decisions)],
            )
            append_audit_event(self.audit_path, event)
            return result

    def export(self) -> dict[str, Any]:
        with self._lock:
            reviewed_rows: list[dict[str, Any]] = []
            training_rows: list[dict[str, Any]] = []
            human_verified_rows: list[dict[str, Any]] = []
            for row in self._rows:
                sample_id = str(row["sample_id"])
                decision = self._decisions.get(sample_id)
                output = dict(row)
                if decision:
                    output["predeploy_original_leaf_id"] = str(row["leaf_id"])
                    output["review_method"] = (
                        "human_actual_labeler_v1"
                        if self.dataset_kind == "actual"
                        else "human_predeploy_labeler_v1"
                    )
                    output["reviewed_at"] = decision["reviewed_at"]
                    output["human_review"] = {
                        "event_id": decision["event_id"],
                        "decision": decision["decision"],
                        "revision": decision["revision"],
                        "note": decision["note"],
                    }
                    if decision["decision"] == "approved":
                        output["leaf_id"] = decision["selected_leaf_id"]
                        output["review_status"] = "approved"
                        # These samples were selected after inspecting current-model errors.
                        output["split_lock"] = "train"
                        human_verified_rows.append(output)
                    else:
                        output["review_status"] = "rejected"
                        output.pop("split_lock", None)
                reviewed_rows.append(output)
                if str(output.get("review_status") or "") in TRAINING_REVIEW_STATUSES:
                    training_rows.append(output)

            corrected_path = self.output_dir / "corrected_manifest.jsonl"
            training_path = self.output_dir / "training_manifest.jsonl"
            verified_path = self.output_dir / "human_verified_manifest.jsonl"
            atomic_write_jsonl(corrected_path, reviewed_rows)
            atomic_write_jsonl(training_path, training_rows)
            atomic_write_jsonl(verified_path, human_verified_rows)
            summary = {
                "schema_version": 1,
                "generated_at": utc_now(),
                "dataset_kind": self.dataset_kind,
                "input_manifest": str(self.input_manifest),
                "input_manifest_sha256": self.manifest_sha256,
                "decisions": self.summary(),
                "corrected_manifest": str(corrected_path),
                "training_manifest": str(training_path),
                "human_verified_manifest": str(verified_path),
                "training_rows": len(training_rows),
                "human_verified_rows": len(human_verified_rows),
                "evaluation_policy": (
                    "Human-reviewed samples are locked to train because they were selected using "
                    "current-model predictions. Deployment metrics require an untouched holdout."
                ),
            }
            summary_path = self.output_dir / "labeling_summary.json"
            atomic_write_json(summary_path, summary)
            return summary


def _request_host(request: Request) -> str:
    return (request.url.hostname or "").casefold()


def _origin_is_loopback(request: Request) -> bool:
    origin = request.headers.get("origin")
    if not origin:
        return True
    return (urlsplit(origin).hostname or "").casefold() in LOOPBACK_HOSTS


def create_app(repository: ReviewRepository) -> FastAPI:
    token = secrets.token_urlsafe(32)
    app = FastAPI(
        title=(
            "CashLog actual-data label review"
            if repository.dataset_kind == "actual"
            else "CashLog pre-deployment label review"
        ),
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        if _request_host(request) not in LOOPBACK_HOSTS or not _origin_is_loopback(request):
            return JSONResponse(
                status_code=403,
                content={"detail": "labeling server is loopback-only"},
            )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    def verify_token(request: Request) -> None:
        supplied = request.headers.get("x-labeler-token")
        if supplied is None or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=401, detail="invalid labeling token")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(LABELER_ASSET_DIR / "index.html", media_type="text/html")

    @app.get("/assets/labeler.css", include_in_schema=False)
    def stylesheet() -> FileResponse:
        return FileResponse(LABELER_ASSET_DIR / "labeler.css", media_type="text/css")

    @app.get("/assets/labeler.js", include_in_schema=False)
    def javascript() -> FileResponse:
        return FileResponse(LABELER_ASSET_DIR / "labeler.js", media_type="text/javascript")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return {**repository.state(), "mutation_token": token}

    @app.get("/api/images/{sample_key}")
    def image(sample_key: str) -> FileResponse:
        try:
            path = repository.image_path_for_key(sample_key)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail="sample image not found") from exc
        return FileResponse(path)

    @app.put("/api/decisions/{sample_id:path}")
    def decide(sample_id: str, payload: DecisionPayload, request: Request) -> dict[str, Any]:
        verify_token(request)
        try:
            decision = repository.decide(sample_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="sample not found") from exc
        except RevisionConflict as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "decision": decision,
            "sample": repository.serialize_sample(sample_id),
            "summary": repository.summary(),
        }

    @app.post("/api/export")
    def export(request: Request) -> dict[str, Any]:
        verify_token(request)
        return repository.export()

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "scope": f"loopback-only-{repository.dataset_kind}-labeling",
            "samples": repository.summary()["total"],
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review current-model errors and correct CashLog 33-leaf training labels."
    )
    parser.add_argument("--input-manifest", type=Path)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--image-root", type=Path, action="append", dest="image_roots")
    parser.add_argument("--host", default="127.0.0.1", choices=sorted(LOOPBACK_HOSTS))
    parser.add_argument("--port", type=int)
    parser.add_argument(
        "--actual",
        action="store_true",
        help="Review only consented CashLog actual images on the isolated queue.",
    )
    parser.add_argument("--export-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.actual:
        input_manifest = args.input_manifest or ACTUAL_INPUT
        output_dir = args.output_dir or ACTUAL_OUTPUT_DIR
        port = args.port if args.port is not None else 8012
        dataset_kind: Literal["proxy", "actual"] = "actual"
        image_roots = args.image_roots or [input_manifest.parent]
        if not input_manifest.is_file():
            raise SystemExit(
                "actual manifest not found; run scripts/sync_cashlog_actual.py first: "
                f"{input_manifest}"
            )
    else:
        input_manifest = args.input_manifest or (
            CURRENT_MODEL_INPUT if CURRENT_MODEL_INPUT.exists() else DEFAULT_INPUT
        )
        output_dir = args.output_dir or DEFAULT_OUTPUT_DIR
        port = args.port if args.port is not None else 8011
        dataset_kind = "proxy"
        image_roots = args.image_roots
    repository = ReviewRepository(
        input_manifest=input_manifest,
        categories_path=args.categories,
        output_dir=output_dir,
        image_roots=image_roots,
        dataset_kind=dataset_kind,
    )
    if args.export_only:
        print(json.dumps(repository.export(), ensure_ascii=False, indent=2), flush=True)
        return
    import uvicorn

    summary = repository.summary()
    print(
        json.dumps(
            {
                "url": f"http://{args.host}:{port}",
                "dataset_kind": dataset_kind,
                "samples": summary["total"],
                "mismatches": summary["mismatches"],
                "output_dir": str(repository.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    uvicorn.run(create_app(repository), host=args.host, port=port, access_log=False)


if __name__ == "__main__":
    main()
