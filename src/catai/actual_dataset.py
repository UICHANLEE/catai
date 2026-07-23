"""Materialize consented CashLog images into a private, de-identified dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import secrets
import uuid
import warnings
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator
from urllib.parse import quote, urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener

from PIL import Image, ImageOps, UnidentifiedImageError

from catai.feedback import load_leaf_ids, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATEGORIES = ROOT / "configs/cashlog/categories.json"
DEFAULT_ACTUAL_DIR = ROOT / "data/raw/cashlog33/actual"
DEFAULT_BUCKET = "cashlog-media"
DEFAULT_MAX_UPLOAD_BYTES = 15 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
BUCKET_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # type: ignore[no-untyped-def]
        return None


NO_REDIRECT_OPENER = build_opener(RejectRedirects)
Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    _ensure_private_directory(path.parent)
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


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    _ensure_private_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        path.chmod(0o600)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    _ensure_private_directory(path.parent)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    path.chmod(0o600)


@contextmanager
def _exclusive_sync_lock(actual_dir: Path) -> Iterator[None]:
    _ensure_private_directory(actual_dir)
    lock_path = actual_dir / ".sync.lock"
    try:
        descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"another actual dataset sync is active: {lock_path}") from error
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            os.fsync(handle.fileno())
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def _safe_object_key(value: Any) -> str:
    key = str(value or "").strip()
    path = PurePosixPath(key)
    if (
        not key
        or len(key) > 1024
        or path.is_absolute()
        or "\\" in key
        or any(ord(character) < 32 for character in key)
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("invalid private image object key")
    return key


def _safe_local_source(source_root: Path, object_key: str) -> Path:
    root = source_root.resolve()
    path = (root / Path(*PurePosixPath(object_key).parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError("private image path escapes source root")
    if not path.is_file():
        raise FileNotFoundError("private source image is missing")
    return path


def _validate_supabase_url(value: str) -> tuple[str, Any]:
    parsed = urlparse(value)
    if (
        parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or not parsed.hostname
    ):
        raise ValueError("Supabase URL must not contain credentials, query, or fragment")
    if parsed.scheme != "https" and parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Supabase URL must use HTTPS outside local development")
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("Supabase URL must use HTTP or HTTPS")
    return value.rstrip("/"), parsed


def fetch_private_supabase_image(
    *,
    supabase_url: str,
    service_role_key: str,
    bucket: str,
    object_key: str,
    maximum_bytes: int,
) -> bytes:
    base_url, parsed = _validate_supabase_url(supabase_url)
    if not service_role_key or len(service_role_key) < 32:
        raise ValueError("Supabase service role key is missing or unexpectedly short")
    if not BUCKET_PATTERN.fullmatch(bucket):
        raise ValueError("invalid Supabase storage bucket name")
    safe_key = _safe_object_key(object_key)
    endpoint = (
        f"{base_url}/storage/v1/object/authenticated/"
        f"{quote(bucket, safe='')}/{quote(safe_key, safe='/-._~')}"
    )
    request = Request(
        endpoint,
        headers={
            "apikey": service_role_key,
            "Authorization": f"Bearer {service_role_key}",
            "Accept": "image/*,application/octet-stream;q=0.8",
            "User-Agent": "CataiActualDataset/1.0",
        },
    )
    with NO_REDIRECT_OPENER.open(request, timeout=45) as response:
        response_url = urlparse(response.geturl())
        if (
            response_url.scheme != parsed.scheme
            or response_url.hostname != parsed.hostname
            or response_url.port != parsed.port
        ):
            raise ValueError("Supabase image download refused a cross-origin redirect")
        content_length = response.headers.get("Content-Length")
        if content_length:
            try:
                declared_size = int(content_length)
            except ValueError as error:
                raise ValueError("invalid image Content-Length") from error
            if declared_size > maximum_bytes:
                raise ValueError("private source image exceeds the size limit")
        payload = response.read(maximum_bytes + 1)
    if len(payload) > maximum_bytes:
        raise ValueError("private source image exceeds the size limit")
    return payload


def _normalize_private_image(payload: bytes) -> tuple[bytes, int, int]:
    if not payload:
        raise ValueError("private source image is empty")
    try:
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as source:
                source.verify()
            with Image.open(io.BytesIO(payload)) as source:
                normalized = ImageOps.exif_transpose(source)
                normalized.load()
                if normalized.width <= 0 or normalized.height <= 0:
                    raise ValueError("private source image has invalid dimensions")
                if normalized.width * normalized.height > MAX_IMAGE_PIXELS:
                    raise ValueError("private source image exceeds the pixel limit")
                if normalized.mode in {"RGBA", "LA"} or (
                    normalized.mode == "P" and "transparency" in normalized.info
                ):
                    rgba = normalized.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, "white")
                    background.alpha_composite(rgba)
                    rgb = background.convert("RGB")
                else:
                    rgb = normalized.convert("RGB")
                width, height = rgb.size
                output = io.BytesIO()
                rgb.save(
                    output,
                    format="JPEG",
                    quality=95,
                    optimize=True,
                    progressive=False,
                )
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
        raise ValueError("private source image exceeds the pixel limit") from error
    except (UnidentifiedImageError, OSError, SyntaxError) as error:
        raise ValueError("private source file is not a valid supported image") from error
    normalized_bytes = output.getvalue()
    if not normalized_bytes:
        raise ValueError("private source image normalization failed")
    return normalized_bytes, width, height


def _deidentified_sample_id(sample_id: str) -> str:
    return "cashlog_actual:" + hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:32]


def _load_existing_manifest(path: Path, actual_dir: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = read_jsonl(path)
    sample_ids: set[str] = set()
    content_hashes: set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        digest = str(row.get("sha256") or "")
        relative_path = Path(str(row.get("relative_path") or ""))
        image_path = (
            relative_path
            if relative_path.is_absolute()
            else (ROOT / relative_path)
        ).resolve()
        if not sample_id.startswith("cashlog_actual:") or len(digest) != 64:
            raise ValueError("existing actual manifest contains an invalid row")
        if sample_id in sample_ids or digest in content_hashes:
            raise ValueError("existing actual manifest contains duplicate samples")
        if not image_path.is_relative_to(actual_dir.resolve()) or not image_path.is_file():
            raise ValueError("existing actual manifest references a missing or unsafe image")
        if hashlib.sha256(image_path.read_bytes()).hexdigest() != digest:
            raise ValueError("existing actual image checksum does not match its manifest")
        sample_ids.add(sample_id)
        content_hashes.add(digest)
    return rows


def _validate_release(
    *,
    events: list[dict[str, Any]],
    secure_images: list[dict[str, Any]],
    leaf_ids: set[str],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    events_by_id: dict[str, dict[str, Any]] = {}
    samples: set[str] = set()
    for event in events:
        event_id = str(event.get("event_id") or "")
        sample_id = str(event.get("sample_id") or "")
        if not event_id or not sample_id or event_id in events_by_id or sample_id in samples:
            raise ValueError("feedback release contains duplicate or missing event identifiers")
        events_by_id[event_id] = event
        samples.add(sample_id)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    secure_events: set[str] = set()
    secure_samples: set[str] = set()
    for secure in secure_images:
        event_id = str(secure.get("event_id") or "")
        sample_id = str(secure.get("sample_id") or "")
        if event_id in secure_events or sample_id in secure_samples:
            raise ValueError("secure image index contains duplicate identifiers")
        event = events_by_id.get(event_id)
        if event is None or str(event.get("sample_id") or "") != sample_id:
            raise ValueError("secure image index does not match the de-identified release")
        leaf_id = str(secure.get("leaf_id") or "")
        secure_status = str(secure.get("review_status") or "")
        if (
            leaf_id not in leaf_ids
            or str(event.get("selected_leaf_id") or "") != leaf_id
            or secure_status not in {"pending", "approved"}
            or event.get("review_status") != secure_status
            or event.get("image_retention_consent") is not True
            or event.get("has_retained_image") is not True
        ):
            raise ValueError(
                "secure image is not pending/approved, consented, and correctly labeled"
            )
        _safe_object_key(secure.get("image_object_key"))
        secure_events.add(event_id)
        secure_samples.add(sample_id)
        pairs.append((event, secure))
    return sorted(pairs, key=lambda pair: str(pair[1]["sample_id"]))


def _manifest_row(
    *,
    event: dict[str, Any],
    sample_id: str,
    relative_path: Path,
    digest: str,
    byte_count: int,
    width: int,
    height: int,
    imported_at: str,
) -> dict[str, Any]:
    top3 = [
        {
            "leaf_id": str(candidate["category"]),
            "score": float(candidate["confidence"]),
        }
        for candidate in event.get("predicted_top3") or []
    ][:3]
    leaf_id = str(event["selected_leaf_id"])
    predicted_leaf = top3[0]["leaf_id"] if top3 else None
    confidence = top3[0]["score"] if top3 else None
    margin = (
        top3[0]["score"] - top3[1]["score"]
        if len(top3) >= 2
        else None
    )
    mismatch = predicted_leaf is not None and predicted_leaf != leaf_id
    occurred_at = str(event.get("occurred_at") or "")
    return {
        "schema_version": 1,
        "sample_id": _deidentified_sample_id(sample_id),
        "leaf_id": leaf_id,
        "relative_path": relative_path.as_posix(),
        "sha256": digest,
        "bytes": byte_count,
        "width": width,
        "height": height,
        "source": "cashlog_actual",
        "provider": "cashlog_app",
        "license": "private_consented",
        "dataset_origin": "actual",
        "source_month": occurred_at[:7] if len(occurred_at) >= 7 else "",
        "feedback_source": str(event.get("source") or ""),
        "model_version": str(event.get("model_version") or ""),
        "taxonomy_version": str(event.get("taxonomy_version") or ""),
        "review_status": "pending",
        "model_review_status": "model_mismatch" if mismatch else "model_match",
        "need_user_check": True,
        "image_retention_consent": True,
        "metadata_stripped": True,
        "imported_at": imported_at,
        "model_review_details": {
            "top3": top3,
            "confidence": confidence,
            "expected_margin": margin,
        },
    }


def materialize_actual_release(
    *,
    release_dir: Path,
    actual_dir: Path = DEFAULT_ACTUAL_DIR,
    categories_path: Path = DEFAULT_CATEGORIES,
    source_root: Path | None = None,
    supabase_url: str | None = None,
    service_role_key: str | None = None,
    bucket: str = DEFAULT_BUCKET,
    maximum_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    remote_fetcher: Callable[..., bytes] = fetch_private_supabase_image,
) -> dict[str, Any]:
    """Import one feedback release into the isolated actual dataset.

    Local sources are removed only after the normalized image and manifest are
    durably written. Supabase objects are downloaded but never deleted here.
    """

    release_dir = release_dir.resolve()
    actual_dir = actual_dir.resolve()
    if maximum_bytes < 1024 or maximum_bytes > 100 * 1024 * 1024:
        raise ValueError("maximum image bytes must be between 1 KiB and 100 MiB")
    events_path = release_dir / "events.jsonl"
    secure_path = release_dir / "secure_image_index.jsonl"
    if not events_path.is_file() or not secure_path.is_file():
        raise FileNotFoundError(
            "feedback release requires events.jsonl and secure_image_index.jsonl"
        )
    if source_root is None and (not supabase_url or not service_role_key):
        raise ValueError(
            "Supabase URL and service role key are required without --source-root"
        )

    events = read_jsonl(events_path)
    secure_images = read_jsonl(secure_path)
    pairs = _validate_release(
        events=events,
        secure_images=secure_images,
        leaf_ids=set(load_leaf_ids(categories_path)),
    )
    images_dir = actual_dir / "images"
    manifest_path = actual_dir / "manifest.jsonl"
    audit_path = actual_dir / "import_audit.jsonl"
    quarantine_path = actual_dir / "import_quarantine.jsonl"
    summary_path = actual_dir / "import_summary.json"
    imported_at = utc_now()
    release_fingerprint = hashlib.sha256(
        events_path.read_bytes() + b"\0" + secure_path.read_bytes()
    ).hexdigest()

    with _exclusive_sync_lock(actual_dir):
        _ensure_private_directory(images_dir)
        if not images_dir.resolve().is_relative_to(actual_dir):
            raise ValueError("actual images directory escapes the private dataset root")
        existing_rows = _load_existing_manifest(manifest_path, actual_dir)
        rows_by_sample = {str(row["sample_id"]): row for row in existing_rows}
        sample_by_digest = {str(row["sha256"]): str(row["sample_id"]) for row in existing_rows}
        added_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        quarantined_rows: list[dict[str, Any]] = []
        local_sources_to_remove: list[Path] = []
        already_present = 0
        duplicate_content = 0

        for event, secure in pairs:
            raw_sample_id = str(secure["sample_id"])
            sample_id = _deidentified_sample_id(raw_sample_id)
            object_key = _safe_object_key(secure["image_object_key"])
            local_source: Path | None = None
            try:
                if source_root is not None:
                    try:
                        local_source = _safe_local_source(source_root, object_key)
                    except FileNotFoundError:
                        if sample_id in rows_by_sample:
                            already_present += 1
                            continue
                        raise
                    if local_source.stat().st_size > maximum_bytes:
                        raise ValueError("private source image exceeds the size limit")
                    payload = local_source.read_bytes()
                    if len(payload) > maximum_bytes:
                        raise ValueError("private source image exceeds the size limit")
                else:
                    payload = remote_fetcher(
                        supabase_url=str(supabase_url),
                        service_role_key=str(service_role_key),
                        bucket=bucket,
                        object_key=object_key,
                        maximum_bytes=maximum_bytes,
                    )
                normalized, width, height = _normalize_private_image(payload)
                if len(normalized) > maximum_bytes:
                    raise ValueError("normalized private image exceeds the size limit")
                digest = hashlib.sha256(normalized).hexdigest()
                existing = rows_by_sample.get(sample_id)
                if existing:
                    if not secrets.compare_digest(str(existing["sha256"]), digest):
                        raise ValueError("an existing actual sample changed image content")
                    already_present += 1
                    if local_source is not None:
                        local_sources_to_remove.append(local_source)
                    continue
                duplicate_sample = sample_by_digest.get(digest)
                if duplicate_sample:
                    duplicate_content += 1
                    quarantined_rows.append(
                        {
                            "sample_id": sample_id,
                            "reason": "duplicate normalized image content",
                            "duplicate_of": duplicate_sample,
                            "release_fingerprint": release_fingerprint,
                            "quarantined_at": imported_at,
                        }
                    )
                    continue

                image_path = images_dir / digest[:2] / f"{digest}.jpg"
                _ensure_private_directory(image_path.parent)
                if image_path.exists():
                    if hashlib.sha256(image_path.read_bytes()).hexdigest() != digest:
                        raise ValueError("actual image path contains unexpected content")
                else:
                    temporary = image_path.with_name(f".{image_path.name}.{uuid.uuid4().hex}.tmp")
                    descriptor = os.open(
                        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
                    )
                    try:
                        with os.fdopen(descriptor, "wb") as handle:
                            handle.write(normalized)
                            handle.flush()
                            os.fsync(handle.fileno())
                        temporary.replace(image_path)
                        image_path.chmod(0o600)
                    finally:
                        if temporary.exists():
                            temporary.unlink()

                try:
                    relative_path = image_path.relative_to(ROOT)
                except ValueError:
                    relative_path = image_path
                row = _manifest_row(
                    event=event,
                    sample_id=raw_sample_id,
                    relative_path=relative_path,
                    digest=digest,
                    byte_count=len(normalized),
                    width=width,
                    height=height,
                    imported_at=imported_at,
                )
                added_rows.append(row)
                rows_by_sample[sample_id] = row
                sample_by_digest[digest] = sample_id
                audit_rows.append(
                    {
                        "sample_id": sample_id,
                        "sha256": digest,
                        "action": "moved_from_local_source"
                        if local_source is not None
                        else "downloaded_from_private_storage",
                        "release_fingerprint": release_fingerprint,
                        "imported_at": imported_at,
                    }
                )
                if local_source is not None:
                    local_sources_to_remove.append(local_source)
            except (KeyError, OSError, TypeError, ValueError) as error:
                quarantined_rows.append(
                    {
                        "sample_id": sample_id,
                        "reason": str(error),
                        "release_fingerprint": release_fingerprint,
                        "quarantined_at": imported_at,
                    }
                )

        final_rows = sorted(rows_by_sample.values(), key=lambda row: str(row["sample_id"]))
        _atomic_write_jsonl(manifest_path, final_rows)
        _append_jsonl(audit_path, audit_rows)
        _append_jsonl(quarantine_path, quarantined_rows)
        summary = {
            "schema_version": 1,
            "generated_at": imported_at,
            "release_fingerprint": release_fingerprint,
            "release_candidates": len(pairs),
            "imported": len(added_rows),
            "already_present": already_present,
            "duplicate_content": duplicate_content,
            "quarantined": len(quarantined_rows),
            "actual_dataset_rows": len(final_rows),
            "manifest": str(manifest_path),
            "images_dir": str(images_dir),
            "source_mode": "local_move" if source_root is not None else "supabase_download",
            "remote_objects_deleted": False,
            "contains_object_keys": False,
            "contains_user_ids": False,
            "metadata_stripped": True,
        }
        _atomic_write_json(summary_path, summary)

        for source in sorted(set(local_sources_to_remove)):
            source.unlink(missing_ok=True)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move or download pending/approved consented CashLog images into the private "
            "actual dataset."
        )
    )
    parser.add_argument("--release-dir", type=Path, required=True)
    parser.add_argument("--actual-dir", type=Path, default=DEFAULT_ACTUAL_DIR)
    parser.add_argument("--categories", type=Path, default=DEFAULT_CATEGORIES)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="Local private-storage mirror. Imported source files are removed after commit.",
    )
    parser.add_argument("--supabase-url", default=os.getenv("SUPABASE_URL"))
    parser.add_argument(
        "--service-role-key-env",
        default="SUPABASE_SERVICE_ROLE_KEY",
        help="Environment variable containing the backend-only service role key.",
    )
    parser.add_argument("--bucket", default=DEFAULT_BUCKET)
    parser.add_argument("--maximum-bytes", type=int, default=DEFAULT_MAX_UPLOAD_BYTES)
    parser.add_argument("--fail-on-quarantine", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service_role_key = (
        None if args.source_root else os.getenv(args.service_role_key_env)
    )
    try:
        summary = materialize_actual_release(
            release_dir=args.release_dir,
            actual_dir=args.actual_dir,
            categories_path=args.categories,
            source_root=args.source_root,
            supabase_url=args.supabase_url,
            service_role_key=service_role_key,
            bucket=args.bucket,
            maximum_bytes=args.maximum_bytes,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    print(Path(summary["manifest"]).resolve(), flush=True)
    if args.fail_on_quarantine and summary["quarantined"]:
        raise SystemExit(
            f"actual dataset sync quarantined {summary['quarantined']} candidate(s)"
        )


if __name__ == "__main__":
    main()
