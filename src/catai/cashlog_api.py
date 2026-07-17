from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import importlib.util
import os
import re
import secrets
import time
import uuid
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from .telemetry import InferenceTelemetry, configure_json_logger, log_event


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = Path(
    os.getenv("CATAI_CASHLOG_REPORT_DIR", PACKAGE_ROOT / "reports/cashlog33/model_report")
)
REPORT_INDEX = REPORT_DIR / "index.html"
REPORT_JSON = REPORT_DIR / "report.json"
ASSET_ROOT = Path(__file__).resolve().parent / "assets/cashlog_category_uecfood_mps"
DEFAULT_CHECKPOINT = ASSET_ROOT / "best.pt"
LOGGER = configure_json_logger("catai.inference")
TELEMETRY = InferenceTelemetry(
    max_records=max(10, int(os.getenv("CATAI_METRICS_WINDOW", "1000")))
)
MODEL_STATE: dict[str, Any] = {
    "loaded": False,
    "load_ms": None,
    "warmup_ms": None,
    "device": None,
    "model": None,
}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")
INFERENCE_SEMAPHORE = asyncio.Semaphore(
    max(1, int(os.getenv("CATAI_MAX_CONCURRENT_INFERENCE", "1")))
)


def env_enabled(name: str, default: bool = False) -> bool:
    fallback = "true" if default else "false"
    return os.getenv(name, fallback).lower() in {"1", "true", "yes"}


@lru_cache(maxsize=1)
def get_classifier() -> Any:
    started = time.perf_counter()
    try:
        from .cashlog_classifier import load_cashlog_classifier_from_env
    except ImportError as exc:
        extra = "hybrid" if os.getenv("CATAI_CASHLOG_HYBRID_CONFIG") else "model"
        raise RuntimeError(
            f'model dependencies are not installed. Run: pip install "catai[{extra}]"'
        ) from exc

    classifier = load_cashlog_classifier_from_env()
    load_ms = (time.perf_counter() - started) * 1000.0
    MODEL_STATE.update(
        {
            "loaded": True,
            "load_ms": load_ms,
            "device": str(getattr(classifier, "device", "unknown")),
            "model": str(
                getattr(classifier, "config", {}).get(
                    "model_version", classifier.__class__.__name__
                )
            ),
        }
    )
    log_event(LOGGER, "model_loaded", **MODEL_STATE)
    return classifier


def model_runtime_available() -> bool:
    if os.getenv("CATAI_CASHLOG_HYBRID_CONFIG"):
        required = [
            "PIL",
            "joblib",
            "onnxruntime",
            "pillow_heif",
            "rapidocr",
            "safetensors",
            "sklearn",
            "torch",
            "torchvision",
            "transformers",
        ]
    else:
        required = ["PIL", "safetensors", "timm", "torch", "torchvision"]
    return all(importlib.util.find_spec(name) is not None for name in required)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if env_enabled("CATAI_EAGER_LOAD_MODEL") and model_runtime_available():
        classifier = await run_in_threadpool(get_classifier)
        if env_enabled("CATAI_WARMUP_MODEL"):
            started = time.perf_counter()
            buffer = io.BytesIO()
            warmup_image = Image.new("RGB", (960, 960), "white")
            draw = ImageDraw.Draw(warmup_image)
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 48)
            except OSError:
                font = ImageFont.load_default()
            for index, text in enumerate(["CASHLOG RECEIPT", "COFFEE 5500", "TOTAL 5500"]):
                draw.text((80, 100 + index * 120), text, fill="black", font=font)
            warmup_image.save(buffer, format="JPEG", quality=88)
            for _ in range(2):
                warmup = await run_in_threadpool(classifier.analyze, buffer.getvalue())
                warmup.pop("_timings_ms", None)
            MODEL_STATE["warmup_ms"] = (time.perf_counter() - started) * 1000.0
            log_event(LOGGER, "model_warmed", **MODEL_STATE)
    yield


app = FastAPI(title="Catai Cashlog Product Image API", lifespan=lifespan)


def cors_allowed_origins() -> list[str]:
    configured = os.getenv(
        "CATAI_CORS_ALLOWED_ORIGINS",
        "http://127.0.0.1:5175,http://localhost:5175,http://127.0.0.1:5173,http://localhost:5173",
    )
    return [origin.strip().rstrip("/") for origin in configured.split(",") if origin.strip()]


app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_allowed_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-Internal-API-Key"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    supplied_request_id = request.headers.get("x-request-id", "")
    request_id = (
        supplied_request_id
        if REQUEST_ID_PATTERN.fullmatch(supplied_request_id)
        else uuid.uuid4().hex
    )
    request.state.request_id = request_id
    started = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        log_event(
            LOGGER,
            "request_failed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            total_ms=round(elapsed_ms, 3),
        )
        raise
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.3f}"
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.url.path != "/health" or env_enabled("CATAI_LOG_HEALTH_REQUESTS"):
        log_event(
            LOGGER,
            "request_completed",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            total_ms=round(elapsed_ms, 3),
        )
    return response

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 30_000_000
MAX_REQUEST_BYTES = 14 * 1024 * 1024
MIME_BY_EXTENSION = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "heic": "image/heic",
    "heif": "image/heif",
}


def detect_image_mime(data: bytes) -> str | None:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12].lower()
        if brand in {b"heic", b"heix", b"hevc", b"hevx"}:
            return "image/heic"
        if brand in {b"mif1", b"msf1"}:
            return "image/heif"
    return None


def validate_image(data: bytes, declared_mime: str, filename: str | None = None) -> None:
    if not data or len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image must be between 1 byte and 10MB")
    normalized_mime = declared_mime.split(";", 1)[0].strip().lower()
    detected_mime = detect_image_mime(data)
    if not detected_mime or detected_mime != normalized_mime:
        raise HTTPException(status_code=400, detail="image MIME type and file signature do not match")
    if filename:
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension and MIME_BY_EXTENSION.get(extension) != detected_mime:
            raise HTTPException(status_code=400, detail="image extension and file signature do not match")
    try:
        with Image.open(io.BytesIO(data)) as image:
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="image dimensions exceed the 30MP limit")
            image.verify()
    except HTTPException:
        raise
    except (Image.DecompressionBombError, UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image payload cannot be decoded safely") from exc


def verify_internal_api_key(value: str | None) -> None:
    required = os.getenv("CATAI_REQUIRE_INTERNAL_API_KEY", "false").lower() in {
        "1",
        "true",
        "yes",
    }
    configured = os.getenv("CATAI_INTERNAL_API_KEY")
    if required and not configured:
        raise HTTPException(status_code=503, detail="internal API authentication is not configured")
    if configured and (value is None or not secrets.compare_digest(value, configured)):
        raise HTTPException(status_code=401, detail="unauthorized")


@app.get("/", include_in_schema=False)
def report_index() -> FileResponse:
    if not REPORT_INDEX.exists():
        raise HTTPException(status_code=404, detail="model report has not been generated")
    return FileResponse(REPORT_INDEX, media_type="text/html")


@app.get("/report", include_in_schema=False)
def report_alias() -> FileResponse:
    return report_index()


@app.get("/report.json", include_in_schema=False)
def report_json() -> FileResponse:
    if not REPORT_JSON.exists():
        raise HTTPException(status_code=404, detail="model report JSON has not been generated")
    return FileResponse(REPORT_JSON, media_type="application/json")


@app.get("/health")
def health() -> dict[str, Any]:
    hybrid_config = os.getenv("CATAI_CASHLOG_HYBRID_CONFIG")
    ensemble_config = os.getenv("CATAI_CASHLOG_ENSEMBLE_CONFIG")
    serving_path = (
        Path(hybrid_config or ensemble_config)
        if (hybrid_config or ensemble_config)
        else DEFAULT_CHECKPOINT
    )
    serving_mode = "hybrid" if hybrid_config else "ensemble" if ensemble_config else "single"
    return {
        "status": "ok",
        "report_available": REPORT_INDEX.exists(),
        "checkpoint_available": serving_path.exists(),
        "checkpoint": str(serving_path),
        "serving_mode": serving_mode,
        "model_runtime_available": model_runtime_available(),
        "model_loaded": bool(MODEL_STATE["loaded"]),
        "model_device": MODEL_STATE["device"],
        "model_version": MODEL_STATE["model"],
        "model_load_ms": MODEL_STATE["load_ms"],
        "model_warmup_ms": MODEL_STATE["warmup_ms"],
    }


@app.get("/metrics")
def metrics(request: Request) -> dict[str, Any]:
    verify_internal_api_key(
        request.headers.get("x-internal-api-key") or request.headers.get("x-api-key")
    )
    return {
        "status": "ok",
        "model": dict(MODEL_STATE),
        "inference": TELEMETRY.snapshot(),
    }


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    image: UploadFile | None = File(default=None, description="Product image file"),
) -> dict:
    declared_length = request.headers.get("content-length")
    if declared_length:
        try:
            if int(declared_length) > MAX_REQUEST_BYTES:
                raise HTTPException(status_code=413, detail="request exceeds the 14MB limit")
        except ValueError as exc:
            raise HTTPException(status_code=400, detail="invalid Content-Length") from exc
    verify_internal_api_key(
        request.headers.get("x-internal-api-key") or request.headers.get("x-api-key")
    )
    try:
        classifier = get_classifier()
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if image is not None:
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        validate_image(image_bytes, image.content_type or "", image.filename)
        return await run_analysis(request, classifier, image_bytes)

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        image_base64 = str(body.get("imageBase64") or "").strip()
        if image_base64:
            try:
                image_bytes = base64.b64decode(image_base64, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=400, detail="invalid imageBase64") from exc
            validate_image(
                image_bytes,
                str(body.get("mimeType") or ""),
                str(body.get("filename") or "") or None,
            )
            return await run_analysis(request, classifier, image_bytes)

    raise HTTPException(status_code=400, detail="image file or imageBase64 is required")


async def run_analysis(request: Request, classifier: Any, image_bytes: bytes) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        async with INFERENCE_SEMAPHORE:
            result = await run_in_threadpool(classifier.analyze, image_bytes)
    except Exception as exc:
        total_ms = (time.perf_counter() - started) * 1000.0
        TELEMETRY.record(
            status="error",
            total_ms=total_ms,
            model=str(MODEL_STATE["model"] or "unknown"),
            device=str(MODEL_STATE["device"] or "unknown"),
        )
        log_event(
            LOGGER,
            "inference_failed",
            request_id=request.state.request_id,
            error_type=type(exc).__name__,
            total_ms=round(total_ms, 3),
        )
        raise HTTPException(status_code=500, detail="model inference failed") from exc

    total_ms = (time.perf_counter() - started) * 1000.0
    timings = {
        str(key): float(value)
        for key, value in result.pop("_timings_ms", {}).items()
    }
    model = str(result.get("model") or MODEL_STATE["model"] or "unknown")
    device = str(MODEL_STATE["device"] or "unknown")
    TELEMETRY.record(
        status="ok",
        total_ms=total_ms,
        stages_ms=timings,
        model=model,
        device=device,
    )
    log_event(
        LOGGER,
        "inference_completed",
        request_id=request.state.request_id,
        model=model,
        device=device,
        category=result.get("recommended_category"),
        confidence=round(float(result.get("confidence", 0.0)), 6),
        need_user_check=bool(result.get("need_user_check")),
        total_ms=round(total_ms, 3),
        stages_ms={key: round(value, 3) for key, value in timings.items()},
    )
    if env_enabled("CATAI_INCLUDE_PERFORMANCE_IN_RESPONSE"):
        result["performance"] = {
            "total_ms": total_ms,
            "stages_ms": timings,
            "device": device,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cashlog product image classification API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("catai.cashlog_api:app", host=args.host, port=args.port, reload=args.reload)
