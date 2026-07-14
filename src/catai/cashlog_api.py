from __future__ import annotations

import argparse
import base64
import binascii
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from starlette.requests import Request


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PACKAGE_ROOT / "reports/cashlog_model_report"
REPORT_INDEX = REPORT_DIR / "index.html"
REPORT_JSON = REPORT_DIR / "report.json"
ASSET_ROOT = Path(__file__).resolve().parent / "assets/cashlog_category_uecfood_mps"
DEFAULT_CHECKPOINT = ASSET_ROOT / "best.pt"


@lru_cache(maxsize=1)
def get_classifier() -> Any:
    try:
        from .cashlog_classifier import CashlogCategoryClassifier
    except ImportError as exc:
        raise RuntimeError('model dependencies are not installed. Run: pip install "catai[model]"') from exc

    return CashlogCategoryClassifier.from_env()


def model_runtime_available() -> bool:
    required = ["PIL", "safetensors", "timm", "torch", "torchvision"]
    return all(importlib.util.find_spec(name) is not None for name in required)


app = FastAPI(title="Catai Cashlog Product Image API")


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
    allow_headers=["Content-Type", "Authorization"],
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Content-Security-Policy"] = "frame-ancestors 'none'; object-src 'none'; base-uri 'self'"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    return response

MAX_IMAGE_BYTES = 10 * 1024 * 1024
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
        if MIME_BY_EXTENSION.get(extension) != detected_mime:
            raise HTTPException(status_code=400, detail="image extension and file signature do not match")


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
def health() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "report_available": REPORT_INDEX.exists(),
        "checkpoint_available": DEFAULT_CHECKPOINT.exists(),
        "checkpoint": str(DEFAULT_CHECKPOINT),
        "model_runtime_available": model_runtime_available(),
    }


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    image: UploadFile | None = File(default=None, description="Product image file"),
) -> dict:
    try:
        classifier = get_classifier()
    except (FileNotFoundError, ImportError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if image is not None:
        image_bytes = await image.read(MAX_IMAGE_BYTES + 1)
        validate_image(image_bytes, image.content_type or "", image.filename)
        return classifier.analyze(image_bytes)

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
            return classifier.analyze(image_bytes)

    raise HTTPException(status_code=400, detail="image file or imageBase64 is required")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cashlog product image classification API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("catai.cashlog_api:app", host=args.host, port=args.port, reload=args.reload)
