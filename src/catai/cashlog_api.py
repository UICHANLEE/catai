from __future__ import annotations

import argparse
import base64
import importlib.util
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
        return classifier.analyze(await image.read())

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        image_base64 = str(body.get("imageBase64") or "").strip()
        if image_base64:
            return classifier.analyze(base64.b64decode(image_base64))

    raise HTTPException(status_code=400, detail="image file or imageBase64 is required")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cashlog product image classification API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("catai.cashlog_api:app", host=args.host, port=args.port, reload=args.reload)
