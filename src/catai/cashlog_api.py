from __future__ import annotations

import argparse
from functools import lru_cache

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.requests import Request

from .cashlog_classifier import CashlogCategoryClassifier, analyze_base64


@lru_cache(maxsize=1)
def get_classifier() -> CashlogCategoryClassifier:
    return CashlogCategoryClassifier.from_env()


app = FastAPI(title="Catai Cashlog Product Image API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    classifier = get_classifier()
    return {
        "status": "ok",
        "device": str(classifier.device),
        "checkpoint": str(classifier.checkpoint_path),
    }


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    image: UploadFile | None = File(default=None, description="Product image file"),
) -> dict:
    classifier = get_classifier()
    if image is not None:
        return classifier.analyze(await image.read())

    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        image_base64 = str(body.get("imageBase64") or "").strip()
        if image_base64:
            return analyze_base64(image_base64, classifier)

    raise HTTPException(status_code=400, detail="image file or imageBase64 is required")


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve Cashlog product image classification API.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run("catai.cashlog_api:app", host=args.host, port=args.port, reload=args.reload)
