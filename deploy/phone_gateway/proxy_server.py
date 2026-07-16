from __future__ import annotations

import os
import secrets

import httpx
from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.responses import Response


MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:18010")
GATEWAY_API_KEY = os.getenv("GATEWAY_API_KEY") or os.getenv("PUBLIC_API_KEY")
MODEL_API_KEY = os.getenv("MODEL_API_KEY")
MAX_REQUEST_BYTES = int(
    os.getenv(
        "MAX_REQUEST_BYTES",
        os.getenv("MAX_UPLOAD_BYTES", str(14 * 1024 * 1024)),
    )
)
EXPOSE_HEALTH = os.getenv("EXPOSE_HEALTH", "false").lower() in {"1", "true", "yes"}

ALLOWED_CONTENT_TYPES = (
    "multipart/form-data",
    "application/json",
)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


def verify_api_key(value: str | None) -> None:
    if not GATEWAY_API_KEY:
        raise RuntimeError("GATEWAY_API_KEY is not configured")
    if value is None or not secrets.compare_digest(value, GATEWAY_API_KEY):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
        )


def validate_content_type(value: str | None) -> str:
    if not value:
        raise HTTPException(status_code=400, detail="Content-Type is required")
    normalized = value.split(";", 1)[0].strip().lower()
    if normalized not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="Unsupported Content-Type")
    return value


def validate_content_length(value: str | None) -> None:
    if not value:
        return
    try:
        declared_size = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    if declared_size > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")


@app.get("/health")
async def health() -> dict[str, str]:
    if not EXPOSE_HEALTH:
        raise HTTPException(status_code=404, detail="Not found")
    return {"status": "gateway-ok"}


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> Response:
    verify_api_key(x_api_key)
    content_type = validate_content_type(request.headers.get("content-type"))
    validate_content_length(request.headers.get("content-length"))

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="Empty request body")
    if len(body) > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")
    if not MODEL_API_KEY:
        raise HTTPException(status_code=503, detail="Model authentication is not configured")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=5.0)) as client:
            upstream = await client.post(
                f"{MODEL_BASE_URL}/analyze-image",
                content=body,
                headers={
                    "content-type": content_type,
                    "x-internal-api-key": MODEL_API_KEY,
                },
            )
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail="Model worker unavailable") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Model worker timed out") from exc

    return Response(
        content=upstream.content,
        status_code=upstream.status_code,
        media_type=upstream.headers.get("content-type", "application/json"),
    )
