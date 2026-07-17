from __future__ import annotations

from collections.abc import AsyncIterator
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
MAX_RESPONSE_BYTES = int(os.getenv("MAX_RESPONSE_BYTES", str(2 * 1024 * 1024)))
EXPOSE_HEALTH = os.getenv("EXPOSE_HEALTH", "false").lower() in {"1", "true", "yes"}

ALLOWED_CONTENT_TYPES = (
    "multipart/form-data",
    "application/json",
)

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


class UploadTooLarge(Exception):
    pass


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
    if normalized == "multipart/form-data" and "boundary=" not in value.lower():
        raise HTTPException(status_code=400, detail="Multipart boundary is required")
    return value


def validate_content_length(value: str | None) -> None:
    if not value:
        return
    try:
        declared_size = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid Content-Length") from exc
    if declared_size < 0:
        raise HTTPException(status_code=400, detail="Invalid Content-Length")
    if declared_size > MAX_REQUEST_BYTES:
        raise HTTPException(status_code=413, detail="Upload too large")


async def limited_body(request: Request) -> AsyncIterator[bytes]:
    received = 0
    async for chunk in request.stream():
        received += len(chunk)
        if received > MAX_REQUEST_BYTES:
            raise UploadTooLarge
        yield chunk


async def upstream_request(
    method: str,
    path: str,
    *,
    content: AsyncIterator[bytes] | None = None,
    content_type: str | None = None,
) -> Response:
    headers = {"accept": "application/json"}
    if content_type:
        headers["content-type"] = content_type
    if MODEL_API_KEY:
        headers["x-internal-api-key"] = MODEL_API_KEY

    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=5.0),
            follow_redirects=False,
        ) as client:
            async with client.stream(
                method,
                f"{MODEL_BASE_URL.rstrip('/')}{path}",
                content=content,
                headers=headers,
            ) as upstream:
                payload = await upstream.aread()
                if len(payload) > MAX_RESPONSE_BYTES:
                    raise HTTPException(status_code=502, detail="Model response too large")
                upstream_status = upstream.status_code
                upstream_content_type = upstream.headers.get(
                    "content-type", "application/json"
                )
    except UploadTooLarge as exc:
        raise HTTPException(status_code=413, detail="Upload too large") from exc
    except httpx.ConnectError as exc:
        raise HTTPException(status_code=503, detail="Model worker unavailable") from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="Model worker timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail="Model worker request failed") from exc

    safe_content_type = (
        upstream_content_type
        if upstream_content_type.lower().startswith("application/json")
        else "application/json"
    )
    return Response(
        content=payload,
        status_code=upstream_status,
        headers={
            "Cache-Control": "no-store",
            "Content-Type": safe_content_type,
        },
    )


@app.get("/health")
async def health(x_api_key: str | None = Header(default=None)) -> Response:
    if not EXPOSE_HEALTH:
        verify_api_key(x_api_key)
    return await upstream_request("GET", "/health")


@app.post("/analyze-image")
async def analyze_image(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> Response:
    verify_api_key(x_api_key)
    content_type = validate_content_type(request.headers.get("content-type"))
    validate_content_length(request.headers.get("content-length"))
    if not MODEL_API_KEY:
        raise HTTPException(status_code=503, detail="Model authentication is not configured")
    return await upstream_request(
        "POST",
        "/analyze-image",
        content=limited_body(request),
        content_type=content_type,
    )
